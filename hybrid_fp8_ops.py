# file: ComfyUI/custom_nodes/HybridFP8Loader/hybrid_fp8_ops.py

import torch
import comfy.ops
from comfy.ops import manual_cast, fp8_linear, cast_bias_weight, uncast_bias_weight
import comfy.model_management
import logging

# Dictionary mapping specific full tensor keys to their desired dtype
_high_precision_keys = {} # Format: { "full.layer.name.weight": torch.float16, ... }


def set_high_precision_keys(key_dtype_map):
    """Sets the dictionary of {full_tensor_name: dtype}."""
    global _high_precision_keys
    _high_precision_keys = key_dtype_map
    print(f"[Hybrid FP8 Ops] High precision overrides set for {len(key_dtype_map)} layers.")


def get_hybrid_fp8_ops(scale_input_enabled=False, disable_fp8_mat_mult=False):
    """
    Dynamically creates and returns a hybrid operations class.
    Combines standard Scaled FP8 logic with granular High-Precision overrides.
    """
    print(f"[Hybrid FP8 Ops] Configuring with scale_input_enabled: {scale_input_enabled}")

    if disable_fp8_mat_mult == False:
       fp8_mat_mult_supported = comfy.model_management.supports_fp8_compute()
    else:
        fp8_mat_mult_supported = False

    logging.info("Using hybrid fp8: fp8 matrix mult: {}, scale input: {}".format(fp8_mat_mult_supported, scale_input_enabled))

    class HybridOps(manual_cast):
        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                # Initialize as FP8 by default.
                # We do not know if this is a high-precision layer until _load_from_state_dict.
                kwargs['dtype'] = torch.float8_e4m3fn
                super().__init__(*args, **kwargs)

            def reset_parameters(self):
                # Standard FP8 Initialization
                if not hasattr(self, 'scale_weight'):
                    self.scale_weight = torch.nn.parameter.Parameter(data=torch.ones((), device=self.weight.device, dtype=torch.float32), requires_grad=False)

                if not scale_input_enabled:
                    self.scale_input = None

                if not hasattr(self, 'scale_input'):
                    self.scale_input = torch.nn.parameter.Parameter(data=torch.ones((), device=self.weight.device, dtype=torch.float32), requires_grad=False)
                return None

            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                # Construct the key as it appears in the map (e.g. "model.diffusion_model...weight")
                weight_key = prefix + 'weight'

                # Determine TARGET_DTYPE based on key existence in the map
                if weight_key in _high_precision_keys:
                    target_dtype = _high_precision_keys[weight_key]
                else:
                    target_dtype = torch.float8_e4m3fn

                # If the target is NOT FP8, we perform the swap to High Precision
                if target_dtype != torch.float8_e4m3fn:
                    # Load weight as High Precision (Native Dtype)
                    weight_tensor = state_dict.pop(weight_key, None)
                    bias_key = prefix + 'bias'
                    bias_tensor = state_dict.pop(bias_key, None)

                    if weight_tensor is not None:
                        # Enforce the dtype from the dictionary (BF16/FP16/FP32)
                        # This replaces the FP8 parameter created in __init__
                        self.weight = torch.nn.Parameter(weight_tensor.to(dtype=target_dtype), requires_grad=False)
                    else:
                        missing_keys.append(weight_key)

                    if bias_tensor is not None:
                        self.bias = torch.nn.Parameter(bias_tensor.to(dtype=target_dtype), requires_grad=False)
                    else:
                        self.bias = None

                    # Clean up unused FP8 scales from state_dict if they exist so they don't trigger unexpected keys
                    state_dict.pop(prefix + 'scale_weight', None)
                    state_dict.pop(prefix + 'scale_input', None)

                    # Nullify the FP8 specific attributes on the layer instance
                    self.scale_weight = None
                    self.scale_input = None

                    # Flag this layer so forward() knows to skip FP8 logic
                    setattr(self, 'is_high_precision_layer', True)
                else:
                    # No override: Load normally as FP8
                    super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

            def forward_comfy_cast_weights(self, input):
                # PATH A: High Precision Layer (LoRA Compatible)
                if getattr(self, 'is_high_precision_layer', False):
                    # Strictly adhere to comfy.ops.manual_cast logic
                    weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
                    out = torch.nn.functional.linear(input, weight, bias)
                    uncast_bias_weight(self, weight, bias, offload_stream)
                    return out

                # PATH B: FP8 Layer (Scaled FP8 Logic)
                if fp8_mat_mult_supported:
                    out = fp8_linear(self, input)
                    if out is not None:
                        return out

                # Manual FP8 Cast Logic
                weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)

                if weight.numel() < input.numel():
                    x = torch.nn.functional.linear(input, weight * self.scale_weight.to(device=weight.device, dtype=weight.dtype), bias)
                else:
                    x = torch.nn.functional.linear(input * self.scale_weight.to(device=weight.device, dtype=weight.dtype), weight, bias)
                uncast_bias_weight(self, weight, bias, offload_stream)
                return x

            def convert_weight(self, weight, inplace=False, **kwargs):
                # If this is a high precision layer, scale_weight is None.
                # We simply return the weight as-is.
                if self.scale_weight is None:
                    self.scale_weight = torch.nn.parameter.Parameter(data=torch.ones((), device=self.weight.device, dtype=torch.float32), requires_grad=False)
                    # FP8 Logic
                    if inplace:
                        weight *= self.scale_weight.to(device=weight.device, dtype=weight.dtype)
                        return weight
                    else:
                        return weight.to(dtype=torch.float32) * self.scale_weight.to(device=weight.device, dtype=torch.float32)
                else:
                    if inplace:
                        weight *= self.scale_weight.to(device=weight.device, dtype=weight.dtype)
                        return weight
                    else:
                        return weight.to(dtype=torch.float32) * self.scale_weight.to(device=weight.device, dtype=torch.float32)

            def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
                # If this is a high precision layer, scale_weight is None.
                # We skip the FP8 stochastic rounding/scaling.
                if self.scale_weight is None:
                    if return_weight:
                        return weight
                    if inplace_update:
                        self.weight.data.copy_(weight)
                    else:
                        self.weight = torch.nn.Parameter(weight, requires_grad=False)
                    return

                # FP8 Logic
                weight = comfy.float.stochastic_rounding(weight / self.scale_weight.to(device=weight.device, dtype=weight.dtype), self.weight.dtype, seed=seed)
                if return_weight:
                    return weight
                if inplace_update:
                    self.weight.data.copy_(weight)
                else:
                    self.weight = torch.nn.Parameter(weight, requires_grad=False)

    return HybridOps

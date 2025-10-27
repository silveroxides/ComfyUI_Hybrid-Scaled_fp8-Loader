# file: ComfyUI/custom_nodes/HybridFP8Loader/hybrid_fp8_ops.py

import torch
import comfy.ops
import comfy.model_management

TARGET_FP8_DTYPE = torch.float8_e4m3fn
_high_precision_keynames = []

def set_high_precision_keynames(keynames):
    """Sets the list of substrings used to identify high-precision layers."""
    global _high_precision_keynames
    _high_precision_keynames = keynames
    print(f"[Hybrid FP8 Ops] High precision keynames set: {keynames}")

def get_hybrid_fp8_ops(scale_input_enabled=False):
    """
    Dynamically creates and returns a hybrid operations class.
    The 'scale_input_enabled' flag is now passed in from the loader.
    """
    print(f"[Hybrid FP8 Ops] Configuring with scale_input_enabled: {scale_input_enabled}")

    fp8_mat_mult_supported = comfy.model_management.supports_fp8_compute()

    base_ops_class = comfy.ops.scaled_fp8_ops(
        fp8_matrix_mult=fp8_mat_mult_supported,
        scale_input=scale_input_enabled,
        override_dtype=TARGET_FP8_DTYPE
    )

    class HybridScaledFP8Linear(base_ops_class.Linear):
        """
        A Linear layer that intelligently handles both scaled FP8 and high-precision weights.
        It now dynamically detects block-wise scaling based on the shape of the scale_weight tensor.
        """
        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            is_excluded = any(name in prefix for name in _high_precision_keynames)

            if is_excluded and _high_precision_keynames:
                print(f"[Hybrid FP8 Ops] Intercepting high-precision layer: {prefix}")

                weight_key = prefix + 'weight'
                bias_key = prefix + 'bias'

                weight_tensor = state_dict.pop(weight_key, None)
                bias_tensor = state_dict.pop(bias_key, None)

                if weight_tensor is None:
                    missing_keys.append(weight_key)
                else:
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)

                if bias_tensor is not None:
                    self.bias = torch.nn.Parameter(bias_tensor, requires_grad=False)
                else:
                    self.bias = None

                state_dict.pop(prefix + 'scale_weight', None)
                state_dict.pop(prefix + 'scale_input', None)

                self.scale_weight = None
                self.scale_input = None

                setattr(self, 'is_high_precision_layer', True)
            else:
                # --- THE FIX ---
                # Before calling the main loader, we must ensure the `scale_weight`
                # parameter on this model layer has the correct shape. The base class
                # initializes it as an empty tensor, causing a size mismatch.

                scale_weight_key = prefix + 'scale_weight'
                if scale_weight_key in state_dict:
                    # Get the tensor from the file's state_dict
                    scale_tensor_from_file = state_dict[scale_weight_key]

                    # Re-create the parameter on this layer with the correct shape and device.
                    # This pre-empts the size mismatch error in the super() call.
                    self.scale_weight = torch.nn.Parameter(
                        torch.empty_like(scale_tensor_from_file), requires_grad=False
                    )

                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
                setattr(self, 'is_high_precision_layer', False)

        def forward(self, input):
            # Case 1: High-precision layer (not quantized)
            if getattr(self, 'is_high_precision_layer', False):
                weight_hp = self.weight.to(input.device, input.dtype)
                bias_hp = self.bias.to(input.device, input.dtype) if self.bias is not None else None
                return torch.nn.functional.linear(input, weight_hp, bias_hp)

            # Case 2: Quantized layer (handle all scaling types)
            else:
                # Handle input scaling for T5-style models first
                if self.scale_input is not None:
                    input = input / self.scale_input.to(input.device, input.dtype)

                # Get weight, scale, and bias tensors ready for compute
                weight = self.weight.to(input.dtype)
                scale = self.scale_weight.to(input.dtype)
                bias = self.bias.to(input.dtype) if self.bias is not None else None

                dequantized_weight = None

                # Check for block-wise scaling (scale tensor has 3 dimensions)
                if scale.ndim == 3:
                    out_features, num_blocks, _ = scale.shape
                    # Reshape weight, multiply by scale, then reshape back to original
                    dequantized_weight = weight.view(out_features, num_blocks, -1) * scale
                    dequantized_weight = dequantized_weight.view(self.weight.shape)
                # Fallback to standard tensor or vector scaling
                else:
                    dequantized_weight = weight * scale

                # Perform the linear operation
                return torch.nn.functional.linear(input, dequantized_weight, bias)

    class HybridOps(base_ops_class):
        class Linear(HybridScaledFP8Linear):
            pass

    return HybridOps
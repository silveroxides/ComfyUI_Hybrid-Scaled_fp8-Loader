# file: ComfyUI/custom_nodes/HybridFP8Loader/hybrid_fp8_ops.py

import torch
import torch.nn as nn
import comfy.ops
import comfy.model_management as mm

from .cpu_offload_ops import BouncingLinearFn

TARGET_FP8_DTYPE = torch.float8_e4m3fn
_high_precision_keynames = []

def set_high_precision_keynames(keynames):
    """Sets the list of substrings used to identify high-precision layers."""
    global _high_precision_keynames
    _high_precision_keynames = keynames
    print(f"[Hybrid FP8 Ops] High precision keynames set: {keynames}")

def get_hybrid_fp8_ops(scale_input_enabled=False, cpu_offload_enabled=False):
    """
    Dynamically creates and returns a hybrid operations class.
    The 'scale_input_enabled' flag is now passed in from the loader.
    """
    print(f"[Hybrid FP8 Ops] Configuring with: scale_input={scale_input_enabled}, cpu_offload={cpu_offload_enabled}")

    fp8_mat_mult_supported = mm.supports_fp8_compute()

    base_ops_class = comfy.ops.scaled_fp8_ops(
        fp8_matrix_mult=fp8_mat_mult_supported,
        scale_input=scale_input_enabled,
        override_dtype=TARGET_FP8_DTYPE
    )

    class HybridScaledFP8Linear(base_ops_class.Linear):
        """
        A Linear layer that intelligently handles both scaled FP8 and high-precision weights.
        """
        def __init__(self, *args, **kwargs):
            # In CPU offload mode, we don't want the FP8 override.
            # We will handle weight creation manually in _load_from_state_dict.
            if cpu_offload_enabled:
                nn.Module.__init__(self)
            else:
                super().__init__(*args, **kwargs)
            self.device = mm.get_torch_device()

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if cpu_offload_enabled:
                print(f"[Hybrid FP8 Ops] Intercepting for CPU Offload: {prefix}")
                weight_key = prefix + 'weight'
                bias_key = prefix + 'bias'

                weight_tensor = state_dict.pop(weight_key, None)
                bias_tensor = state_dict.pop(bias_key, None)

                if weight_tensor is None:
                    missing_keys.append(weight_key)
                else:
                    w_cpu = weight_tensor.to(device='cpu', non_blocking=True).pin_memory()
                    self.weight = nn.Parameter(w_cpu, requires_grad=False)
                
                if bias_tensor is not None:
                    b_cpu = bias_tensor.to(device='cpu', non_blocking=True).pin_memory()
                    self.bias = nn.Parameter(b_cpu, requires_grad=False)
                else:
                    self.bias = None

                state_dict.pop(prefix + 'scale_weight', None)
                state_dict.pop(prefix + 'scale_input', None)
                
                setattr(self, 'is_cpu_offloaded', True)
                return

            is_excluded = any(name in prefix for name in _high_precision_keynames)

            if is_excluded and _high_precision_keynames:
                print(f"[Hybrid FP8 Ops] Intercepting high-precision layer: {prefix}")
                
                weight_key, bias_key = prefix + 'weight', prefix + 'bias'
                weight_tensor = state_dict.pop(weight_key, None)
                bias_tensor = state_dict.pop(bias_key, None)

                if weight_tensor is not None: self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                else: missing_keys.append(weight_key)
                
                if bias_tensor is not None: self.bias = nn.Parameter(bias_tensor, requires_grad=False)
                else: self.bias = None

                state_dict.pop(prefix + 'scale_weight', None)
                state_dict.pop(prefix + 'scale_input', None)
                self.scale_weight = None
                self.scale_input = None

                setattr(self, 'is_high_precision_layer', True)
            else:
                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        def forward(self, input):
            if getattr(self, 'is_cpu_offloaded', False):
                return BouncingLinearFn.apply(input, self.weight, self.bias, self.device)
            elif getattr(self, 'is_high_precision_layer', False):
                weight_hp = self.weight.to(input.device, input.dtype)
                bias_hp = self.bias.to(input.device, input.dtype) if self.bias is not None else None
                return torch.nn.functional.linear(input, weight_hp, bias_hp)
            else:
                return super().forward(input)

    class HybridOps(base_ops_class):
        class Linear(HybridScaledFP8Linear):
            pass

    return HybridOps

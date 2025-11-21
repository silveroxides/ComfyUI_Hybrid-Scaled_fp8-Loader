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

def get_hybrid_fp8_ops(scale_input_enabled=False, is_blockwise=False, disable_fp8_mat_mult=False):
    """
    Dynamically creates and returns a hybrid operations class.
    """
    print(f"[Hybrid FP8 Ops] Config: scale_input={scale_input_enabled}, is_blockwise={is_blockwise}, disable_fp8_mat_mult={disable_fp8_mat_mult}")

    if disable_fp8_mat_mult == False:
       fp8_mat_mult_supported = comfy.model_management.supports_fp8_compute()
    else:
        fp8_mat_mult_supported = False

    # Generate Base Class
    base_ops_class = comfy.ops.scaled_fp8_ops(
        fp8_matrix_mult=fp8_mat_mult_supported,
        scale_input=scale_input_enabled,
        override_dtype=TARGET_FP8_DTYPE
    )

    class HybridScaledFP8Linear(base_ops_class.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.is_blockwise_model = is_blockwise

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            # 1. High Precision Exclusion
            is_excluded = any(name in prefix for name in _high_precision_keynames)
            if is_excluded and _high_precision_keynames:
                weight_key = prefix + 'weight'
                bias_key = prefix + 'bias'

                if weight_key in state_dict:
                    # Manually load as Parameter to avoid FP8 casting in super()
                    self.weight = torch.nn.Parameter(state_dict.pop(weight_key), requires_grad=False)
                else:
                    missing_keys.append(weight_key)

                if bias_key in state_dict:
                    self.bias = torch.nn.Parameter(state_dict.pop(bias_key), requires_grad=False)
                else:
                    self.bias = None

                # Clean up unused buffers
                state_dict.pop(prefix + 'scale_weight', None)
                state_dict.pop(prefix + 'scale_input', None)
                self.scale_weight = None
                self.scale_input = None
                
                setattr(self, 'is_high_precision_layer', True)
                return

            # 2. Standard / Blockwise Loading
            setattr(self, 'is_high_precision_layer', False)

            # Resize scale_weight parameter if the file contains a block-wise (3D) tensor
            scale_weight_key = prefix + 'scale_weight'
            if scale_weight_key in state_dict:
                scale_tensor = state_dict[scale_weight_key]
                # If parameter exists but shape mismatches (scalar vs 3D), resize it
                if self.scale_weight is not None and self.scale_weight.shape != scale_tensor.shape:
                    self.scale_weight = torch.nn.Parameter(torch.empty_like(scale_tensor), requires_grad=False)

            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        def forward_comfy_cast_weights(self, input):
            # Path 1: High Precision Layer
            if getattr(self, 'is_high_precision_layer', False):
                weight, bias, offload_stream = comfy.ops.cast_bias_weight(self, input, offloadable=True)
                x = torch.nn.functional.linear(input, weight, bias)
                comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)
                return x

            # Path 2: Standard / Blockwise FP8
            weight, bias, offload_stream = comfy.ops.cast_bias_weight(self, input, offloadable=True)

            # LoRA Guard: Check if weight is still FP8 (Raw) or BF16 (Patched by LoRA)
            is_fp8_weight = weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

            if is_fp8_weight:
                # --- BLOCKWISE LOGIC ---
                if self.is_blockwise_model and self.scale_weight.ndim == 3:
                    # 1. Cast Weight to Input Dtype (BF16/FP16) FIRST.
                    # Why? Direct FP8 * FP16 broadcast multiplication often fails (returns zeros) 
                    # or isn't supported by PyTorch kernels for transposed views.
                    weight_hp = weight.to(dtype=input.dtype)
                    
                    # 2. Get Scale
                    scale = self.scale_weight.to(device=weight.device, dtype=input.dtype)
                    
                    # 3. Reshape & Broadcast Multiply
                    # Scale shape: [Out, Blocks, 1]
                    # Weight View: [Out, Blocks, BlockSize]
                    out_features, num_blocks, _ = scale.shape
                    
                    # Note: -1 automatically handles the BlockSize dimension
                    dq_weight = weight_hp.view(out_features, num_blocks, -1) * scale
                    
                    # 4. Flatten back to [Out, In]
                    dq_weight = dq_weight.reshape(weight.shape)
                    
                    # 5. Linear Op
                    # Note: We skip scale_input here as you mentioned block-wise models don't use it.
                    x = torch.nn.functional.linear(input, dq_weight, bias)

                # --- STANDARD SCALAR/TENSOR LOGIC ---
                else:
                    # Apply Input Scaling (Only for non-blockwise/Standard FP8)
                    if self.scale_input is not None:
                        input = input * self.scale_input.to(device=input.device, dtype=input.dtype)

                    scale = self.scale_weight.to(device=weight.device, dtype=input.dtype)
                    
                    # Standard dequantization
                    if weight.numel() < input.numel():
                        x = torch.nn.functional.linear(input, weight * scale, bias)
                    else:
                        x = torch.nn.functional.linear(input * scale, weight, bias)
            else:
                # Path 3: Patched/LoRA (Already Dequantized to BF16/FP32)
                x = torch.nn.functional.linear(input, weight, bias)

            comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)
            return x

    class HybridOps(base_ops_class):
        class Linear(HybridScaledFP8Linear):
            pass

    return HybridOps
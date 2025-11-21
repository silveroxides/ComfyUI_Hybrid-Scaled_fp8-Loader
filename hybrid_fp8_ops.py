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
    print(f"[Hybrid FP8 Ops] Configuring - scale_input: {scale_input_enabled}, blockwise: {is_blockwise}")

    if disable_fp8_mat_mult == False:
       fp8_mat_mult_supported = comfy.model_management.supports_fp8_compute()
    else:
        fp8_mat_mult_supported = False

    # Generate the base Comfy class
    base_ops_class = comfy.ops.scaled_fp8_ops(
        fp8_matrix_mult=fp8_mat_mult_supported,
        scale_input=scale_input_enabled,
        override_dtype=TARGET_FP8_DTYPE
    )

    class HybridScaledFP8Linear(base_ops_class.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Determine if we expect blockwise behavior based on the loader flag
            self.is_blockwise_model = is_blockwise

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            # 1. Check for Exclusion (High Precision)
            is_excluded = any(name in prefix for name in _high_precision_keynames)
            if is_excluded and _high_precision_keynames:
                # Handle High Precision Conversion
                weight_key = prefix + 'weight'
                bias_key = prefix + 'bias'

                if weight_key in state_dict:
                    # Convert to Parameter manually to prevent base class interference
                    self.weight = torch.nn.Parameter(state_dict.pop(weight_key), requires_grad=False)
                else:
                    missing_keys.append(weight_key)

                if bias_key in state_dict:
                    self.bias = torch.nn.Parameter(state_dict.pop(bias_key), requires_grad=False)
                else:
                    self.bias = None

                # Remove scaling keys from state_dict so they don't trigger unexpected key errors
                state_dict.pop(prefix + 'scale_weight', None)
                state_dict.pop(prefix + 'scale_input', None)
                
                # Nullify internal buffers
                self.scale_weight = None
                self.scale_input = None
                
                setattr(self, 'is_high_precision_layer', True)
                return # Skip super() for excluded layers

            # 2. Prepare for Quantized Loading
            setattr(self, 'is_high_precision_layer', False)

            scale_weight_key = prefix + 'scale_weight'
            
            # DYNAMIC BUFFER RESIZING
            # Comfy's base class initializes scale_weight as a scalar (0-dim).
            # If we are loading a block-wise model, the state_dict has a 3D tensor.
            # We must resize the parameter *before* super().load() or it will crash.
            if scale_weight_key in state_dict:
                scale_tensor = state_dict[scale_weight_key]
                
                # If the shape in file differs from current parameter shape
                if self.scale_weight is not None and self.scale_weight.shape != scale_tensor.shape:
                    self.scale_weight = torch.nn.Parameter(
                        torch.empty_like(scale_tensor), requires_grad=False
                    )

            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        def forward_comfy_cast_weights(self, input):
            # 1. High Precision Path
            if getattr(self, 'is_high_precision_layer', False):
                # Standard Linear execution
                weight, bias, offload_stream = comfy.ops.cast_bias_weight(self, input, offloadable=True)
                x = torch.nn.functional.linear(input, weight, bias)
                comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)
                return x

            # 2. FP8 / LoRA Path
            # We rely on Comfy's casting utility to move weights to GPU
            weight, bias, offload_stream = comfy.ops.cast_bias_weight(self, input, offloadable=True)

            # --- LORA GUARD ---
            # Check if the weight is actually FP8. 
            # If a LoRA is active, Comfy might have already converted this to BF16/Float32.
            is_fp8_weight = weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

            if is_fp8_weight:
                # Apply Input Scaling if enabled
                if self.scale_input is not None:
                    # Note: Comfy standard usually multiplies, but your snippet showed division.
                    # Adjust operator based on specific model requirements (e.g. Flux vs SDXL).
                    # Assuming standard behavior:
                    scale_input = self.scale_input.to(device=input.device, dtype=input.dtype)
                    input = input * scale_input 

                scale = self.scale_weight.to(device=weight.device, dtype=input.dtype)
                
                # --- BLOCKWISE LOGIC ---
                # Detect based on tensor shape (3D) or explicit flag
                if scale.ndim == 3:
                    out_features, num_blocks, block_size = scale.shape
                    # View as (Out, Blocks, BlockSize) - automatically infer block size from weight if needed
                    # Note: weight.view(...) is fast (metadata only), no data copy
                    dq_weight = weight.view(out_features, num_blocks, -1) * scale
                    dq_weight = dq_weight.reshape(weight.shape)
                    
                    x = torch.nn.functional.linear(input, dq_weight, bias)
                else:
                    # Standard Scalar Scaling (Standard Comfy FP8)
                    if weight.numel() < input.numel():
                        x = torch.nn.functional.linear(input, weight * scale, bias)
                    else:
                        x = torch.nn.functional.linear(input * scale, weight, bias)

            else:
                # --- PATCHED PATH (LoRA) ---
                # The weight is likely BF16/Float32. It contains (Dequantized_Base + LoRA).
                # DO NOT scale it again. Just run.
                x = torch.nn.functional.linear(input, weight, bias)

            # 3. Uncast
            comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)
            return x

    class HybridOps(base_ops_class):
        class Linear(HybridScaledFP8Linear):
            pass

    return HybridOps
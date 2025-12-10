# file: ComfyUI/custom_nodes/HybridFP8Loader/hybrid_fp8_ops.py

import torch
from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight, disable_weight_init
from comfy.quant_ops import QuantizedTensor
import comfy.model_management
import comfy.float
import logging
import os
import concurrent.futures
from . import utils


# Dictionary mapping specific full tensor keys to their desired dtype
_high_precision_keys = {} # Format: { "full.layer.name.weight": torch.float16, ... }

# Global configuration state
_scale_input_enabled = False
_fp8_mat_mult_supported = False
_configured = False
_log_high_precision = False
_use_mmap_for_state_dict = True
_state_dict_worker_override = None
_state_dict_worker_override_enabled = False


def _strip_scaled_fp8_hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    # Remove any global auxiliary tensor that may remain in the checkpoint
    state_dict.pop('scaled_fp8', None)
    state_dict.pop(prefix + 'scaled_fp8', None)


def set_high_precision_keys(key_dtype_map):
    """Sets the dictionary of {full_tensor_name: dtype}."""
    global _high_precision_keys
    _high_precision_keys = key_dtype_map
    print(f"[Hybrid FP8 Ops] High precision overrides set for {len(key_dtype_map)} layers.")


def configure_hybrid_ops(model_path, model_type="none", force_fp8_matmul=False, scale_input_enabled=None, disable_fp8_mat_mult=False, debug_metadata=False, guard_no_tensor_read=False, log_high_precision=False):
    """
    Configure hybrid FP8 operations by inspecting model metadata.
    This sets global configuration state without loading any model tensors.

    Args:
        model_path: Path to the safetensors file (for metadata inspection)
        model_type: Type of model to configure high-precision layers for
        force_fp8_matmul: Force enable FP8 matrix multiplication
        scale_input_enabled: Override for scale_input detection (auto-detect if None)
        disable_fp8_mat_mult: Force disable FP8 matrix multiplication
    """
    global _scale_input_enabled, _fp8_mat_mult_supported, _configured
    global _log_high_precision

    # Model type exclusion lists
    DISTILL_LAYER_KEYNAMES_LARGE = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]
    NERF_LAYER_KEYNAMES_LARGE = ["distilled_guidance_layer", "img_in_patch", "nerf_blocks", "nerf_final_layer_conv", "nerf_image_embedder", "txt_in"]
    DISTILL_LAYER_KEYNAMES_SMALL = ["distilled_guidance_layer"]
    NERF_LAYER_KEYNAMES_SMALL = ["distilled_guidance_layer", "img_in_patch", "nerf_blocks", "nerf_final_layer_conv", "nerf_image_embedder"]
    WAN_LAYER_KEYNAMES = [
        "patch_embedding", "ref_conv", "control_adapter", "motion_encoder.enc.net_app",
        "face_encoder.conv", "pose_patch_embedding", "text_embedding", "time_embedding",
        "time_projection", "head.head", "img_emb.proj", "motion_encoder.dec",
        "motion_encoder.enc.fc", "face_encoder.out_proj", "face_adapter"
    ]
    PONYV7_LAYER_KEYNAMES = ["t_embedder", "cond_seq_linear", "final_linear", "init_x_linear", "modF", "positional_encoding", "register_tokens"]
    QWEN_LAYER_KEYNAMES = ["time_text_embed", "img_in", "norm_out", "proj_out", "txt_in", "norm_added_k", "norm_added_q", "norm_k", "norm_q", "txt_norm"]
    HUNYUAN_LAYER_KEYNAMES = ["layernorm", "img_attn_k_norm", "img_attn_q_norm", "txt_attn_k_norm", "txt_attn_q_norm", "norm1", "norm2", "vision_in.proj.0", "vision_in.proj.4", "img_in.proj", "cond_type_embedding"]
    ZIMAGE_LAYER_KEYNAMES = ["cap_embedder.0", "attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2", "norm_k", "norm_q", "norm1", "norm2"]

    # Detect configuration from model file if path provided
    if model_path is not None:
        excluded_layers = []
        local_disable_fp8_mat_mult = False

        if model_type == "chroma_hybrid_large":
            excluded_layers.extend(DISTILL_LAYER_KEYNAMES_LARGE)
        elif model_type == "radiance_hybrid_large":
            excluded_layers.extend(NERF_LAYER_KEYNAMES_LARGE)
        elif model_type == "chroma_hybrid_small":
            excluded_layers.extend(DISTILL_LAYER_KEYNAMES_SMALL)
        elif model_type == "radiance_hybrid_small":
            excluded_layers.extend(NERF_LAYER_KEYNAMES_SMALL)
        elif model_type == "wan":
            excluded_layers.extend(WAN_LAYER_KEYNAMES)
        elif model_type == "pony_diffusion_v7":
            excluded_layers.extend(PONYV7_LAYER_KEYNAMES)
        elif model_type == "qwen":
            excluded_layers.extend(QWEN_LAYER_KEYNAMES)
            local_disable_fp8_mat_mult = True
        elif model_type == "hunyuan":
            excluded_layers.extend(HUNYUAN_LAYER_KEYNAMES)
        elif model_type == "zimage":
            excluded_layers.extend(ZIMAGE_LAYER_KEYNAMES)

        # Remove duplicates
        high_precision_substrings = list(set(excluded_layers))

        # Detect FP8 optimizations from file metadata (no tensor loading)
        detected_scale_input = False
        excluded_layers_dtype = {}

        # Use CallableMemEffSafeOpen for better metadata access
        # Read all metadata and close file immediately
        try:
            # Optional guard that forbids tensor body reads during configure.
            if guard_no_tensor_read:
                class _HeaderOnly(utils.CallableMemEffSafeOpen):
                    def get_tensor(self, key):
                        raise RuntimeError("Header-only guard: tensor reads are blocked during configure_hybrid_ops.")
                    def get_tensor_as_dict(self, key):
                        raise RuntimeError("Header-only guard: tensor reads are blocked during configure_hybrid_ops.")
                    def get_tensor_info(self, pattern, as_dict=False):
                        raise RuntimeError("Header-only guard: tensor reads are blocked during configure_hybrid_ops.")
                open_cls = _HeaderOnly
            else:
                open_cls = utils.CallableMemEffSafeOpen

            with open_cls(model_path, device="cpu") as f:
                all_tensors = f.list_keys(show_dtype=True, show_shape=True).copy()
            if debug_metadata:
                logging.info("[Hybrid FP8 Loader] Header-only metadata scan read %d entries (shapes+dtypes); file closed immediately.", len(all_tensors))
        except Exception as e:
            print(f"[Hybrid FP8 Loader] Warning: Could not inspect model file: {e}")
            all_tensors = []

        # Build dtype conversion map
        dtype_str_to_torch = {
            'torch.float64': torch.float64, 'torch.float32': torch.float32,
            'torch.float16': torch.float16, 'torch.bfloat16': torch.bfloat16,
            'torch.int64': torch.int64, 'torch.int32': torch.int32,
            'torch.int16': torch.int16, 'torch.int8': torch.int8,
            'torch.uint8': torch.uint8, 'torch.bool': torch.bool,
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_str_to_torch['torch.float8_e5m2'] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_str_to_torch['torch.float8_e4m3fn'] = torch.float8_e4m3fn

        # Process tensors to find high-precision layers (file is now closed)
        for tensor_info in all_tensors:
            key = tensor_info['key']

            # Detect scale_input mode by checking scaled_fp8 tensor shape
            if key == 'scaled_fp8':
                shape = tensor_info.get('shape', [1])
                if shape[0] == 0:
                    print("[Hybrid FP8 Loader] Scale Input model detected (scale_input enabled).")
                    detected_scale_input = True
                else:
                    print("[Hybrid FP8 Loader] Standard UNet-style model detected (scale_input disabled).")
                continue

            # Skip non-weight keys
            if not key.endswith('.weight'):
                continue

            # Check if this key matches any high-precision substring
            for substring in high_precision_substrings:
                if substring in key:
                    dtype_str = tensor_info['dtype']
                    excluded_layers_dtype[key] = dtype_str_to_torch.get(dtype_str, torch.float32)
                    break

        # Strip common prefixes to match ComfyUI's internal key format
        excluded_layers_dtype_stripped = {}
        common_prefixes = ["model.diffusion_model.", "model.", "diffusion_model."]

        for key, dtype in excluded_layers_dtype.items():
            stripped_key = key
            for prefix in common_prefixes:
                if key.startswith(prefix):
                    stripped_key = key[len(prefix):]
                    break
            excluded_layers_dtype_stripped[stripped_key] = dtype

        print(f"[Hybrid FP8 Loader] Mapped {len(excluded_layers_dtype_stripped)} high-precision layers")

        # Set high precision keys globally
        set_high_precision_keys(excluded_layers_dtype_stripped)

        # Use detected scale_input if not overridden
        if scale_input_enabled is None:
            scale_input_enabled = detected_scale_input

        # Apply model-specific settings
        if local_disable_fp8_mat_mult:
            disable_fp8_mat_mult = True
    else:
        # No model path provided, use defaults
        if scale_input_enabled is None:
            scale_input_enabled = False

    print(f"[Hybrid FP8 Ops] Configuring with scale_input_enabled: {scale_input_enabled}")

    if disable_fp8_mat_mult == False:
       fp8_mat_mult_supported = comfy.model_management.supports_fp8_compute()
    elif force_fp8_matmul == True:
        fp8_mat_mult_supported = True
    else:
        fp8_mat_mult_supported = False

    logging.info("Using hybrid fp8: fp8 matrix mult: {}, scale input: {}".format(fp8_mat_mult_supported, scale_input_enabled))

    # Set global configuration
    _scale_input_enabled = scale_input_enabled
    _fp8_mat_mult_supported = fp8_mat_mult_supported
    _configured = True
    _log_high_precision = log_high_precision


def _log_hp(message):
    if _log_high_precision:
        logging.info(message)


def set_state_dict_mmap(enabled: bool):
    """Toggle mmap usage for safetensors state_dict loading."""
    global _use_mmap_for_state_dict
    _use_mmap_for_state_dict = bool(enabled)
    logging.info(f"[Hybrid FP8 Loader] mmap for state_dict is {'enabled' if _use_mmap_for_state_dict else 'disabled'}")


def set_state_dict_workers(count: int, enabled: bool):
    """Set manual worker override for state_dict loading."""
    global _state_dict_worker_override, _state_dict_worker_override_enabled
    _state_dict_worker_override_enabled = bool(enabled)
    if enabled:
        try:
            _state_dict_worker_override = max(1, min(16, int(count)))
        except Exception:
            _state_dict_worker_override = None
    else:
        _state_dict_worker_override = None
    logging.info(f"[Hybrid FP8 Loader] worker override is {'enabled' if _state_dict_worker_override_enabled else 'disabled'}; value={_state_dict_worker_override}")


def load_unet_state_dict(model_path):
    """Load a UNet state dict from safetensors without the scaled_fp8 aux tensor."""
    sd = {}
    metadata = None
    try:
        header, header_size = utils.MemoryEfficientSafeOpen._read_header(model_path)
        metadata = header.get("__metadata__", None)
        keys = [k for k in header.keys() if k not in ("__metadata__", "scaled_fp8")]

        # Precompute tensor sizes and offsets for planning.
        dtype_sizes = {
            'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2,
            'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1,
            'U8': 1, 'BOOL': 1,
            'F8_E5M2': 1, 'F8_E4M3': 1
        }

        entries = []  # (key, start, end, nbytes)
        total_bytes = 0
        for key in keys:
            md = header[key]
            start, end = md["data_offsets"]
            numel = 1
            for dim in md["shape"]:
                numel *= dim
            nbytes = numel * dtype_sizes.get(md["dtype"], 4)
            entries.append((key, start, end, nbytes))
            total_bytes += nbytes

        # Read order by file offset to keep access mostly sequential.
        entries.sort(key=lambda x: x[1])

        # Memory-aware fallback: if estimated model size exceeds free RAM, avoid parallelism.
        cpu_device = comfy.model_management.torch.device("cpu")
        ram_free = comfy.model_management.get_free_memory(cpu_device)
        if _state_dict_worker_override_enabled and _state_dict_worker_override is not None:
            planned_workers = _state_dict_worker_override
        else:
            planned_workers = max(1, min(4, os.cpu_count() or 4))

        worker_count = planned_workers if _use_mmap_for_state_dict else 1
        if ram_free is not None and total_bytes > 0.75 * ram_free:
            worker_count = 1
            logging.info(f"[Hybrid FP8 Loader] Parallel load disabled due to RAM pressure (need~{total_bytes/1e9:.2f}GB, free~{ram_free/1e9:.2f}GB)")

        logging.info(f"[Hybrid FP8 Loader] Loading state_dict with mmap={'on' if _use_mmap_for_state_dict else 'off'} workers={worker_count} tensors={len(entries)} total={total_bytes/1e9:.2f}GB free_ram={(ram_free/1e9 if ram_free else 0):.2f}GB override={_state_dict_worker_override if _state_dict_worker_override_enabled else 'auto'}")

        # Map the entire file and reuse the parsed header for faster sequential loads.
        with utils.MemoryEfficientSafeOpen(model_path, device="cpu", use_mmap=_use_mmap_for_state_dict, header_cache=(header, header_size)) as f:
            # If mmap is unavailable or disabled, fall back to sequential.
            if not _use_mmap_for_state_dict or f.mmap_obj is None or worker_count == 1:
                for key, _, _, _ in entries:
                    sd[key] = f.get_tensor(key)
            else:
                # Parallel load with preallocated result slots.
                results = [None] * len(entries)

                def _load(idx_entry):
                    idx, (key, _, _, _) = idx_entry
                    tensor = f.get_tensor(key)
                    results[idx] = (key, tensor)

                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as ex:
                    ex.map(_load, enumerate(entries))

                for key, tensor in results:
                    sd[key] = tensor
    except Exception as e:
        logging.error(f"[Hybrid FP8 Loader] Failed lazy load state_dict: {e}")
        raise
    return sd, metadata


def load_unet_lazy(model_path):
    """Lazy load UNet via state_dict using safetensors reader and comfy load_diffusion_model_state_dict."""
    sd, metadata = load_unet_state_dict(model_path)
    return sd, metadata

def fp8_linear(self, input):
    """
    Legacy FP8 linear function for backward compatibility.
    Uses QuantizedTensor subclass for dispatch.
    """
    # Get the actual weight data
    weight = self.weight
    if isinstance(weight, torch.nn.Parameter):
        weight = weight.data
    
    # If weight is already a QuantizedTensor, use it directly
    if isinstance(weight, QuantizedTensor):
        input_dtype = input.dtype
        
        if input.ndim == 3 or input.ndim == 2:
            # Prepare input as QuantizedTensor
            scale_input = self.scale_input
            if scale_input is None:
                scale_input = torch.ones((), device=input.device, dtype=torch.float32)
                inp_clamped = torch.clamp(input, min=-448, max=448)
                layout_params_input = {'scale': scale_input, 'orig_dtype': input_dtype}
                quantized_input = QuantizedTensor(inp_clamped.to(torch.float8_e4m3fn).contiguous(), "TensorCoreFP8Layout", layout_params_input)
            else:
                scale_input = scale_input.to(input.device)
                quantized_input = QuantizedTensor.from_float(input, "TensorCoreFP8Layout", scale=scale_input, dtype=torch.float8_e4m3fn)
            
            # Use the weight QuantizedTensor directly
            bias = self.bias
            if bias is not None:
                bias = bias.to(device=input.device, dtype=input_dtype)
            
            o = torch.nn.functional.linear(quantized_input, weight, bias)
            return o
        return None
    
    # Raw FP8 tensor path
    dtype = weight.dtype
    if dtype not in [torch.float8_e4m3fn]:
        return None

    input_dtype = input.dtype

    if input.ndim == 3 or input.ndim == 2:
        w, bias, offload_stream = cast_bias_weight(self, input, dtype=dtype, bias_dtype=input_dtype, offloadable=True)

        scale_weight = self.scale_weight
        scale_input = self.scale_input
        if scale_weight is None:
            scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
        else:
            scale_weight = scale_weight.to(input.device)

        if scale_input is None:
            scale_input = torch.ones((), device=input.device, dtype=torch.float32)
            inp_clamped = torch.clamp(input, min=-448, max=448)
            layout_params_input = {'scale': scale_input, 'orig_dtype': input_dtype}
            quantized_input = QuantizedTensor(inp_clamped.to(dtype).contiguous(), "TensorCoreFP8Layout", layout_params_input)
        else:
            scale_input = scale_input.to(input.device)
            quantized_input = QuantizedTensor.from_float(input, "TensorCoreFP8Layout", scale=scale_input, dtype=dtype)

        # Wrap weight in QuantizedTensor without re-quantizing (w is already FP8);
        # attach scale/orig_dtype so quant_ops TensorCore handler can use _scaled_mm.
        layout_params_weight = {'scale': scale_weight, 'orig_dtype': input_dtype}
        quantized_weight = QuantizedTensor(w, "TensorCoreFP8Layout", layout_params_weight)
        o = torch.nn.functional.linear(quantized_input, quantized_weight, bias)

        uncast_bias_weight(self, w, bias, offload_stream)
        return o

    return None


class HybridOps(manual_cast):
    """
    Hybrid operations class combining Scaled FP8 logic with High-Precision overrides.
    Uses global configuration set by configure_hybrid_ops().

    IMPORTANT: We do NOT set a class-level `dtype` attribute here. This prevents
    ComfyUI from greedily casting all layers to FP8. Each Linear layer decides
    its own dtype based on _high_precision_keys during state dict loading.
    """

    class Linear(manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            # Do NOT force FP8 dtype here - let state dict loading determine dtype
            super().__init__(*args, **kwargs)
            # Initialize scale attributes
            self.scale_weight = None
            self.scale_input = None
            self.is_high_precision_layer = False

        def reset_parameters(self):
            return None

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            # Drop global auxiliary tensor if present
            state_dict.pop('scaled_fp8', None)

            # Construct the key as it appears in the map (e.g. "blocks.0.attn.weight")
            weight_key = prefix + 'weight'

            # Determine TARGET_DTYPE based on key existence in the map
            if weight_key in _high_precision_keys:
                target_dtype = _high_precision_keys[weight_key]
                # Strip FP8 scale keys eagerly so ComfyUI does not attempt FP8 conversion
                state_dict.pop(prefix + 'scale_weight', None)
                state_dict.pop(prefix + 'scale_input', None)
                self.is_high_precision_layer = True
                self.scale_weight = None
                self.scale_input = None
                _log_hp(f"[Hybrid FP8 Loader] Layer {weight_key} marked high-precision with dtype={target_dtype}")
            else:
                # FP8 layer - load scales if present
                target_dtype = torch.float8_e4m3fn

                # Load scale_weight
                scale_weight_key = prefix + 'scale_weight'
                if scale_weight_key in state_dict:
                    self.scale_weight = torch.nn.Parameter(state_dict.pop(scale_weight_key), requires_grad=False)
                else:
                    self.scale_weight = torch.nn.Parameter(torch.ones((), dtype=torch.float32), requires_grad=False)

                # Load scale_input if enabled
                scale_input_key = prefix + 'scale_input'
                if _scale_input_enabled and scale_input_key in state_dict:
                    self.scale_input = torch.nn.Parameter(state_dict.pop(scale_input_key), requires_grad=False)
                else:
                    self.scale_input = None
                    state_dict.pop(scale_input_key, None)  # Remove if present but not needed

            # Now load the weight tensor
            weight_tensor = state_dict.get(weight_key)
            if weight_tensor is not None:
                if self.is_high_precision_layer:
                    # High precision layer: store as regular tensor
                    self.weight = torch.nn.Parameter(weight_tensor.to(dtype=target_dtype), requires_grad=False)
                else:
                    # FP8 layer: wrap in QuantizedTensor so .to() operations work properly
                    # This is critical for LoRA compatibility - QuantizedTensor handles dtype
                    # conversions via __torch_dispatch__ and will dequantize when needed
                    fp8_weight = weight_tensor.to(dtype=target_dtype)
                    layout_params = {
                        'scale': self.scale_weight.data,
                        'orig_dtype': torch.bfloat16  # Default compute dtype
                    }
                    quantized_weight = QuantizedTensor(fp8_weight, "TensorCoreFP8Layout", layout_params)
                    self.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
                state_dict.pop(weight_key)
            else:
                missing_keys.append(weight_key)

            # Handle bias
            bias_key = prefix + 'bias'
            bias_tensor = state_dict.get(bias_key)
            if bias_tensor is not None:
                # Bias should always be in compute dtype (not FP8)
                if self.is_high_precision_layer:
                    self.bias = torch.nn.Parameter(bias_tensor.to(dtype=target_dtype), requires_grad=False)
                else:
                    # For FP8 layers, keep bias in original dtype (usually bf16/fp16)
                    self.bias = torch.nn.Parameter(bias_tensor, requires_grad=False)
                state_dict.pop(bias_key)

        def forward_comfy_cast_weights(self, input):
            # PATH A: High Precision Layer (LoRA Compatible)
            if self.is_high_precision_layer:
                # Strictly adhere to comfy.ops.manual_cast logic
                weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
                out = torch.nn.functional.linear(input, weight, bias)
                uncast_bias_weight(self, weight, bias, offload_stream)
                return out

            # PATH B: FP8 Layer
            # Get the actual weight - might be QuantizedTensor or raw tensor
            weight = self.weight
            if isinstance(weight, torch.nn.Parameter):
                weight = weight.data
            
            # Check if weight is a QuantizedTensor (our preferred storage format)
            if isinstance(weight, QuantizedTensor):
                # FP8 Layer with TensorCore support
                if _fp8_mat_mult_supported:
                    try:
                        out = fp8_linear(self, input)
                        if out is not None:
                            return out
                    except Exception as e:
                        logging.debug(f"[Hybrid FP8 Loader] FP8 matmul failed: {e}, falling back to dequantization")
                
                # Fallback: dequantize and use standard linear
                weight_dequant = weight.dequantize().to(dtype=input.dtype)
                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=input.device, dtype=input.dtype)
                return torch.nn.functional.linear(input, weight_dequant, bias)
            
            # Raw FP8 tensor (legacy path)
            if weight.dtype == torch.float8_e4m3fn:
                # FP8 Layer with TensorCore support
                if _fp8_mat_mult_supported:
                    try:
                        out = fp8_linear(self, input)
                        if out is not None:
                            return out
                    except Exception as e:
                        logging.debug(f"[Hybrid FP8 Loader] FP8 matmul failed: {e}, falling back to dequantization")

                # Manual FP8 dequantization fallback
                if self.scale_weight is not None:
                    scale = self.scale_weight.to(device=weight.device, dtype=input.dtype)
                    weight_dequant = weight.to(dtype=input.dtype) * scale
                else:
                    weight_dequant = weight.to(dtype=input.dtype)
                
                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=input.device, dtype=input.dtype)
                return torch.nn.functional.linear(input, weight_dequant, bias)

            # Non-FP8 weight, use standard manual_cast path
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            out = torch.nn.functional.linear(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return out

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                # Check if we need special handling
                weight = self.weight
                if isinstance(weight, torch.nn.Parameter):
                    weight = weight.data
                
                # QuantizedTensor or FP8 needs our special forward path
                if isinstance(weight, QuantizedTensor) or weight.dtype == torch.float8_e4m3fn:
                    return self.forward_comfy_cast_weights(*args, **kwargs)
                return super().forward(*args, **kwargs)

        def convert_weight(self, weight, inplace=False, **kwargs):
            """
            Convert weight for LoRA patching. This is called during LoRA application.
            For FP8 layers, we must dequantize the weight before any math operations.
            
            Note: The 'weight' parameter passed here may be:
            - A QuantizedTensor (our FP8 storage format)
            - A raw FP8 tensor
            - A regular float tensor (for high precision layers)
            """
            # If this is a high precision layer, return weight as-is
            if self.is_high_precision_layer:
                return weight

            # Handle QuantizedTensor - dequantize it
            if isinstance(weight, QuantizedTensor):
                return weight.dequantize()
            
            # Handle raw FP8 tensor
            if weight.dtype == torch.float8_e4m3fn:
                if self.scale_weight is not None:
                    scale = self.scale_weight.to(device=weight.device, dtype=torch.float32)
                    return weight.to(dtype=torch.float32) * scale
                else:
                    return weight.to(dtype=torch.float32)
            
            # Non-FP8 weight with scale (shouldn't normally happen, but handle it)
            if self.scale_weight is not None:
                if inplace:
                    weight *= self.scale_weight.to(device=weight.device, dtype=weight.dtype)
                    return weight
                else:
                    return weight.to(dtype=torch.float32) * self.scale_weight.to(device=weight.device, dtype=torch.float32)
            return weight

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            """
            Set weight after LoRA patching. Re-quantizes the weight back to FP8 if needed.
            """
            # If this is a high precision layer, just set the weight directly
            if self.is_high_precision_layer or self.scale_weight is None:
                if return_weight:
                    return weight
                if inplace_update:
                    if isinstance(self.weight.data, QuantizedTensor):
                        # Can't inplace update a QuantizedTensor easily
                        self.weight = torch.nn.Parameter(weight, requires_grad=False)
                    else:
                        self.weight.data.copy_(weight)
                else:
                    self.weight = torch.nn.Parameter(weight, requires_grad=False)
                return

            # FP8 Logic: quantize the weight with stochastic rounding
            # First divide by scale, then convert to FP8
            scaled_weight = weight / self.scale_weight.to(device=weight.device, dtype=weight.dtype)
            fp8_weight = comfy.float.stochastic_rounding(scaled_weight, torch.float8_e4m3fn, seed=seed)
            
            # Wrap in QuantizedTensor for proper handling
            layout_params = {
                'scale': self.scale_weight.data,
                'orig_dtype': weight.dtype
            }
            quantized_weight = QuantizedTensor(fp8_weight, "TensorCoreFP8Layout", layout_params)
            
            if return_weight:
                return quantized_weight
            
            self.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)

    # Normalization layers should NEVER be FP8 - use standard manual_cast versions
    # These are inherited from manual_cast and work correctly
    class GroupNorm(manual_cast.GroupNorm):
        pass

    class LayerNorm(manual_cast.LayerNorm):
        pass

    class RMSNorm(manual_cast.RMSNorm):
        pass

    # Convolution layers - use standard manual_cast versions
    class Conv1d(manual_cast.Conv1d):
        pass

    class Conv2d(manual_cast.Conv2d):
        pass

    class Conv3d(manual_cast.Conv3d):
        pass

    class ConvTranspose1d(manual_cast.ConvTranspose1d):
        pass

    class ConvTranspose2d(manual_cast.ConvTranspose2d):
        pass

    # Embedding layer - use standard manual_cast version
    class Embedding(manual_cast.Embedding):
        pass

    @classmethod
    def conv_nd(cls, dims, *args, **kwargs):
        if dims == 2:
            return cls.Conv2d(*args, **kwargs)
        elif dims == 3:
            return cls.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")



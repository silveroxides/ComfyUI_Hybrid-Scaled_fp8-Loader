# file: ComfyUI/custom_nodes/HybridFP8Loader/hybrid_fp8_ops.py

import torch
from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor
import comfy.model_management
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
    dtype = self.weight.dtype
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
            input = torch.clamp(input, min=-448, max=448, out=input)
            layout_params_input = {'scale': scale_input, 'orig_dtype': input_dtype}
            quantized_input = QuantizedTensor(input.to(dtype).contiguous(), "TensorCoreFP8Layout", layout_params_input)
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
    """
    # Specify that this ops class uses FP8 dtype
    dtype = torch.float8_e4m3fn

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Strip stray scaled_fp8 tensors before state dict loading
        self.register_state_dict_pre_hook(_strip_scaled_fp8_hook)

    @classmethod
    def cast_to(cls, dtype):
        """Override to prevent manual casting - we handle dtype internally."""
        # Return self to prevent ComfyUI from trying to cast
        return cls

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

            # Access scale_input_enabled from global config
            if not _scale_input_enabled:
                self.scale_input = None

            if not hasattr(self, 'scale_input'):
                self.scale_input = torch.nn.parameter.Parameter(data=torch.ones((), device=self.weight.device, dtype=torch.float32), requires_grad=False)
            return None

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            # Drop global auxiliary tensor if present
            state_dict.pop('scaled_fp8', None)

            # Construct the key as it appears in the map (e.g. "model.diffusion_model...weight")
            weight_key = prefix + 'weight'

            # Determine TARGET_DTYPE based on key existence in the map
            if weight_key in _high_precision_keys:
                target_dtype = _high_precision_keys[weight_key]
                # Strip FP8 scale keys eagerly so ComfyUI does not attempt FP8 conversion
                state_dict.pop(prefix + 'scale_weight', None)
                state_dict.pop(prefix + 'scale_input', None)
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
                _log_hp(f"[Hybrid FP8 Loader] Marked high-precision layer {weight_key} dtype={target_dtype}")
            else:
                # No override: Load normally as FP8
                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        def forward_comfy_cast_weights(self, input):
            # PATH A: High Precision Layer (LoRA Compatible)
            if getattr(self, 'is_high_precision_layer', False):
                # Sanity: high-precision layers should not carry FP8 scales
                if self.scale_weight is not None or self.scale_input is not None:
                    _log_hp(f"[Hybrid FP8 Loader] Warning: high-precision layer retained scales (scale_weight={self.scale_weight is not None}, scale_input={self.scale_input is not None}) dtype={self.weight.dtype}")
                # Strictly adhere to comfy.ops.manual_cast logic
                weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
                out = torch.nn.functional.linear(input, weight, bias)
                uncast_bias_weight(self, weight, bias, offload_stream)
                _log_hp(f"[Hybrid FP8 Loader] Forward high-precision linear dtype={weight.dtype} shape={tuple(weight.shape)}")
                return out

            # PATH B: FP8 Layer (TensorCore if available, else manual scaled)
            if _fp8_mat_mult_supported:
                out = fp8_linear(self, input)
                if out is not None:
                    return out

            # Manual FP8 Cast Logic fallback
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
            if getattr(self, 'is_high_precision_layer', False):
                _log_hp(f"[Hybrid FP8 Loader] convert_weight high-precision dtype={weight.dtype} inplace={inplace} shape={tuple(weight.shape)}")
                return weight
            else:
                if inplace:
                    weight *= self.scale_weight.to(device=weight.device, dtype=weight.dtype)
                    return weight
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
                _log_hp(f"[Hybrid FP8 Loader] set_weight high-precision dtype={weight.dtype} inplace_update={inplace_update} return_weight={return_weight} shape={tuple(weight.shape)}")
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



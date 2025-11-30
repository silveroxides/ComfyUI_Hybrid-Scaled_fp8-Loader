# file: ComfyUI/custom_nodes/HybridFP8Loader/__init__.py

import folder_paths
import comfy.sd
from safetensors import safe_open
from . import hybrid_fp8_ops

# --- Exclusion Lists ---
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


def detect_fp8_optimizations_and_dtype(model_path, excluded_layers_substrings):
    """
    Peeks into the safetensors file.
    1. Determines if it is scale_input enabled.
    2. Finds specific layers matching 'excluded_layers_substrings' and records their native dtype.
    """
    scale_input = False
    excluded_layers_dtype = {}

    try:
        with safe_open(model_path, framework="pt", device="cpu") as f:
            all_keys = f.keys()

            for key in all_keys:
                for substring in excluded_layers_substrings:
                    if substring in key:
                        tensor = f.get_tensor(key)
                        excluded_layers_dtype[key] = tensor.dtype
                        # Break inner loop to avoid adding same key twice if multiple substrings match
                        break

            if "scaled_fp8" in all_keys:
                scaled_fp8_tensor = f.get_tensor("scaled_fp8")
                if scaled_fp8_tensor.shape[0] == 0:
                    print("[Hybrid FP8 Loader] Scale Input model detected (scale_input enabled).")
                    scale_input = True
                else:
                    print("[Hybrid FP8 Loader] Standard UNet-style model detected (scale_input disabled).")
    except Exception as e:
        print(f"[Hybrid FP8 Loader] Warning: Could not inspect model file: {e}")

    return scale_input, excluded_layers_dtype


def setup_hybrid_ops(model_path, model_type):
    """A helper function to configure the hybrid ops based on user settings and model type."""
    disable_fp8_mat_mult = False
    excluded_layers = []

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
        disable_fp8_mat_mult = True
    elif model_type == "hunyuan":
        excluded_layers.extend(HUNYUAN_LAYER_KEYNAMES)
    elif model_type == "zimage":
        excluded_layers.extend(ZIMAGE_LAYER_KEYNAMES)

    # Use set to remove duplicate substrings
    high_precision_substrings = list(set(excluded_layers))

    scale_input_enabled, excluded_layers_dtype = detect_fp8_optimizations_and_dtype(model_path, high_precision_substrings)

    hybrid_fp8_ops.set_high_precision_keys(excluded_layers_dtype)

    return hybrid_fp8_ops.get_hybrid_fp8_ops(scale_input_enabled=scale_input_enabled, disable_fp8_mat_mult=disable_fp8_mat_mult)

class ScaledFP8HybridUNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("unet"), ),
                "model_type": (["none", "chroma_hybrid_large", "radiance_hybrid_large", "chroma_hybrid_small", "radiance_hybrid_small", "wan", "pony_diffusion_v7", "qwen", "hunyuan", "zimage"], {"default": "none"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/FP8"

    def load_unet(self, model_name, model_type):
        unet_path = folder_paths.get_full_path("unet", model_name)
        ops = setup_hybrid_ops(unet_path, model_type)
        model = comfy.sd.load_diffusion_model(unet_path, model_options={"custom_operations": ops})
        return (model,)

class ScaledFP8HybridCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "model_type": (["none", "chroma_hybrid_large", "radiance_hybrid_large", "chroma_hybrid_small", "radiance_hybrid_small", "wan", "pony_diffusion_v7", "qwen", "hunyuan", "zimage"], {"default": "none"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders/FP8"

    def load_checkpoint(self, ckpt_name, model_type):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        ops = setup_hybrid_ops(ckpt_path, model_type)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options={"custom_operations": ops})
        return out[:3]

NODE_CLASS_MAPPINGS = {
    "ScaledFP8HybridUNetLoader": ScaledFP8HybridUNetLoader,
    "ScaledFP8HybridCheckpointLoader": ScaledFP8HybridCheckpointLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ScaledFP8HybridUNetLoader": "Load FP8 Scaled Diffusion Model (Choose One)",
    "ScaledFP8HybridCheckpointLoader": "Load FP8 Scaled Ckpt (Choose One)",
}
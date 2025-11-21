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

def detect_fp8_optimizations(model_path):
    """
    Peeks into the safetensors file to check the shape of 'scaled_fp8' tensor.
    Returns True if input scaling should be enabled, False otherwise.
    """
    try:
        with safe_open(model_path, framework="pt", device="cpu") as f:
            if "scaled_fp8" in f.keys():
                scaled_fp8_tensor = f.get_tensor("scaled_fp8")
                if scaled_fp8_tensor.shape[0] == 0:
                    print("[Hybrid FP8 Loader] Scale Input model detected (scale_input enabled).")
                    return True
    except Exception as e:
        print(f"[Hybrid FP8 Loader] Warning: Could not inspect model file to determine FP8 type: {e}")

    print("[Hybrid FP8 Loader] Standard UNet-style model detected (scale_input disabled).")
    return False

def setup_hybrid_ops(model_path, model_type):
    """A helper function to configure the hybrid ops based on user settings and model type."""
    disable_fp8_mat_mult = False
    excluded_layers = []
    if model_type == "chroma_hybrid_large":
        excluded_layers.extend(DISTILL_LAYER_KEYNAMES_LARGE)
    if model_type == "radiance_hybrid_large":
        excluded_layers.extend(NERF_LAYER_KEYNAMES_LARGE)
    if model_type == "chroma_hybrid_small":
        excluded_layers.extend(DISTILL_LAYER_KEYNAMES_SMALL)
    if model_type == "radiance_hybrid_small":
        excluded_layers.extend(NERF_LAYER_KEYNAMES_SMALL)
    if model_type == "wan":
        excluded_layers.extend(WAN_LAYER_KEYNAMES)
    if model_type == "pony_diffusion_v7":
        excluded_layers.extend(PONYV7_LAYER_KEYNAMES)
    if model_type == "qwen":
        excluded_layers.extend(QWEN_LAYER_KEYNAMES)
        disable_fp8_mat_mult = True

    hybrid_fp8_ops.set_high_precision_keynames(list(set(excluded_layers)))

    # --- THIS IS THE KEY LOGIC ---
    # Detect model type from the file and pass the correct flag to get_hybrid_fp8_ops
    scale_input_enabled = detect_fp8_optimizations(model_path)
    return hybrid_fp8_ops.get_hybrid_fp8_ops(scale_input_enabled=scale_input_enabled, disable_fp8_mat_mult=disable_fp8_mat_mult)

class ScaledFP8HybridUNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("unet"), ),
                "model_type": (["none", "chroma_hybrid_large", "radiance_hybrid_large", "chroma_hybrid_small", "radiance_hybrid_small", "wan", "pony_diffusion_v7", "qwen"], {"default": "none"}),
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
                "model_type": (["none", "chroma_hybrid_large", "radiance_hybrid_large", "chroma_hybrid_small", "radiance_hybrid_small", "wan", "pony_diffusion_v7", "qwen"], {"default": "none"}),
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

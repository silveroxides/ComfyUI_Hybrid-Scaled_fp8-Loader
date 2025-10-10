# file: ComfyUI/custom_nodes/HybridFP8Loader/__init__.py

import folder_paths
import comfy.sd
from safetensors import safe_open
from . import hybrid_fp8_ops

# --- Exclusion Lists ---
DISTILL_LAYER_KEYNAMES = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]
NERF_LAYER_KEYNAMES = ["img_in_patch", "nerf_blocks", "nerf_final_layer_conv", "nerf_image_embedder"]
DISTILL_LAYER_KEYNAMES_REV2 = ["distilled_guidance_layer"]
NERF_LAYER_KEYNAMES_REV2 = ["nerf_blocks", "nerf_image_embedder"]

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

def setup_hybrid_ops(model_path, keep_distillation, keep_nerf, keep_distillation_rev2, keep_nerf_rev2):
    """A helper function to configure the hybrid ops based on user settings and model type."""
    excluded_layers = []
    if keep_distillation: excluded_layers.extend(DISTILL_LAYER_KEYNAMES)
    if keep_nerf: excluded_layers.extend(NERF_LAYER_KEYNAMES)
    if keep_distillation_rev2: excluded_layers.extend(DISTILL_LAYER_KEYNAMES_REV2)
    if keep_nerf_rev2: excluded_layers.extend(NERF_LAYER_KEYNAMES_REV2)

    hybrid_fp8_ops.set_high_precision_keynames(list(set(excluded_layers)))

    # --- THIS IS THE KEY LOGIC ---
    # Detect model type from the file and pass the correct flag to get_hybrid_fp8_ops
    scale_input_enabled = detect_fp8_optimizations(model_path)
    return hybrid_fp8_ops.get_hybrid_fp8_ops(scale_input_enabled=scale_input_enabled)

class ScaledFP8HybridUNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("unet"), ),
                "keep_distillation": ("BOOLEAN", {"default": False, "tooltip": "Only first revision hybrid models which might have some LoRA issues"}),
                "keep_nerf": ("BOOLEAN", {"default": False, "tooltip": "Only first revision hybrid models which might have some LoRA issues"}),
                "keep_distillation_rev2": ("BOOLEAN", {"default": False, "tooltip": "Use only with second revision hybrid models. LoRA should no longer be an issue"}),
                "keep_nerf_rev2": ("BOOLEAN", {"default": False, "tooltip": "Use only with second revision hybrid models. LoRA should no longer be an issue"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/FP8"

    def load_unet(self, model_name, keep_distillation, keep_nerf, keep_distillation_rev2, keep_nerf_rev2):
        unet_path = folder_paths.get_full_path("unet", model_name)
        ops = setup_hybrid_ops(unet_path, keep_distillation, keep_nerf, keep_distillation_rev2, keep_nerf_rev2)
        model = comfy.sd.load_diffusion_model(unet_path, model_options={"custom_operations": ops})
        return (model,)

class ScaledFP8HybridCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "keep_distillation": ("BOOLEAN", {"default": False, "tooltip": "Only first revision hybrid models which might have some LoRA issues"}),
                "keep_nerf": ("BOOLEAN", {"default": False, "tooltip": "Only first revision hybrid models which might have some LoRA issues"}),
                "keep_distillation_rev2": ("BOOLEAN", {"default": False, "tooltip": "Use only with second revision hybrid models. LoRA should no longer be an issue"}),
                "keep_nerf_rev2": ("BOOLEAN", {"default": False, "tooltip": "Use only with second revision hybrid models. LoRA should no longer be an issue"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders/FP8"

    def load_checkpoint(self, ckpt_name, keep_distillation, keep_nerf, keep_distillation_rev2, keep_nerf_rev2):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        ops = setup_hybrid_ops(ckpt_path, keep_distillation, keep_nerf, keep_distillation_rev2, keep_nerf_rev2)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options={"custom_operations": ops})
        return out[:3]

NODE_CLASS_MAPPINGS = {
    "ScaledFP8HybridUNetLoader": ScaledFP8HybridUNetLoader,
    "ScaledFP8HybridCheckpointLoader": ScaledFP8HybridCheckpointLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ScaledFP8HybridUNetLoader": "Load Scaled FP8 Diffusion Model (Hybrid)",
    "ScaledFP8HybridCheckpointLoader": "Load Scaled FP8 Checkpoint (Hybrid)",
}
# file: ComfyUI/custom_nodes/HybridFP8Loader/__init__.py

import folder_paths
import comfy.sd
from safetensors import safe_open
from . import hybrid_fp8_ops

# --- Exclusion Lists ---
DISTILL_LAYER_KEYNAMES = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]
NERF_LAYER_KEYNAMES = ["nerf_image_embedder", "img_in_patch", "nerf_final_layer_conv"]
RADIANCE_LAYER_KEYNAMES = ["img_in_patch", "nerf_final_layer_conv"]
T5XXL_AVOID_KEY_NAMES = ["norm", "bias", "embed_tokens", "shared"]

def detect_fp8_optimizations(model_path):
    """
    Peeks into the safetensors file to check the shape of 'scaled_fp8' tensor.
    Returns True if T5XXL-style input scaling should be enabled, False otherwise.
    """
    try:
        with safe_open(model_path, framework="pt", device="cpu") as f:
            if "scaled_fp8" in f.keys():
                # T5XXL models have a zero-element tensor. Standard UNets have a 2-element one.
                scaled_fp8_tensor = f.get_tensor("scaled_fp8")
                if scaled_fp8_tensor.shape[0] == 0:
                    print("[Hybrid FP8 Loader] T5XXL-style model detected (scale_input enabled).")
                    return True
    except Exception as e:
        print(f"[Hybrid FP8 Loader] Warning: Could not inspect model file to determine FP8 type: {e}")

    print("[Hybrid FP8 Loader] Standard UNet-style model detected (scale_input disabled).")
    return False

def setup_hybrid_ops(model_path, keep_distillation, keep_nerf, radiance, t5xxl_exclusions):
    """A helper function to configure the hybrid ops based on user settings and model type."""
    excluded_layers = []
    if keep_distillation: excluded_layers.extend(DISTILL_LAYER_KEYNAMES)
    if keep_nerf: excluded_layers.extend(NERF_LAYER_KEYNAMES)
    if radiance: excluded_layers.extend(RADIANCE_LAYER_KEYNAMES)
    if t5xxl_exclusions: excluded_layers.extend(T5XXL_AVOID_KEY_NAMES)

    hybrid_fp8_ops.set_high_precision_keynames(list(set(excluded_layers)))

    # --- THIS IS THE KEY LOGIC ---
    # Detect model type from the file and pass the correct flag to get_hybrid_fp8_ops
    scale_input_enabled = detect_fp8_optimizations(model_path)
    return hybrid_fp8_ops.get_hybrid_fp8_ops(scale_input_enabled=scale_input_enabled)

class ScaledFP8HybridUNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "model_name": (folder_paths.get_filename_list("unet"), ), "keep_distillation": ("BOOLEAN", {"default": True}), "keep_nerf": ("BOOLEAN", {"default": True}), "radiance": ("BOOLEAN", {"default": False}), "t5xxl_exclusions": ("BOOLEAN", {"default": False}), } }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/FP8"

    def load_unet(self, model_name, keep_distillation, keep_nerf, radiance, t5xxl_exclusions):
        unet_path = folder_paths.get_full_path("unet", model_name)
        ops = setup_hybrid_ops(unet_path, keep_distillation, keep_nerf, radiance, t5xxl_exclusions)
        model = comfy.sd.load_diffusion_model(unet_path, model_options={"custom_operations": ops})
        return (model,)

class ScaledFP8HybridCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ), "keep_distillation": ("BOOLEAN", {"default": True}), "keep_nerf": ("BOOLEAN", {"default": True}), "radiance": ("BOOLEAN", {"default": False}), "t5xxl_exclusions": ("BOOLEAN", {"default": False}), } }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders/FP8"

    def load_checkpoint(self, ckpt_name, keep_distillation, keep_nerf, radiance, t5xxl_exclusions):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        ops = setup_hybrid_ops(ckpt_path, keep_distillation, keep_nerf, radiance, t5xxl_exclusions)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options={"custom_operations": ops})
        return out[:3]

NODE_CLASS_MAPPINGS = {
    "ScaledFP8HybridUNetLoader": ScaledFP8HybridUNetLoader,
    "ScaledFP8HybridCheckpointLoader": ScaledFP8HybridCheckpointLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ScaledFP8HybridUNetLoader": "Load Scaled FP8 UNet (Hybrid)",
    "ScaledFP8HybridCheckpointLoader": "Load Scaled FP8 Ckpt (Hybrid)",
}
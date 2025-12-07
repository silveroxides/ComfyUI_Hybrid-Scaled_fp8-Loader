# file: ComfyUI/custom_nodes/HybridFP8Loader/__init__.py

import folder_paths
import comfy.sd
from . import hybrid_fp8_ops

class ScaledFP8HybridUNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("unet"), ),
                "model_type": (["none", "chroma_hybrid_large", "radiance_hybrid_large", "chroma_hybrid_small", "radiance_hybrid_small", "wan", "pony_diffusion_v7", "qwen", "hunyuan", "zimage"], {"default": "none"}),
                "force_fp8_matmul": ("BOOLEAN", {"default": False}),
                "metadata_debug": ("BOOLEAN", {"default": False}),
                "guard_header_only": ("BOOLEAN", {"default": False}),
                "log_high_precision": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/FP8"

    def load_unet(self, model_name, model_type, force_fp8_matmul, metadata_debug, guard_header_only, log_high_precision):
        unet_path = folder_paths.get_full_path("unet", model_name)
        # Configure ops with metadata inspection only (no tensor loading)
        hybrid_fp8_ops.configure_hybrid_ops(
            model_path=unet_path,
            model_type=model_type,
            force_fp8_matmul=force_fp8_matmul,
            debug_metadata=metadata_debug,
            guard_no_tensor_read=guard_header_only,
            log_high_precision=log_high_precision,
        )
        # Load model with ops class - ComfyUI will instantiate it
        model = comfy.sd.load_diffusion_model(unet_path, model_options={"custom_operations": hybrid_fp8_ops.HybridOps})
        return (model,)

class ScaledFP8HybridCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "model_type": (["none", "chroma_hybrid_large", "radiance_hybrid_large", "chroma_hybrid_small", "radiance_hybrid_small", "wan", "pony_diffusion_v7", "qwen", "hunyuan", "zimage"], {"default": "none"}),
                "force_fp8_matmul": ("BOOLEAN", {"default": False}),
                "metadata_debug": ("BOOLEAN", {"default": False}),
                "guard_header_only": ("BOOLEAN", {"default": False}),
                "log_high_precision": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders/FP8"

    def load_checkpoint(self, ckpt_name, model_type, force_fp8_matmul, metadata_debug, guard_header_only, log_high_precision):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        # Configure ops with metadata inspection only (no tensor loading)
        hybrid_fp8_ops.configure_hybrid_ops(
            model_path=ckpt_path,
            model_type=model_type,
            force_fp8_matmul=force_fp8_matmul,
            debug_metadata=metadata_debug,
            guard_no_tensor_read=guard_header_only,
            log_high_precision=log_high_precision,
        )
        # Load checkpoint with ops class - ComfyUI will instantiate it
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options={"custom_operations": hybrid_fp8_ops.HybridOps})
        return out[:3]

NODE_CLASS_MAPPINGS = {
    "ScaledFP8HybridUNetLoader": ScaledFP8HybridUNetLoader,
    "ScaledFP8HybridCheckpointLoader": ScaledFP8HybridCheckpointLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ScaledFP8HybridUNetLoader": "Load FP8 Scaled Diffusion Model (Choose One)",
    "ScaledFP8HybridCheckpointLoader": "Load FP8 Scaled Ckpt (Choose One)",
}
# file: ComfyUI/custom_nodes/HybridFP8Loader/__init__.py

import folder_paths
import comfy.sd
from . import hybrid_fp8_ops


class HybridConfigNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "force_fp8_matmul": ("BOOLEAN", {"default": False, "tooltip": "Force FP8 matrix multiplications even if not detected in metadata"}),
                "metadata_debug": ("BOOLEAN", {"default": False, "tooltip": "Enable detailed logging of metadata during model inspection"}),
                "guard_header_only": ("BOOLEAN", {"default": False, "tooltip": "Only read metadata headers without loading any tensors"}),
                "log_high_precision": ("BOOLEAN", {"default": False, "tooltip": "Log when high-precision tensors are used instead of FP8"}),
                "mmap": ("BOOLEAN", {"default": False, "tooltip": "Enable mmap-backed state dict loading"}),
                "worker_override": ("BOOLEAN", {"default": False, "tooltip": "Override default worker count for state dict loading"}),
                "worker_count": ("INT", {"default": 2, "min": 1, "max": 16, "tooltip": "Number of worker threads for state dict loading"}),
            }
        }

    RETURN_TYPES = ("HYBRID_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "configure"
    CATEGORY = "loaders/FP8"

    def configure(self, force_fp8_matmul, metadata_debug, guard_header_only, log_high_precision, mmap, worker_override, worker_count):
        config = {
            "force_fp8_matmul": force_fp8_matmul,
            "metadata_debug": metadata_debug,
            "guard_header_only": guard_header_only,
            "log_high_precision": log_high_precision,
            "mmap": mmap,
            "worker_override": worker_override,
            "worker_count": worker_count,
        }
        return (config,)


class ScaledFP8HybridUNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("unet"), ),
                "model_type": (["none", "chroma_hybrid_large", "radiance_hybrid_large", "chroma_hybrid_small", "radiance_hybrid_small", "wan", "pony_diffusion_v7", "qwen", "hunyuan", "zimage"], {"default": "none", "tooltip": "Type of the model to load for proper FP8 handling"}),
            },
            "optional": {
                "hybrid_config": ("HYBRID_CONFIG",{"tooltip": "Hybrid FP8 configuration from Hybrid Config node"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/FP8"

    def load_unet(self, model_name, model_type, hybrid_config=None):
        unet_path = folder_paths.get_full_path("unet", model_name)
        # Configure ops with metadata inspection only (no tensor loading)
        if hybrid_config is None:
            force_fp8_matmul = False
            metadata_debug = False
            guard_header_only = False
            log_high_precision = False
            mmap = False
            worker_override = False
            worker_count = 2
        else:
            force_fp8_matmul = hybrid_config.get("force_fp8_matmul", False)
            metadata_debug = hybrid_config.get("metadata_debug", False)
            guard_header_only = hybrid_config.get("guard_header_only", False)
            log_high_precision = hybrid_config.get("log_high_precision", False)
            mmap = hybrid_config.get("mmap", False)
            worker_override = hybrid_config.get("worker_override", False)
            worker_count = hybrid_config.get("worker_count", 2)
        hybrid_fp8_ops.configure_hybrid_ops(
            model_path=unet_path,
            model_type=model_type,
            force_fp8_matmul=force_fp8_matmul,
            debug_metadata=metadata_debug,
            guard_no_tensor_read=guard_header_only,
            log_high_precision=log_high_precision,
        )
        # Toggle mmap-backed state dict loading based on mmap flag
        hybrid_fp8_ops.set_state_dict_mmap(mmap)
        hybrid_fp8_ops.set_state_dict_workers(worker_count, worker_override)
        # Lazy-load state dict via safetensors and use load_diffusion_model_state_dict
        sd, metadata = hybrid_fp8_ops.load_unet_lazy(unet_path)
        print(f"metadata keys: {metadata}")
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": hybrid_fp8_ops.HybridOps, "fp8_optimizations": True})
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
    "HybridConfigNode": HybridConfigNode,
    "ScaledFP8HybridUNetLoader": ScaledFP8HybridUNetLoader,
    "ScaledFP8HybridCheckpointLoader": ScaledFP8HybridCheckpointLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HybridConfigNode": "Hybrid FP8 Config",
    "ScaledFP8HybridUNetLoader": "Load FP8 Scaled Diffusion Model (Choose One)",
    "ScaledFP8HybridCheckpointLoader": "Load FP8 Scaled Ckpt (Choose One)",
}
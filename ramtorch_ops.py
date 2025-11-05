import torch
import comfy.ops
from ramtorch.modules.linear import BouncingLinearFn

def get_ramtorch_ops():
    """
    Returns the custom operations class that replaces torch.nn.Linear
    with a ramtorch-compatible implementation, following the correct
    ComfyUI ops pattern.
    """

    class RamTorchOps(comfy.ops.disable_weight_init):
        class Linear(comfy.ops.disable_weight_init.Linear):
            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                # We call the super init of the ComfyUI ops class we are inheriting from.
                # This sets up the module correctly within the ComfyUI framework.
                super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
                self.target_device = device if device is not None else torch.cuda.current_device()

            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                # This method completely takes over the loading process for this layer.
                # We do NOT call super()._load_from_state_dict() as there is none to call,
                # and we are handling everything manually. This fixes the AttributeError.
                weight_key = f"{prefix}weight"
                bias_key = f"{prefix}bias"

                weight_tensor = state_dict.pop(weight_key, None)
                bias_tensor = state_dict.pop(bias_key, None)

                if weight_tensor is None:
                    if strict:
                        missing_keys.append(weight_key)
                    return

                print(f"[RamTorch Ops] Intercepting layer: {prefix} --> Storing weights on CPU.")

                # Manually create the parameters, ensuring they are on the CPU and pinned.
                self.weight = torch.nn.Parameter(weight_tensor.to(device='cpu', non_blocking=True).pin_memory(), requires_grad=False)
                if bias_tensor is not None:
                    self.bias = torch.nn.Parameter(bias_tensor.to(device='cpu', non_blocking=True).pin_memory(), requires_grad=False)
                else:
                    self.register_parameter('bias', None)

                # Since we handled them, remove them from the list of missing keys.
                if weight_key in missing_keys:
                    missing_keys.remove(weight_key)
                if bias_key in missing_keys and self.bias is not None:
                    missing_keys.remove(bias_key)

            def forward(self, x):
                # The forward pass uses the ramtorch function with our CPU-based weights.
                comfy.ops.run_every_op()
                return BouncingLinearFn.apply(x, self.weight, self.bias, self.target_device)

            # We inherit the no-op reset_parameters from the parent class.
            # We must override _apply and cuda to manage our CPU-bound weights correctly.
            def _apply(self, fn):
                dummy = torch.tensor(0.0, device="cpu")
                result = fn(dummy)
                if result.dtype != dummy.dtype:
                    new_dtype = result.dtype
                    if self.weight is not None:
                        self.weight.data = self.weight.data.to(dtype=new_dtype)
                    if self.bias is not None:
                        self.bias.data = self.bias.data.to(dtype=new_dtype)
                del dummy
                return self

            def cuda(self, device=None):
                if device is not None:
                    self.target_device = device
                return self # Do not move weights to CUDA.

            def cpu(self):
                return self # Already on CPU.

    return RamTorchOps
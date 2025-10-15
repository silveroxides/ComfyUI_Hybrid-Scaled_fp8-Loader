# file: ComfyUI/custom_nodes/HybridFP8Loader/cpu_offload_ops.py

"""
CPU Linear Module

A memory-efficient linear layer implementation that keeps parameters on CPU
and transfers them to GPU on-demand using asynchronous CUDA streams.

This approach interleave compute and data transfer, making it useful for:
- Very large models that don't fit in GPU memory
- Scenarios where GPU memory is limited but CPU memory is abundant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.profiler import record_function
import comfy.model_management as mm

_DEVICE_STATE = {}

def _get_device_state(device):
    """Get or initialize per-device state."""
    if isinstance(device, str):
        device = torch.device(device)

    if device not in _DEVICE_STATE:
        with torch.cuda.device(device):
            _DEVICE_STATE[device] = {
                "transfer_stream": torch.cuda.Stream(device=device),
                "transfer_grad_stream": torch.cuda.Stream(device=device),
                "transfer_forward_finished_event": torch.cuda.Event(),
                "compute_forward_start_event": torch.cuda.Event(),
                "transfer_backward_finished_event": torch.cuda.Event(),
                "transfer_weight_backward_start_event": torch.cuda.Event(),
                "transfer_weight_backward_finished_event": torch.cuda.Event(),
                "compute_backward_start_event": torch.cuda.Event(),
                "compute_backward_finished_event": torch.cuda.Event(),
                "w_buffers": [None, None], "b_buffers": [None, None],
                "w_bwd_buffers": [None, None], "w_grad_buffers": [None, None],
                "b_grad_buffers": [None, None], "w_grad_accum_buffers": [None, None],
                "b_grad_accum_buffers": [None, None],
                "forward_clk": 0, "backward_clk": 0,
            }
    return _DEVICE_STATE[device]

def _invoke_tensor_hooks(tensor, grad):
    if hasattr(tensor, "_ramtorch_backward_hooks") and tensor._ramtorch_backward_hooks:
        for hook_id, hook_fn in tensor._ramtorch_backward_hooks.items():
            result = hook_fn(grad)
            if result is not None:
                grad = result
    return grad

def _invoke_post_accum_tensor_hooks(tensor):
    if (hasattr(tensor, "_ramtorch_post_accumulate_grad_hooks") and tensor._ramtorch_post_accumulate_grad_hooks):
        for hook_id, hook_fn in tensor._ramtorch_post_accumulate_grad_hooks.items():
            hook_fn(tensor)

class BouncingLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device="cuda"):
        state = _get_device_state(device)
        transfer_stream = state["transfer_stream"]
        w_buffers, b_buffers = state["w_buffers"], state["b_buffers"]
        transfer_forward_finished_event = state["transfer_forward_finished_event"]
        compute_forward_start_event = state["compute_forward_start_event"]

        selected_buffer = state["forward_clk"]
        state["forward_clk"] ^= 1

        with torch.cuda.stream(transfer_stream):
            transfer_stream.wait_event(compute_forward_start_event)
            with record_function("forward_weight_bias_transfer"):
                w_buffers[selected_buffer] = weight_cpu.to(device, non_blocking=True)
                b_buffers[selected_buffer] = (bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None)
            transfer_forward_finished_event.record()

        with record_function("forward_linear_compute"):
            torch.cuda.current_stream().wait_event(transfer_forward_finished_event)
            compute_forward_start_event.record()
            out = F.linear(x, w_buffers[selected_buffer], b_buffers[selected_buffer])

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # NOTE: The backward pass is for training. In ComfyUI (inference),
        # this will not be called, but we include it for completeness.
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        state = _get_device_state(device)
        transfer_stream = state["transfer_stream"]
        transfer_grad_stream = state["transfer_grad_stream"]
        w_bwd_buffers, w_grad_buffers, b_grad_buffers = state["w_bwd_buffers"], state["w_grad_buffers"], state["b_grad_buffers"]
        w_grad_accum_buffers, b_grad_accum_buffers = state["w_grad_accum_buffers"], state["b_grad_accum_buffers"]
        transfer_backward_finished_event = state["transfer_backward_finished_event"]
        transfer_weight_backward_finished_event = state["transfer_weight_backward_finished_event"]
        compute_backward_start_event = state["compute_backward_start_event"]
        compute_backward_finished_event = state["compute_backward_finished_event"]

        selected_buffer = state["backward_clk"]
        state["backward_clk"] ^= 1

        with torch.cuda.stream(transfer_stream):
            with record_function("backward_weight_transfer"):
                transfer_stream.wait_event(compute_backward_start_event)
                w_bwd_buffers[selected_buffer] = weight_cpu.to(device, non_blocking=True)
            with record_function("backward_grad_accumulator_transfer"):
                w_grad_accum_buffers[selected_buffer] = (weight_cpu.grad.to(device, non_blocking=True) if weight_cpu.grad is not None else None)
                b_grad_accum_buffers[selected_buffer] = (bias_cpu.grad.to(device, non_blocking=True) if bias_cpu.grad is not None else None)
                transfer_backward_finished_event.record()

        torch.cuda.current_stream().wait_event(transfer_backward_finished_event)
        compute_backward_start_event.record()

        with record_function("backward_linear_compute"):
            grad_input = grad_out @ w_bwd_buffers[selected_buffer]
            torch.cuda.current_stream().wait_event(transfer_weight_backward_finished_event)
            w_grad_buffers[selected_buffer] = grad_out.flatten(0, -2).T @ x.flatten(0, -2)

            with record_function("backward_weight_grad_accumulate"):
                w_grad_buffers[selected_buffer] = _invoke_tensor_hooks(weight_cpu, w_grad_buffers[selected_buffer])
                if w_grad_accum_buffers[selected_buffer] is not None:
                    w_grad_buffers[selected_buffer] += w_grad_accum_buffers[selected_buffer]
                weight_cpu.ramtorch_grad = w_grad_buffers[selected_buffer]
                _invoke_post_accum_tensor_hooks(weight_cpu)
                del weight_cpu.ramtorch_grad

            if bias_cpu is not None:
                reduce_dims = tuple(range(grad_out.ndim - 1))
                b_grad_buffers[selected_buffer] = grad_out.sum(dim=reduce_dims)
                with record_function("backward_bias_grad_accumulate"):
                    b_grad_buffers[selected_buffer] = _invoke_tensor_hooks(bias_cpu, b_grad_buffers[selected_buffer])
                    if b_grad_accum_buffers[selected_buffer] is not None:
                        b_grad_buffers[selected_buffer] += b_grad_accum_buffers[selected_buffer]
                    bias_cpu.ramtorch_grad = b_grad_buffers[selected_buffer]
                    _invoke_post_accum_tensor_hooks(bias_cpu)
                    del bias_cpu.ramtorch_grad
            compute_backward_finished_event.record()

        with record_function("backward_grad_transfer"):
            with torch.cuda.stream(transfer_grad_stream):
                transfer_grad_stream.wait_event(compute_backward_finished_event)
                weight_cpu.grad = w_grad_buffers[selected_buffer].to("cpu", non_blocking=True)
                if bias_cpu is not None:
                    bias_cpu.grad = b_grad_buffers[selected_buffer].to("cpu", non_blocking=True)
                transfer_weight_backward_finished_event.record()

        return grad_input, None, None, None

class CPUBouncingLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=None, device="cuda"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        dtype = dtype if dtype is not None else torch.float32

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device="cpu").share_memory_().pin_memory())
        self.bias = (nn.Parameter(torch.empty(out_features, dtype=dtype, device="cpu").share_memory_().pin_memory()) if bias else None)

        self.weight.is_ramtorch = True
        if self.bias is not None:
            self.bias.is_ramtorch = True

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return BouncingLinearFn.apply(x, self.weight, self.bias, self.device)
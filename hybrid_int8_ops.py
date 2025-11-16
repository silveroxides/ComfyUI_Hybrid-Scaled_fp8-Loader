# file: ComfyUI/custom_nodes/HybridLoader/hybrid_int8_ops.py

import torch
import comfy.ops
import comfy.model_management
import triton
import triton.language as tl
from triton import Config
from typing import Tuple

# --- Triton Kernels and Wrappers from int8_matmul.py ---
# This section contains the high-performance GPU code for INT8 operations.

@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    amax = tl.max(tl.abs(x))
    s = amax / 127.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)

def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    if x.size(-1) % block_size != 0:
        pad = block_size - (x.size(-1) % block_size)
        x = torch.nn.functional.pad(x, (0, pad))
    
    y = torch.empty_like(x, dtype=torch.int8)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s

int8_gemm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]

@triton.autotune(configs=int8_gemm_configs, key=["N", "K"])
@triton.jit
def int8_gemm_kernel(
    a_ptr, b_ptr, c_ptr, a_s_ptr, b_s_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None] # implicit transpose
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + offs_n * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - i * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K - i * BLOCK_SIZE_K), other=0.0)
        a_s = tl.load(a_s_ptrs + i, mask=offs_m < M)
        b_s = tl.load(b_s_ptrs + i, mask=offs_n < N)
        
        dot_prod = tl.dot(a, tl.trans(b))
        accumulator += dot_prod.to(tl.float32) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

def int8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(M, N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    int8_gemm_kernel[grid](a.view(M, K), b, c, a_s.flatten(), b_s.flatten(), M, N, K)
    return c

# --- Base Class for INT8 Matrix Multiplication ---
class Int8MM(comfy.ops.disable_weight_init):
    class Linear(comfy.ops.disable_weight_init.Linear):
        def forward(self, input, *args, **kwargs):
            comfy.ops.run_every_op()
            
            original_shape = input.shape
            input_reshaped = input.view(-1, self.in_features)

            # 1. Quantize activations on-the-fly
            # The GEMM kernel is configured with BLOCK_SIZE_K=128
            input_q, input_s = act_quant(input_reshaped, block_size=128)

            # 2. Use pre-quantized weights (self.weight) and scales (self.scale_weight)
            # The shapes are already correct from the loader:
            # self.weight: [out_features, in_features] (int8)
            # self.scale_weight: [out_features, in_features // block_size] (float32)
            weight_q = self.weight
            weight_s = self.scale_weight
            
            # 3. Perform high-performance scaled INT8 GEMM
            output = int8_gemm(input_q, input_s, weight_q, weight_s)

            # 4. Add bias if it exists
            if self.bias is not None:
                output += self.bias.to(output.device, output.dtype)

            # 5. Reshape to original batch dimensions
            return output.view(*original_shape[:-1], self.out_features)

# --- Hybrid Logic Implementation ---
_high_precision_keynames = []

def set_high_precision_keynames(keynames):
    global _high_precision_keynames
    _high_precision_keynames = keynames
    print(f"[Hybrid INT8 Ops] High precision keynames set: {keynames}")

def get_hybrid_int8_ops():
    print(f"[Hybrid INT8 Ops] Configuring INT8 operations.")
    base_ops_class = Int8MM

    class HybridScaledInt8Linear(base_ops_class.Linear):
        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            is_excluded = any(name in prefix for name in _high_precision_keynames)

            if is_excluded and _high_precision_keynames:
                print(f"[Hybrid INT8 Ops] Intercepting high-precision layer: {prefix}")
                weight_key = prefix + 'weight'
                bias_key = prefix + 'bias'
                weight_tensor = state_dict.pop(weight_key, None)
                bias_tensor = state_dict.pop(bias_key, None)

                if weight_tensor is None:
                    missing_keys.append(weight_key)
                else:
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)

                if bias_tensor is not None:
                    self.bias = torch.nn.Parameter(bias_tensor, requires_grad=False)
                else:
                    self.register_parameter("bias", None)

                state_dict.pop(prefix + 'scale_weight', None)
                self.register_parameter("scale_weight", None)
                setattr(self, 'is_high_precision_layer', True)
            else:
                # 1. Handle scale_weight
                scale_key = prefix + 'scale_weight'
                scale_tensor = state_dict.pop(scale_key, None)

                if scale_tensor is None:
                    error_msgs.append(f"Missing '{scale_key}' for quantized layer {prefix}")
                else:
                    # Assign the scale as a non-trainable Parameter. This ensures it's
                    # managed correctly by ComfyUI (e.g., moved to the correct device).
                    self.scale_weight = torch.nn.Parameter(scale_tensor, requires_grad=False)

                # 2. Handle the int8 weight
                weight_key = prefix + 'weight'
                weight_tensor = state_dict.pop(weight_key, None)
                if weight_tensor is None:
                    missing_keys.append(weight_key)
                elif weight_tensor.dtype != torch.int8:
                    error_msgs.append(f"Weight for quantized layer {prefix} is not int8! Found {weight_tensor.dtype}.")
                else:
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                
                # 3. Now that our custom keys are handled, let the parent class load the standard 'bias'.
                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        def forward(self, input):
            if getattr(self, 'is_high_precision_layer', False):
                comfy.ops.run_every_op()
                weight_hp = self.weight.to(input.device, input.dtype)
                bias_hp = self.bias.to(input.device, input.dtype) if self.bias is not None else None
                return torch.nn.functional.linear(input, weight_hp, bias_hp)
            else:
                return super().forward(input)

    class HybridInt8Ops(base_ops_class):
        class Linear(HybridScaledInt8Linear):
            pass

    return HybridInt8Ops
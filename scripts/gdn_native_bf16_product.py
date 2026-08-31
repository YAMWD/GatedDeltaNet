#!/usr/bin/env python3
"""Emulate the proposed native FPGA BF16 multiplier on an NVIDIA GPU.

The production all-BF16 model normally uses Tensor Cores: BF16 operands are
multiplied exactly and accumulated in FP32, then the completed linear result is
rounded to BF16.  A native ``ap_float<16, 8>`` FPGA multiply has a different
contract: every scalar product is rounded to BF16 *before* FP32 accumulation.

This module implements that contract without materialising an M x N x K tensor.
The Triton kernel also follows the accelerator's reduction structure:

* balanced 16-product reductions;
* four FP32 partial banks over consecutive 16-value input stripes; and
* final ``(p0 + p1) + (p2 + p3)`` reduction.

Only the dense HBM-backed projections are patched.  The tiny a/b projections,
normalisation, convolution and recurrent arithmetic retain their existing
all-BF16-boundary / FP32-compute behavior.  The LM head returns FP32 logits;
other dense projections round their completed outputs to BF16.
"""

from __future__ import annotations

import re
import types
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import triton
import triton.language as tl


_ATTN_DENSE = re.compile(
    r"(?:^|\.)layers\.\d+\.attn\.(?:q_proj|k_proj|v_proj|g_proj|o_proj)$"
)
_MLP_DENSE = re.compile(
    r"(?:^|\.)layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)$"
)
_LAST_COMPILED_KERNEL: Any | None = None


@triton.jit
def _balanced_tree16(
    products,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Force the exact 8/4/2/1 pairwise association of gemv32_tree16."""

    level1 = tl.sum(
        products.reshape((BLOCK_M, BLOCK_N, 8, 2)), axis=3
    )
    level2 = tl.sum(level1.reshape((BLOCK_M, BLOCK_N, 4, 2)), axis=3)
    level3 = tl.sum(level2.reshape((BLOCK_M, BLOCK_N, 2, 2)), axis=3)
    return tl.sum(level3.reshape((BLOCK_M, BLOCK_N, 1, 2)), axis=3).reshape(
        (BLOCK_M, BLOCK_N)
    )


@triton.jit
def _native_bf16_product_linear_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    m_size,
    n_size,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    K_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """BF16-product-rounded GEMM with the FPGA's four-bank FP32 reduction."""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < m_size
    mask_n = offs_n < n_size

    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    lane = tl.arange(0, 16)

    # All accelerator dense input dimensions are divisible by 64.  One loop
    # iteration therefore reproduces two 32-value BF16 weight beats and maps
    # their four balanced dot16 reductions to partial banks 0..3.
    for k_base in range(0, K_SIZE, 64):
        k0 = k_base + lane
        k1 = k_base + 16 + lane
        k2 = k_base + 32 + lane
        k3 = k_base + 48 + lane

        x0 = tl.load(
            x_ptr
            + offs_m[:, None, None] * stride_xm
            + k0[None, None, :] * stride_xk,
            mask=mask_m[:, None, None],
            other=0.0,
        ).to(tl.float32)
        x1 = tl.load(
            x_ptr
            + offs_m[:, None, None] * stride_xm
            + k1[None, None, :] * stride_xk,
            mask=mask_m[:, None, None],
            other=0.0,
        ).to(tl.float32)
        x2 = tl.load(
            x_ptr
            + offs_m[:, None, None] * stride_xm
            + k2[None, None, :] * stride_xk,
            mask=mask_m[:, None, None],
            other=0.0,
        ).to(tl.float32)
        x3 = tl.load(
            x_ptr
            + offs_m[:, None, None] * stride_xm
            + k3[None, None, :] * stride_xk,
            mask=mask_m[:, None, None],
            other=0.0,
        ).to(tl.float32)

        w0 = tl.load(
            w_ptr
            + offs_n[None, :, None] * stride_wn
            + k0[None, None, :] * stride_wk,
            mask=mask_n[None, :, None],
            other=0.0,
        ).to(tl.float32)
        w1 = tl.load(
            w_ptr
            + offs_n[None, :, None] * stride_wn
            + k1[None, None, :] * stride_wk,
            mask=mask_n[None, :, None],
            other=0.0,
        ).to(tl.float32)
        w2 = tl.load(
            w_ptr
            + offs_n[None, :, None] * stride_wn
            + k2[None, None, :] * stride_wk,
            mask=mask_n[None, :, None],
            other=0.0,
        ).to(tl.float32)
        w3 = tl.load(
            w_ptr
            + offs_n[None, :, None] * stride_wn
            + k3[None, None, :] * stride_wk,
            mask=mask_n[None, :, None],
            other=0.0,
        ).to(tl.float32)

        # The downcast is the experiment's only arithmetic change.  It forces
        # an RNE BF16 rounding point after every scalar multiply and before any
        # add.  Widening back to FP32 is exact wiring/conversion.
        p0 = (x0 * w0).to(tl.bfloat16).to(tl.float32)
        p1 = (x1 * w1).to(tl.bfloat16).to(tl.float32)
        p2 = (x2 * w2).to(tl.bfloat16).to(tl.float32)
        p3 = (x3 * w3).to(tl.bfloat16).to(tl.float32)

        acc0 += _balanced_tree16(p0, BLOCK_M, BLOCK_N)
        acc1 += _balanced_tree16(p1, BLOCK_M, BLOCK_N)
        acc2 += _balanced_tree16(p2, BLOCK_M, BLOCK_N)
        acc3 += _balanced_tree16(p3, BLOCK_M, BLOCK_N)

    result = (acc0 + acc1) + (acc2 + acc3)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        result += bias[None, :]

    tl.store(
        out_ptr
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on,
        result,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _launch_shape(m_size: int) -> tuple[int, int, int]:
    if m_size <= 2:
        return 1, 64, 4
    if m_size <= 64:
        return 4, 32, 8
    return 8, 32, 8


@torch.compiler.disable
def native_bf16_product_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    output_fp32: bool = False,
) -> torch.Tensor:
    """Linear with BF16-rounded products and FPGA-associated FP32 reduction."""

    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("native BF16-product evaluation requires CUDA tensors")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError(
            "native BF16-product evaluation requires BF16 input and weight; "
            f"got {x.dtype} and {weight.dtype}"
        )
    if x.shape[-1] != weight.shape[1]:
        raise ValueError(f"linear shape mismatch: {x.shape} and {weight.shape}")
    if weight.shape[1] % 64:
        raise ValueError(
            f"FPGA-associated reduction requires K divisible by 64, got {weight.shape[1]}"
        )
    if bias is not None and bias.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 bias, got {bias.dtype}")

    x_contiguous = x.contiguous()
    weight_contiguous = weight.contiguous()
    m_size = x_contiguous.numel() // x_contiguous.shape[-1]
    k_size = x_contiguous.shape[-1]
    n_size = weight_contiguous.shape[0]
    x_2d = x_contiguous.view(m_size, k_size)
    output_dtype = torch.float32 if output_fp32 else torch.bfloat16
    out_2d = torch.empty(
        (m_size, n_size), device=x.device, dtype=output_dtype
    )
    block_m, block_n, num_warps = _launch_shape(m_size)
    grid = (triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))
    bias_ptr = bias if bias is not None else weight_contiguous
    global _LAST_COMPILED_KERNEL
    _LAST_COMPILED_KERNEL = _native_bf16_product_linear_kernel[grid](
        x_2d,
        weight_contiguous,
        bias_ptr,
        out_2d,
        m_size,
        n_size,
        x_2d.stride(0),
        x_2d.stride(1),
        weight_contiguous.stride(0),
        weight_contiguous.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        K_SIZE=k_size,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=2,
    )
    return out_2d.view(*x.shape[:-1], n_size)


def last_compiled_ptx() -> str:
    if _LAST_COMPILED_KERNEL is None:
        return ""
    return str(getattr(_LAST_COMPILED_KERNEL, "asm", {}).get("ptx", ""))


def fpga_associated_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    output_fp32: bool = False,
) -> torch.Tensor:
    """Small CPU/GPU reference; intentionally materialises output tiles."""

    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("reference operands must both be BF16")
    if x.shape[-1] % 64:
        raise ValueError("reference K must be divisible by 64")
    original_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1]).float()
    weight_fp32 = weight.float()
    banks = [
        torch.zeros(
            (x_2d.shape[0], weight.shape[0]),
            device=x.device,
            dtype=torch.float32,
        )
        for _ in range(4)
    ]
    for k_base in range(0, x.shape[-1], 64):
        for bank in range(4):
            start = k_base + 16 * bank
            products = (
                x_2d[:, None, start : start + 16]
                * weight_fp32[None, :, start : start + 16]
            ).to(torch.bfloat16).float()
            # Explicit balanced tree matching gemv32_tree16.
            level = products
            while level.shape[-1] > 1:
                level = level[..., 0::2] + level[..., 1::2]
            banks[bank] += level[..., 0]
    result = (banks[0] + banks[1]) + (banks[2] + banks[3])
    if bias is not None:
        result += bias.float()
    if not output_fp32:
        result = result.to(torch.bfloat16)
    return result.view(*original_shape, weight.shape[0])


@dataclass(frozen=True)
class ProductRoundedPatch:
    module_names: tuple[str, ...]
    dense_weight_elements: int
    lm_head_fp32_logits: bool


def _is_accelerator_dense(name: str) -> bool:
    return bool(_ATTN_DENSE.search(name) or _MLP_DENSE.search(name) or name == "lm_head")


def install_native_bf16_product_linears(model: nn.Module) -> ProductRoundedPatch:
    """Patch exactly the accelerator's 193 HBM-backed dense projections."""

    patched: list[str] = []
    weight_elements = 0
    for name, module in model.named_modules():
        if not _is_accelerator_dense(name):
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"expected nn.Linear at {name}, got {type(module).__name__}")
        output_fp32 = name == "lm_head"

        def forward(
            self: nn.Linear,
            inputs: torch.Tensor,
            _output_fp32: bool = output_fp32,
        ) -> torch.Tensor:
            return native_bf16_product_linear(
                inputs, self.weight, self.bias, output_fp32=_output_fp32
            )

        module.forward = types.MethodType(forward, module)
        module._gdn_native_bf16_product = True  # type: ignore[attr-defined]
        patched.append(name)
        weight_elements += module.weight.numel()

    # The model has 24 layers, eight HBM-backed dense matrices per layer, and
    # one LM head.  Refuse a partial experiment if upstream naming changes.
    expected = 24 * 8 + 1
    if len(patched) != expected:
        raise RuntimeError(
            f"patched {len(patched)} dense modules, expected {expected}: {patched}"
        )

    # The fused SwiGLU helper receives down_proj.weight directly and bypasses
    # nn.Linear.forward.  Its unfused forward uses the same BF16 SwiGLU kernel,
    # then calls the patched down projection, so this is a routing change only.
    mlp_count = 0
    for name, module in model.named_modules():
        if re.search(r"(?:^|\.)layers\.\d+\.mlp$", name):
            if not hasattr(module, "fuse_swiglu"):
                raise TypeError(f"MLP at {name} has no fuse_swiglu selector")
            module.fuse_swiglu = False
            mlp_count += 1
    if mlp_count != 24:
        raise RuntimeError(f"updated {mlp_count} MLPs, expected 24")

    return ProductRoundedPatch(
        module_names=tuple(patched),
        dense_weight_elements=weight_elements,
        lm_head_fp32_logits=True,
    )


def patch_manifest(patch: ProductRoundedPatch) -> dict[str, Any]:
    return {
        "arithmetic_contract": "bf16_mul_rne_to_bf16_then_fp32_accumulate",
        "reduction_contract": "dot16_balanced_four_banks_final_01_plus_23",
        "patched_dense_module_count": len(patch.module_names),
        "patched_dense_weight_elements": patch.dense_weight_elements,
        "lm_head_logits_dtype": "float32",
        "patched_dense_modules": list(patch.module_names),
    }

#!/usr/bin/env python3
"""Numerical and generated-code gate for native BF16-product emulation."""

from __future__ import annotations

import argparse
import json
import random

import torch
import triton

from gdn_native_bf16_product import (
    fpga_associated_reference,
    last_compiled_ptx,
    native_bf16_product_linear,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-cases", type=int, default=20)
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.manual_seed(20260827)
    random.seed(20260827)
    device = torch.device("cuda")
    cases = [(1, 17, 64), (3, 65, 128), (9, 33, 192)]
    for _ in range(args.random_cases):
        cases.append(
            (
                random.randint(1, 11),
                random.randint(1, 97),
                64 * random.randint(1, 5),
            )
        )

    exact_mismatches = 0
    exact_elements = 0
    for index, (m_size, n_size, k_size) in enumerate(cases):
        x = (torch.randn(m_size, k_size, device=device) * 0.7).to(torch.bfloat16)
        weight = (torch.randn(n_size, k_size, device=device) * 0.4).to(
            torch.bfloat16
        )
        expected = fpga_associated_reference(x, weight, output_fp32=True)
        actual = native_bf16_product_linear(x, weight, output_fp32=True)
        mismatch = int((actual.view(torch.int32) != expected.view(torch.int32)).sum())
        exact_mismatches += mismatch
        exact_elements += actual.numel()
        if mismatch:
            difference = (actual - expected).abs()
            raise AssertionError(
                f"case {index} {(m_size, n_size, k_size)}: {mismatch} exact "
                f"mismatches, max_abs={difference.max().item()}"
            )

    # Show that the test exercises the new contract rather than accidentally
    # reproducing ordinary BF16 Tensor-Core GEMM.
    x = (torch.randn(8, 256, device=device) * 0.7).to(torch.bfloat16)
    weight = (torch.randn(64, 256, device=device) * 0.4).to(torch.bfloat16)
    rounded = native_bf16_product_linear(x, weight, output_fp32=True)
    tensor_core = x.float() @ weight.float().t()
    contract_delta_count = int(
        (rounded.view(torch.int32) != tensor_core.view(torch.int32)).sum()
    )
    if contract_delta_count == 0:
        raise AssertionError("product-rounded path is identical to exact-product GEMM")

    # Triton exposes the compiled PTX on the object returned by the launch.
    # Require BF16 conversions and forbid Tensor-Core MMA so a compiler change
    # cannot silently turn this into the already-evaluated Arm-A contract.
    combined_ptx = last_compiled_ptx()
    if not combined_ptx:
        raise AssertionError("could not retrieve generated PTX")
    bf16_conversions = combined_ptx.count("bf16")
    mma_instructions = combined_ptx.count("mma.sync")
    if bf16_conversions == 0:
        raise AssertionError("generated PTX has no BF16 conversion")
    if mma_instructions:
        raise AssertionError("generated PTX unexpectedly contains Tensor-Core MMA")

    benchmark = None
    if args.benchmark:
        bench_x = torch.randn(256, 2048, device=device).to(torch.bfloat16)
        bench_w = torch.randn(2048, 2048, device=device).to(torch.bfloat16)
        benchmark = {
            "m": 256,
            "n": 2048,
            "k": 2048,
            "milliseconds": triton.testing.do_bench(
                lambda: native_bf16_product_linear(bench_x, bench_w)
            ),
        }

    print(
        json.dumps(
            {
                "status": "PASS",
                "device": torch.cuda.get_device_name(device),
                "cases": len(cases),
                "exact_elements": exact_elements,
                "exact_mismatches": exact_mismatches,
                "ordinary_gemm_different_elements": contract_delta_count,
                "ptx_bf16_mentions": bf16_conversions,
                "ptx_mma_sync_count": mma_instructions,
                "benchmark": benchmark,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

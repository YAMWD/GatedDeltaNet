#!/usr/bin/env python3
"""Create one fast-prefill, boundary-rounded GDNSTAT1 handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from fla.models.gated_deltanet import GatedDeltaNetForCausalLM

from fpga_eval_client import FastStateHandoffProducer
from gdn_native_bf16_product import (
    install_native_bf16_product_linears,
    patch_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default="steady_decode_benchmark")
    args = parser.parse_args()

    tokens = [int(value) for value in args.tokens.read_text().split()]
    if not tokens:
        raise ValueError("handoff token file is empty")
    device = torch.device(args.device)
    model = GatedDeltaNetForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    patch = install_native_bf16_product_linears(model)
    manifest = patch_manifest(patch)
    producer = FastStateHandoffProducer(model, device)
    blank = producer.export(
        tokens,
        args.output,
        args.metadata,
        {
            "caller": "export_fpga_handoff_state",
            "task": args.task,
            "model": str(args.model.resolve()),
            "native_bf16_product": manifest,
        },
    )
    if blank:
        raise RuntimeError("benchmark handoff unexpectedly produced blank state")
    print(json.dumps({
        "state": str(args.output),
        "metadata": str(args.metadata),
        "context_tokens": len(tokens),
        "native_bf16_product": manifest,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


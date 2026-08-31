#!/usr/bin/env python3
"""Load the 1.3B checkpoint and gate the product-rounded dense patch."""

from __future__ import annotations

import argparse
import json

import torch

# Registers the architecture with Transformers AutoModel.
from fla.models.gated_deltanet import GatedDeltaNetConfig  # noqa: F401
from transformers import AutoModelForCausalLM

from gdn_native_bf16_product import (
    install_native_bf16_product_linears,
    patch_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sequence-length", type=int, default=8)
    return parser.parse_args()


def set_recurrent_mode(model: torch.nn.Module) -> None:
    model.config.attn_mode = "fused_recurrent"
    for layer in model.model.layers:
        layer.attn.mode = "fused_recurrent"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.manual_seed(20260827)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    set_recurrent_mode(model)
    patch = install_native_bf16_product_linears(model)

    vocab_size = int(model.config.vocab_size)
    input_ids = torch.arange(1, args.sequence_length + 1, device=device)[None, :]
    input_ids %= vocab_size
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
    logits = outputs.logits
    if logits.dtype != torch.float32:
        raise AssertionError(f"LM-head logits are {logits.dtype}, expected float32")
    if logits.shape != (1, 1, vocab_size):
        raise AssertionError(f"unexpected logits shape: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise AssertionError("non-finite logit from product-rounded model")

    result = patch_manifest(patch)
    result.update(
        {
            "status": "PASS",
            "sequence_length": args.sequence_length,
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
            "logits_checksum": float(logits.double().sum()),
            "logits_min": float(logits.min()),
            "logits_max": float(logits.max()),
            "argmax": int(logits[0, 0].argmax()),
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

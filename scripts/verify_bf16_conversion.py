#!/usr/bin/env python3
"""Verify a BF16 checkpoint conversion before spending GPU hours evaluating it.

Checks, in order of cost:
  1. tensor names, shapes, and element counts survive the cast
  2. every float tensor is BF16 and every non-float tensor is untouched
  3. the largest relative change stays inside the BF16 round-to-nearest bound,
     which also distinguishes round-to-nearest from truncation
  4. (optional, needs a GPU) the converted weights still produce the same
     next-token prediction when loaded back in FP32 - i.e. the "BF16 weights,
     FP32 math" configuration the FPGA would implement

Usage:
    python scripts/verify_bf16_conversion.py <fp32-snapshot> <bf16-checkpoint>
                                             [--expect-tensors N]
                                             [--expect-elements N]
                                             [--skip-logits]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

# BF16 keeps 8 bits of significand (7 stored + 1 implicit), so round-to-nearest
# cannot move a value by more than 2^-8. Truncation would permit up to 2^-7,
# which is how these two rounding modes are told apart below.
BF16_ROUND_TO_NEAREST_BOUND = 2.0**-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="original FP32 snapshot directory")
    p.add_argument("destination", type=Path, help="converted BF16 directory")
    p.add_argument("--expect-tensors", type=int, default=None)
    p.add_argument("--expect-elements", type=int, default=None)
    p.add_argument("--skip-logits", action="store_true",
                   help="skip the GPU forward-pass comparison")
    p.add_argument("--prompt", default="The capital of France is Paris, and "
                                       "the capital of Germany is")
    return p.parse_args()


def shard_names(root: Path) -> list[str]:
    index = root / "model.safetensors.index.json"
    if index.exists():
        return sorted(set(json.loads(index.read_text())["weight_map"].values()))
    return sorted(p.name for p in root.glob("*.safetensors"))


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    n_tensors = n_elements = 0
    worst_rel, worst_name = 0.0, ""
    dtypes: dict[str, int] = {}
    changed_non_float: list[str] = []

    for shard in shard_names(args.source):
        with safe_open(args.source / shard, framework="pt", device="cpu") as a, \
             safe_open(args.destination / shard, framework="pt", device="cpu") as b:
            keys_a, keys_b = sorted(a.keys()), sorted(b.keys())
            if keys_a != keys_b:
                check(f"{shard} tensor names match", False,
                      f"{len(keys_a)} vs {len(keys_b)}")
                continue
            for key in keys_a:
                ta, tb = a.get_tensor(key), b.get_tensor(key)
                n_tensors += 1
                n_elements += ta.numel()
                dtypes[str(tb.dtype)] = dtypes.get(str(tb.dtype), 0) + 1
                if ta.shape != tb.shape:
                    check(f"{key} shape", False, f"{ta.shape} vs {tb.shape}")
                    continue
                if not ta.is_floating_point():
                    if not torch.equal(ta, tb):
                        changed_non_float.append(key)
                    continue
                x, y = ta.float(), tb.float()
                rel = ((y - x).abs() / x.abs().clamp_min(1e-30)).max().item()
                if rel > worst_rel:
                    worst_rel, worst_name = rel, key

    print(f"\ntensors compared      : {n_tensors}")
    print(f"elements compared     : {n_elements:,}")
    print(f"output dtypes         : {dtypes}")
    print(f"worst relative change : {worst_rel:.4e} "
          f"(round-to-nearest bound {BF16_ROUND_TO_NEAREST_BOUND:.4e}) at {worst_name}\n")

    check("all float tensors are bfloat16", set(dtypes) <= {"torch.bfloat16"}, str(dtypes))
    check("non-float tensors unchanged", not changed_non_float,
          f"{len(changed_non_float)} changed")
    check("rounding is round-to-nearest, not truncation",
          worst_rel <= BF16_ROUND_TO_NEAREST_BOUND, f"{worst_rel:.4e}")
    if args.expect_tensors is not None:
        check(f"tensor count is {args.expect_tensors}",
              n_tensors == args.expect_tensors, f"got {n_tensors}")
    if args.expect_elements is not None:
        check(f"element count is {args.expect_elements:,}",
              n_elements == args.expect_elements, f"got {n_elements:,}")

    if not args.skip_logits:
        # Importing fla registers the gated_deltanet architecture with
        # Transformers; without it AutoModel cannot resolve the model type.
        import fla  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("\nloading both checkpoints in FP32 for a forward-pass comparison ...")
        tok = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True)
        ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()

        logits = {}
        for name, path in (("fp32", args.source), ("bf16-weights", args.destination)):
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float32, trust_remote_code=True
            ).cuda().eval()
            model.config.attn_mode = "fused_recurrent"
            with torch.inference_mode():
                logits[name] = model(ids).logits[0, -1].float().cpu()
            del model
            torch.cuda.empty_cache()

        x, y = logits["fp32"], logits["bf16-weights"]
        print(f"\nlogit max absolute difference : {(x - y).abs().max().item():.6f}")
        print(f"fp32 top token                : {tok.decode(x.argmax())!r}")
        print(f"bf16 top token                : {tok.decode(y.argmax())!r}")
        check("same predicted token", x.argmax().item() == y.argmax().item())
        check("same top-5 ordering",
              x.topk(5).indices.tolist() == y.topk(5).indices.tolist())

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

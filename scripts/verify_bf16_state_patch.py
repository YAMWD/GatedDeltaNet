#!/usr/bin/env python
"""Prove the BF16 recurrent-state patch actually takes effect.

Runs a few decode steps and inspects the recurrent state carried in the cache.
FLA allocates that accumulator as float32 whatever the model dtype, so the
tensors stay float32 either way -- what the patch changes is their *values*,
which become BF16-exact (low 16 mantissa bits zero).

Exit 0 only if the observed state matches the expectation for --expect.
A patch that silently fails to apply would otherwise produce a "no difference"
result that reads like a finding.
"""
import argparse, sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def bf16_exact_fraction(t: torch.Tensor) -> float:
    bits = t.detach().float().cpu().contiguous().view(torch.int32)
    return (bits & 0xFFFF == 0).float().mean().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--expect", choices=["bf16", "fp32"], required=True)
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(cfg, "attn_mode"):
        cfg.attn_mode = "fused_recurrent"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=cfg, torch_dtype=getattr(torch, args.dtype),
        trust_remote_code=True).cuda().eval()
    if hasattr(model.config, "attn_mode"):
        model.config.attn_mode = "fused_recurrent"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ids = tok("The capital of France is Paris, and the capital of Italy is",
              return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        for _ in range(args.steps):
            nxt = out.logits[:, -1:].argmax(-1)
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values

    # The recurrent state is the 4-D (batch, heads, K, V) entry; conv state is 3-D.
    states, seen = [], []
    stack = [past]
    while stack:
        o = stack.pop()
        if torch.is_tensor(o):
            seen.append(tuple(o.shape))
            if o.dim() == 4 and o.shape[-1] == o.shape[-2]:
                states.append(o)
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
        elif isinstance(o, dict):
            stack.extend(o.values())
        elif hasattr(o, "__dict__"):
            stack.extend(v for v in vars(o).values()
                         if torch.is_tensor(v) or isinstance(v, (list, tuple, dict)))

    if not states:
        print("FAIL: found no recurrent-state tensor in the cache", file=sys.stderr)
        print(f"  shapes seen: {seen[:12]}", file=sys.stderr)
        return 2

    fracs = [bf16_exact_fraction(s) for s in states]
    worst, mean = min(fracs), sum(fracs) / len(fracs)
    print(f"  recurrent-state tensors : {len(states)}  shape {tuple(states[0].shape)}"
          f"  dtype {states[0].dtype}")
    print(f"  BF16-exact fraction     : mean {mean:.6f}  worst {worst:.6f}")

    if args.expect == "bf16":
        ok = worst > 0.999
        print("  expected BF16-rounded state ->", "PASS" if ok else "FAIL")
        if not ok:
            print("  the patch did NOT take effect; do not trust any result from"
                  " this run", file=sys.stderr)
    else:
        ok = mean < 0.9
        print("  expected unrounded FP32 state ->", "PASS" if ok else "FAIL")
        if not ok:
            print("  state already looks BF16-exact without the patch; the"
                  " control is invalid", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

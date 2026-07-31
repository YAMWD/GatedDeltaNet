#!/usr/bin/env python3
"""TPOT (time-per-output-token) benchmark: GatedDeltaNet vs standard transformers.

Tests the decode-stage premise behind the FPGA work (see
c_impl/doc/decode_disaggregated_gemv.md): linear-attention GDN decode is O(1)/token with a
constant-size recurrent state, whereas a standard transformer is O(n)/token with
a KV cache that grows with context. So GDN's per-token decode latency (TPOT) and
memory should stay *flat* as context grows, while the transformer's climb.

This is a GPU (PyTorch) measurement of the architectural premise — NOT the
FPGA's TPOT. The trustworthy signal is the *slope vs context*, not the absolute
per-token number (at 1.3B on an A100 the model is small and decode is partly
launch/overhead-bound).

Run in the gdn-hf env:
  .micromamba/envs/gdn-hf/bin/python scripts/bench_tpot.py            # full sweep
  .micromamba/envs/gdn-hf/bin/python scripts/bench_tpot.py --quick    # smoke
  .micromamba/envs/gdn-hf/bin/python scripts/bench_tpot.py --models gdn-hybrid
"""
import argparse
import gc
import json
import os
import statistics as st
import sys
import time
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# fla provides GatedDeltaNet but does not auto-register it with AutoModel, so the
# m-a-p `model_type: gated_deltanet` checkpoints fail under a plain
# AutoModelForCausalLM.from_pretrained. Register the config/model explicitly.
_GDN_CLS = None
try:
    import fla  # noqa: F401
    from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    _GDN_CLS = GatedDeltaNetForCausalLM
    try:
        AutoConfig.register("gated_deltanet", GatedDeltaNetConfig)
        AutoModelForCausalLM.register(GatedDeltaNetConfig, GatedDeltaNetForCausalLM)
    except Exception:
        pass  # already registered
    _HAS_FLA = True
except Exception as e:  # pragma: no cover
    _HAS_FLA = False
    print(f"[warn] could not import/register fla ({e}); GDN models will not load", file=sys.stderr)


@dataclass
class ModelSpec:
    key: str
    hf_id: str
    kind: str          # "gdn" | "transformer"
    max_ctx: int       # cap the context sweep (OPT's learned pos-emb caps at 2048)
    trust_remote_code: bool = False


MODELS = {
    "gdn-pure":    ModelSpec("gdn-pure",    "m-a-p/1.3B-100B-GatedDeltaNet-pure",       "gdn",         65536, True),
    "gdn-hybrid":  ModelSpec("gdn-hybrid",  "m-a-p/1.3B-100B-GatedDeltaNet-hybrid-3-1", "gdn",         65536, True),
    "pythia-1.4b": ModelSpec("pythia-1.4b", "EleutherAI/pythia-1.4b",                   "transformer", 65536, False),
    # OPT uses learned positional embeddings (max_position_embeddings=2048); the
    # context + decode steps must stay under that or the position lookup asserts.
    "opt-1.3b":    ModelSpec("opt-1.3b",    "facebook/opt-1.3b",                        "transformer", 1024,  False),
}

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# A neutral English paragraph, repeated/truncated to the target context length.
# Content is irrelevant — we measure latency, not quality.
PROMPT_TEXT = (
    "The history of computing hardware spans many centuries and covers a wide "
    "range of devices, from simple mechanical aids to modern electronic "
    "processors. Early calculating tools such as the abacus gave way to "
    "mechanical calculators, then to programmable machines, and eventually to "
    "the integrated circuits that power today's computers. Each generation "
    "introduced new abstractions, new trade-offs between speed and memory, and "
    "new ways of organizing computation. Understanding this lineage helps "
    "engineers reason about the accelerators they design today. "
)


def build_input_ids(tok, length, device):
    """A single sequence of exactly `length` real tokens (repeat + truncate)."""
    ids = tok(PROMPT_TEXT, return_tensors="pt").input_ids[0]
    while ids.numel() < length:
        ids = torch.cat([ids, ids], dim=0)
    return ids[:length].unsqueeze(0).to(device)


@torch.inference_mode()
def _prefill(model, input_ids):
    out = model(input_ids=input_ids, use_cache=True)
    nxt = out.logits[:, -1:, :].argmax(-1)          # [1, 1]
    return out.past_key_values, nxt


@torch.inference_mode()
def _decode_step(model, next_ids, past):
    out = model(input_ids=next_ids, past_key_values=past, use_cache=True)
    return out.logits[:, -1, :].argmax(-1, keepdim=True), out.past_key_values


@torch.inference_mode()
def bench_one(model, tok, L, G, warmup, device):
    """Return per-(model, context-length) timing + peak-memory dict."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    input_ids = build_input_ids(tok, L, device)

    # ---- warmup: triton/FLA kernels autotune on first call per shape ----
    past, nxt = _prefill(model, input_ids)
    for _ in range(warmup):
        nxt, past = _decode_step(model, nxt, past)
    torch.cuda.synchronize(device)
    del past

    # ---- timed prefill (TTFT) ----
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    past, nxt = _prefill(model, input_ids)
    e1.record()
    torch.cuda.synchronize(device)
    ttft_ms = e0.elapsed_time(e1)

    # ---- timed decode (per-step) ----
    step_ms = []
    for _ in range(G):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        nxt, past = _decode_step(model, nxt, past)
        b.record()
        torch.cuda.synchronize(device)
        step_ms.append(a.elapsed_time(b))

    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
    step_ms.sort()
    del past
    return {
        "context": L,
        "decode_steps": G,
        "ttft_ms": round(ttft_ms, 4),
        "tpot_ms_median": round(st.median(step_ms), 5),
        "tpot_ms_mean": round(sum(step_ms) / len(step_ms), 5),
        "tpot_ms_p10": round(step_ms[int(0.1 * len(step_ms))], 5),
        "tpot_ms_p90": round(step_ms[min(int(0.9 * len(step_ms)), len(step_ms) - 1)], 5),
        "peak_mem_mb": round(peak_mb, 1),
        "tok_per_s": round(1000.0 / st.median(step_ms), 2),
    }


def load_model(spec, dtype, device):
    print(f"\n[load] {spec.key}  ({spec.hf_id}, dtype={dtype})", flush=True)
    t = time.time()
    tok = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code)
    # Load GDN via the explicit fla class (robust even if AutoModel registration
    # did not take); transformers go through AutoModelForCausalLM.
    cls = _GDN_CLS if (spec.kind == "gdn" and _GDN_CLS is not None) else AutoModelForCausalLM
    # Transformers: request PyTorch SDPA (the fair, optimized attention decode
    # path; avoids a flash-attn build). The fla GDN model class rejects the
    # attn_implementation kwarg, so GDN loads without it — the pure model has no
    # attention layers; the hybrid's fla attention requires flash-attn.
    base = dict(torch_dtype=DTYPES[dtype], trust_remote_code=spec.trust_remote_code)
    attn_options = ["sdpa", "eager"] if spec.kind == "transformer" else [None]
    model = None
    for attn_impl in attn_options:
        kw = dict(base)
        if attn_impl is not None:
            kw["attn_implementation"] = attn_impl
        try:
            model = cls.from_pretrained(spec.hf_id, **kw).to(device).eval()
            if attn_impl:
                print(f"[load] {spec.key}: attn_implementation={attn_impl}", flush=True)
            break
        except Exception as e:
            print(f"[load] {spec.key}: attn={attn_impl} failed ({e})", flush=True)
    if model is None:
        raise RuntimeError(f"could not load {spec.key}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] {spec.key}: {n_params/1e9:.3f}B params in {time.time()-t:.1f}s", flush=True)
    return model, tok, n_params


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def maybe_plot(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable ({e}); skipping plots", file=sys.stderr)
        return []
    saved = []
    for metric, ylabel, fname in [
        ("tpot_ms_median", "TPOT (ms/token, median)", "tpot_vs_context.png"),
        ("peak_mem_mb", "Peak GPU memory (MB)", "mem_vs_context.png"),
    ]:
        plt.figure(figsize=(7, 5))
        for key, rows in results.items():
            if not rows:
                continue
            xs = [r["context"] for r in rows]
            ys = [r[metric] for r in rows]
            style = "-o" if key.startswith("gdn") else "--s"
            plt.plot(xs, ys, style, label=key)
        plt.xlabel("context length (tokens)")
        plt.ylabel(ylabel)
        plt.title(ylabel + " vs context — GDN (linear) vs transformer (KV cache)")
        plt.xscale("log", base=2)
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close()
        saved.append(path)
        print(f"[plot] wrote {path}", flush=True)
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--context-lengths", nargs="+", type=int,
                    default=[128, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--out-dir", default="report/tpot")
    ap.add_argument("--quick", action="store_true",
                    help="smoke: contexts [128,512], 16 decode steps")
    args = ap.parse_args()

    if args.quick:
        args.context_lengths = [128, 512]
        args.decode_steps = 16

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda:0")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device: {torch.cuda.get_device_name(0)}  |  dtype: {args.dtype}  |  "
          f"contexts: {args.context_lengths}  |  decode_steps: {args.decode_steps}")

    results = {}
    meta = {}
    for key in args.models:
        spec = MODELS[key]
        try:
            model, tok, n_params = load_model(spec, args.dtype, device)
        except Exception as e:
            print(f"[error] failed to load {key}: {e}", file=sys.stderr)
            results[key] = []
            continue
        meta[key] = {"hf_id": spec.hf_id, "kind": spec.kind, "params_B": round(n_params / 1e9, 3)}
        rows = []
        for L in args.context_lengths:
            if L > spec.max_ctx:
                print(f"[skip] {key} @ ctx={L} (> max_ctx {spec.max_ctx})", flush=True)
                continue
            try:
                r = bench_one(model, tok, L, args.decode_steps, args.warmup, device)
                rows.append(r)
                print(f"[bench] {key:12s} ctx={L:6d}  TTFT={r['ttft_ms']:8.2f}ms  "
                      f"TPOT={r['tpot_ms_median']:7.3f}ms ({r['tok_per_s']:7.1f} tok/s)  "
                      f"peak={r['peak_mem_mb']:8.1f}MB", flush=True)
            except Exception as e:
                print(f"[error] {key} @ ctx={L}: {e}", file=sys.stderr)
        results[key] = rows
        free(model)

    payload = {
        "device": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "decode_steps": args.decode_steps,
        "warmup": args.warmup,
        "models": meta,
        "results": results,
    }
    json_path = os.path.join(args.out_dir, "tpot_results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[out] wrote {json_path}", flush=True)

    csv_path = os.path.join(args.out_dir, "tpot_results.csv")
    with open(csv_path, "w") as f:
        f.write("model,kind,context,ttft_ms,tpot_ms_median,tpot_ms_p10,tpot_ms_p90,tok_per_s,peak_mem_mb\n")
        for key, rows in results.items():
            kind = meta.get(key, {}).get("kind", "?")
            for r in rows:
                f.write(f"{key},{kind},{r['context']},{r['ttft_ms']},{r['tpot_ms_median']},"
                        f"{r['tpot_ms_p10']},{r['tpot_ms_p90']},{r['tok_per_s']},{r['peak_mem_mb']}\n")
    print(f"[out] wrote {csv_path}", flush=True)

    maybe_plot(results, args.out_dir)
    print("\n[done]")


if __name__ == "__main__":
    main()

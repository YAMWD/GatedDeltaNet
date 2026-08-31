#!/usr/bin/env python3
"""Score a rolling-loglikelihood .gdnreq (kind=3) on GPU, as the reference for
the FPGA's on-card WikiText perplexity.

Why this and not the published lm-eval number: the harness tokenizes and
windows the corpus through its own path, so its 16.827 is not guaranteed to be
window-for-window identical to the fixture the accelerator scores. Reading the
same fixture removes that doubt -- both sides score the same token windows and
sum log-probs the same way.

Why not a native (CPU) reference: gdn_eval runs at ~12.8 s/token, so the
328,878-token fixture would need ~1,170 hours. The GPU is the only feasible
same-scale reference.

Contract: BF16 activations, optional native-BF16 product rounding for the
HBM-backed dense matrices, FP32 LM-head reduction with unrounded FP32 logits --
matching the accelerator. Each window is scored from a blank recurrent state,
exactly as the FPGA's reset_decode_state() does.

Caveat recorded in the output: this scores each window with ONE batched forward,
so the per-token BF16 state/conv rounding the FPGA applies between tokens is not
reproduced. That is a sub-0.1%-level difference on perplexity, not a
window-alignment difference, and it is stated in the JSON rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import torch

REQ_MAGIC = b"GDNREQ1\0"
REQ_KIND_ROLLING = 3


def read_fixture(path: Path):
    blob = path.read_bytes()
    magic, version, kind, num_examples = struct.unpack_from("<8s3I", blob, 0)
    if magic != REQ_MAGIC:
        raise SystemExit(f"{path}: bad magic {magic!r}")
    if version != 1 or kind != REQ_KIND_ROLLING:
        raise SystemExit(f"{path}: expected version 1 kind {REQ_KIND_ROLLING}, "
                         f"got version {version} kind {kind}")
    offset = struct.calcsize("<8s3I")
    documents = []
    for _ in range(num_examples):
        word_count, byte_count, num_windows = struct.unpack_from(
            "<3I", blob, offset)
        offset += 12
        windows = []
        for _ in range(num_windows):
            ctx_len, cont_len = struct.unpack_from("<2I", blob, offset)
            offset += 8
            ctx = list(struct.unpack_from(f"<{ctx_len}i", blob, offset))
            offset += 4 * ctx_len
            cont = list(struct.unpack_from(f"<{cont_len}i", blob, offset))
            offset += 4 * cont_len
            windows.append((ctx, cont))
        documents.append({"word_count": word_count,
                          "byte_count": byte_count,
                          "windows": windows})
    return documents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--doc-limit", type=int, default=0)
    parser.add_argument("--native-bf16-product", action="store_true")
    args = parser.parse_args()

    # Load through fla's class and select fused_recurrent per layer -- the
    # pattern compare_gdn_c.load_model uses. AutoModelForCausalLM cannot
    # resolve `gated_deltanet` here even with fla imported.
    from fla.models.gated_deltanet import GatedDeltaNetForCausalLM

    documents = read_fixture(args.fixture)
    if args.doc_limit:
        documents = documents[:args.doc_limit]

    model = GatedDeltaNetForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()
    for layer in model.model.layers:
        if hasattr(layer.attn, "mode"):
            layer.attn.mode = "fused_recurrent"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if args.native_bf16_product:
        from gdn_native_bf16_product import (
            install_native_bf16_product_linears, patch_manifest)
        patch = install_native_bf16_product_linears(model)
        manifest = patch_manifest(patch)
        print(f"native BF16 product patch: {json.dumps(manifest)[:400]}",
              flush=True)

    # FP32 LM head from BF16-exact operands, matching the accelerator: bypass
    # the model's BF16 lm_head and reduce in FP32 so logits are unrounded.
    lm_head_weight = model.get_output_embeddings().weight.detach()

    total_logprob = 0.0
    total_words = total_bytes = scored_tokens = windows_done = 0
    start = time.time()
    with torch.inference_mode():
        for index, doc in enumerate(documents):
            for ctx, cont in doc["windows"]:
                tokens = torch.tensor([ctx + cont], dtype=torch.long,
                                      device="cuda")
                out = model.model(input_ids=tokens, use_cache=False,
                                  return_dict=True)
                hidden = out.last_hidden_state[0]        # [T, hidden], BF16
                # Positions predicting the continuation: the token at index
                # len(ctx)-1 predicts cont[0], and so on.
                first = len(ctx) - 1
                last = len(ctx) + len(cont) - 1          # exclusive
                h = hidden[first:last].float()
                logits = h @ lm_head_weight.float().T     # FP32 reduction
                logprobs = torch.log_softmax(logits, dim=-1)
                targets = torch.tensor(cont, dtype=torch.long, device="cuda")
                total_logprob += logprobs.gather(
                    1, targets.unsqueeze(1)).sum().item()
                scored_tokens += len(cont)
                windows_done += 1
            total_words += doc["word_count"]
            total_bytes += doc["byte_count"]
            print(f"doc {index + 1}/{len(documents)} "
                  f"scored_tokens={scored_tokens} "
                  f"running_word_ppl="
                  f"{math.exp(-total_logprob / max(total_words, 1)):.6f} "
                  f"elapsed={time.time() - start:.0f}s", flush=True)

    word_ppl = math.exp(-total_logprob / total_words)
    byte_ppl = math.exp(-total_logprob / total_bytes)
    bits_per_byte = -total_logprob / total_bytes / math.log(2.0)
    result = {
        "task": "wikitext_rolling",
        "reference": "gpu",
        "fixture": str(args.fixture),
        "model": args.model,
        "native_bf16_product": bool(args.native_bf16_product),
        "documents": len(documents),
        "windows": windows_done,
        "scored_tokens": scored_tokens,
        "words": total_words,
        "bytes": total_bytes,
        "total_logprob": total_logprob,
        "word_perplexity": word_ppl,
        "byte_perplexity": byte_ppl,
        "bits_per_byte": bits_per_byte,
        "caveat": ("one batched forward per window; the accelerator's "
                   "per-token BF16 state/conv rounding is not reproduced"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

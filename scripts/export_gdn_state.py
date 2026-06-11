#!/usr/bin/env python3
"""Export post-prefill GatedDeltaNet state for the FPGA decode-only accelerator.

Disaggregated decode: the GPU prefills an (arbitrary) prompt and dumps the
constant-size recurrent + conv state to disk; the FPGA loads that state and
decodes token-by-token, never prefilling. Because GDN is a linear-attention
(recurrent) model, the entire prompt history compresses into a FIXED-size object
regardless of prompt length:

    recurrent_state : num_layers x H x K x V  fp32   (24*8*256*256*4 = 48 MB)
    conv tails      : num_layers x {q,k,v} x (W-1) x hidden  fp32  (~1.7 MB)

State-layout contract (GPU FLA  ->  FPGA gdn_model.cpp):
  * recurrent_state  FLA final_state [N,H,K,V] (V contiguous)  ==  FPGA
    state[h][k][v] = recurrent_state[(h*K + k)*V + v]  -- IDENTICAL, no transpose.
  * conv_state       FLA cache [N,D,W] (newest at W-1)  ->  FPGA conv tail
    tail[r][d] = cache[0, d, r+1]  for r in 0..W-2  (drop oldest, transpose D<->W).
    Per layer the conv order is q(0), k(1), v(2), matching the FPGA head_buffer
    slot (layer*3 + conv).

The blob layout is two contiguous sections so the host maps each straight into a
BO: Section A = all layers' recurrent_state (-> recurrent_state BO); Section B =
all layers' conv tails (-> head_buffer BO).

Usage:
  python scripts/export_gdn_state.py \
      --fixture c_impl/fixtures_decode/decode.gdnreq \
      --example 0 \
      --output c_impl/fixtures_decode/decode_ex0.gdnstate
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch

# Reuse the package model loader + fixture reader from the reference scorer.
from compare_gdn_c import load_fixture, load_model, REQ_KIND_LL

STATE_MAGIC = b"GDNSTAT1"
STATE_VERSION = 1


@torch.no_grad()
def export_state(
    model,
    device: torch.device,
    prompt_ids: list[int],
    output_path: Path,
    golden_cont: list[int] | None,
    verify_steps: int,
) -> None:
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size

    # ---- GPU prefill of the prompt, capturing the recurrent cache ----
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    last_logits = out.logits[0, -1].float()
    seed_token = int(torch.argmax(last_logits).item())

    # ---- Pull per-layer state, derive dims from layer 0 ----
    st0 = past[0]
    rs0 = st0["recurrent_state"]            # [N, H, K, V]
    cq0, _ck0, _cv0 = st0["conv_state"]     # each [N, D, W]
    _, H, K, V = rs0.shape
    _, D, W = cq0.shape
    assert D == hidden, f"conv dim {D} != hidden {hidden}"

    rec_section = bytearray()   # Section A: [layer][H*K*V]
    conv_section = bytearray()  # Section B: [layer][{q,k,v}][(W-1)*D]

    for layer_idx in range(num_layers):
        st = past[layer_idx]
        rs = st["recurrent_state"][0].contiguous().float().cpu()      # [H, K, V]
        rec_section += rs.flatten().numpy().tobytes()
        for conv in st["conv_state"]:                                 # q, k, v
            # cache [N, D, W] -> tail [W-1, D] = cache[0, :, 1:].T (drop oldest)
            tail = conv[0, :, 1:].transpose(0, 1).contiguous().float().cpu()  # [W-1, D]
            assert tail.shape == (W - 1, D)
            conv_section += tail.flatten().numpy().tobytes()

    # ---- Write the .gdnstate blob ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        fh.write(STATE_MAGIC)
        for val in (STATE_VERSION, num_layers, H, K, V, hidden, W,
                    len(prompt_ids), seed_token):
            fh.write(struct.pack("<I", val if val >= 0 else val & 0xFFFFFFFF))
        fh.write(struct.pack(f"<{len(prompt_ids)}i", *prompt_ids))
        fh.write(rec_section)
        fh.write(conv_section)

    print(f"wrote {output_path}")
    print(f"  layers={num_layers} H={H} K={K} V={V} hidden={hidden} W={W}")
    print(f"  prompt_len={len(prompt_ids)}  seed_token(argmax last pos)={seed_token}")
    print(f"  recurrent section = {len(rec_section)/1e6:.1f} MB   "
          f"conv section = {len(conv_section)/1e3:.1f} KB")
    if golden_cont:
        print(f"  golden_cont[0]={golden_cont[0]}  -> seed matches: "
              f"{seed_token == golden_cont[0]}")

    # ---- Optional: cache-based decode sanity check vs the golden trajectory ----
    if verify_steps > 0:
        traj = [seed_token]
        tok = torch.tensor([[seed_token]], dtype=torch.long, device=device)
        cache = past
        for _ in range(verify_steps - 1):
            step = model(input_ids=tok, past_key_values=cache, use_cache=True)
            cache = step.past_key_values
            nxt = int(torch.argmax(step.logits[0, -1].float()).item())
            traj.append(nxt)
            tok = torch.tensor([[nxt]], dtype=torch.long, device=device)
        print(f"  cache-decode traj[:{verify_steps}] = {traj}")
        if golden_cont:
            gold = golden_cont[:verify_steps]
            match = traj == gold
            print(f"  golden       traj[:{verify_steps}] = {gold}")
            print(f"  EXACT MATCH vs golden: {match}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="m-a-p/1.3B-100B-GatedDeltaNet-pure")
    ap.add_argument("--fixture", type=Path,
                    default=Path("c_impl/fixtures_decode/decode.gdnreq"))
    ap.add_argument("--example", type=int, default=0,
                    help="which LL example's prompt to prefill")
    ap.add_argument("--output", type=Path,
                    default=Path("c_impl/fixtures_decode/decode_ex0.gdnstate"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--verify-steps", type=int, default=8,
                    help="cache-decode this many tokens and diff vs golden (0=off)")
    args = ap.parse_args()

    kind, examples = load_fixture(args.fixture)
    if kind != REQ_KIND_LL:
        raise SystemExit(f"--fixture must be LL-kind (kind=2), got {kind}")
    ex = examples[args.example]
    prompt_ids = ex["context"]
    golden_cont = ex.get("continuation") or None

    model, device = load_model(args.model_name, args.device, args.dtype)
    export_state(model, device, prompt_ids, args.output, golden_cont,
                 args.verify_steps)


if __name__ == "__main__":
    main()

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
import json
import struct
from pathlib import Path

import torch

# Reuse the package model loader + fixture reader from the reference scorer.
from compare_gdn_c import load_fixture, load_model, REQ_KIND_LL

STATE_MAGIC = b"GDNSTAT1"
STATE_VERSION = 1
LOGITS_MAGIC = b"GDNLOG1\0"
LOGITS_VERSION = 1
HIDDEN_MAGIC = b"GDNHID1\0"
HIDDEN_VERSION = 1


def round_recurrent_cache_bf16(past) -> None:
    """Apply the accelerator's persistent-state boundary in place.

    The current token's logits have already consumed the unrounded FP32 update
    when this helper is called. Short-convolution storage is controlled
    separately so Arm-B-plus-state and all-BF16 references remain distinct.
    """
    for layer_idx, state in enumerate(past):
        recurrent = state.get("recurrent_state")
        if recurrent is None:
            raise RuntimeError(
                f"layer {layer_idx} has no recurrent_state in its decode cache"
            )
        state["recurrent_state"] = recurrent.to(torch.bfloat16).float()


def round_conv_cache_bf16(past) -> None:
    """Apply the BF16 persistent short-convolution boundary in place."""
    for layer_idx, state in enumerate(past):
        conv_state = state.get("conv_state")
        if conv_state is None:
            raise RuntimeError(
                f"layer {layer_idx} has no conv_state in its decode cache"
            )
        state["conv_state"] = tuple(
            value.to(torch.bfloat16) for value in conv_state
        )


def assert_bf16_exact_recurrent_cache(past) -> None:
    """Reject a handoff whose recurrent values are not BF16-exact FP32."""
    for layer_idx, state in enumerate(past):
        recurrent = state["recurrent_state"].contiguous().float()
        low_bits = torch.bitwise_and(recurrent.view(torch.int32), 0xFFFF)
        if bool(torch.any(low_bits != 0).item()):
            raise RuntimeError(
                f"layer {layer_idx} recurrent state is not BF16-exact"
            )


def assert_bf16_exact_conv_cache(past) -> None:
    """Reject a handoff whose q/k/v convolution tails are not BF16-exact."""
    for layer_idx, state in enumerate(past):
        for kind, conv in enumerate(state["conv_state"]):
            values = conv.contiguous().float()
            low_bits = torch.bitwise_and(values.view(torch.int32), 0xFFFF)
            if bool(torch.any(low_bits != 0).item()):
                raise RuntimeError(
                    f"layer {layer_idx} conv cache {kind} is not BF16-exact"
                )


@torch.no_grad()
def run_model_step(
    model,
    input_ids: torch.Tensor,
    past,
    lm_head_weight: torch.Tensor | None,
    native_bf16_product: bool,
):
    """Run one token and return cache plus logits in the selected contract.

    A BF16 Hugging Face model normally returns BF16-rounded logits.  The
    accelerator instead widens its BF16 final activation and BF16 LM-head
    weights, reduces the GEMV in FP32, and exposes the unrounded FP32 vector.
    Bypass the model's BF16 ``lm_head`` when an FP32 weight view is supplied so
    the GPU handoff and full-logit reference implement that exact boundary.
    """
    if lm_head_weight is None:
        out = model(
            input_ids=input_ids,
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
        )
        return out.past_key_values, out.logits[0, -1].float(), None

    out = model.model(
        input_ids=input_ids,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
    )
    hidden = out.last_hidden_state[0, -1]
    if hidden.dtype != torch.bfloat16:
        raise RuntimeError(
            f"FP32-logit contract requires BF16 final activation, got {hidden.dtype}"
        )
    if native_bf16_product:
        from gdn_native_bf16_product import native_bf16_product_linear

        if lm_head_weight.dtype != torch.bfloat16:
            raise RuntimeError(
                "native BF16-product LM head requires BF16 weights"
            )
        logits = native_bf16_product_linear(
            hidden.unsqueeze(0), lm_head_weight, output_fp32=True
        )[0]
    else:
        # Both operands are BF16-exact values widened to FP32. CUDA TF32 is
        # disabled by the caller, so this GEMV returns an unrounded FP32 vector.
        logits = torch.mv(lm_head_weight, hidden.float())
    return out.past_key_values, logits, hidden.float()


@torch.no_grad()
def run_prefill(
    model,
    device: torch.device,
    prompt_ids: list[int],
    recurrent_storage: str,
    conv_storage: str,
    lm_head_weight: torch.Tensor | None,
    native_bf16_product: bool,
):
    """Run prefill with an explicit per-token persistence boundary."""
    if not prompt_ids:
        raise ValueError("cannot export state for an empty prompt")

    if (recurrent_storage != "bf16" and conv_storage != "bf16"
            and not native_bf16_product):
        # No per-token rounding boundary is requested, so prefill the whole
        # prompt in one batched call, exactly like the original FP32 exporter.
        # (native_bf16_product keeps the per-token loop so prefill goes through
        # the same fused_recurrent kernel path as its recorded references.)
        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=device)
        past, last_logits, _hidden = run_model_step(
            model, input_ids, None, lm_head_weight, native_bf16_product
        )
        return past, last_logits

    past = None
    last_logits = None
    for token in prompt_ids:
        input_ids = torch.tensor([[token]], dtype=torch.long, device=device)
        past, last_logits, _hidden = run_model_step(
            model, input_ids, past, lm_head_weight, native_bf16_product
        )
        if recurrent_storage == "bf16":
            round_recurrent_cache_bf16(past)
        if conv_storage == "bf16":
            round_conv_cache_bf16(past)

    assert past is not None and last_logits is not None
    if recurrent_storage == "bf16":
        assert_bf16_exact_recurrent_cache(past)
    if conv_storage == "bf16":
        assert_bf16_exact_conv_cache(past)
    return past, last_logits


@torch.no_grad()
def decode_reference(
    model,
    device: torch.device,
    past,
    seed_token: int,
    decode_steps: int,
    recurrent_storage: str,
    conv_storage: str,
    lm_head_weight: torch.Tensor | None,
    native_bf16_product: bool,
) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
    """Decode from the handoff and capture each post-seed full-logit vector."""
    trajectory = [seed_token]
    logits: list[torch.Tensor] = []
    hidden_states: list[torch.Tensor] = []
    token = seed_token
    for _ in range(max(decode_steps - 1, 0)):
        input_ids = torch.tensor([[token]], dtype=torch.long, device=device)
        past, step_logits, step_hidden = run_model_step(
            model, input_ids, past, lm_head_weight, native_bf16_product
        )
        token = int(torch.argmax(step_logits).item())
        trajectory.append(token)
        logits.append(step_logits.cpu())
        if step_hidden is not None:
            hidden_states.append(step_hidden.cpu())
        if recurrent_storage == "bf16":
            round_recurrent_cache_bf16(past)
        if conv_storage == "bf16":
            round_conv_cache_bf16(past)

    return trajectory, logits, hidden_states


def write_logits_reference(path: Path, vocab_size: int,
                           logits: list[torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack(
            "<8sIII", LOGITS_MAGIC, LOGITS_VERSION, vocab_size, len(logits)
        ))
        for step_logits in logits:
            values = step_logits.contiguous().float().numpy()
            if values.size != vocab_size:
                raise RuntimeError(
                    f"logit vector has {values.size} values, expected {vocab_size}"
                )
            handle.write(values.tobytes())


def write_hidden_reference(path: Path, hidden_size: int,
                           hidden_states: list[torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack(
            "<8sIII", HIDDEN_MAGIC, HIDDEN_VERSION,
            hidden_size, len(hidden_states)
        ))
        for step_hidden in hidden_states:
            values = step_hidden.contiguous().float().numpy()
            if values.size != hidden_size:
                raise RuntimeError(
                    f"hidden vector has {values.size} values, expected {hidden_size}"
                )
            handle.write(values.tobytes())


def write_decode_golden(path: Path, prompt_ids: list[int],
                        trajectory: list[int]) -> None:
    payload = {
        "kind": REQ_KIND_LL,
        "decode_len": len(trajectory),
        "num_examples": 1,
        "examples": [{
            "index": 0,
            "prompt_ids": prompt_ids,
            "golden_traj": trajectory,
            "per_step_argmax": trajectory,
            "per_step_logprob": [],
        }],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


@torch.no_grad()
def export_state(
    model,
    device: torch.device,
    prompt_ids: list[int],
    output_path: Path,
    golden_cont: list[int] | None,
    verify_steps: int,
    recurrent_storage: str,
    conv_storage: str,
    decode_steps: int,
    golden_output: Path | None,
    logits_output: Path | None,
    hidden_output: Path | None,
    fp32_logits: bool,
    native_bf16_product: bool,
) -> None:
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size

    lm_head_weight = None
    if fp32_logits:
        if model.lm_head.weight.dtype != torch.bfloat16:
            raise RuntimeError(
                "--fp32-logits requires BF16 LM-head weights before widening"
            )
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        lm_head_weight = (
            model.lm_head.weight.detach()
            if native_bf16_product
            else model.lm_head.weight.detach().float()
        )

    # ---- GPU prefill of the prompt, capturing the recurrent cache ----
    past, last_logits = run_prefill(
        model, device, prompt_ids, recurrent_storage, conv_storage,
        lm_head_weight, native_bf16_product,
    )
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
    print(f"  recurrent persistence = {recurrent_storage}")
    print(f"  convolution persistence = {conv_storage}")
    print(
        "  logits = "
        + (
            "BF16-rounded products / FP32 GEMV output"
            if native_bf16_product
            else "BF16 operands / FP32 GEMV output"
            if fp32_logits
            else "model-native"
        )
    )
    print(f"  recurrent section = {len(rec_section)/1e6:.1f} MB   "
          f"conv section = {len(conv_section)/1e3:.1f} KB")
    if golden_cont:
        print(f"  golden_cont[0]={golden_cont[0]}  -> seed matches: "
              f"{seed_token == golden_cont[0]}")

    # ---- Optional: cache-based decode sanity check vs the golden trajectory ----
    reference_steps = max(verify_steps, decode_steps)
    if reference_steps > 0:
        traj, logits, hidden_states = decode_reference(
            model, device, past, seed_token, reference_steps,
            recurrent_storage, conv_storage, lm_head_weight,
            native_bf16_product,
        )
        print(f"  cache-decode traj[:{verify_steps}] = {traj[:verify_steps]}")
        if golden_cont:
            gold = golden_cont[:verify_steps]
            match = traj[:verify_steps] == gold
            print(f"  golden       traj[:{verify_steps}] = {gold}")
            print(f"  EXACT MATCH vs golden: {match}")
        if golden_output is not None:
            write_decode_golden(golden_output, prompt_ids, traj[:decode_steps])
            print(f"wrote {golden_output}")
        if logits_output is not None:
            write_logits_reference(
                logits_output, cfg.vocab_size, logits[:max(decode_steps - 1, 0)]
            )
            print(f"wrote {logits_output}")
        if hidden_output is not None:
            if len(hidden_states) < max(decode_steps - 1, 0):
                raise RuntimeError(
                    "--hidden-output requires --fp32-logits and enough decode steps"
                )
            write_hidden_reference(
                hidden_output, cfg.hidden_size,
                hidden_states[:max(decode_steps - 1, 0)],
            )
            print(f"wrote {hidden_output}")


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
    ap.add_argument(
        "--recurrent-storage", choices=("fp32", "bf16"), default="fp32",
        help="persistent recurrent-state precision applied after every token",
    )
    ap.add_argument(
        "--conv-storage", choices=("native", "bf16"), default="native",
        help="persistent q/k/v short-convolution precision after every token",
    )
    ap.add_argument(
        "--require-all-bf16", action="store_true",
        help="require bfloat16 model activations plus BF16 recurrent/conv storage",
    )
    ap.add_argument(
        "--fp32-logits", action="store_true",
        help=(
            "widen BF16 final activations/LM-head weights and emit unrounded "
            "FP32 GEMV logits, matching the accelerator boundary"
        ),
    )
    ap.add_argument(
        "--native-bf16-product", action="store_true",
        help=(
            "round every HBM-backed dense BF16 product to BF16 before the "
            "FP32 four-bank reduction, including the FP32-output LM head"
        ),
    )
    ap.add_argument(
        "--decode-steps", type=int, default=0,
        help="emit this many trajectory tokens including the exported seed",
    )
    ap.add_argument(
        "--golden-output", type=Path,
        help="optional JSON trajectory reference for --decode-steps",
    )
    ap.add_argument(
        "--logits-output", type=Path,
        help="optional GDNLOG1 full-logit reference for post-seed steps",
    )
    ap.add_argument(
        "--hidden-output", type=Path,
        help="optional GDNHID1 final-normalized hidden reference for post-seed steps",
    )
    args = ap.parse_args()

    if args.require_all_bf16:
        if args.dtype not in ("bfloat16", "bf16"):
            raise SystemExit("--require-all-bf16 requires --dtype bfloat16")
        if args.recurrent_storage != "bf16":
            raise SystemExit(
                "--require-all-bf16 requires --recurrent-storage bf16"
            )
        if args.conv_storage != "bf16":
            raise SystemExit("--require-all-bf16 requires --conv-storage bf16")
        if not args.fp32_logits:
            raise SystemExit(
                "--require-all-bf16 requires --fp32-logits for the accelerator contract"
            )
    if args.native_bf16_product and not args.require_all_bf16:
        raise SystemExit(
            "--native-bf16-product requires --require-all-bf16"
        )

    kind, examples = load_fixture(args.fixture)
    if kind != REQ_KIND_LL:
        raise SystemExit(f"--fixture must be LL-kind (kind=2), got {kind}")
    ex = examples[args.example]
    prompt_ids = ex["context"]
    golden_cont = ex.get("continuation") or None

    model, device = load_model(args.model_name, args.device, args.dtype)
    if args.native_bf16_product:
        from gdn_native_bf16_product import (
            install_native_bf16_product_linears,
            patch_manifest,
        )

        patch = install_native_bf16_product_linears(model)
        print(json.dumps(patch_manifest(patch), indent=2, sort_keys=True))
    export_state(model, device, prompt_ids, args.output, golden_cont,
                 args.verify_steps, args.recurrent_storage, args.conv_storage,
                 args.decode_steps, args.golden_output, args.logits_output,
                 args.hidden_output, args.fp32_logits,
                 args.native_bf16_product)


if __name__ == "__main__":
    main()

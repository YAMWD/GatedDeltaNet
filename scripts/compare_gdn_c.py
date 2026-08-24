#!/usr/bin/env python3
"""Generate the cached GPU golden for the decode fixture."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch

from fla.models.gated_deltanet import GatedDeltaNetForCausalLM


REQ_HEADER = struct.Struct("<8s3I")
REQ_MAGIC = b"GDNREQ1\0"
REQ_KIND_LL = 2


def read_u32(blob: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", blob, offset)[0], offset + 4


def read_i32s(blob: bytes, offset: int, count: int) -> tuple[list[int], int]:
    if count == 0:
        return [], offset
    values = list(struct.unpack_from(f"<{count}i", blob, offset))
    return values, offset + count * 4


def load_fixture(path: Path) -> tuple[int, list[dict]]:
    blob = path.read_bytes()
    magic, version, kind, num_examples = REQ_HEADER.unpack_from(blob, 0)
    if magic != REQ_MAGIC or version != 1:
        raise ValueError(f"Unsupported fixture header in {path}")
    offset = REQ_HEADER.size
    if kind != REQ_KIND_LL:
        raise ValueError(f"Decode requires an LL-kind fixture, got kind {kind}")
    examples = []
    for _ in range(num_examples):
        ctx_len, offset = read_u32(blob, offset)
        cont_len, offset = read_u32(blob, offset)
        context, offset = read_i32s(blob, offset, ctx_len)
        continuation, offset = read_i32s(blob, offset, cont_len)
        examples.append({"context": context, "continuation": continuation})
    if offset != len(blob):
        raise ValueError(f"Unexpected trailing data in {path}")
    return kind, examples


def write_u32(handle, value: int) -> None:
    handle.write(struct.pack("<I", int(value)))


def write_i32_array(handle, values: list[int]) -> None:
    if values:
        handle.write(struct.pack(f"<{len(values)}i", *map(int, values)))


@torch.no_grad()
def decode_trajectory(model, device: torch.device, prompt_ids: list[int], decode_len: int):
    """Re-prefill greedy decode: each step runs a FULL forward over the growing
    prefix and takes argmax of the last position. Matches the FPGA decode bench
    numerics exactly (no recurrent cache, no model.generate)."""
    prefix = list(prompt_ids)
    golden_traj: list[int] = []
    per_step_logprob: list[float] = []
    for _ in range(decode_len):
        input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        logits = model(input_ids=input_ids).logits[0, -1].float()
        nxt = int(torch.argmax(logits).item())
        lp = float((logits[nxt] - torch.logsumexp(logits, dim=-1)).item())
        golden_traj.append(nxt)
        per_step_logprob.append(lp)
        prefix.append(nxt)
    return golden_traj, per_step_logprob


def decode_golden(model, device: torch.device, fixture_path: Path, output_path: Path) -> None:
    kind, examples = load_fixture(fixture_path)
    if kind != REQ_KIND_LL:
        raise ValueError(f"Decode golden expects an LL-kind fixture, got kind {kind}")

    decode_len = None
    out_examples = []
    rewritten = []
    for index, example in enumerate(examples):
        prompt_ids = example["context"]
        n = len(example["continuation"])
        if decode_len is None:
            decode_len = n
        elif n != decode_len:
            raise ValueError(
                f"Inconsistent cont_len across examples ({n} vs {decode_len}); "
                "all decode examples must share the same N."
            )
        golden_traj, per_step_logprob = decode_trajectory(model, device, prompt_ids, n)
        out_examples.append(
            {
                "index": index,
                "prompt_ids": prompt_ids,
                "golden_traj": golden_traj,
                "per_step_argmax": list(golden_traj),
                "per_step_logprob": per_step_logprob,
            }
        )
        rewritten.append((prompt_ids, golden_traj))

    if decode_len is None:
        decode_len = 0

    result = {
        "kind": REQ_KIND_LL,
        "decode_len": decode_len,
        "num_examples": len(out_examples),
        "examples": out_examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {output_path}")

    # Rewrite the fixture in place so cont[] holds the true golden trajectory.
    with fixture_path.open("wb") as handle:
        handle.write(REQ_HEADER.pack(REQ_MAGIC, 1, REQ_KIND_LL, len(rewritten)))
        for prompt_ids, golden_traj in rewritten:
            write_u32(handle, len(prompt_ids))
            write_u32(handle, len(golden_traj))
            write_i32_array(handle, prompt_ids)
            write_i32_array(handle, golden_traj)
    print(f"updated {fixture_path} (cont[] = golden trajectory)")


def load_model(model_name: str, device_name: str, dtype_name: str):
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    dtype = getattr(torch, dtype_name)
    model = GatedDeltaNetForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    for layer in model.model.layers:
        if hasattr(layer.attn, "mode"):
            layer.attn.mode = "fused_recurrent"
    return model, device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-golden", type=Path, required=True)
    parser.add_argument("--model-name", default="m-a-p/1.3B-100B-GatedDeltaNet-pure")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    model, device = load_model(args.model_name, args.device, args.dtype)
    decode_golden(model, device, args.decode_golden, args.output)



if __name__ == "__main__":
    main()

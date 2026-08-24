#!/usr/bin/env python3
"""Export GatedDeltaNet weights and the decode fixture for the C runtime."""

from __future__ import annotations

import argparse
import json
import struct
from contextlib import ExitStack
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer


WEIGHT_MAGIC = b"GDNWv1\0\0"
REQ_MAGIC = b"GDNREQ1\0"
WEIGHT_HEADER = struct.Struct("<8s10I2if")
REQ_HEADER = struct.Struct("<8s3I")

REQ_KIND_LL = 2

DEFAULT_MODEL = "m-a-p/1.3B-100B-GatedDeltaNet-pure"


def write_u32(handle, value: int) -> None:
    handle.write(struct.pack("<I", int(value)))


def write_i32_array(handle, values: list[int]) -> None:
    if values:
        handle.write(struct.pack(f"<{len(values)}i", *map(int, values)))


def load_snapshot(model_name: str) -> Path:
    # Accept a local directory as well as a Hub repo id, so a converted
    # checkpoint (e.g. the BF16 cast) can be exported without publishing it.
    candidate = Path(model_name)
    if candidate.is_dir() and (candidate / "config.json").exists():
        return candidate
    return Path(snapshot_download(model_name, local_files_only=True))


def tok_encode_default(tokenizer, text: str) -> list[int]:
    # Match lm-eval HFLM.tok_encode(..., add_special_tokens=None).
    return tokenizer.encode(text)


# Curated natural-English prompts for the decode benchmark. Kept short so the
# tokenized prompt lands in the 16-48 token range expected by the C testbench.
DECODE_PROMPTS = [
    "The history of the Roman Empire is a story of ambition, conquest, and "
    "the slow transformation of a small city into a vast",
    "In the early morning the fishermen pushed their boats out onto the calm "
    "water, hoping that the day would bring a good",
    "Scientists have long wondered how migrating birds manage to find their "
    "way across thousands of miles of open ocean without",
    "She opened the old wooden chest in the attic and found a bundle of "
    "letters tied with a faded ribbon, each one written",
    "The recipe calls for fresh tomatoes, a handful of basil, two cloves of "
    "garlic, and a generous drizzle of olive",
    "When the spacecraft finally entered orbit around the distant planet, the "
    "engineers in the control room held their breath and",
    "Learning to play the piano takes patience and daily practice, but the "
    "reward of playing a beautiful piece of music is",
    "The detective studied the room carefully, noting the overturned chair, "
    "the broken glass, and the single muddy footprint near the",
    "Across the rolling hills the farmers worked from dawn until dusk, "
    "gathering the golden wheat before the autumn rains could",
    "A good teacher does more than share facts; she inspires curiosity, "
    "encourages questions, and helps each student discover the joy of",
    "The river wound slowly through the green valley, past sleepy villages "
    "and ancient stone bridges that had stood for many",
    "After years of careful research the team finally announced that they had "
    "discovered a new species of butterfly living deep within the",
]


def build_decode_records(tokenizer, prefix_token_id: int, decode_len: int, limit: int | None):
    prompts = DECODE_PROMPTS if limit is None else DECODE_PROMPTS[:limit]
    records = []
    for text in prompts:
        # tok_encode_default mirrors the parity tooling: the tokenizer prepends
        # its own BOS, so the prompt is the exact prefill the FPGA reproduces.
        prompt_ids = tok_encode_default(tokenizer, text)
        if not prompt_ids or prompt_ids[0] != prefix_token_id:
            prompt_ids = [prefix_token_id] + prompt_ids
        # Placeholder continuation; the golden greedy trajectory is filled in by
        # compare_gdn_c.py and written back into the .gdnreq when regenerated.
        records.append({"prompt_ids": prompt_ids, "decode_len": decode_len})
    return records


def export_decode_fixture(tokenizer, prefix_token_id: int, decode_len: int, output_dir: Path, limit: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "decode.gdnreq"
    manifest_path = output_dir / "decode.json"
    records = build_decode_records(tokenizer, prefix_token_id, decode_len, limit)

    # Decode fixtures reuse the LL kind: ctx[] = prompt ids, cont[] = greedy
    # trajectory of length decode_len. The trajectory is a placeholder here and
    # is rewritten in place by compare_gdn_c.py once the GPU golden is computed.
    placeholder = [prefix_token_id] * decode_len
    with out_path.open("wb") as handle:
        handle.write(REQ_HEADER.pack(REQ_MAGIC, 1, REQ_KIND_LL, len(records)))
        for rec in records:
            prompt_ids = rec["prompt_ids"]
            write_u32(handle, len(prompt_ids))
            write_u32(handle, decode_len)
            write_i32_array(handle, prompt_ids)
            write_i32_array(handle, placeholder)

    manifest = {
        "task": "decode",
        "kind": REQ_KIND_LL,
        "num_examples": len(records),
        "decode_len": decode_len,
        "limit": limit,
        "path": str(out_path),
        "note": "cont[] is a placeholder; run compare_gdn_c.py --decode-golden to fill the golden trajectory",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_path}")


def tensor_names(cfg: dict) -> list[str]:
    names = ["model.embeddings.weight"]
    n = cfg["num_hidden_layers"]
    for i in range(n):
        prefix = f"model.layers.{i}"
        names.extend(
            [
                f"{prefix}.attn_norm.weight",
                f"{prefix}.attn.A_log",
                f"{prefix}.attn.dt_bias",
                f"{prefix}.attn.q_proj.weight",
                f"{prefix}.attn.k_proj.weight",
                f"{prefix}.attn.v_proj.weight",
                f"{prefix}.attn.a_proj.weight",
                f"{prefix}.attn.b_proj.weight",
                f"{prefix}.attn.q_conv1d.weight",
                f"{prefix}.attn.k_conv1d.weight",
                f"{prefix}.attn.v_conv1d.weight",
                f"{prefix}.attn.g_proj.weight",
                f"{prefix}.attn.o_norm.weight",
                f"{prefix}.attn.o_proj.weight",
                f"{prefix}.mlp_norm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        )
    names.extend(["model.norm.weight", "lm_head.weight"])
    return names


def export_weights(model_name: str, output_path: Path) -> None:
    snapshot_dir = load_snapshot(model_name)
    cfg = json.loads((snapshot_dir / "config.json").read_text())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index = json.loads((snapshot_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))

    header = WEIGHT_HEADER.pack(
        WEIGHT_MAGIC,
        1,
        int(cfg["vocab_size"]),
        int(cfg["hidden_size"]),
        int(cfg["num_hidden_layers"]),
        int(cfg["num_heads"]),
        int(cfg.get("num_v_heads") or cfg["num_heads"]),
        int(cfg["head_dim"]),
        int(256 * ((int(cfg["hidden_size"] * cfg["hidden_ratio"] * 2 / 3) + 255) // 256)),
        int(cfg["conv_size"]),
        int(cfg["max_position_embeddings"]),
        int(cfg["bos_token_id"]),
        int(cfg["eos_token_id"]),
        float(cfg["norm_eps"]),
    )

    with output_path.open("wb") as handle, ExitStack() as stack:
        handle.write(header)
        readers = {
            shard: stack.enter_context(safe_open(snapshot_dir / shard, framework="pt", device="cpu"))
            for shard in shard_names
        }
        for name in tensor_names(cfg):
            reader = readers[weight_map[name]]
            tensor = reader.get_tensor(name).float().contiguous()
            if name.endswith("conv1d.weight"):
                tensor = tensor[:, 0, :].contiguous()
            handle.write(tensor.numpy().tobytes(order="C"))

    meta = {
        "model_name": model_name,
        "output": str(output_path),
        "config": {
            "vocab_size": cfg["vocab_size"],
            "hidden_size": cfg["hidden_size"],
            "num_hidden_layers": cfg["num_hidden_layers"],
            "num_heads": cfg["num_heads"],
            "num_v_heads": cfg.get("num_v_heads") or cfg["num_heads"],
            "head_dim": cfg["head_dim"],
            "intermediate_size": int(256 * ((int(cfg["hidden_size"] * cfg["hidden_ratio"] * 2 / 3) + 255) // 256)),
            "conv_size": cfg["conv_size"],
            "max_position_embeddings": cfg["max_position_embeddings"],
            "bos_token_id": cfg["bos_token_id"],
            "eos_token_id": cfg["eos_token_id"],
            "norm_eps": cfg["norm_eps"],
        },
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    weight_parser = subparsers.add_parser("weights")
    weight_parser.add_argument("--model-name", default=DEFAULT_MODEL)
    weight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("c_impl/artifacts/gdn-1.3b-f32.gdnw"),
    )

    decode_parser = subparsers.add_parser("decode")
    decode_parser.add_argument("--model-name", default=DEFAULT_MODEL)
    decode_parser.add_argument("--decode-len", type=int, default=64)
    decode_parser.add_argument("--limit", type=int, default=None)
    decode_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("c_impl/fixtures_decode"),
    )

    args = parser.parse_args()
    if args.cmd == "weights":
        export_weights(args.model_name, args.output)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prefix_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if prefix_token_id is None:
        raise ValueError("Tokenizer must define BOS or EOS token for prefix_token_id.")
    if args.cmd == "decode":
        export_decode_fixture(tokenizer, prefix_token_id, args.decode_len, args.output_dir, args.limit)
        return



if __name__ == "__main__":
    main()

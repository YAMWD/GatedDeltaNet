#!/usr/bin/env python3
"""Losslessly preserve checkpoint structure while casting floating tensors to BF16."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cast every floating-point safetensors tensor to BF16."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace files in an existing destination.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists() and any(destination.iterdir()) and not args.force:
        raise FileExistsError(
            f"{destination} is not empty; pass --force to replace its files"
        )
    destination.mkdir(parents=True, exist_ok=True)

    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        shard_names = [path.name for path in sorted(source.glob("*.safetensors"))]
        index = None
    if not shard_names:
        raise FileNotFoundError(f"No safetensors shards in {source}")

    source_dtypes: Counter[str] = Counter()
    output_dtypes: Counter[str] = Counter()
    tensor_count = 0
    total_size = 0
    shard_records: list[dict[str, Any]] = []

    for shard_name in shard_names:
        source_shard = source / shard_name
        output_shard = destination / shard_name
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(source_shard, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                source_dtypes[dtype_name(tensor.dtype)] += 1
                if tensor.is_floating_point():
                    tensor = tensor.to(torch.bfloat16)
                tensor = tensor.contiguous()
                output_dtypes[dtype_name(tensor.dtype)] += 1
                tensor_count += 1
                total_size += tensor.numel() * tensor.element_size()
                tensors[key] = tensor
        save_file(tensors, output_shard, metadata=metadata)
        shard_records.append(
            {
                "name": shard_name,
                "source_sha256": sha256(source_shard),
                "output_sha256": sha256(output_shard),
                "output_bytes": output_shard.stat().st_size,
            }
        )

    copied_files = []
    for name in (
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        path = source / name
        if path.exists():
            shutil.copy2(path, destination / name)
            copied_files.append(name)

    config = json.loads((source / "config.json").read_text())
    config["torch_dtype"] = "bfloat16"
    (destination / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    if index is not None:
        index.setdefault("metadata", {})["total_size"] = total_size
        (destination / index_path.name).write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )

    manifest = {
        "conversion": "all floating-point tensors cast to torch.bfloat16",
        "source": str(source),
        "destination": str(destination),
        "source_config_sha256": sha256(source / "config.json"),
        "output_config_sha256": sha256(destination / "config.json"),
        "tensor_count": tensor_count,
        "tensor_payload_bytes": total_size,
        "source_tensor_dtypes": dict(sorted(source_dtypes.items())),
        "output_tensor_dtypes": dict(sorted(output_dtypes.items())),
        "copied_files": copied_files,
        "shards": shard_records,
    }
    (destination / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

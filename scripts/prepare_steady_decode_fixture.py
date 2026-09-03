#!/usr/bin/env python3
"""Derive a fixed 4K prompt and teacher stream from the pinned WikiText fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--teacher-length", type=int, default=4096)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixture_tokens(path: Path) -> list[int]:
    blob = path.read_bytes()
    magic, version, kind, documents = struct.unpack_from("<8s3I", blob, 0)
    if magic != b"GDNREQ1\0" or version != 1 or kind != 3:
        raise ValueError("expected a GDNREQ1 rolling-loglikelihood fixture")
    offset = struct.calcsize("<8s3I")
    tokens: list[int] = []
    for _ in range(documents):
        _, _, windows = struct.unpack_from("<3I", blob, offset)
        offset += 12
        for _ in range(windows):
            context_length, continuation_length = struct.unpack_from(
                "<2I", blob, offset
            )
            offset += 8
            context = struct.unpack_from(
                f"<{context_length}i", blob, offset
            )
            offset += context_length * 4
            continuation = struct.unpack_from(
                f"<{continuation_length}i", blob, offset
            )
            offset += continuation_length * 4
            tokens.extend(context)
            tokens.extend(continuation)
    if offset != len(blob):
        raise ValueError("fixture has trailing bytes")
    return tokens


def write_tokens(path: Path, values: list[int]) -> None:
    path.write_text("\n".join(str(value) for value in values) + "\n")


def main() -> None:
    args = parse_args()
    if args.prompt_length <= 1 or args.teacher_length <= 0:
        raise ValueError("invalid prompt/teacher length")
    tokens = fixture_tokens(args.fixture)
    required = args.prompt_length + args.teacher_length
    if len(tokens) < required:
        raise ValueError(f"fixture has {len(tokens)} tokens, need {required}")
    prompt = tokens[: args.prompt_length]
    teacher = tokens[args.prompt_length : required]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = args.output_dir / "prompt_4096.tokens"
    teacher_path = args.output_dir / "teacher_4096.tokens"
    write_tokens(prompt_path, prompt)
    write_tokens(teacher_path, teacher)
    manifest = {
        "schema": "gdn-steady-fixture-v1",
        "source_fixture": str(args.fixture.resolve()),
        "source_fixture_sha256": sha256(args.fixture),
        "construction": "concatenate rolling-window contexts and continuations in fixture order",
        "prompt_length": len(prompt),
        "teacher_length": len(teacher),
        "prompt_sha256": sha256(prompt_path),
        "teacher_sha256": sha256(teacher_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


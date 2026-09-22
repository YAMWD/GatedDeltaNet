#!/usr/bin/env python3
"""Capture immutable artifact, source, dataset, and runtime identities."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_COMMIT = "a072ac63535ae7384ce6ba79d2a54c0835122937"
EXPECTED_TREE = "d717ea9583b871ca5decf4bc9f1292a835d92fc2"
EXPECTED_CHECKPOINT = "930ed6ae4ac629c86cb9855bb3dcb0a0974a29aa"
EXPECTED_XCLBIN = "fb4fc63f76bc1ee485665f21102596270930d6d39643b8eb5ae7d4f899d289ab"
EXPECTED_WEIGHTS = "ba81d3536e868e1057b81cc71354060cfba968c1b356b2fa70add3b83a84c298"
EXPECTED_KERNEL_CPP = "2bc240e6a5cf24b23bd83aa3ad518552bd475e72c8ac5c54a6cdf8bc991ad020"
EXPECTED_KERNEL_HEADER = "906b11e5ca368b08da0c4dfbdb2e3a8ddf10683efbc912e4f07223124f89ac53"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--xclbin", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fpga-bdf")
    parser.add_argument(
        "--artifact", action="append", default=[], metavar="NAME=PATH"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*values: str) -> str | None:
    try:
        completed = subprocess.run(
            values, check=True, capture_output=True, text=True, timeout=30
        )
        return completed.stdout.strip() or completed.stderr.strip()
    except Exception:
        return None


def git(repo: Path, *values: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *values],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    commit = git(repo, "rev-parse", "HEAD")
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    checkpoint = args.checkpoint.resolve()
    xclbin_hash = sha256(args.xclbin)
    weight_hash = sha256(args.weights)
    failures = []
    ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor",
         EXPECTED_COMMIT, commit]
    ).returncode == 0
    if not ancestry:
        failures.append(f"evaluation commit {commit} is not based on {EXPECTED_COMMIT}")
    if checkpoint.name != EXPECTED_CHECKPOINT:
        failures.append(
            f"checkpoint snapshot {checkpoint.name} != {EXPECTED_CHECKPOINT}"
        )
    if xclbin_hash != EXPECTED_XCLBIN:
        failures.append(f"xclbin {xclbin_hash} != {EXPECTED_XCLBIN}")
    if weight_hash != EXPECTED_WEIGHTS:
        failures.append(f"weights {weight_hash} != {EXPECTED_WEIGHTS}")
    if sha256(repo / "c_impl/gdn_model.cpp") != EXPECTED_KERNEL_CPP:
        failures.append("gdn_model.cpp does not match the retained XCLBIN source")
    if sha256(repo / "c_impl/gdn_model.h") != EXPECTED_KERNEL_HEADER:
        failures.append("gdn_model.h does not match the retained XCLBIN source")
    if failures:
        raise RuntimeError("identity gate failed: " + "; ".join(failures))

    files: dict[str, dict[str, Any]] = {}
    defaults = {
        "xclbin": args.xclbin,
        "weights": args.weights,
        "gdn_model.cpp": repo / "c_impl/gdn_model.cpp",
        "gdn_model.h": repo / "c_impl/gdn_model.h",
        "host.cpp": repo / "c_impl/host.cpp",
    }
    for item in args.artifact:
        name, separator, raw_path = item.partition("=")
        if not separator or not name:
            raise ValueError(f"invalid --artifact {item!r}")
        defaults[name] = Path(raw_path)
    for name, path in defaults.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        files[name] = {
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved),
        }

    gpu_query = command(
        "nvidia-smi",
        "--query-gpu=uuid,name,driver_version",
        "--format=csv,noheader",
    )
    manifest = {
        "schema": "gdn-fpga-evaluation-manifest-v1",
        "repository": str(repo),
        "source_base_commit": EXPECTED_COMMIT,
        "source_base_tree": EXPECTED_TREE,
        "evaluation_commit": commit,
        "evaluation_tree": tree,
        "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_status": git(repo, "status", "--short"),
        "checkpoint_snapshot": str(checkpoint),
        "checkpoint_snapshot_id": checkpoint.name,
        "files": files,
        "environment": {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "python": platform.python_version(),
            "pytorch": package_version("torch"),
            "transformers": package_version("transformers"),
            "fla": package_version("flash-linear-attention", "fla"),
            "lm_eval": package_version("lm_eval"),
            "datasets": package_version("datasets"),
            "xrt": command("xbutil", "--version"),
            "xocl_driver": (
                Path("/sys/module/xocl/version").read_text().strip()
                if Path("/sys/module/xocl/version").exists()
                else None
            ),
            "gpu_query": gpu_query,
            "fpga_bdf": args.fpga_bdf,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "identity_gate": "pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

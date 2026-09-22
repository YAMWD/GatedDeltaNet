#!/usr/bin/env python3
"""Run a standalone NVIDIA GPU latency/energy evaluation from a fresh clone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path


MODEL_ID = "m-a-p/1.3B-100B-GatedDeltaNet-pure"
MODEL_REVISION = "930ed6ae4ac629c86cb9855bb3dcb0a0974a29aa"
REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="local checkpoint; omitted downloads the pinned model")
    parser.add_argument("--output-dir", type=Path, required=True, help="new or empty directory")
    parser.add_argument("--device", default="cuda:0", help="logical CUDA device within the allocation")
    parser.add_argument("--require-gpu-name", help="fail unless the actual GPU name contains this string")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--decode-mode", choices=("fused_recurrent", "chunk"), default="fused_recurrent")
    parser.add_argument("--idle-seconds", type=int, default=60)
    parser.add_argument("--warmup-seconds", type=int, default=30)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--intervals", type=int, default=3)
    parser.add_argument("--power-period", type=float, default=0.2)
    parser.add_argument("--fixture", type=Path, default=REPO / "c_impl/fixtures_full/wikitext.gdnreq")
    return parser.parse_args()


def run_logged(command: list[str], log: Path) -> None:
    print("[run] " + " ".join(command), flush=True)
    with log.open("w") as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdout is not None:
                process.stdout.close()


def git(*args: str) -> str | None:
    result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing run: {output}")

    def terminate(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    status = "failed"
    try:
        if args.model is None:
            from huggingface_hub import snapshot_download

            model = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION))
        else:
            model = args.model.resolve()
        if not (model / "config.json").is_file():
            raise FileNotFoundError(f"not a local model checkpoint: {model}")
        source_names = [Path(__file__).name, "benchmark_gpu_fp32.py", "sample_device_power.py",
                        "gpu_power.py", "aggregate_power_eval.py", "prepare_steady_decode_fixture.py"]
        manifest = {
            "schema": "gdn-gpu-eval-run-v1", "command": sys.argv,
            "source_commit": git("rev-parse", "HEAD"),
            "source_status": git("status", "--short"),
            "source_sha256": {name: hashlib.sha256((REPO / "scripts" / name).read_bytes()).hexdigest()
                              for name in source_names},
            "python": sys.version, "platform": platform.platform(),
            "model": str(model),
            "model_revision": MODEL_REVISION if args.model is None else "user-supplied local checkpoint",
            "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "configuration": {key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()},
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True,
                                capture_output=True, text=True)
        (output / "environment.txt").write_text(freeze.stdout)
        scripts = REPO / "scripts"
        run_logged([sys.executable, str(scripts / "prepare_steady_decode_fixture.py"),
                    "--fixture", str(args.fixture.resolve()), "--output-dir", str(output / "fixture")],
                   output / "fixture.log")
        command = [sys.executable, "-u", str(scripts / "benchmark_gpu_fp32.py"),
                   "--model", str(model), "--device", args.device,
                   "--dtype", args.dtype, "--decode-mode", args.decode_mode,
                   "--prompt-tokens", str(output / "fixture/prompt_4096.tokens"),
                   "--teacher-tokens", str(output / "fixture/teacher_4096.tokens"),
                   "--output", str(output / "gpu_timing.json"),
                   "--power-output", str(output / "gpu_power.jsonl")]
        for name in ("idle_seconds", "warmup_seconds", "interval_seconds", "intervals", "power_period"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        if args.require_gpu_name:
            command.extend(["--require-gpu-name", args.require_gpu_name])
        run_logged(command, output / "benchmark.log")
        run_logged([sys.executable, str(scripts / "aggregate_power_eval.py"),
                    "--gpu-timing", str(output / "gpu_timing.json"),
                    "--gpu-power", str(output / "gpu_power.jsonl"),
                    "--output-json", str(output / "summary.json"),
                    "--output-csv", str(output / "summary.csv")], output / "aggregation.log")
        summary = json.loads((output / "summary.json").read_text())["gpu"]
        print(json.dumps({key: summary[key] for key in (
            "device", "production_tpot_ms", "tokens_per_second",
            "gross_joules_per_token", "gross_tokens_per_joule")}, indent=2))
        status = "complete"
    finally:
        (output / "status.json").write_text(json.dumps({"status": status}) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure NVIDIA GPU eager decode TPOT with the common teacher-forced stream.

The historical filename and FP32 default remain compatible with existing jobs.
The standalone power launcher defaults to BF16 and enables managed telemetry.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import statistics
import signal
import time
from pathlib import Path
from typing import Any

import torch
from fla.models.gated_deltanet import GatedDeltaNetForCausalLM

from gpu_power import NvmlDevice, PowerSampler, cuda_uuid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=Path, required=True)
    parser.add_argument("--teacher-tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-gpu-name")
    parser.add_argument("--idle-seconds", type=int, default=60)
    parser.add_argument("--warmup-seconds", type=int, default=30)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--intervals", type=int, default=3)
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--decode-mode", default="fused_recurrent",
                        choices=["fused_recurrent", "chunk"])
    parser.add_argument("--power-output", type=Path)
    parser.add_argument("--power-period", type=float, default=0.2)
    return parser.parse_args()


def read_tokens(path: Path) -> list[int]:
    tokens = [int(value) for value in path.read_text().split()]
    if not tokens:
        raise ValueError(f"empty token file: {path}")
    return tokens


def set_mode(model: torch.nn.Module, mode: str) -> None:
    if hasattr(model.config, "attn_mode"):
        model.config.attn_mode = mode
    for layer in model.model.layers:
        layer.attn.mode = mode


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean_ms": statistics.fmean(values) * 1000.0,
        "median_ms": statistics.median(values) * 1000.0,
        "p95_ms": ordered[p95_index] * 1000.0,
        "min_ms": ordered[0] * 1000.0,
        "max_ms": ordered[-1] * 1000.0,
    }


@torch.inference_mode()
def benchmark(args: argparse.Namespace, stack: ExitStack) -> None:
    if args.interval_seconds <= 0 or args.intervals <= 0:
        raise ValueError("interval duration/count must be positive")
    if args.idle_seconds <= 0 or args.warmup_seconds < 0:
        raise ValueError("idle must be positive and warmup nonnegative")
    if not math.isfinite(args.power_period) or args.power_period <= 0:
        raise ValueError("power period must be finite and positive")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this benchmark requires a CUDA device")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if args.require_gpu_name and args.require_gpu_name.lower() not in properties.name.lower():
        raise ValueError(f"expected {args.require_gpu_name}, allocated {properties.name}")
    gpu_uuid = cuda_uuid(torch, device)
    gpu = NvmlDevice(gpu_uuid=gpu_uuid) if args.power_output else None
    identity = gpu.identity() if gpu else {}
    prompt = read_tokens(args.prompt_tokens)
    teacher = read_tokens(args.teacher_tokens)
    if len(prompt) != 4096:
        raise ValueError(f"expected a 4096-token prompt, got {len(prompt)}")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = GatedDeltaNetForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    # State corresponds to prompt[:-1], exactly matching the FPGA handoff.
    set_mode(model, "chunk")
    prefix = torch.tensor([prompt[:-1]], dtype=torch.long, device=device)
    prefill = model.model(
        input_ids=prefix, use_cache=True, return_dict=True
    )
    past = prefill.past_key_values
    del prefill, prefix
    torch.cuda.synchronize(device)
    set_mode(model, args.decode_mode)

    if args.power_output:
        stack.enter_context(PowerSampler(args.power_output, gpu_uuid, args.power_period))

    token_stream = [prompt[-1], *teacher]
    token_index = 0

    def run_one() -> tuple[float, float]:
        nonlocal past, token_index
        host_start = time.perf_counter()
        token = torch.tensor(
            [[token_stream[token_index]]], dtype=torch.long, device=device
        )
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
        output = model(
            input_ids=token,
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        selected = output.logits[:, -1, :].argmax(dim=-1)
        event_end.record()
        selected_token = int(selected.item())
        del selected_token
        event_end.synchronize()
        device_seconds = event_start.elapsed_time(event_end) / 1000.0
        host_seconds = time.perf_counter() - host_start
        past = output.past_key_values
        token_index = (token_index + 1) % len(token_stream)
        return host_seconds, device_seconds

    idle_energy_start = gpu.energy_mj() if gpu else None
    idle_start_unix = time.time()
    print(f"[benchmark] loaded-idle window {args.idle_seconds}s", flush=True)
    time.sleep(args.idle_seconds)
    warmup_start_unix = time.time()
    idle_energy_end = gpu.energy_mj() if gpu else None
    warmup_deadline = time.monotonic() + args.warmup_seconds
    warmup_steps = 0
    while time.monotonic() < warmup_deadline:
        run_one()
        warmup_steps += 1
    warmup_end_unix = time.time()
    print(f"[benchmark] warmup complete steps={warmup_steps}", flush=True)

    intervals: list[dict[str, Any]] = []
    for interval_index in range(args.intervals):
        energy_start = gpu.energy_mj() if gpu else None
        start_unix = time.time()
        deadline = time.monotonic() + args.interval_seconds
        host_values: list[float] = []
        device_values: list[float] = []
        while time.monotonic() < deadline:
            host_seconds, device_seconds = run_one()
            host_values.append(host_seconds)
            device_values.append(device_seconds)
        end_unix = time.time()
        energy_end = gpu.energy_mj() if gpu else None
        intervals.append({
            "index": interval_index,
            "start_unix": start_unix,
            "end_unix": end_unix,
            "steps": len(host_values),
            "energy_start_mj": energy_start,
            "energy_end_mj": energy_end,
            "tokens_per_second": len(host_values) / (end_unix - start_unix),
            "production_tpot": summarize(host_values),
            "kernel": summarize(device_values),
        })
        print(
            f"[benchmark] interval {interval_index + 1}/{args.intervals} "
            f"steps={len(host_values)}",
            flush=True,
        )

    result = {
        "schema": "gdn-steady-tpot-v1",
        "device": properties.name,
        "execution": f"{args.dtype} eager FLA/PyTorch, decode={args.decode_mode}",
        "model": str(args.model.resolve()),
        "prompt_token_file": str(args.prompt_tokens.resolve()),
        "teacher_token_file": str(args.teacher_tokens.resolve()),
        "prompt_sha256": hashlib.sha256(args.prompt_tokens.read_bytes()).hexdigest(),
        "teacher_sha256": hashlib.sha256(args.teacher_tokens.read_bytes()).hexdigest(),
        "prompt_tokens": len(prompt),
        "teacher_stream_tokens": len(teacher),
        "teacher_forced": True,
        "full_lm_head": True,
        "argmax": True,
        "torch_compile": False,
        "tf32": False,
        "cuda_graphs": False,
        "dtype": args.dtype,
        "decode_mode": args.decode_mode,
        "prefill_mode": "chunk",
        "state_reset_between_intervals": False,
        "token_stream": "cyclic [prompt[-1], *teacher], including warmup",
        "energy_boundary": "decode-only GPU board; host/server and prefill excluded",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "nvml": identity,
        "gpu_uuid": gpu_uuid,
        "gpu_name": properties.name,
        "idle_energy_start_mj": idle_energy_start,
        "idle_energy_end_mj": idle_energy_end,
        "idle_start_unix": idle_start_unix,
        "warmup_start_unix": warmup_start_unix,
        "warmup_end_unix": warmup_end_unix,
        "warmup_steps": warmup_steps,
        "intervals": intervals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[benchmark] wrote {args.output}", flush=True)


def main() -> None:
    args = parse_args()
    # Slurm TERM must run ExitStack cleanup for the sampler as well.
    def terminate(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    with ExitStack() as stack:
        benchmark(args, stack)


if __name__ == "__main__":
    main()

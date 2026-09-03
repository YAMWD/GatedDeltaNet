#!/usr/bin/env python3
"""Measure A100 FP32 eager decode TPOT with the common teacher-forced stream."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from fla.models.gated_deltanet import GatedDeltaNetForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=Path, required=True)
    parser.add_argument("--teacher-tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--idle-seconds", type=int, default=60)
    parser.add_argument("--warmup-seconds", type=int, default=30)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--intervals", type=int, default=3)
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
def main() -> None:
    args = parse_args()
    if args.interval_seconds <= 0 or args.intervals <= 0:
        raise ValueError("interval duration/count must be positive")
    device = torch.device(args.device)
    prompt = read_tokens(args.prompt_tokens)
    teacher = read_tokens(args.teacher_tokens)
    if len(prompt) != 4096:
        raise ValueError(f"expected a 4096-token prompt, got {len(prompt)}")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = GatedDeltaNetForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
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
    set_mode(model, "fused_recurrent")

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

    idle_start_unix = time.time()
    print(f"[benchmark] loaded-idle window {args.idle_seconds}s", flush=True)
    time.sleep(args.idle_seconds)
    warmup_start_unix = time.time()
    warmup_deadline = time.monotonic() + args.warmup_seconds
    warmup_steps = 0
    while time.monotonic() < warmup_deadline:
        run_one()
        warmup_steps += 1
    warmup_end_unix = time.time()
    print(f"[benchmark] warmup complete steps={warmup_steps}", flush=True)

    intervals: list[dict[str, Any]] = []
    for interval_index in range(args.intervals):
        start_unix = time.time()
        deadline = time.monotonic() + args.interval_seconds
        host_values: list[float] = []
        device_values: list[float] = []
        while time.monotonic() < deadline:
            host_seconds, device_seconds = run_one()
            host_values.append(host_seconds)
            device_values.append(device_seconds)
        end_unix = time.time()
        intervals.append({
            "index": interval_index,
            "start_unix": start_unix,
            "end_unix": end_unix,
            "steps": len(host_values),
            "tokens_per_second": len(host_values) / (end_unix - start_unix),
            "production_tpot": summarize(host_values),
            "kernel": summarize(device_values),
        })
        print(
            f"[benchmark] interval {interval_index + 1}/{args.intervals} "
            f"steps={len(host_values)}",
            flush=True,
        )

    try:
        import pynvml

        pynvml.nvmlInit()
        visible = __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "0")
        physical_index = int(visible.split(",")[0]) if visible.split(",")[0].isdigit() else 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
        gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_uuid, bytes):
            gpu_uuid = gpu_uuid.decode()
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
    except Exception:
        gpu_uuid = None
        gpu_name = None

    result = {
        "schema": "gdn-steady-tpot-v1",
        "device": "A100",
        "execution": "FP32 eager FLA/PyTorch",
        "model": str(args.model.resolve()),
        "prompt_token_file": str(args.prompt_tokens.resolve()),
        "teacher_token_file": str(args.teacher_tokens.resolve()),
        "prompt_tokens": len(prompt),
        "teacher_stream_tokens": len(teacher),
        "teacher_forced": True,
        "full_lm_head": True,
        "argmax": True,
        "torch_compile": False,
        "tf32": False,
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "idle_start_unix": idle_start_unix,
        "warmup_start_unix": warmup_start_unix,
        "warmup_end_unix": warmup_end_unix,
        "warmup_steps": warmup_steps,
        "intervals": intervals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[benchmark] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()


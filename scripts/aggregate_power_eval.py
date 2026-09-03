#!/usr/bin/env python3
"""Integrate timestamped device power over matched TPOT benchmark windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpga-timing", type=Path, required=True)
    parser.add_argument("--fpga-power", type=Path, required=True)
    parser.add_argument("--gpu-timing", type=Path, required=True)
    parser.add_argument("--gpu-power", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_power(path: Path) -> tuple[dict[str, Any], list[tuple[float, float]]]:
    identity = None
    samples = []
    errors = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["kind"] == "identity":
            identity = record
        elif record["kind"] == "sample":
            samples.append((float(record["unix_seconds"]), float(record["watts"])))
        elif record["kind"] == "error":
            errors.append(record)
    if identity is None or len(samples) < 2:
        raise RuntimeError(f"power telemetry unavailable/incomplete: {path}")
    if errors and len(errors) > max(3, len(samples) // 20):
        raise RuntimeError(
            f"too many telemetry failures in {path}: {len(errors)} errors, "
            f"{len(samples)} samples"
        )
    samples.sort()
    return identity, samples


def interpolate(samples: list[tuple[float, float]], timestamp: float) -> float:
    if timestamp < samples[0][0] or timestamp > samples[-1][0]:
        raise RuntimeError("power trace does not cover a timing boundary")
    left = samples[0]
    for right in samples[1:]:
        if right[0] >= timestamp:
            if right[0] == left[0]:
                return right[1]
            fraction = (timestamp - left[0]) / (right[0] - left[0])
            return left[1] + fraction * (right[1] - left[1])
        left = right
    return samples[-1][1]


def window_samples(
    samples: list[tuple[float, float]], start: float, end: float
) -> list[tuple[float, float]]:
    if end <= start:
        raise ValueError("invalid integration window")
    inner = [(t, p) for t, p in samples if start < t < end]
    return [
        (start, interpolate(samples, start)),
        *inner,
        (end, interpolate(samples, end)),
    ]


def integrate(trace: list[tuple[float, float]]) -> float:
    return sum(
        (right_t - left_t) * (left_p + right_p) * 0.5
        for (left_t, left_p), (right_t, right_p) in zip(trace, trace[1:])
    )


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def summarize_device(
    name: str,
    timing_path: Path,
    power_path: Path,
) -> dict[str, Any]:
    timing = json.loads(timing_path.read_text())
    identity, samples = load_power(power_path)
    idle_trace = window_samples(
        samples,
        float(timing["idle_start_unix"]),
        float(timing["warmup_start_unix"]),
    )
    idle_values = [power for _, power in idle_trace]
    idle_watts = statistics.fmean(idle_values)

    interval_results = []
    all_active_values: list[float] = []
    total_steps = 0
    total_seconds = 0.0
    total_energy = 0.0
    production_weighted = 0.0
    kernel_weighted = 0.0
    for interval in timing["intervals"]:
        start = float(interval["start_unix"])
        end = float(interval["end_unix"])
        steps = int(interval["steps"])
        trace = window_samples(samples, start, end)
        energy = integrate(trace)
        seconds = end - start
        active_values = [power for _, power in trace]
        all_active_values.extend(active_values)
        total_steps += steps
        total_seconds += seconds
        total_energy += energy
        production_weighted += float(interval["production_tpot"]["mean_ms"]) * steps
        kernel_weighted += float(interval["kernel"]["mean_ms"]) * steps
        interval_results.append({
            "index": int(interval["index"]),
            "steps": steps,
            "seconds": seconds,
            "gross_joules": energy,
            "gross_joules_per_token": energy / steps,
            "idle_subtracted_joules_per_token": (
                energy - idle_watts * seconds
            ) / steps,
            "mean_watts": statistics.fmean(active_values),
        })
    gross_joules_per_token = total_energy / total_steps
    idle_subtracted_joules_per_token = (
        total_energy - idle_watts * total_seconds
    ) / total_steps
    return {
        "device": name,
        "timing_path": str(timing_path.resolve()),
        "power_path": str(power_path.resolve()),
        "power_identity": identity,
        "steps": total_steps,
        "seconds": total_seconds,
        "production_tpot_ms": production_weighted / total_steps,
        "device_time_ms": kernel_weighted / total_steps,
        "tokens_per_second": total_steps / total_seconds,
        "idle_mean_watts": idle_watts,
        "active_mean_watts": statistics.fmean(all_active_values),
        "active_median_watts": statistics.median(all_active_values),
        "active_p95_watts": p95(all_active_values),
        "gross_joules_per_token": gross_joules_per_token,
        "idle_subtracted_joules_per_token": idle_subtracted_joules_per_token,
        "gross_tokens_per_joule": 1.0 / gross_joules_per_token,
        "idle_subtracted_tokens_per_joule": (
            1.0 / idle_subtracted_joules_per_token
        ),
        "tokens_per_second_per_watt": (
            total_steps / total_seconds
        ) / statistics.fmean(all_active_values),
        "intervals": interval_results,
    }


def main() -> None:
    args = parse_args()
    fpga = summarize_device("U55C all-BF16 accelerator", args.fpga_timing, args.fpga_power)
    gpu = summarize_device("A100 FP32 eager", args.gpu_timing, args.gpu_power)
    comparison = {
        "headline": "U55C all-BF16 accelerator versus A100 FP32 eager",
        "energy_boundary": "decode-only board energy; host/server and prefill excluded",
        "fpga_over_gpu_production_tpot": (
            fpga["production_tpot_ms"] / gpu["production_tpot_ms"]
        ),
        "fpga_over_gpu_device_time": (
            fpga["device_time_ms"] / gpu["device_time_ms"]
        ),
        "fpga_over_gpu_gross_energy_efficiency": (
            fpga["gross_tokens_per_joule"] / gpu["gross_tokens_per_joule"]
        ),
        "fpga_over_gpu_idle_subtracted_energy_efficiency": (
            fpga["idle_subtracted_tokens_per_joule"] /
            gpu["idle_subtracted_tokens_per_joule"]
        ),
    }
    result = {
        "schema": "gdn-power-evaluation-v1",
        "fpga": fpga,
        "gpu": gpu,
        "comparison": comparison,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    fields = [
        "device", "steps", "seconds", "production_tpot_ms",
        "device_time_ms", "tokens_per_second", "idle_mean_watts",
        "active_mean_watts", "active_median_watts", "active_p95_watts",
        "gross_joules_per_token", "idle_subtracted_joules_per_token",
        "gross_tokens_per_joule", "idle_subtracted_tokens_per_joule",
        "tokens_per_second_per_watt",
    ]
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: fpga[key] for key in fields})
        writer.writerow({key: gpu[key] for key in fields})


if __name__ == "__main__":
    main()


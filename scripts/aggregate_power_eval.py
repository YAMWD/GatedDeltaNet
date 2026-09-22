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
    parser.add_argument("--fpga-timing", type=Path)
    parser.add_argument("--fpga-power", type=Path)
    parser.add_argument("--gpu-timing", type=Path, required=True)
    parser.add_argument("--gpu-power", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--boundary-tolerance-seconds", type=float, default=0.0,
        help="hold a boundary at the nearest edge sample when it lies at most "
             "this far outside the power trace (default 0: strict)",
    )
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
            timestamp, watts = float(record["unix_seconds"]), float(record["watts"])
            if not math.isfinite(timestamp) or not math.isfinite(watts) or watts <= 0:
                raise ValueError(f"invalid power sample in {path}")
            samples.append((timestamp, watts))
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


BOUNDARY_TOLERANCE_SECONDS = 0.0
BOUNDARY_CLAMPS: list[dict] = []


def interpolate(samples: list[tuple[float, float]], timestamp: float) -> float:
    if timestamp < samples[0][0] or timestamp > samples[-1][0]:
        # A boundary just outside the trace (within --boundary-tolerance-seconds,
        # at most one sample period) is held at the edge sample's power. Off by
        # default; every clamp is recorded in the output JSON.
        edge = samples[0] if timestamp < samples[0][0] else samples[-1]
        gap = abs(timestamp - edge[0])
        if gap <= BOUNDARY_TOLERANCE_SECONDS:
            BOUNDARY_CLAMPS.append({
                "boundary_unix": timestamp,
                "edge_sample_unix": edge[0],
                "gap_seconds": gap,
                "held_watts": edge[1],
            })
            return edge[1]
        raise RuntimeError(
            "power trace does not cover a timing boundary "
            f"(gap {gap:.3f} s beyond the trace edge)"
        )
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
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
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


def counter_joules(start: Any, end: Any) -> float | None:
    """Use a supported, advancing counter; never hide a counter reset."""
    if start is None or end is None:
        return None
    start, end = float(start), float(end)
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
        raise ValueError("invalid or reset NVML energy counter")
    return (end - start) / 1000.0 if end > start else None


def positive_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or numerator <= 0 or denominator <= 0:
        return None
    return numerator / denominator


def summarize_device(
    name: str | None,
    timing_path: Path,
    power_path: Path,
) -> dict[str, Any]:
    timing = json.loads(timing_path.read_text())
    identity, samples = load_power(power_path)
    if timing.get("gpu_uuid") and timing["gpu_uuid"] != identity.get("gpu_uuid"):
        raise ValueError("timing and power traces refer to different GPU UUIDs")
    if name is None:
        name = " ".join(filter(None, [identity.get("gpu_name", timing.get("device", "GPU")),
                                     timing.get("execution")]))
    idle_trace = window_samples(
        samples,
        float(timing["idle_start_unix"]),
        float(timing["warmup_start_unix"]),
    )
    idle_seconds = float(timing["warmup_start_unix"]) - float(timing["idle_start_unix"])
    idle_energy = counter_joules(timing.get("idle_energy_start_mj"),
                                 timing.get("idle_energy_end_mj"))
    idle_source = "nvml_energy_counter" if idle_energy is not None else "sampled_power"
    if idle_energy is None:
        idle_energy = integrate(idle_trace)
    idle_watts = idle_energy / idle_seconds

    interval_results = []
    all_active_values: list[float] = []
    total_steps = 0
    total_seconds = 0.0
    total_energy = 0.0
    total_sampled_energy = 0.0
    production_weighted = 0.0
    kernel_weighted = 0.0
    for interval in timing["intervals"]:
        start = float(interval["start_unix"])
        end = float(interval["end_unix"])
        steps = int(interval["steps"])
        if steps <= 0:
            raise ValueError("timing interval has no decoded tokens")
        trace = window_samples(samples, start, end)
        sampled_energy = integrate(trace)
        energy = counter_joules(interval.get("energy_start_mj"), interval.get("energy_end_mj"))
        energy_source = "nvml_energy_counter" if energy is not None else "sampled_power"
        if energy is None:
            energy = sampled_energy
        seconds = end - start
        active_values = [power for _, power in trace]
        all_active_values.extend(active_values)
        total_steps += steps
        total_seconds += seconds
        total_energy += energy
        total_sampled_energy += sampled_energy
        production_weighted += float(interval["production_tpot"]["mean_ms"]) * steps
        kernel_weighted += float(interval["kernel"]["mean_ms"]) * steps
        interval_results.append({
            "index": int(interval["index"]),
            "steps": steps,
            "seconds": seconds,
            "gross_joules": energy,
            "energy_source": energy_source,
            "sampled_joules": sampled_energy,
            "gross_joules_per_token": energy / steps,
            "idle_subtracted_joules_per_token": (
                energy - idle_watts * seconds
            ) / steps,
            "mean_watts": energy / seconds,
        })
    if total_steps <= 0:
        raise ValueError("timing file has no completed intervals")
    gross_joules_per_token = total_energy / total_steps
    idle_subtracted_joules_per_token = (
        total_energy - idle_watts * total_seconds
    ) / total_steps
    return {
        "device": name,
        "timing_path": str(timing_path.resolve()),
        "power_path": str(power_path.resolve()),
        "power_identity": identity,
        "execution": timing.get("execution"),
        "prompt_sha256": timing.get("prompt_sha256"),
        "teacher_sha256": timing.get("teacher_sha256"),
        "steps": total_steps,
        "seconds": total_seconds,
        "production_tpot_ms": production_weighted / total_steps,
        "device_time_ms": kernel_weighted / total_steps,
        "tokens_per_second": total_steps / total_seconds,
        "idle_mean_watts": idle_watts,
        "idle_energy_source": idle_source,
        "active_mean_watts": total_energy / total_seconds,
        "sampled_active_mean_watts": total_sampled_energy / total_seconds,
        "active_median_watts": statistics.median(all_active_values),
        "active_p95_watts": p95(all_active_values),
        "gross_joules_per_token": gross_joules_per_token,
        "idle_subtracted_joules_per_token": idle_subtracted_joules_per_token,
        "gross_tokens_per_joule": 1.0 / gross_joules_per_token,
        "idle_subtracted_tokens_per_joule": positive_ratio(1.0, idle_subtracted_joules_per_token),
        "tokens_per_second_per_watt": total_steps / total_energy,
        "intervals": interval_results,
    }


def main() -> None:
    args = parse_args()
    global BOUNDARY_TOLERANCE_SECONDS
    if bool(args.fpga_timing) != bool(args.fpga_power):
        raise ValueError("provide both --fpga-timing and --fpga-power, or neither")
    if not math.isfinite(args.boundary_tolerance_seconds) or args.boundary_tolerance_seconds < 0:
        raise ValueError("boundary tolerance must be finite and nonnegative")
    BOUNDARY_TOLERANCE_SECONDS = args.boundary_tolerance_seconds
    BOUNDARY_CLAMPS.clear()
    fpga = (summarize_device("U55C all-BF16 accelerator", args.fpga_timing, args.fpga_power)
            if args.fpga_timing else None)
    gpu = summarize_device(None, args.gpu_timing, args.gpu_power)
    boundary = "decode-only board energy; host/server and prefill excluded"
    comparison = None
    if fpga is not None:
        for key in ("prompt_sha256", "teacher_sha256"):
            if fpga.get(key) and gpu.get(key) and fpga[key] != gpu[key]:
                raise ValueError(f"FPGA/GPU workload mismatch: {key}")
        comparison = {
            "headline": f"{fpga['device']} versus {gpu['device']}",
            "energy_boundary": boundary,
            "fpga_over_gpu_production_tpot": fpga["production_tpot_ms"] / gpu["production_tpot_ms"],
            "fpga_over_gpu_device_time": fpga["device_time_ms"] / gpu["device_time_ms"],
            "fpga_over_gpu_gross_energy_efficiency": (
                fpga["gross_tokens_per_joule"] / gpu["gross_tokens_per_joule"]),
            "fpga_over_gpu_idle_subtracted_energy_efficiency": positive_ratio(
                fpga["idle_subtracted_tokens_per_joule"], gpu["idle_subtracted_tokens_per_joule"]),
        }
    result = {
        "schema": "gdn-power-evaluation-v1",
        "energy_boundary": boundary,
        "fpga": fpga,
        "gpu": gpu,
        "boundary_tolerance_seconds": BOUNDARY_TOLERANCE_SECONDS,
        "boundary_clamps": BOUNDARY_CLAMPS,
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
        if fpga is not None:
            writer.writerow({key: fpga[key] for key in fields})
        writer.writerow({key: gpu[key] for key in fields})


if __name__ == "__main__":
    main()

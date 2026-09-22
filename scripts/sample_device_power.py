#!/usr/bin/env python3
"""Write timestamped NVIDIA GPU or U55C board-power samples as JSON Lines."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from gpu_power import NvmlDevice, cuda_uuid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("gpu", "fpga"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--period", type=float)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--gpu-index", type=int, help="explicit physical NVML index")
    selector.add_argument("--gpu-uuid")
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--fpga-bdf")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(fraction * len(ordered) + 0.999) - 1))
    return ordered[index]


def find_electrical_power(value: Any, path: str = "") -> tuple[float, str] | None:
    """Find the explicit board-power field in version-varying xbutil JSON."""

    if isinstance(value, dict):
        priority = (
            "power_consumption_watts",
            "power_consumption",
            "board_power_watts",
            "power_watts",
            "power",
        )
        lowered = {str(key).lower(): key for key in value}
        for wanted in priority:
            if wanted in lowered:
                key = lowered[wanted]
                raw = value[key]
                if isinstance(raw, (int, float)):
                    return float(raw), f"{path}/{key}"
                if isinstance(raw, str):
                    match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
                    if match:
                        return float(match.group()), f"{path}/{key}"
        for key, child in value.items():
            found = find_electrical_power(child, f"{path}/{key}")
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = find_electrical_power(child, f"{path}/{index}")
            if found is not None:
                return found
    return None


def make_gpu_reader(
    index_arg: int | None, gpu_uuid: str | None = None, device: str = "cuda:0"
) -> tuple[Callable[[], float], dict[str, Any]]:
    if index_arg is None and gpu_uuid is None:
        import torch

        gpu_uuid = cuda_uuid(torch, torch.device(device))
    gpu = NvmlDevice(gpu_uuid=gpu_uuid, index=index_arg)
    return gpu.watts, gpu.identity()


def xbutil_power(bdf: str) -> tuple[float, str]:
    command = [
        "xbutil", "examine", "--device", bdf,
        "--report", "electrical", "--format", "JSON",
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=10
    )
    report = json.loads(completed.stdout)
    found = find_electrical_power(report)
    if found is None:
        raise RuntimeError("xbutil electrical JSON has no board-power field")
    return found


def discover_hwmon_power(bdf: str) -> Path:
    roots = list(Path(f"/sys/bus/pci/devices/{bdf}/hwmon").glob("hwmon*"))
    candidates = [path for root in roots for path in root.glob("power*_input")]
    if not candidates:
        raise RuntimeError(f"no power input under PCI device {bdf}")
    # Prefer a channel explicitly labelled total/board/card/xmc when present.
    for candidate in candidates:
        label = candidate.with_name(candidate.name.replace("_input", "_label"))
        if label.exists() and any(
            word in label.read_text().lower()
            for word in ("total", "board", "card", "xmc")
        ):
            return candidate
    return candidates[0]


def make_fpga_reader(bdf: str | None) -> tuple[Callable[[], float], dict[str, Any]]:
    if not bdf:
        raise ValueError("--fpga-bdf is required for FPGA power sampling")
    try:
        initial, field = xbutil_power(bdf)
    except Exception as xbutil_error:
        sensor = discover_hwmon_power(bdf)

        def read_hwmon() -> float:
            # Linux power*_input is expressed in microwatts.
            return float(sensor.read_text().strip()) / 1_000_000.0

        return read_hwmon, {
            "source": "sysfs_hwmon",
            "fpga_bdf": bdf,
            "sensor": str(sensor),
            "xbutil_error": str(xbutil_error),
        }

    first = [initial]

    def read_xbutil() -> float:
        if first:
            return first.pop()
        return xbutil_power(bdf)[0]

    return read_xbutil, {
        "source": "xbutil_electrical_json",
        "fpga_bdf": bdf,
        "json_field": field,
    }


def main() -> None:
    args = parse_args()
    if args.device == "gpu":
        reader, identity = make_gpu_reader(args.gpu_index, args.gpu_uuid, args.cuda_device)
        period = args.period if args.period is not None else 0.1
    else:
        reader, identity = make_fpga_reader(args.fpga_bdf)
        period = args.period if args.period is not None else 1.0
    if not math.isfinite(period) or period <= 0:
        raise ValueError("sample period must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.ready_file:
        args.ready_file.unlink(missing_ok=True)
    args.stop_file.unlink(missing_ok=True)
    values: list[float] = []
    with args.output.open("w", encoding="utf-8") as output:
        output.write(json.dumps({
            "kind": "identity",
            "device": args.device,
            "period_seconds": period,
            **identity,
        }, sort_keys=True) + "\n")
        output.flush()
        while not args.stop_file.exists():
            started = time.monotonic()
            try:
                watts = reader()
                if not math.isfinite(watts) or watts <= 0:
                    raise RuntimeError(f"non-positive power reading {watts}")
                values.append(watts)
                record = {
                    "kind": "sample",
                    "unix_seconds": time.time(),
                    "monotonic_seconds": time.monotonic(),
                    "watts": watts,
                }
            except Exception as error:
                record = {
                    "kind": "error",
                    "unix_seconds": time.time(),
                    "error": str(error),
                }
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            if args.ready_file and record["kind"] == "sample" and not args.ready_file.exists():
                temporary = args.ready_file.with_suffix(args.ready_file.suffix + ".tmp")
                temporary.write_text(json.dumps(identity) + "\n")
                os.replace(temporary, args.ready_file)
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
        # One closing sample after the stop request, so the trace always
        # extends past the benchmark's last timing boundary. Without it the
        # final boundary can fall inside the last sleep (job 3389: the last
        # sample landed 0.19 s before interval 2 ended and aggregation failed).
        try:
            watts = reader()
            if not math.isfinite(watts) or watts <= 0:
                raise RuntimeError(f"non-positive power reading {watts}")
            values.append(watts)
            record = {
                "kind": "sample",
                "unix_seconds": time.time(),
                "monotonic_seconds": time.monotonic(),
                "watts": watts,
                "closing": True,
            }
        except Exception as error:
            record = {"kind": "error", "unix_seconds": time.time(), "error": str(error)}
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        output.write(json.dumps({
            "kind": "summary",
            "samples": len(values),
            "mean_watts": sum(values) / len(values) if values else None,
            "median_watts": percentile(values, 0.5) if values else None,
            "p95_watts": percentile(values, 0.95) if values else None,
        }, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

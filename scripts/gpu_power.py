"""CUDA/NVML identity and managed board-power telemetry for evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def cuda_uuid(torch_module: Any, device: Any) -> str:
    """Use CUDA's actual device UUID, never a guessed physical ordinal."""
    raw = str(torch_module.cuda.get_device_properties(device).uuid)
    if raw.startswith("MIG-"):
        raise ValueError("board energy requires a whole GPU, not a MIG instance")
    value = raw.removeprefix("GPU-")
    return "GPU-" + str(uuid.UUID(value))


class NvmlDevice:
    def __init__(self, gpu_uuid: str | None = None, index: int | None = None):
        import pynvml

        if (gpu_uuid is None) == (index is None):
            raise ValueError("specify exactly one GPU UUID or physical NVML index")
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = (pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
                       if gpu_uuid is not None
                       else pynvml.nvmlDeviceGetHandleByIndex(index))
        if pynvml.nvmlDeviceIsMigDeviceHandle(self.handle):
            raise ValueError("board energy cannot be attributed to a MIG instance")

    def optional(self, function: str, *args: Any) -> Any:
        try:
            return getattr(self.nvml, function)(*args)
        except (self.nvml.NVMLError, AttributeError):
            return None

    def identity(self) -> dict[str, Any]:
        def text(value: Any) -> Any:
            return value.decode() if isinstance(value, bytes) else value

        n, h = self.nvml, self.handle
        limit = self.optional("nvmlDeviceGetPowerManagementLimit", h)
        return {
            "source": "nvml",
            "gpu_uuid": text(n.nvmlDeviceGetUUID(h)),
            "gpu_name": text(n.nvmlDeviceGetName(h)),
            "gpu_index": n.nvmlDeviceGetIndex(h),
            "driver_version": text(n.nvmlSystemGetDriverVersion()),
            "power_limit_watts": limit / 1000.0 if limit is not None else None,
            "sm_clock_mhz": self.optional("nvmlDeviceGetClockInfo", h, n.NVML_CLOCK_SM),
            "memory_clock_mhz": self.optional("nvmlDeviceGetClockInfo", h, n.NVML_CLOCK_MEM),
            "energy_counter_supported": self.energy_mj() is not None,
            "energy_scope": "whole GPU board; requires exclusive use",
        }

    def watts(self) -> float:
        return self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def energy_mj(self) -> int | None:
        try:
            return int(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))
        except (self.nvml.NVMLError_NotSupported,
                self.nvml.NVMLError_FunctionNotFound):
            return None


class PowerSampler:
    """Start sampling before timing and obtain a closing sample on shutdown."""

    def __init__(self, path: Path, gpu_uuid: str, period: float):
        self.path = path
        self.stop = path.with_suffix(path.suffix + ".stop")
        self.ready = path.with_suffix(path.suffix + ".ready")
        self.gpu_uuid = gpu_uuid
        self.period = period
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> PowerSampler:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for path in (self.path, self.ready, self.stop):
            if path.exists():
                raise FileExistsError(f"use a fresh power-output path: {path}")
        self.process = subprocess.Popen([
            sys.executable, str(Path(__file__).with_name("sample_device_power.py")),
            "--device", "gpu", "--gpu-uuid", self.gpu_uuid,
            "--period", str(self.period), "--output", str(self.path),
            "--stop-file", str(self.stop), "--ready-file", str(self.ready),
        ])
        try:
            deadline = time.monotonic() + 30
            while not self.ready.exists():
                if self.process.poll() is not None:
                    raise RuntimeError("power sampler failed before its first sample")
                if time.monotonic() >= deadline:
                    raise TimeoutError("power sampler did not produce its first sample")
                time.sleep(0.05)
            if json.loads(self.ready.read_text())["gpu_uuid"] != self.gpu_uuid:
                raise RuntimeError("power sampler selected a different GPU")
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        if self.process is None:
            return
        self.stop.touch()
        try:
            code = self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError("power sampler did not stop")
        if code:
            raise RuntimeError(f"power sampler failed with exit code {code}")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

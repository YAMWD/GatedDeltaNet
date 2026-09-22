#!/usr/bin/env python3
"""CPU tests for energy accounting, GPU identity and telemetry lifecycle."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import aggregate_power_eval as aggregate
import sample_device_power as sampler
from gpu_power import NvmlDevice, cuda_uuid


UUID = "GPU-12345678-1234-1234-1234-123456789abc"


class EnergyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.timing_path = self.root / "timing.json"
        self.power_path = self.root / "power.jsonl"
        aggregate.BOUNDARY_TOLERANCE_SECONDS = 0.0
        aggregate.BOUNDARY_CLAMPS.clear()
        self.timing = {
            "device": "A100",  # legacy label must not override actual identity
            "gpu_uuid": UUID,
            "execution": "bfloat16 eager FLA/PyTorch",
            "idle_start_unix": 0, "warmup_start_unix": 2,
            "intervals": [{"index": 0, "start_unix": 3, "end_unix": 7,
                           "steps": 40, "production_tpot": {"mean_ms": 90},
                           "kernel": {"mean_ms": 80}}],
        }
        # Uneven samples along the same straight line: arithmetic means are wrong.
        self.samples = [(-1, 10), (0, 10), (2, 10), (3, 20), (3.1, 20.5),
                        (3.2, 21), (7, 40), (8, 40)]
        self.write()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self) -> None:
        self.timing_path.write_text(json.dumps(self.timing))
        records = [{"kind": "identity", "device": "gpu", "gpu_uuid": UUID,
                    "gpu_name": "NVIDIA H100 80GB HBM3"}]
        records += [{"kind": "sample", "unix_seconds": t, "watts": w}
                    for t, w in self.samples]
        self.power_path.write_text("".join(json.dumps(row) + "\n" for row in records))

    def summary(self):
        self.write()
        return aggregate.summarize_device(None, self.timing_path, self.power_path)

    def test_integrated_energy_and_time_weighted_power(self):
        result = self.summary()
        self.assertAlmostEqual(result["gross_joules_per_token"], 3)
        self.assertAlmostEqual(result["active_mean_watts"], 30)
        self.assertAlmostEqual(result["idle_mean_watts"], 10)
        self.assertAlmostEqual(result["idle_subtracted_joules_per_token"], 2)
        self.assertAlmostEqual(result["gross_tokens_per_joule"], 1 / 3)
        self.assertAlmostEqual(result["tokens_per_second_per_watt"], 1 / 3)
        self.assertIn("H100", result["device"])
        self.assertNotIn("A100", result["device"])

    def test_counter_energy_units_and_source(self):
        self.timing.update(idle_energy_start_mj=1000, idle_energy_end_mj=25000)
        self.timing["intervals"][0].update(energy_start_mj=50000, energy_end_mj=210000)
        result = self.summary()
        self.assertEqual(result["gross_joules_per_token"], 4)
        self.assertEqual(result["idle_mean_watts"], 12)
        self.assertEqual(result["active_mean_watts"], 40)
        self.assertEqual(result["intervals"][0]["sampled_joules"], 120)
        self.assertEqual(result["intervals"][0]["energy_source"], "nvml_energy_counter")

    def test_counter_reset_rejected(self):
        self.timing["intervals"][0].update(energy_start_mj=50000, energy_end_mj=100)
        with self.assertRaisesRegex(ValueError, "reset"):
            self.summary()

    def test_stale_counter_falls_back_to_sampled_power(self):
        self.timing["intervals"][0].update(energy_start_mj=100, energy_end_mj=100)
        result = self.summary()
        self.assertEqual(result["intervals"][0]["energy_source"], "sampled_power")

    def test_wrong_gpu_is_rejected(self):
        self.timing["gpu_uuid"] = "GPU-other"
        with self.assertRaisesRegex(ValueError, "different GPU"):
            self.summary()

    def test_missing_closing_sample_is_rejected(self):
        self.samples = self.samples[:-2]
        with self.assertRaisesRegex(RuntimeError, "timing boundary"):
            self.summary()

    def test_explicit_boundary_tolerance_is_recorded(self):
        self.samples = [(-1, 10), (0, 10), (2, 10), (3, 20), (6.9, 39.5)]
        aggregate.BOUNDARY_TOLERANCE_SECONDS = 0.2
        self.summary()
        self.assertEqual(len(aggregate.BOUNDARY_CLAMPS), 1)

    def test_zero_net_energy_does_not_divide_by_zero(self):
        self.samples = [(t, 10) for t, _ in self.samples]
        result = self.summary()
        self.assertEqual(result["idle_subtracted_joules_per_token"], 0)
        self.assertIsNone(result["idle_subtracted_tokens_per_joule"])

    def test_empty_interval_rejected(self):
        self.timing["intervals"][0]["steps"] = 0
        with self.assertRaisesRegex(ValueError, "no decoded tokens"):
            self.summary()

    def test_nonfinite_power_rejected(self):
        self.samples[0] = (-1, float("nan"))
        with self.assertRaisesRegex(ValueError, "invalid power"):
            self.summary()

    def test_legacy_no_counter_or_uuid(self):
        del self.timing["gpu_uuid"]
        self.assertEqual(self.summary()["intervals"][0]["energy_source"], "sampled_power")

    def cli(self, *extra):
        return subprocess.run([
            sys.executable, str(Path(aggregate.__file__)),
            "--gpu-timing", str(self.timing_path), "--gpu-power", str(self.power_path),
            "--output-json", str(self.root / "summary.json"),
            "--output-csv", str(self.root / "summary.csv"), *extra,
        ], capture_output=True, text=True)

    def test_gpu_only_cli_json_and_csv(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads((self.root / "summary.json").read_text())
        self.assertIsNone(output["fpga"])
        self.assertIsNone(output["comparison"])
        with (self.root / "summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertIn("H100", rows[0]["device"])

    def test_paired_cli_remains_supported(self):
        result = self.cli("--fpga-timing", str(self.timing_path),
                          "--fpga-power", str(self.power_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(output["comparison"]["fpga_over_gpu_gross_energy_efficiency"], 1)
        self.assertIn("H100", output["comparison"]["headline"])

    def test_partial_fpga_arguments_rejected(self):
        self.assertNotEqual(self.cli("--fpga-timing", str(self.timing_path)).returncode, 0)


class DeviceTest(unittest.TestCase):
    def test_cuda_uuid_not_visible_device_ordinal(self):
        seen = []
        def properties(device):
            seen.append(device)
            return SimpleNamespace(uuid=UUID.removeprefix("GPU-"))
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=properties))
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,0"}):
            self.assertEqual(cuda_uuid(torch, "cuda:1"), UUID)
        self.assertEqual(seen, ["cuda:1"])

    def test_mig_cuda_uuid_rejected(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(
            get_device_properties=lambda device: SimpleNamespace(uuid="MIG-abc")))
        with self.assertRaisesRegex(ValueError, "MIG"):
            cuda_uuid(torch, "cuda:0")

    def test_nvml_binds_uuid_and_rejects_mig_handles(self):
        seen = []
        n = SimpleNamespace(nvmlInit=lambda: None,
                            nvmlDeviceGetHandleByUUID=lambda uuid: seen.append(uuid) or 17,
                            nvmlDeviceIsMigDeviceHandle=lambda handle: False)
        with patch.dict(sys.modules, {"pynvml": n}):
            self.assertEqual(NvmlDevice(gpu_uuid=UUID).handle, 17)
            self.assertEqual(seen, [UUID])
            n.nvmlDeviceIsMigDeviceHandle = lambda handle: True
            with self.assertRaisesRegex(ValueError, "MIG"):
                NvmlDevice(gpu_uuid=UUID)

    def test_sampler_ready_and_closing_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stop, ready, output = root / "stop", root / "ready", root / "power.jsonl"
            def read():
                stop.touch()
                return 123.0
            args = ["sample", "--device", "gpu", "--gpu-uuid", UUID,
                    "--output", str(output), "--stop-file", str(stop),
                    "--ready-file", str(ready), "--period", "0.0001"]
            with patch.object(sys, "argv", args), patch.object(
                sampler, "make_gpu_reader", return_value=(read, {"gpu_uuid": UUID})
            ):
                sampler.main()
            records = [json.loads(line) for line in output.read_text().splitlines()]
            samples = [row for row in records if row["kind"] == "sample"]
            self.assertEqual(len(samples), 2)
            self.assertTrue(samples[-1]["closing"])
            self.assertEqual(json.loads(ready.read_text())["gpu_uuid"], UUID)


if __name__ == "__main__":
    unittest.main()

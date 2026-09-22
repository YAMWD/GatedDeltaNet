#!/usr/bin/env python3
"""Unit tests for the GPU/FPGA evaluation handoff protocol."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from fpga_eval_client import (
    FLAG_BLANK_STATE,
    FpgaFileQueue,
    MODE_GENERATE,
    MODE_SCORE,
    REQUEST_VERSION,
    encode_generate_request,
    encode_score_request,
    parse_request_for_test,
    stable_request_id,
    write_gdn_state,
)


class FakeProducer:
    def export(self, context_tokens, state_path, metadata_path, metadata):
        blank = len(context_tokens) == 1
        if not blank:
            state_path.write_bytes(b"fake-state")
        metadata_path.write_text(json.dumps(metadata))
        return blank


class ProtocolTest(unittest.TestCase):
    def test_generate_round_trip(self) -> None:
        payload = encode_generate_request(
            17, 128, [[2], [13, 10]], blank_state=True
        )
        decoded = parse_request_for_test(payload)
        self.assertEqual(decoded["version"], REQUEST_VERSION)
        self.assertEqual(decoded["mode"], MODE_GENERATE)
        self.assertEqual(decoded["flags"], FLAG_BLANK_STATE)
        self.assertEqual(decoded["boundary_token"], 17)
        self.assertEqual(decoded["max_new_tokens"], 128)
        self.assertEqual(decoded["stop_sequences"], [[2], [13, 10]])

    def test_score_round_trip(self) -> None:
        payload = encode_score_request(
            -1, [[4], [5, 6, 7]], blank_state=False
        )
        decoded = parse_request_for_test(payload)
        self.assertEqual(decoded["mode"], MODE_SCORE)
        self.assertEqual(decoded["boundary_token"], -1)
        self.assertEqual(decoded["choices"], [[4], [5, 6, 7]])

    def test_request_id_is_stable(self) -> None:
        value = {"tokens": [1, 2, 3], "mode": "score"}
        self.assertEqual(
            stable_request_id("table 3", value),
            stable_request_id("table 3", dict(reversed(list(value.items())))),
        )

    def test_state_layout(self) -> None:
        recurrent = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        conv = tuple(
            torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)
            for _ in range(3)
        )
        past = [
            {
                "recurrent_state": recurrent.to(torch.bfloat16).float(),
                "conv_state": tuple(value.to(torch.bfloat16) for value in conv),
            },
            {
                "recurrent_state": recurrent.to(torch.bfloat16).float(),
                "conv_state": tuple(value.to(torch.bfloat16) for value in conv),
            },
        ]
        model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.gdnstate"
            digest = write_gdn_state(path, model, past, [11, 12], 13)
            blob = path.read_bytes()
        self.assertEqual(len(digest), 64)
        self.assertEqual(blob[:8], b"GDNSTAT1")
        values = struct.unpack_from("<9I", blob, 8)
        self.assertEqual(values, (1, 2, 2, 3, 4, 8, 3, 2, 13))
        expected = 8 + 36 + 2 * 4 + 2 * 2 * 3 * 4 * 4 + 2 * 3 * 2 * 8 * 4
        self.assertEqual(len(blob), expected)

    def test_queue_resume_and_blank_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = FpgaFileQueue(root / "queue", root / "results", FakeProducer())
            handle = queue.submit_generate(
                prefix="pilot",
                context_tokens=[42],
                max_new_tokens=2,
                stop_sequences=[[2]],
                metadata={"task": "unit"},
            )
            request = parse_request_for_test(
                (root / "queue" / f"{handle.request_id}.req.ready").read_bytes()
            )
            self.assertEqual(request["flags"], FLAG_BLANK_STATE)
            handle.result_path.write_text(
                json.dumps({"id": handle.request_id, "tokens": [7, 8]})
            )
            self.assertEqual(queue.wait(handle)["tokens"], [7, 8])
            queue.finish()
            self.assertTrue((root / "queue" / "producer.done").exists())


if __name__ == "__main__":
    unittest.main()


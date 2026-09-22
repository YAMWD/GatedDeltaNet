#!/usr/bin/env python3
"""GPU-prefill/FPGA-decode handoff and file-queue protocol.

The FPGA host is a persistent process.  This module prepares the fixed-size
Gated DeltaNet state on the GPU, publishes at most a small number of requests
through a node-local directory, and waits for atomic JSON results.  Large state
and logit artifacts deliberately stay outside Git.

Protocol ``GDNEVQ1`` is little-endian.  Every request starts with an eight-byte
magic and six uint32 fields: version, mode, flags, boundary token, primary
count, and stop-sequence count.  Generation requests append length-prefixed
stop-token sequences.  Scoring requests append length-prefixed candidate
continuations.  A sibling ``<id>.gdnstate`` contains the existing ``GDNSTAT1``
state format unless the blank-state flag is set.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from export_gdn_state import (
    STATE_MAGIC,
    STATE_VERSION,
    assert_bf16_exact_conv_cache,
    assert_bf16_exact_recurrent_cache,
    round_conv_cache_bf16,
    round_recurrent_cache_bf16,
)


REQUEST_MAGIC = b"GDNEVQ1\0"
REQUEST_VERSION = 1
MODE_GENERATE = 1
MODE_SCORE = 2
FLAG_BLANK_STATE = 1 << 0
CONTRACT = "native-bf16-product/all-bf16-boundaries/fp32-logits-v1"


def _atomic_write(path: Path, payload: bytes) -> None:
    """Publish one file only after its complete payload reaches the filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def token_sha256(tokens: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack("<i", int(token)))
    return digest.hexdigest()


def stable_request_id(prefix: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:32]
    safe_prefix = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in prefix
    ).strip("-")
    return f"{safe_prefix[:48] or 'request'}-{digest}"


def _pack_i32(values: Iterable[int]) -> bytes:
    values = [int(value) for value in values]
    if not values:
        return b""
    return struct.pack(f"<{len(values)}i", *values)


def encode_generate_request(
    boundary_token: int,
    max_new_tokens: int,
    stop_sequences: Sequence[Sequence[int]],
    *,
    blank_state: bool,
) -> bytes:
    if max_new_tokens <= 0 or max_new_tokens > 4096:
        raise ValueError("max_new_tokens must be in [1, 4096]")
    if len(stop_sequences) > 64:
        raise ValueError("at most 64 stop sequences are supported")
    payload = bytearray(REQUEST_MAGIC)
    payload += struct.pack(
        "<6I",
        REQUEST_VERSION,
        MODE_GENERATE,
        FLAG_BLANK_STATE if blank_state else 0,
        int(boundary_token) & 0xFFFFFFFF,
        int(max_new_tokens),
        len(stop_sequences),
    )
    for sequence in stop_sequences:
        if not sequence or len(sequence) > 32:
            raise ValueError("stop sequences must contain 1..32 tokens")
        payload += struct.pack("<I", len(sequence))
        payload += _pack_i32(sequence)
    return bytes(payload)


def encode_score_request(
    boundary_token: int,
    choices: Sequence[Sequence[int]],
    *,
    blank_state: bool,
) -> bytes:
    if not choices or len(choices) > 64:
        raise ValueError("scoring requests require 1..64 choices")
    payload = bytearray(REQUEST_MAGIC)
    payload += struct.pack(
        "<6I",
        REQUEST_VERSION,
        MODE_SCORE,
        FLAG_BLANK_STATE if blank_state else 0,
        int(boundary_token) & 0xFFFFFFFF,
        len(choices),
        0,
    )
    for choice in choices:
        if not choice or len(choice) > 8192:
            raise ValueError("choices must contain 1..8192 tokens")
        payload += struct.pack("<I", len(choice))
        payload += _pack_i32(choice)
    return bytes(payload)


def _set_attention_mode(model: torch.nn.Module, mode: str) -> None:
    if hasattr(model, "config") and hasattr(model.config, "attn_mode"):
        model.config.attn_mode = mode
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Gated DeltaNet model layers were not found")
    for layer in layers:
        attention = getattr(layer, "attn", None)
        if attention is None or not hasattr(attention, "mode"):
            raise RuntimeError("Gated DeltaNet attention mode is unavailable")
        attention.mode = mode


def _state_dimensions(model: torch.nn.Module, past: Any) -> dict[str, int]:
    layers = list(past)
    if not layers:
        raise RuntimeError("prefill produced an empty recurrent cache")
    recurrent = layers[0]["recurrent_state"]
    conv = layers[0]["conv_state"][0]
    if recurrent.ndim != 4 or recurrent.shape[0] != 1:
        raise RuntimeError(f"unexpected recurrent state shape {recurrent.shape}")
    if conv.ndim != 3 or conv.shape[0] != 1:
        raise RuntimeError(f"unexpected convolution state shape {conv.shape}")
    _, heads, key_dim, value_dim = recurrent.shape
    _, hidden, conv_width = conv.shape
    configured_layers = int(
        getattr(model.config, "num_hidden_layers", len(layers))
    )
    if configured_layers != len(layers):
        raise RuntimeError(
            f"cache has {len(layers)} layers, expected {configured_layers}"
        )
    return {
        "num_layers": len(layers),
        "heads": int(heads),
        "key_dim": int(key_dim),
        "value_dim": int(value_dim),
        "hidden": int(hidden),
        "conv_width": int(conv_width),
    }


def write_gdn_state(
    path: Path,
    model: torch.nn.Module,
    past: Any,
    prefix_tokens: Sequence[int],
    boundary_token: int,
) -> str:
    """Write the existing GDNSTAT1 ABI and return its SHA-256 digest."""

    assert_bf16_exact_recurrent_cache(past)
    assert_bf16_exact_conv_cache(past)
    dimensions = _state_dimensions(model, past)
    header_values = (
        STATE_VERSION,
        dimensions["num_layers"],
        dimensions["heads"],
        dimensions["key_dim"],
        dimensions["value_dim"],
        dimensions["hidden"],
        dimensions["conv_width"],
        len(prefix_tokens),
        int(boundary_token) & 0xFFFFFFFF,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()

    def emit(handle: Any, value: bytes) -> None:
        handle.write(value)
        digest.update(value)

    with temporary.open("wb") as handle:
        emit(handle, STATE_MAGIC)
        emit(handle, struct.pack("<9I", *header_values))
        emit(handle, _pack_i32(prefix_tokens))
        for layer in past:
            recurrent = (
                layer["recurrent_state"][0]
                .contiguous()
                .float()
                .cpu()
                .numpy()
                .tobytes()
            )
            emit(handle, recurrent)
        for layer in past:
            for conv in layer["conv_state"]:
                tail = (
                    conv[0, :, 1:]
                    .transpose(0, 1)
                    .contiguous()
                    .float()
                    .cpu()
                    .numpy()
                    .tobytes()
                )
                emit(handle, tail)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


class FastStateHandoffProducer:
    """Run one fused/chunk GPU prompt prefill and serialize its final cache."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        if device.type != "cuda":
            raise ValueError("FPGA handoff prefill requires a CUDA GPU")
        parameter = next(model.parameters())
        if parameter.dtype != torch.bfloat16:
            raise TypeError(
                f"FPGA handoff requires a BF16 model, got {parameter.dtype}"
            )
        dense_count = sum(
            1
            for module in model.modules()
            if getattr(module, "_gdn_native_bf16_product", False)
        )
        if dense_count == 0:
            raise RuntimeError(
                "native BF16-product patch is not installed on the GPU model"
            )
        self.native_dense_count = dense_count

    @torch.inference_mode()
    def export(
        self,
        context_tokens: Sequence[int],
        state_path: Path,
        metadata_path: Path,
        metadata: dict[str, Any],
    ) -> bool:
        """Prepare state for context[:-1]; return whether blank state is used."""

        context = [int(token) for token in context_tokens]
        if not context:
            raise ValueError("FPGA requests require a non-empty context")
        boundary_token = context[-1]
        prefix = context[:-1]
        state_metadata = {
            **metadata,
            "contract": CONTRACT,
            "context_tokens": len(context),
            "prefix_tokens": len(prefix),
            "context_token_sha256": token_sha256(context),
            "prefix_token_sha256": token_sha256(prefix),
            "boundary_token": boundary_token,
            "native_product_dense_modules": self.native_dense_count,
        }
        if not prefix:
            state_metadata["blank_state"] = True
            state_metadata["state_sha256"] = None
            _atomic_write_json(metadata_path, state_metadata)
            return True

        _set_attention_mode(self.model, "chunk")
        input_ids = torch.tensor(
            [prefix], dtype=torch.long, device=self.device
        )
        output = self.model.model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        round_recurrent_cache_bf16(past)
        round_conv_cache_bf16(past)
        torch.cuda.synchronize(self.device)
        state_sha256 = write_gdn_state(
            state_path, self.model, past, prefix, boundary_token
        )
        state_metadata["blank_state"] = False
        state_metadata["state_sha256"] = state_sha256
        _atomic_write_json(metadata_path, state_metadata)
        del output, past, input_ids
        return False


@dataclass(frozen=True)
class RequestHandle:
    request_id: str
    result_path: Path
    error_path: Path


class FpgaFileQueue:
    """Client for one persistent FPGA host, with deterministic resume keys."""

    def __init__(
        self,
        queue_dir: Path,
        results_dir: Path,
        producer: FastStateHandoffProducer,
        *,
        result_timeout_seconds: float = 7200.0,
        poll_seconds: float = 0.05,
    ):
        self.queue_dir = queue_dir.resolve()
        self.results_dir = results_dir.resolve()
        self.producer = producer
        self.result_timeout_seconds = float(result_timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.producer_done = self.queue_dir / "producer.done"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.producer_done.unlink(missing_ok=True)

    def _handle(self, request_id: str) -> RequestHandle:
        return RequestHandle(
            request_id=request_id,
            result_path=self.results_dir / f"{request_id}.json",
            error_path=self.results_dir / f"{request_id}.error.json",
        )

    def _submit(
        self,
        *,
        prefix: str,
        identity: dict[str, Any],
        context_tokens: Sequence[int],
        request_payload: bytes,
        metadata: dict[str, Any],
    ) -> RequestHandle:
        request_id = stable_request_id(prefix, identity)
        handle = self._handle(request_id)
        if handle.result_path.exists():
            return handle
        if handle.error_path.exists():
            raise RuntimeError(handle.error_path.read_text())

        state_path = self.queue_dir / f"{request_id}.gdnstate"
        metadata_path = self.results_dir / f"{request_id}.request.json"
        blank_state = self.producer.export(
            context_tokens, state_path, metadata_path, metadata
        )
        # The blank-state bit is encoded by the mode-specific caller after it
        # knows whether a state file exists.  Rebuild the flags word in place.
        mutable = bytearray(request_payload)
        struct.pack_into(
            "<I", mutable, 8 + 2 * 4,
            FLAG_BLANK_STATE if blank_state else 0,
        )
        _atomic_write(
            self.queue_dir / f"{request_id}.req.ready", bytes(mutable)
        )
        return handle

    def submit_generate(
        self,
        *,
        prefix: str,
        context_tokens: Sequence[int],
        max_new_tokens: int,
        stop_sequences: Sequence[Sequence[int]],
        metadata: dict[str, Any],
    ) -> RequestHandle:
        identity = {
            "contract": CONTRACT,
            "mode": "generate",
            "context": [int(token) for token in context_tokens],
            "max_new_tokens": int(max_new_tokens),
            "stop_sequences": [list(map(int, value)) for value in stop_sequences],
            "metadata": metadata,
        }
        return self._submit(
            prefix=prefix,
            identity=identity,
            context_tokens=context_tokens,
            request_payload=encode_generate_request(
                int(context_tokens[-1]), max_new_tokens, stop_sequences,
                blank_state=False,
            ),
            metadata={
                **metadata,
                "mode": "generate",
                "max_new_tokens": int(max_new_tokens),
                "stop_sequence_token_sha256": [
                    token_sha256(value) for value in stop_sequences
                ],
            },
        )

    def submit_score(
        self,
        *,
        prefix: str,
        context_tokens: Sequence[int],
        choices: Sequence[Sequence[int]],
        metadata: dict[str, Any],
    ) -> RequestHandle:
        identity = {
            "contract": CONTRACT,
            "mode": "score",
            "context": [int(token) for token in context_tokens],
            "choices": [list(map(int, value)) for value in choices],
            "metadata": metadata,
        }
        return self._submit(
            prefix=prefix,
            identity=identity,
            context_tokens=context_tokens,
            request_payload=encode_score_request(
                int(context_tokens[-1]), choices, blank_state=False
            ),
            metadata={
                **metadata,
                "mode": "score",
                "choice_token_counts": [len(value) for value in choices],
                "choice_token_sha256": [token_sha256(value) for value in choices],
            },
        )

    def wait(self, handle: RequestHandle) -> dict[str, Any]:
        deadline = time.monotonic() + self.result_timeout_seconds
        while time.monotonic() < deadline:
            if handle.result_path.exists():
                result = json.loads(handle.result_path.read_text())
                if result.get("id") != handle.request_id:
                    raise RuntimeError(
                        f"result identity mismatch for {handle.request_id}"
                    )
                return result
            if handle.error_path.exists():
                error = json.loads(handle.error_path.read_text())
                raise RuntimeError(
                    f"FPGA request {handle.request_id} failed: "
                    f"{error.get('error', error)}"
                )
            time.sleep(self.poll_seconds)
        raise TimeoutError(
            f"timed out waiting for FPGA request {handle.request_id}"
        )

    def finish(self) -> None:
        _atomic_write(self.producer_done, b"done\n")


def parse_request_for_test(payload: bytes) -> dict[str, Any]:
    """Small independent decoder used by protocol unit tests."""

    if payload[:8] != REQUEST_MAGIC:
        raise ValueError("bad request magic")
    version, mode, flags, boundary, primary, auxiliary = struct.unpack_from(
        "<6I", payload, 8
    )
    offset = 8 + 24
    decoded: dict[str, Any] = {
        "version": version,
        "mode": mode,
        "flags": flags,
        "boundary_token": struct.unpack("<i", struct.pack("<I", boundary))[0],
    }
    values = []
    count = auxiliary if mode == MODE_GENERATE else primary
    for _ in range(count):
        (length,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        row = list(struct.unpack_from(f"<{length}i", payload, offset))
        offset += 4 * length
        values.append(row)
    decoded["stop_sequences" if mode == MODE_GENERATE else "choices"] = values
    decoded["max_new_tokens"] = primary if mode == MODE_GENERATE else None
    if offset != len(payload):
        raise ValueError("request has trailing bytes")
    return decoded

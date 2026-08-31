#!/usr/bin/env python3
"""Compare every value in two GDNLOG1 trajectories.

The accelerator and its native C++ model have one fixed arithmetic schedule,
so hardware/native validation is bit-exact elsewhere.  An independent CUDA
all-BF16 model is a different implementation: FP32 reductions are associated
differently before each BF16 boundary.  For that comparison, this tool gates
the complete vectors using scale-aware error, direction, and ranking metrics
instead of applying an FP32 bit-reproduction tolerance to every element.
"""

from __future__ import annotations

import argparse
import heapq
import math
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path


HEADER = struct.Struct("<8sIII")

# Independent-GPU envelope for the evaluated all-BF16 contract.  These are
# deliberately wider than the measured 64-token reference (0.00473 global
# NRMSE, 0.0277 worst-step NRMSE, 0.999827 minimum cosine, 0.0677 worst
# max-error/reference-RMS, 0.214 maximum absolute error, exact top-5 sets).
# Hardware/native remains bit-exact; this envelope never authorizes hardware
# to differ from the native reference.
GPU_GLOBAL_NRMSE_MAX = 0.01
GPU_WORST_STEP_NRMSE_MAX = 0.04
GPU_MIN_STEP_COSINE_MIN = 0.9995
GPU_WORST_MAX_OVER_RMS_MAX = 0.10
GPU_MAX_ABS_MAX = 0.50
GPU_MIN_TOP5_OVERLAP = 5


@dataclass(frozen=True)
class LogitsFile:
    path: Path
    vocab: int
    steps: int
    values: array
    payload: bytes


def load_logits(path: Path) -> LogitsFile:
    blob = path.read_bytes()
    if len(blob) < HEADER.size:
        raise ValueError(f"truncated GDNLOG header: {path}")
    magic, version, vocab, steps = HEADER.unpack_from(blob)
    expected_bytes = HEADER.size + vocab * steps * 4
    if magic[:7] != b"GDNLOG1" or version != 1 or len(blob) != expected_bytes:
        raise ValueError(
            f"invalid GDNLOG: {path} magic={magic!r} version={version} "
            f"vocab={vocab} steps={steps} bytes={len(blob)}/{expected_bytes}"
        )
    values = array("f")
    values.frombytes(blob[HEADER.size:])
    return LogitsFile(path, vocab, steps, values, blob[HEADER.size:])


def top_indices(values: array, base: int, count: int, k: int) -> list[int]:
    # (-index) makes lower vocabulary indices win exact ties, matching the
    # strict-'>' first-wins scan in native and XRT hosts.
    return heapq.nlargest(
        k, range(count), key=lambda index: (values[base + index], -index)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument(
        "--profile", choices=("bf16-gpu", "report-only"),
        default="bf16-gpu",
    )
    args = parser.parse_args()

    candidate = load_logits(args.candidate)
    reference = load_logits(args.reference)
    if candidate.vocab != reference.vocab or candidate.steps > reference.steps:
        raise ValueError(
            "GDNLOG shape mismatch: "
            f"candidate=({candidate.steps},{candidate.vocab}) "
            f"reference=({reference.steps},{reference.vocab})"
        )

    vocab = candidate.vocab
    steps = candidate.steps
    bit_mismatches = 0
    nonfinite_mismatches = 0
    argmax_mismatches = 0
    min_top5_overlap = 5
    max_abs = 0.0
    global_error_sq = 0.0
    global_candidate_sq = 0.0
    global_reference_sq = 0.0
    global_dot = 0.0
    worst_step_nrmse = 0.0
    min_step_cosine = 1.0
    worst_max_over_rms = 0.0
    worst_step = 0

    for step in range(steps):
        base = step * vocab
        step_error_sq = 0.0
        step_candidate_sq = 0.0
        step_reference_sq = 0.0
        step_dot = 0.0
        step_max_abs = 0.0
        for lane in range(vocab):
            index = base + lane
            actual = candidate.values[index]
            expected = reference.values[index]
            byte_offset = index * 4
            exact = (
                candidate.payload[byte_offset:byte_offset + 4]
                == reference.payload[byte_offset:byte_offset + 4]
            )
            if not exact:
                bit_mismatches += 1
            if not math.isfinite(actual) or not math.isfinite(expected):
                if not exact:
                    nonfinite_mismatches += 1
                continue
            error = actual - expected
            abs_error = abs(error)
            step_error_sq += error * error
            step_candidate_sq += actual * actual
            step_reference_sq += expected * expected
            step_dot += actual * expected
            step_max_abs = max(step_max_abs, abs_error)

        if step_reference_sq == 0.0 or step_candidate_sq == 0.0:
            step_nrmse = math.inf
            step_cosine = -1.0
            max_over_rms = math.inf
        else:
            step_nrmse = math.sqrt(step_error_sq / step_reference_sq)
            step_cosine = step_dot / math.sqrt(
                step_candidate_sq * step_reference_sq
            )
            max_over_rms = step_max_abs / math.sqrt(step_reference_sq / vocab)
        if step_nrmse > worst_step_nrmse:
            worst_step_nrmse = step_nrmse
            worst_step = step
        min_step_cosine = min(min_step_cosine, step_cosine)
        worst_max_over_rms = max(worst_max_over_rms, max_over_rms)
        max_abs = max(max_abs, step_max_abs)
        global_error_sq += step_error_sq
        global_candidate_sq += step_candidate_sq
        global_reference_sq += step_reference_sq
        global_dot += step_dot

        candidate_top5 = top_indices(candidate.values, base, vocab, 5)
        reference_top5 = top_indices(reference.values, base, vocab, 5)
        top5_overlap = len(set(candidate_top5) & set(reference_top5))
        min_top5_overlap = min(min_top5_overlap, top5_overlap)
        if candidate_top5[0] != reference_top5[0]:
            argmax_mismatches += 1

    global_nrmse = (
        math.sqrt(global_error_sq / global_reference_sq)
        if global_reference_sq != 0.0 else math.inf
    )
    global_cosine = (
        global_dot / math.sqrt(global_candidate_sq * global_reference_sq)
        if global_candidate_sq != 0.0 and global_reference_sq != 0.0
        else -1.0
    )
    values = vocab * steps
    print(
        f"GPU_ALL_LOGITS steps={steps} values={values} "
        f"bit_mismatch={bit_mismatches} nonfinite_mismatch={nonfinite_mismatches} "
        f"argmax_mismatch={argmax_mismatches} min_top5_overlap={min_top5_overlap} "
        f"global_nrmse={global_nrmse:.9g} "
        f"worst_step_nrmse={worst_step_nrmse:.9g}@{worst_step} "
        f"global_cosine={global_cosine:.12g} "
        f"min_step_cosine={min_step_cosine:.12g} "
        f"max_abs={max_abs:.9g} "
        f"worst_max_over_rms={worst_max_over_rms:.9g}"
    )

    if args.profile == "report-only":
        print("GPU_ALL_LOGITS_REPORT_ONLY")
        return 0

    failures: list[str] = []
    if nonfinite_mismatches:
        failures.append(f"nonfinite_mismatches={nonfinite_mismatches}")
    if argmax_mismatches:
        failures.append(f"argmax_mismatches={argmax_mismatches}")
    if min_top5_overlap < GPU_MIN_TOP5_OVERLAP:
        failures.append(f"min_top5_overlap={min_top5_overlap}")
    if global_nrmse > GPU_GLOBAL_NRMSE_MAX:
        failures.append(f"global_nrmse={global_nrmse:.9g}")
    if worst_step_nrmse > GPU_WORST_STEP_NRMSE_MAX:
        failures.append(f"worst_step_nrmse={worst_step_nrmse:.9g}")
    if min_step_cosine < GPU_MIN_STEP_COSINE_MIN:
        failures.append(f"min_step_cosine={min_step_cosine:.12g}")
    if worst_max_over_rms > GPU_WORST_MAX_OVER_RMS_MAX:
        failures.append(f"worst_max_over_rms={worst_max_over_rms:.9g}")
    if max_abs > GPU_MAX_ABS_MAX:
        failures.append(f"max_abs={max_abs:.9g}")

    if failures:
        print("GPU_ALL_LOGITS_FAIL " + " ".join(failures))
        return 1
    print("GPU_ALL_LOGITS_PASS profile=bf16-gpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

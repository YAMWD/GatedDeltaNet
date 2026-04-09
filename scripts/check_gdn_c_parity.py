#!/usr/bin/env python3
"""Compare Python and C GatedDeltaNet fixture results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def max_abs_diff(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError(f"Length mismatch: {len(xs)} vs {len(ys)}")
    if not xs:
        return 0.0
    return max(abs(float(a) - float(b)) for a, b in zip(xs, ys))


def compare_task(python_path: Path, c_path: Path) -> dict[str, float | int | str]:
    py = json.loads(python_path.read_text())
    c = json.loads(c_path.read_text())
    if py["kind"] != c["kind"]:
        raise ValueError(f"{python_path.stem}: kind mismatch {py['kind']} vs {c['kind']}")
    if py["num_examples"] != c["num_examples"]:
        raise ValueError(
            f"{python_path.stem}: num_examples mismatch {py['num_examples']} vs {c['num_examples']}"
        )

    kind = int(py["kind"])
    task = python_path.stem
    summary: dict[str, float | int | str] = {"task": task, "kind": kind, "num_examples": int(py["num_examples"])}

    if kind == 1:
        summary["acc_diff"] = abs(float(py["metrics"]["acc"]) - float(c["metrics"]["acc"]))
        summary["acc_norm_diff"] = abs(float(py["metrics"]["acc_norm"]) - float(c["metrics"]["acc_norm"]))
        score_diff = 0.0
        score_norm_diff = 0.0
        pred_mismatch = 0
        pred_norm_mismatch = 0
        for py_ex, c_ex in zip(py["examples"], c["examples"]):
            score_diff = max(
                score_diff,
                max_abs_diff([s["logprob"] for s in py_ex["scores"]], c_ex["scores"]),
            )
            score_norm_diff = max(
                score_norm_diff,
                max_abs_diff([s["logprob_norm"] for s in py_ex["scores"]], c_ex["scores_norm"]),
            )
            pred_mismatch += int(int(py_ex["pred"]) != int(c_ex["pred"]))
            pred_norm_mismatch += int(int(py_ex["pred_norm"]) != int(c_ex["pred_norm"]))
        summary["max_score_diff"] = score_diff
        summary["max_score_norm_diff"] = score_norm_diff
        summary["pred_mismatch"] = pred_mismatch
        summary["pred_norm_mismatch"] = pred_norm_mismatch
    elif kind == 2:
        summary["acc_diff"] = abs(float(py["metrics"]["acc"]) - float(c["metrics"]["acc"]))
        summary["perplexity_diff"] = abs(
            float(py["metrics"]["perplexity"]) - float(c["metrics"]["perplexity"])
        )
        logprob_diff = 0.0
        greedy_mismatch = 0
        for py_ex, c_ex in zip(py["examples"], c["examples"]):
            logprob_diff = max(logprob_diff, abs(float(py_ex["logprob"]) - float(c_ex["logprob"])))
            greedy_mismatch += int(bool(py_ex["greedy"]) != bool(c_ex["greedy"]))
        summary["max_logprob_diff"] = logprob_diff
        summary["greedy_mismatch"] = greedy_mismatch
    elif kind == 3:
        summary["word_perplexity_diff"] = abs(
            float(py["metrics"]["word_perplexity"]) - float(c["metrics"]["word_perplexity"])
        )
        summary["byte_perplexity_diff"] = abs(
            float(py["metrics"]["byte_perplexity"]) - float(c["metrics"]["byte_perplexity"])
        )
        summary["bits_per_byte_diff"] = abs(
            float(py["metrics"]["bits_per_byte"]) - float(c["metrics"]["bits_per_byte"])
        )
        logprob_diff = 0.0
        for py_ex, c_ex in zip(py["examples"], c["examples"]):
            logprob_diff = max(logprob_diff, abs(float(py_ex["logprob"]) - float(c_ex["logprob"])))
        summary["max_logprob_diff"] = logprob_diff
    else:
        raise ValueError(f"{task}: unsupported kind {kind}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-dir", type=Path, required=True)
    parser.add_argument("--c-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summaries = []
    for python_path in sorted(args.python_dir.glob("*.json")):
        c_path = args.c_dir / python_path.name
        if not c_path.exists():
            raise FileNotFoundError(f"Missing C result for {python_path.name}")
        summaries.append(compare_task(python_path, c_path))

    payload = {"tasks": summaries}
    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()

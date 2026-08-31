#!/usr/bin/env python3
"""Two analyses of stored evaluation artifacts that the summarizer does not do.

`summarize_gdn_quality_eval.py` renders the headline tables. This script backs
the two supporting claims recorded in
`c_impl/doc/fp32_bf16_quality_evaluation.md`:

  --table5-truncated   Rescore Table 5 with each answer cut at its first line.
                       Six QA tasks apply no answer truncation, so a model that
                       answers correctly and then keeps generating is punished
                       by F1's precision term. This quantifies that effect.

  --pair-table2        Count per-sample answer flips between two arms on the
                       RULER samples. A small aggregate delta can hide either
                       many cancelling changes or one sample moving; only the
                       paired count distinguishes them.

  --pair-table5-preds  Count byte-identical generated answers between two arms.

Usage:
    python scripts/analyze_gdn_eval_artifacts.py --table5-truncated <arm-dir>
    python scripts/analyze_gdn_eval_artifacts.py --pair-table2 <arm-a> <arm-b>
    python scripts/analyze_gdn_eval_artifacts.py --pair-table5-preds <arm-a> <arm-b>
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_gdn_quality_eval import TABLE5_PAPER  # noqa: E402

# The six Table 5 tasks that LongBench does not newline-stop, and which
# therefore carry untruncated continuations into an F1 score.
UNTRUNCATED_QA = ("narrativeqa", "qasper", "multifieldqa_en",
                  "hotpotqa", "2wikimqa", "musique")


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def token_f1(prediction: str, ground_truth: str) -> float:
    """Same scorer as run_gdn_longbench_eval.token_f1, duplicated to keep this
    script importable without pulling in torch."""
    pred, gold = normalize(prediction).split(), normalize(ground_truth).split()
    same = sum((Counter(pred) & Counter(gold)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(gold)
    return 2 * precision * recall / (precision + recall)


def table5_truncated(arm: Path) -> None:
    scores = json.loads((arm / "table5" / "result.json").read_text())["scores_unrounded"]
    print(f"{'task':18s}{'paper':>8s}{'as run':>9s}{'first line':>12s}{'change':>9s}")
    as_run_total = trunc_total = 0.0
    for task, paper in TABLE5_PAPER.items():
        as_run = float(scores[task])
        if task in UNTRUNCATED_QA:
            values = []
            for line in (arm / "table5" / "pred" / f"{task}.jsonl").read_text().splitlines():
                row = json.loads(line)
                first_line = row["pred"].strip().split("\n")[0]
                values.append(max(token_f1(first_line, a) for a in row["answers"]))
            trunc = 100.0 * sum(values) / len(values)
        else:
            trunc = as_run
        as_run_total += as_run
        trunc_total += trunc
        print(f"{task:18s}{paper:8.1f}{as_run:9.2f}{trunc:12.2f}{trunc - as_run:+9.2f}")
    n = len(TABLE5_PAPER)
    paper_avg = sum(TABLE5_PAPER.values()) / n
    print(f"{'AVERAGE':18s}{paper_avg:8.2f}{as_run_total / n:9.2f}"
          f"{trunc_total / n:12.2f}{(trunc_total - as_run_total) / n:+9.2f}")


def _table2_scores(arm: Path, pattern: str) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for path in glob.glob(str(arm / "table2" / "**" / pattern), recursive=True):
        for line in open(path):
            row = json.loads(line)
            doc = row["doc"]
            score = [v for k, v in row.items() if k == str(doc["max_length"])]
            if score:
                out[(doc["max_length"], doc["index"])] = score[0]
    return out


def pair_table2(arm_a: Path, arm_b: Path) -> None:
    print(f"{'task':6s}{'length':>8s}{'a->correct':>12s}{'a->wrong':>10s}{'net':>6s}{'n':>7s}")
    up = down = unchanged = 0
    for label, pattern in (("S1", "samples_gdn_niah_single_1_*.jsonl"),
                           ("S2", "samples_gdn_niah_single_2_*.jsonl"),
                           ("S3", "samples_gdn_niah_single_3_*.jsonl")):
        a, b = _table2_scores(arm_a, pattern), _table2_scores(arm_b, pattern)
        buckets: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
        for key, va in a.items():
            if key not in b or va is None:
                continue
            slot = 0 if b[key] > va else (1 if b[key] < va else 2)
            buckets[key[0]][slot] += 1
        for length in sorted(buckets):
            u, d, s = buckets[length]
            up, down, unchanged = up + u, down + d, unchanged + s
            print(f"{label:6s}{length:8d}{u:12d}{d:10d}{u - d:+6d}{u + d + s:7d}")
    total = up + down + unchanged
    if total == 0:
        raise SystemExit(
            "pair-table2: no overlapping samples found -- were both arms run "
            "with --log_samples, and do the two roots point at the same tasks?")
    print(f"\nwrong->right {up}, right->wrong {down}, unchanged {unchanged}")
    print(f"changed {up + down} of {total} samples = {100 * (up + down) / total:.2f}%")
    if up + down:
        from math import comb
        k, n = min(up, down), up + down
        p = sum(comb(n, i) for i in range(k + 1)) / 2**n
        print(f"P(split this lopsided | symmetric noise) = {p:.4f}")


def pair_table5_preds(arm_a: Path, arm_b: Path) -> None:
    total = identical = 0
    worst = (1.0, "")
    for task in TABLE5_PAPER:
        a = [json.loads(l)["pred"] for l in (arm_a / "table5" / "pred" / f"{task}.jsonl").read_text().splitlines()]
        b = [json.loads(l)["pred"] for l in (arm_b / "table5" / "pred" / f"{task}.jsonl").read_text().splitlines()]
        if len(a) != len(b):
            print(f"{task}: LENGTH MISMATCH {len(a)} vs {len(b)}")
            continue
        same = sum(1 for x, y in zip(a, b) if x == y)
        total += len(a)
        identical += same
        if same / len(a) < worst[0]:
            worst = (same / len(a), task)
    if total == 0:
        raise SystemExit(
            "pair-table5-preds: no comparable predictions -- every task was "
            "missing or length-mismatched between the two arms")
    print(f"{identical}/{total} generated answers byte-identical "
          f"= {100 * identical / total:.1f}%")
    if worst[1]:
        print(f"worst task: {worst[1]} at {100 * worst[0]:.1f}%")
    else:
        print("worst task: none (every compared task fully identical)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table5-truncated", type=Path, metavar="ARM")
    p.add_argument("--pair-table2", type=Path, nargs=2, metavar=("ARM_A", "ARM_B"))
    p.add_argument("--pair-table5-preds", type=Path, nargs=2, metavar=("ARM_A", "ARM_B"))
    args = p.parse_args()

    if args.table5_truncated:
        table5_truncated(args.table5_truncated)
    if args.pair_table2:
        pair_table2(*args.pair_table2)
    if args.pair_table5_preds:
        pair_table5_preds(*args.pair_table5_preds)
    if not any((args.table5_truncated, args.pair_table2, args.pair_table5_preds)):
        p.print_help()


if __name__ == "__main__":
    main()

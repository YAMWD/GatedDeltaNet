#!/usr/bin/env python3
"""Validate full evaluation counts and render paper/FP32/BF16 delta tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TABLE2_PAPER = {
    ("S1", 1024): 98.4,
    ("S1", 2048): 88.4,
    ("S1", 4096): 91.4,
    ("S1", 8192): 91.8,
    ("S2", 1024): 100.0,
    ("S2", 2048): 99.8,
    ("S2", 4096): 92.2,
    ("S2", 8192): 29.6,
    ("S3", 1024): 86.6,
    ("S3", 2048): 84.2,
    ("S3", 4096): 27.6,
}

TABLE3_ROWS = [
    ("WikiText PPL", "wikitext", "word_perplexity,none", 16.42, "ppl"),
    ("LAMBADA PPL", "lambada_openai", "perplexity,none", 12.17, "ppl"),
    ("LAMBADA accuracy", "lambada_openai", "acc,none", 46.65, "acc"),
    ("PIQA", "piqa", "acc,none", 72.25, "acc"),
    ("HellaSwag", "hellaswag", "acc_norm,none", 55.76, "acc"),
    ("WinoGrande", "winogrande", "acc,none", 57.45, "acc"),
    ("ARC-Easy", "arc_easy", "acc,none", 71.21, "acc"),
    ("ARC-Challenge", "arc_challenge", "acc_norm,none", 38.39, "acc"),
    ("SocialIQA", "social_iqa", "acc,none", 40.63, "acc"),
    ("BoolQ", "boolq", "acc,none", 60.24, "acc"),
]

TABLE3_EXPECTED_COUNTS = {
    "wikitext": 62,
    "lambada_openai": 5153,
    "piqa": 1838,
    "hellaswag": 10042,
    "winogrande": 1267,
    "arc_easy": 2376,
    "arc_challenge": 1172,
    "social_iqa": 1954,
    "boolq": 3270,
}

TABLE5_PAPER = {
    "narrativeqa": 14.1,
    "qasper": 14.0,
    "multifieldqa_en": 23.3,
    "hotpotqa": 13.7,
    "2wikimqa": 14.4,
    "musique": 5.8,
    "gov_report": 7.5,
    "qmsum": 16.4,
    "multi_news": 7.9,
    "trec": 30.0,
    "triviaqa": 22.4,
    "samsum": 23.0,
    "lcc": 18.7,
    "repobench-p": 22.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="Root containing fp32/ and bf16/ result directories.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def find_lm_eval_result(path: Path) -> dict[str, Any]:
    candidates = sorted(path.rglob("results_*.json"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one lm-eval result under {path}, got {candidates}")
    return load_json(candidates[0])


def read_table2(precision_root: Path) -> dict[tuple[str, int], float]:
    s12 = find_lm_eval_result(precision_root / "table2" / "s12")
    s3 = find_lm_eval_result(precision_root / "table2" / "s3")
    values: dict[tuple[str, int], float] = {}
    for label, task, result, expected in (
        ("S1", "gdn_niah_single_1", s12, 2000),
        ("S2", "gdn_niah_single_2", s12, 2000),
        ("S3", "gdn_niah_single_3", s3, 1500),
    ):
        actual = int(result["results"][task]["sample_len"])
        effective = int(result["n-samples"][task]["effective"])
        if actual != expected or effective != expected:
            raise ValueError(
                f"{precision_root.name} {label} count {actual}/{effective}, "
                f"expected {expected}"
            )
        for cell_label, length in TABLE2_PAPER:
            if cell_label == label:
                value = float(result["results"][task][f"{length},none"])
                if value < 0:
                    raise ValueError(f"Missing {label} {length} score")
                values[(label, length)] = 100.0 * value
    return values


def read_table3(precision_root: Path) -> dict[str, float]:
    result = find_lm_eval_result(precision_root / "table3" / "results")
    values: dict[str, float] = {}
    accuracies = []
    for name, task, metric, _, kind in TABLE3_ROWS:
        effective = int(result["n-samples"][task]["effective"])
        expected = TABLE3_EXPECTED_COUNTS[task]
        if effective != expected:
            raise ValueError(
                f"{precision_root.name} {task} count {effective}, expected {expected}"
            )
        value = float(result["results"][task][metric])
        if kind == "acc":
            value *= 100.0
            accuracies.append(value)
        values[name] = value
    values["Accuracy average"] = sum(accuracies) / len(accuracies)
    return values


def read_table5(precision_root: Path) -> dict[str, float]:
    result = load_json(precision_root / "table5" / "result.json")
    if set(result["scores_unrounded"]) != set(TABLE5_PAPER):
        raise ValueError("Table 5 task set does not match the paper")
    if sum(int(value) for value in result["sample_counts"].values()) != 3350:
        raise ValueError("Table 5 does not contain all 3,350 samples")
    values = {
        task: float(result["scores_unrounded"][task]) for task in TABLE5_PAPER
    }
    values["Average"] = sum(values.values()) / len(TABLE5_PAPER)
    return values


def signed(value: float) -> str:
    return f"{value:+.2f}"


def render_table2(fp32: dict, bf16: dict) -> None:
    print("### Table 2: RULER S-NIAH")
    print()
    print("| Cell | Paper | FP32 | FP32 - paper | BF16 | BF16 - paper | BF16 - FP32 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for key, paper in TABLE2_PAPER.items():
        fvalue, bvalue = fp32[key], bf16[key]
        label, length = key
        print(
            f"| {label} {length // 1024}K | {paper:.2f} | {fvalue:.2f} | "
            f"{signed(fvalue - paper)} | {bvalue:.2f} | "
            f"{signed(bvalue - paper)} | {signed(bvalue - fvalue)} |"
        )
    paper_avg = sum(TABLE2_PAPER.values()) / len(TABLE2_PAPER)
    fp32_avg = sum(fp32.values()) / len(fp32)
    bf16_avg = sum(bf16.values()) / len(bf16)
    print(
        f"| **Macro average** | **{paper_avg:.2f}** | **{fp32_avg:.2f}** | "
        f"**{signed(fp32_avg - paper_avg)}** | **{bf16_avg:.2f}** | "
        f"**{signed(bf16_avg - paper_avg)}** | "
        f"**{signed(bf16_avg - fp32_avg)}** |"
    )


def render_table3(fp32: dict, bf16: dict) -> None:
    print("### Table 3: short-context results")
    print()
    print("| Metric | Paper | FP32 | FP32 - paper | BF16 | BF16 - paper | BF16 - FP32 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    rows = [(name, paper) for name, _, _, paper, _ in TABLE3_ROWS]
    rows.append(("Accuracy average", 55.32))
    for name, paper in rows:
        fvalue, bvalue = fp32[name], bf16[name]
        emphasis = "**" if name == "Accuracy average" else ""
        print(
            f"| {emphasis}{name}{emphasis} | {emphasis}{paper:.2f}{emphasis} | "
            f"{emphasis}{fvalue:.2f}{emphasis} | "
            f"{emphasis}{signed(fvalue - paper)}{emphasis} | "
            f"{emphasis}{bvalue:.2f}{emphasis} | "
            f"{emphasis}{signed(bvalue - paper)}{emphasis} | "
            f"{emphasis}{signed(bvalue - fvalue)}{emphasis} |"
        )


def render_table5(fp32: dict, bf16: dict) -> None:
    print("### Table 5: LongBench v1")
    print()
    print("| Task | Paper | FP32 | FP32 - paper | BF16 | BF16 - paper | BF16 - FP32 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for task, paper in list(TABLE5_PAPER.items()) + [("Average", 16.6)]:
        fvalue, bvalue = fp32[task], bf16[task]
        emphasis = "**" if task == "Average" else ""
        print(
            f"| {emphasis}{task}{emphasis} | {emphasis}{paper:.2f}{emphasis} | "
            f"{emphasis}{fvalue:.2f}{emphasis} | "
            f"{emphasis}{signed(fvalue - paper)}{emphasis} | "
            f"{emphasis}{bvalue:.2f}{emphasis} | "
            f"{emphasis}{signed(bvalue - paper)}{emphasis} | "
            f"{emphasis}{signed(bvalue - fvalue)}{emphasis} |"
        )


def render_verdicts(
    fp32_t2: dict,
    bf16_t2: dict,
    fp32_t3: dict,
    bf16_t3: dict,
    fp32_t5: dict,
    bf16_t5: dict,
) -> None:
    paper_t2 = sum(TABLE2_PAPER.values()) / len(TABLE2_PAPER)
    fp32_t2_avg = sum(fp32_t2.values()) / len(fp32_t2)
    bf16_t2_avg = sum(bf16_t2.values()) / len(bf16_t2)
    fp32_paper_pass = {
        "Table 2": fp32_t2_avg >= paper_t2 - 3.0,
        "Table 3": (
            fp32_t3["Accuracy average"] >= 55.32 - 1.0
            and fp32_t3["WikiText PPL"] <= 16.42 * 1.05
            and fp32_t3["LAMBADA PPL"] <= 12.17 * 1.05
        ),
        "Table 5": fp32_t5["Average"] >= 16.6 - 1.0,
    }
    bf16_fp32_pass = {
        "Table 2": (
            bf16_t2_avg >= fp32_t2_avg - 2.0
            and min(bf16_t2[key] - fp32_t2[key] for key in fp32_t2) >= -5.0
        ),
        "Table 3": (
            bf16_t3["Accuracy average"] >= fp32_t3["Accuracy average"] - 0.5
            and bf16_t3["WikiText PPL"] <= fp32_t3["WikiText PPL"] * 1.01
            and bf16_t3["LAMBADA PPL"] <= fp32_t3["LAMBADA PPL"] * 1.01
        ),
        "Table 5": (
            bf16_t5["Average"] >= fp32_t5["Average"] - 0.5
            and min(
                bf16_t5[key] - fp32_t5[key] for key in TABLE5_PAPER
            ) >= -2.0
        ),
    }
    print("### Acceptance verdicts")
    print()
    print("| Comparison | Table 2 | Table 3 | Table 5 | Overall |")
    print("|---|---:|---:|---:|---:|")
    for name, verdicts in (
        ("FP32 vs paper", fp32_paper_pass),
        ("BF16 vs FP32", bf16_fp32_pass),
    ):
        labels = ["pass" if verdicts[key] else "fail" for key in ("Table 2", "Table 3", "Table 5")]
        overall = "pass" if all(verdicts.values()) else "fail"
        print(f"| {name} | {labels[0]} | {labels[1]} | {labels[2]} | **{overall}** |")


def main() -> None:
    root = parse_args().root.resolve()
    fp32_root, bf16_root = root / "fp32", root / "bf16"
    fp32_t2, bf16_t2 = read_table2(fp32_root), read_table2(bf16_root)
    fp32_t3, bf16_t3 = read_table3(fp32_root), read_table3(bf16_root)
    fp32_t5, bf16_t5 = read_table5(fp32_root), read_table5(bf16_root)
    render_table2(fp32_t2, bf16_t2)
    print()
    render_table3(fp32_t3, bf16_t3)
    print()
    render_table5(fp32_t5, bf16_t5)
    print()
    render_verdicts(fp32_t2, bf16_t2, fp32_t3, bf16_t3, fp32_t5, bf16_t5)


if __name__ == "__main__":
    main()

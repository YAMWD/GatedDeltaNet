#!/usr/bin/env python3
"""Validate cross-family FPGA pilot completeness before committing harness code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TABLE3_TASKS = {
    "wikitext", "lambada_openai", "piqa", "hellaswag", "winogrande",
    "arc_easy", "arc_challenge", "social_iqa", "boolq",
}
TABLE5_TASKS = {
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
    "samsum", "lcc", "repobench-p",
}


def one_lm_result(root: Path) -> dict:
    candidates = list(root.rglob("results_*.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one lm-eval result under {root}: {candidates}")
    return json.loads(candidates[0].read_text())


def raw_results(path: Path) -> list[dict]:
    errors = list(path.glob("*.error.json"))
    if errors:
        raise RuntimeError(f"FPGA host errors under {path}: {errors}")
    results = [json.loads(item.read_text()) for item in path.glob("*.json")
               if not item.name.endswith(".request.json")]
    if not results:
        raise RuntimeError(f"no FPGA raw results under {path}")
    for result in results:
        if int(result.get("steps", 0)) <= 0:
            raise RuntimeError(f"empty FPGA result: {result}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    logits = json.loads((root / "full_logits_8.json").read_text())
    parity = logits.get("gpu_logits_parity", {})
    if parity.get("argmax_mismatches") != 0 or parity.get("nonfinite_mismatches") != 0:
        raise RuntimeError(f"full-logit gate failed: {parity}")

    s12 = one_lm_result(root / "table2/s12")
    s3 = one_lm_result(root / "table2/s3")
    if set(s12["results"]) != {"gdn_niah_single_1", "gdn_niah_single_2"}:
        raise RuntimeError("Table 2 S1/S2 task mismatch")
    if set(s3["results"]) != {"gdn_niah_single_3"}:
        raise RuntimeError("Table 2 S3 task mismatch")

    table3 = one_lm_result(root / "table3/results")
    if set(table3["results"]) != TABLE3_TASKS:
        raise RuntimeError(
            f"Table 3 task mismatch: {set(table3['results']) ^ TABLE3_TASKS}"
        )
    table5 = json.loads((root / "table5/result.json").read_text())
    if set(table5["sample_counts"]) != TABLE5_TASKS:
        raise RuntimeError("Table 5 task mismatch")
    if any(int(count) != 1 for count in table5["sample_counts"].values()):
        raise RuntimeError("Table 5 pilot did not run exactly one sample/task")

    raw = {
        phase: raw_results(root / phase / "raw")
        for phase in ("table2_s12", "table2_s3", "table3", "table5")
    }
    modes = {result["mode"] for values in raw.values() for result in values}
    if modes != {"generate", "score"}:
        raise RuntimeError(f"pilot did not cover generation and scoring: {modes}")
    snapshots = {
        result.get("snapshot_mode")
        for result in raw["table3"]
        if result["mode"] == "score"
    }
    if not snapshots & {"device_copy", "host_copy"}:
        raise RuntimeError("multiple-choice state snapshot path was not exercised")

    verdict = {
        "schema": "gdn-fpga-eval-pilot-v1",
        "pass": True,
        "full_logit_steps": parity.get("checked_steps"),
        "full_logit_argmax_mismatches": parity.get("argmax_mismatches"),
        "table2_tasks": 3,
        "table3_tasks": len(TABLE3_TASKS),
        "table5_tasks": len(TABLE5_TASKS),
        "raw_requests": {
            phase: len(values) for phase, values in raw.items()
        },
        "state_snapshot_modes": sorted(value for value in snapshots if value),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Verify committed native-BF16 aggregate results against the summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text())


def require_close(label: str, actual: float, expected: float,
                  tolerance: float = 0.011) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{label}: aggregate={actual:.9g}, summary={expected:.9g}"
        )


def verify_table2(summary: dict[str, Any]) -> None:
    s12 = load("table2_s12_results.json")
    s3 = load("table2_s3_results.json")
    values: dict[str, float] = {}
    for label, task, result, expected_count in (
        ("S1", "gdn_niah_single_1", s12, 2000),
        ("S2", "gdn_niah_single_2", s12, 2000),
        ("S3", "gdn_niah_single_3", s3, 1500),
    ):
        assert int(result["results"][task]["sample_len"]) == expected_count
        assert int(result["n-samples"][task]["effective"]) == expected_count
        lengths = (1024, 2048, 4096, 8192) if label != "S3" else (1024, 2048, 4096)
        for length in lengths:
            key = f"{label}_{length // 1024}K"
            value = 100.0 * float(result["results"][task][f"{length},none"])
            values[key] = value
            require_close(
                f"Table 2 {key}", value,
                float(summary["table2_accuracy_percent"][key]["native_product_bf16"]),
            )
    macro = sum(values.values()) / len(values)
    require_close(
        "Table 2 macro", macro,
        float(summary["table2_accuracy_percent"]["macro"]["native_product_bf16"]),
    )


def verify_table3(summary: dict[str, Any]) -> None:
    result = load("table3_results.json")
    expected_counts = summary["sample_counts"]["table3"]
    for task, expected in expected_counts.items():
        assert int(result["n-samples"][task]["effective"]) == int(expected)

    metrics = (
        ("wikitext_word_perplexity", "wikitext", "word_perplexity,none", 1.0, 0.0011),
        ("lambada_perplexity", "lambada_openai", "perplexity,none", 1.0, 0.0011),
        ("lambada_accuracy_percent", "lambada_openai", "acc,none", 100.0, 0.011),
        ("piqa_accuracy_percent", "piqa", "acc,none", 100.0, 0.011),
        ("hellaswag_normalized_accuracy_percent", "hellaswag", "acc_norm,none", 100.0, 0.011),
        ("winogrande_accuracy_percent", "winogrande", "acc,none", 100.0, 0.011),
        ("arc_easy_accuracy_percent", "arc_easy", "acc,none", 100.0, 0.011),
        ("arc_challenge_normalized_accuracy_percent", "arc_challenge", "acc_norm,none", 100.0, 0.011),
        ("social_iqa_accuracy_percent", "social_iqa", "acc,none", 100.0, 0.011),
        ("boolq_accuracy_percent", "boolq", "acc,none", 100.0, 0.011),
    )
    accuracies = []
    for summary_key, task, metric, scale, tolerance in metrics:
        value = scale * float(result["results"][task][metric])
        require_close(
            f"Table 3 {summary_key}", value,
            float(summary["table3"][summary_key]["native_product_bf16"]),
            tolerance,
        )
        if scale == 100.0:
            accuracies.append(value)
    require_close(
        "Table 3 accuracy average", sum(accuracies) / len(accuracies),
        float(summary["table3"]["metric_mapped_accuracy_average_percent"]["native_product_bf16"]),
    )


def verify_table5(summary: dict[str, Any]) -> None:
    result = load("table5_results.json")
    assert sum(int(value) for value in result["sample_counts"].values()) == 3350
    for task, value in result["scores_unrounded"].items():
        require_close(
            f"Table 5 {task}", float(value),
            float(summary["table5_as_run"][task]["native_product_bf16"]),
        )
    require_close(
        "Table 5 macro", float(result["macro_average"]),
        float(summary["table5_as_run"]["macro"]["native_product_bf16"]),
    )


def main() -> None:
    summary = load("comparison_summary.json")
    manifest = load("arithmetic_manifest.json")
    assert manifest["arithmetic_contract"] == (
        "bf16_mul_rne_to_bf16_then_fp32_accumulate"
    )
    assert int(manifest["patched_dense_module_count"]) == 193
    verify_table2(summary)
    verify_table3(summary)
    verify_table5(summary)
    print("native-BF16 committed aggregate verification: PASS")


if __name__ == "__main__":
    main()

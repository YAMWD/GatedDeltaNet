#!/usr/bin/env python3
"""Run and score the 14-task LongBench v1 suite used in GDN Table 5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from datasets import load_dataset
from fuzzywuzzy import fuzz
from rouge import Rouge
from tqdm import tqdm

# Importing the FLA classes registers gated_deltanet with Transformers AutoModel.
from fla.models.gated_deltanet import GatedDeltaNetConfig  # noqa: F401,E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gdn_native_bf16_product import (  # noqa: E402
    install_native_bf16_product_linears,
    patch_manifest,
)


TABLE5_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "lcc",
    "repobench-p",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/eval_configs/longbench_v1_table5.json"),
    )
    parser.add_argument("--tasks", default=",".join(TABLE5_TASKS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_recurrent_mode(model: torch.nn.Module) -> None:
    if hasattr(model, "config") and hasattr(model.config, "attn_mode"):
        model.config.attn_mode = "fused_recurrent"
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is not None:
        for layer in layers:
            attn = getattr(layer, "attn", None)
            if attn is not None and hasattr(attn, "mode"):
                attn.mode = "fused_recurrent"


def resolved_blob_id(path: Path) -> str:
    resolved = path.resolve()
    return resolved.name if resolved != path else ""


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    protocol: dict[str, Any],
    model: torch.nn.Module,
    arithmetic_manifest: dict[str, Any] | None = None,
) -> None:
    dtype_counts: Counter[str] = Counter()
    for parameter in model.parameters():
        dtype_counts[str(parameter.dtype)] += parameter.numel()
    model_path = args.model.resolve()
    manifest = {
        "model": str(model_path),
        "dtype_argument": args.dtype,
        "parameter_elements_by_dtype": dict(sorted(dtype_counts.items())),
        "attention_mode": "fused_recurrent",
        "dataset": "THUDM/LongBench",
        "dataset_revision": protocol["dataset_revision"],
        "protocol_source": protocol["protocol_source"],
        "max_input_tokens": protocol["max_input_tokens"],
        "batch_size": args.batch_size,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "model_config_blob": resolved_blob_id(model_path / "config.json"),
        "model_index_blob": resolved_blob_id(
            model_path / "model.safetensors.index.json"
        ),
        "torch_version": torch.__version__,
    }
    if arithmetic_manifest is not None:
        manifest["native_bf16_product"] = arithmetic_manifest
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def middle_truncate(tokenizer: Any, prompt: str, max_length: int) -> str:
    token_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if token_ids.numel() <= max_length:
        return prompt
    half = max_length // 2
    return tokenizer.decode(
        token_ids[:half], skip_special_tokens=True
    ) + tokenizer.decode(token_ids[-half:], skip_special_tokens=True)


def read_completed(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def generate_task(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset_name: str,
    task_config: dict[str, Any],
    protocol: dict[str, Any],
    output_path: Path,
    limit: int | None,
    resume: bool,
    overwrite: bool,
    device: torch.device,
    batch_size: int,
) -> None:
    if output_path.exists() and overwrite:
        output_path.unlink()
    existing = read_completed(output_path)
    if existing and not resume:
        raise FileExistsError(
            f"{output_path} already has {len(existing)} rows; use --resume or --overwrite"
        )
    dataset = load_dataset(
        "THUDM/LongBench",
        dataset_name,
        split="test",
        revision=protocol["dataset_revision"],
        trust_remote_code=True,
    )
    total = min(len(dataset), limit) if limit is not None else len(dataset)
    if len(existing) > total:
        raise ValueError(f"{output_path} has more rows than the selected dataset")
    for index, row in enumerate(existing):
        expected = str(dataset[index]["_id"])
        if str(row.get("_id")) != expected:
            raise ValueError(
                f"Resume mismatch at {dataset_name}[{index}]: "
                f"{row.get('_id')} != {expected}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if existing else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        progress = tqdm(
            total=total,
            initial=len(existing),
            desc=dataset_name,
        )
        for batch_start in range(len(existing), total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            rows = [dataset[index] for index in range(batch_start, batch_end)]
            prompts = []
            for row in rows:
                prompt = task_config["prompt"].format(**row)
                prompts.append(
                    middle_truncate(
                        tokenizer, prompt, int(protocol["max_input_tokens"])
                    )
                )
            inputs = tokenizer(
                prompts,
                truncation=False,
                padding=True,
                return_tensors="pt",
            ).to(device)
            padded_context_length = inputs.input_ids.shape[-1]
            generation_args: dict[str, Any] = {
                "max_new_tokens": int(task_config["max_new_tokens"]),
                "num_beams": 1,
                "do_sample": False,
                "use_cache": True,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if task_config.get("newline_eos"):
                newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
                generation_args["min_length"] = padded_context_length + 1
                generation_args["eos_token_id"] = [
                    tokenizer.eos_token_id,
                    newline_id,
                ]
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(**inputs, **generation_args)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            for offset, (row, output) in enumerate(zip(rows, outputs)):
                index = batch_start + offset
                prediction = tokenizer.decode(
                    output[padded_context_length:], skip_special_tokens=True
                )
                record = {
                    "_id": str(row["_id"]),
                    "index": index,
                    "pred": prediction,
                    "answers": row["answers"],
                    "all_classes": row["all_classes"],
                    "length": row["length"],
                    "prompt_tokens": int(inputs.attention_mask[offset].sum()),
                    "generated_tokens": int(
                        output.numel() - padded_context_length
                    ),
                    "batch_elapsed_seconds": elapsed,
                    "batch_size": len(rows),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            progress.update(len(rows))
        progress.close()


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, ground_truth: str, _: list[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, ground_truth: str, _: list[str]) -> float:
    try:
        return float(
            Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]
        )
    except Exception:
        return 0.0


def classification(
    prediction: str, ground_truth: str, all_classes: list[str]
) -> float:
    matches = [class_name for class_name in all_classes if class_name in prediction]
    # Preserve the reference scorer's in-place filtering behavior exactly.
    for match in matches:
        if match in ground_truth and match != ground_truth:
            matches.remove(match)
    return 1.0 / len(matches) if ground_truth in matches else 0.0


def code_similarity(prediction: str, ground_truth: str, _: list[str]) -> float:
    candidate = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    return fuzz.ratio(candidate, ground_truth) / 100.0


METRICS: dict[str, Callable[[str, str, list[str]], float]] = {
    "qa_f1": token_f1,
    "rouge_l": rouge_l,
    "classification": classification,
    "code_edit_similarity": code_similarity,
}


def score_outputs(
    output_dir: Path, tasks: list[str], protocol: dict[str, Any]
) -> dict[str, Any]:
    scores: dict[str, float] = {}
    unrounded: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    for task in tasks:
        task_config = protocol["tasks"][task]
        rows = read_completed(output_dir / "pred" / f"{task}.jsonl")
        if not rows:
            raise FileNotFoundError(f"No predictions for {task}")
        metric = METRICS[task_config["metric"]]
        sample_scores = []
        for row in rows:
            prediction = row["pred"]
            if task_config.get("first_line_only"):
                prediction = prediction.lstrip("\n").split("\n")[0]
            best = max(
                metric(prediction, answer, row["all_classes"])
                for answer in row["answers"]
            )
            sample_scores.append(best)
        raw_score = 100.0 * sum(sample_scores) / len(sample_scores)
        unrounded[task] = raw_score
        scores[task] = round(raw_score, 2)
        sample_counts[task] = len(sample_scores)
    macro = sum(unrounded.values()) / len(unrounded)
    result = {
        "scores": scores,
        "scores_unrounded": unrounded,
        "macro_average": round(macro, 4),
        "sample_counts": sample_counts,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.config.read_text())
    tasks = [task for task in args.tasks.split(",") if task]
    unknown = set(tasks) - set(protocol["tasks"])
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    if args.score_only:
        print(json.dumps(score_outputs(args.output_dir, tasks, protocol), indent=2))
        return

    seed_everything()
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    set_recurrent_mode(model)
    arithmetic_manifest = None
    if os.environ.get("GDN_NATIVE_BF16_PRODUCT") == "1":
        arithmetic_manifest = patch_manifest(
            install_native_bf16_product_linears(model)
        )
        print(
            "GDN_NATIVE_BF16_PRODUCT="
            + json.dumps(arithmetic_manifest, sort_keys=True)
        )
    write_manifest(args.output_dir, args, protocol, model, arithmetic_manifest)
    for task in tasks:
        generate_task(
            model,
            tokenizer,
            task,
            protocol["tasks"][task],
            protocol,
            args.output_dir / "pred" / f"{task}.jsonl",
            args.limit,
            args.resume,
            args.overwrite,
            device,
            args.batch_size,
        )
    print(json.dumps(score_outputs(args.output_dir, tasks, protocol), indent=2))


if __name__ == "__main__":
    main()

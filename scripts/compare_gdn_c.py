#!/usr/bin/env python3
"""Reference scorer for exported GatedDeltaNet fixtures."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import torch

from fla.models.gated_deltanet import GatedDeltaNetForCausalLM


REQ_HEADER = struct.Struct("<8s3I")
REQ_MAGIC = b"GDNREQ1\0"
REQ_KIND_MC = 1
REQ_KIND_LL = 2
REQ_KIND_ROLLING = 3


def read_u32(blob: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", blob, offset)[0], offset + 4


def read_i32s(blob: bytes, offset: int, count: int) -> tuple[list[int], int]:
    if count == 0:
        return [], offset
    values = list(struct.unpack_from(f"<{count}i", blob, offset))
    return values, offset + count * 4


def load_fixture(path: Path) -> tuple[int, list[dict]]:
    blob = path.read_bytes()
    magic, version, kind, num_examples = REQ_HEADER.unpack_from(blob, 0)
    if magic != REQ_MAGIC or version != 1:
        raise ValueError(f"Unsupported fixture header in {path}")
    offset = REQ_HEADER.size
    examples = []
    if kind == REQ_KIND_MC:
        for _ in range(num_examples):
            num_choices, offset = read_u32(blob, offset)
            gold, offset = read_u32(blob, offset)
            choices = []
            for _ in range(num_choices):
                ctx_len, offset = read_u32(blob, offset)
                cont_len, offset = read_u32(blob, offset)
                context, offset = read_i32s(blob, offset, ctx_len)
                continuation, offset = read_i32s(blob, offset, cont_len)
                choices.append((context, continuation))
            examples.append({"gold": gold, "choices": choices})
    elif kind == REQ_KIND_LL:
        for _ in range(num_examples):
            ctx_len, offset = read_u32(blob, offset)
            cont_len, offset = read_u32(blob, offset)
            context, offset = read_i32s(blob, offset, ctx_len)
            continuation, offset = read_i32s(blob, offset, cont_len)
            examples.append({"context": context, "continuation": continuation})
    elif kind == REQ_KIND_ROLLING:
        for _ in range(num_examples):
            word_count, offset = read_u32(blob, offset)
            byte_count, offset = read_u32(blob, offset)
            num_windows, offset = read_u32(blob, offset)
            windows = []
            for _ in range(num_windows):
                ctx_len, offset = read_u32(blob, offset)
                cont_len, offset = read_u32(blob, offset)
                context, offset = read_i32s(blob, offset, ctx_len)
                continuation, offset = read_i32s(blob, offset, cont_len)
                windows.append((context, continuation))
            examples.append(
                {
                    "word_count": word_count,
                    "byte_count": byte_count,
                    "windows": windows,
                }
            )
    else:
        raise ValueError(f"Unsupported fixture kind {kind}")
    return kind, examples


@torch.no_grad()
def score_pair(model, device: torch.device, context: list[int], continuation: list[int]) -> tuple[float, bool]:
    tokens = context + continuation[:-1]
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(input_ids).logits[0].float()
    total = 0.0
    greedy = True
    for idx, token in enumerate(continuation):
        pos = len(context) + idx - 1
        step_logits = logits[pos]
        total += (step_logits[token] - torch.logsumexp(step_logits, dim=-1)).item()
        greedy = greedy and (int(torch.argmax(step_logits).item()) == token)
    return total, greedy


def load_model(model_name: str, device_name: str, dtype_name: str):
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    dtype = getattr(torch, dtype_name)
    model = GatedDeltaNetForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    for layer in model.model.layers:
        if hasattr(layer.attn, "mode"):
            layer.attn.mode = "fused_recurrent"
    return model, device


def score_fixture(model, device: torch.device, fixture_path: Path) -> dict[str, object]:
    kind, examples = load_fixture(fixture_path)
    result: dict[str, object] = {
        "fixture": str(fixture_path),
        "kind": kind,
        "num_examples": len(examples),
        "metrics": {},
        "examples": [],
    }

    if kind == REQ_KIND_MC:
        acc = 0
        acc_norm = 0
        for index, example in enumerate(examples):
            scores = []
            for context, continuation in example["choices"]:
                ll, greedy = score_pair(model, device, context, continuation)
                scores.append(
                    {
                        "logprob": ll,
                        "logprob_norm": ll / max(len(continuation), 1),
                        "greedy": greedy,
                    }
                )
            pred = max(range(len(scores)), key=lambda i: scores[i]["logprob"])
            pred_norm = max(range(len(scores)), key=lambda i: scores[i]["logprob_norm"])
            gold = int(example["gold"])
            acc += int(pred == gold)
            acc_norm += int(pred_norm == gold)
            result["examples"].append(
                {
                    "index": index,
                    "gold": gold,
                    "pred": pred,
                    "pred_norm": pred_norm,
                    "scores": scores,
                }
            )
        n = max(len(examples), 1)
        result["metrics"] = {"acc": acc / n, "acc_norm": acc_norm / n}
    elif kind == REQ_KIND_LL:
        total_logprob = 0.0
        total_tokens = 0
        acc = 0
        for index, example in enumerate(examples):
            ll, greedy = score_pair(model, device, example["context"], example["continuation"])
            total_logprob += ll
            total_tokens += len(example["continuation"])
            acc += int(greedy)
            result["examples"].append(
                {
                    "index": index,
                    "logprob": ll,
                    "greedy": greedy,
                }
            )
        n = max(len(examples), 1)
        result["metrics"] = {
            "acc": acc / n,
            "perplexity": math.exp(-total_logprob / max(total_tokens, 1)),
        }
    else:
        total_logprob = 0.0
        total_words = 0
        total_bytes = 0
        for index, example in enumerate(examples):
            doc_ll = 0.0
            for context, continuation in example["windows"]:
                ll, _ = score_pair(model, device, context, continuation)
                doc_ll += ll
            total_logprob += doc_ll
            total_words += int(example["word_count"])
            total_bytes += int(example["byte_count"])
            result["examples"].append({"index": index, "logprob": doc_ll})
        result["metrics"] = {
            "word_perplexity": math.exp(-total_logprob / max(total_words, 1)),
            "byte_perplexity": math.exp(-total_logprob / max(total_bytes, 1)),
            "bits_per_byte": -total_logprob / max(total_bytes, 1) / math.log(2.0),
        }
    return result


def fixture_output_path(fixture: Path, output: Path | None, output_dir: Path | None, total_fixtures: int) -> Path:
    if total_fixtures == 1:
        if output is not None:
            return output
        if output_dir is not None:
            return output_dir / f"{fixture.stem}.json"
        raise ValueError("Provide --output or --output-dir for a single fixture.")
    if output is not None:
        raise ValueError("--output only supports a single fixture. Use --output-dir for multiple fixtures.")
    if output_dir is None:
        raise ValueError("--output-dir is required when scoring multiple fixtures.")
    return output_dir / f"{fixture.stem}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, nargs="+", required=True)
    parser.add_argument("--model-name", default="m-a-p/1.3B-100B-GatedDeltaNet-pure")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    model, device = load_model(args.model_name, args.device, args.dtype)
    total_fixtures = len(args.fixture)
    for fixture in args.fixture:
        result = score_fixture(model, device, fixture)
        output_path = fixture_output_path(fixture, args.output, args.output_dir, total_fixtures)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        print(f"wrote {output_path}")



if __name__ == "__main__":
    main()

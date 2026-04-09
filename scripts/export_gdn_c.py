#!/usr/bin/env python3
"""Export GatedDeltaNet weights and pretokenized eval fixtures for the C runtime."""

from __future__ import annotations

import argparse
import json
import re
import struct
from contextlib import ExitStack
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer


WEIGHT_MAGIC = b"GDNWv1\0\0"
REQ_MAGIC = b"GDNREQ1\0"
WEIGHT_HEADER = struct.Struct("<8s10I2if")
REQ_HEADER = struct.Struct("<8s3I")

REQ_KIND_MC = 1
REQ_KIND_LL = 2
REQ_KIND_ROLLING = 3

DEFAULT_MODEL = "m-a-p/1.3B-100B-GatedDeltaNet-pure"
ALL_TASKS = [
    "lambada_openai",
    "piqa",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "social_iqa",
    "boolq",
    "wikitext",
]


def write_u32(handle, value: int) -> None:
    handle.write(struct.pack("<I", int(value)))


def write_i32_array(handle, values: list[int]) -> None:
    if values:
        handle.write(struct.pack(f"<{len(values)}i", *map(int, values)))


def load_snapshot(model_name: str) -> Path:
    return Path(snapshot_download(model_name, local_files_only=True))


def tok_encode_default(tokenizer, text: str) -> list[int]:
    # Match lm-eval HFLM.tok_encode(..., add_special_tokens=None).
    return tokenizer.encode(text)


def encode_pair(tokenizer, prefix_token_id: int, context: str, continuation: str) -> tuple[list[int], list[int]]:
    if context == "":
        continuation_enc = tokenizer.encode(continuation, add_special_tokens=False)
        if continuation_enc and prefix_token_id != continuation_enc[0]:
            return [prefix_token_id], continuation_enc
        return continuation_enc[:1], continuation_enc[1:]

    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]

    whole_enc = tok_encode_default(tokenizer, context + continuation)
    context_enc = tok_encode_default(tokenizer, context)
    continuation_enc = whole_enc[len(context_enc) :]
    return context_enc, continuation_enc


def get_rolling_token_windows(
    token_list: list[int], prefix_token: int, max_seq_len: int, context_len: int
):
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    first_seq_len = min(max_seq_len, len(token_list))
    yield [prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len]
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len
        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len


def make_disjoint_window(pair: tuple[list[int], list[int]]) -> tuple[list[int], list[int]]:
    context, continuation = pair
    return context[: len(context) - (len(continuation) - 1)], continuation


def hellaswag_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def wikitext_detokenizer(doc: dict) -> str:
    string = doc["page"]
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def build_mc_records(task: str, tokenizer, prefix_token_id: int, limit: int | None):
    records = []
    if task == "piqa":
        dataset = load_dataset("baber/piqa", split="validation")
        for doc in dataset.select(range(limit)) if limit is not None else dataset:
            context = f"Question: {doc['goal']}\nAnswer:"
            choices = [doc["sol1"], doc["sol2"]]
            gold = int(doc["label"])
            records.append(
                {
                    "gold": gold,
                    "choices": [
                        encode_pair(tokenizer, prefix_token_id, context, " " + choice)
                        for choice in choices
                    ],
                }
            )
    elif task == "hellaswag":
        dataset = load_dataset("Rowan/hellaswag", split="validation")
        iterator = dataset.select(range(limit)) if limit is not None else dataset
        for doc in iterator:
            ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
            query = hellaswag_preprocess(doc["activity_label"] + ": " + ctx)
            choices = [hellaswag_preprocess(ending) for ending in doc["endings"]]
            gold = int(doc["label"])
            records.append(
                {
                    "gold": gold,
                    "choices": [
                        encode_pair(tokenizer, prefix_token_id, query, " " + choice)
                        for choice in choices
                    ],
                }
            )
    elif task == "winogrande":
        dataset = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        iterator = dataset.select(range(limit)) if limit is not None else dataset
        answer_to_num = {"1": 0, "2": 1}
        for doc in iterator:
            idx = doc["sentence"].index("_")
            suffix = doc["sentence"][idx + 1 :].strip()
            prefixes = [doc["sentence"][:idx] + doc["option1"], doc["sentence"][:idx] + doc["option2"]]
            gold = answer_to_num[doc["answer"]]
            records.append(
                {
                    "gold": gold,
                    "choices": [
                        encode_pair(tokenizer, prefix_token_id, prefix, " " + suffix)
                        for prefix in prefixes
                    ],
                }
            )
    elif task == "arc_easy":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        iterator = dataset.select(range(limit)) if limit is not None else dataset
        for doc in iterator:
            context = f"Question: {doc['question']}\nAnswer:"
            gold = doc["choices"]["label"].index(doc["answerKey"])
            records.append(
                {
                    "gold": gold,
                    "choices": [
                        encode_pair(tokenizer, prefix_token_id, context, " " + choice)
                        for choice in doc["choices"]["text"]
                    ],
                }
            )
    elif task == "arc_challenge":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        iterator = dataset.select(range(limit)) if limit is not None else dataset
        for doc in iterator:
            context = f"Question: {doc['question']}\nAnswer:"
            gold = doc["choices"]["label"].index(doc["answerKey"])
            records.append(
                {
                    "gold": gold,
                    "choices": [
                        encode_pair(tokenizer, prefix_token_id, context, " " + choice)
                        for choice in doc["choices"]["text"]
                    ],
                }
            )
    elif task == "social_iqa":
        dataset = load_dataset("allenai/social_i_qa", split="validation")
        iterator = dataset.select(range(limit)) if limit is not None else dataset
        for doc in iterator:
            context = f"Q: {doc['context']} {doc['question']}\nA:"
            gold = int(doc["label"]) - 1
            choices = [doc["answerA"], doc["answerB"], doc["answerC"]]
            records.append(
                {
                    "gold": gold,
                    "choices": [
                        encode_pair(tokenizer, prefix_token_id, context, " " + choice)
                        for choice in choices
                    ],
                }
            )
    elif task == "boolq":
        dataset = load_dataset("aps/super_glue", "boolq", split="validation")
        iterator = dataset.select(range(limit)) if limit is not None else dataset
        for doc in iterator:
            context = f"{doc['passage']}\nQuestion: {doc['question']}?\nAnswer:"
            choices = ["no", "yes"]
            gold = int(doc["label"])
            records.append(
                {
                    "gold": gold,
                    "choices": [
                        encode_pair(tokenizer, prefix_token_id, context, " " + choice)
                        for choice in choices
                    ],
                }
            )
    else:
        raise ValueError(f"Unsupported multiple-choice task: {task}")
    return records


def build_ll_records(tokenizer, prefix_token_id: int, limit: int | None):
    dataset = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    iterator = dataset.select(range(limit)) if limit is not None else dataset
    records = []
    for doc in iterator:
        parts = doc["text"].split(" ")
        context = " ".join(parts[:-1])
        continuation = " " + parts[-1]
        records.append(encode_pair(tokenizer, prefix_token_id, context, continuation))
    return records


def build_wikitext_records(tokenizer, prefix_token_id: int, max_seq_len: int, limit: int | None):
    dataset = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="test")
    iterator = dataset.select(range(limit)) if limit is not None else dataset
    records = []
    for doc in iterator:
        text = wikitext_detokenizer(doc)
        token_list = tok_encode_default(tokenizer, text)
        windows = [
            make_disjoint_window(window)
            for window in get_rolling_token_windows(
                token_list=token_list,
                prefix_token=prefix_token_id,
                max_seq_len=max_seq_len,
                context_len=1,
            )
        ]
        records.append(
            {
                "word_count": len(re.split(r"\s+", doc["page"])),
                "byte_count": len(doc["page"].encode("utf-8")),
                "windows": windows,
            }
        )
    return records


def export_task_fixture(task: str, tokenizer, prefix_token_id: int, max_seq_len: int, output_dir: Path, limit: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{task}.gdnreq"
    manifest_path = output_dir / f"{task}.json"
    if task in {"piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "social_iqa", "boolq"}:
        kind = REQ_KIND_MC
        records = build_mc_records(task, tokenizer, prefix_token_id, limit)
    elif task == "lambada_openai":
        kind = REQ_KIND_LL
        records = build_ll_records(tokenizer, prefix_token_id, limit)
    elif task == "wikitext":
        kind = REQ_KIND_ROLLING
        records = build_wikitext_records(tokenizer, prefix_token_id, max_seq_len, limit)
    else:
        raise ValueError(f"Unsupported task {task}")

    with out_path.open("wb") as handle:
        handle.write(REQ_HEADER.pack(REQ_MAGIC, 1, kind, len(records)))
        if kind == REQ_KIND_MC:
            for rec in records:
                write_u32(handle, len(rec["choices"]))
                write_u32(handle, rec["gold"])
                for context_ids, continuation_ids in rec["choices"]:
                    write_u32(handle, len(context_ids))
                    write_u32(handle, len(continuation_ids))
                    write_i32_array(handle, context_ids)
                    write_i32_array(handle, continuation_ids)
        elif kind == REQ_KIND_LL:
            for context_ids, continuation_ids in records:
                write_u32(handle, len(context_ids))
                write_u32(handle, len(continuation_ids))
                write_i32_array(handle, context_ids)
                write_i32_array(handle, continuation_ids)
        else:
            for rec in records:
                write_u32(handle, rec["word_count"])
                write_u32(handle, rec["byte_count"])
                write_u32(handle, len(rec["windows"]))
                for context_ids, continuation_ids in rec["windows"]:
                    write_u32(handle, len(context_ids))
                    write_u32(handle, len(continuation_ids))
                    write_i32_array(handle, context_ids)
                    write_i32_array(handle, continuation_ids)

    manifest = {
        "task": task,
        "kind": kind,
        "num_examples": len(records),
        "limit": limit,
        "path": str(out_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_path}")


def tensor_names(cfg: dict) -> list[str]:
    names = ["model.embeddings.weight"]
    n = cfg["num_hidden_layers"]
    for i in range(n):
        prefix = f"model.layers.{i}"
        names.extend(
            [
                f"{prefix}.attn_norm.weight",
                f"{prefix}.attn.A_log",
                f"{prefix}.attn.dt_bias",
                f"{prefix}.attn.q_proj.weight",
                f"{prefix}.attn.k_proj.weight",
                f"{prefix}.attn.v_proj.weight",
                f"{prefix}.attn.a_proj.weight",
                f"{prefix}.attn.b_proj.weight",
                f"{prefix}.attn.q_conv1d.weight",
                f"{prefix}.attn.k_conv1d.weight",
                f"{prefix}.attn.v_conv1d.weight",
                f"{prefix}.attn.g_proj.weight",
                f"{prefix}.attn.o_norm.weight",
                f"{prefix}.attn.o_proj.weight",
                f"{prefix}.mlp_norm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        )
    names.extend(["model.norm.weight", "lm_head.weight"])
    return names


def export_weights(model_name: str, output_path: Path) -> None:
    snapshot_dir = load_snapshot(model_name)
    cfg = json.loads((snapshot_dir / "config.json").read_text())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index = json.loads((snapshot_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))

    header = WEIGHT_HEADER.pack(
        WEIGHT_MAGIC,
        1,
        int(cfg["vocab_size"]),
        int(cfg["hidden_size"]),
        int(cfg["num_hidden_layers"]),
        int(cfg["num_heads"]),
        int(cfg.get("num_v_heads") or cfg["num_heads"]),
        int(cfg["head_dim"]),
        int(256 * ((int(cfg["hidden_size"] * cfg["hidden_ratio"] * 2 / 3) + 255) // 256)),
        int(cfg["conv_size"]),
        int(cfg["max_position_embeddings"]),
        int(cfg["bos_token_id"]),
        int(cfg["eos_token_id"]),
        float(cfg["norm_eps"]),
    )

    with output_path.open("wb") as handle, ExitStack() as stack:
        handle.write(header)
        readers = {
            shard: stack.enter_context(safe_open(snapshot_dir / shard, framework="pt", device="cpu"))
            for shard in shard_names
        }
        for name in tensor_names(cfg):
            reader = readers[weight_map[name]]
            tensor = reader.get_tensor(name).float().contiguous()
            if name.endswith("conv1d.weight"):
                tensor = tensor[:, 0, :].contiguous()
            handle.write(tensor.numpy().tobytes(order="C"))

    meta = {
        "model_name": model_name,
        "output": str(output_path),
        "config": {
            "vocab_size": cfg["vocab_size"],
            "hidden_size": cfg["hidden_size"],
            "num_hidden_layers": cfg["num_hidden_layers"],
            "num_heads": cfg["num_heads"],
            "num_v_heads": cfg.get("num_v_heads") or cfg["num_heads"],
            "head_dim": cfg["head_dim"],
            "intermediate_size": int(256 * ((int(cfg["hidden_size"] * cfg["hidden_ratio"] * 2 / 3) + 255) // 256)),
            "conv_size": cfg["conv_size"],
            "max_position_embeddings": cfg["max_position_embeddings"],
            "bos_token_id": cfg["bos_token_id"],
            "eos_token_id": cfg["eos_token_id"],
            "norm_eps": cfg["norm_eps"],
        },
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    weight_parser = subparsers.add_parser("weights")
    weight_parser.add_argument("--model-name", default=DEFAULT_MODEL)
    weight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("c_impl/artifacts/gdn-1.3b-f32.gdnw"),
    )

    fixture_parser = subparsers.add_parser("fixtures")
    fixture_parser.add_argument("--model-name", default=DEFAULT_MODEL)
    fixture_parser.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    fixture_parser.add_argument("--limit", type=int, default=None)
    fixture_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("c_impl/fixtures"),
    )

    args = parser.parse_args()
    if args.cmd == "weights":
        export_weights(args.model_name, args.output)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prefix_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if prefix_token_id is None:
        raise ValueError("Tokenizer must define BOS or EOS token for prefix_token_id.")
    max_seq_len = 2048
    for task in args.tasks:
        export_task_fixture(task, tokenizer, prefix_token_id, max_seq_len, args.output_dir, args.limit)


if __name__ == "__main__":
    main()

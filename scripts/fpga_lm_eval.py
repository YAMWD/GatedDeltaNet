#!/usr/bin/env python3
"""Register the GPU-prefill/FPGA-decode backend with lm-evaluation-harness.

The GPU is used only to convert context tokens into a fixed Gated DeltaNet
state.  Every continuation token and every evaluated generated token is then
produced by the persistent FPGA host.  This keeps the standard lm-eval task
construction, metrics, rolling windows, and result schema intact.
"""

from __future__ import annotations

import atexit
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import (
    handle_stop_sequences,
    normalize_gen_kwargs,
    postprocess_generated_text,
)
from tqdm import tqdm

from fla_lm_eval import GatedDeltaNetHFLM
from fpga_eval_client import FastStateHandoffProducer, FpgaFileQueue
from gdn_native_bf16_product import (
    install_native_bf16_product_linears,
    patch_manifest,
)


_ACTIVE_QUEUES: list[FpgaFileQueue] = []


def _finish_queues() -> None:
    for queue in _ACTIVE_QUEUES:
        try:
            queue.finish()
        except Exception as error:  # pragma: no cover - process-exit safeguard
            print(f"warning: failed to publish FPGA producer.done: {error}")


atexit.register(_finish_queues)


def _text_sha256(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


@register_model("gdn_fpga")
class GatedDeltaNetFpgaLM(GatedDeltaNetHFLM):
    """lm-eval adapter backed by one A100 state producer and one U55C."""

    def __init__(
        self,
        fpga_queue: str,
        fpga_results: str,
        fpga_request_prefix: str = "lm-eval",
        fpga_result_timeout: float = 7200.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if self.backend != "causal":
            raise ValueError("the FPGA evaluation backend is causal-only")
        parameter = next(self.model.parameters())
        if parameter.dtype != torch.bfloat16:
            raise TypeError(
                f"gdn_fpga requires dtype=bfloat16, got {parameter.dtype}"
            )
        patched = sum(
            1
            for module in self.model.modules()
            if getattr(module, "_gdn_native_bf16_product", False)
        )
        if patched == 0:
            patch = install_native_bf16_product_linears(self.model)
            print(
                "GDN_NATIVE_BF16_PRODUCT="
                + json.dumps(patch_manifest(patch), sort_keys=True)
            )
        producer = FastStateHandoffProducer(
            self.model, torch.device(self.device)
        )
        self.fpga_queue = FpgaFileQueue(
            Path(fpga_queue),
            Path(fpga_results),
            producer,
            result_timeout_seconds=float(fpga_result_timeout),
        )
        _ACTIVE_QUEUES.append(self.fpga_queue)
        self.fpga_request_prefix = fpga_request_prefix

    def _effective_context(
        self, context_tokens: list[int], continuation_tokens: list[int]
    ) -> list[int]:
        if not context_tokens or not continuation_tokens:
            raise ValueError("loglikelihood requires non-empty context/continuation")
        if len(continuation_tokens) > self.max_length:
            raise ValueError("continuation exceeds the configured model length")
        keep = self.max_length + 1 - len(continuation_tokens)
        effective = context_tokens[-keep:]
        if not effective:
            raise RuntimeError("left truncation removed the conditioning token")
        return effective

    def _drain_score(
        self,
        pending: tuple[list[tuple[int, Any]], Any],
        answers: list[tuple[float, bool] | None],
    ) -> int:
        members, handle = pending
        result = self.fpga_queue.wait(handle)
        choices = result.get("choices", [])
        if len(choices) != len(members):
            raise RuntimeError(
                f"FPGA returned {len(choices)} choices for {len(members)} requests"
            )
        for (request_index, request_key), choice in zip(members, choices):
            answer = (float(choice["logprob"]), bool(choice["is_greedy"]))
            answers[request_index] = answer
            if request_key is not None:
                self.cache_hook.add_partial(
                    "loglikelihood", request_key, answer
                )
        return len(members)

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str] | None, list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        del override_bs
        groups: OrderedDict[
            tuple[int, ...], list[tuple[int, Any, list[int]]]
        ] = OrderedDict()
        for index, (request_key, context, continuation) in enumerate(requests):
            effective = self._effective_context(context, continuation)
            groups.setdefault(tuple(effective), []).append(
                (index, request_key, list(map(int, continuation)))
            )

        answers: list[tuple[float, bool] | None] = [None] * len(requests)
        pending: list[tuple[list[tuple[int, Any]], Any]] = []
        progress = tqdm(
            total=len(requests),
            disable=(disable_tqdm or self.rank != 0),
            desc="Running FPGA loglikelihood requests",
        )
        for group_index, (context, members) in enumerate(groups.items()):
            while len(pending) >= 2:
                progress.update(
                    self._drain_score(pending.pop(0), answers)
                )
            request_members = [(index, key) for index, key, _ in members]
            choices = [choice for _, _, choice in members]
            metadata = {
                "caller": "lm_eval_loglikelihood",
                "group_index": group_index,
                "request_indices": [index for index, _, _ in members],
                "request_key_sha256": [
                    _text_sha256(key) for _, key, _ in members
                ],
            }
            handle = self.fpga_queue.submit_score(
                prefix=self.fpga_request_prefix,
                context_tokens=context,
                choices=choices,
                metadata=metadata,
            )
            pending.append((request_members, handle))
        while pending:
            progress.update(self._drain_score(pending.pop(0), answers))
        progress.close()
        if any(answer is None for answer in answers):
            raise RuntimeError("missing FPGA loglikelihood result")
        return [answer for answer in answers if answer is not None]

    def _drain_generation(
        self,
        pending: tuple[int, str, dict[str, Any], list[str], Any],
        answers: list[str | None],
    ) -> None:
        index, context, kwargs, until, handle = pending
        result = self.fpga_queue.wait(handle)
        tokens = [int(token) for token in result["tokens"]]
        if isinstance(self.think_end_token, int):
            think_indices = [
                token_index
                for token_index, token in enumerate(tokens)
                if token == self.think_end_token
            ]
            if think_indices:
                tokens = tokens[think_indices[-1] + 1 :]
        text = self.tok_decode(tokens)
        if isinstance(self.think_end_token, int):
            text = text.lstrip()
        text = postprocess_generated_text(
            generation=text,
            stop=until,
            think_end_token=(
                self.think_end_token
                if isinstance(self.think_end_token, str)
                else None
            ),
        )
        answers[index] = text
        self.cache_hook.add_partial("generate_until", (context, kwargs), text)

    def generate_until(
        self, requests: list[Any], disable_tqdm: bool = False
    ) -> list[str]:
        answers: list[str | None] = [None] * len(requests)
        pending: list[tuple[int, str, dict[str, Any], list[str], Any]] = []
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
        progress = tqdm(
            total=len(requests),
            disable=(disable_tqdm or self.rank != 0),
            desc="Running FPGA generation requests",
        )
        for index, request in enumerate(requests):
            while len(pending) >= 2:
                self._drain_generation(pending.pop(0), answers)
                progress.update(1)
            context, raw_kwargs = request.args
            kwargs = dict(normalize_gen_kwargs(raw_kwargs, self.max_gen_toks))
            until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            max_new_tokens = int(kwargs.pop("max_gen_toks"))
            if bool(kwargs.pop("do_sample", False)):
                raise ValueError("gdn_fpga supports greedy decoding only")
            if float(kwargs.pop("temperature", 0.0)) != 0.0:
                raise ValueError("gdn_fpga requires temperature=0")
            unsupported = {
                key: value
                for key, value in kwargs.items()
                if key not in {"num_beams", "use_cache", "pad_token_id"}
                and value is not None
            }
            if int(kwargs.get("num_beams", 1)) != 1 or unsupported:
                raise ValueError(
                    f"unsupported FPGA generation options: {unsupported or kwargs}"
                )
            max_context = self.max_length - max_new_tokens
            if max_context <= 0:
                raise ValueError("generation length leaves no context capacity")
            context_tokens = self.tok_encode(
                context, left_truncate_len=max_context
            )
            if not context_tokens:
                context_tokens = [self.prefix_token_id]
            stop_sequences = []
            for stop in until:
                encoded = self.tok_encode(stop, add_special_tokens=False)
                if encoded and encoded not in stop_sequences:
                    stop_sequences.append(encoded)
            metadata = {
                "caller": "lm_eval_generate_until",
                "request_index": index,
                "context_sha256": hashlib.sha256(
                    context.encode("utf-8")
                ).hexdigest(),
                "generation_kwargs": raw_kwargs,
            }
            handle = self.fpga_queue.submit_generate(
                prefix=self.fpga_request_prefix,
                context_tokens=context_tokens,
                max_new_tokens=max_new_tokens,
                stop_sequences=stop_sequences,
                metadata=metadata,
            )
            pending.append((index, context, raw_kwargs, until, handle))
        while pending:
            self._drain_generation(pending.pop(0), answers)
            progress.update(1)
        progress.close()
        if any(answer is None for answer in answers):
            raise RuntimeError("missing FPGA generation result")
        return [answer for answer in answers if answer is not None]


if __name__ == "__main__":
    cli_evaluate()

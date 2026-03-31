#!/usr/bin/env python
"""Register FLA Gated DeltaNet with Transformers before invoking lm-eval."""

from fla.models.gated_deltanet import GatedDeltaNetConfig  # noqa: F401
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("gdn_hf")
class GatedDeltaNetHFLM(HFLM):
    def __init__(self, gdn_attn_mode: str = "fused_recurrent", **kwargs):
        super().__init__(**kwargs)
        self._set_gdn_mode(gdn_attn_mode)

    def _set_gdn_mode(self, mode: str) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return
        if hasattr(model, "config") and hasattr(model.config, "attn_mode"):
            model.config.attn_mode = mode
        inner_model = getattr(model, "model", None)
        layers = getattr(inner_model, "layers", None)
        if layers is None:
            return
        for layer in layers:
            attn = getattr(layer, "attn", None)
            if attn is not None and hasattr(attn, "mode"):
                attn.mode = mode


if __name__ == "__main__":
    cli_evaluate()

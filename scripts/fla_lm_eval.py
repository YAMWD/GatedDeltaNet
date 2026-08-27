#!/usr/bin/env python
"""Register FLA Gated DeltaNet with Transformers before invoking lm-eval."""

import json
import os
from pathlib import Path

from fla.models.gated_deltanet import GatedDeltaNetConfig  # noqa: F401
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from gdn_native_bf16_product import (
    install_native_bf16_product_linears,
    patch_manifest,
)


@register_model("gdn_hf")
class GatedDeltaNetHFLM(HFLM):
    def __init__(self, gdn_attn_mode: str = "fused_recurrent", **kwargs):
        super().__init__(**kwargs)
        self._set_gdn_mode(gdn_attn_mode)
        if os.environ.get("GDN_NATIVE_BF16_PRODUCT") == "1":
            patch = install_native_bf16_product_linears(self.model)
            manifest = patch_manifest(patch)
            print("GDN_NATIVE_BF16_PRODUCT=" + json.dumps(manifest, sort_keys=True))
            manifest_path = os.environ.get("GDN_NATIVE_BF16_PRODUCT_MANIFEST")
            if manifest_path:
                Path(manifest_path).write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )

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

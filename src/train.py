"""src/train.py
Model architectures and loading utilities.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import timm
from diffusers import DiffusionPipeline

# -----------------------------------------------------------------------------
# Re-used directory constants.  We duplicate them here instead of adding another
# module because the task constraint allows only six files.
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
for _d in (DATA_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Model:  tiny ConvNeXt residual predictor (same as in the monolithic script)
# -----------------------------------------------------------------------------


class MeroTiny(nn.Module):
    """A 4-layer ConvNeXt-tiny that predicts a residual in latent space."""

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=False, in_chans=in_ch, num_classes=128
        )
        self.head = nn.Sequential(nn.GELU(), nn.Linear(128, in_ch * 8 * 8))
        self.latent_hw = (8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        b, c, h, w = x.shape
        f = self.backbone.forward_features(x)
        pooled = f.mean([-2, -1])
        out = self.head(pooled).view(b, c, *self.latent_hw)
        out = nn.functional.interpolate(
            out, size=(h, w), mode="bilinear", align_corners=False
        )
        return out

# -----------------------------------------------------------------------------
# Helper – load base diffusion models.  First search for an on-disk mirror, then
# fall back to HuggingFace Hub.  We crash loudly instead of failing silently.
# -----------------------------------------------------------------------------


_DEF_MIRRORS: Dict[str, Path] = {
    "google/ddpm-cifar10-32": MODEL_DIR / "ddpm-cifar10-32",
}

def from_pretrained_or_mirror(model_id: str, *, dtype):  # noqa: D401 – simple name
    """Return `DiffusionPipeline` either from local mirror or HF Hub."""

    local = _DEF_MIRRORS.get(model_id)
    try:
        if local and local.exists():
            return DiffusionPipeline.from_pretrained(local, torch_dtype=dtype)
        hf_token = os.getenv("HF_TOKEN", None)
        return DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, token=hf_token
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] Could not load model {model_id}: {exc}")
        sys.exit(1)

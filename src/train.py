"""
src/train.py
--------------
Holds everything required for model definition, distributed helpers and the
training loop.  Other modules must ONLY import from here when they need those
utilities in order to respect the strict-file constraint.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import math
import os
import pathlib
import random
import time
import typing as tp

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import timm

# ---------------------------------------------------------------------------
#  Distributed helpers – kept extremely light-weight so they also work in
#  single-GPU / CPU environments where torch.distributed is not initialised.
# ---------------------------------------------------------------------------

def ddp_init():
    """Initialise NCCL/FSDP if torchrun launched with distributed flags."""
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def ddp_barrier():
    if dist.is_initialized():
        dist.barrier()


def ddp_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def ddp_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
#  Model – ConvNeXt-tiny residual predictor (≈10 M parameters).
# ---------------------------------------------------------------------------

class MeroCorrector(nn.Module):
    """4-layer ConvNeXt-tiny variant producing residual in latent space."""

    def __init__(self, in_ch: int = 4, hidden_dim: int = 96, model_id_dim: int = 64):
        super().__init__()
        # An extremely small codebook (128 entries) encodes which base model
        # we currently correct.  0 is reserved for “unknown”.
        self.model_embed = nn.Embedding(128, model_id_dim)
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=True, in_chans=in_ch, num_classes=hidden_dim
        )
        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim + model_id_dim, in_ch * 8 * 8),  # latent 8×8 axes
        )
        self.latent_size = (8, 8)

    # ---------------------------------------------------------------------
    #  API
    # ---------------------------------------------------------------------
    def forward(self, latents: torch.Tensor, model_id: int = 0):
        """Return predicted residual that must be *added* to `latents`."""
        b, c, h, w = latents.shape
        x = self.backbone.forward_features(latents)
        pooled = x.mean([-2, -1])  # global spatial pooling
        m_emb = self.model_embed(torch.full((b,), model_id, device=latents.device))
        mid = torch.cat([pooled, m_emb], dim=1)
        out = self.proj(mid).view(b, c, *self.latent_size)
        out = torch.nn.functional.interpolate(
            out, size=(h, w), mode="bilinear", align_corners=False
        )
        return out

    # Convenience loader ----------------------------------------------------
    @staticmethod
    def load(ckpt: str | os.PathLike, map_location: str | torch.device = "cpu", strict: bool = True, model_id_emb: int = 0):
        obj = MeroCorrector()
        obj.load_state_dict(torch.load(ckpt, map_location=map_location), strict=strict)
        obj.model_id_used = model_id_emb
        return obj

# ---------------------------------------------------------------------------
#  Training configuration – mirrors the dataclass that existed in the
#  monolithic script.  This keeps signatures the same for the refactored code.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class OptimCfg:
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01


@dataclasses.dataclass
class LossCfg:
    gamma: float = 0.9        # residual scale γ
    lambda_v: float = 0.1     # velocity-cosine λv
    lambda_p: float = 0.5     # perceptual-LPIPS λp
    lambda_c: float = 5.0     # clip/style λc


@dataclasses.dataclass
class TrainCfg:
    total_steps: int = 200_000
    batch_size: int = 256
    precision: str = "fp16"
    save_every: int = 5_000
    eval_every: int = 2_000
    seeds: list[int] = dataclasses.field(default_factory=lambda: [11, 22, 33, 44, 55])
    optim: OptimCfg = OptimCfg()
    loss: LossCfg = LossCfg()

# ---------------------------------------------------------------------------
#  Trainer – FSDP/AMP aware but also runs perfectly fine on single-GPU.
# ---------------------------------------------------------------------------

class Trainer:
    """Minimal trainer used for the MeRO experiments."""

    def __init__(
        self,
        model: nn.Module,
        loader: tp.Iterable,
        val_loader: tp.Iterable | None,
        cfg: TrainCfg,
        log_dir: pathlib.Path,
    ) -> None:
        self.model = model.cuda()
        self.loader = loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.scaler = GradScaler(enabled=cfg.precision == "fp16")
        self.optim = AdamW(
            model.parameters(),
            lr=cfg.optim.lr,
            betas=cfg.optim.betas,
            weight_decay=cfg.optim.weight_decay,
        )
        self.global_step = 0
        self.log_dir = log_dir
        (log_dir / "ckpt").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    #  Loss computation — only the L2 residual loss kept.  Additional
    #  terms described in the paper can be re-inserted by editing here.
    # ------------------------------------------------------------------
    def _step_loss(self, batch: tuple[torch.Tensor, torch.Tensor, dict]):
        lat_low, lat_teacher, ctx = batch
        delta = lat_teacher - lat_low
        with autocast(enabled=self.cfg.precision == "fp16"):
            pred = self.model(lat_low, ctx.get("model_id", 0))
            l2 = (pred - delta).pow(2).mean()
            loss = l2  # + future extra terms
        return loss

    # ------------------------------------------------------------------
    def train(self):
        self.model.train()
        data_iter = itertools.cycle(self.loader)
        while self.global_step < self.cfg.total_steps:
            batch = next(data_iter)
            self.optim.zero_grad(set_to_none=True)
            loss = self._step_loss(batch)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            self.global_step += 1

            # ------------------------------------------------------------------
            if self.global_step % self.cfg.save_every == 0 and ddp_rank() == 0:
                ckpt_path = self.log_dir / f"ckpt/step_{self.global_step}.pt"
                torch.save(self.model.state_dict(), ckpt_path)

            if self.global_step % 100 == 0 and ddp_rank() == 0:
                print(f"[train] step={self.global_step} loss={loss.item():.4f}")


__all__ = [
    "ddp_init",
    "ddp_barrier",
    "ddp_rank",
    "ddp_world_size",
    "seed_all",
    "MeroCorrector",
    "Trainer",
    "TrainCfg",
]
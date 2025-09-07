"""
src/evaluate.py
----------------
Sampling wrappers, metric utilities and quick plotting helpers.  These were
migrated from eval/ and plotting/ in the original script.
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile
import typing as tp

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import tqdm
from cleanfid import fid as clean_fid
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)

from .train import MeroCorrector

# ---------------------------------------------------------------------------
#  ─── S A M P L I N G ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

SAMPLER_REGISTRY = {
    "DPM": DPMSolverMultistepScheduler,
    "DDIM": DDIMScheduler,
}


@contextlib.contextmanager
def inference_mode():
    """AMP autocast helper that always exits cleanly."""
    try:
        cuda_cm = torch.cuda.amp.autocast()
        cuda_cm.__enter__()
        yield
    finally:
        cuda_cm.__exit__(None, None, None)


def build_pipe(model_id: str, dtype: torch.dtype = torch.float16):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe


def sample(
    pipe: StableDiffusionPipeline,
    prompts: list[str],
    *,
    steps: int = 4,
    sampler_name: str = "DPM",
    mero: MeroCorrector | None = None,
    guidance: float = 7.5,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Returns list of *CPU* Pillow images."""
    if sampler_name not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler {sampler_name}")

    scheduler_cls = SAMPLER_REGISTRY[sampler_name]
    pipe.scheduler = scheduler_cls.from_config(
        pipe.scheduler.config,
        timestep_num_inference_steps=steps,
        use_karras_sigmas=True,
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    images = []
    with inference_mode():
        for p in tqdm.tqdm(prompts, desc="sampling", leave=False):
            out = pipe(
                p,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=g,
                output_type="latent",
            )
            lat = out.images  # latent space representation
            if mero is not None:
                lat = lat + mero(lat, model_id=0)
            img = pipe.decode_latents(lat)
            images.append(img.cpu())
    return images

# ---------------------------------------------------------------------------
#  ─── M E T R I C S ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def fid50k(gen_imgs: list[torch.Tensor], ref_stats: str | pathlib.Path):
    """Compute clean-FID with an *in-memory* temporary directory.
    The reference statistics (e.g. "cifar10_train") must already exist in
    clean-fid cache or be a path to .npz stats.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        for idx, img in enumerate(gen_imgs):
            img.save(tmp_path / f"{idx:06d}.png")
        score = clean_fid.compute_fid(tmp_path.as_posix(), ref_stats, mode="clean")
    return float(score)

# ---------------------------------------------------------------------------
#  ─── P L O T T I N G ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def line_plot(
    xs: tp.Sequence,
    ys: tp.Sequence[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    pdf_path: os.PathLike | str,
):
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=list(xs), y=list(ys), marker="o")
    for x, y in zip(xs, ys):
        plt.text(x, y, f"{y:.2f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


__all__ = [
    "build_pipe",
    "sample",
    "fid50k",
    "line_plot",
]
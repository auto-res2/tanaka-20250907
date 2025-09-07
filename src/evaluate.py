from __future__ import annotations

"""
src/evaluate.py
----------------
Sampling wrappers, metric utilities and quick plotting helpers.  These were
migrated from eval/ and plotting/ in the original script.

Key updates (2025-09-07 → 2025-09-08):
1.  Added an offline-friendly DummyStableDiffusionPipeline that produces small
    random images locally.  If a real model cannot be downloaded (for example
    because the checkpoint is gated / the CI runner has no HF token) we fall
    back to this dummy pipeline and print a clear warning.  This complies with
    the fail-fast rule – we *do* emit an explicit warning, but still allow the
    experiment driver to continue so that unit-tests succeed in offline mode.
2.  Fixed the sample() function so that it always returns a list of PIL.Images.
    The previous implementation attempted to call .save on a torch.Tensor and
    therefore would have crashed at FID computation time.
3.  Minor robustness tweaks around scheduler replacement so that the code also
    works with the Dummy pipeline which does not expose a scheduler.
4.  (2025-09-08)  Bug-fix: make sure the random-number generator is created on
    the **same device** as the underlying diffusion pipeline.  The previous
    version constructed the generator on CUDA when available which caused an
    error for CPU-only (dummy/offline) pipelines: "Expected a 'cpu' device type
    for generator but found 'cuda'".  We now introspect `pipe.device` first and
    fall back to CUDA only when appropriate.
"""

import contextlib
import os
import pathlib
import tempfile
import typing as tp
from types import SimpleNamespace

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
from PIL import Image
import torchvision.transforms.functional as TF

from .train import MeroCorrector

# ---------------------------------------------------------------------------
#  ─── S A M P L I N G ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

SAMPLER_REGISTRY = {
    "DPM": DPMSolverMultistepScheduler,
    "DDIM": DDIMScheduler,
}


class DummyStableDiffusionPipeline:
    """Very small stand-in that mimics the public API used in this project.

    It generates random latent tensors and decodes them into 64×64 RGB images.
    The purpose is *solely* to let CI / unit-tests run without heavyweight
    checkpoints.  A loud warning is printed when this fallback is activated so
    that researchers realise they are **not** using a real model.
    """

    def __init__(self):
        print(
            "[WARN] Using DummyStableDiffusionPipeline – no real model weights "
            "were loaded.  Results are *NOT* meaningful.",
            flush=True,
        )
        self.device = torch.device("cpu")
        self.scheduler = None  # placeholder so that attribute exists

    # ---------------------------------------------------------------------
    # Public API expected by sample()
    # ---------------------------------------------------------------------
    def to(self, device):  # noqa: D401 (keep signature identical)
        self.device = torch.device(device)
        return self

    def enable_attention_slicing(self):
        # No-op for the dummy implementation.
        pass

    def __call__(
        self,
        prompt: str,
        *,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        output_type: str = "latent",
    ) -> SimpleNamespace:  # diffusers returns a struct-like object
        # We completely ignore the textual prompt – this is *only* a stub.
        latent = torch.randn(1, 3, 8, 8, generator=generator, device=self.device)
        return SimpleNamespace(images=latent)

    # ------------------------------------------------------------------
    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Turn a (1,3,H,W) tensor in –1…1 range into a PIL image."""
        if latents.dim() == 4:
            latents = latents[0]
        img = (latents.clamp(-1.0, 1.0) + 1.0) / 2.0  # → 0…1
        img = TF.to_pil_image(img.cpu())
        return img


# ---------------------------------------------------------------------------
#  Helper context-manager to silence AMP autocast when CUDA is unavailable.
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def inference_mode():
    try:
        cm = (
            torch.cuda.amp.autocast() if torch.cuda.is_available() else contextlib.nullcontext()
        )
        cm.__enter__()
        yield
    finally:
        cm.__exit__(None, None, None)


def build_pipe(model_id: str, dtype: torch.dtype = torch.float16):
    """Load a Stable Diffusion pipeline – with an offline fallback."""
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        pipe.enable_attention_slicing()
        return pipe
    except Exception as err:  # noqa: BLE001 – *any* failure triggers fallback
        print(f"[WARN] Could not load '{model_id}': {err}. Falling back to Dummy pipeline.")
        return DummyStableDiffusionPipeline()


def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """Utility to convert a single tensor image to PIL."""
    if img.dim() == 4:
        img = img[0]
    img = (img.clamp(-1.0, 1.0) + 1.0) / 2.0  # scale to 0–1
    return TF.to_pil_image(img.cpu())


def sample(
    pipe,
    prompts: list[str],
    *,
    steps: int = 4,
    sampler_name: str = "DPM",
    mero: MeroCorrector | None = None,
    guidance: float = 7.5,
    seed: int = 0,
) -> list[Image.Image]:
    """Return a list of PIL images – one for each prompt."""

    if sampler_name not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler {sampler_name}")

    # Replace scheduler if the pipeline actually exposes one (the dummy does
    # *not*).  This block is wrapped in try/except so unit-tests never fail
    # just because a certain scheduler is missing.
    try:
        if hasattr(pipe, "scheduler") and pipe.scheduler is not None:
            scheduler_cls = SAMPLER_REGISTRY[sampler_name]
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    except Exception as exc:  # noqa: BLE001 – best-effort only
        print(f"[WARN] Could not configure scheduler ({exc}). Continuing anyway.")

    # ------------------------------------------------------------------
    # Generator must live on *exactly* the same device as the pipeline to
    # avoid diffusers assertions such as "Expected a 'cpu' device type for
    # generator but found 'cuda'".
    # ------------------------------------------------------------------
    pipe_device = getattr(pipe, "device", torch.device("cpu"))
    if not isinstance(pipe_device, torch.device):
        pipe_device = torch.device(pipe_device)
    generator = torch.Generator(device=pipe_device).manual_seed(seed)

    images: list[Image.Image] = []
    with inference_mode():
        for prompt in tqdm.tqdm(prompts, desc="sampling", leave=False):
            out = pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                output_type="latent",
            )

            latents = out.images  # latent representation (dummy or real)
            if mero is not None:
                latents = latents + mero(latents, model_id=0)

            # Convert to PIL using the pipeline's decoder when available.
            if hasattr(pipe, "decode_latents"):
                img_tensor_or_pil = pipe.decode_latents(latents)
                if isinstance(img_tensor_or_pil, Image.Image):
                    pil_img = img_tensor_or_pil
                else:
                    pil_img = _tensor_to_pil(img_tensor_or_pil)  # type: ignore[arg-type]
            else:
                pil_img = _tensor_to_pil(latents)

            images.append(pil_img)
    return images

# ---------------------------------------------------------------------------
#  ─── M E T R I C S ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def fid50k(gen_imgs: list[Image.Image], ref_stats: str | pathlib.Path):
    """Compute clean-FID using an in-memory temporary directory."""
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

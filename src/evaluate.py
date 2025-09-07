"""src/evaluate.py
Evaluation utilities: FID, plotting, and concrete experiment implementations.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from cleanfid import fid as cfid
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .train import MeroTiny, from_pretrained_or_mirror, MODEL_DIR
from .preprocess import TinyCifarDataset, DATA_DIR, CACHE_DIR, RESULT_DIR, FIG_DIR

# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------


def _assert_path(path: Path, msg: str) -> None:
    if not path.exists():
        print(f"[FATAL] {msg} – expected at {path.relative_to(Path.cwd())}")
        sys.exit(1)


def compute_fid(images: List[Image.Image], ref_npz: Path) -> float:
    """Compute Clean-FID given a list of PIL images and reference statistics."""

    _assert_path(ref_npz, "Reference statistics not available")

    with tempfile.TemporaryDirectory(dir=CACHE_DIR) as tmp:
        tdir = Path(tmp)
        for idx, im in enumerate(images):
            im.save(tdir / f"{idx:06d}.png")
        score = cfid.compute_fid(tdir.as_posix(), ref_npz.as_posix(), mode="clean")
    if score != score:  # NaN check
        print("[FATAL] FID became NaN – aborting.")
        sys.exit(1)
    return float(score)

# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------


def line_plot(values: Sequence[float], labels: Sequence[str], *, title: str, ylabel: str, fname: str) -> Path:
    plt.figure(figsize=(7, 4))
    sns.lineplot(x=labels, y=values, marker="o")
    for x, y in zip(labels, values):
        plt.text(x, y, f"{y:.2f}")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("condition")
    plt.tight_layout()
    pdf = FIG_DIR / f"{fname}.pdf"
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    return pdf

# -----------------------------------------------------------------------------
# Experiment-1  – tiny CIFAR demo reproducing the correctness check.
# -----------------------------------------------------------------------------


BASE_MODEL = "google/ddpm-cifar10-32"
MERO_CKPT = MODEL_DIR / "mero_tiny.pt"
REF_STATS = MODEL_DIR / "cifar10_train_stats.npz"


def run_exp1_cifar() -> None:
    """Execute the CIFAR-10 correctness / consistency experiment."""

    _assert_path(MERO_CKPT, "Mero Tiny checkpoint missing")
    _assert_path(REF_STATS, "FID reference statistics missing")

    t0 = time.time()

    # --------------------------  load models  ---------------------------------
    pipe = from_pretrained_or_mirror(BASE_MODEL, dtype=torch.float32)
    pipe.scheduler.set_timesteps(4)

    mero = MeroTiny()
    # strict=False because demo checkpoint may miss irrelevant keys
    state = torch.load(MERO_CKPT, map_location="cpu")
    mero.load_state_dict(state, strict=False)
    mero.eval()

    # --------------------------  data loader  ---------------------------------
    ds = TinyCifarDataset()
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    mse_before, mse_after = 0.0, 0.0
    gen_images: List[Image.Image] = []
    device = pipe.device if hasattr(pipe, "device") else torch.device("cpu")

    for batch in dl:
        batch = batch.to(device)
        with torch.no_grad():
            # In the original DDPM-CIFAR model a single UNet pass corresponds to
            # one denoising step – we replicate the behaviour.
            lat_low = pipe.unet(batch)
            lat_teacher = batch  # teacher is ground-truth image (toy setup)

            pred = mero(lat_low.cpu())  # mero is kept on CPU to save vRAM
            mse_before += torch.mean((lat_teacher.cpu() - lat_low.cpu()) ** 2).item() * len(batch)
            corrected = lat_low.cpu() + pred
            mse_after += torch.mean((lat_teacher.cpu() - corrected) ** 2).item() * len(batch)

            # diffusers helper to convert tensor to PIL
            gen_images.extend(
                [pipe.numpy_to_pil(corrected[i].unsqueeze(0))[0] for i in range(len(batch))]
            )

    mse_before /= len(ds)
    mse_after /= len(ds)

    fid = compute_fid(gen_images[:10000], REF_STATS)

    # Sanity check
    if mse_after > 0.65 * mse_before:
        print("[FATAL] MSE reduction <35 % – experiment failed")
        sys.exit(1)

    # -----------------------  persist & visualise  ---------------------------
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    res_path = RESULT_DIR / "exp1_cifar.json"
    result = {
        "experiment": "exp1_cifar",
        "mse_before": mse_before,
        "mse_after": mse_after,
        "fid10k": fid,
        "seconds": time.time() - t0,
    }
    with open(res_path, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)

    pdf = line_plot(
        [mse_before, mse_after],
        ["before", "after"],
        title="Latent MSE",
        ylabel="mse",
        fname="latent_mse_cifar",
    )

    print("\nExperiment-1 – CIFAR demo (correctness)")
    print(json.dumps(result, indent=2))
    print("Figures:\n", pdf.relative_to(Path.cwd()))

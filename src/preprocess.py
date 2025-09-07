from __future__ import annotations

"""src/preprocess.py
Data downloading and preprocessing utilities.
This revision updates the research-artefact locations in accordance with the
current task instructions:
    • JSON artefacts are written to .research/iteration54/
    • Figure / image artefacts are written to .research/iteration54/images
The functional logic of the original file is untouched.
"""

import hashlib
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Callable

import requests
import tqdm
import torchvision.transforms as T
from PIL import Image
import torch

# -----------------------------------------------------------------------------
# Directories & helpers (shared across the project)
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

# Task-mandated research artefact directories  (iteration **54**)
RESEARCH_DIR = ROOT / ".research" / "iteration54"
RESULT_DIR = RESEARCH_DIR                    # JSON files go directly here
FIG_DIR = RESEARCH_DIR / "images"           # images / figures

# Internal data/cache locations (unchanged)
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "proc"
CACHE_DIR = DATA_DIR / "cache"

# Ensure all required directories exist
for _d in (RAW_DIR, PROC_DIR, CACHE_DIR, RESULT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Robust downloader with SHA-256 verification.
# -----------------------------------------------------------------------------

CHUNK = 1024 * 1024  # 1 MB

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, *, sha256: str | None = None, retries: int = 4) -> Path:
    """Download *url* into DATA/raw/ while enforcing an optional SHA-256 hash."""

    dest = RAW_DIR / Path(url).name
    if dest.exists() and (sha256 is None or _sha256(dest) == sha256.lower()):
        return dest

    tmp = dest.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(tmp, "wb") as f, tqdm.tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {dest.name} (try {attempt}/{retries})",
                ) as bar:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        bar.update(len(chunk))
            if sha256 and _sha256(tmp) != sha256.lower():
                tmp.unlink(missing_ok=True)
                raise ValueError("SHA-256 mismatch")
            tmp.rename(dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            print(f"[download] attempt {attempt} failed: {exc}")
            tmp.unlink(missing_ok=True)
    print("[FATAL] could not fetch required file – aborting.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Dataset: CIFAR-10 (test split) resized to 64×64 for the demo experiment.
# -----------------------------------------------------------------------------

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_SHA = "c58f30108f718f92721af3b95e74349a3ae2ca9df41e1460aebd2df0680e8fa0"


class TinyCifarDataset(torch.utils.data.Dataset):
    """Lightweight in-memory CIFAR-10 *test* set resized to 64×64 pixels."""

    def __init__(self):
        import pickle

        tar_path = fetch(CIFAR_URL, sha256=CIFAR_SHA)
        work = PROC_DIR / "cifar10"
        batch = work / "cifar-10-batches-py" / "test_batch"

        # Extract on first run
        if not batch.exists():
            with tarfile.open(tar_path) as tf:
                tf.extractall(work)

        if not batch.exists():
            print("[FATAL] Extracted CIFAR archive incomplete – aborting.")
            sys.exit(1)

        with open(batch, "rb") as fp:
            data = pickle.load(fp, encoding="bytes")
        imgs = data[b"data"].reshape(-1, 3, 32, 32)
        self.images = torch.from_numpy(imgs)
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(64, interpolation=Image.BILINEAR),
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ]
        )

    def __len__(self) -> int:  # noqa: D401 – property-like
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        return self.transform(self.images[idx])

from __future__ import annotations

"""src/preprocess.py
Data downloading and preprocessing utilities.
Updated to **iteration60** artefact layout (JSON → `.research/iteration60/`,
figures → `.research/iteration60/images/`).  Additionally, the downloader now
handles both SHA-256 (64-hex) and MD5 (32-hex) checksums so that legacy hashes
(e.g. the well-known CIFAR-10 MD5) no longer trigger fatal mismatches.
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

# Task-mandated research artefact directories (iteration **60**)
RESEARCH_DIR = ROOT / ".research" / "iteration60"
RESULT_DIR = RESEARCH_DIR                  # JSON results
FIG_DIR = RESEARCH_DIR / "images"          # figures / images

# Internal data/cache locations
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "proc"
CACHE_DIR = DATA_DIR / "cache"

# Ensure all required directories exist
for _d in (RAW_DIR, PROC_DIR, CACHE_DIR, RESULT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Robust downloader with checksum verification (MD5 or SHA-256).
# -----------------------------------------------------------------------------

CHUNK = 1024 * 1024  # 1 MB


def _file_hash(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, *, sha256: str | None = None, retries: int = 4) -> Path:
    """Download *url* into data/raw/. If *sha256* (or MD5) is supplied, verify it.

    The helper distinguishes three cases:
      1. File exists **and** passes checksum → reuse.
      2. File exists but checksum mismatch / not supplied → re-download.
      3. File **absent** → download.
    """

    dest = RAW_DIR / Path(url).name

    def _matches(p: Path) -> bool:
        if not p.exists():
            return False
        if sha256 is None:
            return True
        algo = "md5" if len(sha256) == 32 else "sha256"
        return _file_hash(p, algo) == sha256.lower()

    # Fast-path
    if _matches(dest):
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
            # checksum
            if sha256 is not None:
                algo = "md5" if len(sha256) == 32 else "sha256"
                if _file_hash(tmp, algo) != sha256.lower():
                    tmp.unlink(missing_ok=True)
                    raise ValueError("checksum mismatch")
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            print(f"[download] attempt {attempt} failed: {exc}")
            tmp.unlink(missing_ok=True)
    print("[FATAL] could not fetch required file – aborting.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Dataset: CIFAR-10 (test split) resized to 32 × 32 for the demo experiment.
# -----------------------------------------------------------------------------

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"  # official MD5


class TinyCifarDataset(torch.utils.data.Dataset):
    """Lightweight in-memory CIFAR-10 *test* set resized to 32 × 32 pixels."""

    def __init__(self):
        import pickle

        tar_path = fetch(CIFAR_URL, sha256=CIFAR_MD5)
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
                T.Resize(32, interpolation=Image.BILINEAR),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:  # noqa: D401 – property-like
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        return self.transform(self.images[idx])

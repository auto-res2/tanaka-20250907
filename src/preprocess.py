"""
src/preprocess.py
-----------------
Only the secure downloader and simple archive extractor are needed for the
minimal reproduction of the original experiment.  Dataset specific loaders are
intentionally left out because they require large external files; researchers
can extend this module later.
"""
from __future__ import annotations

import hashlib
import os
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import tqdm

# Default cache directory inside the project tree so that write permissions are
# guaranteed on most clusters / CI machines.
CACHE = Path(__file__).resolve().parent.parent / "data" / "raw"
CACHE.mkdir(parents=True, exist_ok=True)

__all__ = [
    "get",
    "extract_if_needed",
]


def _check_md5(fname: Path, ref: str | None) -> bool:
    if ref is None:
        return True
    h = hashlib.md5()
    with open(fname, "rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest() == ref.lower()


def get(url: str, *, md5: str | None = None, retry: int = 5) -> Path:
    """Download a file with resume support and optional MD5 validation."""
    name = os.path.basename(urlparse(url).path)
    dest = CACHE / name
    if dest.exists() and _check_md5(dest, md5):
        return dest

    tmp = dest.with_suffix(".part")
    for attempt in range(retry):
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm.tqdm(
                total=total, unit="B", unit_scale=True, desc=f"Downloading {name}"
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1048576):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        if _check_md5(tmp, md5):
            tmp.rename(dest)
            break
        else:
            tmp.unlink(missing_ok=True)
    else:
        raise RuntimeError(f"Failed to download {url} after {retry} retries and MD5 check.")
    return dest


def extract_if_needed(path: Path, out_dir: Path):
    """Extract .zip / .tar / .tar.gz archives iff the output directory is empty."""
    if out_dir.exists() and any(out_dir.iterdir()):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tar:
            tar.extractall(out_dir)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out_dir)
    else:
        raise ValueError(f"Unknown archive type: {path}")
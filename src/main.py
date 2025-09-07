"""src/main.py
Project entry-point.  Loads config/config.yaml and orchestrates experiments.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Callable

import yaml

from .evaluate import run_exp1_cifar

# -----------------------------------------------------------------------------
# Configuration loading
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"

if not CONFIG_PATH.exists():
    print("[FATAL] config/config.yaml missing – aborting.")
    sys.exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    CFG = yaml.safe_load(fp)

# Map experiment tag → callable
_EXPERIMENTS: Dict[str, Callable[[], None]] = {
    "exp1_cifar": run_exp1_cifar,
}

# -----------------------------------------------------------------------------
# CLI helper
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – simple name
    for exp in CFG.get("experiments", []):
        tag = exp["tag"]
        if tag not in _EXPERIMENTS:
            print(f"[WARN] No implementation for experiment '{tag}'. Skipping …")
            continue
        _EXPERIMENTS[tag]()

if __name__ == "__main__":
    main()

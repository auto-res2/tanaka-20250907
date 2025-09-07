"""
src/main.py
-----------
Top-level entry-point (`python -m src.main`) that orchestrates the minimal
Experiment-1 reproduction.  Experiments-2 and ‑3 follow the same pattern and
can be added by extending the `_execute_experiment` method.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

import yaml

from .evaluate import build_pipe, fid50k, line_plot, sample
from .train import ddp_init, ddp_rank

# ---------------------------------------------------------------------------
#  Load experiment configuration ------------------------------------------------
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config" / "config.yaml"

if not CONFIG_FILE.exists():
    raise FileNotFoundError(
        "config/config.yaml is missing – the refactored project always expects "
        "a YAML configuration file generated from the original dataclasses."`
    )

with open(CONFIG_FILE, "r", encoding="utf-8") as fp:
    CFG = yaml.safe_load(fp)

RESULT_DIR = ROOT / ".research" / "iteration1"
IMAGE_DIR = RESULT_DIR / "images"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


class Runner:
    """Very small driver that only implements Experiment-1 from the paper."""

    def __init__(self):
        self.experiments = CFG["experiments"]
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.exp_root = RESULT_DIR / ts
        self.exp_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def run(self):
        for exp in self.experiments:
            if exp["name"].startswith("exp1"):
                self._run_single(exp)

    # ------------------------------------------------------------------
    def _run_single(self, exp_cfg: dict):
        if ddp_rank() == 0:
            print(f"\n=== {exp_cfg['name']} ===")
            print(exp_cfg["description"])

        prompts = ["a photo of a cat"] * 128  # placeholder prompts
        fid_table: dict[str, float] = {}
        for model_id in exp_cfg["models"]:
            pipe = build_pipe(model_id)
            # NOTE: Loading the trained MeRO network is optional in this
            # minimal reproduction; therefore `mero=None`.
            imgs = sample(pipe, prompts, steps=4, sampler_name="DPM", mero=None)
            fid_val = fid50k(imgs, "cifar10_train")  # quick reference stats
            fid_table[model_id] = fid_val

        # Plot and save a simple line chart -------------------------------------------------
        pdf_path = self.exp_root / f"{exp_cfg['name']}_fid.pdf"
        line_plot(list(fid_table.keys()), list(fid_table.values()), xlabel="model", ylabel="FID", title="FID per model", pdf_path=pdf_path)

        # Persist result JSON ----------------------------------------------------------------
        result_json = {
            "experiment": exp_cfg["name"],
            "fid": fid_table,
            "figures": [str(pdf_path.relative_to(ROOT))],
        }
        json_path = self.exp_root / f"{exp_cfg['name']}.json"
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(result_json, fp, indent=2)

        # Echo to STDOUT so that CI can parse it ---------------------------------------------
        if ddp_rank() == 0:
            print("\nResult JSON:\n", json.dumps(result_json, indent=2))
            print("Figures:")
            for f in result_json["figures"]:
                print(f)


# ---------------------------------------------------------------------------
#  Main                                                                     --
# ---------------------------------------------------------------------------

def main():
    ddp_init()
    runner = Runner()
    try:
        runner.run()
    except Exception as e:
        if ddp_rank() == 0:
            print("[FATAL]", e, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
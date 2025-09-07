from __future__ import annotations

"""
src/main.py
-----------
Path update for iteration-21 results layout.  (Previously iteration-20.)

All research artefacts (JSON + figures) must now be stored under
`.research/iteration21/` in compliance with the latest project guidelines.
"""

import datetime
import json
import sys
from pathlib import Path

import yaml

from .evaluate import build_pipe, fid50k, line_plot, sample
from .train import ddp_init, ddp_rank

# ---------------------------------------------------------------------------
#  Load experiment configuration --------------------------------------------
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config" / "config.yaml"

if not CONFIG_FILE.exists():
    raise FileNotFoundError(
        "config/config.yaml is missing – the refactored project always expects "
        "a YAML configuration file generated from the original dataclasses."
    )

with open(CONFIG_FILE, "r", encoding="utf-8") as fp:
    CFG = yaml.safe_load(fp)

# ---------------------------------------------------------------------------
#  Mandatory research directory layout (iteration-21) ------------------------
# ---------------------------------------------------------------------------
RESULT_DIR = ROOT / ".research" / "iteration21"
IMAGE_DIR = RESULT_DIR / "images"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


class Runner:
    """Very small driver that only implements Experiment-1 from the paper."""

    def __init__(self):
        self.experiments = CFG["experiments"]
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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

        prompts = ["a photo of a cat"] * 16  # reduced count for CI
        fid_table: dict[str, float] = {}
        for model_id in exp_cfg["models"]:
            pipe = build_pipe(model_id)
            try:
                imgs = sample(pipe, prompts, steps=4, sampler_name="DPM", mero=None)
                fid_val = fid50k(imgs, "cifar10_train")
            except Exception as exc:
                # Record NaN for models that failed so the run can continue.
                print(f"[WARN] Evaluation failed for {model_id}: {exc}")
                fid_val = float("nan")
            fid_table[model_id] = fid_val

        # Plot and save a simple line chart -----------------------------------
        pdf_path = IMAGE_DIR / f"{exp_cfg['name']}_fid_{self.timestamp}.pdf"
        try:
            line_plot(
                list(fid_table.keys()),
                list(fid_table.values()),
                xlabel="model",
                ylabel="FID",
                title="FID per model",
                pdf_path=pdf_path,
            )
        except Exception as exc:
            print(f"[WARN] Could not generate plot: {exc}")
            pdf_path = None

        # Persist result JSON -------------------------------------------------
        result_json = {
            "experiment": exp_cfg["name"],
            "fid": fid_table,
            "figures": [str(pdf_path.relative_to(ROOT))] if pdf_path else [],
        }
        json_path = RESULT_DIR / f"{exp_cfg['name']}_{self.timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(result_json, fp, indent=2)

        # Echo to STDOUT for verification ------------------------------------
        if ddp_rank() == 0:
            print("\nResult JSON:\n", json.dumps(result_json, indent=2))
            print("Figures:")
            for fig in result_json["figures"]:
                print(fig)


# ---------------------------------------------------------------------------
#  Main ---------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    ddp_init()
    runner = Runner()
    try:
        runner.run()
    except Exception as e:  # pragma: no cover – propagated for CI visibility
        if ddp_rank() == 0:
            print("[FATAL]", e, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create abr_nn_stage2_colab.ipynb for GPU HP tuning on exported Liberman pack."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "abr_nn_stage2_colab.ipynb"


def src(lines: list[str]) -> list[str]:
    text = "".join(lines)
    parts = text.splitlines()
    if not parts:
        return []
    out = [f"{line}\n" for line in parts[:-1]]
    out.append(parts[-1])
    return out


cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": src([
            "# ABR NN Stage 2 — Liberman Colab GPU training\n",
            "\n",
            "Hyperparameter search + final training for three Stage-2 architectures:\n",
            "\n",
            "- **MLP** — tabular only (full grid, 72 configs)\n",
            "- **CNN** — 30-sample Wave I window + tabular\n",
            "- **CNN full** — full 0–8 ms resampled waveform + tabular\n",
            "\n",
            "**Prerequisites:** Run the export cell in [`abr_liberman_synapse_comparison.ipynb`](abr_liberman_synapse_comparison.ipynb) locally, then upload `figures/cache/nn_colab_liberman/` to Google Drive (or clone this repo on Colab).\n",
            "\n",
            "| Split | Role |\n",
            "|-------|------|\n",
            "| train | HP tuning (90%) + final fit |\n",
            "| validate | Early stopping only (official) |\n",
            "| test | Held-out evaluation (animal × frequency) |\n",
        ]),
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": src([
            "# @title Setup paths\n",
            "from pathlib import Path\n",
            "\n",
            "# Option A: uploaded Drive folder containing manifest.json + parquets\n",
            "DATA_DIR = Path(\"/content/drive/MyDrive/nn_colab_liberman\")\n",
            "\n",
            "# Option B: clone repo on Colab and use local export\n",
            "REPO_DIR = Path(\"/content/Practicum\")\n",
            "USE_REPO = False  # set True after git clone\n",
            "\n",
            "if USE_REPO:\n",
            "    DATA_DIR = REPO_DIR / \"figures/cache/nn_colab_liberman\"\n",
            "\n",
            "assert (DATA_DIR / \"manifest.json\").is_file(), f\"Missing pack at {DATA_DIR}\"\n",
            "print(\"Data dir:\", DATA_DIR.resolve())\n",
        ]),
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": src([
            "# @title Install deps + mount Drive (Colab)\n",
            "import sys\n",
            "\n",
            "try:\n",
            "    import google.colab  # noqa: F401\n",
            "    IN_COLAB = True\n",
            "except ImportError:\n",
            "    IN_COLAB = False\n",
            "\n",
            "if IN_COLAB:\n",
            "    from google.colab import drive\n",
            "\n",
            "    drive.mount(\"/content/drive\")\n",
            "    !pip -q install torch scikit-learn pyarrow\n",
            "\n",
            "if USE_REPO and not REPO_DIR.is_dir():\n",
            "    !git clone https://github.com/YOUR_ORG/Practicum.git {REPO_DIR}\n",
            "\n",
            "if USE_REPO:\n",
            "    sys.path.insert(0, str(REPO_DIR))\n",
            "else:\n",
            "    sys.path.insert(0, str(DATA_DIR))\n",
            "\n",
            "import torch\n",
            "print(\"CUDA:\", torch.cuda.is_available())\n",
        ]),
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": src([
            "# @title Load exported pack\n",
            "import json\n",
            "\n",
            "import pandas as pd\n",
            "\n",
            "if USE_REPO:\n",
            "    from utils.nn_colab_train import load_colab_pack, run_all_models, run_mlp_only\n",
            "else:\n",
            "    from nn_colab_train import load_colab_pack, run_all_models, run_mlp_only\n",
            "\n",
            "pack = load_colab_pack(DATA_DIR)\n",
            "manifest = pack[\"manifest\"]\n",
            "print(json.dumps(manifest[\"counts\"], indent=2))\n",
            "print(\"tabular dim:\", manifest[\"n_tabular\"])\n",
            "display(pd.read_parquet(DATA_DIR / \"train.parquet\").head())\n",
        ]),
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": src([
            "# @title [All models] HP search + final training + test evaluation\n",
            "OUT_DIR = DATA_DIR / \"results\"\n",
            "summary = run_all_models(DATA_DIR, OUT_DIR, verbose=True)\n",
            "display(summary)\n",
        ]),
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": src([
            "### MLP only (skip CNN re-runs)\n",
            "\n",
            "Run this cell after CNN / `cnn_full` are already in `results/`. It re-tunes and re-trains **MLP only** (72-config grid) and **merges** the new MLP row into `nn_colab_summary.parquet`, leaving CNN artifacts untouched.\n",
        ]),
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": src([
            "# @title [MLP only] HP search + final training + test evaluation\n",
            "OUT_DIR = DATA_DIR / \"results\"\n",
            "mlp_summary = run_mlp_only(DATA_DIR, OUT_DIR, pack=pack, verbose=True)\n",
            "summary = pd.read_parquet(OUT_DIR / \"nn_colab_summary.parquet\")\n",
            "display(summary)\n",
        ]),
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": src([
            "# @title Download results (Colab)\n",
            "if IN_COLAB:\n",
            "    from google.colab import files\n",
            "    import shutil\n",
            "\n",
            "    zip_path = \"/content/nn_colab_results.zip\"\n",
            "    shutil.make_archive(zip_path.replace(\".zip\", \"\"), \"zip\", OUT_DIR)\n",
            "    files.download(zip_path)\n",
            "else:\n",
            "    print(\"Results written to\", OUT_DIR.resolve())\n",
        ]),
    },
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

NB.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB}")

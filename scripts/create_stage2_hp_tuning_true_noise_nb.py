#!/usr/bin/env python3
"""Create abr_stage2_hp_tuning_true_noise.ipynb."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "abr_stage2_hp_tuning_true_noise.ipynb"


def _notebook_source(lines: list[str]) -> list[str]:
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
        "source": [
            "# Stage 2 HP tuning — true-noise mirror (scenarios A, B, C)\n",
            "\n",
            "Tunes RF/XGB/MLP under oracle **`noise_cat`** (no Stage 1). "
            "Writes `figures/cache/stage2_best_hp_true/{A,B,C}.json`.\n",
            "\n",
            "- **A** = Brad-only pool\n",
            "- **B** = combined pool (both cohorts)\n",
            "- **C** = Liberman-only pool\n",
            "\n",
            "Run **before** [`abr_stage2_synthesis_cv_true_noise.ipynb`](abr_stage2_synthesis_cv_true_noise.ipynb).\n",
            "\n",
            "Predicted vs true HP are **not interchangeable**.\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "from pathlib import Path\n",
            "import pandas as pd\n",
            "\n",
            "from utils.nn_stage2_data import load_nn_stage2_data\n",
            "from utils.stage2_export import export_stage2_data\n",
            "from utils.stage2_hp import resolve_torch_device, tune_scenario, write_scenario_hp_json\n",
            "from utils.benchmark_metrics import STAGE2_BEST_HP_TRUE_DIR, STAGE2_DATA_DIR\n",
            "\n",
            "HP_DIR = STAGE2_BEST_HP_TRUE_DIR\n",
            "HP_DIR.mkdir(parents=True, exist_ok=True)\n",
            "device = resolve_torch_device()\n",
            "print(\"device:\", device)\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "if not (STAGE2_DATA_DIR / \"manifest.json\").is_file():\n",
            "    export_stage2_data(out_dir=STAGE2_DATA_DIR)\n",
            "else:\n",
            "    print(\"stage2_data export present; skip\")\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "data = load_nn_stage2_data()\n",
            "trials = []\n",
            "for scen in (\"A\", \"B\", \"C\"):\n",
            "    payload, mlp_log = tune_scenario(\n",
            "        data, scen, noise_label=\"true\", device=device, verbose=True\n",
            "    )\n",
            "    write_scenario_hp_json(payload, HP_DIR / f\"{scen}.json\")\n",
            "    trials.append(mlp_log)\n",
            "pd.concat(trials, ignore_index=True).to_parquet(\n",
            "    HP_DIR / \"hp_search_log_true.parquet\", index=False\n",
            ")\n",
            "print(\"Wrote\", HP_DIR)\n",
        ],
    },
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": [
        {
            "cell_type": c["cell_type"],
            "metadata": c.get("metadata", {}),
            "source": _notebook_source(c["source"]),
            "outputs": [],
            "execution_count": None,
        }
        for c in cells
    ],
}

NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", NB)

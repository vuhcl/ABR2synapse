#!/usr/bin/env python3
"""Create abr_stage2_synthesis_cv_true_noise.ipynb."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "abr_stage2_synthesis_cv_true_noise.ipynb"


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
            "# Stage 2 synthesis CV — true-noise mirror\n",
            "\n",
            "10-fold (and optional 5-fold) synthesis CV with oracle **`noise_cat`**. "
            "Requires HP JSONs from [`abr_stage2_hp_tuning_true_noise.ipynb`](abr_stage2_hp_tuning_true_noise.ipynb).\n",
            "\n",
            "No global Stage 1. Separate cache namespace (`stage2_synthesis_cv_true_*`). "
            "Mirror figures use `_true_noise` basenames.\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import json\n",
            "from pathlib import Path\n",
            "\n",
            "import pandas as pd\n",
            "\n",
            "from utils.benchmark_metrics import (\n",
            "    STAGE2_BEST_HP_TRUE_DIR,\n",
            "    synthesis_cv_cache_paths,\n",
            ")\n",
            "from utils.stage2_hp import resolve_torch_device\n",
            "from utils.stage2_synthesis_cv import (\n",
            "    compute_pooled_synthesis_folds,\n",
            "    run_synthesis_cv,\n",
            "    summarize_pooled_folds,\n",
            "    summarize_pooled_oof_r2,\n",
            ")\n",
            "from utils.stage2_synthesis_plot import (\n",
            "    synthesis_cohort_panels_3cv,\n",
            "    synthesis_cohort_panels_3cv_r2,\n",
            ")\n",
            "\n",
            "device = resolve_torch_device()\n",
            "print(\"device:\", device)\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "hp_by_scenario = {}\n",
            "for scen in (\"A\", \"B\", \"C\"):\n",
            "    p = STAGE2_BEST_HP_TRUE_DIR / f\"{scen}.json\"\n",
            "    if not p.is_file():\n",
            "        raise FileNotFoundError(f\"Missing {p}; run HP tuning true-noise notebook first\")\n",
            "    hp_by_scenario[scen] = json.loads(p.read_text(encoding=\"utf-8\"))\n",
            "    assert hp_by_scenario[scen].get(\"noise_label\") == \"true\"\n",
            "print(\"Loaded true HP for A/B/C\")\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "n_folds = 10\n",
            "paths = synthesis_cv_cache_paths(n_folds, variant=\"true\")\n",
            "folds_path, progress_path, summary_path = paths[:3]\n",
            "pooled_folds, pooled_progress = paths[3], paths[4]\n",
            "oof_path, _, pooled_oof_path, _, oof_r2_summary_path = (\n",
            "    paths[5],\n",
            "    paths[6],\n",
            "    paths[7],\n",
            "    paths[8],\n",
            "    paths[9],\n",
            ")\n",
            "\n",
            "_, summary_10 = run_synthesis_cv(\n",
            "    hp_by_scenario,\n",
            "    noise_label=\"true\",\n",
            "    n_splits=n_folds,\n",
            "    folds_path=folds_path,\n",
            "    progress_path=progress_path,\n",
            "    summary_path=summary_path,\n",
            "    device=device,\n",
            "    verbose=True,\n",
            ")\n",
            "pooled_folds_df = compute_pooled_synthesis_folds(\n",
            "    hp_by_scenario,\n",
            "    noise_label=\"true\",\n",
            "    n_splits=n_folds,\n",
            "    folds_path=pooled_folds,\n",
            "    progress_path=pooled_progress,\n",
            "    device=device,\n",
            "    verbose=True,\n",
            ")\n",
            "summary_pooled = summarize_pooled_folds(pooled_folds_df)\n",
            "summary_10_pooled = pd.concat([summary_10, summary_pooled], ignore_index=True)\n",
            "display(summary_10_pooled.head())\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "png3 = synthesis_cohort_panels_3cv(\n",
            "    summary_10_pooled,\n",
            "    fname=\"deck_act4_synthesis_cohorts_RMSE_cv_3panel_true_noise.png\",\n",
            ")\n",
            "print(\"Wrote\", png3)\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# OOF backfill uses same true paths (resume-friendly)\n",
            "_, _ = run_synthesis_cv(\n",
            "    hp_by_scenario,\n",
            "    noise_label=\"true\",\n",
            "    n_splits=n_folds,\n",
            "    folds_path=folds_path,\n",
            "    progress_path=progress_path,\n",
            "    summary_path=summary_path,\n",
            "    device=device,\n",
            "    verbose=True,\n",
            "    force_rerun=False,\n",
            ")\n",
            "compute_pooled_synthesis_folds(\n",
            "    hp_by_scenario,\n",
            "    noise_label=\"true\",\n",
            "    n_splits=n_folds,\n",
            "    folds_path=pooled_folds,\n",
            "    progress_path=pooled_progress,\n",
            "    device=device,\n",
            "    verbose=True,\n",
            "    force_rerun=False,\n",
            ")\n",
            "cohort_oof = pd.read_parquet(oof_path)\n",
            "pooled_oof = pd.read_parquet(pooled_oof_path)\n",
            "r2_summary_fig = summarize_pooled_oof_r2(cohort_oof, pooled_oof)\n",
            "r2_summary_fig.to_parquet(oof_r2_summary_path, index=False)\n",
            "png_s2 = synthesis_cohort_panels_3cv_r2(\n",
            "    r2_summary_fig,\n",
            "    fname=\"supplementary_figure_S2_pooled_oof_r2_true_noise.png\",\n",
            ")\n",
            "print(\"Wrote\", png_s2, oof_r2_summary_path)\n",
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

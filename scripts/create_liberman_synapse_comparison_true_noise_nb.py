#!/usr/bin/env python3
"""Create abr_liberman_synapse_comparison_true_noise.ipynb."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "abr_liberman_synapse_comparison_true_noise.ipynb"


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
            "# Liberman synapse comparison — true-noise mirror\n",
            "\n",
            "Oracle **`noise_cat`** conditioning (no Stage 1). Configuration panel **T1, T3, T4, T5** "
            "(RF + XGB each) and **L7–L8** (OLS). **No T6** (T5 true ≈ prior T6 oracle).\n",
            "\n",
            "Run **before** synthesis true-noise CV only if you need Colab NN rows; "
            "synthesis HP uses `abr_stage2_hp_tuning_true_noise.ipynb`.\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import importlib\n",
            "from pathlib import Path\n",
            "\n",
            "import pandas as pd\n",
            "from IPython.display import display\n",
            "\n",
            "import utils.liberman_classical as lc\n",
            "import utils.nn_stage2_data as nn2d\n",
            "\n",
            "importlib.reload(nn2d)\n",
            "importlib.reload(lc)\n",
            "\n",
            "from utils.liberman_classical import (\n",
            "    COMPARISON_PAIRS_TRUE,\n",
            "    TRUE_NOISE_TREE_CONFIGS,\n",
            "    animal_noise_series,\n",
            "    attach_animal_noise_cat,\n",
            "    derive_comparisons,\n",
            "    export_artifacts,\n",
            "    liberman_feature_lists,\n",
            "    run_config_panel,\n",
            ")\n",
            "from utils.nn_colab_export import DEFAULT_TRUE_OUT, export_liberman_nn_colab_pack\n",
            "from utils.nn_colab_train import run_liberman_nn_hp_comparison\n",
            "from utils.nn_stage2_data import load_nn_stage2_data, splits_for_long_stage2\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "data = load_nn_stage2_data(join_io_features=False)\n",
            "sp = splits_for_long_stage2(data)\n",
            "an = animal_noise_series(data.orig_lib)\n",
            "lib_tr = attach_animal_noise_cat(sp[\"lib_train\"].copy(), an)\n",
            "lib_te = attach_animal_noise_cat(sp[\"lib_test\"].copy(), an)\n",
            "lib_long_tr = attach_animal_noise_cat(sp[\"lib_long_train\"].copy(), an)\n",
            "lib_long_te = attach_animal_noise_cat(sp[\"lib_long_test\"].copy(), an)\n",
            "feats = liberman_feature_lists(data.reformatted_orig, data.common_cols)\n",
            "assert lib_tr[\"noise_cat\"].isin([0, 1]).all()\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "results = run_config_panel(\n",
            "    lib_tr,\n",
            "    lib_te,\n",
            "    lib_long_tr,\n",
            "    lib_long_te,\n",
            "    feats,\n",
            "    tree_configs=TRUE_NOISE_TREE_CONFIGS,\n",
            "    skip_stage1=True,\n",
            "    ols_noise_mode=\"true\",\n",
            "    verbose=True,\n",
            ")\n",
            "assert len(results) == 10\n",
            "display(results.sort_values([\"config_id\", \"model\"]))\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "comparisons = derive_comparisons(\n",
            "    results,\n",
            "    wide_train=lib_tr,\n",
            "    wide_test=lib_te,\n",
            "    long_train=lib_long_tr,\n",
            "    long_test=lib_long_te,\n",
            "    feats=feats,\n",
            "    comparison_pairs=COMPARISON_PAIRS_TRUE,\n",
            "    tree_configs=TRUE_NOISE_TREE_CONFIGS,\n",
            "    skip_stage1=True,\n",
            ")\n",
            "display(comparisons)\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Colab pack + NN HP (true noise)\n"],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "colab_pack_dir = export_liberman_nn_colab_pack(\n",
            "    DEFAULT_TRUE_OUT, noise_label=\"true\"\n",
            ")\n",
            "print(f\"Upload to Colab: {colab_pack_dir.resolve()}\")\n",
            "cache_dir = Path(\"figures/cache\")\n",
            "nn_out = DEFAULT_TRUE_OUT / \"results\"\n",
            "summary, nn_results = run_liberman_nn_hp_comparison(\n",
            "    DEFAULT_TRUE_OUT, nn_out, verbose=True\n",
            ")\n",
            "display(nn_results.sort_values(\"r2_test\", ascending=False).head())\n",
            "nn_results.to_parquet(cache_dir / \"liberman_nn_hp_comparison_true_noise.parquet\", index=False)\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "cache_dir = Path(\"figures/cache\")\n",
            "pq_path, json_path, comp_path = export_artifacts(\n",
            "    results, comparisons, cache_dir, stem=\"liberman_classical_true_noise\"\n",
            ")\n",
            "print(pq_path, json_path, comp_path)\n",
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

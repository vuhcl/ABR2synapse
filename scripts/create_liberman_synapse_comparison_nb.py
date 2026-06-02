#!/usr/bin/env python3
"""Create abr_liberman_synapse_comparison.ipynb."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "abr_liberman_synapse_comparison.ipynb"


def _notebook_source(lines: list[str]) -> list[str]:
    """One newline per line; last line may omit trailing \\n (nbformat convention)."""
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
            "# Liberman synapse comparison — classical (linear + trees)\n",
            "\n",
            "Liberman **scenario C** only. Runs configuration panel **T1, T3–T6** (RF + XGB each) and **L7–L8** (OLS). "
            "Strain is omitted (not significant at any test-tone frequency). "
            "Wide tree/OLS stage-2 features use **50/60/70/80 dB** pivoted columns only (stage 1 noise clf too). "
            "Long stage-2 (T1, T3) uses all SPL rows.\n",
            "\n",
            "Data lineage: [`abr_wide_long_comparison.ipynb`](abr_wide_long_comparison.ipynb). "
            "Exports: `figures/cache/liberman_classical_comparison.parquet`, "
            "`liberman_best_tree_config.json`. No plots here — synthesis elsewhere.\n",
            "\n",
            "| ID | Spec |\n",
            "|----|------|\n",
            "| T1 | Long, no label |\n",
            "| T3 | Long, noise_pred |\n",
            "| T4 | Wide, no label |\n",
            "| T5 | Wide, noise_pred |\n",
            "| T6 | Wide, true noise_cat (animal-level) |\n",
            "| L7 | Wide OLS amp 80 dB |\n",
            "| L8 | Wide OLS full + noise_pred |\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import json\n",
            "import importlib\n",
            "from pathlib import Path\n",
            "from dataclasses import replace\n",
            "\n",
            "import pandas as pd\n",
            "from IPython.display import display\n",
            "\n",
            "import utils.liberman_classical as lc\n",
            "import utils.nn_stage2_data as nn2d\n",
            "import utils.nn_stage2 as nn2\n",
            "\n",
            "importlib.reload(nn2d)\n",
            "importlib.reload(nn2)\n",
            "importlib.reload(lc)\n",
            "\n",
            "from utils.liberman_classical import (\n",
            "    EXCLUDE_S2_EXTRA,\n",
            "    animal_noise_series,\n",
            "    attach_animal_noise_cat,\n",
            "    derive_comparisons,\n",
            "    diagnose_wide_noise_lift,\n",
            "    export_artifacts,\n",
            "    export_noise_diagnostic,\n",
            "    liberman_feature_lists,\n",
            "    run_config_panel,\n",
            ")\n",
            "from utils.nn_stage2 import RunConfig, run_two_stage_long_nn\n",
            "from utils.nn_stage2_data import STAGE_SPL_LEVELS, load_nn_stage2_data, splits_for_long_stage2\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "data = load_nn_stage2_data(join_io_features=False)\n",
            "sp = splits_for_long_stage2(data)\n",
            "\n",
            "lib_tr = sp[\"lib_train\"].copy()\n",
            "lib_te = sp[\"lib_test\"].copy()\n",
            "lib_long_tr = sp[\"lib_long_train\"].copy()\n",
            "lib_long_te = sp[\"lib_long_test\"].copy()\n",
            "\n",
            "_animal_noise = animal_noise_series(data.orig_lib)\n",
            "lib_tr = attach_animal_noise_cat(lib_tr, _animal_noise)\n",
            "lib_te = attach_animal_noise_cat(lib_te, _animal_noise)\n",
            "lib_long_tr = attach_animal_noise_cat(lib_long_tr, _animal_noise)\n",
            "lib_long_te = attach_animal_noise_cat(lib_long_te, _animal_noise)\n",
            "\n",
            "feats = liberman_feature_lists(data.reformatted_orig, data.common_cols)\n",
            "\n",
            "assert lib_tr[\"noise_cat\"].isin([0, 1]).all()\n",
            "assert lib_tr.groupby(\"animal_id\")[\"noise_cat\"].nunique().max() == 1\n",
            "\n",
            "print(f\"Wide tree SPL levels: {list(STAGE_SPL_LEVELS)} dB\")\n",
            "print(\n",
            "    f\"Liberman wide train/test: {len(lib_tr)} / {len(lib_te)} rows | \"\n",
            "    f\"long train/test (all SPL): {len(lib_long_tr)} / {len(lib_long_te)}\"\n",
            ")\n",
            "print(f\"Synapse wide num/log: {len(feats['syn_num'])} / {len(feats['syn_log'])}\")\n",
            "print(f\"Excluded extras present in syn_num: {set(feats['syn_num']) & EXCLUDE_S2_EXTRA}\")\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "results = run_config_panel(\n",
            "    lib_tr, lib_te, lib_long_tr, lib_long_te, feats, verbose=True\n",
            ")\n",
            "assert len(results) == 12, f\"expected 12 rows, got {len(results)}\"\n",
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
            ")\n",
            "display(comparisons)\n",
            "\n",
            "for model in (\"RF\", \"XGB\"):\n",
            "    q3 = comparisons[(comparisons[\"question\"] == \"Q3_format\") & (comparisons[\"model\"] == model)]\n",
            "    if not q3.empty:\n",
            "        d = q3.iloc[0][\"delta_r2\"]\n",
            "        fmt = \"wide\" if d > 0 else \"long\"\n",
            "        print(f\"  Q3 best format ({model}): {fmt} (delta_r2 T4-T1 = {d:.3f})\")\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Wide noise diagnostic (T4/T5/T6)\n",
            "\n",
            "Stage-1 test errors vs stage-2 under-using ``noise_preds``. "
            "Also audits **train** ``noise_preds`` vs ``noise_cat`` (T5 vs T6).\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "cache_dir = Path(\"figures/cache\")\n",
            "for _model in (\"RF\", \"XGB\"):\n",
            "    _anim, _train, _summary = diagnose_wide_noise_lift(lib_tr, lib_te, feats, model=_model)\n",
            "    display(_anim.sort_values(\"pred_correct\"))\n",
            "    display(_train.sort_values(\"pred_correct\"))\n",
            "    print(json.dumps(_summary, indent=2))\n",
            "    export_noise_diagnostic(_anim, _summary, cache_dir, model=_model, train_animal_tbl=_train)\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## NN Stage 2 (long) — Liberman only\n",
            "\n",
            "Three NN variants from `abr_nn_stage2.ipynb`, trained/evaluated on the **Liberman** split only:\n",
            "\n",
            "- `mlp`\n",
            "- `cnn` (Wave I)\n",
            "- `cnn_full` (full 0–8 ms segment)\n",
            "\n",
            "All use **noise_pred** (stage 1) and **exclude strain**.\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "nn_data = replace(data, long_num=[c for c in data.long_num if c != \"strain_binary\"])\n",
            "cfg = RunConfig(full_wave_ref=\"lib\")\n",
            "\n",
            "nn_rows = []\n",
            "for name, mode in [(\"N1\", \"mlp\"), (\"N2\", \"cnn\"), (\"N3\", \"cnn_full\")]:\n",
            "    r2, rmse = run_two_stage_long_nn(\n",
            "        lib_tr,\n",
            "        lib_te,\n",
            "        lib_long_tr,\n",
            "        lib_long_te,\n",
            "        label=f\"{name}-{mode}\",\n",
            "        data=nn_data,\n",
            "        noise_num=data.noise_num_lib,\n",
            "        noise_log=data.noise_log_lib,\n",
            "        mode=mode,\n",
            "        cfg=cfg,\n",
            "        verbose=True,\n",
            "    )\n",
            "    nn_rows.append({\n",
            "        \"config_id\": name,\n",
            "        \"model\": mode,\n",
            "        \"format\": \"long\",\n",
            "        \"noise_label\": \"predicted\",\n",
            "        \"r2_test\": float(r2),\n",
            "        \"rmse_test\": float(rmse),\n",
            "    })\n",
            "\n",
            "nn_results = pd.DataFrame(nn_rows)\n",
            "display(nn_results)\n",
        ],
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "cache_dir = Path(\"figures/cache\")\n",
            "pq_path, json_path, comp_path = export_artifacts(results, comparisons, cache_dir)\n",
            "print(f\"Wrote {pq_path}\")\n",
            "print(f\"Wrote {comp_path}\")\n",
            "print(f\"Wrote {json_path}\")\n",
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

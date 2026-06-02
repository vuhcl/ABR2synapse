#!/usr/bin/env python3
"""Generate Section 5.3 HP tuning and synthesis CV notebooks."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _nb(cells):
    return {
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


def _md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code(source: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def build_hp_nb():
    return _nb(
        [
            _md(
                "# Stage 2 HP tuning (scenarios A & B)\n\n"
                "Tunes T5 wide RF/XGB + MLP on conservative non-test pools; "
                "assembles scenario **C** from 5.2 exports.\n"
            ),
            _code(
                "from pathlib import Path\n"
                "import json\n"
                "import pandas as pd\n\n"
                "from utils.nn_stage2_data import load_nn_stage2_data\n"
                "from utils.stage2_export import export_stage2_data\n"
                "from utils.stage2_hp import resolve_torch_device, tune_scenario_ab, write_scenario_hp_json\n"
                "from utils.benchmark_metrics import STAGE2_BEST_HP_DIR, STAGE2_DATA_DIR\n\n"
                "HP_DIR = STAGE2_BEST_HP_DIR\n"
                "HP_DIR.mkdir(parents=True, exist_ok=True)\n"
                "device = resolve_torch_device()\n"
                "print('device:', device)\n"
            ),
            _code(
                "# Idempotent export snapshot for HP notebook\n"
                "if not (STAGE2_DATA_DIR / 'manifest.json').is_file():\n"
                "    export_stage2_data(out_dir=STAGE2_DATA_DIR)\n"
                "else:\n"
                "    print('stage2_data export present; skip')\n"
            ),
            _code(
                "data = load_nn_stage2_data()\n"
                "trials = []\n"
                "for scen in ('A', 'B'):\n"
                "    payload, mlp_log = tune_scenario_ab(data, scen, device=device, verbose=True)\n"
                "    write_scenario_hp_json(payload, HP_DIR / f'{scen}.json')\n"
                "    trials.append(mlp_log)\n"
                "pd.concat(trials, ignore_index=True).to_parquet(HP_DIR / 'hp_search_log.parquet', index=False)\n"
            ),
            _code(
                "# Scenario C: merge T5 sklearn + Colab MLP (run scripts/export_liberman_t5_hp.py + merge if missing)\n"
                "import subprocess\n"
                "t5 = HP_DIR / 'liberman_t5_sklearn.json'\n"
                "if not t5.is_file():\n"
                "    subprocess.run(['python', 'scripts/export_liberman_t5_hp.py'], check=True, cwd='.')\n"
                "c_out = HP_DIR / 'C.json'\n"
                "if not c_out.is_file():\n"
                "    subprocess.run(['python', 'scripts/merge_liberman_hp.py'], check=True, cwd='.')\n"
                "assert c_out.is_file(), 'C.json missing'\n"
                "print('C.json OK')\n"
            ),
        ]
    )


def build_cv_nb():
    return _nb(
        [
            _md(
                "# Stage 2 synthesis CV (Section 5.3)\n\n"
                "Act IV synthesis CV; separate cache for 5-fold vs 10-fold.\n"
            ),
            _code(
                "import json\n"
                "from pathlib import Path\n\n"
                "import pandas as pd\n\n"
                "from utils.benchmark_metrics import (\n"
                "    LIBERMAN_GROUP_MEAN_BASELINE_PARQUET,\n"
                "    STAGE2_BEST_HP_DIR,\n"
                "    stage2_synthesis_cv_paths,\n"
                ")\n"
                "from utils.nn_stage2_data import load_nn_stage2_data\n"
                "from utils.stage2_hp import resolve_torch_device\n"
                "from utils.stage2_synthesis_cv import run_synthesis_cv\n"
                "from utils.stage2_synthesis_plot import synthesis_act4_grid_cv, synthesis_cohort_panels_cv\n"
                "from utils.subject_cv import eval_stratum_mean_baseline_cv\n\n"
                "device = resolve_torch_device()\n"
                "print('device:', device)\n"
            ),
            _code(
                "hp = {}\n"
                "for scen in ('A', 'B', 'C'):\n"
                "    p = STAGE2_BEST_HP_DIR / f'{scen}.json'\n"
                "    hp[scen] = json.loads(p.read_text())\n"
                "SYNTHESIS_RUNS = [(10, False), (5, True)]\n"
                "summaries = {}\n"
                "for n_folds, force_rerun in SYNTHESIS_RUNS:\n"
                "    _, summary = run_synthesis_cv(\n"
                "        hp, device=device, n_splits=n_folds, force_rerun=force_rerun, verbose=True\n"
                "    )\n"
                "    summaries[n_folds] = summary\n"
                "summary_10, summary_5 = summaries[10], summaries[5]\n"
                "summary_10\n"
            ),
            _code(
                "# Ceiling sanity: Liberman vs cached baseline; Brad inline recompute\n"
                "STRATUM_COLS = ['noise_cat', 'frequency']\n"
                "data = load_nn_stage2_data()\n"
                "lib_cv = eval_stratum_mean_baseline_cv(\n"
                "    data.reformatted_orig.reset_index(drop=True), STRATUM_COLS\n"
                ")\n"
                "lib_ref = pd.read_parquet(LIBERMAN_GROUP_MEAN_BASELINE_PARQUET)\n"
                "print('Liberman CV rmse', lib_cv['rmse'], '±', lib_cv['rmse_sem'])\n"
                "print('Liberman parquet rmse', float(lib_ref['rmse'].iloc[0]), '±', float(lib_ref['rmse_sem'].iloc[0]))\n"
                "brad_cv = eval_stratum_mean_baseline_cv(\n"
                "    data.reformatted.reset_index(drop=True), STRATUM_COLS\n"
                ")\n"
                "print('Brad inline CV rmse', brad_cv['rmse'], '±', brad_cv['rmse_sem'])\n"
                "for n_folds, summary in summaries.items():\n"
                "    print(f'\\n=== {n_folds}-fold ceiling ===')\n"
                "    print(summary.loc[summary['model'].eq('ceiling')])\n"
            ),
            _code(
                "for n_folds, summary in summaries.items():\n"
                "    png = synthesis_act4_grid_cv(summary, n_folds=n_folds)\n"
                "    print(f'{n_folds}-fold saved', png)\n"
            ),
            _code(
                "png = synthesis_cohort_panels_cv(summary_10)\n"
                "svg = png.with_suffix('.svg')\n"
                "print('10-fold cohort panels saved', png, '+', svg)\n"
            ),
            _code(
                "# Act IV-style digest (mean ± SEM)\n"
                "for ts in ('Brad', 'Liberman'):\n"
                "    for scen in ('A', 'B', 'C'):\n"
                "        sub = summary.loc[summary['test_set'].eq(ts) & summary['scenario'].eq(scen)]\n"
                "        print(f'\\n=== {ts} test | train {scen} ===')\n"
                "        print(sub[['model','RMSE','RMSE_sem','source']].to_string(index=False))\n"
            ),
        ]
    )


def main():
    (ROOT / "abr_stage2_hp_tuning.ipynb").write_text(
        json.dumps(build_hp_nb(), indent=1) + "\n", encoding="utf-8"
    )
    (ROOT / "abr_stage2_synthesis_cv.ipynb").write_text(
        json.dumps(build_cv_nb(), indent=1) + "\n", encoding="utf-8"
    )
    print("Wrote abr_stage2_hp_tuning.ipynb and abr_stage2_synthesis_cv.ipynb")


if __name__ == "__main__":
    main()

"""Smoke: one pooled fold (display=3, MLP) + report table shape."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.benchmark_metrics import STAGE2_BEST_HP_DIR, STAGE2_SYNTHESIS_CV_FOLDS_PARQUET
from utils.liberman_classical import liberman_feature_lists, stage2_feature_lists
from utils.nn_stage2_data import load_nn_stage2_data, splits_for_long_stage2
from utils.stage2_hp import resolve_torch_device
from utils.stage2_sklearn import metrics_from_animal_frequency_agg
from utils.stage2_synthesis_cv import (
    _attach_global_stage1,
    _mlp_eval_agg,
    _mlp_holdout_seed,
    _pooled_rmse_from_aggs,
    build_cv_scenario_frames,
    fit_global_stage1_for_scenario,
    generate_cv_folds,
    panel_c_rmse_report_table,
    summarize_pooled_folds,
    train_scenarios_for_display,
)


def _load_hp() -> dict:
    hp = {}
    for scen in ("A", "B", "C"):
        hp[scen] = json.loads((STAGE2_BEST_HP_DIR / f"{scen}.json").read_text())
    return hp


def smoke_one_fold_mlp() -> float:
    hp = _load_hp()
    device = resolve_torch_device()
    data = load_nn_stage2_data()
    splits = splits_for_long_stage2(data)
    global_s1 = {
        s: fit_global_stage1_for_scenario(s, data, splits, verbose=False)
        for s in ("A", "B", "C")
    }
    bb_wide = data.reformatted.reset_index(drop=True)
    lib_wide = data.reformatted_orig.reset_index(drop=True)
    _, _, tr_anim_bb, te_anim_bb = generate_cv_folds(bb_wide)[0]
    _, _, tr_anim_lib, te_anim_lib = generate_cv_folds(lib_wide)[0]

    feats = liberman_feature_lists(data.reformatted_orig, data.common_cols)
    _, _, _, long_num, long_cat = stage2_feature_lists(feats, noise_label="predicted")
    long_log = list(feats["long_log"])

    display = 3
    scen_map = train_scenarios_for_display(display)
    aggs = []
    fold_i = 0
    for eval_cohort, scen in scen_map.items():
        tr_anim = tr_anim_bb if eval_cohort == "Brad" else tr_anim_lib
        te_anim = te_anim_bb if eval_cohort == "Brad" else te_anim_lib
        w_tr, w_ev, l_tr, l_ev, _, _ = build_cv_scenario_frames(
            eval_cohort, scen, tr_anim, te_anim, data, splits  # type: ignore[arg-type]
        )
        w_aug_tr, w_aug_ev, l_aug_tr, l_aug_ev = _attach_global_stage1(
            w_tr, w_ev, l_tr, l_ev, global_s1[scen]["animal_preds"]
        )
        aggs.append(
            _mlp_eval_agg(
                l_aug_tr,
                l_aug_ev,
                long_num,
                long_cat,
                long_log,
                hp[scen]["MLP"],
                holdout_seed=_mlp_holdout_seed(eval_cohort, scen, fold_i),
                device=device,
            )
        )

    rmse_pool = _pooled_rmse_from_aggs(*aggs)
    assert np.isfinite(rmse_pool), rmse_pool

    cohort_rmses = [metrics_from_animal_frequency_agg(a)[1] for a in aggs]
    mean_cohorts = float(np.mean(cohort_rmses))
    assert abs(rmse_pool - mean_cohorts) > 1e-6, (
        f"pooled {rmse_pool} should differ from mean cohorts {mean_cohorts}"
    )
    print(f"fold0 display=3 MLP pooled_rmse={rmse_pool:.4f} (≠ mean cohorts)")
    return rmse_pool


def smoke_report_table_shape() -> None:
    folds = pd.DataFrame(
        [
            {"fold": 0, "display": d, "model": m, "rmse": 1.0 + 0.1 * d, "source": "nn"}
            for d in (1, 2, 3)
            for m in ("L7", "RF", "XGB", "MLP")
        ]
        + [{"fold": 0, "display": 0, "model": "ceiling", "rmse": 2.0, "source": "ceiling"}]
    )
    summary = summarize_pooled_folds(folds)
    tbl = panel_c_rmse_report_table(summary)
    assert len(tbl) == 5, tbl
    assert list(tbl.columns) == [
        "model",
        "Within-cohort",
        "Cross-cohort",
        "Combined",
    ]
    print("panel_c_rmse_report_table shape OK")


def smoke_fold_cache_present() -> None:
    cache = Path(STAGE2_SYNTHESIS_CV_FOLDS_PARQUET)
    if not cache.is_file():
        print("skip: no per-cohort fold cache for parity check")
        return
    cached = pd.read_parquet(cache)
    print(f"fold cache rows={len(cached)} (parity vs refactored scorers: run full CV)")


if __name__ == "__main__":
    smoke_one_fold_mlp()
    smoke_report_table_shape()
    smoke_fold_cache_present()
    print("smoke_pooled_synthesis_panel_c OK")

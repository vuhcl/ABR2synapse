#!/usr/bin/env python3
"""Smoke test: subject-level CV has no leakage; tuning uses Train-only rows."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from utils.nn_stage2_data import load_nn_stage2_data, splits_for_long_stage2
from utils.subject_cv import (
    DEFAULT_CEILING_STRATUM_COLS,
    EXPERIMENTAL_GROUP_COL,
    assert_no_group_leakage,
    build_subject_cv,
    eval_stratum_mean_baseline_cv,
    fit_subject_search,
    prepare_strat_groups,
    prepare_xy_groups,
    train_only_frame,
)
from utils.stage2_sklearn import RF_PARAMS


def main() -> None:
    data = load_nn_stage2_data()
    sp = splits_for_long_stage2(data)
    long_tr = sp["bb_long_train"]
    long_num = data.long_num
    long_cat = data.long_cat
    long_log = data.long_log
    feat = [c for c in long_num + long_cat + long_log if c in long_tr.columns]
    assert feat, "no feature columns found on long_train"

    X, y, groups, _ = prepare_xy_groups(long_tr, feat, "synapses")
    assert len(train_only_frame(long_tr)) <= len(long_tr)
    if "DataGroup" in long_tr.columns:
        assert (long_tr.loc[X.index, "DataGroup"] == "Train").all()

    cv = build_subject_cv(y.to_numpy(), groups, prefer_stratified=False)
    assert_no_group_leakage(cv, X, y, groups)

    wide = sp["bb_wide_train"]
    if EXPERIMENTAL_GROUP_COL in wide.columns:
        w = wide.reset_index(drop=True)
        g_exp, strat_exp, _ = prepare_strat_groups(w, EXPERIMENTAL_GROUP_COL)
        cv_s = build_subject_cv(
            strat_exp, g_exp, n_splits=3, random_state=1, prefer_stratified=True
        )
        assert type(cv_s).__name__ == "StratifiedGroupKFold"
        assert_no_group_leakage(
            cv_s, w[[EXPERIMENTAL_GROUP_COL]], pd.Series(strat_exp), g_exp
        )

    if EXPERIMENTAL_GROUP_COL in wide.columns:
        out = eval_stratum_mean_baseline_cv(
            wide, list(DEFAULT_CEILING_STRATUM_COLS), n_splits=3, random_state=1
        )
        assert len(out["pred_oof"]) == len(wide)
        assert out["n_cv_folds"] == 3

    pipe = Pipeline([("rf", RandomForestRegressor(n_estimators=5, random_state=1))])
    search = fit_subject_search(
        pipe,
        {"rf__max_depth": [3, 5]},
        long_tr,
        feat,
        "synapses",
        prefer_stratified=False,
        n_iter=2,
        n_jobs=1,
    )
    n_fit = len(prepare_xy_groups(long_tr, feat, "synapses")[0])
    assert search.n_features_in_ == len(feat)
    print(f"OK: subject CV smoke passed ({n_fit} Train-only rows, best_depth={search.best_params_})")


if __name__ == "__main__":
    main()

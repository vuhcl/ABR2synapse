#!/usr/bin/env python3
"""Smoke one MLP SHAP fold (GradientExplainer + synthesis-aligned training)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.benchmark_metrics import STAGE2_SYNTHESIS_CV_FOLDS_PARQUET
from utils.liberman_classical import liberman_feature_lists, stage2_feature_lists
from utils.nn_stage2_data import load_nn_stage2_data, splits_for_long_stage2
from utils.stage2_hp import resolve_torch_device
from utils.stage2_mlp_shap import (
    _EXPECTED_N_FEATURES,
    compute_fold_shap,
    fit_mlp_fold,
    mlp_hp_for_scenario,
)
from utils.stage2_synthesis_cv import (
    _mlp_holdout_seed,
    attach_global_stage1,
    build_cv_scenario_frames,
    fit_global_stage1_for_scenario,
    generate_cv_folds,
)

COHORT_MAP = {
    "buran": ("Brad", "A"),
    "liberman": ("Liberman", "C"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", choices=COHORT_MAP.keys(), default="liberman")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--rmse-tol", type=float, default=1e-3)
    p.add_argument(
        "--strict-rmse",
        action="store_true",
        help="Exit non-zero if synthesis CV fold RMSE differs (cache may be stale).",
    )
    args = p.parse_args()

    eval_cohort, scen = COHORT_MAP[args.cohort]
    hp = mlp_hp_for_scenario(scen)
    device = resolve_torch_device()

    data = load_nn_stage2_data()
    splits = splits_for_long_stage2(data)
    feats = liberman_feature_lists(data.reformatted_orig, data.common_cols)
    _syn_num, _syn_log, _syn_cat, long_num, long_cat = stage2_feature_lists(
        feats, noise_label="predicted"
    )
    long_log = list(feats["long_log"])

    gs1 = fit_global_stage1_for_scenario(scen, data, splits, verbose=False)
    wide_all = (
        data.reformatted.reset_index(drop=True)
        if eval_cohort == "Brad"
        else data.reformatted_orig.reset_index(drop=True)
    )
    folds = generate_cv_folds(wide_all, n_splits=args.n_splits, random_state=22)
    _tr_idx, _te_idx, tr_anim, te_anim = folds[args.fold]

    _w_tr, _w_ev, l_tr, l_ev, _, _ = build_cv_scenario_frames(
        eval_cohort, scen, tr_anim, te_anim, data, splits
    )
    _w_tr, _w_ev, l_tr, l_ev = attach_global_stage1(
        _w_tr, _w_ev, l_tr, l_ev, gs1["animal_preds"]
    )

    holdout_seed = _mlp_holdout_seed(eval_cohort, scen, args.fold)
    art = fit_mlp_fold(
        l_tr, l_ev, long_num, long_cat, long_log, hp,
        holdout_seed=holdout_seed, device=device,
    )
    shap_df, feat_df = compute_fold_shap(art, holdout_seed=holdout_seed, shap_batch_size=32)

    assert shap_df.shape[1] == _EXPECTED_N_FEATURES == 13, shap_df.shape
    assert len(shap_df) == len(art.meta_eval)
    assert np.isfinite(shap_df.values).all()

    n_af = art.meta_eval.groupby(["animal_id", "frequency"]).ngroups
    print(
        f"OK cohort={args.cohort} fold={args.fold} "
        f"long_rows={len(shap_df)} animal_freq_pairs={n_af} "
        f"rmse_af={art.rmse_animal_freq:.4f}"
    )

    if STAGE2_SYNTHESIS_CV_FOLDS_PARQUET.is_file():
        folds_df = pd.read_parquet(STAGE2_SYNTHESIS_CV_FOLDS_PARQUET)
        row = folds_df.loc[
            folds_df["eval_cohort"].eq(eval_cohort)
            & folds_df["train_scenario"].eq(scen)
            & folds_df["model"].eq("MLP")
            & folds_df["fold"].eq(args.fold)
        ]
        if len(row) == 1:
            ref = float(row["rmse"].iloc[0])
            diff = abs(art.rmse_animal_freq - ref)
            print(f"synthesis_cv rmse={ref:.4f} diff={diff:.6f}")
            if diff > args.rmse_tol:
                msg = (
                    f"RMSE mismatch vs synthesis folds: {art.rmse_animal_freq} vs {ref} "
                    f"(diff={diff:.4f}; cache may be stale — re-run synthesis CV)"
                )
                if args.strict_rmse:
                    raise SystemExit(msg)
                print("warn:", msg)
        else:
            print("warn: no matching synthesis CV fold row; skip RMSE check")
    else:
        print("warn: synthesis folds parquet missing; skip RMSE check")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run MLP SHAP pipeline for one cohort (for batch / smoke extension)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.benchmark_metrics import apply_slide_rcparams
from utils.liberman_classical import liberman_feature_lists, stage2_feature_lists
from utils.nn_stage2_data import load_nn_stage2_data, splits_for_long_stage2
from utils.stage2_hp import resolve_torch_device
from utils.stage2_mlp_shap import (
    build_long_feature_annotation_table,
    compute_fold_shap,
    fit_mlp_fold,
    fold_cache_path,
    load_fold_cache,
    mlp_hp_for_scenario,
    run_mlp_shap_plots,
    save_fold_cache,
)
from utils.stage2_synthesis_cv import (
    _mlp_holdout_seed,
    attach_global_stage1,
    build_cv_scenario_frames,
    fit_global_stage1_for_scenario,
    generate_cv_folds,
)
from utils.subject_cv import DEFAULT_CEILING_CV_RANDOM_STATE

COHORT_MAP = {"buran": ("Brad", "A"), "liberman": ("Liberman", "C")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", choices=COHORT_MAP.keys(), required=True)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    args = p.parse_args()

    cohort_slug = args.cohort
    eval_cohort, scen = COHORT_MAP[cohort_slug]
    cache_dir = Path("figures/cache/shap/mlp")
    fig_dir = Path("figures/shap/mlp")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    apply_slide_rcparams()
    device = resolve_torch_device()
    hp = mlp_hp_for_scenario(scen)

    data = load_nn_stage2_data()
    splits = splits_for_long_stage2(data)
    feats = liberman_feature_lists(data.reformatted_orig, data.common_cols)
    _, _, _, long_num, long_cat = stage2_feature_lists(feats, noise_label="predicted")
    long_log = list(feats["long_log"])

    build_long_feature_annotation_table(long_num + list(long_cat) + long_log).to_parquet(
        cache_dir / "feature_names.parquet", index=False
    )

    global_s1 = {
        s: fit_global_stage1_for_scenario(s, data, splits, verbose=False)
        for s in ("A", "B", "C")
    }
    wide_all = (
        data.reformatted.reset_index(drop=True)
        if eval_cohort == "Brad"
        else data.reformatted_orig.reset_index(drop=True)
    )
    folds = generate_cv_folds(
        wide_all, n_splits=args.n_splits, random_state=DEFAULT_CEILING_CV_RANDOM_STATE
    )
    shap_rows, feat_rows, meta_rows = [], [], []
    n_folds = len(folds) if args.max_folds is None else min(args.max_folds, len(folds))

    for fold_id in range(n_folds):
        _tr_idx, _te_idx, tr_anim, te_anim = folds[fold_id]
        cache_p = fold_cache_path(cache_dir, cohort_slug, fold_id)
        if cache_p.is_file() and not args.force_rerun:
            shap_df, feat_df, meta_df = load_fold_cache(cache_p)
        else:
            w_tr, w_ev, l_tr, l_ev, _, _ = build_cv_scenario_frames(
                eval_cohort, scen, tr_anim, te_anim, data, splits
            )
            w_tr, w_ev, l_tr, l_ev = attach_global_stage1(
                w_tr, w_ev, l_tr, l_ev, global_s1[scen]["animal_preds"]
            )
            holdout_seed = _mlp_holdout_seed(eval_cohort, scen, fold_id)
            art = fit_mlp_fold(
                l_tr, l_ev, long_num, long_cat, long_log, hp,
                holdout_seed=holdout_seed, device=device,
            )
            shap_df, feat_df = compute_fold_shap(art, holdout_seed=holdout_seed)
            meta_df = art.meta_eval.copy()
            meta_df["fold_id"] = fold_id
            meta_df["cohort"] = cohort_slug
            save_fold_cache(cache_p, shap_df=shap_df, feat_df=feat_df, meta_df=meta_df)
            print(f"fold {fold_id} rmse_af={art.rmse_animal_freq:.4f} rows={len(shap_df)}")
        shap_rows.append(shap_df)
        feat_rows.append(feat_df)
        meta_rows.append(meta_df)

    shap_oof = pd.concat(shap_rows, ignore_index=True)
    feat_oof = pd.concat(feat_rows, ignore_index=True)
    meta_oof = pd.concat(meta_rows, ignore_index=True)

    shap_renamed = shap_oof.rename(
        columns={"frequency": "frequency_feature", "level": "level_feature"}
    )
    shap_export = pd.concat(
        [meta_oof[["animal_id", "frequency", "level"]], shap_renamed],
        axis=1,
    )
    shap_export.to_parquet(cache_dir / f"shap_values_{cohort_slug}.parquet", index=False)
    meta_oof.to_parquet(cache_dir / f"shap_metadata_{cohort_slug}.parquet", index=False)

    if not args.skip_plots:
        run_mlp_shap_plots(shap_oof, feat_oof, meta_oof, cohort_slug, fig_dir)

    print("done", cohort_slug, "rows", len(meta_oof))


if __name__ == "__main__":
    main()

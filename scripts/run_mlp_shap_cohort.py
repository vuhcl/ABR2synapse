#!/usr/bin/env python3
"""Run MLP SHAP pipeline for one or both cohorts (for batch / smoke extension)."""
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
    COHORT_SLUGS_DECK,
    MLP_SHAP_CACHE_DIR,
    MLP_SHAP_FIG_DIR,
    aggregate_oof_long_grain,
    build_long_feature_annotation_table,
    compute_fold_shap,
    export_feature_names_mlp,
    export_mlp_shap_parquet,
    fit_mlp_fold,
    fold_cache_path,
    load_cohort_outputs_from_cache,
    load_fold_cache,
    mlp_hp_for_scenario,
    run_mlp_shap_deck_figures,
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


def run_one_cohort(
    cohort_slug: str,
    *,
    cache_dir: Path,
    n_splits: int,
    max_folds: int | None,
    force_rerun: bool,
    max_background: int | None,
    device,
    data,
    splits,
    long_num,
    long_cat,
    long_log,
    global_s1,
) -> dict[str, pd.DataFrame]:
    eval_cohort, scen = COHORT_MAP[cohort_slug]
    hp = mlp_hp_for_scenario(scen)
    wide_all = (
        data.reformatted.reset_index(drop=True)
        if eval_cohort == "Brad"
        else data.reformatted_orig.reset_index(drop=True)
    )
    folds = generate_cv_folds(
        wide_all, n_splits=n_splits, random_state=DEFAULT_CEILING_CV_RANDOM_STATE
    )
    shap_rows, feat_rows, meta_rows = [], [], []
    n_folds = len(folds) if max_folds is None else min(max_folds, len(folds))

    for fold_id in range(n_folds):
        _tr_idx, _te_idx, tr_anim, te_anim = folds[fold_id]
        cache_p = fold_cache_path(cache_dir, cohort_slug, fold_id)
        if cache_p.is_file() and not force_rerun:
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
                l_tr,
                l_ev,
                long_num,
                long_cat,
                long_log,
                hp,
                holdout_seed=holdout_seed,
                device=device,
            )
            shap_df, feat_df = compute_fold_shap(
                art,
                holdout_seed=holdout_seed,
                max_background=max_background,
            )
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
    shap_oof, feat_oof, meta_oof = aggregate_oof_long_grain(shap_oof, feat_oof, meta_oof)
    export_mlp_shap_parquet(cohort_slug, shap_oof, feat_oof, meta_oof, cache_dir)
    return {"shap": shap_oof, "feat": feat_oof, "meta": meta_oof}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", choices=[*COHORT_MAP.keys(), "all"], default="all")
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument(
        "--max-background",
        type=int,
        default=None,
        nargs="?",
        const=None,
        help="Cap training rows for GradientExplainer background (default: all rows).",
    )
    p.add_argument(
        "--deck-figures-only",
        action="store_true",
        help="Skip SHAP computation; load fold caches and write deck figures.",
    )
    p.add_argument("--skip-deck-figures", action="store_true")
    args = p.parse_args()

    cache_dir = MLP_SHAP_CACHE_DIR
    fig_dir = MLP_SHAP_FIG_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    apply_slide_rcparams()

    if args.deck_figures_only:
        cohort_outputs = {
            slug: load_cohort_outputs_from_cache(cache_dir, slug, n_folds=args.n_splits)
            for slug in COHORT_SLUGS_DECK
        }
        paths = run_mlp_shap_deck_figures(cohort_outputs, fig_dir)
        print("deck figures:", len(paths), "files")
        return

    device = resolve_torch_device()
    data = load_nn_stage2_data()
    splits = splits_for_long_stage2(data)
    feats = liberman_feature_lists(data.reformatted_orig, data.common_cols)
    _, _, _, long_num, long_cat = stage2_feature_lists(feats, noise_label="predicted")
    long_log = list(feats["long_log"])
    feature_names = long_num + list(long_cat) + long_log
    export_feature_names_mlp(feature_names, cache_dir)

    global_s1 = {
        s: fit_global_stage1_for_scenario(s, data, splits, verbose=False)
        for s in ("A", "B", "C")
    }

    slugs = list(COHORT_MAP.keys()) if args.cohort == "all" else [args.cohort]
    cohort_outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for cohort_slug in slugs:
        cohort_outputs[cohort_slug] = run_one_cohort(
            cohort_slug,
            cache_dir=cache_dir,
            n_splits=args.n_splits,
            max_folds=args.max_folds,
            force_rerun=args.force_rerun,
            max_background=args.max_background,
            device=device,
            data=data,
            splits=splits,
            long_num=long_num,
            long_cat=long_cat,
            long_log=long_log,
            global_s1=global_s1,
        )
        print("exported", cohort_slug, "rows", len(cohort_outputs[cohort_slug]["meta"]))

    if not args.skip_deck_figures and len(cohort_outputs) == len(COHORT_SLUGS_DECK):
        run_mlp_shap_deck_figures(cohort_outputs, fig_dir)
        print("deck figures written to", fig_dir)


if __name__ == "__main__":
    main()

"""
10-fold synthesis CV for Section 5.3 (Act IV grid on unified CV protocol).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Set, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from utils.liberman_classical import (
    liberman_feature_lists,
    ols_wide_amp80,
    resolve_amp80_column,
    stage2_feature_lists,
)
from utils.nn_colab_train import aggregate_animal_freq_metrics
from utils.nn_stage2 import (
    _animal_auc,
    _animal_label_accuracy,
    attach_noise_preds_long,
    fit_stage1_wide_best,
    stage1_animal_pred_covering,
)
from utils.nn_stage2_data import NNStage2Data, load_nn_stage2_data, splits_for_long_stage2, wide_stage1_fit, wide_stage1_val
from utils.stage2_mlp_shap import fit_mlp_fold
from utils.stage2_hp import resolve_torch_device
from utils.stage2_sklearn import XGB_N_JOBS, _syn_prep, agg_animal_frequency, metrics_from_animal_frequency_agg
from utils.subject_cv import (
    DEFAULT_CEILING_CV_RANDOM_STATE,
    DEFAULT_CEILING_CV_SPLITS,
    DEFAULT_CEILING_STRATUM_COLS,
    EXPERIMENTAL_GROUP_COL,
    assert_no_group_leakage,
    build_subject_cv,
    prepare_strat_groups,
    train_only_frame,
)

EvalCohort = Literal["Brad", "Liberman"]
TrainScenario = Literal["A", "B", "C"]
ModelId = Literal["L7", "RF", "XGB", "MLP"]

DECK_LABELS: Dict[str, str] = {
    "L7": "LR baseline",
    "RF": "RF",
    "XGB": "XGB",
    "MLP": "MLP",
    "ceiling": "ceiling",
}

CV_SEED = DEFAULT_CEILING_CV_RANDOM_STATE
N_FOLDS = DEFAULT_CEILING_CV_SPLITS

# Bump when fold stratification, ceiling strata, or S1 label universe change.
SYNTHESIS_CV_CACHE_VERSION = 7


def _mlp_holdout_seed(eval_cohort: str, scenario: str, fold: int) -> int:
    key = f"{CV_SEED}|{eval_cohort}|{scenario}|{fold}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def _filter_animals(df: pd.DataFrame, animals: Set, *, include: bool = True) -> pd.DataFrame:
    aid = df["animal_id"].astype(str)
    mask = aid.isin({str(a) for a in animals})
    if not include:
        mask = ~mask
    return df.loc[mask].reset_index(drop=True)


def build_cv_scenario_frames(
    eval_cohort: EvalCohort,
    scenario: TrainScenario,
    fold_train_animals: Set,
    fold_test_animals: Set,
    data: NNStage2Data,
    splits: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Eval-cohort-specific train/eval wide+long frames (5.3 aligned long pools).

    Returns ``wide_train, wide_eval, long_train, long_eval, noise_num, noise_log``.
    """
    bb_wide = data.reformatted.reset_index(drop=True)
    lib_wide = data.reformatted_orig.reset_index(drop=True)
    bb_long_all = splits["bb_long_all"]
    lib_long_all = splits["lib_long_all"]

    bb_long_fold_tr = _filter_animals(bb_long_all, fold_train_animals)
    lib_long_fold_tr = _filter_animals(lib_long_all, fold_train_animals)

    if eval_cohort == "Brad":
        wide_eval = _filter_animals(bb_wide, fold_test_animals)
        long_eval = _filter_animals(bb_long_all, fold_test_animals)
        bb_wide_tr = _filter_animals(bb_wide, fold_train_animals)
        if scenario == "A":
            wide_train, long_train = bb_wide_tr, bb_long_fold_tr
            nn, nl = list(data.noise_num_bb), list(data.noise_log_bb)
        elif scenario == "B":
            wide_train = pd.concat([bb_wide_tr, lib_wide], ignore_index=True)
            long_train = pd.concat([bb_long_fold_tr, lib_long_all], ignore_index=True)
            nn, nl = list(data.noise_num_common), list(data.noise_log_common)
        else:
            wide_train, long_train = lib_wide, lib_long_all
            nn, nl = list(data.noise_num_lib), list(data.noise_log_lib)
    else:
        wide_eval = _filter_animals(lib_wide, fold_test_animals)
        long_eval = _filter_animals(lib_long_all, fold_test_animals)
        lib_wide_tr = _filter_animals(lib_wide, fold_train_animals)
        if scenario == "A":
            wide_train, long_train = bb_wide, bb_long_all
            nn, nl = list(data.noise_num_bb), list(data.noise_log_bb)
        elif scenario == "B":
            wide_train = pd.concat([bb_wide, lib_wide_tr], ignore_index=True)
            long_train = pd.concat([bb_long_all, lib_long_fold_tr], ignore_index=True)
            nn, nl = list(data.noise_num_common), list(data.noise_log_common)
        else:
            wide_train, long_train = lib_wide_tr, lib_long_fold_tr
            nn, nl = list(data.noise_num_lib), list(data.noise_log_lib)

    return wide_train, wide_eval, long_train, long_eval, nn, nl


def _scenario_animals_universe(
    scenario: TrainScenario,
    data: NNStage2Data,
) -> Set[str]:
    """Animals that need Stage-1 ``noise_preds`` for synthesis scenario A/B/C."""
    bb = set(data.reformatted["animal_id"].astype(str))
    lib = set(data.reformatted_orig["animal_id"].astype(str))
    # All scenarios: both cohorts. Scenario A trains on Brad-only rows for Brad
    # within-cohort, but Liberman cross-cohort (eval Cohort A / train Brad) still
    # needs S1 labels on Liberman holdout mice in ``build_scenario_full_pools``.
    return bb | lib


def build_scenario_full_pools(
    eval_cohort: EvalCohort,
    scenario: TrainScenario,
    data: NNStage2Data,
    splits: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Wide/long pools covering every animal that appears in fold train or eval.

    Cross-cohort scenarios (e.g. Brad eval + Liberman train) must include eval-cohort
    rows so ``stage1_animal_pred_covering`` assigns ``noise_preds`` on held-out animals.
    """
    bb_wide = data.reformatted.reset_index(drop=True)
    lib_wide = data.reformatted_orig.reset_index(drop=True)
    bb_long_all = splits["bb_long_all"]
    lib_long_all = splits["lib_long_all"]
    universe = _scenario_animals_universe(scenario, data)
    bb_animals = set(bb_wide["animal_id"].astype(str))
    lib_animals = set(lib_wide["animal_id"].astype(str))

    wide_tr, _, long_tr, _, nn, nl = build_cv_scenario_frames(
        eval_cohort, scenario, universe, set(), data, splits
    )
    train_animals = set(wide_tr["animal_id"].astype(str))
    missing = universe - train_animals
    if missing:
        parts_w, parts_l = [wide_tr], [long_tr]
        bb_missing = missing & bb_animals
        lib_missing = missing & lib_animals
        if bb_missing:
            parts_w.append(_filter_animals(bb_wide, bb_missing))
            parts_l.append(_filter_animals(bb_long_all, bb_missing))
        if lib_missing:
            parts_w.append(_filter_animals(lib_wide, lib_missing))
            parts_l.append(_filter_animals(lib_long_all, lib_missing))
        wide_pool = pd.concat(parts_w, ignore_index=True)
        long_pool = pd.concat(parts_l, ignore_index=True)
    else:
        wide_pool, long_pool = wide_tr, long_tr
    return wide_pool, long_pool, nn, nl


def _official_s1_wide_frames(
    scenario: TrainScenario,
    splits: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Official train/test wide frames for global Stage 1 (matches §5.2 training scenarios)."""
    bb_tr = splits["bb_wide_train"]
    bb_te = splits["bb_wide_test"]
    lib_tr = splits["lib_train"]
    lib_te = splits["lib_test"]
    if scenario == "A":
        return bb_tr, bb_te
    if scenario == "B":
        return (
            pd.concat([bb_tr, lib_tr], ignore_index=True),
            pd.concat([bb_te, lib_te], ignore_index=True),
        )
    return lib_tr, lib_te


def _noise_feats_for_scenario(
    scenario: TrainScenario, data: NNStage2Data
) -> Tuple[List[str], List[str]]:
    if scenario == "A":
        return list(data.noise_num_bb), list(data.noise_log_bb)
    if scenario == "B":
        return list(data.noise_num_common), list(data.noise_log_common)
    return list(data.noise_num_lib), list(data.noise_log_lib)


def stage1_test_metrics_by_cohort(
    s1: Dict[str, Any],
    *,
    brad_test_animals: Set[str],
    liberman_test_animals: Set[str],
) -> Dict[str, Dict[str, float | int]]:
    """Animal-level test accuracy/AUC on Brad vs Liberman holdout mice."""
    y = s1["animal_y_te"]
    prob = s1["animal_prob_te"]
    if y is None or len(y) == 0:
        empty = {"test_acc": float("nan"), "test_auc": float("nan"), "n_animals": 0}
        return {"Brad": dict(empty), "Liberman": dict(empty)}
    thr = float(s1["stage1_threshold"])
    pred = (prob > thr).astype(int)

    def _one(cohort_animals: Set[str]) -> Dict[str, float | int]:
        animals = {str(a) for a in cohort_animals}
        idx = y.index.astype(str).isin(animals)
        if not idx.any():
            return {"test_acc": float("nan"), "test_auc": float("nan"), "n_animals": 0}
        yt = y.loc[idx]
        pr = prob.loc[idx]
        pd_pred = pred.loc[idx]
        return {
            "test_acc": _animal_label_accuracy(yt, pd_pred),
            "test_auc": _animal_auc(yt, pr),
            "n_animals": int(idx.sum()),
        }

    return {"Brad": _one(brad_test_animals), "Liberman": _one(liberman_test_animals)}


def fit_global_stage1_for_scenario(
    scenario: TrainScenario,
    data: NNStage2Data,
    splits: Dict[str, pd.DataFrame],
    *,
    eval_cohort: EvalCohort = "Brad",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fit Stage 1 once per training scenario; return animal-level ``noise_preds``.

    ``eval_cohort`` only selects which wide/long pool layout to use when attaching
    labels for synthesis CV (A/B/C frame builder); S1 train/test mice are scenario-defined.
    """
    wide_pool, long_pool, nn, nl = build_scenario_full_pools(
        eval_cohort, scenario, data, splits
    )
    s1_train, s1_test = _official_s1_wide_frames(scenario, splits)
    s1 = fit_stage1_wide_best(
        wide_stage1_fit(s1_train),
        wide_stage1_val(s1_train),
        s1_test,
        nn,
        nl,
        random_state=1,
        verbose=verbose,
    )
    animal_preds = stage1_animal_pred_covering(
        wide_pool, s1, nn, nl, long_pool=long_pool
    )
    if verbose:
        by_c = stage1_test_metrics_by_cohort(
            s1,
            brad_test_animals=set(splits["bb_wide_test"]["animal_id"].astype(str)),
            liberman_test_animals=set(splits["lib_test"]["animal_id"].astype(str)),
        )
        print(
            f"  global S1 scen={scenario} ({s1['stage1_model'].upper()}): "
            f"fit/val {s1_train['animal_id'].nunique()} animals, "
            f"official test {s1_test['animal_id'].nunique()} animals, "
            f"labels for {len(animal_preds)} animals | "
            f"Brad test acc={by_c['Brad']['test_acc']:.3f} AUC={by_c['Brad']['test_auc']:.3f} | "
            f"Lib test acc={by_c['Liberman']['test_acc']:.3f} AUC={by_c['Liberman']['test_auc']:.3f}"
        )
    return {
        "s1": s1,
        "animal_preds": animal_preds,
        "noise_num": nn,
        "noise_log": nl,
        "scenario": scenario,
    }


def build_stage1_classification_table(
    data: NNStage2Data | None = None,
    splits: Dict[str, pd.DataFrame] | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Stage 1 noise-classifier performance by training scenario and test cohort.

    Rows: (1) within-cohort, (2) cross-cohort, (3) combined training.
    Columns: Brad / Liberman test accuracy and AUC (animal-level @ calibrated threshold).
    """
    data = data or load_nn_stage2_data()
    splits = splits or splits_for_long_stage2(data)
    bb_te = splits["bb_wide_test"]
    lib_te = splits["lib_test"]
    bb_tr = splits["bb_wide_train"]
    lib_tr = splits["lib_train"]
    brad_test = set(bb_te["animal_id"].astype(str))
    lib_test = set(lib_te["animal_id"].astype(str))

    def _row(
        scenario: int,
        training_data: str,
        s1: Dict[str, Any],
        *,
        brad_acc: float | None = None,
        brad_auc: float | None = None,
        lib_acc: float | None = None,
        lib_auc: float | None = None,
    ) -> Dict[str, Any]:
        by_c = stage1_test_metrics_by_cohort(
            s1, brad_test_animals=brad_test, liberman_test_animals=lib_test
        )
        return {
            "scenario": scenario,
            "training_data": training_data,
            "stage1_model": s1.get("stage1_model", ""),
            "brad_test_acc": brad_acc if brad_acc is not None else by_c["Brad"]["test_acc"],
            "brad_test_auc": brad_auc if brad_auc is not None else by_c["Brad"]["test_auc"],
            "brad_n_test": by_c["Brad"]["n_animals"],
            "liberman_test_acc": lib_acc
            if lib_acc is not None
            else by_c["Liberman"]["test_acc"],
            "liberman_test_auc": lib_auc
            if lib_auc is not None
            else by_c["Liberman"]["test_auc"],
            "liberman_n_test": by_c["Liberman"]["n_animals"],
        }

    rows: List[Dict[str, Any]] = []

    if verbose:
        print("\n=== Stage 1 table: within-cohort (Brad) ===")
    nn_bb, nl_bb = _noise_feats_for_scenario("A", data)
    s1_bb = fit_stage1_wide_best(
        wide_stage1_fit(bb_tr),
        wide_stage1_val(bb_tr),
        bb_te,
        nn_bb,
        nl_bb,
        random_state=1,
        verbose=verbose,
    )
    if verbose:
        print("\n=== Stage 1 table: within-cohort (Liberman) ===")
    nn_lib, nl_lib = _noise_feats_for_scenario("C", data)
    s1_lib = fit_stage1_wide_best(
        wide_stage1_fit(lib_tr),
        wide_stage1_val(lib_tr),
        lib_te,
        nn_lib,
        nl_lib,
        random_state=1,
        verbose=verbose,
    )
    by_bb = stage1_test_metrics_by_cohort(
        s1_bb, brad_test_animals=brad_test, liberman_test_animals=lib_test
    )
    by_lib = stage1_test_metrics_by_cohort(
        s1_lib, brad_test_animals=brad_test, liberman_test_animals=lib_test
    )
    rows.append(
        {
            "scenario": 1,
            "training_data": "Within-cohort",
            "stage1_model": f"Brad:{s1_bb['stage1_model']}; Lib:{s1_lib['stage1_model']}",
            "brad_test_acc": by_bb["Brad"]["test_acc"],
            "brad_test_auc": by_bb["Brad"]["test_auc"],
            "brad_n_test": by_bb["Brad"]["n_animals"],
            "liberman_test_acc": by_lib["Liberman"]["test_acc"],
            "liberman_test_auc": by_lib["Liberman"]["test_auc"],
            "liberman_n_test": by_lib["Liberman"]["n_animals"],
        }
    )

    if verbose:
        print("\n=== Stage 1 table: cross-cohort (train Brad → test Liberman) ===")
    s1_bb_on_lib = fit_stage1_wide_best(
        wide_stage1_fit(bb_tr),
        wide_stage1_val(bb_tr),
        lib_te,
        nn_bb,
        nl_bb,
        random_state=1,
        verbose=verbose,
    )
    if verbose:
        print("\n=== Stage 1 table: cross-cohort (train Liberman → test Brad) ===")
    s1_lib_on_bb = fit_stage1_wide_best(
        wide_stage1_fit(lib_tr),
        wide_stage1_val(lib_tr),
        bb_te,
        nn_lib,
        nl_lib,
        random_state=1,
        verbose=verbose,
    )
    by_cross_bb = stage1_test_metrics_by_cohort(
        s1_bb_on_lib, brad_test_animals=brad_test, liberman_test_animals=lib_test
    )
    by_cross_lib = stage1_test_metrics_by_cohort(
        s1_lib_on_bb, brad_test_animals=brad_test, liberman_test_animals=lib_test
    )
    rows.append(
        {
            "scenario": 2,
            "training_data": "Cross-cohort",
            "stage1_model": f"Brad→Lib:{s1_bb_on_lib['stage1_model']}; Lib→Brad:{s1_lib_on_bb['stage1_model']}",
            "brad_test_acc": by_cross_lib["Brad"]["test_acc"],
            "brad_test_auc": by_cross_lib["Brad"]["test_auc"],
            "brad_n_test": by_cross_lib["Brad"]["n_animals"],
            "liberman_test_acc": by_cross_bb["Liberman"]["test_acc"],
            "liberman_test_auc": by_cross_bb["Liberman"]["test_auc"],
            "liberman_n_test": by_cross_bb["Liberman"]["n_animals"],
        }
    )

    if verbose:
        print("\n=== Stage 1 table: combined training ===")
    nn_both, nl_both = _noise_feats_for_scenario("B", data)
    both_tr = pd.concat([bb_tr, lib_tr], ignore_index=True)
    both_te = pd.concat([bb_te, lib_te], ignore_index=True)
    s1_both = fit_stage1_wide_best(
        wide_stage1_fit(both_tr),
        wide_stage1_val(both_tr),
        both_te,
        nn_both,
        nl_both,
        random_state=1,
        verbose=verbose,
    )
    rows.append(_row(3, "Combined", s1_both))

    return pd.DataFrame(rows)


def _attach_global_stage1(
    wide_train: pd.DataFrame,
    wide_eval: pd.DataFrame,
    long_train: pd.DataFrame,
    long_eval: pd.DataFrame,
    animal_preds: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map precomputed global Stage-1 labels onto fold train/eval frames."""
    w_tr = attach_noise_preds_long(
        wide_train,
        animal_preds,
        fallback_noise_cat=False,
        require_full_coverage=True,
    )
    w_ev = attach_noise_preds_long(
        wide_eval, animal_preds, fallback_noise_cat=False
    )
    l_tr = attach_noise_preds_long(
        long_train,
        animal_preds,
        fallback_noise_cat=False,
        require_full_coverage=True,
    )
    l_ev = attach_noise_preds_long(long_eval, animal_preds, fallback_noise_cat=False)
    return w_tr, w_ev, l_tr, l_ev


attach_global_stage1 = _attach_global_stage1


def generate_cv_folds(
    wide_df: pd.DataFrame,
    *,
    n_splits: int = N_FOLDS,
    random_state: int = CV_SEED,
) -> List[Tuple[np.ndarray, np.ndarray, Set, Set]]:
    """Return list of (train_idx, test_idx, train_animals, test_animals) on wide_df."""
    work = wide_df.reset_index(drop=True)
    groups, strat, _ = prepare_strat_groups(work, EXPERIMENTAL_GROUP_COL, "animal_id")
    cv = build_subject_cv(
        strat, groups, n_splits=n_splits, random_state=random_state, prefer_stratified=True
    )
    assert_no_group_leakage(
        cv, work[[EXPERIMENTAL_GROUP_COL, "frequency"]], pd.Series(strat), groups
    )
    split_x = np.zeros(len(work))
    folds: List[Tuple[np.ndarray, np.ndarray, Set, Set]] = []
    for tr_idx, te_idx in cv.split(split_x, strat, groups):
        tr_anim = set(work.iloc[tr_idx]["animal_id"].astype(str))
        te_anim = set(work.iloc[te_idx]["animal_id"].astype(str))
        folds.append((tr_idx, te_idx, tr_anim, te_anim))
    return folds


# Display bucket 1/2/3 → cohort-specific train scenario letters (within / cross / combined).
DISPLAY_TRAIN_SCENARIOS: Dict[int, Dict[str, str]] = {
    1: {"Brad": "A", "Liberman": "C"},
    2: {"Brad": "C", "Liberman": "A"},
    3: {"Brad": "B", "Liberman": "B"},
}
# Pooled summary ``scenario`` column (A/B/C = within / cross / combined pooled).
POOLED_SCENARIO_BY_DISPLAY: Dict[int, str] = {1: "A", 2: "B", 3: "C"}
POOLED_DISPLAY_LABELS: Dict[int, str] = {
    1: "Within-cohort",
    2: "Cross-cohort",
    3: "Combined",
}


def train_scenarios_for_display(display: int) -> Dict[str, str]:
    """Brad/Liberman train slice letters for a display bucket (1–3)."""
    if display not in DISPLAY_TRAIN_SCENARIOS:
        raise ValueError(f"display must be 1, 2, or 3; got {display}")
    return dict(DISPLAY_TRAIN_SCENARIOS[display])


def pooled_summary_scenario(display: int) -> str:
    """Summary ``scenario`` value for Panel C rows."""
    return POOLED_SCENARIO_BY_DISPLAY[display]


def fold_stratum_mean_rmse(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    *,
    stratum_cols: Tuple[str, ...] = DEFAULT_CEILING_STRATUM_COLS,
    target_col: str = "synapses",
) -> float:
    agg = _fold_stratum_mean_agg(
        train_rows, test_rows, stratum_cols=stratum_cols, target_col=target_col
    )
    _, rmse = metrics_from_animal_frequency_agg(agg)
    return rmse


def _fold_stratum_mean_agg(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    *,
    stratum_cols: Tuple[str, ...] = DEFAULT_CEILING_STRATUM_COLS,
    target_col: str = "synapses",
) -> pd.DataFrame:
    """Stratum-mean baseline predictions at animal×frequency grain."""
    stratum_means = train_rows.groupby(list(stratum_cols))[target_col].mean()
    global_mean = float(train_rows[target_col].mean())
    multi_idx = pd.MultiIndex.from_frame(test_rows[list(stratum_cols)])
    pred = (
        stratum_means.reindex(multi_idx).astype(float).fillna(global_mean).to_numpy(dtype=float)
    )
    ev = test_rows[["animal_id", "frequency", target_col]].copy()
    ev["y_pred"] = pred
    return agg_animal_frequency(ev)


def _fit_fixed_rf(
    tr_aug: pd.DataFrame,
    syn_num: List[str],
    syn_cat: List[str],
    syn_log: List[str],
    best_params: Dict[str, Any],
) -> Pipeline:
    feat = syn_num + syn_cat + syn_log
    pipe = Pipeline(
        [("prep", _syn_prep(syn_num, syn_cat, syn_log)), ("rf", RandomForestRegressor(random_state=1))]
    )
    pipe.set_params(**best_params)
    tr = train_only_frame(tr_aug)
    X = tr[feat].dropna()
    y = tr.loc[X.index, "synapses"]
    pipe.fit(X, y)
    return pipe


def _fit_fixed_xgb(
    tr_aug: pd.DataFrame,
    syn_num: List[str],
    syn_cat: List[str],
    syn_log: List[str],
    best_params: Dict[str, Any],
) -> Pipeline:
    feat = syn_num + syn_cat + syn_log
    pipe = Pipeline(
        [
            ("prep", _syn_prep(syn_num, syn_cat, syn_log)),
            ("xgb", xgb.XGBRegressor(random_state=1, n_jobs=XGB_N_JOBS)),
        ]
    )
    pipe.set_params(**best_params)
    tr = train_only_frame(tr_aug)
    X = tr[feat].dropna()
    y = tr.loc[X.index, "synapses"]
    pipe.fit(X, y)
    return pipe


def _predict_wide_agg(
    reg: Pipeline,
    te_aug: pd.DataFrame,
    feat_cols: List[str],
) -> pd.DataFrame:
    Xs = te_aug[feat_cols].dropna()
    if len(Xs) == 0:
        raise ValueError(
            f"No complete feature rows on eval wide frame (n={len(te_aug)}); "
            f"check global Stage-1 noise_preds coverage."
        )
    pred = reg.predict(Xs)
    ev = te_aug.loc[Xs.index, ["animal_id", "frequency", "synapses"]].copy()
    ev["y_pred"] = pd.Series(pred, index=Xs.index)
    return agg_animal_frequency(ev)


def _wide_amp80_eval_agg(
    wide_train: pd.DataFrame, wide_eval: pd.DataFrame
) -> pd.DataFrame:
    amp = resolve_amp80_column(wide_train.columns)
    if amp is None:
        raise ValueError("wide frame missing amplitude_80 column for L7 baseline")
    model = smf.ols(f"synapses ~ Q('{amp}')", data=wide_train).fit()
    ev = wide_eval[["animal_id", "frequency", "synapses"]].copy()
    ev["y_pred"] = model.predict(wide_eval)
    return agg_animal_frequency(ev)


def _rf_xgb_eval_agg(
    model: ModelId,
    w_tr: pd.DataFrame,
    w_ev: pd.DataFrame,
    syn_num: List[str],
    syn_cat: List[str],
    syn_log: List[str],
    best_params: Dict[str, Any],
) -> pd.DataFrame:
    feat = syn_num + syn_cat + syn_log
    fit_fn = _fit_fixed_rf if model == "RF" else _fit_fixed_xgb
    reg = fit_fn(w_tr, syn_num, syn_cat, syn_log, best_params)
    return _predict_wide_agg(reg, w_ev, feat)


def _mlp_eval_agg(
    l_tr: pd.DataFrame,
    l_ev: pd.DataFrame,
    long_num: List[str],
    long_cat: List[str],
    long_log: List[str],
    hp: Dict[str, Any],
    *,
    holdout_seed: int,
    device: torch.device,
) -> pd.DataFrame:
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
    _, _, agg = aggregate_animal_freq_metrics(
        art.meta_eval[["animal_id", "frequency", "synapses"]],
        art.y_pred_eval,
    )
    return agg


def _pooled_rmse_from_aggs(*aggs: pd.DataFrame) -> float:
    pool = pd.concat(list(aggs), ignore_index=True)
    _, rmse = metrics_from_animal_frequency_agg(pool)
    return rmse


def _score_l7(wide_train: pd.DataFrame, wide_eval: pd.DataFrame) -> float:
    agg = _wide_amp80_eval_agg(wide_train, wide_eval)
    _, rmse = metrics_from_animal_frequency_agg(agg)
    return rmse


def _score_rf_xgb(
    model: ModelId,
    w_tr: pd.DataFrame,
    w_ev: pd.DataFrame,
    syn_num: List[str],
    syn_cat: List[str],
    syn_log: List[str],
    best_params: Dict[str, Any],
) -> float:
    agg = _rf_xgb_eval_agg(model, w_tr, w_ev, syn_num, syn_cat, syn_log, best_params)
    _, rmse = metrics_from_animal_frequency_agg(agg)
    return rmse


def _score_mlp(
    l_tr: pd.DataFrame,
    l_ev: pd.DataFrame,
    long_num: List[str],
    long_cat: List[str],
    long_log: List[str],
    hp: Dict[str, Any],
    *,
    holdout_seed: int,
    device: torch.device,
) -> float:
    agg = _mlp_eval_agg(
        l_tr, l_ev, long_num, long_cat, long_log, hp,
        holdout_seed=holdout_seed, device=device,
    )
    _, rmse = metrics_from_animal_frequency_agg(agg)
    return rmse


def _progress_key(
    eval_cohort: str, fold: int, scenario: str, model: str
) -> str:
    return f"{eval_cohort}|{fold}|{scenario}|{model}"


def _load_progress(path: Path, *, n_splits: int) -> Tuple[Set[str], int]:
    if not path.is_file():
        return set(), SYNTHESIS_CV_CACHE_VERSION
    data = json.loads(path.read_text(encoding="utf-8"))
    version = int(data.get("version", 1))
    if version != SYNTHESIS_CV_CACHE_VERSION:
        return set(), version
    if int(data.get("n_splits", n_splits)) != n_splits:
        return set(), version
    return set(data.get("completed", [])), version


def _save_progress(path: Path, completed: Set[str], *, n_splits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": SYNTHESIS_CV_CACHE_VERSION,
                "n_splits": n_splits,
                "completed": sorted(completed),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def reset_synthesis_oof_cache(
    *,
    oof_path: Path | None = None,
    oof_progress_path: Path | None = None,
) -> None:
    """Remove cohort OOF cache only (RMSE folds unchanged)."""
    from utils.benchmark_metrics import (
        STAGE2_SYNTHESIS_CV_OOF_PARQUET,
        STAGE2_SYNTHESIS_CV_OOF_PROGRESS_JSON,
    )

    for p in (
        Path(oof_path or STAGE2_SYNTHESIS_CV_OOF_PARQUET),
        Path(oof_progress_path or STAGE2_SYNTHESIS_CV_OOF_PROGRESS_JSON),
    ):
        if p.is_file():
            p.unlink()


def reset_pooled_oof_cache(
    *,
    oof_path: Path | None = None,
    oof_progress_path: Path | None = None,
) -> None:
    """Remove pooled OOF cache only (pooled RMSE folds unchanged)."""
    from utils.benchmark_metrics import (
        STAGE2_SYNTHESIS_CV_POOLED_OOF_PARQUET,
        STAGE2_SYNTHESIS_CV_POOLED_OOF_PROGRESS_JSON,
    )

    for p in (
        Path(oof_path or STAGE2_SYNTHESIS_CV_POOLED_OOF_PARQUET),
        Path(oof_progress_path or STAGE2_SYNTHESIS_CV_POOLED_OOF_PROGRESS_JSON),
    ):
        if p.is_file():
            p.unlink()


def reset_synthesis_cv_cache(
    *,
    folds_path: Path | None = None,
    progress_path: Path | None = None,
    summary_path: Path | None = None,
    oof_path: Path | None = None,
    oof_progress_path: Path | None = None,
) -> None:
    """Remove incremental CV cache (folds, progress, summary, OOF)."""
    from utils.benchmark_metrics import (
        STAGE2_SYNTHESIS_CV_FOLDS_PARQUET,
        STAGE2_SYNTHESIS_CV_OOF_PARQUET,
        STAGE2_SYNTHESIS_CV_OOF_PROGRESS_JSON,
        STAGE2_SYNTHESIS_CV_PROGRESS_JSON,
        STAGE2_SYNTHESIS_CV_SUMMARY_PARQUET,
    )

    for p in (
        Path(folds_path or STAGE2_SYNTHESIS_CV_FOLDS_PARQUET),
        Path(progress_path or STAGE2_SYNTHESIS_CV_PROGRESS_JSON),
        Path(summary_path or STAGE2_SYNTHESIS_CV_SUMMARY_PARQUET),
        Path(oof_path or STAGE2_SYNTHESIS_CV_OOF_PARQUET),
        Path(oof_progress_path or STAGE2_SYNTHESIS_CV_OOF_PROGRESS_JSON),
    ):
        if p.is_file():
            p.unlink()


def _append_fold_row(rows: List[Dict[str, Any]], row: Dict[str, Any], out_path: Path) -> None:
    rows.append(row)
    pd.DataFrame(rows).to_parquet(out_path, index=False)


def _load_oof_progress(path: Path, *, n_splits: int) -> Set[str]:
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("version", 1)) != SYNTHESIS_CV_CACHE_VERSION:
        return set()
    if int(data.get("n_splits", n_splits)) != n_splits:
        return set()
    return set(data.get("completed", []))


def _save_oof_progress(path: Path, completed: Set[str], *, n_splits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": SYNTHESIS_CV_CACHE_VERSION,
                "n_splits": n_splits,
                "completed": sorted(completed),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _agg_to_oof_records(
    agg: pd.DataFrame,
    *,
    fold: int,
    model: str,
    eval_cohort: str | None = None,
    train_scenario: str | None = None,
    display: int | None = None,
) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for _, r in agg.iterrows():
        row: Dict[str, Any] = {
            "fold": fold,
            "model": model,
            "animal_id": r["animal_id"],
            "frequency": r["frequency"],
            "y_true": float(r["y_true"]),
            "y_pred": float(r["y_pred"]),
        }
        if eval_cohort is not None:
            row["eval_cohort"] = eval_cohort
        if train_scenario is not None:
            row["train_scenario"] = train_scenario
        if display is not None:
            row["display"] = display
        recs.append(row)
    return recs


def _append_oof_records(
    oof_rows: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    oof_rows.extend(records)
    pd.DataFrame(oof_rows).to_parquet(out_path, index=False)


def _oof_r2_and_ceiling_sem(oof_sub: pd.DataFrame) -> Tuple[float, float]:
    """Pooled OOF R² plus per-fold R² SEM (for baseline band)."""
    r2 = float(r2_score(oof_sub["y_true"], oof_sub["y_pred"]))
    fold_r2s = [
        float(r2_score(g["y_true"], g["y_pred"]))
        for _, g in oof_sub.groupby("fold", sort=True)
        if len(g) >= 2
    ]
    sem = (
        float(np.std(fold_r2s, ddof=1) / np.sqrt(len(fold_r2s)))
        if len(fold_r2s) > 1
        else 0.0
    )
    return r2, sem


def run_synthesis_cv(
    hp_by_scenario: Dict[str, Dict[str, Any]],
    *,
    data: NNStage2Data | None = None,
    n_splits: int | None = None,
    folds_path: Path | None = None,
    progress_path: Path | None = None,
    summary_path: Path | None = None,
    oof_path: Path | None = None,
    oof_progress_path: Path | None = None,
    device: torch.device | None = None,
    force_rerun: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from utils.benchmark_metrics import (
        STAGE2_SYNTHESIS_CV_OOF_PARQUET,
        STAGE2_SYNTHESIS_CV_OOF_PROGRESS_JSON,
        stage2_synthesis_cv_paths,
    )

    n_splits = int(n_splits or N_FOLDS)
    default_folds, default_progress, default_summary = stage2_synthesis_cv_paths(n_splits)

    data = data or load_nn_stage2_data()
    splits = splits_for_long_stage2(data)
    feats = liberman_feature_lists(data.reformatted_orig, data.common_cols)
    syn_num, syn_log, syn_cat, long_num, long_cat = stage2_feature_lists(
        feats, noise_label="predicted"
    )
    long_log = list(feats["long_log"])

    folds_path = Path(folds_path or default_folds)
    progress_path = Path(progress_path or default_progress)
    summary_path = Path(summary_path or default_summary)
    oof_path = Path(oof_path or STAGE2_SYNTHESIS_CV_OOF_PARQUET)
    oof_progress_path = Path(oof_progress_path or STAGE2_SYNTHESIS_CV_OOF_PROGRESS_JSON)
    device = device or resolve_torch_device()

    if verbose:
        print(f"Synthesis CV: {n_splits}-fold → {summary_path.name}")

    if force_rerun:
        reset_synthesis_cv_cache(
            folds_path=folds_path,
            progress_path=progress_path,
            summary_path=summary_path,
            oof_path=oof_path,
            oof_progress_path=oof_progress_path,
        )

    completed, cache_version = _load_progress(progress_path, n_splits=n_splits)
    oof_completed = _load_oof_progress(oof_progress_path, n_splits=n_splits)
    fold_rows: List[Dict[str, Any]] = []
    oof_rows: List[Dict[str, Any]] = []
    if folds_path.is_file() and cache_version == SYNTHESIS_CV_CACHE_VERSION:
        fold_rows = pd.read_parquet(folds_path).to_dict("records")
    elif folds_path.is_file() and verbose:
        print(
            f"Ignoring stale {folds_path.name} "
            f"(cache v{cache_version} → v{SYNTHESIS_CV_CACHE_VERSION}); recomputing."
        )
    if oof_path.is_file():
        oof_rows = pd.read_parquet(oof_path).to_dict("records")

    scenarios: Tuple[TrainScenario, ...] = ("A", "B", "C")
    models: Tuple[ModelId, ...] = ("L7", "RF", "XGB", "MLP")

    if verbose:
        print("\n=== Global Stage 1 (per training scenario A/B/C) ===")
    global_s1: Dict[TrainScenario, Dict[str, Any]] = {}
    for scen in scenarios:
        global_s1[scen] = fit_global_stage1_for_scenario(
            scen, data, splits, verbose=verbose
        )

    for eval_cohort in ("Brad", "Liberman"):
        wide_all = (
            data.reformatted if eval_cohort == "Brad" else data.reformatted_orig
        ).reset_index(drop=True)

        folds = generate_cv_folds(wide_all, n_splits=n_splits)

        for fold_i, (_tr_idx, te_idx, tr_anim, te_anim) in enumerate(folds):
            wide_te = wide_all.iloc[te_idx]
            wide_tr_ceiling = wide_all.iloc[_tr_idx]

            ck = _progress_key(eval_cohort, fold_i, "all", "ceiling")
            need_rmse = ck not in completed
            need_oof = ck not in oof_completed
            if need_rmse or need_oof:
                agg_c = _fold_stratum_mean_agg(wide_tr_ceiling, wide_te)
                if need_rmse:
                    rmse_c = metrics_from_animal_frequency_agg(agg_c)[1]
                    row = {
                        "eval_cohort": eval_cohort,
                        "fold": fold_i,
                        "model": "ceiling",
                        "train_scenario": "all",
                        "rmse": rmse_c,
                        "n_cells": len(wide_te),
                    }
                    _append_fold_row(fold_rows, row, folds_path)
                    completed.add(ck)
                    _save_progress(progress_path, completed, n_splits=n_splits)
                    if verbose:
                        print(f"{eval_cohort} fold {fold_i} ceiling rmse={rmse_c:.4f}")
                if need_oof:
                    _append_oof_records(
                        oof_rows,
                        _agg_to_oof_records(
                            agg_c,
                            fold=fold_i,
                            model="ceiling",
                            eval_cohort=eval_cohort,
                            train_scenario="all",
                        ),
                        oof_path,
                    )
                    oof_completed.add(ck)
                    _save_oof_progress(oof_progress_path, oof_completed, n_splits=n_splits)

            for scen in scenarios:
                w_tr, w_ev, l_tr, l_ev, _, _ = build_cv_scenario_frames(
                    eval_cohort, scen, tr_anim, te_anim, data, splits
                )
                gs1 = global_s1[scen]
                w_aug_tr, w_aug_ev, l_aug_tr, l_aug_ev = _attach_global_stage1(
                    w_tr, w_ev, l_tr, l_ev, gs1["animal_preds"]
                )
                hp = hp_by_scenario[scen]

                for model in models:
                    ck = _progress_key(eval_cohort, fold_i, scen, model)
                    need_rmse = ck not in completed
                    need_oof = ck not in oof_completed
                    if not need_rmse and not need_oof:
                        continue

                    if model == "L7":
                        agg = _wide_amp80_eval_agg(w_tr, w_ev)
                        source = "classical"
                    elif model == "RF":
                        agg = _rf_xgb_eval_agg(
                            "RF",
                            w_aug_tr,
                            w_aug_ev,
                            syn_num,
                            syn_cat,
                            syn_log,
                            hp["RF"]["best_params"],
                        )
                        source = "classical"
                    elif model == "XGB":
                        agg = _rf_xgb_eval_agg(
                            "XGB",
                            w_aug_tr,
                            w_aug_ev,
                            syn_num,
                            syn_cat,
                            syn_log,
                            hp["XGB"]["best_params"],
                        )
                        source = "classical"
                    else:
                        agg = _mlp_eval_agg(
                            l_aug_tr,
                            l_aug_ev,
                            long_num,
                            long_cat,
                            long_log,
                            hp["MLP"],
                            holdout_seed=_mlp_holdout_seed(eval_cohort, scen, fold_i),
                            device=device,
                        )
                        source = "nn"

                    if need_rmse:
                        rmse = metrics_from_animal_frequency_agg(agg)[1]
                        row = {
                            "eval_cohort": eval_cohort,
                            "fold": fold_i,
                            "model": model,
                            "train_scenario": scen,
                            "rmse": rmse,
                            "n_cells": len(w_ev),
                            "source": source,
                        }
                        _append_fold_row(fold_rows, row, folds_path)
                        completed.add(ck)
                        _save_progress(progress_path, completed, n_splits=n_splits)
                        if verbose:
                            print(
                                f"{eval_cohort} f{fold_i} scen={scen} {model} rmse={rmse:.4f}"
                            )
                    if need_oof:
                        _append_oof_records(
                            oof_rows,
                            _agg_to_oof_records(
                                agg,
                                fold=fold_i,
                                model=model,
                                eval_cohort=eval_cohort,
                                train_scenario=scen,
                            ),
                            oof_path,
                        )
                        oof_completed.add(ck)
                        _save_oof_progress(oof_progress_path, oof_completed, n_splits=n_splits)

    folds_df = pd.DataFrame(fold_rows)
    summary = summarize_cv_folds(folds_df)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(summary_path, index=False)
    return folds_df, summary


def summarize_cv_folds(folds_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold RMSE → mean ± SEM; deck-compatible columns."""
    recs: List[Dict[str, Any]] = []
    ceil = folds_df.loc[folds_df["model"].eq("ceiling")]
    for eval_cohort in ("Brad", "Liberman"):
        sub_c = ceil.loc[ceil["eval_cohort"].eq(eval_cohort), "rmse"]
        if len(sub_c):
            recs.append(
                {
                    "test_set": eval_cohort,
                    "scenario": "all",
                    "model": "ceiling",
                    "RMSE": float(sub_c.mean()),
                    "RMSE_sem": float(sub_c.std(ddof=1) / np.sqrt(len(sub_c)))
                    if len(sub_c) > 1
                    else 0.0,
                    "source": "ceiling",
                    "n_folds": int(len(sub_c)),
                }
            )

    models = ("L7", "RF", "XGB", "MLP")
    for eval_cohort in ("Brad", "Liberman"):
        for scen in ("A", "B", "C"):
            for model in models:
                sub = folds_df.loc[
                    folds_df["eval_cohort"].eq(eval_cohort)
                    & folds_df["train_scenario"].eq(scen)
                    & folds_df["model"].eq(model),
                    "rmse",
                ]
                if sub.empty:
                    continue
                src_row = folds_df.loc[
                    folds_df["eval_cohort"].eq(eval_cohort)
                    & folds_df["train_scenario"].eq(scen)
                    & folds_df["model"].eq(model),
                    "source",
                ].iloc[0]
                recs.append(
                    {
                        "test_set": eval_cohort,
                        "scenario": scen,
                        "model": DECK_LABELS[model],
                        "RMSE": float(sub.mean()),
                        "RMSE_sem": float(sub.std(ddof=1) / np.sqrt(len(sub)))
                        if len(sub) > 1
                        else 0.0,
                        "source": src_row,
                        "n_folds": int(len(sub)),
                    }
                )
    return pd.DataFrame(recs)


def _pooled_progress_key(fold: int, display: int, model: str) -> str:
    return f"{fold}|{display}|{model}"


def _pooled_ceiling_key(fold: int) -> str:
    return f"{fold}|ceiling"


def _load_pooled_progress(path: Path, *, n_splits: int) -> Set[str]:
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("version", 1)) != SYNTHESIS_CV_CACHE_VERSION:
        return set()
    if int(data.get("n_splits", n_splits)) != n_splits:
        return set()
    return set(data.get("completed", []))


def _save_pooled_progress(path: Path, completed: Set[str], *, n_splits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": SYNTHESIS_CV_CACHE_VERSION,
                "n_splits": n_splits,
                "completed": sorted(completed),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def reset_pooled_synthesis_cv_cache(
    *,
    folds_path: Path | None = None,
    progress_path: Path | None = None,
    oof_path: Path | None = None,
    oof_progress_path: Path | None = None,
) -> None:
    from utils.benchmark_metrics import (
        STAGE2_SYNTHESIS_CV_POOLED_FOLDS_PARQUET,
        STAGE2_SYNTHESIS_CV_POOLED_OOF_PARQUET,
        STAGE2_SYNTHESIS_CV_POOLED_OOF_PROGRESS_JSON,
        STAGE2_SYNTHESIS_CV_POOLED_PROGRESS_JSON,
    )

    for p in (
        Path(folds_path or STAGE2_SYNTHESIS_CV_POOLED_FOLDS_PARQUET),
        Path(progress_path or STAGE2_SYNTHESIS_CV_POOLED_PROGRESS_JSON),
        Path(oof_path or STAGE2_SYNTHESIS_CV_POOLED_OOF_PARQUET),
        Path(oof_progress_path or STAGE2_SYNTHESIS_CV_POOLED_OOF_PROGRESS_JSON),
    ):
        if p.is_file():
            p.unlink()


def _append_pooled_row(rows: List[Dict[str, Any]], row: Dict[str, Any], out_path: Path) -> None:
    rows.append(row)
    pd.DataFrame(rows).to_parquet(out_path, index=False)


def compute_pooled_synthesis_folds(
    hp_by_scenario: Dict[str, Dict[str, Any]],
    *,
    data: NNStage2Data | None = None,
    n_splits: int | None = None,
    folds_path: Path | None = None,
    progress_path: Path | None = None,
    oof_path: Path | None = None,
    oof_progress_path: Path | None = None,
    device: torch.device | None = None,
    force_rerun: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Per-fold RMSE pooling Brad + Liberman holdout animal×frequency predictions.

  Display buckets 1–3 map to cohort-specific train scenarios; summary uses
    ``scenario`` A/B/C = within / cross / combined pooled.
    """
    from utils.benchmark_metrics import (
        STAGE2_SYNTHESIS_CV_POOLED_FOLDS_PARQUET,
        STAGE2_SYNTHESIS_CV_POOLED_OOF_PARQUET,
        STAGE2_SYNTHESIS_CV_POOLED_OOF_PROGRESS_JSON,
        STAGE2_SYNTHESIS_CV_POOLED_PROGRESS_JSON,
    )

    n_splits = int(n_splits or N_FOLDS)
    folds_path = Path(folds_path or STAGE2_SYNTHESIS_CV_POOLED_FOLDS_PARQUET)
    progress_path = Path(progress_path or STAGE2_SYNTHESIS_CV_POOLED_PROGRESS_JSON)
    oof_path = Path(oof_path or STAGE2_SYNTHESIS_CV_POOLED_OOF_PARQUET)
    oof_progress_path = Path(
        oof_progress_path or STAGE2_SYNTHESIS_CV_POOLED_OOF_PROGRESS_JSON
    )
    device = device or resolve_torch_device()

    if force_rerun:
        reset_pooled_synthesis_cv_cache(
            folds_path=folds_path,
            progress_path=progress_path,
            oof_path=oof_path,
            oof_progress_path=oof_progress_path,
        )

    completed = _load_pooled_progress(progress_path, n_splits=n_splits)
    oof_completed = _load_oof_progress(oof_progress_path, n_splits=n_splits)
    fold_rows: List[Dict[str, Any]] = []
    oof_rows: List[Dict[str, Any]] = []
    if folds_path.is_file() and not force_rerun:
        fold_rows = pd.read_parquet(folds_path).to_dict("records")
    if oof_path.is_file() and not force_rerun:
        oof_rows = pd.read_parquet(oof_path).to_dict("records")

    data = data or load_nn_stage2_data()
    splits = splits_for_long_stage2(data)
    feats = liberman_feature_lists(data.reformatted_orig, data.common_cols)
    syn_num, syn_log, syn_cat, long_num, long_cat = stage2_feature_lists(
        feats, noise_label="predicted"
    )
    long_log = list(feats["long_log"])

    scenarios: Tuple[TrainScenario, ...] = ("A", "B", "C")
    models: Tuple[ModelId, ...] = ("L7", "RF", "XGB", "MLP")

    if verbose:
        print(f"Pooled synthesis CV: {n_splits}-fold → {folds_path.name}")

    global_s1: Dict[TrainScenario, Dict[str, Any]] = {}
    for scen in scenarios:
        global_s1[scen] = fit_global_stage1_for_scenario(
            scen, data, splits, verbose=False
        )

    bb_wide = data.reformatted.reset_index(drop=True)
    lib_wide = data.reformatted_orig.reset_index(drop=True)
    folds_bb = generate_cv_folds(bb_wide, n_splits=n_splits)
    folds_lib = generate_cv_folds(lib_wide, n_splits=n_splits)

    for fold_i in range(n_splits):
        _tr_bb, te_idx_bb, tr_anim_bb, te_anim_bb = folds_bb[fold_i]
        _tr_lib, te_idx_lib, tr_anim_lib, te_anim_lib = folds_lib[fold_i]
        wide_te_bb = bb_wide.iloc[te_idx_bb]
        wide_tr_bb = bb_wide.iloc[_tr_bb]
        wide_te_lib = lib_wide.iloc[te_idx_lib]
        wide_tr_lib = lib_wide.iloc[_tr_lib]

        ck_ceil = _pooled_ceiling_key(fold_i)
        need_rmse = ck_ceil not in completed
        need_oof = ck_ceil not in oof_completed
        if need_rmse or need_oof:
            wide_tr_pool = pd.concat([wide_tr_bb, wide_tr_lib], ignore_index=True)
            wide_te_pool = pd.concat([wide_te_bb, wide_te_lib], ignore_index=True)
            agg_c = _fold_stratum_mean_agg(wide_tr_pool, wide_te_pool)
            if need_rmse:
                rmse_c = metrics_from_animal_frequency_agg(agg_c)[1]
                _append_pooled_row(
                    fold_rows,
                    {
                        "fold": fold_i,
                        "display": 0,
                        "model": "ceiling",
                        "rmse": rmse_c,
                        "n_cells": len(wide_te_pool),
                        "source": "ceiling",
                    },
                    folds_path,
                )
                completed.add(ck_ceil)
                _save_pooled_progress(progress_path, completed, n_splits=n_splits)
                if verbose:
                    print(f"pooled fold {fold_i} ceiling rmse={rmse_c:.4f}")
            if need_oof:
                _append_oof_records(
                    oof_rows,
                    _agg_to_oof_records(
                        agg_c,
                        fold=fold_i,
                        model="ceiling",
                        display=0,
                    ),
                    oof_path,
                )
                oof_completed.add(ck_ceil)
                _save_oof_progress(oof_progress_path, oof_completed, n_splits=n_splits)

        for display in (1, 2, 3):
            scen_map = train_scenarios_for_display(display)
            for model in models:
                ck = _pooled_progress_key(fold_i, display, model)
                need_rmse = ck not in completed
                need_oof = ck not in oof_completed
                if not need_rmse and not need_oof:
                    continue

                aggs: List[pd.DataFrame] = []
                source = "classical"
                for eval_cohort, scen in scen_map.items():
                    tr_anim = tr_anim_bb if eval_cohort == "Brad" else tr_anim_lib
                    te_anim = te_anim_bb if eval_cohort == "Brad" else te_anim_lib
                    w_tr, w_ev, l_tr, l_ev, _, _ = build_cv_scenario_frames(
                        eval_cohort, scen, tr_anim, te_anim, data, splits  # type: ignore[arg-type]
                    )
                    gs1 = global_s1[scen]
                    w_aug_tr, w_aug_ev, l_aug_tr, l_aug_ev = _attach_global_stage1(
                        w_tr, w_ev, l_tr, l_ev, gs1["animal_preds"]
                    )
                    hp = hp_by_scenario[scen]

                    if model == "L7":
                        agg = _wide_amp80_eval_agg(w_tr, w_ev)
                        source = "classical"
                    elif model == "RF":
                        agg = _rf_xgb_eval_agg(
                            "RF",
                            w_aug_tr,
                            w_aug_ev,
                            syn_num,
                            syn_cat,
                            syn_log,
                            hp["RF"]["best_params"],
                        )
                        source = "classical"
                    elif model == "XGB":
                        agg = _rf_xgb_eval_agg(
                            "XGB",
                            w_aug_tr,
                            w_aug_ev,
                            syn_num,
                            syn_cat,
                            syn_log,
                            hp["XGB"]["best_params"],
                        )
                        source = "classical"
                    else:
                        agg = _mlp_eval_agg(
                            l_aug_tr,
                            l_aug_ev,
                            long_num,
                            long_cat,
                            long_log,
                            hp["MLP"],
                            holdout_seed=_mlp_holdout_seed(eval_cohort, scen, fold_i),
                            device=device,
                        )
                        source = "nn"
                    aggs.append(agg)

                pool_agg = pd.concat(aggs, ignore_index=True)
                if need_rmse:
                    rmse = _pooled_rmse_from_aggs(*aggs)
                    _append_pooled_row(
                        fold_rows,
                        {
                            "fold": fold_i,
                            "display": display,
                            "model": model,
                            "rmse": rmse,
                            "n_cells": len(pool_agg),
                            "source": source,
                        },
                        folds_path,
                    )
                    completed.add(ck)
                    _save_pooled_progress(progress_path, completed, n_splits=n_splits)
                    if verbose:
                        print(
                            f"pooled f{fold_i} display={display} {model} rmse={rmse:.4f}"
                        )
                if need_oof:
                    _append_oof_records(
                        oof_rows,
                        _agg_to_oof_records(
                            pool_agg,
                            fold=fold_i,
                            model=model,
                            display=display,
                        ),
                        oof_path,
                    )
                    oof_completed.add(ck)
                    _save_oof_progress(oof_progress_path, oof_completed, n_splits=n_splits)

    return pd.DataFrame(fold_rows)


def summarize_pooled_folds(folds_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pooled fold RMSE → deck rows with ``test_set='Pooled'``."""
    recs: List[Dict[str, Any]] = []
    ceil = folds_df.loc[folds_df["model"].eq("ceiling")]
    if len(ceil):
        sub = ceil["rmse"]
        recs.append(
            {
                "test_set": "Pooled",
                "scenario": "all",
                "model": "ceiling",
                "RMSE": float(sub.mean()),
                "RMSE_sem": float(sub.std(ddof=1) / np.sqrt(len(sub)))
                if len(sub) > 1
                else 0.0,
                "source": "ceiling",
                "n_folds": int(len(sub)),
            }
        )

    models: Tuple[ModelId, ...] = ("L7", "RF", "XGB", "MLP")
    for display in (1, 2, 3):
        scen_col = pooled_summary_scenario(display)
        for model in models:
            sub = folds_df.loc[
                folds_df["display"].eq(display) & folds_df["model"].eq(model),
                "rmse",
            ]
            if sub.empty:
                continue
            src = folds_df.loc[
                folds_df["display"].eq(display) & folds_df["model"].eq(model),
                "source",
            ].iloc[0]
            recs.append(
                {
                    "test_set": "Pooled",
                    "scenario": scen_col,
                    "model": DECK_LABELS[model],
                    "RMSE": float(sub.mean()),
                    "RMSE_sem": float(sub.std(ddof=1) / np.sqrt(len(sub)))
                    if len(sub) > 1
                    else 0.0,
                    "source": src,
                    "n_folds": int(len(sub)),
                }
            )
    return pd.DataFrame(recs)


_PANEL_C_MODELS: Tuple[str, ...] = (
    "LR baseline",
    "RF",
    "XGB",
    "MLP",
)


def panel_c_rmse_report_table(summary_pooled: pd.DataFrame) -> pd.DataFrame:
    """Readable Panel C table: models × within/cross/combined (+ ceiling)."""
    models = list(_PANEL_C_MODELS)
    cols = {
        "A": "Within-cohort",
        "B": "Cross-cohort",
        "C": "Combined",
    }
    rows: List[Dict[str, Any]] = []
    sub = summary_pooled.loc[summary_pooled["test_set"].eq("Pooled")]
    for model in models:
        row: Dict[str, Any] = {"model": model}
        for scen, label in cols.items():
            r = sub.loc[sub["scenario"].eq(scen) & sub["model"].eq(model)]
            if r.empty:
                row[label] = ""
            else:
                m = float(r["RMSE"].iloc[0])
                s = float(r["RMSE_sem"].iloc[0])
                row[label] = f"{m:.4f} ± {s:.4f}"
        rows.append(row)
    ceil = sub.loc[sub["model"].eq("ceiling")]
    if not ceil.empty:
        m = float(ceil["RMSE"].iloc[0])
        s = float(ceil["RMSE_sem"].iloc[0])
        rows.append(
            {
                "model": "ceiling (stratum mean)",
                "Within-cohort": "",
                "Cross-cohort": "",
                "Combined": f"{m:.4f} ± {s:.4f}",
            }
        )
    return pd.DataFrame(rows)


def summarize_cohort_oof_r2(cohort_oof: pd.DataFrame) -> pd.DataFrame:
    """Liberman/Brad pooled OOF R² summary rows (``test_set`` = eval cohort)."""
    recs: List[Dict[str, Any]] = []
    ceil = cohort_oof.loc[cohort_oof["model"].eq("ceiling")]
    for eval_cohort in ("Brad", "Liberman"):
        sub_c = ceil.loc[ceil["eval_cohort"].eq(eval_cohort)]
        if sub_c.empty:
            continue
        r2, sem = _oof_r2_and_ceiling_sem(sub_c)
        recs.append(
            {
                "test_set": eval_cohort,
                "scenario": "all",
                "model": "ceiling",
                "R2": r2,
                "R2_baseline_sem": sem,
                "source": "ceiling",
                "n_cells": int(len(sub_c)),
                "n_folds": int(sub_c["fold"].nunique()),
            }
        )

    models: Tuple[ModelId, ...] = ("L7", "RF", "XGB", "MLP")
    for eval_cohort in ("Brad", "Liberman"):
        for scen in ("A", "B", "C"):
            for model in models:
                sub = cohort_oof.loc[
                    cohort_oof["eval_cohort"].eq(eval_cohort)
                    & cohort_oof["train_scenario"].eq(scen)
                    & cohort_oof["model"].eq(model)
                ]
                if sub.empty:
                    continue
                r2 = float(r2_score(sub["y_true"], sub["y_pred"]))
                src = "nn" if model == "MLP" else "classical"
                recs.append(
                    {
                        "test_set": eval_cohort,
                        "scenario": scen,
                        "model": DECK_LABELS[model],
                        "R2": r2,
                        "R2_baseline_sem": float("nan"),
                        "source": src,
                        "n_cells": int(len(sub)),
                        "n_folds": int(sub["fold"].nunique()),
                    }
                )
    return pd.DataFrame(recs)


def summarize_pooled_panel_oof_r2(pooled_oof: pd.DataFrame) -> pd.DataFrame:
    """Panel C rows with ``test_set='Pooled'`` from pooled-synthesis OOF cache."""
    recs: List[Dict[str, Any]] = []
    ceil = pooled_oof.loc[pooled_oof["model"].eq("ceiling")]
    if not ceil.empty:
        r2, sem = _oof_r2_and_ceiling_sem(ceil)
        recs.append(
            {
                "test_set": "Pooled",
                "scenario": "all",
                "model": "ceiling",
                "R2": r2,
                "R2_baseline_sem": sem,
                "source": "ceiling",
                "n_cells": int(len(ceil)),
                "n_folds": int(ceil["fold"].nunique()),
            }
        )

    models: Tuple[ModelId, ...] = ("L7", "RF", "XGB", "MLP")
    for display in (1, 2, 3):
        scen_col = pooled_summary_scenario(display)
        for model in models:
            sub = pooled_oof.loc[
                pooled_oof["display"].eq(display) & pooled_oof["model"].eq(model)
            ]
            if sub.empty:
                continue
            r2 = float(r2_score(sub["y_true"], sub["y_pred"]))
            src = "nn" if model == "MLP" else "classical"
            recs.append(
                {
                    "test_set": "Pooled",
                    "scenario": scen_col,
                    "model": DECK_LABELS[model],
                    "R2": r2,
                    "R2_baseline_sem": float("nan"),
                    "source": src,
                    "n_cells": int(len(sub)),
                    "n_folds": int(sub["fold"].nunique()),
                }
            )
    return pd.DataFrame(recs)


def summarize_pooled_oof_r2(
    cohort_oof: pd.DataFrame,
    pooled_oof: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full OOF R² summary: cohort panels + optional pooled panel."""
    parts = [summarize_cohort_oof_r2(cohort_oof)]
    if pooled_oof is not None and not pooled_oof.empty:
        parts.append(summarize_pooled_panel_oof_r2(pooled_oof))
    return pd.concat(parts, ignore_index=True)


def pooled_oof_r2_report_table(r2_summary: pd.DataFrame) -> pd.DataFrame:
    """§5.2.2 table: models × scenario with Cohort A / B / Pooled columns."""
    scen_labels = {
        "A": "Within-cohort",
        "B": "Cross-cohort",
        "C": "Combined",
    }
    cohort_cols = {
        "Liberman": "Cohort A (Liberman)",
        "Brad": "Cohort B (Brad)",
        "Pooled": "Pooled",
    }
    models = list(_PANEL_C_MODELS)
    rows: List[Dict[str, Any]] = []
    for model in models:
        row: Dict[str, Any] = {"model": model}
        for test_set, col_name in cohort_cols.items():
            sub = r2_summary.loc[
                r2_summary["test_set"].eq(test_set) & r2_summary["model"].eq(model)
            ]
            parts: List[str] = []
            for scen, label in scen_labels.items():
                r = sub.loc[sub["scenario"].eq(scen)]
                if r.empty:
                    continue
                parts.append(f"{label}: {float(r['R2'].iloc[0]):.4f}")
            row[col_name] = "; ".join(parts) if parts else ""
        rows.append(row)

    ceil_parts: Dict[str, str] = {}
    for test_set, col_name in cohort_cols.items():
        r = r2_summary.loc[
            r2_summary["test_set"].eq(test_set) & r2_summary["model"].eq("ceiling")
        ]
        if r.empty:
            ceil_parts[col_name] = ""
        else:
            m = float(r["R2"].iloc[0])
            s = float(r["R2_baseline_sem"].iloc[0])
            ceil_parts[col_name] = f"{m:.4f} ± {s:.4f}"
    rows.append({"model": "ceiling (stratum mean)", **ceil_parts})
    return pd.DataFrame(rows)

"""Two-stage sklearn runners for abr_wide_long_comparison (subject-level Stage 2 CV)."""
from __future__ import annotations

import multiprocessing
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.nn_stage2 import (
    RF_PARAMS,
    attach_noise_preds_long,
    fit_stage1_wide_best,
    mk_log_pipe,
)
from utils.nn_stage2_data import wide_stage1_fit, wide_stage1_val
from utils.subject_cv import (
    DEFAULT_CV_RANDOM_STATE,
    fit_subject_search,
)

XGB_N_JOBS = max(1, multiprocessing.cpu_count() // 2)
XGB_STAGE2_PARAM_GRID = {
    "xgb__n_estimators": [10, 25, 50],
    "xgb__max_depth": [4, 6, 8],
    "xgb__learning_rate": [0.05, 0.1, 0.2],
}


def agg_animal_frequency(
    df: pd.DataFrame,
    *,
    animal_col: str = "animal_id",
    freq_col: str = "frequency",
    y_col: str = "synapses",
    pred_col: str = "y_pred",
) -> pd.DataFrame:
    """
    Collapse row-level preds to one row per animal×frequency.

    Long format has multiple SPL rows per cell; wide may have one row per cell already.
    Uses first ``synapses`` and mean ``y_pred`` within each cell.
    """
    return (
        df.groupby([animal_col, freq_col], as_index=False)
        .agg(y_true=(y_col, "first"), y_pred=(pred_col, "mean"))
    )


def metrics_from_animal_frequency_agg(agg: pd.DataFrame) -> Tuple[float, float]:
    """R² and RMSE from an animal×frequency aggregation table."""
    r2 = float(r2_score(agg["y_true"], agg["y_pred"]))
    rmse = float(np.sqrt(np.mean((agg["y_true"] - agg["y_pred"]) ** 2)))
    return r2, rmse


def _syn_prep(syn_num: List[str], syn_cat: List[str], syn_log: List[str]) -> ColumnTransformer:
    transformers = [
        ("num", StandardScaler(), syn_num),
        ("log", mk_log_pipe(), syn_log),
    ]
    if syn_cat:
        transformers.insert(
            1,
            ("cat", OneHotEncoder(drop="first", sparse_output=False), syn_cat),
        )
    return ColumnTransformer(transformers)


def _fit_stage2_rf(
    tr_aug: pd.DataFrame,
    syn_num: List[str],
    syn_cat: List[str],
    syn_log: List[str],
    *,
    random_state: int = DEFAULT_CV_RANDOM_STATE,
    n_iter: int = 10,
) -> Tuple[Pipeline, float, Dict[str, Any]]:
    feat_cols = syn_num + syn_cat + syn_log
    pipe = Pipeline(
        [("prep", _syn_prep(syn_num, syn_cat, syn_log)), ("rf", RandomForestRegressor(random_state=1))]
    )
    search = fit_subject_search(
        pipe,
        RF_PARAMS,
        tr_aug,
        feat_cols,
        "synapses",
        scoring="r2",
        prefer_stratified=False,
        random_state=random_state,
        n_iter=n_iter,
        n_jobs=-1,
    )
    return search.best_estimator_, float(search.best_score_), dict(search.best_params_)


def _fit_stage2_xgb(
    tr_aug: pd.DataFrame,
    syn_num: List[str],
    syn_cat: List[str],
    syn_log: List[str],
) -> Tuple[Pipeline, float, Dict[str, Any]]:
    feat_cols = syn_num + syn_cat + syn_log
    xgb_pipe = Pipeline(
        [
            ("prep", _syn_prep(syn_num, syn_cat, syn_log)),
            ("xgb", xgb.XGBRegressor(random_state=1, n_jobs=XGB_N_JOBS)),
        ]
    )
    search = fit_subject_search(
        xgb_pipe,
        XGB_STAGE2_PARAM_GRID,
        tr_aug,
        feat_cols,
        "synapses",
        scoring="r2",
        search_cls=GridSearchCV,
        prefer_stratified=False,
        n_jobs=1,
    )
    return search.best_estimator_, float(search.best_score_), dict(search.best_params_)


def _long_stage2_test_agg(
    wide_train: pd.DataFrame,
    wide_test: pd.DataFrame,
    long_train: pd.DataFrame,
    long_test: pd.DataFrame,
    *,
    noise_num: List[str],
    noise_log: List[str],
    long_num: List[str],
    long_cat: List[str],
    long_log: List[str],
    model: Literal["RF", "XGB"],
) -> Tuple[pd.DataFrame, float, Dict[str, object]]:
    """Animal×frequency test predictions after long Stage 2 (mean pred per cell)."""
    s1_out = fit_stage1_wide_best(
        wide_stage1_fit(wide_train),
        wide_stage1_val(wide_train),
        wide_test,
        noise_num,
        noise_log,
        random_state=1,
        verbose=False,
    )
    long_tr = attach_noise_preds_long(
        long_train,
        s1_out["animal_pred_non_test"],
        fallback_noise_cat=False,
        require_full_coverage=True,
    )
    long_te = attach_noise_preds_long(
        long_test, s1_out["animal_pred_te"], fallback_noise_cat=False
    )
    fit_fn = _fit_stage2_rf if model == "RF" else _fit_stage2_xgb
    reg, cv_r2, _ = fit_fn(long_tr, long_num, long_cat, long_log)
    feat_cols = long_num + long_cat + long_log
    Xs_te = long_te[feat_cols].dropna()
    y_pred_rows = reg.predict(Xs_te)
    _eval = long_te.loc[Xs_te.index, ["animal_id", "frequency", "synapses"]].copy()
    _eval["y_pred"] = pd.Series(y_pred_rows, index=Xs_te.index)
    agg = agg_animal_frequency(_eval)
    return agg, cv_r2, s1_out


def long_stage2_test_agg(
    wide_train: pd.DataFrame,
    wide_test: pd.DataFrame,
    long_train: pd.DataFrame,
    long_test: pd.DataFrame,
    *,
    noise_num: List[str],
    noise_log: List[str],
    long_num: List[str],
    long_cat: List[str],
    long_log: List[str],
    model: Literal["RF", "XGB"] = "RF",
) -> pd.DataFrame:
    """Public wrapper: test-set synapse preds aggregated per animal×frequency."""
    agg, _, _ = _long_stage2_test_agg(
        wide_train,
        wide_test,
        long_train,
        long_test,
        noise_num=noise_num,
        noise_log=noise_log,
        long_num=long_num,
        long_cat=long_cat,
        long_log=long_log,
        model=model,
    )
    return agg


def wide_stage2_test_agg(
    wide_train: pd.DataFrame,
    wide_test: pd.DataFrame,
    *,
    noise_num: List[str],
    noise_log: List[str],
    syn_num: List[str],
    syn_cat: List[str],
    syn_log: List[str],
    model: Literal["RF", "XGB"] = "RF",
) -> pd.DataFrame:
    """Test-set synapse preds aggregated per animal×frequency (wide Stage 2)."""
    s1_out = fit_stage1_wide_best(
        wide_stage1_fit(wide_train),
        wide_stage1_val(wide_train),
        wide_test,
        noise_num,
        noise_log,
        random_state=1,
        verbose=False,
    )
    tr_aug = attach_noise_preds_long(
        wide_train,
        s1_out["animal_pred_non_test"],
        fallback_noise_cat=False,
        require_full_coverage=True,
    )
    te_aug = attach_noise_preds_long(
        wide_test, s1_out["animal_pred_te"], fallback_noise_cat=False
    )
    fit_fn = _fit_stage2_rf if model == "RF" else _fit_stage2_xgb
    reg, _, _ = fit_fn(tr_aug, syn_num, syn_cat, syn_log)
    feat_cols = syn_num + syn_cat + syn_log
    Xs_te = te_aug[feat_cols].dropna()
    y_pred = reg.predict(Xs_te)
    _eval = te_aug.loc[Xs_te.index, ["animal_id", "frequency", "synapses"]].copy()
    _eval["y_pred"] = pd.Series(y_pred, index=Xs_te.index)
    return agg_animal_frequency(_eval)


def run_two_stage_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label: str,
    *,
    noise_num: Optional[List[str]] = None,
    noise_log: Optional[List[str]] = None,
    syn_num: Optional[List[str]] = None,
    syn_cat: Optional[List[str]] = None,
    syn_log: Optional[List[str]] = None,
) -> Tuple[float, float, pd.Series, np.ndarray]:
    """
    Two-stage RF: Stage 1 via fit_stage1_wide_best; Stage 2 RF with subject-level CV on Train only.
    """
    if noise_num is None or noise_log is None or syn_num is None or syn_cat is None or syn_log is None:
        raise ValueError(
            "noise_num, noise_log, syn_num, syn_cat, syn_log must be set by the notebook"
        )

    print(f"  [{label}]  S1/S2 train wide rows: {len(train_df)}")

    s1_out = fit_stage1_wide_best(
        wide_stage1_fit(train_df),
        wide_stage1_val(train_df),
        test_df,
        list(noise_num),
        list(noise_log),
        random_state=1,
        verbose=True,
    )
    tr_aug = attach_noise_preds_long(
        train_df,
        s1_out["animal_pred_non_test"],
        fallback_noise_cat=False,
        require_full_coverage=True,
    )
    te_aug = attach_noise_preds_long(
        test_df, s1_out["animal_pred_te"], fallback_noise_cat=False
    )

    reg, cv_r2, _ = _fit_stage2_rf(tr_aug, syn_num, syn_cat, syn_log)
    Xs_te = te_aug[syn_num + syn_cat + syn_log].dropna()
    ys_te = te_aug.loc[Xs_te.index, "synapses"]
    y_pred = reg.predict(Xs_te)
    r2 = float(r2_score(ys_te, y_pred))
    rmse = float(np.sqrt(np.mean((ys_te.values - y_pred) ** 2)))

    print(
        f"           Synapse reg — CV R²: {cv_r2:.3f}, "
        f"test R²: {r2:.3f}, RMSE: {rmse:.3f}"
    )
    return r2, rmse, ys_te, y_pred


def run_two_stage_long_s2(
    wide_train: pd.DataFrame,
    wide_test: pd.DataFrame,
    long_train: pd.DataFrame,
    long_test: pd.DataFrame,
    label: str,
    *,
    noise_num: Optional[List[str]] = None,
    noise_log: Optional[List[str]] = None,
    long_num: Optional[List[str]] = None,
    long_cat: Optional[List[str]] = None,
    long_log: Optional[List[str]] = None,
) -> Tuple[float, float]:
    if noise_num is None or noise_log is None or long_num is None or long_cat is None or long_log is None:
        raise ValueError("noise_num, noise_log, long_num, long_cat, long_log required")

    print(
        f"  [{label}]  S1 wide train: {len(wide_train)} | long train: {len(long_train)} rows"
    )
    agg, cv_r2, s1_out = _long_stage2_test_agg(
        wide_train,
        wide_test,
        long_train,
        long_test,
        noise_num=noise_num,
        noise_log=noise_log,
        long_num=long_num,
        long_cat=long_cat,
        long_log=long_log,
        model="RF",
    )
    print(
        f"           {s1_out['stage1_model'].upper()}  — val acc (row): {s1_out['s1_val_acc_pre']:.3f} | "
        f"non-test acc (animal@cal): {s1_out['s1_non_test_acc_animal']:.3f} | "
        f"threshold (Fit+Val): {s1_out['stage1_threshold']:.3f} | "
        f"test acc (animal): {s1_out['s1_test_acc']:.3f}, AUC: {s1_out['s1_test_auc']:.3f}"
    )
    r2, rmse = metrics_from_animal_frequency_agg(agg)
    print(
        f"           Synapse reg — CV R²: {cv_r2:.3f}, "
        f"test R² (animal×freq): {r2:.3f}, RMSE: {rmse:.3f}"
    )
    return r2, rmse


def run_two_stage_long_s2_xgb(
    wide_train: pd.DataFrame,
    wide_test: pd.DataFrame,
    long_train: pd.DataFrame,
    long_test: pd.DataFrame,
    label: str,
    *,
    noise_num: Optional[List[str]] = None,
    noise_log: Optional[List[str]] = None,
    long_num: Optional[List[str]] = None,
    long_cat: Optional[List[str]] = None,
    long_log: Optional[List[str]] = None,
) -> Tuple[float, float]:
    """Stage 1 pool = wide_train; Stage 2 XGB on long rows (subject-level CV on Train)."""
    if noise_num is None or noise_log is None or long_num is None or long_cat is None or long_log is None:
        raise ValueError("noise_num, noise_log, long_num, long_cat, long_log required")

    print(f"  [{label}]  S1/S2 wide train: {len(wide_train)} | long train: {len(long_train)} rows")
    agg, cv_r2, s1_out = _long_stage2_test_agg(
        wide_train,
        wide_test,
        long_train,
        long_test,
        noise_num=noise_num,
        noise_log=noise_log,
        long_num=long_num,
        long_cat=long_cat,
        long_log=long_log,
        model="XGB",
    )
    print(
        f"           {s1_out['stage1_model'].upper()}  — val acc (row): {s1_out['s1_val_acc_pre']:.3f} | "
        f"non-test acc (animal@cal): {s1_out['s1_non_test_acc_animal']:.3f} | "
        f"threshold (Fit+Val): {s1_out['stage1_threshold']:.3f} | "
        f"test acc (animal): {s1_out['s1_test_acc']:.3f}, AUC: {s1_out['s1_test_auc']:.3f}"
    )
    r2, rmse = metrics_from_animal_frequency_agg(agg)
    print(
        f"           Synapse XGB — CV R²: {cv_r2:.3f}, "
        f"test R² (animal×freq): {r2:.3f}, RMSE: {rmse:.3f}"
    )
    return r2, rmse


def run_two_stage_model_xgb_wide(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label: str,
    *,
    noise_num: Optional[List[str]] = None,
    noise_log: Optional[List[str]] = None,
    syn_num: Optional[List[str]] = None,
    syn_cat: Optional[List[str]] = None,
    syn_log: Optional[List[str]] = None,
) -> Tuple[float, float]:
    """Stage 1 wide RF/LR; Stage 2 XGB on wide rows."""
    if noise_num is None or noise_log is None or syn_num is None or syn_cat is None or syn_log is None:
        raise ValueError("noise_num, noise_log, syn_num, syn_cat, syn_log required")

    print(f"  [{label}]  S1/S2 train wide rows: {len(train_df)}")

    s1_out = fit_stage1_wide_best(
        wide_stage1_fit(train_df),
        wide_stage1_val(train_df),
        test_df,
        noise_num,
        noise_log,
        random_state=1,
        verbose=False,
    )
    tr_aug = attach_noise_preds_long(
        train_df,
        s1_out["animal_pred_non_test"],
        fallback_noise_cat=False,
        require_full_coverage=True,
    )
    te_aug = attach_noise_preds_long(
        test_df, s1_out["animal_pred_te"], fallback_noise_cat=False
    )

    reg, cv_r2, _ = _fit_stage2_xgb(tr_aug, syn_num, syn_cat, syn_log)
    Xs_te = te_aug[syn_num + syn_cat + syn_log].dropna()
    ys_te = te_aug.loc[Xs_te.index, "synapses"]
    y_pred = reg.predict(Xs_te)
    r2 = float(r2_score(ys_te, y_pred))
    rmse = float(np.sqrt(np.mean((ys_te.values - y_pred) ** 2)))
    print(
        f"  [{label}] XGB-wide — CV R²: {cv_r2:.3f}, "
        f"test R²: {r2:.3f}, RMSE: {rmse:.3f}"
    )
    return r2, rmse


def wide_xgb_triple(
    train_A: pd.DataFrame,
    train_B: pd.DataFrame,
    train_C: pd.DataFrame,
    test_df: pd.DataFrame,
    noise_num: List[str],
    noise_log: List[str],
    *,
    headers: Optional[Dict[str, str]] = None,
    label_suffix: str = "",
    **syn_kw,
) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for lab, tr in [("A", train_A), ("B", train_B), ("C", train_C)]:
        if headers and lab in headers:
            print(headers[lab])
        lbl = f"{lab}{label_suffix}" if label_suffix else lab
        out[lab] = run_two_stage_model_xgb_wide(
            tr,
            test_df,
            lbl,
            noise_num=noise_num,
            noise_log=noise_log,
            **syn_kw,
        )
    return out

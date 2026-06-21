"""
Stage-1 wide RF noise classifier + Stage-2 synapse regression on **long** rows.

Stage 1: Train-wide fit, Validate-wide HP selection, Youden threshold on Fit+Val animals, Test metrics; SPL 50/60/70/80.
Stage 2 long: MLP or hybrid ``CNNPlusTabular`` (waveform CNN branch + tabular MLP, fused).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from utils.nn_stage2_data import (
    MOUSE_SPLIT_RANDOM_STATE,
    NN_TRAIN_RANDOM_STATE,
    NNStage2Data,
    full_wave_matrix,
    unified_grid_wave_i_matrix,
    wide_stage1_fit,
    wide_stage1_val,
)

RF_PARAMS = {
    "rf__n_estimators": [25, 50, 100, 150, 200, 300, 400, 500],
    "rf__max_depth": [5, 10, 20, 30, 40, 50, None],
    "rf__min_samples_leaf": [2, 5, 10, 15, 20, 30],
    "rf__min_samples_split": [2, 5, 10, 15, 20, 30],
    "rf__max_features": ["sqrt", "log2", 0.3, 0.5],
}
_RF_PARAMS = RF_PARAMS  # backward compatible


def mk_log_pipe():
    return Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, validate=True)),
            ("scaler", StandardScaler()),
        ]
    )


def _youden_j_threshold(y_true, y_score) -> float:
    """Youden J (tpr - fpr) optimal cut on animal-level scores."""
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    if np.unique(y).size < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    if not valid.any():
        return 0.5
    j = tpr[valid] - fpr[valid]
    return float(thresholds[valid][np.argmax(j)])


def animal_noise_gt_series(wide_df: pd.DataFrame) -> pd.Series:
    """One ground-truth noise label per animal (animal-level, not per frequency row)."""
    by_animal = wide_df.groupby("animal_id")["noise_cat"]
    nuniq = by_animal.nunique().round()
    bad = nuniq[nuniq > 1]
    if len(bad):
        sample = bad.index[:3].tolist()
        raise ValueError(
            f"noise_cat varies within animal_id for {len(bad)} animals; e.g. {sample}"
        )
    return by_animal.first().round().astype(int)


def _animal_prob_and_gt(
    wide_df: pd.DataFrame,
    row_index: pd.Index,
    proba: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Mean row probability per animal; GT is animal-level ``noise_cat``."""
    rows = wide_df.loc[row_index, ["animal_id"]].copy()
    rows["proba"] = proba.values
    animal_prob = rows.groupby("animal_id")["proba"].mean()
    animal_gt = animal_noise_gt_series(wide_df.loc[row_index]).reindex(animal_prob.index)
    return animal_prob, animal_gt


def aggregate_noise_to_animal(
    df: pd.DataFrame,
    indices: pd.Index,
    proba_col: pd.Series,
    *,
    threshold: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
    """Post-aggregate row probabilities to animal level, then threshold."""
    rows = df.loc[indices, ["animal_id"]].copy()
    rows["proba"] = proba_col.values
    agg_proba = rows.groupby("animal_id")["proba"].mean()
    return (agg_proba > threshold).astype(int), agg_proba


def _animal_label_accuracy(animal_gt: pd.Series, animal_pred: pd.Series) -> float:
    """Accuracy on aligned animal-level label Series (index = animal_id)."""
    pred = animal_pred.reindex(animal_gt.index).astype(int)
    if pred.isna().any():
        missing = pred.index[pred.isna()].astype(str).tolist()[:5]
        raise ValueError(f"Missing animal-level predictions for {missing}")
    return float((animal_gt.astype(int) == pred).mean())


def _animal_auc(animal_gt: pd.Series, animal_prob: pd.Series) -> float:
    """ROC AUC on aligned animal-level scores (index = animal_id)."""
    prob = animal_prob.reindex(animal_gt.index).astype(float)
    if prob.isna().any():
        missing = prob.index[prob.isna()].astype(str).tolist()[:5]
        raise ValueError(f"Missing animal-level probabilities for {missing}")
    y = animal_gt.astype(int).values
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, prob.values))


def _row_noise_accuracy(y_true: pd.Series, proba: pd.Series) -> float:
    """Pre-aggregation Validate score: row-level accuracy (proba >= 0.5)."""
    y_hat = (proba.values >= 0.5).astype(int)
    return float(accuracy_score(y_true.astype(int), y_hat))


def _animal_preds_non_test(
    clf: Pipeline,
    s1_wide_fit: pd.DataFrame,
    s1_wide_val: pd.DataFrame,
    feat_cols: List[str],
    *,
    threshold: float,
) -> pd.Series:
    """Train + Validate animal labels (mean proba, then threshold); excludes Test."""
    X_fit = s1_wide_fit[feat_cols].dropna()
    X_val = s1_wide_val[feat_cols].dropna()
    proba_tr = pd.Series(clf.predict_proba(X_fit)[:, 1], index=X_fit.index)
    pred_tr, _ = aggregate_noise_to_animal(
        s1_wide_fit, X_fit.index, proba_tr, threshold=threshold
    )
    parts = [pred_tr]
    if len(X_val):
        proba_va = pd.Series(clf.predict_proba(X_val)[:, 1], index=X_val.index)
        pred_va, _ = aggregate_noise_to_animal(
            s1_wide_val, X_val.index, proba_va, threshold=threshold
        )
        parts.append(pred_va)
    return pd.concat(parts)


def stage1_animal_pred_covering(
    wide_pool: pd.DataFrame,
    s1: dict,
    noise_num: List[str],
    noise_log: List[str],
    *,
    long_pool: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Animal-level Stage-1 labels for every animal in ``wide_pool`` (and ``long_pool``).

    Uses Fit+Validate preds, any holdout preds already in ``s1["animal_pred_te"]`` that
    appear in the pools, then applies the fitted classifier to remaining animals (e.g.
    official ``DataGroup == "Test"`` rows still present in a CV train pool).
    """
    parts: List[pd.Series] = [s1["animal_pred_non_test"]]
    te = s1.get("animal_pred_te")
    if te is not None and len(te):
        pool_animals = set(wide_pool["animal_id"].astype(str))
        if long_pool is not None:
            pool_animals |= set(long_pool["animal_id"].astype(str))
        te_in_pool = te[te.index.astype(str).isin(pool_animals)]
        if len(te_in_pool):
            parts.append(te_in_pool)
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="last")]

    need = set(wide_pool["animal_id"].astype(str))
    if long_pool is not None:
        need |= set(long_pool["animal_id"].astype(str))
    missing = need - set(out.index.astype(str))
    if not missing:
        return out

    feat_cols = list(noise_num) + list(noise_log)
    clf = s1["clf"]
    t_cal = float(s1["stage1_threshold"])
    sub = wide_pool.loc[wide_pool["animal_id"].astype(str).isin(missing)].reset_index(
        drop=True
    )
    X = sub[feat_cols].dropna()
    if len(X) == 0:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"Stage 1 missing wide features for {len(missing)} animals; e.g. {sample}"
        )
    proba = pd.Series(clf.predict_proba(X)[:, 1], index=X.index)
    pred, _ = aggregate_noise_to_animal(sub, X.index, proba, threshold=t_cal)
    out = pd.concat([out, pred])
    out = out[~out.index.duplicated(keep="last")]
    still = need - set(out.index.astype(str))
    if still:
        sample = sorted(still)[:5]
        raise ValueError(
            f"Stage 1 missing noise_preds for {len(still)} animals; e.g. {sample}"
        )
    return out


STAGE1_TIE_MODEL = "rf"
_LR_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def _animal_noise_metrics(
    wide_df: pd.DataFrame,
    row_index: pd.Index,
    proba: pd.Series,
    *,
    threshold: float,
) -> tuple[float, float, pd.Series, pd.Series]:
    animal_prob, animal_gt = _animal_prob_and_gt(wide_df, row_index, proba)
    animal_pred = (animal_prob > threshold).astype(int)
    acc = float((animal_pred == animal_gt).mean())
    auc = float(roc_auc_score(animal_gt, animal_prob))
    return acc, auc, animal_pred, animal_prob


def _stage1_noise_prep(noise_num: List[str], noise_log: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), noise_num),
            ("log", mk_log_pipe(), noise_log),
        ]
    )


def _pack_stage1_result(
    clf: Pipeline,
    s1_wide_fit: pd.DataFrame,
    s1_wide_val: pd.DataFrame,
    wide_test: pd.DataFrame,
    X_fit: pd.DataFrame,
    X_val: pd.DataFrame,
    X_te: pd.DataFrame,
    feat_cols: List[str],
    *,
    verbose: bool,
    label: str,
) -> dict:
    proba_fit = pd.Series(clf.predict_proba(X_fit)[:, 1], index=X_fit.index)
    if len(X_te):
        proba_te = pd.Series(clf.predict_proba(X_te)[:, 1], index=X_te.index)
    else:
        proba_te = pd.Series(dtype=float)
    y_fit = s1_wide_fit.loc[X_fit.index, "noise_cat"].round().astype(int)
    s1_fit_acc_row = _row_noise_accuracy(y_fit, proba_fit)

    animal_prob_fit, animal_gt_fit = _animal_prob_and_gt(
        s1_wide_fit, X_fit.index, proba_fit
    )

    if len(X_val):
        proba_va = pd.Series(clf.predict_proba(X_val)[:, 1], index=X_val.index)
        y_va = s1_wide_val.loc[X_val.index, "noise_cat"].round().astype(int)
        s1_val_acc_pre = _row_noise_accuracy(y_va, proba_va)
        animal_prob_va, animal_gt_va = _animal_prob_and_gt(
            s1_wide_val, X_val.index, proba_va
        )
        t_youden_val_only = _youden_j_threshold(
            animal_gt_va.values, animal_prob_va.values
        )
        s1_val_acc_animal_val_youden = float(
            ((animal_prob_va > t_youden_val_only).astype(int) == animal_gt_va).mean()
        )
        pool_prob = pd.concat([animal_prob_fit, animal_prob_va])
        pool_gt = pd.concat([animal_gt_fit, animal_gt_va])
    else:
        proba_va = pd.Series(dtype=float)
        s1_val_acc_pre = float("nan")
        s1_val_acc_animal_val_youden = float("nan")
        t_youden_val_only = 0.5
        pool_prob = animal_prob_fit
        pool_gt = animal_gt_fit

    wide_non_test = pd.concat([s1_wide_fit, s1_wide_val], ignore_index=True)
    gt_non_test = animal_noise_gt_series(wide_non_test)

    t_cal = _youden_j_threshold(pool_gt.values, pool_prob.values)

    animal_pred_tr, _ = aggregate_noise_to_animal(
        s1_wide_fit, X_fit.index, proba_fit, threshold=t_cal
    )
    if len(X_te):
        animal_pred_te, animal_prob = aggregate_noise_to_animal(
            wide_test, X_te.index, proba_te, threshold=t_cal
        )
    else:
        animal_pred_te = pd.Series(dtype=int)
        animal_prob = pd.Series(dtype=float)
    animal_pred_non_test = _animal_preds_non_test(
        clf, s1_wide_fit, s1_wide_val, feat_cols, threshold=t_cal
    )
    prob_non_test = pool_prob.reindex(gt_non_test.index)
    s1_non_test_acc_animal = _animal_label_accuracy(gt_non_test, animal_pred_non_test)
    s1_non_test_auc_animal = _animal_auc(gt_non_test, prob_non_test)
    s1_fit_acc_animal = _animal_label_accuracy(
        animal_noise_gt_series(s1_wide_fit), animal_pred_tr
    )
    if len(X_val):
        pred_va, _ = aggregate_noise_to_animal(
            s1_wide_val, X_val.index, proba_va, threshold=t_cal
        )
        s1_val_acc_animal = _animal_label_accuracy(
            animal_noise_gt_series(s1_wide_val), pred_va
        )
    else:
        s1_val_acc_animal = float("nan")
    if len(X_te):
        _, animal_gt_te = _animal_prob_and_gt(wide_test, X_te.index, proba_te)
        s1_test_acc, s1_test_auc, _, _ = _animal_noise_metrics(
            wide_test, X_te.index, proba_te, threshold=t_cal
        )
        s1_test_acc_0p5, _, _, _ = _animal_noise_metrics(
            wide_test, X_te.index, proba_te, threshold=0.5
        )
    else:
        animal_gt_te = pd.Series(dtype=int)
        s1_test_acc = float("nan")
        s1_test_auc = float("nan")
        s1_test_acc_0p5 = float("nan")
    if verbose:
        print(
            f"           {label}  — fit acc (row): {s1_fit_acc_row:.3f} | "
            f"fit acc (animal@cal): {s1_fit_acc_animal:.3f} | "
            f"val acc (row): {s1_val_acc_pre:.3f} | "
            f"val acc (animal@cal): {s1_val_acc_animal:.3f} | "
            f"non-test acc (animal@cal): {s1_non_test_acc_animal:.3f} | "
            f"non-test AUC (animal): {s1_non_test_auc_animal:.3f} | "
            f"threshold (Fit+Val Youden): {t_cal:.3f} | "
            f"threshold (Val-only Youden): {t_youden_val_only:.3f} | "
            f"test acc (animal@cal): {s1_test_acc:.3f} | "
            f"test acc@0.5: {s1_test_acc_0p5:.3f} | "
            f"AUC: {s1_test_auc:.3f}"
        )
    return {
        "clf": clf,
        "animal_pred_tr": animal_pred_tr,
        "animal_pred_te": animal_pred_te,
        "animal_pred_non_test": animal_pred_non_test,
        "animal_prob_te": animal_prob.astype(float),
        "animal_y_te": animal_gt_te.astype(np.int8),
        "stage1_threshold": float(t_cal),
        "stage1_threshold_val_youden": float(t_youden_val_only),
        "s1_fit_acc_row": s1_fit_acc_row,
        "s1_fit_acc_animal": s1_fit_acc_animal,
        "s1_val_acc_pre": s1_val_acc_pre,
        "s1_val_acc_animal": s1_val_acc_animal,
        "s1_val_acc_animal_val_youden": s1_val_acc_animal_val_youden,
        "s1_non_test_acc_animal": s1_non_test_acc_animal,
        "s1_non_test_auc_animal": s1_non_test_auc_animal,
        "s1_val_acc": s1_val_acc_pre,
        "s1_test_acc": s1_test_acc,
        "s1_test_acc_0p5": s1_test_acc_0p5,
        "s1_test_auc": s1_test_auc,
        "noise_acc": s1_test_acc,
        "noise_auc": s1_test_auc,
    }


def fit_stage1_wide_rf(
    s1_wide_fit: pd.DataFrame,
    s1_wide_val: pd.DataFrame,
    wide_test: pd.DataFrame,
    noise_num: List[str],
    noise_log: List[str],
    *,
    random_state: int = 1,
    n_iter: int = 24,
    verbose: bool = True,
):
    """Stage-1 RF: HP on Validate row-level accuracy; test metrics post-aggregation."""
    prep = _stage1_noise_prep(noise_num, noise_log)
    feat_cols = noise_num + noise_log

    X_fit = s1_wide_fit[feat_cols].dropna()
    y_fit = s1_wide_fit.loc[X_fit.index, "noise_cat"].round().astype(int)
    X_val = s1_wide_val[feat_cols].dropna()
    X_te = wide_test[feat_cols].dropna()

    best_score = -1.0
    best_params: dict = {}
    sampler = ParameterSampler(_RF_PARAMS, n_iter=n_iter, random_state=random_state)
    for params in sampler:
        pipe = Pipeline(
            [("prep", prep), ("rf", RandomForestClassifier(random_state=1))]
        ).set_params(**params)
        pipe.fit(X_fit, y_fit)
        if len(X_val) == 0:
            continue
        proba_va = pd.Series(pipe.predict_proba(X_val)[:, 1], index=X_val.index)
        y_va = s1_wide_val.loc[X_val.index, "noise_cat"].round().astype(int)
        score = _row_noise_accuracy(y_va, proba_va)
        if score > best_score:
            best_score = score
            best_params = params

    if not best_params:
        best_params = next(ParameterSampler(_RF_PARAMS, n_iter=1, random_state=random_state))

    clf = Pipeline(
        [("prep", prep), ("rf", RandomForestClassifier(random_state=1))]
    ).set_params(**best_params)
    clf.fit(X_fit, y_fit)
    return _pack_stage1_result(
        clf, s1_wide_fit, s1_wide_val, wide_test, X_fit, X_val, X_te, feat_cols,
        verbose=verbose, label="Noise RF",
    )


def fit_stage1_wide_logistic(
    s1_wide_fit: pd.DataFrame,
    s1_wide_val: pd.DataFrame,
    wide_test: pd.DataFrame,
    noise_num: List[str],
    noise_log: List[str],
    *,
    random_state: int = 1,
    verbose: bool = True,
):
    """Stage-1 L2 logistic: HP on Validate row-level accuracy; test metrics post-aggregation."""
    prep = _stage1_noise_prep(noise_num, noise_log)
    feat_cols = noise_num + noise_log

    X_fit = s1_wide_fit[feat_cols].dropna()
    y_fit = s1_wide_fit.loc[X_fit.index, "noise_cat"].round().astype(int)
    X_val = s1_wide_val[feat_cols].dropna()
    X_te = wide_test[feat_cols].dropna()

    best_score = -1.0
    best_C = float(_LR_C_GRID[0])
    for C in _LR_C_GRID:
        pipe = Pipeline(
            [
                ("prep", prep),
                (
                    "lr",
                    LogisticRegression(
                        C=C,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        pipe.fit(X_fit, y_fit)
        if len(X_val) == 0:
            continue
        proba_va = pd.Series(pipe.predict_proba(X_val)[:, 1], index=X_val.index)
        y_va = s1_wide_val.loc[X_val.index, "noise_cat"].round().astype(int)
        val_acc = _row_noise_accuracy(y_va, proba_va)
        if val_acc > best_score or (val_acc == best_score and float(C) < best_C):
            best_score = val_acc
            best_C = float(C)

    clf = Pipeline(
        [
            ("prep", prep),
            (
                "lr",
                LogisticRegression(
                    C=best_C,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    clf.fit(X_fit, y_fit)
    return _pack_stage1_result(
        clf, s1_wide_fit, s1_wide_val, wide_test, X_fit, X_val, X_te, feat_cols,
        verbose=verbose, label="Noise LR",
    )


def _pick_stage1_candidate(rf: dict, lr: dict) -> tuple[dict, str]:
    """Pick RF vs LR by non-test animal accuracy; ties broken by non-test animal AUC (else RF)."""
    r_acc = rf["s1_non_test_acc_animal"]
    l_acc = lr["s1_non_test_acc_animal"]
    if np.isfinite(l_acc) and np.isfinite(r_acc):
        if l_acc > r_acc:
            return lr, "lr"
        if r_acc > l_acc:
            return rf, "rf"
        r_auc = rf["s1_non_test_auc_animal"]
        l_auc = lr["s1_non_test_auc_animal"]
        if np.isfinite(l_auc) and np.isfinite(r_auc):
            if l_auc > r_auc:
                return lr, "lr"
            if r_auc > l_auc:
                return rf, "rf"
    return rf, "rf"


def fit_stage1_wide_best(
    s1_wide_fit: pd.DataFrame,
    s1_wide_val: pd.DataFrame,
    wide_test: pd.DataFrame,
    noise_num: List[str],
    noise_log: List[str],
    *,
    random_state: int = 1,
    n_iter: int = 24,
    verbose: bool = True,
) -> dict:
    """RF vs LR; pick by non-test animal acc; tie-break by non-test animal AUC; else RF."""
    rf = fit_stage1_wide_rf(
        s1_wide_fit, s1_wide_val, wide_test, noise_num, noise_log,
        random_state=random_state, n_iter=n_iter, verbose=False,
    )
    lr = fit_stage1_wide_logistic(
        s1_wide_fit, s1_wide_val, wide_test, noise_num, noise_log,
        random_state=random_state, verbose=False,
    )
    winner, name = _pick_stage1_candidate(rf, lr)
    out = {**winner, "stage1_model": name, "candidates": {"rf": rf, "lr": lr}}
    if verbose:
        print(
            f"           Selected {name.upper()}  — fit acc (animal@cal): {out['s1_fit_acc_animal']:.3f} | "
            f"non-test acc (animal@cal): {out['s1_non_test_acc_animal']:.3f} | "
            f"threshold (Fit+Val): {out['stage1_threshold']:.3f} | "
            f"test acc (animal@cal): {out['s1_test_acc']:.3f} | "
            f"test acc@0.5: {out['s1_test_acc_0p5']:.3f} | "
            f"AUC: {out['s1_test_auc']:.3f}"
        )
    return out


def assert_train_noise_preds_coverage(
    train_df: pd.DataFrame,
    animal_pred: pd.Series,
    *,
    id_col: str = "animal_id",
) -> None:
    need = set(train_df[id_col].astype(str).unique())
    have = set(pd.Index(animal_pred.index).astype(str))
    missing = need - have
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"Stage 1 missing noise_preds for {len(missing)} animals; e.g. {sample}"
        )


def attach_noise_preds_long(
    long_df: pd.DataFrame,
    animal_pred: pd.Series,
    *,
    fallback_noise_cat: bool = False,
    require_full_coverage: bool = False,
) -> pd.DataFrame:
    """Broadcast animal-level ``animal_pred`` to every row (same label per animal_id)."""
    out = long_df.copy().reset_index(drop=True)
    out["noise_preds"] = out["animal_id"].map(animal_pred)
    if fallback_noise_cat:
        miss = out["noise_preds"].isna()
        if miss.any():
            out.loc[miss, "noise_preds"] = (
                out.loc[miss, "noise_cat"].round().astype(int)
            )
    if require_full_coverage:
        assert_train_noise_preds_coverage(out, animal_pred)
    per_animal_nuniq = out.groupby("animal_id")["noise_preds"].nunique(dropna=True)
    if (per_animal_nuniq > 1).any():
        bad = per_animal_nuniq[per_animal_nuniq > 1].index[:3].tolist()
        raise ValueError(
            f"noise_preds not constant within animal_id; e.g. {bad}"
        )
    return out


def syn_prep_transformer(long_num, long_cat, long_log):
    return ColumnTransformer(
        [
            ("num", StandardScaler(), long_num),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), long_cat),
            ("log", mk_log_pipe(), long_log),
        ]
    )


def prepare_long_xy(
    long_df: pd.DataFrame,
    long_num: List[str],
    long_cat: List[str],
    long_log: List[str],
    prep: ColumnTransformer,
    *,
    fit: bool,
) -> Tuple[pd.Index, np.ndarray, np.ndarray]:
    cols = long_num + long_cat + long_log
    X_df = long_df[cols]
    valid = X_df.dropna().index
    X_part = X_df.loc[valid]
    if fit:
        X_t = prep.fit_transform(X_part)
    else:
        X_t = prep.transform(X_part)
    y = long_df.loc[valid, "synapses"].values.astype(np.float32)
    return valid, np.asarray(X_t, dtype=np.float32), y


class MLPRegressor(nn.Module):
    def __init__(self, n_in: int, hidden: Tuple[int, ...] = (128, 64), dropout: float = 0.2):
        super().__init__()
        layers = []
        d = n_in
        for h in hidden:
            layers.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)])
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNNPlusTabular(nn.Module):
    """Hybrid Stage-2 model: 1D-CNN on resampled Wave-I segment + MLP on tabular features."""

    def __init__(self, tab_dim: int, dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.tab = nn.Sequential(
            nn.Linear(tab_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 48),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(32 + 48, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, wave, tab):
        # wave: (B, 30) -> (B, 1, 30)
        z = self.cnn(wave.unsqueeze(1)).squeeze(-1)
        t = self.tab(tab)
        return self.head(torch.cat([z, t], dim=1)).squeeze(-1)


def _train_loop(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    second_input: Optional[np.ndarray] = None,
    second_val: Optional[np.ndarray] = None,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    huber: bool = True,
    device: Optional[torch.device] = None,
    forward_fn: Optional[Callable] = None,
) -> nn.Module:
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=1.0) if huber else nn.MSELoss()

    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train)
    Xv = torch.from_numpy(X_val)
    yv = torch.from_numpy(y_val)

    if second_input is not None:
        St = torch.from_numpy(second_input)
        Sv = torch.from_numpy(second_val)  # type: ignore
    n = Xt.shape[0]

    best_state = None
    best_val = float("inf")
    patience, bad = 15, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = Xt[idx].to(device)
            yb = yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            if forward_fn is None:
                pred = model(xb)
            else:
                sb = St[idx].to(device)
                pred = forward_fn(model, xb, sb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            if forward_fn is None:
                pv = model(Xv.to(device)).cpu().numpy()
            else:
                pv = forward_fn(model, Xv.to(device), Sv.to(device)).cpu().numpy()
            vl = float(np.mean((pv - yv.numpy()) ** 2))
        if vl < best_val - 1e-6:
            best_val = vl
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def resolve_full_wave_len(data: NNStage2Data, cfg: RunConfig) -> int:
    if cfg.full_wave_target_len is not None:
        return cfg.full_wave_target_len
    ref = (cfg.full_wave_ref or "lib").strip().lower()
    if ref in ("brad", "bb", "b"):
        return data.full_wave_target_len_brad
    return data.full_wave_target_len_lib


def aggregate_r2_rmse(
    long_te: pd.DataFrame,
    valid_idx: pd.Index,
    y_pred: np.ndarray,
    *,
    return_dataframe: bool = False,
):
    _eval = long_te.loc[valid_idx, ["animal_id", "frequency", "synapses"]].copy()
    _eval["y_pred"] = pd.Series(y_pred, index=valid_idx)
    _agg = _eval.groupby(["animal_id", "frequency"]).agg(
        y_true=("synapses", "first"),
        y_pred=("y_pred", "mean"),
    )
    r2 = r2_score(_agg["y_true"], _agg["y_pred"])
    rmse = float(np.sqrt(np.mean((_agg["y_true"] - _agg["y_pred"]) ** 2)))
    if return_dataframe:
        return r2, rmse, _agg.reset_index()
    return r2, rmse


@dataclass
class RunConfig:
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    huber: bool = True
    val_frac: float = 0.1  # fallback only if no Validate rows after dropna
    random_state: int = NN_TRAIN_RANDOM_STATE
    split_random_state: int = MOUSE_SPLIT_RANDOM_STATE
    full_wave_target_len: Optional[int] = None
    full_wave_ref: str = "lib"


def run_two_stage_long_nn(
    wide_train: pd.DataFrame,
    wide_test: pd.DataFrame,
    long_train: pd.DataFrame,
    long_test: pd.DataFrame,
    label: str,
    data: NNStage2Data,
    *,
    s1_wide_train: Optional[pd.DataFrame] = None,
    noise_num: Optional[List[str]] = None,
    noise_log: Optional[List[str]] = None,
    mode: str = "mlp",
    cfg: Optional[RunConfig] = None,
    verbose: bool = True,
    return_agg: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, pd.DataFrame]]:
    cfg = cfg or RunConfig()
    if s1_wide_train is not None and s1_wide_train is not wide_train:
        warnings.warn(
            "s1_wide_train is deprecated; Stage 1 pool must equal wide_train",
            stacklevel=2,
        )
    s1_pool = wide_train
    s1_fit = wide_stage1_fit(s1_pool)
    s1_val = wide_stage1_val(s1_pool)
    noise_num = noise_num or data.noise_num_common
    noise_log = noise_log or data.noise_log_common
    long_num = data.long_num
    long_cat = data.long_cat
    long_log = data.long_log

    if verbose:
        extra = ""
        if mode in ("cnn", "cnn_full"):
            extra = f" | full_wave_ref={cfg.full_wave_ref}"
        print(
            f"  [{label}]  S1 fit/val wide: {len(s1_fit)}/{len(s1_val)} | "
            f"long train: {len(long_train)} rows | Stage2={mode}{extra}"
        )

    s1 = fit_stage1_wide_best(
        s1_fit,
        s1_val,
        wide_test,
        noise_num,
        noise_log,
        random_state=cfg.random_state,
        verbose=verbose,
    )

    long_tr = attach_noise_preds_long(
        long_train,
        s1["animal_pred_non_test"],
        fallback_noise_cat=False,
        require_full_coverage=True,
    )
    long_te = attach_noise_preds_long(
        long_test, s1["animal_pred_te"], fallback_noise_cat=False
    )

    if "DataGroup" not in long_tr.columns:
        raise ValueError("long_train requires DataGroup column")
    long_fit = long_tr[long_tr["DataGroup"] == "Train"]
    long_val = long_tr[long_tr["DataGroup"] == "Validate"]
    fit_animals = set(long_fit["animal_id"].unique())
    val_animals = set(long_val["animal_id"].unique())
    if fit_animals & val_animals:
        raise ValueError("Train and Validate animals must be disjoint")

    prep = syn_prep_transformer(long_num, long_cat, long_log)
    tr_idx, X_sub, y_sub = prepare_long_xy(
        long_fit, long_num, long_cat, long_log, prep, fit=True
    )

    use_datagroup_val = len(long_val) > 0
    va_idx: pd.Index
    if use_datagroup_val:
        va_idx, X_va, y_va = prepare_long_xy(
            long_val, long_num, long_cat, long_log, prep, fit=False
        )
    else:
        warnings.warn(
            "No Validate rows for Stage 2; falling back to random val_frac holdout",
            stacklevel=2,
        )
        rng = np.random.RandomState(cfg.split_random_state)
        n = X_sub.shape[0]
        if n < 10:
            raise ValueError("too few training rows after dropna")
        sh = rng.permutation(n)
        n_val = max(1, int(n * cfg.val_frac))
        va_ix = sh[:n_val]
        tr_ix = sh[n_val:]
        if tr_ix.size == 0:
            tr_ix, va_ix = sh, sh[:1]
        X_va, y_va = X_sub[va_ix], y_sub[va_ix]
        X_sub, y_sub = X_sub[tr_ix], y_sub[tr_ix]
        tr_idx = tr_idx[tr_ix]

    if verbose:
        print(
            f"           Stage2 rows — train: {len(long_fit)} | validate: {len(long_val)} | "
            f"animals train/val: {len(fit_animals)}/{len(val_animals)}"
        )

    n_train = X_sub.shape[0]
    if n_train < 10:
        raise ValueError("too few training rows after dropna")

    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    wave_len_full = resolve_full_wave_len(data, cfg)
    if verbose and mode in ("cnn", "cnn_full"):
        print(f"           Unified 0–8 ms resampled length T={wave_len_full}")

    if mode == "mlp":
        model = MLPRegressor(X_sub.shape[1])
        model = _train_loop(
            model,
            X_sub,
            y_sub,
            X_va,
            y_va,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            huber=cfg.huber,
        )
        device = next(model.parameters()).device
        model.eval()
        te_idx, X_te, y_te = prepare_long_xy(
            long_te, long_num, long_cat, long_log, prep, fit=False
        )
        with torch.no_grad():
            pred = (
                model(torch.from_numpy(X_te).to(device)).cpu().numpy().astype(np.float64)
            )
    elif mode == "cnn":
        W_sub = unified_grid_wave_i_matrix(long_fit, tr_idx, wave_len_full)
        if use_datagroup_val:
            W_va_s = unified_grid_wave_i_matrix(long_val, va_idx, wave_len_full)
        else:
            W_all = unified_grid_wave_i_matrix(
                long_fit, tr_idx if isinstance(tr_idx, pd.Index) else pd.Index(tr_idx),
                wave_len_full,
            )
            rng = np.random.RandomState(cfg.split_random_state)
            n_w = W_all.shape[0]
            sh = rng.permutation(n_w)
            n_val = max(1, int(n_w * cfg.val_frac))
            W_va_s = W_all[sh[:n_val]]

        model = CNNPlusTabular(X_sub.shape[1])

        def fwd(m, xb, wb):
            return m(wb, xb)

        model = _train_loop(
            model,
            X_sub,
            y_sub,
            X_va,
            y_va,
            second_input=W_sub,
            second_val=W_va_s,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            huber=cfg.huber,
            forward_fn=fwd,
        )
        device = next(model.parameters()).device
        te_idx, X_te, y_te = prepare_long_xy(
            long_te, long_num, long_cat, long_log, prep, fit=False
        )
        W_te = unified_grid_wave_i_matrix(long_te, te_idx, wave_len_full)
        model.eval()
        with torch.no_grad():
            pred = (
                fwd(
                    model,
                    torch.from_numpy(X_te).to(device),
                    torch.from_numpy(W_te).to(device),
                )
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    elif mode == "cnn_full":
        W_sub = full_wave_matrix(long_fit, tr_idx, wave_len_full)
        if use_datagroup_val:
            W_va_s = full_wave_matrix(long_val, va_idx, wave_len_full)
        else:
            W_all = full_wave_matrix(long_fit, tr_idx, wave_len_full)
            rng = np.random.RandomState(cfg.split_random_state)
            n_w = W_all.shape[0]
            sh = rng.permutation(n_w)
            n_val = max(1, int(n_w * cfg.val_frac))
            W_va_s = W_all[sh[:n_val]]

        model = CNNPlusTabular(X_sub.shape[1])

        def fwd_full(m, xb, wb):
            return m(wb, xb)

        model = _train_loop(
            model,
            X_sub,
            y_sub,
            X_va,
            y_va,
            second_input=W_sub,
            second_val=W_va_s,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            huber=cfg.huber,
            forward_fn=fwd_full,
        )
        device = next(model.parameters()).device
        te_idx, X_te, y_te = prepare_long_xy(
            long_te, long_num, long_cat, long_log, prep, fit=False
        )
        W_te = full_wave_matrix(long_te, te_idx, wave_len_full)
        model.eval()
        with torch.no_grad():
            pred = (
                fwd_full(
                    model,
                    torch.from_numpy(X_te).to(device),
                    torch.from_numpy(W_te).to(device),
                )
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    else:
        raise ValueError("mode must be 'mlp', 'cnn', or 'cnn_full'")

    agg_out = aggregate_r2_rmse(
        long_te, te_idx, pred, return_dataframe=return_agg
    )
    if return_agg:
        r2, rmse, agg_df = agg_out
    else:
        r2, rmse = agg_out  # type: ignore[misc]
    if verbose:
        print(f"           Synapse NN — test R²: {r2:.3f}, RMSE: {rmse:.3f}")
    if return_agg:
        return r2, rmse, agg_df
    return r2, rmse


def bb_long_s2_rows(data: NNStage2Data, splits: Dict):
    """Six-tuples: wide_train, wide_test, long_train, long_test, noise_num, noise_log."""
    _bb_wide_train = splits["bb_wide_train"]
    _bb_wide_test = splits["bb_wide_test"]
    _bb_long_train = splits["bb_long_train"]
    _bb_long_test = splits["bb_long_test"]
    ref_o = data.reformatted_orig
    lib_long_train = splits["lib_long_train"]
    nc = data.noise_num_common
    nlog = data.noise_log_common

    return [
        (
            _bb_wide_train,
            _bb_wide_test,
            _bb_long_train,
            _bb_long_test,
            data.noise_num_bb,
            data.noise_log_bb,
        ),
        (
            pd.concat([_bb_wide_train, ref_o], ignore_index=True),
            _bb_wide_test,
            pd.concat([_bb_long_train, lib_long_train], ignore_index=True),
            _bb_long_test,
            nc,
            nlog,
        ),
        (
            ref_o.reset_index(drop=True),
            _bb_wide_test,
            lib_long_train,
            _bb_long_test,
            data.noise_num_lib,
            data.noise_log_lib,
        ),
    ]


def lib_long_s2_rows(data: NNStage2Data, splits: Dict):
    """Six-tuples: wide_train, wide_test, long_train, long_test, noise_num, noise_log."""
    ref = data.reformatted
    ref_o = data.reformatted_orig
    bb_long_train = splits["bb_long_train"]
    _lib_train = splits["lib_train"]
    _lib_test = splits["lib_test"]
    _lib_long_train = splits["lib_long_train"]
    _lib_long_test = splits["lib_long_test"]
    nc = data.noise_num_common
    nlog = data.noise_log_common

    return [
        (
            ref.reset_index(drop=True),
            _lib_test,
            bb_long_train,
            _lib_long_test,
            data.noise_num_bb,
            data.noise_log_bb,
        ),
        (
            pd.concat([ref.reset_index(drop=True), _lib_train], ignore_index=True),
            _lib_test,
            pd.concat([bb_long_train, _lib_long_train], ignore_index=True),
            _lib_long_test,
            nc,
            nlog,
        ),
        (
            _lib_train,
            _lib_test,
            _lib_long_train,
            _lib_long_test,
            data.noise_num_lib,
            data.noise_log_lib,
        ),
    ]


def long_s2_triple_nn(
    runner,
    rows: List[Tuple],
    data: NNStage2Data,
    *,
    mode: str,
    cfg: Optional[RunConfig] = None,
    **runner_kw,
):
    hdr = (
        "\n[A] Train on Brad Buran's dataset only:",
        "\n[B] Train on Brad Buran's + Liberman dataset:",
        "\n[C] Train on Liberman dataset only:",
    )
    out = {}
    for lab, row, h in zip("ABC", rows, hdr):
        print(h)
        wt, wte, lt, lte, nn, nl = row
        out[lab] = runner(
            wt,
            wte,
            lt,
            lte,
            lab,
            data,
            noise_num=nn,
            noise_log=nl,
            mode=mode,
            cfg=cfg,
            **runner_kw,
        )
    return out

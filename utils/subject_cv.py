"""Subject-level cross-validation for sklearn hyperparameter search."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    GroupKFold,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)

DEFAULT_CV_SPLITS = 5
DEFAULT_CV_RANDOM_STATE = 1
DEFAULT_CEILING_CV_SPLITS = 10
DEFAULT_CEILING_CV_RANDOM_STATE = 22

# Animal-level experimental group (Brad ``tx``, Liberman ``Group``) on wide frames.
EXPERIMENTAL_GROUP_COL = "experimental_group"
# Ceiling prediction strata (matches Stage 2 / §5.2: binary noise + frequency).
DEFAULT_CEILING_STRATUM_COLS = ("noise_cat", "frequency")

SearchCls = Type[Union[RandomizedSearchCV, GridSearchCV]]


def resolve_subject_col(df: pd.DataFrame, id_col: Optional[str] = None) -> str:
    if id_col is not None:
        if id_col not in df.columns:
            raise KeyError(f"id_col {id_col!r} not in frame columns")
        return id_col
    if "animal_id" in df.columns:
        return "animal_id"
    if "Subject" in df.columns:
        return "Subject"
    raise KeyError("frame must contain 'animal_id' or 'Subject'")


def train_only_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "DataGroup" not in df.columns:
        return df
    return df.loc[df["DataGroup"] == "Train"].copy()


def effective_n_splits(n_groups: int, n_splits: int = DEFAULT_CV_SPLITS) -> int:
    if n_groups < 2:
        raise ValueError(
            f"Need at least 2 subjects for grouped CV, got {n_groups} unique groups"
        )
    return max(2, min(n_splits, n_groups))


def prepare_xy_groups(
    frame: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    id_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, str]:
    """Train-only rows, dropna on features, aligned X / y / groups."""
    sub_col = resolve_subject_col(frame, id_col)
    sub = train_only_frame(frame)
    X = sub[feature_cols].dropna()
    y = sub.loc[X.index, target_col]
    groups = sub.loc[X.index, sub_col].to_numpy()
    return X, y, groups, sub_col


def build_subject_cv(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_CV_RANDOM_STATE,
    *,
    prefer_stratified: bool = True,
) -> BaseCrossValidator:
    """
    Subject-level CV splitter.

    When ``prefer_stratified=True``, use ``StratifiedGroupKFold`` only (no
    ``GroupKFold`` fallback). Regression HP search uses ``prefer_stratified=False``.
    """
    n_unique = len(np.unique(groups))
    n_splits = effective_n_splits(n_unique, n_splits)
    n_samples = len(y)

    if prefer_stratified:
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        list(cv.split(np.zeros(n_samples), y, groups))
        return cv

    return GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def prepare_strat_groups(
    frame: pd.DataFrame,
    strat_col: str,
    id_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Row-aligned group ids and integer strat labels (constant within each subject)."""
    sub_col = resolve_subject_col(frame, id_col)
    groups = frame[sub_col].to_numpy()
    per_animal = frame.groupby(sub_col)[strat_col].first()
    if pd.api.types.is_numeric_dtype(per_animal):
        animal_strat = per_animal.round().astype(int)
    else:
        codes, _ = pd.factorize(per_animal.astype(str))
        animal_strat = pd.Series(codes, index=per_animal.index, dtype=int)
    strat = frame[sub_col].map(animal_strat).astype(int).to_numpy()
    return groups, strat, sub_col


def eval_stratum_mean_baseline_cv(
    frame: pd.DataFrame,
    stratum_cols: List[str],
    *,
    target_col: str = "synapses",
    strat_col: str = EXPERIMENTAL_GROUP_COL,
    id_col: Optional[str] = None,
    n_splits: int = DEFAULT_CEILING_CV_SPLITS,
    random_state: int = DEFAULT_CEILING_CV_RANDOM_STATE,
    assert_no_leakage: bool = True,
) -> Dict[str, Any]:
    """
    Group-mean baseline with StratifiedGroupKFold on subjects.

    Folds stratify on ``strat_col`` (default ``experimental_group``). Each fold
    fits train-animal means per ``stratum_cols`` (default ``noise_cat``×
    ``frequency``; missing strata → global train mean), scores held-out animals,
    and returns mean ± SEM metrics plus out-of-fold predictions aligned to
    ``frame.index``.
    """
    from sklearn.metrics import r2_score

    work = frame.reset_index(drop=True)
    groups, strat, _ = prepare_strat_groups(work, strat_col, id_col)
    cv = build_subject_cv(
        strat,
        groups,
        n_splits=n_splits,
        random_state=random_state,
        prefer_stratified=True,
    )
    if assert_no_leakage:
        assert_no_group_leakage(
            cv,
            work[stratum_cols + [target_col]],
            pd.Series(strat),
            groups,
        )

    pred_oof = np.full(len(work), np.nan, dtype=float)
    fold_rmse: List[float] = []
    fold_r2: List[float] = []
    split_x = np.zeros(len(work))

    for tr_idx, te_idx in cv.split(split_x, strat, groups):
        tr_rows = work.iloc[tr_idx]
        te_rows = work.iloc[te_idx]
        stratum_means = tr_rows.groupby(stratum_cols)[target_col].mean()
        global_mean = float(tr_rows[target_col].mean())
        multi_idx = pd.MultiIndex.from_frame(te_rows[stratum_cols])
        pred = (
            stratum_means.reindex(multi_idx)
            .astype(float)
            .fillna(global_mean)
            .to_numpy(dtype=float)
        )
        y = te_rows[target_col].to_numpy(dtype=float)
        fold_rmse.append(float(np.sqrt(np.mean((y - pred) ** 2))))
        fold_r2.append(float(r2_score(y, pred)))
        pred_oof[te_rows.index.to_numpy()] = pred

    n_folds = len(fold_rmse)
    return {
        "r2": float(np.mean(fold_r2)),
        "r2_sem": float(np.std(fold_r2, ddof=1) / np.sqrt(n_folds)),
        "rmse": float(np.mean(fold_rmse)),
        "rmse_sem": float(np.std(fold_rmse, ddof=1) / np.sqrt(n_folds)),
        "n_cv_folds": n_folds,
        "pred_oof": pred_oof,
    }


def assert_no_group_leakage(
    cv: BaseCrossValidator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
) -> None:
    """Raise if any subject appears in both train and validation for a fold."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    for tr_idx, va_idx in cv.split(X_arr, y_arr, groups):
        tr_g = set(groups[tr_idx])
        va_g = set(groups[va_idx])
        overlap = tr_g & va_g
        if overlap:
            raise AssertionError(
                f"Subject leakage in CV fold: {len(overlap)} groups in both splits"
            )


def fit_subject_search(
    estimator: Any,
    param_grid: Dict[str, List[Any]],
    frame: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    *,
    scoring: str = "r2",
    search_cls: SearchCls = RandomizedSearchCV,
    prefer_stratified: bool = False,
    strat_col: Optional[str] = None,
    n_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_CV_RANDOM_STATE,
    n_iter: int = 10,
    n_jobs: int = -1,
    id_col: Optional[str] = None,
    **search_kw: Any,
) -> Union[RandomizedSearchCV, GridSearchCV]:
    """
    Hyperparameter search on Train-only rows with subject-level CV.

    When ``prefer_stratified`` and ``strat_col`` are set, folds stratify on
    ``strat_col`` (e.g. experimental group) while the estimator still fits ``target_col``.

    Final estimator is refit on the same Train-only data (sklearn refit=True).
    """
    X, y, groups, sub_col = prepare_xy_groups(frame, feature_cols, target_col, id_col)
    if prefer_stratified and strat_col is not None:
        work = train_only_frame(frame).loc[X.index]
        _, cv_strat, _ = prepare_strat_groups(work, strat_col, sub_col)
    else:
        cv_strat = y.to_numpy()
    cv = build_subject_cv(
        cv_strat,
        groups,
        n_splits=n_splits,
        random_state=random_state,
        prefer_stratified=prefer_stratified,
    )

    common = dict(
        estimator=estimator,
        cv=cv,
        scoring=scoring,
        refit=True,
        n_jobs=n_jobs,
        **search_kw,
    )
    if search_cls is RandomizedSearchCV:
        search = RandomizedSearchCV(
            param_distributions=param_grid,
            random_state=random_state,
            n_iter=n_iter,
            **common,
        )
    else:
        search = search_cls(param_grid=param_grid, **common)

    search.fit(X, y, groups=groups)
    return search

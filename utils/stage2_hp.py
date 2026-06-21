"""Hyperparameter search for Stage-2 scenarios A/B/C (Section 5.3)."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from utils.liberman_classical import NoiseLabel, liberman_feature_lists, stage2_feature_lists
from utils.nn_stage2 import (
    MLPRegressor,
    _train_loop,
    attach_noise_preds_long,
    fit_stage1_wide_best,
    prepare_long_xy,
    stage1_animal_pred_covering,
    syn_prep_transformer,
)
from utils.nn_stage2_data import NNStage2Data, wide_stage1_fit, wide_stage1_val
from utils.nn_colab_train import (
    DROPOUT_GRID,
    LR_GRID,
    MLP_HIDDEN_GRID,
    WEIGHT_DECAY_GRID,
    _holdout_split,
    aggregate_animal_freq_metrics,
)
from utils.stage2_sklearn import _fit_stage2_rf, _fit_stage2_xgb
from utils.subject_cv import train_only_frame


def resolve_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def stage1_augment_wide(
    wide_df: pd.DataFrame,
    noise_num: List[str],
    noise_log: List[str],
) -> pd.DataFrame:
    return augment_wide_for_hp(wide_df, noise_num, noise_log, noise_label="predicted")


def augment_wide_for_hp(
    wide_df: pd.DataFrame,
    noise_num: List[str],
    noise_log: List[str],
    *,
    noise_label: NoiseLabel = "predicted",
) -> pd.DataFrame:
    wide = wide_df.reset_index(drop=True)
    if noise_label == "true":
        if "noise_cat" not in wide.columns:
            raise ValueError("true-noise HP requires noise_cat on wide frame")
        return wide
    wide_test = wide.loc[wide["DataGroup"].eq("Test")].reset_index(drop=True)
    s1 = fit_stage1_wide_best(
        wide_stage1_fit(wide),
        wide_stage1_val(wide),
        wide_test,
        noise_num,
        noise_log,
        random_state=1,
        verbose=False,
    )
    animal_pred = stage1_animal_pred_covering(wide, s1, noise_num, noise_log)
    return attach_noise_preds_long(
        wide,
        animal_pred,
        fallback_noise_cat=False,
        require_full_coverage=True,
    )


def hp_pool_non_test(
    data: NNStage2Data,
    scenario: Literal["A", "B"],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
]:
    """Backward-compatible wrapper: scenarios A/B only, predicted noise."""
    return hp_pool_for_scenario(data, scenario, noise_label="predicted")


def hp_pool_for_scenario(
    data: NNStage2Data,
    scenario: Literal["A", "B", "C"],
    *,
    noise_label: NoiseLabel = "predicted",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
]:
    """Conservative non-test pools for HP tuning (wide + long)."""
    feats = liberman_feature_lists(data.reformatted_orig, data.common_cols)
    syn_num, syn_log, syn_cat, long_num, long_cat = stage2_feature_lists(
        feats, noise_label=noise_label
    )
    long_log = list(feats["long_log"])
    bb_w = data.reformatted.loc[data.reformatted["DataGroup"] != "Test"].reset_index(
        drop=True
    )
    lib_w = data.reformatted_orig.loc[
        data.reformatted_orig["DataGroup"] != "Test"
    ].reset_index(drop=True)
    bb_l = data.brad_buran_df.loc[
        ~data.brad_buran_df["animal_id"].isin(
            set(data.reformatted.loc[data.reformatted["DataGroup"] == "Test", "animal_id"])
        )
    ].reset_index(drop=True)
    lib_l = data.orig_lib.loc[data.orig_lib["DataGroup"] != "Test"].reset_index(
        drop=True
    )

    if scenario == "A":
        wide, long = bb_w, bb_l
        nn, nl = list(data.noise_num_bb), list(data.noise_log_bb)
    elif scenario == "B":
        wide = pd.concat([bb_w, lib_w], ignore_index=True)
        long = pd.concat([bb_l, lib_l], ignore_index=True)
        nn, nl = list(data.noise_num_common), list(data.noise_log_common)
    else:
        wide, long = lib_w, lib_l
        nn, nl = list(data.noise_num_lib), list(data.noise_log_lib)

    return wide, long, nn, nl, syn_num, syn_log, syn_cat, long_num, long_cat, long_log


def tune_scenario_sklearn(
    wide_df: pd.DataFrame,
    noise_num: List[str],
    noise_log: List[str],
    syn_num: List[str],
    syn_cat: List[str],
    syn_log: List[str],
    *,
    noise_label: NoiseLabel = "predicted",
    rf_n_iter: int = 24,
) -> Dict[str, Any]:
    aug = augment_wide_for_hp(wide_df, noise_num, noise_log, noise_label=noise_label)
    tr = train_only_frame(aug).reset_index(drop=True)
    _, rf_cv, rf_bp = _fit_stage2_rf(
        tr, syn_num, syn_cat, syn_log, n_iter=rf_n_iter, random_state=1
    )
    _, xgb_cv, xgb_bp = _fit_stage2_xgb(tr, syn_num, syn_cat, syn_log)
    return {
        "RF": {"best_params": _json_sanitize(rf_bp), "cv_score": rf_cv},
        "XGB": {"best_params": _json_sanitize(xgb_bp), "cv_score": xgb_cv},
    }


def tune_scenario_mlp(
    wide_df: pd.DataFrame,
    long_df: pd.DataFrame,
    noise_num: List[str],
    noise_log: List[str],
    long_num: List[str],
    long_cat: List[str],
    long_log: List[str],
    *,
    noise_label: NoiseLabel = "predicted",
    device: torch.device | None = None,
    holdout_rs: int = 1,
    verbose: bool = False,
    max_trials: int | None = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Grid search; early stop on official Validate; pick by Train 10% holdout RMSE."""
    device = device or resolve_torch_device()
    if noise_label == "true":
        long_aug = long_df.reset_index(drop=True)
    else:
        aug_w = augment_wide_for_hp(wide_df, noise_num, noise_log, noise_label="predicted")
        s1 = fit_stage1_wide_best(
            wide_stage1_fit(wide_df),
            wide_stage1_val(wide_df),
            wide_df.loc[wide_df["DataGroup"].eq("Test")].reset_index(drop=True),
            noise_num,
            noise_log,
            verbose=False,
        )
        long_aug = attach_noise_preds_long(
            long_df.reset_index(drop=True),
            s1["animal_pred_non_test"],
            fallback_noise_cat=False,
            require_full_coverage=True,
        )
    long_tr = long_aug[long_aug["DataGroup"].eq("Train")].reset_index(drop=True)
    long_val = long_aug[long_aug["DataGroup"].eq("Validate")].reset_index(drop=True)
    if len(long_val) == 0:
        raise ValueError("MLP HP tuning requires official Validate rows")

    prep = syn_prep_transformer(long_num, long_cat, long_log)
    tr_idx, X_tr, y_tr = prepare_long_xy(long_tr, long_num, long_cat, long_log, prep, fit=True)
    _, X_va, y_va = prepare_long_xy(long_val, long_num, long_cat, long_log, prep, fit=False)
    ho_tr, ho_va = _holdout_split(len(y_tr), val_frac=0.1, random_state=holdout_rs)

    meta_tr = long_tr.loc[tr_idx].reset_index(drop=True)
    grid = list(
        itertools.product(MLP_HIDDEN_GRID, DROPOUT_GRID, LR_GRID, WEIGHT_DECAY_GRID)
    )
    if max_trials is not None:
        grid = grid[: int(max_trials)]
    rows: List[Dict[str, Any]] = []
    best_cfg: Dict[str, Any] | None = None
    best_rmse = float("inf")

    for i, (hidden, dropout, lr, wd) in enumerate(grid):
        torch.manual_seed(holdout_rs)
        np.random.seed(holdout_rs)
        hidden_t = tuple(int(h) for h in hidden)
        model = MLPRegressor(X_tr.shape[1], hidden=hidden_t, dropout=float(dropout))
        model = _train_loop(
            model,
            X_tr[ho_tr],
            y_tr[ho_tr],
            X_va,
            y_va,
            epochs=80,
            batch_size=256,
            lr=float(lr),
            weight_decay=float(wd),
            device=device,
        )
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X_tr[ho_va]).to(device)).cpu().numpy()
        rmse = float(np.sqrt(np.mean((y_tr[ho_va] - pred) ** 2)))
        r2_hold = float(r2_score(y_tr[ho_va], pred))
        _, rmse_af, _ = aggregate_animal_freq_metrics(
            meta_tr.iloc[ho_va].reset_index(drop=True), pred
        )
        row = {
            "hidden": list(hidden_t),
            "dropout": float(dropout),
            "lr": float(lr),
            "weight_decay": float(wd),
            "holdout_rmse_row": rmse,
            "holdout_rmse_animal_freq": rmse_af,
            "holdout_r2": r2_hold,
            "trial": i,
        }
        rows.append(row)
        if rmse_af < best_rmse:
            best_rmse = rmse_af
            best_cfg = {
                "hidden": list(hidden_t),
                "dropout": float(dropout),
                "lr": float(lr),
                "weight_decay": float(wd),
            }
        if verbose:
            print(f"  MLP trial {i+1}/{len(grid)} rmse_af={rmse_af:.4f}")

    assert best_cfg is not None
    return best_cfg, pd.DataFrame(rows)


def tune_scenario(
    data: NNStage2Data,
    scenario: Literal["A", "B", "C"],
    *,
    noise_label: NoiseLabel = "predicted",
    rf_n_iter: int = 24,
    mlp_max_trials: int | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    wide, long, nn, nl, syn_num, syn_log, syn_cat, long_num, long_cat, long_log = (
        hp_pool_for_scenario(data, scenario, noise_label=noise_label)
    )
    test_ids = set()
    if "Test" in wide["DataGroup"].values:
        test_ids = set(wide.loc[wide["DataGroup"] == "Test", "animal_id"])
    assert not test_ids, f"scenario {scenario}: Test animals in tuning wide frame"

    if verbose:
        print(
            f"Scenario {scenario} ({noise_label}): wide={len(wide)} rows "
            f"({wide['animal_id'].nunique()} animals), long={len(long)} rows"
        )

    sk = tune_scenario_sklearn(
        wide, nn, nl, syn_num, syn_cat, syn_log,
        noise_label=noise_label, rf_n_iter=rf_n_iter,
    )
    mlp_hp, mlp_trials = tune_scenario_mlp(
        wide,
        long,
        nn,
        nl,
        long_num,
        long_cat,
        long_log,
        noise_label=noise_label,
        device=device,
        verbose=verbose,
        max_trials=mlp_max_trials,
    )
    payload = {
        "scenario": scenario,
        "config_id": "T5",
        "format": "wide",
        "noise_label": noise_label,
        **sk,
        "MLP": mlp_hp,
    }
    mlp_trials["scenario"] = scenario
    mlp_trials["noise_label"] = noise_label
    return _json_sanitize(payload), mlp_trials


def tune_scenario_ab(
    data: NNStage2Data,
    scenario: Literal["A", "B"],
    *,
    rf_n_iter: int = 24,
    device: torch.device | None = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    return tune_scenario(
        data,
        scenario,
        noise_label="predicted",
        rf_n_iter=rf_n_iter,
        device=device,
        verbose=verbose,
    )


def write_scenario_hp_json(payload: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_sanitize(payload), indent=2) + "\n", encoding="utf-8")

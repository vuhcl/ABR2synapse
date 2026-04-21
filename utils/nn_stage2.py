"""
Stage-1 wide RF noise classifier + Stage-2 synapse regression: **long** rows (MLP or
1D-CNN + tabular) or **pivoted wide** rows (MLP tabular only — no waveforms on wide).

Targets are raw ``synapses``; tabular inputs use the same ColumnTransformer pattern as
``run_two_stage_long_s2`` / wide XGB in the comparison notebook.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from utils.nn_stage2_data import (
    NNStage2Data,
    full_wave_matrix,
    syn_feats_from_wide_common,
    unified_grid_wave_i_matrix,
)

_RF_PARAMS = {
    "rf__n_estimators": [25, 50, 100, 150, 200, 300, 400, 500],
    "rf__max_depth": [5, 10, 20, 30, 40, 50, None],
    "rf__min_samples_leaf": [2, 5, 10, 15, 20, 30],
    "rf__min_samples_split": [2, 5, 10, 15, 20, 30],
    "rf__max_features": ["sqrt", "log2", 0.3, 0.5],
}


def mk_log_pipe():
    return Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, validate=True)),
            ("scaler", StandardScaler()),
        ]
    )


def aggregate_noise_to_animal(df, indices, proba_col):
    rows = df.loc[indices, ["animal_id", "noise_cat"]].copy()
    rows["proba"] = proba_col.values
    agg_proba = rows.groupby("animal_id")["proba"].mean()
    return (agg_proba > 0.5).astype(int), agg_proba


def fit_stage1_wide_rf(
    s1_wide_train: pd.DataFrame,
    wide_test: pd.DataFrame,
    noise_num: List[str],
    noise_log: List[str],
    random_state: int = 1,
    verbose: bool = True,
):
    noise_prep = ColumnTransformer(
        [
            ("num", StandardScaler(), noise_num),
            ("log", mk_log_pipe(), noise_log),
        ]
    )
    noise_search = RandomizedSearchCV(
        Pipeline([("prep", noise_prep), ("rf", RandomForestClassifier(random_state=1))]),
        _RF_PARAMS,
        cv=5,
        random_state=random_state,
        n_jobs=-1,
    )
    Xn_tr = s1_wide_train[noise_num + noise_log].dropna()
    yn_tr = s1_wide_train.loc[Xn_tr.index, "noise_cat"].round().astype(int)
    Xn_te = wide_test[noise_num + noise_log].dropna()

    noise_search.fit(Xn_tr, yn_tr)
    clf = noise_search.best_estimator_

    proba_tr = pd.Series(clf.predict_proba(Xn_tr)[:, 1], index=Xn_tr.index)
    proba_te = pd.Series(clf.predict_proba(Xn_te)[:, 1], index=Xn_te.index)
    animal_pred_tr, _ = aggregate_noise_to_animal(s1_wide_train, Xn_tr.index, proba_tr)
    animal_pred_te, animal_prob = aggregate_noise_to_animal(
        wide_test, Xn_te.index, proba_te
    )
    animal_gt_te = (
        wide_test.loc[Xn_te.index, ["animal_id", "noise_cat"]]
        .groupby("animal_id")["noise_cat"]
        .first()
        .round()
        .astype(int)
    )
    noise_acc = (animal_pred_te == animal_gt_te).mean()
    noise_auc = roc_auc_score(animal_gt_te, animal_prob)
    if verbose:
        print(
            f"           Noise clf  — CV acc: {noise_search.best_score_:.3f} | "
            f"animal-level test acc: {noise_acc:.3f}, AUC: {noise_auc:.3f}"
        )
    return {
        "clf": clf,
        "animal_pred_tr": animal_pred_tr,
        "animal_pred_te": animal_pred_te,
        "s1_cv_acc": noise_search.best_score_,
        "noise_acc": noise_acc,
        "noise_auc": noise_auc,
    }


def attach_noise_preds_long(
    long_df: pd.DataFrame,
    animal_pred: pd.Series,
    *,
    fallback_noise_cat: bool,
) -> pd.DataFrame:
    out = long_df.copy().reset_index(drop=True)
    out["noise_preds"] = out["animal_id"].map(animal_pred)
    if fallback_noise_cat:
        miss = out["noise_preds"].isna()
        if miss.any():
            out.loc[miss, "noise_preds"] = (
                out.loc[miss, "noise_cat"].round().astype(int)
            )
    # Test split: keep NaN where Stage 1 has no animal-level pred (same as RF + dropna).
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
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    val_frac: float = 0.1
    random_state: int = 1
    # If set, overrides cohort-specific targets below for ``cnn`` / ``cnn_full``.
    full_wave_target_len: Optional[int] = None
    # ``lib`` = resample to Liberman median crop length; ``brad`` = Brad median.
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
    s1_wide_train = s1_wide_train if s1_wide_train is not None else wide_train
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
            f"  [{label}]  S1 wide train: {len(s1_wide_train)} | long train: {len(long_train)} rows | Stage2={mode}{extra}"
        )

    s1 = fit_stage1_wide_rf(
        s1_wide_train,
        wide_test,
        noise_num,
        noise_log,
        random_state=cfg.random_state,
        verbose=verbose,
    )

    long_tr = attach_noise_preds_long(
        long_train, s1["animal_pred_tr"], fallback_noise_cat=True
    )
    long_te = attach_noise_preds_long(
        long_test, s1["animal_pred_te"], fallback_noise_cat=False
    )

    prep = syn_prep_transformer(long_num, long_cat, long_log)
    tr_idx, X_tr, y_tr = prepare_long_xy(
        long_tr, long_num, long_cat, long_log, prep, fit=True
    )

    rng = np.random.RandomState(cfg.random_state)
    n = X_tr.shape[0]
    if n < 10:
        raise ValueError("too few training rows after dropna")
    sh = rng.permutation(n)
    n_val = max(1, int(n * cfg.val_frac))
    va = sh[:n_val]
    tr = sh[n_val:]
    if tr.size == 0:
        tr, va = sh, sh[:1]

    X_sub, y_sub = X_tr[tr], y_tr[tr]
    X_va, y_va = X_tr[va], y_tr[va]

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        te_idx, X_te, y_te = prepare_long_xy(
            long_te, long_num, long_cat, long_log, prep, fit=False
        )
        with torch.no_grad():
            pred = (
                model(torch.from_numpy(X_te).to(device)).cpu().numpy().astype(np.float64)
            )
    elif mode == "cnn":
        W_tr = unified_grid_wave_i_matrix(long_tr, tr_idx, wave_len_full)
        W_sub, W_va_s = W_tr[tr], W_tr[va]

        model = CNNPlusTabular(X_sub.shape[1])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        W_tr = full_wave_matrix(long_tr, tr_idx, wave_len_full)
        W_sub, W_va_s = W_tr[tr], W_tr[va]

        model = CNNPlusTabular(X_sub.shape[1])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def bb_long_s2_rows(data: NNStage2Data, splits: Dict, noise_num_bb: List[str]):
    _bb_wide_train = splits["bb_wide_train"]
    _bb_wide_test = splits["bb_wide_test"]
    _bb_long_train = splits["bb_long_train"]
    _bb_long_test = splits["bb_long_test"]
    ref_o = data.reformatted_orig
    lib_all = splits["lib_long_all"]

    return [
        (
            _bb_wide_train,
            _bb_wide_test,
            _bb_long_train,
            _bb_long_test,
            _bb_wide_train,
            noise_num_bb,
            data.noise_log_bb,
        ),
        (
            pd.concat([_bb_wide_train, ref_o], ignore_index=True),
            _bb_wide_test,
            pd.concat([_bb_long_train, lib_all], ignore_index=True),
            _bb_long_test,
            _bb_wide_train,
            noise_num_bb,
            data.noise_log_bb,
        ),
        (
            ref_o.reset_index(drop=True),
            _bb_wide_test,
            lib_all,
            _bb_long_test,
            _bb_wide_train,
            noise_num_bb,
            data.noise_log_bb,
        ),
    ]


def lib_long_s2_rows(data: NNStage2Data, splits: Dict, noise_num_lib: List[str]):
    ref = data.reformatted
    ref_o = data.reformatted_orig
    bb_all = splits["bb_long_all"]
    _lib_train = splits["lib_train"]
    _lib_test = splits["lib_test"]
    _lib_long_train = splits["lib_long_train"]
    _lib_long_test = splits["lib_long_test"]

    return [
        (
            ref.reset_index(drop=True),
            _lib_test,
            bb_all,
            _lib_long_test,
            _lib_train,
            noise_num_lib,
            data.noise_log_lib,
        ),
        (
            pd.concat([ref.reset_index(drop=True), _lib_train], ignore_index=True),
            _lib_test,
            pd.concat([bb_all, _lib_long_train], ignore_index=True),
            _lib_long_test,
            _lib_train,
            noise_num_lib,
            data.noise_log_lib,
        ),
        (
            _lib_train,
            _lib_test,
            _lib_long_train,
            _lib_long_test,
            _lib_train,
            noise_num_lib,
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
        wt, wte, lt, lte, s1, nn, nl = row
        out[lab] = runner(
            wt,
            wte,
            lt,
            lte,
            lab,
            data,
            s1_wide_train=s1,
            noise_num=nn,
            noise_log=nl,
            mode=mode,
            cfg=cfg,
            **runner_kw,
        )
    return out


def run_two_stage_wide_nn(
    wide_train: pd.DataFrame,
    wide_test: pd.DataFrame,
    wide_s2_train: pd.DataFrame,
    wide_s2_test: pd.DataFrame,
    label: str,
    data: NNStage2Data,
    *,
    s1_wide_train: Optional[pd.DataFrame] = None,
    noise_num: Optional[List[str]] = None,
    noise_log: Optional[List[str]] = None,
    syn_num: Optional[List[str]] = None,
    syn_log: Optional[List[str]] = None,
    syn_cat: Optional[List[str]] = None,
    mode: str = "mlp",
    cfg: Optional[RunConfig] = None,
    verbose: bool = True,
    return_agg: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, pd.DataFrame]]:
    """
    Stage 1: wide RF (same as long pipeline). Stage 2: **pivoted wide** tabular features
    and ``synapses`` target (mean per animal × frequency in the wide table).

    **CNN / cnn_full are not supported:** waveforms are defined per **long** row; wide
    rows do not carry raw traces aligned to each target.
    """
    cfg = cfg or RunConfig()
    if mode != "mlp":
        raise ValueError(
            "run_two_stage_wide_nn only supports mode='mlp'. "
            "Use run_two_stage_long_nn with mode='cnn' or 'cnn_full' for 1D-CNN on long rows."
        )
    s1_wide_train = s1_wide_train if s1_wide_train is not None else wide_train
    noise_num = noise_num or data.noise_num_common
    noise_log = noise_log or data.noise_log_common
    if syn_num is None or syn_log is None or syn_cat is None:
        syn_num, syn_log, syn_cat = syn_feats_from_wide_common(
            list(data.common_cols),
            has_strain=data.has_strain,
        )

    if verbose:
        print(
            f"  [{label}]  S1 wide train: {len(s1_wide_train)} | "
            f"wide Stage2 train: {len(wide_s2_train)} rows | Stage2=mlp (wide tabular)"
        )

    s1 = fit_stage1_wide_rf(
        s1_wide_train,
        wide_test,
        noise_num,
        noise_log,
        random_state=cfg.random_state,
        verbose=verbose,
    )

    w_tr = attach_noise_preds_long(
        wide_s2_train, s1["animal_pred_tr"], fallback_noise_cat=True
    )
    w_te = attach_noise_preds_long(
        wide_s2_test, s1["animal_pred_te"], fallback_noise_cat=False
    )

    prep = syn_prep_transformer(syn_num, syn_cat, syn_log)
    tr_idx, X_tr, y_tr = prepare_long_xy(
        w_tr, syn_num, syn_cat, syn_log, prep, fit=True
    )

    rng = np.random.RandomState(cfg.random_state)
    n = X_tr.shape[0]
    if n < 10:
        raise ValueError("too few training rows after dropna")
    sh = rng.permutation(n)
    n_val = max(1, int(n * cfg.val_frac))
    va = sh[:n_val]
    tr = sh[n_val:]
    if tr.size == 0:
        tr, va = sh, sh[:1]

    X_sub, y_sub = X_tr[tr], y_tr[tr]
    X_va, y_va = X_tr[va], y_tr[va]

    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    te_idx, X_te, y_te = prepare_long_xy(
        w_te, syn_num, syn_cat, syn_log, prep, fit=False
    )
    with torch.no_grad():
        pred = (
            model(torch.from_numpy(X_te).to(device)).cpu().numpy().astype(np.float64)
        )

    agg_out = aggregate_r2_rmse(w_te, te_idx, pred, return_dataframe=return_agg)
    if return_agg:
        r2, rmse, agg_df = agg_out
    else:
        r2, rmse = agg_out  # type: ignore[misc]
    if verbose:
        print(f"           Synapse NN — test R²: {r2:.3f}, RMSE: {rmse:.3f}")
    if return_agg:
        return r2, rmse, agg_df
    return r2, rmse


def bb_wide_s2_rows(data: NNStage2Data, splits: Dict, noise_num_bb: List[str]):
    _bb_wide_train = splits["bb_wide_train"]
    _bb_wide_test = splits["bb_wide_test"]
    ref_o = data.reformatted_orig

    return [
        (
            _bb_wide_train,
            _bb_wide_test,
            _bb_wide_train,
            _bb_wide_test,
            _bb_wide_train,
            noise_num_bb,
            data.noise_log_bb,
        ),
        (
            pd.concat([_bb_wide_train, ref_o], ignore_index=True),
            _bb_wide_test,
            pd.concat([_bb_wide_train, ref_o], ignore_index=True),
            _bb_wide_test,
            _bb_wide_train,
            noise_num_bb,
            data.noise_log_bb,
        ),
        (
            ref_o.reset_index(drop=True),
            _bb_wide_test,
            ref_o.reset_index(drop=True),
            _bb_wide_test,
            _bb_wide_train,
            noise_num_bb,
            data.noise_log_bb,
        ),
    ]


def lib_wide_s2_rows(data: NNStage2Data, splits: Dict, noise_num_lib: List[str]):
    ref = data.reformatted
    _lib_train = splits["lib_train"]
    _lib_test = splits["lib_test"]

    return [
        (
            ref.reset_index(drop=True),
            _lib_test,
            ref.reset_index(drop=True),
            _lib_test,
            _lib_train,
            noise_num_lib,
            data.noise_log_lib,
        ),
        (
            pd.concat([ref.reset_index(drop=True), _lib_train], ignore_index=True),
            _lib_test,
            pd.concat([ref.reset_index(drop=True), _lib_train], ignore_index=True),
            _lib_test,
            _lib_train,
            noise_num_lib,
            data.noise_log_lib,
        ),
        (
            _lib_train,
            _lib_test,
            _lib_train,
            _lib_test,
            _lib_train,
            noise_num_lib,
            data.noise_log_lib,
        ),
    ]


def wide_s2_triple_nn(
    runner,
    rows: List[Tuple],
    data: NNStage2Data,
    *,
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
        wt, wte, w2t, w2te, s1, nn, nl = row
        out[lab] = runner(
            wt,
            wte,
            w2t,
            w2te,
            lab,
            data,
            s1_wide_train=s1,
            noise_num=nn,
            noise_log=nl,
            cfg=cfg,
            **runner_kw,
        )
    return out

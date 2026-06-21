"""
HP search + final training for Liberman NN Stage-2 on exported Colab packs.

Self-contained: copied into the artifact folder as ``nn_colab_train.py`` for Colab
(upload folder to Drive; no full repo required).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

NN_TRAIN_RANDOM_STATE = 1

# Smaller MLP grid (72 configs): drop standalone (128,), drop wd=1e-5.
MLP_HIDDEN_GRID = [(64,), (128, 64), (256, 128), (128, 64, 32)]
DROPOUT_GRID = [0.1, 0.2, 0.3]
LR_GRID = [1e-4, 1e-3, 5e-3]
WEIGHT_DECAY_GRID = [1e-4, 1e-3]
MLP_GRID_SIZE = len(MLP_HIDDEN_GRID) * len(DROPOUT_GRID) * len(LR_GRID) * len(WEIGHT_DECAY_GRID)


def _format_mlp_hp(hidden: tuple[int, ...], dropout: float, lr: float, wd: float) -> str:
    return (
        f"hidden={hidden} dropout={dropout} lr={lr:g} wd={wd:g}"
    )


class MLPRegressor(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        d = n_in
        for h in hidden:
            layers.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)])
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNNPlusTabular(nn.Module):
    """1D-CNN on waveform + MLP on tabular features."""

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
    device: Optional[torch.device] = None,
    forward_fn: Optional[Callable] = None,
) -> nn.Module:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=1.0)

    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train)
    Xv = torch.from_numpy(X_val)
    yv = torch.from_numpy(y_val)
    if second_input is not None:
        St = torch.from_numpy(second_input)
        Sv = torch.from_numpy(second_val)  # type: ignore[arg-type]
    n = Xt.shape[0]

    best_state = None
    best_val = float("inf")
    patience, bad = 15, 0

    for _epoch in range(epochs):
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
                pred = forward_fn(model, xb, St[idx].to(device))
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


def load_colab_pack(data_dir: Path | str) -> dict[str, Any]:
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())

    def _split(name: str) -> dict[str, np.ndarray]:
        df = pd.read_parquet(data_dir / f"{name}.parquet")
        tab_cols = manifest["tabular_columns"]
        return {
            "meta": df,
            "X_tab": df[tab_cols].to_numpy(dtype=np.float32),
            "y": df["synapses"].to_numpy(dtype=np.float32),
            "wave_i": np.load(data_dir / f"wave_i_{name}.npy"),
            "wave_full": np.load(data_dir / f"wave_full_{name}.npy"),
        }

    return {
        "manifest": manifest,
        "train": _split("train"),
        "validate": _split("validate"),
        "test": _split("test"),
        "data_dir": data_dir,
    }


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _holdout_split(
    n: int,
    val_frac: float = 0.1,
    random_state: int = NN_TRAIN_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_ix = perm[:n_val]
    tr_ix = perm[n_val:]
    if tr_ix.size == 0:
        tr_ix, val_ix = perm, perm[:1]
    return tr_ix, val_ix


def _predict_mlp(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).to(device)).cpu().numpy()


def _predict_cnn(
    model: nn.Module,
    X_tab: np.ndarray,
    wave: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return (
            model(
                torch.from_numpy(wave).to(device),
                torch.from_numpy(X_tab).to(device),
            )
            .cpu()
            .numpy()
        )


def aggregate_animal_freq_metrics(
    meta: pd.DataFrame,
    y_pred: np.ndarray,
) -> tuple[float, float, pd.DataFrame]:
    ev = meta[["animal_id", "frequency", "synapses"]].copy()
    ev["y_pred"] = y_pred
    agg = ev.groupby(["animal_id", "frequency"], as_index=False).agg(
        y_true=("synapses", "first"),
        y_pred=("y_pred", "mean"),
    )
    r2 = float(r2_score(agg["y_true"], agg["y_pred"]))
    rmse = _rmse(agg["y_true"].values, agg["y_pred"].values)
    return r2, rmse, agg


def tune_mlp(
    pack: dict[str, Any],
    *,
    val_frac: float = 0.1,
    epochs: int = 80,
    batch_size: int = 256,
    random_state: int = NN_TRAIN_RANDOM_STATE,
    verbose: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    tr = pack["train"]
    X, y = tr["X_tab"], tr["y"]
    tr_ix, va_ix = _holdout_split(len(y), val_frac=val_frac, random_state=random_state)

    grid = list(
        itertools.product(MLP_HIDDEN_GRID, DROPOUT_GRID, LR_GRID, WEIGHT_DECAY_GRID)
    )
    rows: list[dict[str, Any]] = []
    best_cfg: dict[str, Any] | None = None
    best_rmse = float("inf")

    for i, (hidden, dropout, lr, wd) in enumerate(grid):
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        hidden_t = tuple(hidden)
        model = MLPRegressor(X.shape[1], hidden=hidden_t, dropout=float(dropout))
        model = _train_loop(
            model,
            X[tr_ix],
            y[tr_ix],
            X[va_ix],
            y[va_ix],
            epochs=epochs,
            batch_size=batch_size,
            lr=float(lr),
            weight_decay=float(wd),
        )
        device = next(model.parameters()).device
        pred = _predict_mlp(model, X[va_ix], device)
        rmse = _rmse(y[va_ix], pred)
        row = {
            "hidden": list(hidden_t),
            "dropout": float(dropout),
            "lr": float(lr),
            "weight_decay": float(wd),
            "holdout_rmse": rmse,
            "trial": i,
        }
        rows.append(row)
        if rmse < best_rmse:
            best_rmse = rmse
            best_cfg = {
                "hidden": hidden_t,
                "dropout": float(dropout),
                "lr": float(lr),
                "weight_decay": float(wd),
            }
        if verbose:
            print(
                f"  [mlp {i+1}/{len(grid)}] "
                f"{_format_mlp_hp(hidden_t, dropout, lr, wd)} rmse={rmse:.4f}"
            )

    assert best_cfg is not None
    return best_cfg, pd.DataFrame(rows)


def tune_cnn(
    pack: dict[str, Any],
    *,
    mode: str = "cnn",
    val_frac: float = 0.1,
    epochs: int = 80,
    batch_size: int = 256,
    random_state: int = NN_TRAIN_RANDOM_STATE,
    verbose: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if mode not in ("cnn", "cnn_full"):
        raise ValueError("mode must be 'cnn' or 'cnn_full'")
    tr = pack["train"]
    X, y = tr["X_tab"], tr["y"]
    wave = tr["wave_i"] if mode == "cnn" else tr["wave_full"]
    tr_ix, va_ix = _holdout_split(len(y), val_frac=val_frac, random_state=random_state)

    grid = list(itertools.product(DROPOUT_GRID, LR_GRID, WEIGHT_DECAY_GRID))
    rows: list[dict[str, Any]] = []
    best_cfg: dict[str, Any] | None = None
    best_rmse = float("inf")

    def fwd(m, xb, wb):
        return m(wb, xb)

    for i, (dropout, lr, wd) in enumerate(grid):
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        model = CNNPlusTabular(X.shape[1], dropout=float(dropout))
        model = _train_loop(
            model,
            X[tr_ix],
            y[tr_ix],
            X[va_ix],
            y[va_ix],
            second_input=wave[tr_ix],
            second_val=wave[va_ix],
            epochs=epochs,
            batch_size=batch_size,
            lr=float(lr),
            weight_decay=float(wd),
            forward_fn=fwd,
        )
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            pred = fwd(
                model,
                torch.from_numpy(X[va_ix]).to(device),
                torch.from_numpy(wave[va_ix]).to(device),
            ).cpu().numpy()
        rmse = _rmse(y[va_ix], pred)
        row = {
            "dropout": float(dropout),
            "lr": float(lr),
            "weight_decay": float(wd),
            "holdout_rmse": rmse,
            "trial": i,
        }
        rows.append(row)
        if rmse < best_rmse:
            best_rmse = rmse
            best_cfg = {
                "dropout": float(dropout),
                "lr": float(lr),
                "weight_decay": float(wd),
            }
        if verbose:
            print(
                f"  [{mode} {i+1}/{len(grid)}] dropout={dropout} lr={lr} wd={wd} rmse={rmse:.4f}"
            )

    assert best_cfg is not None
    return best_cfg, pd.DataFrame(rows)


def train_final_mlp(
    pack: dict[str, Any],
    hp: dict[str, Any],
    *,
    epochs: int = 80,
    batch_size: int = 256,
) -> nn.Module:
    tr, va = pack["train"], pack["validate"]
    torch.manual_seed(NN_TRAIN_RANDOM_STATE)
    np.random.seed(NN_TRAIN_RANDOM_STATE)
    hidden = tuple(hp["hidden"])
    model = MLPRegressor(tr["X_tab"].shape[1], hidden=hidden, dropout=float(hp["dropout"]))
    return _train_loop(
        model,
        tr["X_tab"],
        tr["y"],
        va["X_tab"],
        va["y"],
        epochs=epochs,
        batch_size=batch_size,
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
    )


def train_final_cnn(
    pack: dict[str, Any],
    hp: dict[str, Any],
    *,
    mode: str = "cnn",
    epochs: int = 80,
    batch_size: int = 256,
) -> nn.Module:
    tr, va = pack["train"], pack["validate"]
    torch.manual_seed(NN_TRAIN_RANDOM_STATE)
    np.random.seed(NN_TRAIN_RANDOM_STATE)
    wave_tr = tr["wave_i"] if mode == "cnn" else tr["wave_full"]
    wave_va = va["wave_i"] if mode == "cnn" else va["wave_full"]

    def fwd(m, xb, wb):
        return m(wb, xb)

    model = CNNPlusTabular(tr["X_tab"].shape[1], dropout=float(hp["dropout"]))
    return _train_loop(
        model,
        tr["X_tab"],
        tr["y"],
        va["X_tab"],
        va["y"],
        second_input=wave_tr,
        second_val=wave_va,
        epochs=epochs,
        batch_size=batch_size,
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
        forward_fn=fwd,
    )


def evaluate_pack_model(
    pack: dict[str, Any],
    model: nn.Module,
    *,
    mode: str = "mlp",
) -> tuple[float, float, pd.DataFrame]:
    te = pack["test"]
    device = next(model.parameters()).device
    if mode == "mlp":
        pred = _predict_mlp(model, te["X_tab"], device)
    elif mode == "cnn":
        pred = _predict_cnn(model, te["X_tab"], te["wave_i"], device)
    elif mode == "cnn_full":
        pred = _predict_cnn(model, te["X_tab"], te["wave_full"], device)
    else:
        raise ValueError(mode)
    return aggregate_animal_freq_metrics(te["meta"], pred)


def _training_kwargs(pack: dict[str, Any]) -> dict[str, Any]:
    manifest = pack["manifest"]
    return {
        "epochs": int(manifest["training"]["epochs"]),
        "batch_size": int(manifest["training"]["batch_size"]),
        "val_frac": float(manifest["hp_holdout_frac"]),
    }


def _save_summary(out_dir: Path, row: pd.DataFrame, *, merge_models: tuple[str, ...]) -> pd.DataFrame:
    """Write one model row; keep other models from an existing summary file."""
    summary_path = out_dir / "nn_colab_summary.parquet"
    if summary_path.is_file():
        prev = pd.read_parquet(summary_path)
        prev = prev[~prev["model"].isin(merge_models)]
        summary = pd.concat([prev, row], ignore_index=True)
    else:
        summary = row
    summary = summary.sort_values("model").reset_index(drop=True)
    summary.to_parquet(summary_path, index=False)
    (out_dir / "nn_colab_summary.json").write_text(
        summary.to_json(orient="records", indent=2)
    )
    return summary


def run_one_model(
    pack: dict[str, Any],
    out_dir: Path | str,
    model_name: str,
    eval_mode: str,
    tune_fn,
    train_fn,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Tune, final-train, and evaluate a single architecture."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = _training_kwargs(pack)

    if verbose:
        print(f"\n=== HP tuning: {model_name} ===")
    best_hp, hp_log = tune_fn(pack, verbose=verbose, **kw)
    hp_log.to_parquet(out_dir / f"{model_name}_hp_trials.parquet", index=False)
    (out_dir / f"{model_name}_best_hp.json").write_text(
        json.dumps(best_hp, indent=2, default=str)
    )

    if verbose:
        print(f"\n=== Final train: {model_name} ===")
    model = train_fn(
        pack,
        best_hp,
        epochs=kw["epochs"],
        batch_size=kw["batch_size"],
    )
    torch.save(model.state_dict(), out_dir / f"{model_name}_best.pt")

    r2, rmse, pred_df = evaluate_pack_model(pack, model, mode=eval_mode)
    pred_df.insert(0, "model", model_name)
    pred_df.to_parquet(out_dir / f"{model_name}_test_predictions.parquet", index=False)

    row = pd.DataFrame(
        [
            {
                "model": model_name,
                "r2_test": r2,
                "rmse_test": rmse,
                **{f"hp_{k}": v for k, v in best_hp.items()},
            }
        ]
    )
    if verbose:
        print(f"  test R²={r2:.4f}, RMSE={rmse:.4f}")
    return row


def run_all_models(
    data_dir: Path | str,
    out_dir: Path | str | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir or data_dir / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    pack = load_colab_pack(data_dir)
    specs = [
        ("mlp", "mlp", tune_mlp, train_final_mlp),
        ("cnn", "cnn", lambda p, **kw: tune_cnn(p, mode="cnn", **kw), lambda p, hp, **kw: train_final_cnn(p, hp, mode="cnn", **kw)),
        (
            "cnn_full",
            "cnn_full",
            lambda p, **kw: tune_cnn(p, mode="cnn_full", **kw),
            lambda p, hp, **kw: train_final_cnn(p, hp, mode="cnn_full", **kw),
        ),
    ]

    summary_rows: list[pd.DataFrame] = []
    for model_name, eval_mode, tune_fn, train_fn in specs:
        summary_rows.append(
            run_one_model(
                pack,
                out_dir,
                model_name,
                eval_mode,
                tune_fn,
                train_fn,
                verbose=verbose,
            )
        )

    summary = pd.concat(summary_rows, ignore_index=True).sort_values("model")
    summary.to_parquet(out_dir / "nn_colab_summary.parquet", index=False)
    (out_dir / "nn_colab_summary.json").write_text(
        summary.to_json(orient="records", indent=2)
    )
    return summary


NN_COMPARISON_IDS = {"mlp": "N1", "cnn": "N2", "cnn_full": "N3"}


def hp_summary_to_comparison_rows(
    summary: pd.DataFrame,
    *,
    noise_label: str | None = None,
) -> pd.DataFrame:
    """Map ``run_all_models`` summary to Liberman comparison panel schema."""
    rows: list[dict[str, Any]] = []
    nl = noise_label or "predicted"
    for _, r in summary.iterrows():
        model = str(r["model"])
        rows.append(
            {
                "config_id": NN_COMPARISON_IDS.get(model, model),
                "model": model,
                "format": "long",
                "noise_label": nl,
                "r2_test": float(r["r2_test"]),
                "rmse_test": float(r["rmse_test"]),
            }
        )
    return pd.DataFrame(rows)


def run_liberman_nn_hp_comparison(
    data_dir: Path | str,
    out_dir: Path | str | None = None,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    HP-tune mlp / cnn / cnn_full on an exported Colab pack.

    Returns (full summary with HP columns, comparison rows for the classical panel).
    """
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    summary = run_all_models(data_dir, out_dir, verbose=verbose)
    nl = str(manifest.get("noise_label", "predicted"))
    return summary, hp_summary_to_comparison_rows(summary, noise_label=nl)


def run_mlp_only(
    data_dir: Path | str,
    out_dir: Path | str | None = None,
    pack: dict[str, Any] | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Re-run MLP only; merge into ``nn_colab_summary`` (keeps existing CNN rows)."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir or data_dir / "results")
    pack = pack if pack is not None else load_colab_pack(data_dir)
    row = run_one_model(
        pack,
        out_dir,
        "mlp",
        "mlp",
        tune_mlp,
        train_final_mlp,
        verbose=verbose,
    )
    return _save_summary(out_dir, row, merge_models=("mlp",))

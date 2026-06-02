"""
MLP Stage-2 SHAP via ``shap.GradientExplainer`` (long-grain, synthesis CV aligned).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.compose import ColumnTransformer

from utils.benchmark_metrics import STAGE2_BEST_HP_DIR
from utils.nn_colab_train import _holdout_split, aggregate_animal_freq_metrics
from utils.nn_stage2 import MLPRegressor, _train_loop, prepare_long_xy, syn_prep_transformer
from utils.nn_stage2_data import LONG_LOG, LONG_NUM_BASE
from utils.stage2_hp import resolve_torch_device

_MLP_HP_KEYS = ("hidden", "dropout", "lr", "weight_decay")
_EXPECTED_N_FEATURES = len(LONG_NUM_BASE) + 1 + len(LONG_LOG)  # num + noise_preds + log

FEATURE_PAPER_LABELS: dict[str, str] = {
    "amplitude": "Amplitude",
    "distance": "Peak-to-trough latency",
    "slope": "Wave I slope",
    "level": "SPL (dB)",
    "total_variance": "Total variance",
    "PeakIEarlyCurvature": "Peak I curvature (early)",
    "PeakICentralCurvature": "Peak I curvature (central)",
    "PeakILateCurvature": "Peak I curvature (late)",
    "TroughIEarlyCurvature": "Trough I curvature (early)",
    "TroughICentralCurvature": "Trough I curvature (central)",
    "TroughILateCurvature": "Trough I curvature (late)",
    "noise_preds": "Predicted noise",
    "frequency": "Frequency (kHz)",
}

DEPENDENCE_GROUPS: list[list[str]] = [
    ["amplitude", "total_variance", "slope", "distance"],
    [
        "PeakIEarlyCurvature",
        "PeakICentralCurvature",
        "PeakILateCurvature",
    ],
    [
        "TroughIEarlyCurvature",
        "TroughICentralCurvature",
        "TroughILateCurvature",
    ],
]

NOISE_EXPOSED_COLOR = "#E69F00"
UNEXPOSED_COLOR = "#009E73"


@dataclass(frozen=True)
class FoldArtifacts:
    model: MLPRegressor
    prep: ColumnTransformer
    feature_names: List[str]
    X_tr: np.ndarray
    X_eval: np.ndarray
    tr_idx: pd.Index
    te_idx: pd.Index
    meta_eval: pd.DataFrame
    y_pred_eval: np.ndarray
    rmse_animal_freq: float


def mlp_hp_for_scenario(scenario: str, *, hp_dir: Path | None = None) -> dict[str, Any]:
    path = Path(hp_dir or STAGE2_BEST_HP_DIR) / f"{scenario}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing MLP HP file {path}; run abr_stage2_hp_tuning.ipynb "
            f"(scenario {scenario})."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "MLP" not in payload:
        raise KeyError(f"{path} has no 'MLP' block")
    hp = dict(payload["MLP"])
    missing = [k for k in _MLP_HP_KEYS if k not in hp]
    if missing:
        raise KeyError(f"{path} MLP block missing keys: {missing}")
    return hp


def long_transformed_feature_names(
    long_num: Sequence[str],
    long_cat: Sequence[str],
    long_log: Sequence[str],
    *,
    n_cols: int | None = None,
) -> list[str]:
    """Column order matches ``syn_prep_transformer``: num, cat (OHE drop-first), log."""
    cat_names = list(long_cat)
    names = list(long_num) + cat_names + list(long_log)
    if n_cols is not None and len(names) != n_cols:
        raise ValueError(
            f"Feature name count {len(names)} != transformed width {n_cols}: {names}"
        )
    if len(names) != _EXPECTED_N_FEATURES:
        raise ValueError(
            f"Expected {_EXPECTED_N_FEATURES} transformed features, got {len(names)}: {names}"
        )
    return names


def build_long_feature_annotation_table(feature_names: Sequence[str]) -> pd.DataFrame:
    log_set = set(LONG_LOG)
    rows = []
    for f in feature_names:
        rows.append(
            {
                "feature_name": f,
                "feature_type": f,
                "is_log_branch": f in log_set,
                "is_stage1_noise_pred": f == "noise_preds",
                "is_frequency": f == "frequency",
                "is_level": f == "level",
                "paper_label": FEATURE_PAPER_LABELS.get(f, f),
            }
        )
    return pd.DataFrame(rows)


def fit_mlp_fold(
    l_tr: pd.DataFrame,
    l_ev: pd.DataFrame,
    long_num: List[str],
    long_cat: List[str],
    long_log: List[str],
    hp: dict[str, Any],
    *,
    holdout_seed: int,
    device: torch.device | None = None,
) -> FoldArtifacts:
    """Train one MLP fold (same contract as ``stage2_synthesis_cv._score_mlp``)."""
    device = device or resolve_torch_device()
    prep = syn_prep_transformer(long_num, long_cat, long_log)
    tr_idx, X_tr, y_tr = prepare_long_xy(l_tr, long_num, long_cat, long_log, prep, fit=True)
    ho_tr, ho_va = _holdout_split(len(y_tr), val_frac=0.1, random_state=holdout_seed)
    hidden = tuple(int(h) for h in hp["hidden"])
    model = MLPRegressor(
        X_tr.shape[1], hidden=hidden, dropout=float(hp.get("dropout", 0.2))
    )
    model = _train_loop(
        model,
        X_tr[ho_tr],
        y_tr[ho_tr],
        X_tr[ho_va],
        y_tr[ho_va],
        epochs=80,
        batch_size=256,
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
        device=device,
    )
    te_idx, X_te, _ = prepare_long_xy(l_ev, long_num, long_cat, long_log, prep, fit=False)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_te).to(device)).cpu().numpy()

    meta_cols = [
        "animal_id",
        "frequency",
        "level",
        "noise_cat",
        "noise_preds",
        "synapses",
    ]
    if "experimental_group" in l_ev.columns:
        meta_cols.append("experimental_group")
    elif "tx" in l_ev.columns:
        meta_cols.append("tx")
    meta = l_ev.loc[te_idx, meta_cols].reset_index(drop=True)
    _, rmse_af, _ = aggregate_animal_freq_metrics(
        meta[["animal_id", "frequency", "synapses"]], pred
    )
    feat_names = long_transformed_feature_names(
        long_num, long_cat, long_log, n_cols=X_tr.shape[1]
    )
    model_cpu = model.cpu().eval()
    return FoldArtifacts(
        model=model_cpu,
        prep=prep,
        feature_names=feat_names,
        X_tr=X_tr,
        X_eval=X_te,
        tr_idx=tr_idx,
        te_idx=te_idx,
        meta_eval=meta,
        y_pred_eval=pred,
        rmse_animal_freq=float(rmse_af),
    )


def _background_indices(n_train: int, *, seed: int, max_bg: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(max_bg, n_train)
    return rng.choice(n_train, size=k, replace=False)


class _ShapMLPWrapper(torch.nn.Module):
    """GradientExplainer expects shape ``(batch, n_outputs)``; MLP returns ``(batch,)``."""

    def __init__(self, inner: MLPRegressor) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x).reshape(-1, 1)


def gradient_shap_values(
    model: MLPRegressor,
    X_background: np.ndarray,
    X_eval: np.ndarray,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Out-of-fold SHAP on CPU via ``GradientExplainer``."""
    model = model.cpu().eval()
    wrapped = _ShapMLPWrapper(model)
    bg_t = torch.from_numpy(np.asarray(X_background, dtype=np.float32))
    explainer = shap.GradientExplainer(wrapped, bg_t)
    parts: list[np.ndarray] = []
    n = len(X_eval)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.from_numpy(np.asarray(X_eval[start:end], dtype=np.float32))
        sv = explainer.shap_values(xb)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv, dtype=np.float64)
        if sv.ndim == 3:
            sv = sv[..., 0]
        parts.append(sv)
    out = np.vstack(parts)
    if out.shape != (n, X_eval.shape[1]):
        raise ValueError(f"SHAP shape mismatch: got {out.shape}, expected ({n}, {X_eval.shape[1]})")
    if not np.isfinite(out).all():
        raise ValueError("SHAP values contain non-finite numbers")
    return out


def compute_fold_shap(
    artifacts: FoldArtifacts,
    *,
    holdout_seed: int,
    shap_batch_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP + feature matrices aligned to ``artifacts.meta_eval``."""
    bg_ix = _background_indices(len(artifacts.X_tr), seed=holdout_seed)
    shap_vals = gradient_shap_values(
        artifacts.model,
        artifacts.X_tr[bg_ix],
        artifacts.X_eval,
        batch_size=shap_batch_size,
    )
    cols = artifacts.feature_names
    shap_df = pd.DataFrame(shap_vals, columns=cols)
    feat_df = pd.DataFrame(artifacts.X_eval, columns=cols)
    return shap_df, feat_df


def fold_cache_path(cache_dir: Path, cohort_slug: str, fold_id: int) -> Path:
    return cache_dir / f"fold_{cohort_slug}_{fold_id}.parquet"


def save_fold_cache(
    path: Path,
    *,
    shap_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat(
        [
            meta_df.reset_index(drop=True),
            feat_df.add_prefix("feat_"),
            shap_df.add_prefix("shap_"),
        ],
        axis=1,
    )
    out.to_parquet(path, index=False)


def load_fold_cache(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(path)
    meta_cols = [
        "animal_id",
        "frequency",
        "level",
        "noise_cat",
        "noise_preds",
        "synapses",
        "experimental_group",
        "tx",
        "fold_id",
        "cohort",
    ]
    meta = df[[c for c in meta_cols if c in df.columns]].copy()
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    shap_cols = [c for c in df.columns if c.startswith("shap_")]
    feat_df = df[feat_cols].rename(columns=lambda c: c.removeprefix("feat_"))
    shap_df = df[shap_cols].rename(columns=lambda c: c.removeprefix("shap_"))
    return shap_df, feat_df, meta


# --- Plotting (long-grain MLP) ---


def save_current_fig(base_no_ext: Path, *, tight: bool = True) -> None:
    base_no_ext.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(base_no_ext.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(base_no_ext.with_suffix(".svg"), bbox_inches="tight")
    plt.close()


def true_noise_point_colors(meta: pd.DataFrame) -> np.ndarray:
    exposed = meta["noise_cat"].astype(int).eq(1).values
    colors = np.empty(len(meta), dtype=object)
    colors[exposed] = NOISE_EXPOSED_COLOR
    colors[~exposed] = UNEXPOSED_COLOR
    return colors


def noise_group_legend_handles() -> list:
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=NOISE_EXPOSED_COLOR,
            markeredgecolor=NOISE_EXPOSED_COLOR,
            markersize=8,
            label="Noise-exposed",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=UNEXPOSED_COLOR,
            markeredgecolor=UNEXPOSED_COLOR,
            markersize=8,
            label="Unexposed",
        ),
    ]


def beeswarm_dot_y_offsets(shaps: np.ndarray, *, row_height: float = 0.4) -> np.ndarray:
    nbins = 100
    quant = np.round(
        nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8)
    )
    inds = np.argsort(quant + np.random.randn(len(shaps)) * 1e-6)
    layer = 0
    last_bin = -1
    ys = np.zeros(len(shaps))
    for ind in inds:
        if quant[ind] != last_bin:
            layer = 0
        ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[ind]
    ys *= 0.9 * (row_height / np.max(ys + 1))
    return ys


def summary_beeswarm_true_noise(
    values: pd.DataFrame,
    features: pd.DataFrame,
    meta: pd.DataFrame,
    out_base: Path,
    *,
    max_display: int = 13,
    title: str = "",
) -> None:
    cols = list(values.columns)
    paper = [FEATURE_PAPER_LABELS.get(c, c) for c in cols]
    shap_vals = values[cols].values
    dot_colors = true_noise_point_colors(meta)
    feature_order = np.argsort(np.sum(np.abs(shap_vals), axis=0))
    feature_order = feature_order[-min(max_display, len(feature_order)) :]

    row_height = 0.4
    plt.figure(figsize=(8, min(len(feature_order), max_display) * row_height + 1.5))
    ax = plt.gca()
    ax.axvline(x=0, color="#999999", zorder=-1)
    for pos, i in enumerate(feature_order):
        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = shap_vals[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        shaps = shaps[inds]
        colors = dot_colors[inds]
        ys = beeswarm_dot_y_offsets(shaps, row_height=row_height)
        ax.scatter(
            shaps,
            pos + ys,
            c=colors,
            s=12,
            alpha=0.85,
            linewidth=0,
            rasterized=len(shaps) > 800,
        )
    ax.set_yticks(range(len(feature_order)), [paper[i] for i in feature_order])
    ax.set_xlabel("SHAP value")
    ax.set_title(title or "MLP SHAP (animal × frequency × SPL level)")
    for spine in ("right", "top", "left"):
        ax.spines[spine].set_visible(False)
    ax.legend(
        handles=noise_group_legend_handles(),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10,
    )
    save_current_fig(out_base, tight=True)


def summary_bar(values: pd.DataFrame, out_base: Path, *, title: str = "") -> None:
    means = values.abs().mean(axis=0).sort_values(ascending=True)
    labels = [FEATURE_PAPER_LABELS.get(c, c) for c in means.index]
    plt.figure(figsize=(8, 5))
    plt.barh(labels, means.values, color="#4c72b0")
    plt.xlabel("Mean |SHAP|")
    plt.title(title or "Mean |SHAP| (MLP, long rows)")
    save_current_fig(out_base)


def plot_level_scalar_shap(
    shap_df: pd.DataFrame,
    meta: pd.DataFrame,
    scalar_features: Sequence[str],
    out_base: Path,
    *,
    cohort_slug: str,
) -> None:
    """Mean |SHAP| for scalar features, columns = SPL level."""
    levels = sorted(meta["level"].dropna().unique())
    level_labels = [f"{int(lv)}dB" for lv in levels]
    mat = []
    for feat in scalar_features:
        if feat not in shap_df.columns:
            continue
        row = []
        for lv in levels:
            mask = meta["level"].astype(float).eq(float(lv))
            row.append(float(shap_df.loc[mask, feat].abs().mean()) if mask.any() else 0.0)
        mat.append(row)
    if not mat:
        return
    data = pd.DataFrame(mat, index=scalar_features, columns=level_labels)
    plt.figure(figsize=(8, 4 + 0.3 * len(data)))
    im = plt.imshow(data.values, aspect="auto", cmap="Blues")
    plt.colorbar(im, label="Mean |SHAP|")
    plt.yticks(range(len(data)), [FEATURE_PAPER_LABELS.get(c, c) for c in data.index])
    plt.xticks(range(len(level_labels)), level_labels)
    plt.title(f"MLP |SHAP| by SPL level ({cohort_slug})")
    save_current_fig(out_base)


def plot_dependence_grouped_panel(
    shap_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    meta: pd.DataFrame,
    cohort_slug: str,
    out_base: Path,
    *,
    color_by: str,
) -> None:
    from matplotlib.gridspec import GridSpec

    use_noise = color_by == "noise_preds"
    if use_noise:
        point_colors = true_noise_point_colors(meta)
    elif color_by == "frequency":
        color_vals = meta["frequency"].astype(float).values
        vmin, vmax = float(np.nanmin(color_vals)), float(np.nanmax(color_vals))
    else:
        raise ValueError(f"Unsupported color_by={color_by!r}")

    fig = plt.figure(figsize=(10.5, 8.5))
    spec = GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.0, 1.0], hspace=0.55, wspace=0.38)
    panel_slots: list[tuple[int, int, str]] = []
    for col, ftype in enumerate(DEPENDENCE_GROUPS[0]):
        panel_slots.append((0, col, ftype))
    for col, ftype in enumerate(DEPENDENCE_GROUPS[1]):
        panel_slots.append((1, col, ftype))
    for col, ftype in enumerate(DEPENDENCE_GROUPS[2]):
        panel_slots.append((2, col, ftype))

    last_sc = None
    for row, col, ftype in panel_slots:
        ax = fig.add_subplot(spec[row, col])
        if ftype not in shap_df.columns:
            ax.set_visible(False)
            continue
        if use_noise:
            last_sc = ax.scatter(
                feat_df[ftype].values,
                shap_df[ftype].values,
                c=point_colors,
                s=10,
                alpha=0.65,
                linewidths=0,
            )
        else:
            last_sc = ax.scatter(
                feat_df[ftype].values,
                shap_df[ftype].values,
                c=color_vals,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                s=10,
                alpha=0.65,
                linewidths=0,
            )
        ax.set_xlabel(FEATURE_PAPER_LABELS.get(ftype, ftype), fontsize=9)
        if col > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("SHAP", fontsize=9)

    if last_sc is None:
        plt.close(fig)
        return
    if not use_noise:
        fig.colorbar(last_sc, ax=fig.axes, label="Frequency (kHz)", shrink=0.6)
    else:
        fig.legend(
            handles=noise_group_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
        )
    fig.suptitle(
        f"MLP SHAP dependence ({cohort_slug}; animal×freq×level rows)",
        y=1.04,
        fontsize=11,
    )
    save_current_fig(out_base, tight=True)


def run_mlp_shap_plots(
    shap_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    meta: pd.DataFrame,
    cohort_slug: str,
    fig_dir: Path,
) -> None:
    """Write standard MLP SHAP figure bundle for one cohort."""
    fig_dir = Path(fig_dir)
    summary_beeswarm_true_noise(
        shap_df,
        feat_df,
        meta,
        fig_dir / f"mlp_shap_beeswarm_{cohort_slug}",
        title=f"MLP SHAP ({cohort_slug}; long rows)",
    )
    summary_bar(
        shap_df,
        fig_dir / f"mlp_shap_bar_{cohort_slug}",
        title=f"Mean |SHAP| MLP ({cohort_slug})",
    )
    plot_level_scalar_shap(
        shap_df,
        meta,
        ["amplitude", "slope", "distance"],
        fig_dir / f"mlp_shap_level_scalars_{cohort_slug}",
        cohort_slug=cohort_slug,
    )
    plot_dependence_grouped_panel(
        shap_df,
        feat_df,
        meta,
        cohort_slug,
        fig_dir / f"mlp_shap_dependence_noise_{cohort_slug}",
        color_by="noise_preds",
    )
    plot_dependence_grouped_panel(
        shap_df,
        feat_df,
        meta,
        cohort_slug,
        fig_dir / f"mlp_shap_dependence_freq_{cohort_slug}",
        color_by="frequency",
    )

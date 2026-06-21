"""
MLP Stage-2 SHAP via ``shap.GradientExplainer`` (long-grain, synthesis CV aligned).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.compose import ColumnTransformer

from utils.abr_univariate_eda import NOISE_LABELS
from utils.benchmark_metrics import STAGE2_BEST_HP_DIR
from utils.nn_colab_train import _holdout_split, aggregate_animal_freq_metrics
from utils.nn_stage2 import MLPRegressor, _train_loop, prepare_long_xy, syn_prep_transformer
from utils.nn_stage2_data import LONG_LOG, LONG_NUM_BASE, STRAIN_BINARY_COL
from utils.stage2_hp import resolve_torch_device

_MLP_HP_KEYS = ("hidden", "dropout", "lr", "weight_decay")

MLP_SHAP_CACHE_DIR = Path("figures/cache/shap_mlp")
MLP_SHAP_FIG_DIR = Path("figures/shap_mlp")
COHORT_A_SLUG = "liberman"
COHORT_B_SLUG = "buran"
COHORT_SLUGS_DECK = (COHORT_A_SLUG, COHORT_B_SLUG)
SHAP_EXPORT_RENAMES = {"frequency": "shap_frequency", "level": "shap_level"}
INVERSE_SHAP_EXPORT_RENAMES = {v: k for k, v in SHAP_EXPORT_RENAMES.items()}
MLP_EXPORT_KEY_COLS = frozenset({"animal_id", "frequency", "level"})
# Long OOF rows after dedupe (animal × frequency × SPL); pre-dedupe totals are larger.
EXPECTED_ROW_COUNTS = {COHORT_A_SLUG: 5901, COHORT_B_SLUG: 2781}
OOF_GRAIN_KEYS = ("animal_id", "frequency", "level")
OOF_ANIMAL_FREQ_KEYS = ("animal_id", "frequency")
DECK_FIGURE_STEMS = (
    "mlp_shap_beeswarm_cohorts",
    "mlp_shap_scatter_noise_cohorts",
    "mlp_shap_beeswarm_by_noise",
    "mlp_shap_scatter_frequency_cohorts",
    "mlp_shap_scatter_level_frequency",
)

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
    STRAIN_BINARY_COL: "Strain (CBA/CaJ vs C57BL/6J)",
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
    return names


def expected_long_transformed_width(
    long_num: Sequence[str],
    long_cat: Sequence[str],
    long_log: Sequence[str],
) -> int:
    """Transformed column count for long Stage-2 tabular features (cat = one OHE column)."""
    return len(long_num) + len(list(long_cat)) + len(list(long_log))


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

    meta_cols = ["animal_id", "frequency", "level", "synapses"]
    for col in ("noise_cat", "noise_preds"):
        if col in l_ev.columns:
            meta_cols.insert(3, col)
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


def _background_indices(
    n_train: int, *, seed: int, max_bg: int | None = 200
) -> np.ndarray:
    if max_bg is None or max_bg >= n_train:
        return np.arange(n_train)
    rng = np.random.default_rng(seed)
    return rng.choice(n_train, size=max_bg, replace=False)


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
    max_background: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP + feature matrices aligned to ``artifacts.meta_eval``."""
    bg_ix = _background_indices(
        len(artifacts.X_tr), seed=holdout_seed, max_bg=max_background
    )
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


def aggregate_oof_long_grain(
    shap_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One row per animal × frequency × SPL (mean SHAP/features; first meta labels)."""
    keys = list(OOF_GRAIN_KEYS)
    shap_cols = list(shap_df.columns)
    feat_cols = list(feat_df.columns)
    meta_cols = [
        c
        for c in meta_df.columns
        if c not in shap_cols and c not in feat_cols and c not in keys
    ]
    combined = meta_df[[*keys, *meta_cols]].reset_index(drop=True).copy()
    combined[shap_cols] = shap_df.reset_index(drop=True)
    combined[feat_cols] = feat_df.reset_index(drop=True)

    agg_spec: dict[str, str] = {c: "mean" for c in shap_cols + feat_cols + ["synapses"]}
    for c in meta_cols:
        if c in keys or c in agg_spec:
            continue
        agg_spec[c] = "first"

    out = combined.groupby(keys, as_index=False).agg(agg_spec)
    meta_out = out[[*keys, *meta_cols]].copy()
    if "noise_preds" in out.columns and "noise_preds" not in meta_out.columns:
        meta_out["noise_preds"] = out["noise_preds"]
    return out[shap_cols], out[feat_cols], meta_out


def aggregate_oof_animal_freq_grain(
    shap_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One row per animal × frequency (mean SHAP/features across SPL levels)."""
    keys = list(OOF_ANIMAL_FREQ_KEYS)
    shap_cols = list(shap_df.columns)
    feat_cols = list(feat_df.columns)
    meta_cols = [
        c
        for c in meta_df.columns
        if c not in shap_cols and c not in feat_cols and c not in keys
    ]
    combined = meta_df[[*keys, *meta_cols]].reset_index(drop=True).copy()
    combined[shap_cols] = shap_df.reset_index(drop=True)
    combined[feat_cols] = feat_df.reset_index(drop=True)

    agg_spec: dict[str, str] = {c: "mean" for c in shap_cols + feat_cols + ["synapses"]}
    for c in meta_cols:
        if c in keys or c in agg_spec:
            continue
        agg_spec[c] = "first"

    out = combined.groupby(keys, as_index=False).agg(agg_spec)
    meta_out = out[[*keys, *meta_cols]].copy()
    if "noise_preds" in out.columns and "noise_preds" not in meta_out.columns:
        meta_out["noise_preds"] = out["noise_preds"]
    return out[shap_cols], out[feat_cols], meta_out


def load_cohort_outputs_from_cache(
    cache_dir: Path,
    cohort_slug: str,
    *,
    n_folds: int = 10,
) -> dict[str, pd.DataFrame]:
    """Reload OOF SHAP / features / meta from per-fold parquet caches."""
    shap_rows: list[pd.DataFrame] = []
    feat_rows: list[pd.DataFrame] = []
    meta_rows: list[pd.DataFrame] = []
    for fold_id in range(n_folds):
        path = fold_cache_path(cache_dir, cohort_slug, fold_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing fold cache: {path}")
        shap_df, feat_df, meta_df = load_fold_cache(path)
        shap_rows.append(shap_df)
        feat_rows.append(feat_df)
        meta_rows.append(meta_df)
    shap_oof = pd.concat(shap_rows, ignore_index=True)
    feat_oof = pd.concat(feat_rows, ignore_index=True)
    meta_oof = pd.concat(meta_rows, ignore_index=True)
    shap_oof, feat_oof, meta_oof = aggregate_oof_long_grain(shap_oof, feat_oof, meta_oof)
    return {"shap": shap_oof, "feat": feat_oof, "meta": meta_oof}


def export_mlp_shap_parquet(
    cohort_slug: str,
    shap_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    cache_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write ``shap_values_mlp_*`` and ``shap_metadata_mlp_*`` parquets."""
    cache_dir = Path(cache_dir or MLP_SHAP_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    shap_features = shap_df.rename(columns=SHAP_EXPORT_RENAMES)
    shap_export = pd.concat(
        [meta_df[["animal_id", "frequency", "level"]].reset_index(drop=True), shap_features],
        axis=1,
    )
    if not shap_export.columns.is_unique:
        dupes = shap_export.columns[shap_export.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns in shap export: {dupes}")

    meta_export_cols = [
        "animal_id",
        "frequency",
        "noise_cat",
        "noise_preds",
        "synapses",
        "fold_id",
        "cohort",
        "experimental_group",
        "tx",
    ]
    meta_export = meta_df.copy()
    meta_export["stimulus_level"] = meta_export["level"]
    meta_out = meta_export[
        [c for c in meta_export_cols if c in meta_export.columns] + ["stimulus_level"]
    ]
    # stable column order
    ordered = [
        "animal_id",
        "frequency",
        "stimulus_level",
        "noise_cat",
        "noise_preds",
        "synapses",
        "fold_id",
        "cohort",
        "experimental_group",
        "tx",
    ]
    meta_out = meta_out[[c for c in ordered if c in meta_out.columns]]

    values_path = cache_dir / f"shap_values_mlp_{cohort_slug}.parquet"
    meta_path = cache_dir / f"shap_metadata_mlp_{cohort_slug}.parquet"
    shap_export.to_parquet(values_path, index=False)
    meta_out.to_parquet(meta_path, index=False)
    return values_path, meta_path


def load_mlp_cohort_from_export_parquets(
    cache_dir: Path | str,
    cohort_slug: str,
) -> dict[str, pd.DataFrame]:
    """Reload consolidated MLP SHAP export parquets (long grain)."""
    cache_dir = Path(cache_dir)
    values_path = cache_dir / f"shap_values_mlp_{cohort_slug}.parquet"
    meta_path = cache_dir / f"shap_metadata_mlp_{cohort_slug}.parquet"
    if not values_path.is_file():
        raise FileNotFoundError(
            f"Missing {values_path}; run abr_stage2_mlp_shap.ipynb export first."
        )
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing {meta_path}; run abr_stage2_mlp_shap.ipynb export first."
        )
    values = pd.read_parquet(values_path)
    meta = pd.read_parquet(meta_path)
    if len(values) != len(meta):
        raise ValueError(
            f"{cohort_slug}: values rows ({len(values)}) != metadata rows ({len(meta)})"
        )
    shap_cols = [c for c in values.columns if c not in MLP_EXPORT_KEY_COLS]
    shap_df = values[shap_cols].rename(columns=INVERSE_SHAP_EXPORT_RENAMES)
    return {"shap": shap_df, "meta": meta}


def aggregate_mlp_export_animal_freq(
    shap_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average SHAP across SPL levels within each animal × frequency pair."""
    keys = list(OOF_ANIMAL_FREQ_KEYS)
    shap_cols = list(shap_df.columns)
    meta_cols = [
        c
        for c in meta_df.columns
        if c not in shap_cols and c not in keys
    ]
    combined = meta_df[[*keys, *meta_cols]].reset_index(drop=True).copy()
    combined[shap_cols] = shap_df.reset_index(drop=True)
    agg_spec: dict[str, str] = {c: "mean" for c in shap_cols}
    for c in meta_cols:
        if c not in agg_spec:
            agg_spec[c] = "first"
    out = combined.groupby(keys, as_index=False).agg(agg_spec)
    meta_out = out[[*keys, *[c for c in meta_cols if c in out.columns]]].copy()
    if "noise_preds" in out.columns and "noise_preds" not in meta_out.columns:
        meta_out["noise_preds"] = out["noise_preds"]
    return out[shap_cols], meta_out


def export_feature_names_mlp(
    feature_names: Sequence[str],
    cache_dir: Path | None = None,
) -> Path:
    cache_dir = Path(cache_dir or MLP_SHAP_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "feature_names_mlp.parquet"
    build_long_feature_annotation_table(feature_names).to_parquet(path, index=False)
    return path


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
            label=NOISE_LABELS[1],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=UNEXPOSED_COLOR,
            markeredgecolor=UNEXPOSED_COLOR,
            markersize=8,
            label=NOISE_LABELS[0],
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


def feature_columns(shap_df: pd.DataFrame) -> list[str]:
    return [c for c in shap_df.columns if c in FEATURE_PAPER_LABELS]


def mean_abs_shap_series(shap_df: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per model feature (one scalar per column)."""
    cols = feature_columns(shap_df)
    return shap_df[cols].abs().mean(axis=0)


def mean_abs_shap_table(shap_df: pd.DataFrame) -> pd.DataFrame:
    """Tidy mean |SHAP| table sorted descending (rank 1 = most important)."""
    means = mean_abs_shap_series(shap_df).sort_values(ascending=False)
    out = pd.DataFrame(
        {
            "feature": means.index,
            "paper_label": [FEATURE_PAPER_LABELS.get(f, f) for f in means.index],
            "mean_abs_shap": means.values,
        }
    )
    out["rank"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True)


def _shap_for_mean_abs_summary(
    payload: Mapping[str, pd.DataFrame],
    *,
    grain: str,
) -> pd.DataFrame:
    shap_df = payload["shap"]
    if grain == "long":
        return shap_df
    if grain == "animal_freq":
        feat_df = payload["feat"]
        meta_df = payload["meta"]
        shap_df, _, _ = aggregate_oof_animal_freq_grain(shap_df, feat_df, meta_df)
        return shap_df
    raise ValueError(f"Unsupported grain={grain!r}; expected 'long' or 'animal_freq'")


def mean_abs_shap_tables_by_cohort(
    cohort_outputs: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    grain: str = "animal_freq",
) -> dict[str, pd.DataFrame]:
    """Per-cohort mean |SHAP| tables for each entry in ``COHORT_SLUGS_DECK``.

    Default ``grain='animal_freq'`` averages SHAP across SPL levels within each
    animal × frequency pair before computing mean |SHAP| (fairer vs wide XGB).
    Use ``grain='long'`` for animal × frequency × SPL rows.
    """
    tables: dict[str, pd.DataFrame] = {}
    for slug in COHORT_SLUGS_DECK:
        if slug not in cohort_outputs:
            raise KeyError(f"Missing cohort_outputs entry for {slug!r}")
        shap_df = _shap_for_mean_abs_summary(cohort_outputs[slug], grain=grain)
        tables[slug] = mean_abs_shap_table(shap_df)
    return tables


def pooled_feature_order(cohort_outputs: Mapping[str, Mapping[str, pd.DataFrame]]) -> list[str]:
    cols = feature_columns(next(iter(cohort_outputs.values()))["shap"])
    pooled = pd.concat(
        [cohort_outputs[s]["shap"][cols] for s in COHORT_SLUGS_DECK if s in cohort_outputs],
        ignore_index=True,
    )
    return pooled.abs().mean().sort_values(ascending=False).index.tolist()


def _draw_panel_letter(fig: plt.Figure, ax: plt.Axes, letter: str) -> None:
    pos = ax.get_position()
    fig.text(
        pos.x0 - 0.02,
        pos.y1 + 0.01,
        letter,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def _feature_value_colors(values: np.ndarray) -> np.ndarray:
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if vmax - vmin < 1e-12:
        norm = np.zeros(len(values))
    else:
        norm = (values - vmin) / (vmax - vmin)
    cmap = plt.cm.coolwarm
    return cmap(norm)


def _beeswarm_on_ax(
    ax: plt.Axes,
    shap_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    feature_order: Sequence[str],
    *,
    row_height: float = 0.38,
) -> None:
    ax.axvline(x=0, color="#999999", zorder=-1, lw=0.8)
    for pos, feat in enumerate(feature_order):
        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = shap_df[feat].values
        feat_vals = feat_df[feat].values
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        shaps = shaps[inds]
        feat_vals = feat_vals[inds]
        colors = _feature_value_colors(feat_vals)
        ys = beeswarm_dot_y_offsets(shaps, row_height=row_height)
        ax.scatter(
            shaps,
            pos + ys,
            c=colors,
            s=10,
            alpha=0.85,
            linewidth=0,
            rasterized=len(shaps) > 800,
        )
    labels = [FEATURE_PAPER_LABELS.get(f, f) for f in feature_order]
    ax.set_yticks(range(len(feature_order)), labels)
    ax.set_xlabel("SHAP value", fontsize=14)
    for spine in ("right", "top", "left"):
        ax.spines[spine].set_visible(False)


def plot_beeswarm_cohorts_1x2(
    cohort_outputs: Mapping[str, Mapping[str, pd.DataFrame]],
    out_base: Path,
    *,
    feature_order: Sequence[str] | None = None,
) -> None:
    feature_order = list(feature_order or pooled_feature_order(cohort_outputs))
    n_feat = len(feature_order)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, n_feat * 0.38 + 1.5)), sharey=True)
    titles = {"liberman": "Cohort A (Liberman)", "buran": "Cohort B (Brad)"}
    letters = ("A", "B")
    for ax, slug, letter in zip(axes, COHORT_SLUGS_DECK, letters):
        payload = cohort_outputs[slug]
        _beeswarm_on_ax(ax, payload["shap"], payload["feat"], feature_order)
        ax.set_title(titles[slug], fontsize=15)
        _draw_panel_letter(fig, ax, letter)
    fig.suptitle("MLP SHAP (out-of-fold; colored by feature value)", fontsize=16, y=1.02)
    save_current_fig(out_base, tight=True)


def plot_scatter_grid_noise_13x2(
    cohort_outputs: Mapping[str, Mapping[str, pd.DataFrame]],
    out_base: Path,
    *,
    feature_order: Sequence[str] | None = None,
) -> None:
    feature_order = list(feature_order or pooled_feature_order(cohort_outputs))
    n_rows = len(feature_order)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, max(8, n_rows * 2.6)),
        sharex="col",
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.array([axes])
    titles = {"liberman": "Cohort A", "buran": "Cohort B"}
    for col, slug in enumerate(COHORT_SLUGS_DECK):
        payload = cohort_outputs[slug]
        colors = true_noise_point_colors(payload["meta"])
        for row, feat in enumerate(feature_order):
            ax = axes[row, col]
            ax.scatter(
                payload["feat"][feat],
                payload["shap"][feat],
                c=colors,
                s=8,
                alpha=0.65,
                linewidths=0,
                rasterized=len(colors) > 500,
            )
            if col == 0:
                ax.set_ylabel(FEATURE_PAPER_LABELS.get(feat, feat), fontsize=13)
            if row == 0:
                ax.set_title(titles[slug], fontsize=14)
            if row == n_rows - 1:
                ax.set_xlabel("Feature value (model input)", fontsize=12)
    fig.legend(
        handles=noise_group_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    fig.suptitle("SHAP vs feature value (colored by true noise group)", fontsize=16, y=1.03)
    save_current_fig(out_base, tight=False)


def plot_beeswarm_by_noise_2x2(
    cohort_outputs: Mapping[str, Mapping[str, pd.DataFrame]],
    out_base: Path,
    *,
    feature_order: Sequence[str] | None = None,
) -> None:
    feature_order = list(feature_order or pooled_feature_order(cohort_outputs))
    n_feat = len(feature_order)
    fig, axes = plt.subplots(
        2, 2, figsize=(16, max(10, n_feat * 0.72 + 2)), sharey=True, sharex=True
    )
    layout = [
        (COHORT_A_SLUG, 1, NOISE_LABELS[1]),
        (COHORT_A_SLUG, 0, NOISE_LABELS[0]),
        (COHORT_B_SLUG, 1, NOISE_LABELS[1]),
        (COHORT_B_SLUG, 0, NOISE_LABELS[0]),
    ]
    for ax, (slug, noise_val, subtitle) in zip(axes.flat, layout):
        payload = cohort_outputs[slug]
        mask = payload["meta"]["noise_cat"].astype(int).eq(noise_val).values
        if not mask.any():
            ax.set_visible(False)
            continue
        _beeswarm_on_ax(
            ax,
            payload["shap"].loc[mask].reset_index(drop=True),
            payload["feat"].loc[mask].reset_index(drop=True),
            feature_order,
        )
        cohort_label = "Cohort A" if slug == COHORT_A_SLUG else "Cohort B"
        ax.set_title(f"{cohort_label} — {subtitle}", fontsize=13)
    fig.suptitle("MLP SHAP by true noise group (feature-colored)", fontsize=16, y=1.02)
    save_current_fig(out_base, tight=True)


def plot_scatter_grid_frequency_13x2(
    cohort_outputs: Mapping[str, Mapping[str, pd.DataFrame]],
    out_base: Path,
    *,
    feature_order: Sequence[str] | None = None,
) -> None:
    feature_order = list(feature_order or pooled_feature_order(cohort_outputs))
    n_rows = len(feature_order)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, max(8, n_rows * 2.6)),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.array([axes])
    titles = {"liberman": "Cohort A", "buran": "Cohort B"}
    last_sc = None
    for col, slug in enumerate(COHORT_SLUGS_DECK):
        payload = cohort_outputs[slug]
        freq = payload["meta"]["frequency"].astype(float).values
        vmin, vmax = float(np.nanmin(freq)), float(np.nanmax(freq))
        for row, feat in enumerate(feature_order):
            ax = axes[row, col]
            last_sc = ax.scatter(
                payload["feat"][feat],
                payload["shap"][feat],
                c=freq,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                s=8,
                alpha=0.65,
                linewidths=0,
                rasterized=len(freq) > 500,
            )
            if col == 0:
                ax.set_ylabel(FEATURE_PAPER_LABELS.get(feat, feat), fontsize=13)
            if row == 0:
                ax.set_title(titles[slug], fontsize=14)
            if row == n_rows - 1:
                ax.set_xlabel("Feature value (model input)", fontsize=12)
    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes.ravel().tolist(), label="Frequency (kHz)", shrink=0.6)
    fig.suptitle("SHAP vs feature value (colored by frequency)", fontsize=16, y=1.03)
    save_current_fig(out_base, tight=False)


def plot_level_scatter_frequency_1x2(
    cohort_outputs: Mapping[str, Mapping[str, pd.DataFrame]],
    out_base: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    titles = {"liberman": "Cohort A (Liberman)", "buran": "Cohort B (Brad)"}
    letters = ("A", "B")
    last_sc = None
    for ax, slug, letter in zip(axes, COHORT_SLUGS_DECK, letters):
        payload = cohort_outputs[slug]
        freq = payload["meta"]["frequency"].astype(float).values
        x = payload["meta"]["level"].astype(float).values
        y = payload["shap"]["level"].values
        vmin, vmax = float(np.nanmin(freq)), float(np.nanmax(freq))
        last_sc = ax.scatter(
            x,
            y,
            c=freq,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=12,
            alpha=0.7,
            linewidths=0,
        )
        ax.set_xlabel("Stimulus level (dB SPL)", fontsize=14)
        ax.set_ylabel("SHAP (level)", fontsize=14)
        ax.set_title(titles[slug], fontsize=15)
        _draw_panel_letter(fig, ax, letter)
    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes.tolist(), label="Frequency (kHz)", shrink=0.85)
    fig.suptitle("SHAP for stimulus level vs SPL (colored by frequency)", fontsize=16, y=1.05)
    save_current_fig(out_base, tight=False)


def run_mlp_shap_deck_figures(
    cohort_outputs: Mapping[str, Mapping[str, pd.DataFrame]],
    fig_dir: Path | None = None,
    *,
    plot_seed: int = 22,
) -> list[Path]:
    """Write deck-style two-panel MLP SHAP figures (PNG + SVG)."""
    fig_dir = Path(fig_dir or MLP_SHAP_FIG_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(plot_seed)
    feature_order = pooled_feature_order(cohort_outputs)
    specs = [
        ("mlp_shap_beeswarm_cohorts", plot_beeswarm_cohorts_1x2),
        ("mlp_shap_scatter_noise_cohorts", plot_scatter_grid_noise_13x2),
        ("mlp_shap_beeswarm_by_noise", plot_beeswarm_by_noise_2x2),
        ("mlp_shap_scatter_frequency_cohorts", plot_scatter_grid_frequency_13x2),
        ("mlp_shap_scatter_level_frequency", plot_level_scatter_frequency_1x2),
    ]
    written: list[Path] = []
    for stem, plot_fn in specs:
        base = fig_dir / stem
        if plot_fn is plot_level_scatter_frequency_1x2:
            plot_fn(cohort_outputs, base)
        else:
            plot_fn(cohort_outputs, base, feature_order=feature_order)
        written.extend([base.with_suffix(".png"), base.with_suffix(".svg")])
    return written


def run_mlp_shap_plots(
    shap_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    meta: pd.DataFrame,
    cohort_slug: str,
    fig_dir: Path,
) -> None:
    """Write legacy per-cohort MLP SHAP figure bundle (superseded by deck figures)."""
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

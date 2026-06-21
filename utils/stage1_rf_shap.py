"""
Stage-1 wide RF noise classifier SHAP via ``shap.TreeExplainer``.

Evaluates Validate + Test wide rows from RF fitted on Train/Fit split only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestClassifier

from utils.benchmark_metrics import apply_slide_rcparams
from utils.nn_stage2 import fit_stage1_wide_rf
from utils.nn_stage2_data import (
    NNStage2Data,
    load_nn_stage2_data,
    splits_for_long_stage2,
    wide_stage1_fit,
    wide_stage1_val,
)
from utils.stage2_xgb_shap import (
    COHORT_A_SLUG,
    COHORT_B_SLUG,
    FEATURE_PAPER_LABELS,
    FEATURE_TYPE_ORDER,
    _cohort_shap,
    _rank_descending,
    _type_mean_abs_score,
    feature_annotations,
)

STAGE1_SHAP_CACHE_DIR = Path("figures/cache/shap_stage1")
STAGE1_SHAP_FIG_DIR = Path("figures/shap_stage1")
COHORT_SLUGS_DECK = (COHORT_A_SLUG, COHORT_B_SLUG)
COHORT_TO_LAB = {COHORT_A_SLUG: "Liberman", COHORT_B_SLUG: "Brad"}

STAGE1_FEATURE_TYPE_ORDER = FEATURE_TYPE_ORDER
STAGE1_AGGREGATED_TYPE_ORDER: tuple[str, ...] = STAGE1_FEATURE_TYPE_ORDER + ("frequency",)
STAGE1_ROW_ORDER = list(STAGE1_FEATURE_TYPE_ORDER) + ["frequency"]

STAGE1_EXPORT_META_COLS = ("animal_id",)
N_STAGE1_WIDE_FEATURES = 41

# Deck figures: 3 bases per cohort (PNG + SVG each).
STAGE1_DECK_FIGURE_STEMS = (
    "s1_shap_beeswarm_featuretypes",
    "s1_shap_bar_featuretypes",
    "s1_shap_beeswarm_featuretypes_noisecolor",
)

NOISE_EXPOSED_COLOR = "#E69F00"
UNEXPOSED_COLOR = "#009E73"


def prep_feature_names(clf, feat_cols: Sequence[str], *, n_transformed: int) -> list[str]:
    """Map prep output columns to raw wide names (order: num branch then log branch)."""
    if n_transformed != len(feat_cols):
        raise ValueError(
            f"Transformed width mismatch: expected {len(feat_cols)}, got {n_transformed}"
        )
    prep = clf.named_steps["prep"]
    try:
        raw = prep.get_feature_names_out()
    except AttributeError:
        # mk_log_pipe lacks get_feature_names_out on inner FunctionTransformer; order is stable.
        return list(feat_cols)
    stripped = [n.split("__", 1)[-1] if "__" in n else n for n in raw]
    if list(stripped) != list(feat_cols):
        raise ValueError(
            "Prep feature name mismatch. "
            f"expected={list(feat_cols)}, got={stripped}"
        )
    return stripped


def binary_class1_shap(explainer: shap.TreeExplainer, X_t: np.ndarray) -> np.ndarray:
    """Return class-1 SHAP matrix ``(n_rows, n_features)``."""
    shap_raw = explainer.shap_values(X_t)
    if isinstance(shap_raw, (list, tuple)):
        shap_vals = np.asarray(shap_raw[1], dtype=float)
    else:
        shap_vals = np.asarray(shap_raw, dtype=float)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]
    if shap_vals.ndim != 2:
        raise ValueError(f"Expected 2D SHAP array, got shape {shap_vals.shape}")
    return shap_vals


def _cohort_wide_frames(
    cohort_slug: str,
    data: NNStage2Data,
    splits: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    if cohort_slug == COHORT_B_SLUG:
        wide_tr = splits["bb_wide_train"]
        wide_te = splits["bb_wide_test"]
        noise_num = list(data.noise_num_bb)
        noise_log = list(data.noise_log_bb)
    elif cohort_slug == COHORT_A_SLUG:
        wide_tr = splits["lib_train"]
        wide_te = splits["lib_test"]
        noise_num = list(data.noise_num_lib)
        noise_log = list(data.noise_log_lib)
    else:
        raise KeyError(f"Unknown cohort_slug={cohort_slug!r}")
    fit = wide_stage1_fit(wide_tr)
    val = wide_stage1_val(wide_tr)
    return fit, val, wide_te, noise_num, noise_log


def compute_cohort_stage1_rf_shap(
    cohort_slug: str,
    data: NNStage2Data | None = None,
    splits: Mapping[str, pd.DataFrame] | None = None,
    *,
    max_eval_rows: int | None = None,
    random_state: int = 1,
) -> dict[str, Any]:
    """Fit Stage-1 RF on Train; TreeExplainer SHAP on Validate + Test rows."""
    data = data or load_nn_stage2_data()
    splits = splits or splits_for_long_stage2(data)

    fit, val, test, noise_num, noise_log = _cohort_wide_frames(cohort_slug, data, splits)
    feat_cols = noise_num + noise_log
    if len(feat_cols) != N_STAGE1_WIDE_FEATURES:
        raise ValueError(
            f"{cohort_slug}: expected {N_STAGE1_WIDE_FEATURES} features, got {len(feat_cols)}"
        )

    s1 = fit_stage1_wide_rf(
        fit, val, test, noise_num, noise_log, random_state=random_state, verbose=False
    )
    clf = s1["clf"]
    rf = clf.named_steps["rf"]
    if not isinstance(rf, RandomForestClassifier):
        raise TypeError(f"Expected RandomForestClassifier, got {type(rf)!r}")

    eval_wide = pd.concat([val, test], ignore_index=True)
    X_eval_raw = eval_wide[feat_cols].dropna()
    if max_eval_rows is not None and len(X_eval_raw) > max_eval_rows:
        X_eval_raw = X_eval_raw.sample(n=max_eval_rows, random_state=random_state)

    prep = clf.named_steps["prep"]
    X_t = prep.transform(X_eval_raw)
    feat_names = prep_feature_names(clf, feat_cols, n_transformed=X_t.shape[1])

    explainer = shap.TreeExplainer(rf)
    shap_vals = binary_class1_shap(explainer, X_t)

    proba = pd.Series(clf.predict_proba(X_eval_raw)[:, 1], index=X_eval_raw.index)
    threshold = float(s1["stage1_threshold"])
    y_pred = (proba.values >= threshold).astype(np.int8)

    meta_cols = ["animal_id", "frequency", "noise_cat", "experimental_group", "DataGroup"]
    meta_df = eval_wide.loc[X_eval_raw.index, meta_cols].copy()
    meta_df["proba"] = proba.values
    meta_df["y_pred"] = y_pred
    meta_df["cohort"] = cohort_slug

    shap_df = pd.DataFrame(shap_vals, columns=feat_names, index=X_eval_raw.index)
    feat_df = pd.DataFrame(
        X_eval_raw.values, columns=feat_names, index=X_eval_raw.index
    )
    feature_table = feature_annotations(feat_names)

    return {
        "shap": shap_df.reset_index(drop=True),
        "feat": feat_df.reset_index(drop=True),
        "meta": meta_df.reset_index(drop=True),
        "feature_table": feature_table,
        "s1": s1,
        "feat_cols": feat_cols,
    }


def export_stage1_shap_parquet(
    cohort_slug: str,
    shap_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    feature_table: pd.DataFrame | None = None,
    cache_dir: Path | str | None = None,
) -> dict[str, Path]:
    cache = Path(cache_dir or STAGE1_SHAP_CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)

    shap_export = pd.concat(
        [meta_df[list(STAGE1_EXPORT_META_COLS)], shap_df],
        axis=1,
    )
    if not shap_export.columns.is_unique:
        dupes = shap_export.columns[shap_export.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns in shap export: {dupes}")

    shap_path = cache / f"shap_values_{cohort_slug}.parquet"
    meta_path = cache / f"shap_metadata_{cohort_slug}.parquet"
    shap_export.to_parquet(shap_path, index=False)
    meta_df.to_parquet(meta_path, index=False)

    paths = {"shap": shap_path, "meta": meta_path}
    if feature_table is not None:
        feat_path = cache / "feature_names_stage1.parquet"
        feature_table.to_parquet(feat_path, index=False)
        paths["feature_names"] = feat_path
    return paths


def load_stage1_cohort_outputs_from_cache(
    cache_dir: Path | str | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Reload Stage-1 SHAP values and metadata from exported parquets."""
    cache = Path(cache_dir or STAGE1_SHAP_CACHE_DIR)
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for slug in COHORT_SLUGS_DECK:
        shap_path = cache / f"shap_values_{slug}.parquet"
        meta_path = cache / f"shap_metadata_{slug}.parquet"
        if not shap_path.is_file():
            raise FileNotFoundError(shap_path)
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)
        shap_full = pd.read_parquet(shap_path)
        drop_cols = [c for c in STAGE1_EXPORT_META_COLS if c in shap_full.columns]
        outputs[slug] = {
            "shap": shap_full.drop(columns=drop_cols),
            "meta": pd.read_parquet(meta_path),
        }
    return outputs


def aggregated_feature_type_comparison_table_stage1(
    cohort_outputs: Mapping[str, Mapping[str, pd.DataFrame]],
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """Sum of per-column mean |SHAP| within each Stage-1 feature type."""
    shap_a = _cohort_shap(cohort_outputs, COHORT_A_SLUG)
    shap_b = _cohort_shap(cohort_outputs, COHORT_B_SLUG)
    pooled = pd.concat([shap_a, shap_b], ignore_index=True)
    by_type = feature_table.groupby("feature_type")["feature_name"].apply(list)

    rows: list[dict[str, object]] = []
    pooled_scores: dict[str, float] = {}
    mean_a: dict[str, float] = {}
    mean_b: dict[str, float] = {}

    for ftype in STAGE1_AGGREGATED_TYPE_ORDER:
        type_cols = [c for c in by_type.get(ftype, []) if c in shap_a.columns]
        if not type_cols:
            continue
        score_a = _type_mean_abs_score(shap_a, type_cols)
        score_b = _type_mean_abs_score(shap_b, type_cols)
        pooled_scores[ftype] = _type_mean_abs_score(pooled, type_cols)
        mean_a[ftype] = score_a
        mean_b[ftype] = score_b
        rows.append(
            {
                "feature_type": ftype,
                "feature_type_label": FEATURE_PAPER_LABELS.get(ftype, ftype),
                "cohort_a_mean_abs_shap": score_a,
                "cohort_b_mean_abs_shap": score_b,
            }
        )

    if not rows:
        raise ValueError("No aggregated feature types with SHAP columns")

    order = sorted(pooled_scores, key=lambda k: pooled_scores[k], reverse=True)
    out = pd.DataFrame(rows).set_index("feature_type").loc[order].reset_index()
    rank_a = _rank_descending(pd.Series(mean_a))
    rank_b = _rank_descending(pd.Series(mean_b))
    out["cohort_a_rank"] = [int(rank_a[ftype]) for ftype in order]
    out["cohort_b_rank"] = [int(rank_b[ftype]) for ftype in order]
    return out.reset_index(drop=True)


def _save_current_fig(base_no_ext: Path, *, tight: bool = True) -> None:
    base_no_ext.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(base_no_ext.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(base_no_ext.with_suffix(".svg"), bbox_inches="tight")
    plt.close()


def collapse_stage1_shap_by_type(
    shap_df: pd.DataFrame,
    feature_table: pd.DataFrame,
    *,
    include_frequency: bool = False,
) -> pd.DataFrame:
    """Collapse wide Stage-1 SHAP columns to ABR feature types (public API)."""
    dummy_feat = shap_df.copy()
    vals_type, _ = _collapse_by_feature_type(shap_df, dummy_feat, feature_table)
    if include_frequency and "frequency" in shap_df.columns:
        vals_type["frequency"] = shap_df["frequency"]
    if not include_frequency:
        keep = [c for c in STAGE1_FEATURE_TYPE_ORDER if c in vals_type.columns]
        vals_type = vals_type[keep]
    return vals_type


def _collapse_by_feature_type(
    df_vals: pd.DataFrame,
    df_feat: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_type = feature_table.groupby("feature_type")["feature_name"].apply(list)
    val_out: dict[str, pd.Series] = {}
    feat_out: dict[str, pd.Series] = {}
    for ftype in STAGE1_FEATURE_TYPE_ORDER:
        cols = by_type.get(ftype, [])
        keep = [c for c in cols if c in df_vals.columns]
        if keep:
            val_out[ftype] = df_vals[keep].mean(axis=1)
            feat_out[ftype] = df_feat[keep].mean(axis=1)
    return pd.DataFrame(val_out, index=df_vals.index), pd.DataFrame(
        feat_out, index=df_feat.index
    )


def _build_featuretype_frames(
    df_vals: pd.DataFrame,
    df_feat: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vals_type, feat_type = _collapse_by_feature_type(df_vals, df_feat, feature_table)
    for col in ("frequency",):
        if col in df_vals.columns:
            vals_type[col] = df_vals[col]
            feat_type[col] = df_feat[col]
    row_order = [c for c in STAGE1_ROW_ORDER if c in vals_type.columns]
    rename = {c: FEATURE_PAPER_LABELS.get(c, c) for c in row_order}
    return (
        vals_type[row_order].rename(columns=rename),
        feat_type[row_order].rename(columns=rename),
    )


def _summary_beeswarm(
    values: pd.DataFrame,
    features: pd.DataFrame,
    title: str,
    out_base: Path,
    max_display: int = 10,
) -> None:
    shap.summary_plot(
        values.values,
        features=features.values,
        feature_names=list(values.columns),
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title(title)
    _save_current_fig(out_base)


def _summary_bar(
    values: pd.DataFrame,
    features: pd.DataFrame,
    title: str,
    out_base: Path,
    max_display: int = 10,
) -> None:
    shap.summary_plot(
        values.values,
        features=features.values,
        feature_names=list(values.columns),
        max_display=max_display,
        show=False,
        plot_type="bar",
    )
    plt.title(title)
    _save_current_fig(out_base)


def _true_noise_point_colors(meta: pd.DataFrame) -> np.ndarray:
    exposed = meta["noise_cat"].astype(int).eq(1).values
    colors = np.empty(len(meta), dtype=object)
    colors[exposed] = NOISE_EXPOSED_COLOR
    colors[~exposed] = UNEXPOSED_COLOR
    return colors


def _noise_group_legend_handles() -> list[Line2D]:
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


def _beeswarm_dot_y_offsets(shaps: np.ndarray, *, row_height: float = 0.4) -> np.ndarray:
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


def _summary_beeswarm_true_noise_color(
    values: pd.DataFrame,
    features: pd.DataFrame,
    meta: pd.DataFrame,
    out_base: Path,
    *,
    max_display: int = 10,
) -> None:
    cols = [c for c in STAGE1_FEATURE_TYPE_ORDER if c in values.columns]
    shap_vals = values[cols].values
    feature_names = [FEATURE_PAPER_LABELS.get(c, c) for c in cols]
    dot_colors = _true_noise_point_colors(meta)

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
        ys = _beeswarm_dot_y_offsets(shaps, row_height=row_height)
        ax.scatter(
            shaps,
            pos + ys,
            c=colors,
            s=16,
            alpha=1.0,
            linewidth=0,
            zorder=3,
            rasterized=len(shaps) > 500,
        )

    ax.set_yticks(range(len(feature_order)), [feature_names[i] for i in feature_order])
    ax.tick_params(axis="y", labelsize=13, length=20, width=0.5)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_ylim(-1, len(feature_order))
    ax.set_xlabel("SHAP value")
    for spine in ("right", "top", "left"):
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.legend(
        handles=_noise_group_legend_handles(),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=11,
    )
    _save_current_fig(out_base, tight=True)


def run_stage1_shap_deck_figures(
    cohort_outputs: Mapping[str, Mapping[str, pd.DataFrame]],
    fig_dir: Path | str | None = None,
    *,
    feature_table: pd.DataFrame | None = None,
) -> list[str]:
    """Write 3 PNG+SVG SHAP figures per cohort; return created base paths."""
    apply_slide_rcparams()
    fig_dir = Path(fig_dir or STAGE1_SHAP_FIG_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if feature_table is None:
        sample = next(iter(cohort_outputs.values()))["shap"]
        feature_table = feature_annotations(sample.columns.tolist())

    created: list[str] = []
    for cohort_slug, payload in cohort_outputs.items():
        vals_raw = payload["shap"]
        feat_raw = payload["feat"]
        meta = payload["meta"]

        vals_type, feat_type = _collapse_by_feature_type(vals_raw, feat_raw, feature_table)
        vals_ft, feat_ft = _build_featuretype_frames(vals_raw, feat_raw, feature_table)

        p = fig_dir / f"s1_shap_beeswarm_featuretypes_{cohort_slug}"
        _summary_beeswarm(
            vals_ft,
            feat_ft,
            f"Feature types + frequency ({cohort_slug})",
            p,
            max_display=len(vals_ft.columns),
        )
        created.append(str(p))

        p = fig_dir / f"s1_shap_bar_featuretypes_{cohort_slug}"
        _summary_bar(
            vals_type,
            feat_type,
            f"All feature types mean |SHAP| ({cohort_slug})",
            p,
            max_display=10,
        )
        created.append(str(p))

        p = fig_dir / f"s1_shap_beeswarm_featuretypes_noisecolor_{cohort_slug}"
        _summary_beeswarm_true_noise_color(
            vals_type,
            feat_type,
            meta,
            p,
            max_display=10,
        )
        created.append(str(p))

    return created

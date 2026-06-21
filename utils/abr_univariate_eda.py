"""
Univariate EDA: raw (pre-log) ABR features vs. synapses per IHC.

Dual grain: long (animal × frequency × SPL) and animal_freq (80 dB nearest-level dedupe).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from utils.benchmark_metrics import apply_slide_rcparams
from utils.nn_stage2_data import LONG_LOG, NNStage2Data, load_nn_stage2_data, splits_for_long_stage2

logger = logging.getLogger(__name__)

DEDUP_TARGET_DB = 80.0
GRAIN_LONG = "long"
GRAIN_ANIMAL_FREQ = "animal_freq"

RAW_EDA_FEATURES: tuple[str, ...] = (
    "amplitude",
    "distance",
    "slope",
    "total_variance",
    "PeakIEarlyCurvature",
    "PeakICentralCurvature",
    "PeakILateCurvature",
    "TroughIEarlyCurvature",
    "TroughICentralCurvature",
    "TroughILateCurvature",
)

RAW_FEATURE_LABELS: dict[str, str] = {
    "amplitude": "Amplitude",
    "distance": "Peak-to-trough latency",
    "slope": "Wave I slope",
    "total_variance": "Total variance",
    "PeakIEarlyCurvature": "Peak I curvature (early)",
    "PeakICentralCurvature": "Peak I curvature (central)",
    "PeakILateCurvature": "Peak I curvature (late)",
    "TroughIEarlyCurvature": "Trough I curvature (early)",
    "TroughICentralCurvature": "Trough I curvature (central)",
    "TroughILateCurvature": "Trough I curvature (late)",
}

# Print scatter axes: wrap long x-labels onto a second line where needed.
PRINT_FEATURE_LABELS: dict[str, str] = {
    "amplitude": "Amplitude",
    "distance": "Peak-to-trough\nlatency",
    "slope": "Wave I slope",
    "total_variance": "Total variance",
    "PeakIEarlyCurvature": "P1 curvature\n(early)",
    "PeakICentralCurvature": "P1 curvature\n(central)",
    "PeakILateCurvature": "P1 curvature\n(late)",
    "TroughIEarlyCurvature": "T1 curvature\n(early)",
    "TroughICentralCurvature": "T1 curvature\n(central)",
    "TroughILateCurvature": "T1 curvature\n(late)",
}

COHORT_A_LABEL = "A"
COHORT_B_LABEL = "B"
COHORT_TITLES = {COHORT_A_LABEL: "Cohort A (Liberman)", COHORT_B_LABEL: "Cohort B (Brad)"}

NOISE_LOWER_COLOR = "#0173B2"
NOISE_HIGHER_COLOR = "#DE8F05"
NOISE_MARKERS = {0: "o", 1: "^"}
PRINT_SCATTER_MARKER = "o"
NOISE_LABELS = {0: "Low noise exposure", 1: "High noise exposure"}
NOISE_CAPTION_COLOR_PHRASE = (
    f"blue, {NOISE_LABELS[0].lower()}; orange, {NOISE_LABELS[1].lower()}"
)

CROSS_COHORT_NOTE = (
    "Brad vs. Liberman **amplitude** and **peak-to-trough latency** (`distance`) are "
    "constructed with the **same formulas** but on **different waveform grids**. "
    "Cohort B waveforms were resampled to the Liberman timebase (**LibT**), so numeric "
    "ranges are **broadly comparable** across cohorts, with small residual differences "
    "from **resampling interpolation** and **landmark detection method** (in-house "
    "peaks/troughs vs IO latencies). Cross-cohort feature comparisons in the EDA should "
    "be interpreted with this in mind, but are **not precluded**."
)

LONG_GRAIN_FOOTER = (
    "Synapses per IHC is constant within (animal, frequency); multiple SPL levels share y."
)

FIG_EDA_DIR = Path("figures/eda")
CACHE_EDA_DIR = Path("figures/cache/eda")

BONFERRONI_ALPHA = 0.005
BONFERRONI_N_TESTS = len(RAW_EDA_FEATURES)
FIGURE_04_STEM = "figure_04_raw_vs_synapses_cohort_A"
FIGURE_05_STEM = "figure_05_raw_vs_synapses_cohort_B"
FIGURE_COMBINED_STEM = "figure_combined_raw_vs_synapses_cohorts_5x4"
PRINT_FIGSIZE = (7.0, 10.0)
PRINT_FIG_WIDTH_IN = PRINT_FIGSIZE[0]
PRINT_DPI = 300
PRINT_PAD_INCHES = 0.08
PRINT_MARGIN_LR_INCHES = 0.25
PRINT_LR_INSET = PRINT_MARGIN_LR_INCHES / PRINT_FIG_WIDTH_IN
SCATTER_ALPHA = 0.35
LEGEND_ROW_RATIO = 0.07
LEGEND_PANEL_HSPACE = 0.0
LEGEND_FRAME_COLOR = "#888888"
LEGEND_FRAME_LINEWIDTH = 0.5
# Panel ID / stat sit just above the axes box (outside data area).
# Panel IDs use ha="right": x is where the label *ends* (right edge), not where it starts.
PANEL_ID_COHORT_XR = 0.0
PANEL_ID_COHORT_Y = 1.03
PANEL_ID_COMBINED_XR = 0.01
PANEL_ID_COMBINED_Y = 1.05
STAT_COHORT_XY = (1.0, 1.03)
STAT_COMBINED_XY = (1.0, 1.05)
STAT_FONT_COHORT = 9.0
STAT_FONT_COMBINED = 8.0
PRINT_PANEL_HSPACE_COHORT = 0.54
PRINT_PANEL_HSPACE_COMBINED = 0.73
PRINT_SUBPLOT_LEFT_COHORT = 0.14
PRINT_SUBPLOT_RIGHT_COHORT = 0.96
PRINT_SUBPLOT_LEFT_COMBINED = 0.11
PRINT_SUBPLOT_RIGHT_COMBINED = 0.97
PRINT_XLABEL_FONTSIZE_COHORT = 10.0
PRINT_XLABEL_FONTSIZE_COMBINED = 9.0
PRINT_RC: dict[str, float | int | str] = {
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
}

FIGURE_STEM_BY_COHORT = {
    COHORT_A_LABEL: (FIGURE_04_STEM, 4),
    COHORT_B_LABEL: (FIGURE_05_STEM, 5),
}

SKEW_THRESHOLD = 1.0
NONLINEAR_RHO_DELTA = 0.15
STRONG_RHO_MIN = 0.25
STRONG_P_MAX = 0.05
STRONG_N_MIN = 20
GRAIN_DELTA_RHO_WARN = 0.10
SMALL_STRATUM_N = 20


def verify_feature_identity() -> None:
    """Log feature column contract (distance = latency, slope = Wave I slope)."""
    excluded = {"Slope_all", "Slope_high4", "p1_latency"}
    overlap_log = set(RAW_EDA_FEATURES) & set(LONG_LOG)
    logger.info(
        "EDA features: %d columns; distance=peak-to-trough latency; "
        "slope=-amplitude/distance (Wave I); log1p candidates=%s",
        len(RAW_EDA_FEATURES),
        sorted(overlap_log),
    )
    logger.info("Excluded from RAW_EDA_FEATURES: %s", sorted(excluded))


def _level_at_target(level: float, target: float = DEDUP_TARGET_DB, tol: float = 1e-6) -> bool:
    return bool(np.isfinite(level) and abs(float(level) - target) <= tol)


def dedupe_animal_freq(
    df: pd.DataFrame,
    *,
    target_db: float = DEDUP_TARGET_DB,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per (animal_id, frequency): level nearest to ``target_db``."""
    need = ["animal_id", "frequency", "level"] + list(RAW_EDA_FEATURES)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"dedupe_animal_freq missing columns: {missing}")

    work = df.copy()
    work["level"] = work["level"].astype(float)
    work["_level_dist"] = (work["level"] - float(target_db)).abs()
    work = work.sort_values(
        ["animal_id", "frequency", "_level_dist", "level"],
        kind="mergesort",
    )
    out = work.groupby(["animal_id", "frequency"], as_index=False).head(1)
    n_at_80 = int(out["level"].map(lambda x: _level_at_target(x, target_db)).sum())
    stats_out = {
        "n_rows": len(out),
        "n_at_80": n_at_80,
        "n_nearest_fallback": len(out) - n_at_80,
    }
    return out.drop(columns=["_level_dist"]), stats_out


def prepare_eda_frame(
    df: pd.DataFrame,
    cohort: str,
    grain: Literal["long", "animal_freq"],
    *,
    dedup_stats_out: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Subset features, dropna, valid noise_cat; attach cohort label."""
    cols = list(RAW_EDA_FEATURES) + ["synapses", "noise_cat", "animal_id", "frequency", "level"]
    out = df[cols].copy()
    out["cohort"] = cohort
    out["noise_cat"] = out["noise_cat"].astype(int)
    out = out[out["noise_cat"].isin([0, 1])]
    out = out.dropna(subset=list(RAW_EDA_FEATURES) + ["synapses"])

    if grain == GRAIN_ANIMAL_FREQ:
        out, st = dedupe_animal_freq(out)
        if dedup_stats_out is not None:
            dedup_stats_out.update(st)
    out["grain"] = grain
    return out.reset_index(drop=True)


def _spearman_one(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x.loc[mask], y.loc[mask])
    return float(rho), float(p), n


def spearman_table(
    frames: Mapping[str, pd.DataFrame],
    features: Sequence[str] = RAW_EDA_FEATURES,
    grain: str = GRAIN_LONG,
) -> pd.DataFrame:
    """Long-format Spearman: feature × cohort × noise_group (+ cohort all)."""
    rows: list[dict[str, Any]] = []
    for cohort, df in frames.items():
        for noise_group, noise_val in [("lower", 0), ("higher", 1), ("all", None)]:
            sub = df if noise_val is None else df.loc[df["noise_cat"] == noise_val]
            for feat in features:
                rho, p, n = _spearman_one(sub[feat], sub["synapses"])
                rows.append(
                    {
                        "grain": grain,
                        "feature": feat,
                        "cohort": cohort,
                        "noise_group": noise_group,
                        "n": n,
                        "rho": rho,
                        "p_value": p,
                    }
                )
    return pd.DataFrame(rows)


def spearman_table_wide(table: pd.DataFrame) -> pd.DataFrame:
    """Pivot rho for notebook display."""
    if table.empty:
        return table
    t = table.copy()
    t["feature_label"] = t["feature"].map(lambda f: RAW_FEATURE_LABELS.get(f, f))
    t["stratum"] = t["cohort"] + " / " + t["noise_group"]
    return t.pivot_table(
        index="feature_label",
        columns="stratum",
        values="rho",
        aggfunc="first",
    )


def compare_grain_spearman(long_tbl: pd.DataFrame, af_tbl: pd.DataFrame) -> pd.DataFrame:
    """Join long vs animal_freq on feature, cohort, noise_group."""
    keys = ["feature", "cohort", "noise_group"]
    a = long_tbl[keys + ["rho", "n", "p_value"]].rename(
        columns={"rho": "rho_long", "n": "n_long", "p_value": "p_long"}
    )
    b = af_tbl[keys + ["rho", "n", "p_value"]].rename(
        columns={"rho": "rho_animal_freq", "n": "n_animal_freq", "p_value": "p_animal_freq"}
    )
    out = a.merge(b, on=keys, how="outer")
    out["delta_rho"] = out["rho_long"] - out["rho_animal_freq"]
    out["abs_delta_rho"] = out["delta_rho"].abs()
    return out.sort_values(["abs_delta_rho", "feature"], ascending=[False, True])


def feature_skewness_table(
    frames: Mapping[str, pd.DataFrame],
    features: Sequence[str] = RAW_EDA_FEATURES,
    *,
    grain: str = GRAIN_LONG,
    threshold: float = SKEW_THRESHOLD,
) -> pd.DataFrame:
    """Skewness of raw feature values (long grain) for every cohort × feature."""
    log_feats = set(LONG_LOG) & set(features)
    rows: list[dict[str, Any]] = []
    for cohort, df in frames.items():
        for feat in features:
            vals = df[feat].dropna()
            n = int(len(vals))
            if n < 8:
                sk = np.nan
            else:
                sk = float(stats.skew(vals))
            abs_sk = abs(sk) if np.isfinite(sk) else np.nan
            in_log = feat in log_feats
            rows.append(
                {
                    "grain": grain,
                    "cohort": cohort,
                    "feature": feat,
                    "feature_label": RAW_FEATURE_LABELS.get(feat, feat),
                    "in_LONG_LOG": in_log,
                    "n": n,
                    "skew": sk,
                    "abs_skew": abs_sk,
                    "skew_threshold": threshold,
                    "flag_log_candidate": bool(
                        in_log and np.isfinite(sk) and abs_sk > threshold
                    ),
                }
            )
    return pd.DataFrame(rows)


def flag_skewed_features(
    skew_tbl: pd.DataFrame,
    *,
    threshold: float = SKEW_THRESHOLD,
) -> list[dict[str, Any]]:
    """Subset of ``feature_skewness_table`` with |skew| > threshold and in LONG_LOG."""
    flags = []
    for _, row in skew_tbl.iterrows():
        if not row["in_LONG_LOG"] or not row["flag_log_candidate"]:
            continue
        flags.append(
            {
                "feature": row["feature"],
                "cohort": row["cohort"],
                "skew": row["skew"],
                "threshold": threshold,
                "reason": "log1p candidate (LONG_LOG)",
            }
        )
    return flags


def flag_nonlinear_candidates(
    frames: Mapping[str, pd.DataFrame],
    features: Sequence[str] = RAW_EDA_FEATURES,
    grain: str = GRAIN_LONG,
    *,
    delta: float = NONLINEAR_RHO_DELTA,
) -> list[dict[str, Any]]:
    flags = []
    for cohort, df in frames.items():
        for feat in features:
            x, y = df[feat], df["synapses"]
            mask = x.notna() & y.notna()
            if mask.sum() < STRONG_N_MIN:
                continue
            rho_s, _ = stats.spearmanr(x.loc[mask], y.loc[mask])
            rho_p, _ = stats.pearsonr(x.loc[mask], y.loc[mask])
            gap = abs(float(rho_s) - float(rho_p))
            if gap > delta:
                flags.append(
                    {
                        "grain": grain,
                        "feature": feat,
                        "cohort": cohort,
                        "rho_spearman": float(rho_s),
                        "rho_pearson": float(rho_p),
                        "gap": gap,
                    }
                )
    return flags


def flag_strong_associations(
    table: pd.DataFrame,
    *,
    rho_min: float = STRONG_RHO_MIN,
    p_max: float = STRONG_P_MAX,
    n_min: int = STRONG_N_MIN,
) -> list[dict[str, Any]]:
    flags = []
    for _, row in table.iterrows():
        if row["noise_group"] == "all":
            continue
        if row["n"] < n_min or not np.isfinite(row["rho"]):
            continue
        if abs(row["rho"]) >= rho_min and row["p_value"] < p_max:
            flags.append(row.to_dict())
    return flags


def _save_fig(
    fig: plt.Figure,
    base: Path,
    *,
    dpi: int = 300,
    pad_inches: float = 0.08,
) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    fig.savefig(base.with_suffix(".png"), **kw)
    fig.savefig(base.with_suffix(".svg"), **kw)
    plt.close(fig)


def _spearman_ci95(rho: float, n: int) -> tuple[float, float]:
    """Approximate 95% CI for Spearman ρ via Fisher z (matches example-style parentheses)."""
    if not np.isfinite(rho) or n < 4:
        return np.nan, np.nan
    z = np.arctanh(float(np.clip(rho, -0.999, 0.999)))
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def _format_spearman_stat_line(rho: float, p: float, n: int) -> str:
    """Print annotation: ``*0.34 (-0.12, 0.51)``; leading ``*`` if p < Bonferroni α."""
    if not np.isfinite(rho):
        return "—"
    star = "*" if np.isfinite(p) and p < BONFERRONI_ALPHA else ""
    lo, hi = _spearman_ci95(rho, n)
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{star}{rho:.2f} ({lo:.2f}, {hi:.2f})"
    return f"{star}{rho:.2f} (p={p:.3g})"


def _panel_id(cohort: str, feature_index: int) -> str:
    """Cohort letter + 1-based feature index (A1–A10, B1–B10)."""
    return f"{cohort}{feature_index + 1}"


def _style_print_scatter_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _build_row_shared_axes(
    fig: plt.Figure,
    panel_gs: Any,
    n_row: int,
    n_col: int,
) -> np.ndarray:
    """One shared y-scale per row; y tick labels only on column 0."""
    axes = np.empty((n_row, n_col), dtype=object)
    for row in range(n_row):
        for col in range(n_col):
            if col == 0:
                axes[row, col] = fig.add_subplot(panel_gs[row, col])
            else:
                axes[row, col] = fig.add_subplot(
                    panel_gs[row, col], sharey=axes[row, 0]
                )
                axes[row, col].tick_params(axis="y", labelleft=False)
    return axes


def _plot_raw_synapses_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    feat: str,
    *,
    panel_id: str,
    cohort: str,
    figure_num: int,
    spearman_rows: list[dict[str, Any]] | None,
    show_ylabel: bool,
    show_yticklabels: bool = True,
    record_spearman: bool = True,
    layout: Literal["cohort", "combined"] = "cohort",
) -> None:
    if layout == "combined":
        panel_xr, panel_y = PANEL_ID_COMBINED_XR, PANEL_ID_COMBINED_Y
    else:
        panel_xr, panel_y = PANEL_ID_COHORT_XR, PANEL_ID_COHORT_Y
    stat_xy = STAT_COMBINED_XY if layout == "combined" else STAT_COHORT_XY
    stat_fs = STAT_FONT_COMBINED if layout == "combined" else STAT_FONT_COHORT
    for noise_val in (0, 1):
        m = df["noise_cat"] == noise_val
        sub = df.loc[m]
        if sub.empty:
            continue
        ax.scatter(
            sub[feat].astype(float),
            sub["synapses"].astype(float),
            c=NOISE_LOWER_COLOR if noise_val == 0 else NOISE_HIGHER_COLOR,
            marker=PRINT_SCATTER_MARKER,
            s=10 if layout == "combined" else 12,
            alpha=SCATTER_ALPHA,
            linewidths=0,
            rasterized=len(sub) > 400,
            zorder=3,
        )

    rho_all, p_all, n_all = _spearman_one(df[feat], df["synapses"])
    sig_all = bool(np.isfinite(p_all) and p_all < BONFERRONI_ALPHA)
    ci_lo, ci_hi = _spearman_ci95(rho_all, n_all)
    if record_spearman and spearman_rows is not None:
        spearman_rows.append(
            {
                "figure": figure_num,
                "panel": panel_id,
                "cohort": cohort,
                "feature": feat,
                "noise_group": "all",
                "n": n_all,
                "rho": rho_all,
                "p_value": p_all,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "significant_bonferroni": sig_all,
            }
        )
    x_all = df[feat].astype(float).values
    y_all = df["synapses"].astype(float).values
    mask_all = np.isfinite(x_all) & np.isfinite(y_all)
    if sig_all and mask_all.sum() >= 3:
        _overlay_ols_trend(ax, x_all[mask_all], y_all[mask_all])

    ax.text(
        stat_xy[0],
        stat_xy[1],
        _format_spearman_stat_line(rho_all, p_all, n_all),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=stat_fs,
        clip_on=False,
    )
    ax.text(
        panel_xr,
        panel_y,
        panel_id,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10 if layout == "combined" else 11,
        fontweight="bold",
        clip_on=False,
    )
    xlab_fs = (
        PRINT_XLABEL_FONTSIZE_COMBINED
        if layout == "combined"
        else PRINT_XLABEL_FONTSIZE_COHORT
    )
    ax.set_xlabel(PRINT_FEATURE_LABELS.get(feat, feat), fontsize=xlab_fs)
    if show_ylabel:
        ax.set_ylabel("Synapses per IHC", fontsize=10)
    if not show_yticklabels:
        ax.tick_params(axis="y", labelleft=False)
    _style_print_scatter_ax(ax)


def _noise_legend_handles(*, markersize: float = 6.0) -> list[plt.Line2D]:
    return [
        plt.Line2D(
            [0],
            [0],
            marker=PRINT_SCATTER_MARKER,
            color="w",
            markerfacecolor=NOISE_LOWER_COLOR if k == 0 else NOISE_HIGHER_COLOR,
            markeredgecolor="0.25",
            markersize=markersize,
            label=NOISE_LABELS[k],
            linestyle="None",
        )
        for k in (0, 1)
    ]


def _draw_framed_noise_legend(ax: plt.Axes, *, ncol: int = 2) -> None:
    """Framed legend strip (matches synthesis cohort-panel print style)."""
    ax.set_axis_off()
    leg = ax.legend(
        handles=_noise_legend_handles(),
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fancybox=False,
        edgecolor=LEGEND_FRAME_COLOR,
        facecolor="white",
        framealpha=1.0,
    )
    leg.get_frame().set_linewidth(LEGEND_FRAME_LINEWIDTH)


def _overlay_ols_trend(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str = "#222222",
    linestyle: str = "-",
) -> None:
    if x.size < 3:
        return
    coef = np.polyfit(x, y, 1)
    x_line = np.linspace(float(x.min()), float(x.max()), 50)
    y_line = coef[0] * x_line + coef[1]
    ax.plot(x_line, y_line, color=color, linestyle=linestyle, linewidth=1.5, alpha=0.95, zorder=4)


def _panel_legend_for_caption(cohort: str) -> str:
    parts = [
        f"{_panel_id(cohort, i)} {RAW_FEATURE_LABELS[f]}"
        for i, f in enumerate(RAW_EDA_FEATURES)
    ]
    return "; ".join(parts)


def build_figure_caption(cohort: str) -> str:
    """Journal-style figure caption (Fig. 4 / Fig. 5); cohort named here only."""
    fig_num = 4 if cohort == COHORT_A_LABEL else 5
    cohort_name = "Cohort A (Liberman)" if cohort == COHORT_A_LABEL else "Cohort B (Brad)"
    panel_list = _panel_legend_for_caption(cohort)
    return (
        f"Fig. {fig_num} Correlations between raw auditory brainstem response (ABR) "
        f"features and synapse count per inner hair cell (IHC) in {cohort_name}. "
        f"Each panel shows the relationship between one raw, untransformed ABR feature "
        f"and synapse count (synapses per IHC) at the animal × frequency stratum, "
        f"using the ABR measurement at the SPL level nearest to {DEDUP_TARGET_DB:.0f} dB. "
        f"Panels are labeled {cohort}1–{cohort}{len(RAW_EDA_FEATURES)} and arranged as follows: "
        f"{panel_list}. "
        f"For all plots, symbols indicate individual animal × frequency strata and "
        f"color indicates true noise exposure group ({NOISE_CAPTION_COLOR_PHRASE}). "
        f"Spearman rank correlation coefficients (ρ) with approximate 95% confidence "
        f"intervals are shown at the top right of each panel (format: *value (lower, "
        f"upper); asterisk if p < {BONFERRONI_ALPHA}, Bonferroni-corrected for "
        f"{BONFERRONI_N_TESTS} feature comparisons per figure). "
        f"When significant, a single solid regression line is shown as a visual aid."
    )


def _plot_single_cohort_figure(
    df: pd.DataFrame,
    cohort: str,
    out_base: Path,
    *,
    figure_num: int,
    spearman_rows: list[dict[str, Any]],
) -> None:
    """One standalone 5×2 figure per cohort (Figure 4 or 5 only — separate file)."""
    n_row, n_col = 5, 2
    features = list(RAW_EDA_FEATURES)

    with plt.rc_context(PRINT_RC):
        fig = plt.figure(figsize=PRINT_FIGSIZE)
        outer = fig.add_gridspec(
            2, 1, height_ratios=[LEGEND_ROW_RATIO, 1.0], hspace=LEGEND_PANEL_HSPACE
        )
        _draw_framed_noise_legend(fig.add_subplot(outer[0]))
        panel_gs = outer[1].subgridspec(
            n_row, n_col, wspace=0.32, hspace=PRINT_PANEL_HSPACE_COHORT
        )
        axes = _build_row_shared_axes(fig, panel_gs, n_row, n_col)
        for fi, feat in enumerate(features):
            row, col = divmod(fi, n_col)
            _plot_raw_synapses_panel(
                axes[row, col],
                df,
                feat,
                panel_id=_panel_id(cohort, fi),
                cohort=cohort,
                figure_num=figure_num,
                spearman_rows=spearman_rows,
                show_ylabel=col == 0,
                show_yticklabels=col == 0,
                layout="cohort",
            )
        fig.subplots_adjust(
            left=PRINT_SUBPLOT_LEFT_COHORT + PRINT_LR_INSET,
            right=PRINT_SUBPLOT_RIGHT_COHORT - PRINT_LR_INSET,
            bottom=0.06,
            top=0.995,
        )
        _save_fig(fig, out_base, dpi=PRINT_DPI, pad_inches=PRINT_PAD_INCHES)


def build_combined_figure_caption() -> str:
    panel_a = _panel_legend_for_caption(COHORT_A_LABEL)
    panel_b = _panel_legend_for_caption(COHORT_B_LABEL)
    return (
        "Combined raw ABR feature vs. synapse count (synapses per IHC) at the "
        f"animal × frequency stratum ({DEDUP_TARGET_DB:.0f} dB nearest SPL). "
        "Five rows × four columns: columns 1–2, Cohort A (Liberman); columns 3–4, "
        "Cohort B (Brad). Within each cohort block, two features per row (left then right). "
        f"Panel IDs A1–A10 (Liberman): {panel_a}. "
        f"Panel IDs B1–B10 (Brad): {panel_b}. "
        "Symbols are individual strata; color indicates true noise group "
        f"({NOISE_CAPTION_COLOR_PHRASE}). "
        f"Pooled Spearman ρ with 95% CI at top right (* if p < {BONFERRONI_ALPHA}, "
        f"Bonferroni for {BONFERRONI_N_TESTS} features per cohort). "
        "Significant panels show one pooled OLS trend line."
    )


def _plot_combined_cohorts_figure(
    af_frames: Mapping[str, pd.DataFrame],
    out_base: Path,
) -> None:
    """5×4 print figure: cols 0–1 Liberman, cols 2–3 Brad; panels A1–A10 / B1–B10."""
    n_row, n_col = 5, 4
    features = list(RAW_EDA_FEATURES)
    with plt.rc_context(PRINT_RC):
        fig = plt.figure(figsize=PRINT_FIGSIZE)
        outer = fig.add_gridspec(
            2, 1, height_ratios=[LEGEND_ROW_RATIO, 1.0], hspace=LEGEND_PANEL_HSPACE
        )
        _draw_framed_noise_legend(fig.add_subplot(outer[0]))
        panel_gs = outer[1].subgridspec(
            n_row, n_col, wspace=0.30, hspace=PRINT_PANEL_HSPACE_COMBINED
        )
        axes = _build_row_shared_axes(fig, panel_gs, n_row, n_col)
        for row in range(n_row):
            for col in range(n_col):
                fi = row * 2 + (col % 2)
                cohort = COHORT_A_LABEL if col < 2 else COHORT_B_LABEL
                feat = features[fi]
                _plot_raw_synapses_panel(
                    axes[row, col],
                    af_frames[cohort],
                    feat,
                    panel_id=_panel_id(cohort, fi),
                    cohort=cohort,
                    figure_num=0,
                    spearman_rows=None,
                    show_ylabel=col == 0,
                    show_yticklabels=col == 0,
                    record_spearman=False,
                    layout="combined",
                )
        fig.subplots_adjust(
            left=PRINT_SUBPLOT_LEFT_COMBINED + PRINT_LR_INSET,
            right=PRINT_SUBPLOT_RIGHT_COMBINED - PRINT_LR_INSET,
            bottom=0.05,
            top=0.995,
        )
        _save_fig(fig, out_base, dpi=PRINT_DPI, pad_inches=PRINT_PAD_INCHES)


def plot_figure04_05_scatter(
    af_frames: Mapping[str, pd.DataFrame],
    out_dir: Path | str,
    cache_dir: Path | str = CACHE_EDA_DIR,
) -> tuple[Path, Path, Path, Path]:
    """Export Figure 4/5 (5×2 each) plus combined 5×4 cohort figure."""
    out_dir = Path(out_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    spearman_rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for cohort in (COHORT_A_LABEL, COHORT_B_LABEL):
        stem, figure_num = FIGURE_STEM_BY_COHORT[cohort]
        out_base = out_dir / stem
        _plot_single_cohort_figure(
            af_frames[cohort],
            cohort,
            out_base,
            figure_num=figure_num,
            spearman_rows=spearman_rows,
        )
        paths.append(out_base.with_suffix(".png"))
        caption_path = out_dir / f"figure_{figure_num:02d}_caption.txt"
        caption_path.write_text(build_figure_caption(cohort))

    combined_base = out_dir / FIGURE_COMBINED_STEM
    _plot_combined_cohorts_figure(af_frames, combined_base)
    combined_path = combined_base.with_suffix(".png")
    (out_dir / "figure_combined_caption.txt").write_text(build_combined_figure_caption())

    csv_path = cache_dir / "figure_04_05_spearman_animal_freq.csv"
    pd.DataFrame(spearman_rows).to_csv(csv_path, index=False)
    return paths[0], paths[1], combined_path, csv_path


def plot_scatter_grid(
    frames: Mapping[str, pd.DataFrame],
    out_base: Path,
    *,
    grain: str,
    features: Sequence[str] = RAW_EDA_FEATURES,
) -> None:
    """10 × 2 scatter: feature × cohort; color + marker by noise."""
    n_rows = len(features)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, max(8, n_rows * 2.4)),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.array([axes])

    cohort_order = [COHORT_A_LABEL, COHORT_B_LABEL]
    for col, cohort in enumerate(cohort_order):
        df = frames[cohort]
        for row, feat in enumerate(features):
            ax = axes[row, col]
            for noise_val in (0, 1):
                m = df["noise_cat"] == noise_val
                if not m.any():
                    continue
                ax.scatter(
                    df.loc[m, feat],
                    df.loc[m, "synapses"],
                    c=NOISE_LOWER_COLOR if noise_val == 0 else NOISE_HIGHER_COLOR,
                    marker=NOISE_MARKERS[noise_val],
                    s=10,
                    alpha=0.55,
                    linewidths=0,
                    label=NOISE_LABELS[noise_val],
                    rasterized=int(m.sum()) > 800,
                )
            rho, p, n = _spearman_one(df[feat], df["synapses"])
            ax.text(
                0.03,
                0.97,
                f"ρ={rho:.3f}\np={p:.3g}\nn={n}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75),
            )
            if col == 0:
                ax.set_ylabel("Synapses per IHC", fontsize=11)
            if row == 0:
                ax.set_title(COHORT_TITLES[cohort], fontsize=12)
            if row == n_rows - 1:
                ax.set_xlabel(RAW_FEATURE_LABELS.get(feat, feat), fontsize=10)

    grain_title = "all SPL levels" if grain == GRAIN_LONG else f"80 dB nearest (animal × frequency)"
    fig.suptitle(
        f"Raw feature vs. synapses ({grain_title})",
        fontsize=14,
        y=1.01,
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=NOISE_MARKERS[k],
            color="w",
            markerfacecolor=NOISE_LOWER_COLOR if k == 0 else NOISE_HIGHER_COLOR,
            markeredgecolor=NOISE_LOWER_COLOR if k == 0 else NOISE_HIGHER_COLOR,
            markersize=8,
            label=NOISE_LABELS[k],
        )
        for k in (0, 1)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=10)
    if grain == GRAIN_LONG:
        fig.text(0.5, 0.01, LONG_GRAIN_FOOTER, ha="center", fontsize=8, color="#444444")
    _save_fig(fig, out_base)


def plot_feature_distributions(
    frames: Mapping[str, pd.DataFrame],
    out_base: Path,
    *,
    features: Sequence[str] = RAW_EDA_FEATURES,
) -> None:
    """10 × 2 histogram/KDE overlays by noise (long grain)."""
    n_rows = len(features)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, max(8, n_rows * 2.2)),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.array([axes])

    for col, cohort in enumerate([COHORT_A_LABEL, COHORT_B_LABEL]):
        df = frames[cohort]
        for row, feat in enumerate(features):
            ax = axes[row, col]
            for noise_val in (0, 1):
                vals = df.loc[df["noise_cat"] == noise_val, feat].dropna()
                if vals.empty:
                    continue
                color = NOISE_LOWER_COLOR if noise_val == 0 else NOISE_HIGHER_COLOR
                ax.hist(
                    vals,
                    bins=30,
                    density=True,
                    alpha=0.45,
                    color=color,
                    label=NOISE_LABELS[noise_val],
                )
            if col == 0:
                ax.set_ylabel(RAW_FEATURE_LABELS.get(feat, feat), fontsize=11)
            if row == 0:
                ax.set_title(COHORT_TITLES[cohort], fontsize=12)
            if row == n_rows - 1:
                ax.set_xlabel("Raw value", fontsize=10)

    fig.suptitle("Raw feature distributions (long grain)", fontsize=14, y=1.01)
    fig.legend(loc="upper center", ncol=2, frameon=False, fontsize=10)
    _save_fig(fig, out_base)


def build_report_markdown(
    *,
    long_frames: Mapping[str, pd.DataFrame],
    af_frames: Mapping[str, pd.DataFrame],
    long_tbl: pd.DataFrame,
    af_tbl: pd.DataFrame,
    grain_cmp: pd.DataFrame,
    dedup_stats: Mapping[str, dict[str, int]],
    flags: dict[str, Any],
) -> str:
    lines = [
        "# ABR raw-feature univariate EDA report",
        "",
        "## 1. Feature identity",
        "- `distance` = peak-to-trough latency (trough time − peak time).",
        "- `slope` = Wave I slope (−amplitude / distance).",
        "- Excluded: `Slope_all`, `Slope_high4` (Buran amplitude–SPL slopes).",
        "",
        "## 2. Cross-cohort note",
        CROSS_COHORT_NOTE,
        "",
        "## 3. Grain comparison (|Δρ| ≥ {:.2f})".format(GRAIN_DELTA_RHO_WARN),
    ]
    warn = grain_cmp.loc[grain_cmp["abs_delta_rho"] >= GRAIN_DELTA_RHO_WARN]
    if warn.empty:
        lines.append("- No strata exceed the warning threshold.")
    else:
        for _, r in warn.head(15).iterrows():
            lines.append(
                f"- **{r['feature']}** ({r['cohort']}, {r['noise_group']}): "
                f"ρ_long={r['rho_long']:.3f}, ρ_af={r['rho_animal_freq']:.3f}, "
                f"Δρ={r['delta_rho']:.3f}"
            )
    lines.extend(["", "## 4. Strong associations (prefer animal_freq for SHAP prior)"])
    for grain_name, key in [(GRAIN_ANIMAL_FREQ, "strong_animal_freq"), (GRAIN_LONG, "strong_long")]:
        items = flags.get(key, [])
        lines.append(f"### {grain_name}")
        if not items:
            lines.append("- None flagged.")
        else:
            for it in items[:12]:
                lines.append(
                    f"- {it['feature']} ({it['cohort']}, {it['noise_group']}): "
                    f"ρ={it['rho']:.3f}, p={it['p_value']:.4g}, n={it['n']}"
                )
    lines.extend(
        [
            "",
            "## 5. Feature skewness (long grain)",
            "- Full table: `figures/cache/eda/feature_skewness_long.csv` (all features, all cohorts).",
            "- Log1p flags (|skew|>1, LONG_LOG only):",
        ]
    )
    for it in flags.get("skew", []):
        lines.append(f"- {it['feature']} ({it['cohort']}): skew={it['skew']:.2f}")
    if not flags.get("skew"):
        lines.append("- None flagged.")
    below = flags.get("skew_below_threshold")
    if below:
        lines.append("- Below threshold (|skew|≤1) among LONG_LOG:")
        for it in below:
            lines.append(
                f"- {it['feature']} ({it['cohort']}): skew={it['skew']:.3f}"
            )
    lines.extend(["", "## 6. Nonlinear / scaling candidates"])
    for it in flags.get("nonlinear", []):
        lines.append(
            f"- {it['feature']} ({it['cohort']}, {it['grain']}): "
            f"|ρ_s−ρ_p|={it['gap']:.3f}"
        )
    if not flags.get("nonlinear"):
        lines.append("- None flagged.")
    lines.extend(["", "## 7. Data notes"])
    for cohort in (COHORT_A_LABEL, COHORT_B_LABEL):
        lines.append(
            f"- Cohort {cohort}: n_long={len(long_frames[cohort])}, "
            f"n_animal_freq={len(af_frames[cohort])}, "
            f"dedup at_80={dedup_stats[cohort]['n_at_80']}, "
            f"fallback={dedup_stats[cohort]['n_nearest_fallback']}"
        )
    lines.append("- Includes train, validate, and test animals (exploratory).")
    lines.append("- Scatter grids are tall (EDA layout); not print-ready without resizing.")
    return "\n".join(lines) + "\n"


def run_univariate_eda(
    data: NNStage2Data | None = None,
    *,
    out_dir: Path | str = FIG_EDA_DIR,
    cache_dir: Path | str = CACHE_EDA_DIR,
    write_figures: bool = True,
) -> dict[str, Any]:
    """Run dual-grain EDA; write figures and cache artifacts."""
    apply_slide_rcparams()
    data = data or load_nn_stage2_data()
    verify_feature_identity()

    splits = splits_for_long_stage2(data)
    lib_long = splits["lib_long_all"]
    brad_long = splits["bb_long_all"]

    long_frames = {
        COHORT_A_LABEL: prepare_eda_frame(lib_long, COHORT_A_LABEL, GRAIN_LONG),
        COHORT_B_LABEL: prepare_eda_frame(brad_long, COHORT_B_LABEL, GRAIN_LONG),
    }
    dedup_stats: dict[str, dict[str, int]] = {}
    af_frames = {}
    for cohort, df in [(COHORT_A_LABEL, lib_long), (COHORT_B_LABEL, brad_long)]:
        st: dict[str, int] = {}
        af_frames[cohort] = prepare_eda_frame(
            df, cohort, GRAIN_ANIMAL_FREQ, dedup_stats_out=st
        )
        dedup_stats[cohort] = st

    out_dir = Path(out_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    long_tbl = spearman_table(long_frames, grain=GRAIN_LONG)
    af_tbl = spearman_table(af_frames, grain=GRAIN_ANIMAL_FREQ)
    grain_cmp = compare_grain_spearman(long_tbl, af_tbl)

    long_tbl.to_csv(cache_dir / "spearman_raw_features_long.csv", index=False)
    af_tbl.to_csv(cache_dir / "spearman_raw_features_animal_freq.csv", index=False)
    grain_cmp.to_csv(cache_dir / "spearman_grain_comparison.csv", index=False)

    skew_tbl = feature_skewness_table(long_frames, grain=GRAIN_LONG)
    skew_tbl.to_csv(cache_dir / "feature_skewness_long.csv", index=False)

    skew_flagged = flag_skewed_features(skew_tbl)
    skew_below_log = skew_tbl.loc[
        skew_tbl["in_LONG_LOG"] & skew_tbl["abs_skew"].le(SKEW_THRESHOLD)
    ]
    skew_below_records = skew_below_log.to_dict(orient="records")

    flags: dict[str, Any] = {
        "skew": skew_flagged,
        "skew_below_threshold": skew_below_records,
        "nonlinear": flag_nonlinear_candidates(long_frames, grain=GRAIN_LONG)
        + flag_nonlinear_candidates(af_frames, grain=GRAIN_ANIMAL_FREQ),
        "strong_long": flag_strong_associations(long_tbl),
        "strong_animal_freq": flag_strong_associations(af_tbl),
        "dedup_stats": dedup_stats,
        "n_long": {k: len(v) for k, v in long_frames.items()},
        "n_animal_freq": {k: len(v) for k, v in af_frames.items()},
    }
    flags_path = cache_dir / "abr_univariate_eda_flags.json"
    with flags_path.open("w") as f:
        json.dump(flags, f, indent=2, default=str)

    report = build_report_markdown(
        long_frames=long_frames,
        af_frames=af_frames,
        long_tbl=long_tbl,
        af_tbl=af_tbl,
        grain_cmp=grain_cmp,
        dedup_stats=dedup_stats,
        flags=flags,
    )
    report_path = cache_dir / "abr_univariate_eda_report.md"
    report_path.write_text(report)

    figure45_paths: tuple[Path, Path] | None = None
    figure_combined_path: Path | None = None
    figure45_csv: Path | None = None
    if write_figures:
        plot_scatter_grid(
            long_frames,
            out_dir / "abr_raw_vs_synapses_scatter_long",
            grain=GRAIN_LONG,
        )
        fig4_png, fig5_png, fig_combined, figure45_csv = plot_figure04_05_scatter(
            af_frames, out_dir, cache_dir=cache_dir
        )
        figure45_paths = (fig4_png, fig5_png)
        figure_combined_path = fig_combined
        plot_feature_distributions(
            long_frames,
            out_dir / "abr_raw_feature_distributions",
        )

    return {
        "long_frames": long_frames,
        "af_frames": af_frames,
        "long_tbl": long_tbl,
        "af_tbl": af_tbl,
        "grain_cmp": grain_cmp,
        "skew_tbl": skew_tbl,
        "flags": flags,
        "report_path": report_path,
        "flags_path": flags_path,
        "skew_path": cache_dir / "feature_skewness_long.csv",
        "figure_04_path": figure45_paths[0] if figure45_paths else None,
        "figure_05_path": figure45_paths[1] if figure45_paths else None,
        "figure_combined_path": figure_combined_path,
        "figure_04_05_spearman_path": figure45_csv,
    }

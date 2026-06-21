"""Act IV synthesis figures from 10-fold CV summary parquet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from utils.benchmark_metrics import (
    R2_Y_AXIS_MARGIN_HINT,
    RMSE_Y_AXIS_MARGIN_HINT,
    apply_slide_rcparams,
)

try:
    from brokenaxes import brokenaxes
except ImportError as exc:  # pragma: no cover
    brokenaxes = None  # type: ignore[misc, assignment]
    _BROKENAXES_IMPORT_ERROR = exc
else:
    _BROKENAXES_IMPORT_ERROR = None

ACT4_SYNTHESIS_MODELS: Tuple[str, ...] = (
    "LR baseline",
    "RF",
    "XGB",
    "MLP",
)
ACT4_SYNTHESIS_LABELS: Tuple[str, ...] = (
    "Linear Regression (Amp@80)",
    "Random Forest",
    "XGBoost",
    "Neural Network (MLP)",
)
ACT4_X_POSITIONS: Tuple[float, ...] = (0.0, 2.6, 5.2, 7.8)
ACT4_MODEL_X = dict(zip(ACT4_SYNTHESIS_MODELS, ACT4_X_POSITIONS))

# 2-panel cohort figure (models on x at 0..3)
MODEL_CENTERS: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)
MODEL_X_COHORT: Dict[str, float] = dict(
    zip(ACT4_SYNTHESIS_MODELS, MODEL_CENTERS)
)
DISPLAY_SCENARIO_ORDER: Tuple[int, ...] = (1, 2, 3)
DISPLAY_SCENARIO_LABELS: Dict[int, str] = {
    1: "Within-cohort training",
    2: "Cross-cohort training",
    3: "Combined training",
}
DISPLAY_SCENARIO_COLORS: Dict[int, str] = {
    1: "#F5C000",
    2: "#7B2D8B",
    3: "#2D8B45",
}
# Train slice A/B/C → display scenario (1 within, 2 cross, 3 combined).
WITHIN_COHORT_TRAIN: Dict[str, str] = {"Brad": "A", "Liberman": "C"}
CROSS_COHORT_TRAIN: Dict[str, str] = {"Brad": "C", "Liberman": "A"}
COMBINED_TRAIN = "B"
DISPLAY_SCENARIO_X_OFFSET: Dict[int, float] = {1: -0.12, 2: 0.0, 3: 0.12}
COHORT_PANEL_FNAME = "deck_act4_synthesis_cohorts_RMSE_cv.png"
COHORT_PANEL_3_FNAME = "deck_act4_synthesis_cohorts_RMSE_cv_3panel.png"
COHORT_PANEL_3_R2_FNAME = "supplementary_figure_S2_pooled_oof_r2.png"
COHORT_PANEL_FIGSIZE = (16.0, 10.0)
# Two-column print (~7.2 in); not scaled down from slide size.
COHORT_PANEL_3_FIGSIZE = (7.2, 3.4)
COHORT_PANEL_3_WSPACE = 0.55
COHORT_PANEL_3_LEGEND_ROW_RATIO = 0.15
COHORT_PANEL_3_DPI = 300
# Margin label (axes coords): left of y tick labels; more negative = more gap.
COHORT_PANEL_3_YLABEL_X = -0.38
COHORT_PANEL_3_SUBPLOT_LEFT = 0.17
COHORT_GRID_COLOR = "0.82"
COHORT_PANEL_WSPACE = 0.72
COHORT_YLABEL_X = -0.38
COHORT_PANEL_LETTER_FS = 30
COHORT_PANEL_LETTER_DX = -0.010
COHORT_PANEL_LETTER_DY = 0.018
COHORT_TICK_FONTSIZE = 22
COHORT_XLABEL_FONTSIZE = 22
COHORT_MARGIN_HINT_FONTSIZE = 22
COHORT_LEGEND_FONTSIZE = 22
COHORT_BREAK_DX = 0.008
COHORT_BREAK_DY = 0.022
COHORT_LEGEND_ROW_RATIO = 0.11
COHORT_LEGEND_PANEL_HSPACE = 0.10
# Fine-tune within the legend strip (axes coords): left of center, slightly high.
COHORT_LEGEND_ANCHOR_X = 0.46
COHORT_LEGEND_ANCHOR_Y = 1
X_LIM_COHORT = (-0.55, 3.55)
Y_TICK_STEP = 0.5
MIN_BREAK_GAP = 0.12
BASELINE_BAND_COLOR = "#6FA8DC"
BASELINE_LINE_COLOR = "#1c4587"


@dataclass(frozen=True)
class CohortPanelDrawStyle:
    """Typography and mark sizes for cohort RMSE panels (3-panel = print defaults)."""

    tick_fontsize: float
    xlabel_fontsize: float
    margin_fontsize: float
    letter_fontsize: float
    legend_fontsize: float
    legend_markersize: float
    errorbar_ms: float
    errorbar_elinewidth: float
    errorbar_capthick: float
    errorbar_capsize: float
    errorbar_mew: float
    axis_linewidth: float
    tick_width: float
    tick_length: float
    legend_frame_linewidth: float
    ceiling_line_lw: float
    ytick_pad: float


COHORT_PANEL_3_PRINT_STYLE = CohortPanelDrawStyle(
    tick_fontsize=9.0,
    xlabel_fontsize=8.0,
    margin_fontsize=8.0,
    letter_fontsize=11.0,
    legend_fontsize=8.0,
    legend_markersize=6.0,
    errorbar_ms=4.5,
    errorbar_elinewidth=0.9,
    errorbar_capthick=0.8,
    errorbar_capsize=2.5,
    errorbar_mew=0.5,
    axis_linewidth=0.5,
    tick_width=0.45,
    tick_length=2.5,
    legend_frame_linewidth=0.5,
    ceiling_line_lw=1.0,
    ytick_pad=3.0,
)


@dataclass(frozen=True)
class CohortPanelMetricConfig:
    value_col: str
    model_sem_col: str | None
    ceiling_sem_col: str
    ylabel_title: str
    ylabel_hint: str


RMSE_COHORT_METRIC = CohortPanelMetricConfig(
    value_col="RMSE",
    model_sem_col="RMSE_sem",
    ceiling_sem_col="RMSE_sem",
    ylabel_title="Prediction error",
    ylabel_hint=RMSE_Y_AXIS_MARGIN_HINT,
)
R2_COHORT_METRIC = CohortPanelMetricConfig(
    value_col="R2",
    model_sem_col=None,
    ceiling_sem_col="R2_baseline_sem",
    ylabel_title=r"Prediction accuracy ($R^2$)",
    ylabel_hint=R2_Y_AXIS_MARGIN_HINT,
)


def _act4_axes_grid(
    ax, x_positions: Tuple[float, ...], *, solid_vertical: bool = False
) -> None:
    ax.set_axisbelow(True)
    ax.grid(
        True,
        axis="y",
        alpha=0.38,
        linestyle="-",
        linewidth=0.75,
        color="0.82",
        zorder=0,
    )
    vline_ls = "-" if solid_vertical else (0, (1, 2))
    vline_lw = 0.75
    vline_color = COHORT_GRID_COLOR if solid_vertical else "0.70"
    vline_alpha = 0.38 if solid_vertical else 0.88
    for xv in x_positions:
        ax.axvline(
            xv,
            color=vline_color,
            linewidth=vline_lw,
            linestyle=vline_ls,
            alpha=vline_alpha,
            zorder=0,
        )


def _act4_draw_ceiling(
    ax,
    ceiling_rmse: float,
    ceiling_rmse_sem: float | None,
    *,
    annotate: bool = False,
    line_lw: float | None = None,
) -> None:
    band_color = BASELINE_BAND_COLOR
    line_color = BASELINE_LINE_COLOR
    short_baseline = "Condition-stratified baseline (mean ± SEM)"
    has_sem = ceiling_rmse_sem is not None and ceiling_rmse_sem > 0
    if has_sem:
        y_lo = float(ceiling_rmse - ceiling_rmse_sem)
        y_hi = float(ceiling_rmse + ceiling_rmse_sem)
        ax.axhspan(
            y_lo,
            y_hi,
            facecolor=band_color,
            edgecolor="none",
            alpha=0.35,
            zorder=0,
            clip_on=True,
        )
        ax.axhline(
            ceiling_rmse,
            color=line_color,
            ls="-",
            lw=line_lw if line_lw is not None else 1.8,
            zorder=1,
        )
    else:
        line_color = "#555555"
        ax.axhline(
            ceiling_rmse,
            color=line_color,
            ls="--",
            lw=line_lw if line_lw is not None else 1.6,
            zorder=1,
        )
    if not annotate:
        return
    x_right = float(ACT4_X_POSITIONS[-1]) + 0.42
    if has_sem:
        y_lo = float(ceiling_rmse - ceiling_rmse_sem)
        ax.annotate(
            short_baseline,
            xy=(x_right - 0.25, y_lo + 0.05),
            xytext=(0, 10),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=line_color,
            zorder=11,
            clip_on=False,
        )


def scatter_act4_synthesis_rmse_cv(ax, sub: pd.DataFrame) -> None:
    """Four model columns with vertical error bars (RMSE_sem)."""
    dot = "#E35F00"
    sub2 = sub.loc[sub["model"].isin(ACT4_SYNTHESIS_MODELS)].copy()
    if sub2.empty:
        ax.text(
            0.5,
            0.5,
            "No rows",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return
    for _, row in sub2.iterrows():
        m = row["model"]
        if m not in ACT4_MODEL_X:
            continue
        xi = ACT4_MODEL_X[m]
        z = 8 if row.get("source") == "nn" else 5
        yerr = float(row["RMSE_sem"]) if pd.notna(row.get("RMSE_sem")) else None
        ax.errorbar(
            xi,
            float(row["RMSE"]),
            yerr=yerr,
            fmt="o",
            ms=8 if z == 8 else 6,
            mfc=dot,
            mec="0.25",
            mew=0.75,
            ecolor="0.35",
            elinewidth=1.2,
            capsize=3,
            zorder=5,
        )


def _padded_limits(
    vals: Sequence[float],
    *,
    pad_frac: float = 0.05,
    pad_floor: float = 0.06,
) -> Tuple[float, float]:
    if not vals:
        return 0.0, 1.0
    lo, hi = float(min(vals)), float(max(vals))
    pad = max(pad_floor, pad_frac * (hi - lo))
    return lo - pad, hi + pad


def _row_y_extent(
    row: pd.Series, metric: CohortPanelMetricConfig = RMSE_COHORT_METRIC
) -> Tuple[float, float]:
    y = float(row[metric.value_col])
    sem = 0.0
    if metric.model_sem_col and pd.notna(row.get(metric.model_sem_col)):
        sem = float(row[metric.model_sem_col])
    return y - sem, y + sem


def _collect_y_values(
    rows: pd.DataFrame,
    *,
    include_ceiling: pd.Series | None = None,
    metric: CohortPanelMetricConfig = RMSE_COHORT_METRIC,
) -> List[float]:
    out: List[float] = []
    for _, row in rows.iterrows():
        lo, hi = _row_y_extent(row, metric)
        out.extend([lo, float(row[metric.value_col]), hi])
    if include_ceiling is not None and len(include_ceiling):
        ce = float(include_ceiling[metric.value_col].iloc[0])
        sem = (
            float(include_ceiling[metric.ceiling_sem_col].iloc[0])
            if pd.notna(include_ceiling[metric.ceiling_sem_col].iloc[0])
            else 0.0
        )
        out.extend([ce - sem, ce, ce + sem])
    return out


def _cohort_shared_ylim(
    summary: pd.DataFrame,
    *,
    test_sets: Tuple[str, ...] = ("Liberman", "Brad"),
    metric: CohortPanelMetricConfig = RMSE_COHORT_METRIC,
) -> Tuple[float, float]:
    """Matched linear y-limits (models + ceiling bands) for cohort panel(s)."""
    vals: List[float] = []
    for test_set in test_sets:
        df = _models_for_cohort(summary, test_set)
        ceil = summary.loc[
            summary["test_set"].eq(test_set) & summary["model"].eq("ceiling")
        ]
        vals.extend(_collect_y_values(df, include_ceiling=ceil, metric=metric))
    return _padded_limits(vals)


def _draw_linear_cohort_panel(
    ax,
    df: pd.DataFrame,
    *,
    test_set: str,
    ceiling_value: float,
    ceiling_sem: float | None,
    ylim: Tuple[float, float],
    style: CohortPanelDrawStyle,
    metric: CohortPanelMetricConfig = RMSE_COHORT_METRIC,
) -> None:
    _draw_scenarios_on_ax(
        ax, df, test_set=test_set, show_xlabels=True, style=style, metric=metric
    )
    _draw_baseline_band(
        ax,
        ceiling_value,
        ceiling_sem,
        line_lw=style.ceiling_line_lw,
    )
    ax.set_ylim(*ylim)


def _cohort_ceiling(
    summary: pd.DataFrame,
    test_set: str,
    metric: CohortPanelMetricConfig = RMSE_COHORT_METRIC,
) -> Tuple[float, float | None]:
    ceil = summary.loc[
        summary["model"].eq("ceiling") & summary["test_set"].eq(test_set)
    ]
    if ceil.empty:
        return float("nan"), None
    ce = float(ceil[metric.value_col].iloc[0])
    ce_sem = (
        float(ceil[metric.ceiling_sem_col].iloc[0])
        if pd.notna(ceil[metric.ceiling_sem_col].iloc[0])
        else None
    )
    return ce, ce_sem


def _models_for_cohort(summary: pd.DataFrame, test_set: str) -> pd.DataFrame:
    if test_set == "Pooled":
        scenario_values = ("1", "2", "3")
    else:
        scenario_values = ("A", "B", "C")
    return summary.loc[
        summary["test_set"].eq(test_set)
        & summary["scenario"].isin(scenario_values)
        & summary["model"].isin(ACT4_SYNTHESIS_MODELS)
    ].copy()


def _display_scenario_index(test_set: str, train_scenario: str) -> int:
    """Map train slice A/B/C to display scenarios 1/2/3 (within / cross / combined)."""
    if test_set == "Pooled":
        return int(train_scenario)
    if train_scenario == COMBINED_TRAIN:
        return 3
    if train_scenario == WITHIN_COHORT_TRAIN[test_set]:
        return 1
    if train_scenario == CROSS_COHORT_TRAIN[test_set]:
        return 2
    raise ValueError(
        f"Unknown train scenario {train_scenario!r} for test_set {test_set!r}"
    )


def _brad_y_split(df_models: pd.DataFrame) -> float:
    lows = df_models.loc[df_models["RMSE"] < 5.0, "RMSE"]
    highs = df_models.loc[df_models["RMSE"] > 5.0, "RMSE"]
    if lows.empty or highs.empty:
        return 4.85
    return 0.5 * (float(lows.max()) + float(highs.min()))


def _brad_segment_ylims(
    summary: pd.DataFrame,
    *,
    pad_frac: float = 0.05,
    pad_floor: float = 0.06,
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    df = _models_for_cohort(summary, "Brad")
    split = _brad_y_split(df)
    ceil_row = summary.loc[
        summary["test_set"].eq("Brad") & summary["model"].eq("ceiling")
    ]
    lower_rows = df.loc[df["RMSE"] <= split]
    upper_rows = df.loc[df["RMSE"] > split]

    lower_vals = _collect_y_values(lower_rows, include_ceiling=ceil_row)
    upper_vals = _collect_y_values(upper_rows)

    if not upper_vals:
        raise ValueError(
            "Brad upper segment has no points; cannot build broken y-axis."
        )

    lower_lim = _padded_limits(
        lower_vals, pad_frac=pad_frac, pad_floor=pad_floor
    )
    upper_lim = _padded_limits(
        upper_vals, pad_frac=pad_frac, pad_floor=pad_floor
    )

    if upper_lim[0] < lower_lim[1] + MIN_BREAK_GAP:
        upper_lim = (lower_lim[1] + MIN_BREAK_GAP, upper_lim[1])

    return lower_lim, upper_lim, split


def _shared_cohort_broken_ylims(
    summary: pd.DataFrame,
    *,
    pad_frac: float = 0.05,
    pad_floor: float = 0.06,
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Broken-axis limits shared across Liberman, Brad, and Pooled panels."""
    parts = [
        _models_for_cohort(summary, ts)
        for ts in ("Liberman", "Brad", "Pooled")
    ]
    df_all = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if df_all.empty:
        raise ValueError("No cohort model rows for shared broken y-limits.")
    split = _brad_y_split(df_all)

    lower_vals: List[float] = []
    upper_vals: List[float] = []
    for test_set in ("Liberman", "Brad", "Pooled"):
        df = _models_for_cohort(summary, test_set)
        ceil_row = summary.loc[
            summary["test_set"].eq(test_set) & summary["model"].eq("ceiling")
        ]
        lower_rows = df.loc[df["RMSE"] <= split]
        upper_rows = df.loc[df["RMSE"] > split]
        ce = float(ceil_row["RMSE"].iloc[0]) if len(ceil_row) else float("nan")
        lower_inc = ceil_row if len(ceil_row) and ce <= split else None
        upper_inc = ceil_row if len(ceil_row) and ce > split else None
        lower_vals.extend(
            _collect_y_values(lower_rows, include_ceiling=lower_inc)
        )
        upper_vals.extend(
            _collect_y_values(upper_rows, include_ceiling=upper_inc)
        )

    if not upper_vals:
        flat = _cohort_shared_ylim(summary)
        pooled = _models_for_cohort(summary, "Pooled")
        if not pooled.empty:
            flat = _padded_limits(
                _collect_y_values(pooled)
                + _collect_y_values(_models_for_cohort(summary, "Liberman"))
                + _collect_y_values(_models_for_cohort(summary, "Brad")),
                pad_frac=pad_frac,
                pad_floor=pad_floor,
            )
        mid = 0.5 * (flat[0] + flat[1])
        return (flat[0], mid), (mid + MIN_BREAK_GAP, flat[1]), 4.85

    lower_lim = _padded_limits(
        lower_vals, pad_frac=pad_frac, pad_floor=pad_floor
    )
    upper_lim = _padded_limits(
        upper_vals, pad_frac=pad_frac, pad_floor=pad_floor
    )
    if upper_lim[0] < lower_lim[1] + MIN_BREAK_GAP:
        upper_lim = (lower_lim[1] + MIN_BREAK_GAP, upper_lim[1])
    return lower_lim, upper_lim, split


def _axis_matching_ylim(
    axs: Sequence,
    target: Tuple[float, float],
    *,
    atol: float = 1e-4,
) -> plt.Axes:
    for ax in axs:
        lo, hi = ax.get_ylim()
        if abs(lo - target[0]) <= atol and abs(hi - target[1]) <= atol:
            return ax
    raise RuntimeError(f"No axes match ylim {target}")


def _draw_baseline_band(
    ax,
    ceiling_rmse: float,
    ceiling_rmse_sem: float | None,
    *,
    line_lw: float | None = None,
) -> None:
    _act4_draw_ceiling(
        ax, ceiling_rmse, ceiling_rmse_sem, annotate=False, line_lw=line_lw
    )


def _errorbar_scenario_row(
    ax,
    row: pd.Series,
    *,
    test_set: str,
    style: CohortPanelDrawStyle | None = None,
    metric: CohortPanelMetricConfig = RMSE_COHORT_METRIC,
) -> None:
    model = str(row["model"])
    train_scen = str(row["scenario"])
    if model not in MODEL_X_COHORT:
        return
    disp = _display_scenario_index(test_set, train_scen)
    xi = MODEL_X_COHORT[model] + DISPLAY_SCENARIO_X_OFFSET[disp]
    color = DISPLAY_SCENARIO_COLORS[disp]
    yerr: float | None = None
    if metric.model_sem_col and pd.notna(row.get(metric.model_sem_col)):
        yerr = float(row[metric.model_sem_col])
    ms = style.errorbar_ms if style else 10.0
    elw = style.errorbar_elinewidth if style else 1.2
    cthick = style.errorbar_capthick if style else 1.4
    csize = style.errorbar_capsize if style else 4.0
    mew = style.errorbar_mew if style else 0.75
    ax.errorbar(
        xi,
        float(row[metric.value_col]),
        yerr=yerr,
        fmt="o",
        ms=ms,
        mfc=color,
        mec="0.25",
        mew=mew,
        ecolor=color,
        elinewidth=elw,
        capsize=csize,
        capthick=cthick,
        zorder=5,
    )


def _style_model_xaxis(
    ax, *, show_xlabels: bool = True, style: CohortPanelDrawStyle | None = None
) -> None:
    x_fs = style.xlabel_fontsize if style else COHORT_XLABEL_FONTSIZE
    ax.set_xticks(list(MODEL_CENTERS))
    if show_xlabels:
        ax.set_xticklabels(
            list(ACT4_SYNTHESIS_LABELS),
            rotation=90,
            ha="center",
            va="top",
            fontsize=x_fs,
        )
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_xlim(*X_LIM_COHORT)
    _act4_axes_grid(ax, MODEL_CENTERS, solid_vertical=style is not None)
    ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEP))
    if style is not None:
        for spine in ax.spines.values():
            spine.set_linewidth(style.axis_linewidth)
        ax.tick_params(
            axis="both",
            width=style.tick_width,
            length=style.tick_length,
        )
    sns.despine(ax=ax, top=True, right=True)


def _yaxis_spine_lw() -> float:
    return float(plt.rcParams.get("axes.linewidth", 1.0))


def _set_cohort_metric_ylabel(
    ax,
    metric: CohortPanelMetricConfig,
    *,
    label_x: float = COHORT_YLABEL_X,
    margin_fontsize: float | None = None,
) -> None:
    margin_lbl = f"{metric.ylabel_title}\n{metric.ylabel_hint}"
    ax.text(
        label_x,
        0.5,
        margin_lbl,
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=margin_fontsize or COHORT_MARGIN_HINT_FONTSIZE,
        linespacing=1.85,
        clip_on=False,
    )


def _set_cohort_prediction_error_ylabel(
    ax,
    *,
    label_x: float = COHORT_YLABEL_X,
    margin_fontsize: float | None = None,
) -> None:
    margin_lbl = "Prediction error\n" + RMSE_Y_AXIS_MARGIN_HINT
    ax.text(
        label_x,
        0.5,
        margin_lbl,
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=margin_fontsize or COHORT_MARGIN_HINT_FONTSIZE,
        linespacing=1.85,
        clip_on=False,
    )


def _draw_panel_letter(
    fig, ax, letter: str, *, letter_fontsize: float | None = None
) -> None:
    pos = ax.get_position()
    fig.text(
        pos.x0 + COHORT_PANEL_LETTER_DX,
        pos.y1 + COHORT_PANEL_LETTER_DY,
        letter,
        fontsize=letter_fontsize or COHORT_PANEL_LETTER_FS,
        fontweight="bold",
        va="bottom",
        ha="right",
        clip_on=False,
    )


def _draw_scenarios_on_ax(
    ax,
    df: pd.DataFrame,
    *,
    test_set: str,
    show_xlabels: bool = True,
    style: CohortPanelDrawStyle | None = None,
    metric: CohortPanelMetricConfig = RMSE_COHORT_METRIC,
) -> None:
    for _, row in df.iterrows():
        _errorbar_scenario_row(ax, row, test_set=test_set, style=style, metric=metric)
    _style_model_xaxis(ax, show_xlabels=show_xlabels, style=style)
    tick_fs = style.tick_fontsize if style else COHORT_TICK_FONTSIZE
    if style is not None:
        y_pad = style.ytick_pad
    else:
        y_pad = 4
    ax.tick_params(axis="y", labelsize=tick_fs, pad=y_pad)


def _legend_handles(*, legend_markersize: float = 13.0) -> List:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=DISPLAY_SCENARIO_COLORS[s],
            markeredgecolor="0.25",
            markersize=legend_markersize,
            label=DISPLAY_SCENARIO_LABELS[s],
        )
        for s in DISPLAY_SCENARIO_ORDER
    ]


def _clear_brokenaxes_diags(bax) -> None:
    for handle in list(bax.diag_handles):
        handle.remove()
    bax.diag_handles.clear()


def _draw_broken_y_break_marks(fig: plt.Figure, bax) -> None:
    """One slash per gap edge on the left y-axis (match left-spine weight)."""
    _clear_brokenaxes_diags(bax)
    spine = bax.axs[0].spines["left"]
    lw = float(spine.get_linewidth() or _yaxis_spine_lw())
    color = spine.get_edgecolor()
    dx, dy = COHORT_BREAK_DX, COHORT_BREAK_DY

    def _slash_at(ax, *, at_top: bool) -> None:
        pos = ax.get_position()
        x = pos.x0
        y = pos.y1 if at_top else pos.y0
        fig.add_artist(
            Line2D(
                [x - dx, x + dx],
                [y - dy, y + dy],
                transform=fig.transFigure,
                color=color,
                lw=lw,
                solid_capstyle="butt",
                clip_on=False,
                zorder=100,
            )
        )

    for ax in bax.axs:
        spec = ax.get_subplotspec()
        if not spec.is_last_row():
            _slash_at(ax, at_top=False)
        if not spec.is_first_row():
            _slash_at(ax, at_top=True)

    for ax in bax.axs:
        ax.spines["top"].set_visible(False)
        if not ax.get_subplotspec().is_first_row():
            ax.spines["top"].set_linewidth(0.0)


def _style_broken_cohort_segment_axes(
    lower_ax: plt.Axes,
    upper_ax: plt.Axes,
    *,
    lower_lim: Tuple[float, float],
    show_xlabels: bool,
) -> None:
    for ax in (lower_ax, upper_ax):
        ax.set_xlim(*X_LIM_COHORT)
        ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEP))
        ax.tick_params(axis="y", labelsize=COHORT_TICK_FONTSIZE)
        ax.set_axisbelow(True)
        ax.grid(
            True,
            axis="y",
            alpha=0.38,
            linestyle="-",
            linewidth=0.75,
            color=COHORT_GRID_COLOR,
            zorder=0,
        )
        for xv in MODEL_CENTERS:
            ax.axvline(
                xv,
                color="0.70",
                linewidth=0.9,
                linestyle=(0, (1, 2)),
                alpha=0.88,
                zorder=0,
            )
        sns.despine(ax=ax, top=True, right=True)

    _style_model_xaxis(lower_ax, show_xlabels=show_xlabels)
    upper_ax.set_xticks(list(MODEL_CENTERS))
    upper_ax.set_xticklabels([])
    upper_ax.tick_params(axis="x", labelbottom=False)

    upper_ax.spines["bottom"].set_visible(False)
    upper_ax.spines["bottom"].set_linewidth(0.0)
    lower_ax.spines["top"].set_visible(False)
    lower_ax.spines["top"].set_linewidth(0.0)
    lower_yticks = [t for t in lower_ax.get_yticks() if t < lower_lim[1] - 1e-6]
    if lower_yticks:
        lower_ax.set_yticks(lower_yticks)


def _draw_broken_cohort_panel(
    fig: plt.Figure,
    subplot_spec,
    *,
    df: pd.DataFrame,
    test_set: str,
    lower_lim: Tuple[float, float],
    upper_lim: Tuple[float, float],
    y_split: float,
    ceiling_rmse: float,
    ceiling_sem: float | None,
    draw_break_marks: bool = True,
):
    """Broken y-axis cohort panel; ``y_split`` / limits follow Brad (panel B) definition."""
    bax = brokenaxes(
        ylims=(lower_lim, upper_lim),
        hspace=0.14,
        subplot_spec=subplot_spec,
        d=0.0,
    )
    _clear_brokenaxes_diags(bax)
    lower_ax = _axis_matching_ylim(bax.axs, lower_lim)
    upper_ax = _axis_matching_ylim(bax.axs, upper_lim)

    for _, row in df.iterrows():
        target = upper_ax if float(row["RMSE"]) > y_split else lower_ax
        _errorbar_scenario_row(target, row, test_set=test_set)

    ce_ax = upper_ax if ceiling_rmse > y_split else lower_ax
    _draw_baseline_band(ce_ax, ceiling_rmse, ceiling_sem)

    _style_broken_cohort_segment_axes(
        lower_ax, upper_ax, lower_lim=lower_lim, show_xlabels=True
    )
    for ax in (lower_ax, upper_ax):
        ax.tick_params(axis="y", left=True, labelleft=True)

    if draw_break_marks:
        _draw_broken_y_break_marks(fig, bax)
    return bax


def _assert_summary_10_fold(summary: pd.DataFrame) -> None:
    if "n_folds" not in summary.columns:
        return
    nf = summary["n_folds"].dropna().unique()
    if len(nf) and not np.all(nf == 10):
        raise ValueError(
            f"synthesis_cohort_panels_cv expects 10-fold summary; got n_folds={nf.tolist()}"
        )


def synthesis_act4_grid_cv_fname(n_folds: int) -> str:
    """Presentation filename for Act IV CV grid by fold count."""
    if n_folds == 10:
        return "deck_act4_synthesis_grid_RMSE_cv.png"
    return f"deck_act4_synthesis_grid_RMSE_cv_{n_folds}fold.png"


def synthesis_act4_grid_cv(
    summary: pd.DataFrame,
    *,
    out_dir: Path | None = None,
    fname: str | None = None,
    n_folds: int | None = None,
) -> Path:
    """2×3 grid from a synthesis CV summary parquet."""
    if fname is None:
        nf = int(
            n_folds
            or (
                summary["n_folds"].max() if "n_folds" in summary.columns else 10
            )
        )
        fname = synthesis_act4_grid_cv_fname(nf)
    if out_dir is None:
        out_dir = (
            Path(__file__).resolve().parent.parent / "figures" / "presentation"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ("A", "B", "C")
    row_tests = ("Brad", "Liberman")
    ceiling_df = summary.loc[summary["model"].eq("ceiling")]

    fig, axes = plt.subplots(
        2, 3, figsize=(14.5, 10), sharey="row", squeeze=False
    )

    for ri, ts in enumerate(row_tests):
        ceil_row = ceiling_df.loc[ceiling_df["test_set"].eq(ts)]
        ce = float(ceil_row["RMSE"].iloc[0]) if len(ceil_row) else float("nan")
        ce_sem = (
            float(ceil_row["RMSE_sem"].iloc[0])
            if len(ceil_row) and pd.notna(ceil_row["RMSE_sem"].iloc[0])
            else None
        )
        for ci, scen in enumerate(scenarios):
            ax = axes[ri, ci]
            sub = summary.loc[
                summary["test_set"].eq(ts)
                & summary["scenario"].eq(scen)
                & summary["model"].isin(ACT4_SYNTHESIS_MODELS)
            ]
            scatter_act4_synthesis_rmse_cv(ax, sub)
            ax.set_xlim(-0.75, float(ACT4_X_POSITIONS[-1]) + 0.55)
            _act4_draw_ceiling(ax, ce, ce_sem, annotate=False)
            _act4_axes_grid(ax, ACT4_X_POSITIONS)
            if ri == 0:
                ax.set_title(f"Training scenario {scen}", fontsize=13, pad=10)
                ax.tick_params(axis="x", labelbottom=False)
            if ci == 0:
                ax.set_ylabel("RMSE (synapses)")
                ax.text(
                    -0.44,
                    0.5,
                    f"{ts} test",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=14,
                    clip_on=False,
                )
            sns.despine(ax=ax)

    fig.tight_layout()
    png_path = out_dir / fname
    svg_path = out_dir / fname.replace(".png", ".svg")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def synthesis_cohort_panels_cv(
    summary: pd.DataFrame,
    *,
    out_dir: Path | None = None,
    fname: str | None = None,
) -> Path:
    """1×2 cohort panels: Liberman (A) | Brad (B); shared broken y (limits from Brad)."""
    if brokenaxes is None:
        raise ImportError(
            "brokenaxes is required for synthesis_cohort_panels_cv. "
            "Install with: pip install 'brokenaxes>=0.6.0'"
        ) from _BROKENAXES_IMPORT_ERROR

    _assert_summary_10_fold(summary)

    if out_dir is None:
        out_dir = (
            Path(__file__).resolve().parent.parent / "figures" / "presentation"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = fname or COHORT_PANEL_FNAME

    apply_slide_rcparams()

    lib_df = _models_for_cohort(summary, "Liberman")
    brad_df = _models_for_cohort(summary, "Brad")
    lib_ce, lib_ce_sem = _cohort_ceiling(summary, "Liberman")
    brad_ce, brad_ce_sem = _cohort_ceiling(summary, "Brad")

    lower_lim, upper_lim, brad_split = _brad_segment_ylims(summary)

    fig = plt.figure(figsize=COHORT_PANEL_FIGSIZE)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[COHORT_LEGEND_ROW_RATIO, 1.0],
        width_ratios=[1, 1],
        wspace=COHORT_PANEL_WSPACE,
        hspace=COHORT_LEGEND_PANEL_HSPACE,
    )
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.set_axis_off()
    ax_leg.legend(
        handles=_legend_handles(),
        ncol=3,
        loc="center",
        bbox_to_anchor=(COHORT_LEGEND_ANCHOR_X, COHORT_LEGEND_ANCHOR_Y),
        frameon=True,
        fancybox=False,
        edgecolor=COHORT_GRID_COLOR,
        facecolor="white",
        framealpha=1.0,
        fontsize=COHORT_LEGEND_FONTSIZE,
    )

    lib_bax = _draw_broken_cohort_panel(
        fig,
        gs[1, 0],
        df=lib_df,
        test_set="Liberman",
        lower_lim=lower_lim,
        upper_lim=upper_lim,
        y_split=brad_split,
        ceiling_rmse=lib_ce,
        ceiling_sem=lib_ce_sem,
        draw_break_marks=False,
    )
    brad_bax = _draw_broken_cohort_panel(
        fig,
        gs[1, 1],
        df=brad_df,
        test_set="Brad",
        lower_lim=lower_lim,
        upper_lim=upper_lim,
        y_split=brad_split,
        ceiling_rmse=brad_ce,
        ceiling_sem=brad_ce_sem,
        draw_break_marks=False,
    )

    fig.subplots_adjust(bottom=0.18, left=0.28, right=0.98, top=0.97)

    _draw_broken_y_break_marks(fig, lib_bax)
    _draw_broken_y_break_marks(fig, brad_bax)

    _set_cohort_prediction_error_ylabel(lib_bax.big_ax)
    _set_cohort_prediction_error_ylabel(brad_bax.big_ax)
    _draw_panel_letter(fig, lib_bax.big_ax, "A")
    _draw_panel_letter(fig, brad_bax.big_ax, "B")

    png_path = out_dir / fname
    svg_path = out_dir / fname.replace(".png", ".svg")
    save_kw = dict(dpi=150, bbox_inches="tight", pad_inches=0.5)
    fig.savefig(png_path, **save_kw)
    fig.savefig(svg_path, **save_kw)
    plt.close(fig)
    return png_path


def _synthesis_cohort_panels_3cv_impl(
    summary: pd.DataFrame,
    *,
    metric: CohortPanelMetricConfig,
    out_dir: Path | None = None,
    fname: str,
) -> Path:
    """1×3 cohort panels: Liberman | Brad | Pooled; shared linear y."""
    if metric is RMSE_COHORT_METRIC:
        pooled = summary.loc[summary["test_set"].eq("Pooled")]
        if pooled.empty:
            raise ValueError(
                "synthesis_cohort_panels_3cv requires Pooled rows in summary "
                "(run compute_pooled_synthesis_folds + summarize_pooled_folds)."
            )

    _assert_summary_10_fold(summary)

    if out_dir is None:
        out_dir = (
            Path(__file__).resolve().parent.parent / "figures" / "presentation"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    style = COHORT_PANEL_3_PRINT_STYLE

    lib_df = _models_for_cohort(summary, "Liberman")
    brad_df = _models_for_cohort(summary, "Brad")
    pool_df = _models_for_cohort(summary, "Pooled")
    lib_ce, lib_ce_sem = _cohort_ceiling(summary, "Liberman", metric)
    brad_ce, brad_ce_sem = _cohort_ceiling(summary, "Brad", metric)
    pool_ce, pool_ce_sem = _cohort_ceiling(summary, "Pooled", metric)

    ylim = _cohort_shared_ylim(
        summary, test_sets=("Liberman", "Brad", "Pooled"), metric=metric
    )

    fig = plt.figure(figsize=COHORT_PANEL_3_FIGSIZE)
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[COHORT_PANEL_3_LEGEND_ROW_RATIO, 1.0],
        width_ratios=[1, 1, 1],
        wspace=COHORT_PANEL_3_WSPACE,
        hspace=COHORT_LEGEND_PANEL_HSPACE,
    )
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.set_axis_off()
    leg = ax_leg.legend(
        handles=_legend_handles(legend_markersize=style.legend_markersize),
        ncol=3,
        loc="center",
        bbox_to_anchor=(COHORT_LEGEND_ANCHOR_X, COHORT_LEGEND_ANCHOR_Y),
        frameon=True,
        fancybox=False,
        edgecolor=COHORT_GRID_COLOR,
        facecolor="white",
        framealpha=1.0,
        fontsize=style.legend_fontsize,
    )
    leg.get_frame().set_linewidth(style.legend_frame_linewidth)

    ax_lib = fig.add_subplot(gs[1, 0])
    ax_brad = fig.add_subplot(gs[1, 1])
    ax_pool = fig.add_subplot(gs[1, 2])

    _draw_linear_cohort_panel(
        ax_lib,
        lib_df,
        test_set="Liberman",
        ceiling_value=lib_ce,
        ceiling_sem=lib_ce_sem,
        ylim=ylim,
        style=style,
        metric=metric,
    )
    _draw_linear_cohort_panel(
        ax_brad,
        brad_df,
        test_set="Brad",
        ceiling_value=brad_ce,
        ceiling_sem=brad_ce_sem,
        ylim=ylim,
        style=style,
        metric=metric,
    )
    _draw_linear_cohort_panel(
        ax_pool,
        pool_df,
        test_set="Pooled",
        ceiling_value=pool_ce,
        ceiling_sem=pool_ce_sem,
        ylim=ylim,
        style=style,
        metric=metric,
    )

    fig.subplots_adjust(
        bottom=0.20,
        left=COHORT_PANEL_3_SUBPLOT_LEFT,
        right=0.99,
        top=0.96,
    )

    for ax in (ax_lib, ax_brad, ax_pool):
        _set_cohort_metric_ylabel(
            ax,
            metric,
            label_x=COHORT_PANEL_3_YLABEL_X,
            margin_fontsize=style.margin_fontsize,
        )
    _draw_panel_letter(fig, ax_lib, "A", letter_fontsize=style.letter_fontsize)
    _draw_panel_letter(fig, ax_brad, "B", letter_fontsize=style.letter_fontsize)
    _draw_panel_letter(fig, ax_pool, "C", letter_fontsize=style.letter_fontsize)

    png_path = out_dir / fname
    svg_path = out_dir / fname.replace(".png", ".svg")
    save_kw = dict(dpi=COHORT_PANEL_3_DPI, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, **save_kw)
    fig.savefig(svg_path, **save_kw)
    plt.close(fig)
    return png_path


def synthesis_cohort_panels_3cv(
    summary: pd.DataFrame,
    *,
    out_dir: Path | None = None,
    fname: str | None = None,
) -> Path:
    """1×3 cohort panels (two-column print size): Liberman | Brad | Pooled; shared linear y."""
    return _synthesis_cohort_panels_3cv_impl(
        summary,
        metric=RMSE_COHORT_METRIC,
        out_dir=out_dir,
        fname=fname or COHORT_PANEL_3_FNAME,
    )


def synthesis_cohort_panels_3cv_r2(
    summary: pd.DataFrame,
    *,
    out_dir: Path | None = None,
    fname: str | None = None,
) -> Path:
    """Supplementary Figure S2: 3-panel pooled OOF R² companion to Figure 3."""
    return _synthesis_cohort_panels_3cv_impl(
        summary,
        metric=R2_COHORT_METRIC,
        out_dir=out_dir,
        fname=fname or COHORT_PANEL_3_R2_FNAME,
    )

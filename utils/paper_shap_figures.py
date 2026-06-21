"""
Print-ready paper figures 6–8: Stage-1 RF SHAP and Stage-2 MLP SHAP.

Outputs under ``figures/eda/`` at 7 in print width (matches Figures 4–5).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.abr_univariate_eda import (
    FIG_EDA_DIR,
    LEGEND_FRAME_COLOR,
    LEGEND_FRAME_LINEWIDTH,
    NOISE_CAPTION_COLOR_PHRASE,
    NOISE_HIGHER_COLOR,
    NOISE_LOWER_COLOR,
    PRINT_DPI,
    PRINT_FIG_WIDTH_IN,
    PRINT_PAD_INCHES,
    PRINT_RC,
    RAW_EDA_FEATURES,
    SCATTER_ALPHA,
    _noise_legend_handles,
)
from utils.nn_stage2_data import STRAIN_BINARY_COL
from utils.stage1_rf_shap import (
    STAGE1_FEATURE_TYPE_ORDER,
    STAGE1_SHAP_CACHE_DIR,
    COHORT_SLUGS_DECK,
    COHORT_TO_LAB,
    FEATURE_PAPER_LABELS,
    collapse_stage1_shap_by_type,
    load_stage1_cohort_outputs_from_cache,
)
from utils.stage2_mlp_shap import (
    COHORT_SLUGS_DECK,
    FEATURE_PAPER_LABELS as MLP_FEATURE_PAPER_LABELS,
    MLP_SHAP_CACHE_DIR,
    aggregate_mlp_export_animal_freq,
    load_mlp_cohort_from_export_parquets,
    mean_abs_shap_series,
    mean_abs_shap_table,
)

PAPER_FIG_WIDTH_IN = PRINT_FIG_WIDTH_IN  # 7.0

PAPER_PANEL_TOP = 0.80
PAPER_PANEL_BOTTOM = 0.12
PAPER_PANEL_LEFT = 0.28
PAPER_PANEL_RIGHT = 0.98
# Full figure-width strip (add_axes); sits above panel letters (A/B at ~y1+0.02).
PAPER_LEGEND_AX_BBOX = (0.0, 0.905, 1.0, 0.085)

FIGURE_06_STEM = "figure_06_stage1_shap_noise_colored"
FIGURE_07_STEM = "figure_07_stage2_mlp_shap_importance"
FIGURE_08_STEM = "figure_08_stage2_mlp_shap_abr_noise_colored"

FIGURE_06_HEIGHT_IN = 4.5
FIGURE_07_HEIGHT_IN = 5.0
FIGURE_08_HEIGHT_IN = 4.5

# Fixed y-order (top → bottom when reading): ABR block (Fig. 4 sequence), then context inputs.
MLP_PAPER_FEATURE_ORDER: tuple[str, ...] = (
    *RAW_EDA_FEATURES,
    STRAIN_BINARY_COL,
    "level",
    "frequency",
    "noise_preds",
)

# Paper-specific labels (``level`` is stimulus level in dB SPL, not a feature named SPL).
PAPER_MLP_LABELS: dict[str, str] = {
    **MLP_FEATURE_PAPER_LABELS,
    "level": "Stimulus level (dB SPL)",
}

CAPTION_06 = (
    "Fig. 6 SHAP summary for the Stage-1 random forest noise classifier. "
    "Each panel shows SHAP values for ten aggregated ABR feature types "
    "(mean across SPL levels within type) at the animal × frequency stratum; "
    "frequency is omitted. Dots are colored by true noise exposure group "
    f"({NOISE_CAPTION_COLOR_PHRASE}). The model was trained on the "
    "Train split; SHAP values are computed on Validate and Test wide rows. "
    "SHAP values are on the probability scale for class-1 (noise-exposed). "
    "Features follow the same order as Fig. 4 (amplitude at top, reading downward). "
    "Panel A, Cohort A (Liberman); panel B, Cohort B (Brad)."
)

CAPTION_07 = (
    "Fig. 7 Mean absolute SHAP values for the Stage-2 multilayer perceptron "
    "synapse predictor at the animal × frequency grain (SPL levels averaged). "
    "Features follow a fixed top-to-bottom order: the ten ABR morphology features "
    "in the same sequence as Figs. 4 and 6, then strain (when present), stimulus "
    "level (dB SPL), frequency, and predicted noise. "
    "SHAP values are in synapses per inner hair cell (IHC) units. "
    "Models were fit with 10-fold subject-wise out-of-fold cross-validation "
    "within each cohort (Liberman scenario C; Brad scenario A). "
    "Panel A, Cohort A (Liberman); panel B, Cohort B (Brad)."
)

CAPTION_08 = (
    "Fig. 8 SHAP summary for the Stage-2 MLP synapse model restricted to the ten "
    "ABR morphology inputs (noise prediction, frequency, stimulus level, and strain excluded). "
    "Each panel shows out-of-fold SHAP values at the animal × frequency grain "
    f"(SPL levels averaged), with dots colored by true noise exposure group "
    f"({NOISE_CAPTION_COLOR_PHRASE}). SHAP values are in synapses per IHC. "
    "Feature order matches Figs. 4 and 6. Panel A, Cohort A (Liberman); "
    "panel B, Cohort B (Brad)."
)


def _paper_panel_letter(fig: plt.Figure, ax: plt.Axes, letter: str) -> None:
    pos = ax.get_position()
    fig.text(
        pos.x0 - 0.010,
        pos.y1 + 0.018,
        letter,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="right",
        clip_on=False,
    )


def _save_paper_figure(
    fig: plt.Figure,
    stem: str,
    *,
    out_dir: Path | str = FIG_EDA_DIR,
    tight: bool = True,
) -> Path:
    base = Path(out_dir) / stem
    base.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        kw = dict(dpi=PRINT_DPI, bbox_inches="tight", pad_inches=PRINT_PAD_INCHES)
    else:
        kw = dict(dpi=PRINT_DPI, bbox_inches=None, pad_inches=0, facecolor="white")
    fig.savefig(base.with_suffix(".png"), **kw)
    fig.savefig(base.with_suffix(".svg"), **kw)
    plt.close(fig)
    return base


def _write_caption(fig_num: int, text: str, *, out_dir: Path | str = FIG_EDA_DIR) -> Path:
    path = Path(out_dir) / f"figure_{fig_num:02d}_caption.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def _add_paper_fullwidth_legend(fig: plt.Figure) -> None:
    """Framed noise legend in a full-width axes (true 7 in center, not panel inset)."""
    leg_ax = fig.add_axes(PAPER_LEGEND_AX_BBOX, frameon=False, zorder=20)
    leg_ax.set_axis_off()
    leg = leg_ax.legend(
        handles=_noise_legend_handles(markersize=6.0),
        ncol=2,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        bbox_transform=leg_ax.transAxes,
        borderaxespad=0.0,
        frameon=True,
        fancybox=False,
        edgecolor=LEGEND_FRAME_COLOR,
        facecolor="white",
        framealpha=1.0,
        fontsize=9,
    )
    leg.get_frame().set_linewidth(LEGEND_FRAME_LINEWIDTH)


def _apply_paper_panel_margins(fig: plt.Figure) -> None:
    fig.subplots_adjust(
        left=PAPER_PANEL_LEFT,
        right=PAPER_PANEL_RIGHT,
        top=PAPER_PANEL_TOP,
        bottom=PAPER_PANEL_BOTTOM,
    )


def _y_pos_top_first(n: int, rank: int) -> int:
    """Map rank 0 (first feature in list) to the top row."""
    return n - 1 - rank


def _finalize_paper_1x2_figure(
    fig: plt.Figure,
    axes: Sequence[plt.Axes],
    *,
    left_yticklabels: Sequence[str] | None = None,
    left_yticks: Sequence[float] | None = None,
    y_tick_labelsize: float = 8,
) -> None:
    """Apply panel margins, full-width legend, then panel letters."""
    _apply_paper_panel_margins(fig)
    _add_paper_fullwidth_legend(fig)
    if left_yticklabels is not None and left_yticks is not None:
        axes[0].set_yticks(list(left_yticks))
        axes[0].set_yticklabels(list(left_yticklabels))
        axes[0].tick_params(
            axis="y",
            labelsize=y_tick_labelsize,
            length=0,
            labelleft=True,
            left=True,
        )
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    for letter, ax in zip(("A", "B"), axes, strict=True):
        _paper_panel_letter(fig, ax, letter)


def _true_noise_point_colors_paper(meta: pd.DataFrame) -> np.ndarray:
    exposed = meta["noise_cat"].astype(int).eq(1).values
    colors = np.empty(len(meta), dtype=object)
    colors[exposed] = NOISE_HIGHER_COLOR
    colors[~exposed] = NOISE_LOWER_COLOR
    return colors


def _beeswarm_dot_y_offsets(shaps: np.ndarray, *, row_height: float = 0.4) -> np.ndarray:
    nbins = 100
    span = np.max(shaps) - np.min(shaps) + 1e-8
    quant = np.round(nbins * (shaps - np.min(shaps)) / span)
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


def _draw_beeswarm_panel(
    ax: plt.Axes,
    values: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    feature_order: Sequence[str],
    label_map: Mapping[str, str],
    row_height: float = 0.35,
    show_yticklabels: bool = True,
    dot_alpha: float = SCATTER_ALPHA,
) -> None:
    cols = [c for c in feature_order if c in values.columns]
    shap_vals = values[cols].values
    feature_names = [label_map.get(c, c) for c in cols]
    dot_colors = _true_noise_point_colors_paper(meta)

    ax.axvline(x=0, color="#999999", zorder=-1, lw=0.8)
    n_feat = len(cols)
    for rank, col_i in enumerate(range(n_feat)):
        pos = _y_pos_top_first(n_feat, rank)
        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = shap_vals[:, col_i]
        inds = np.arange(len(shaps))
        rng = np.random.default_rng(42 + col_i)
        rng.shuffle(inds)
        shaps = shaps[inds]
        colors = dot_colors[inds]
        ys = _beeswarm_dot_y_offsets(shaps, row_height=row_height)
        ax.scatter(
            shaps,
            pos + ys,
            c=colors,
            s=12,
            alpha=dot_alpha,
            linewidth=0,
            zorder=3,
            rasterized=len(shaps) > 500,
        )

    y_pos = [_y_pos_top_first(n_feat, rank) for rank in range(n_feat)]
    ax.set_yticks(y_pos)
    if show_yticklabels:
        ax.set_yticklabels(feature_names)
        ax.tick_params(axis="y", labelsize=8, length=0, width=0.5)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylim(-1, n_feat)
    ax.set_xlabel("SHAP value")
    for spine in ("right", "top", "left"):
        ax.spines[spine].set_visible(False)
    if show_yticklabels:
        ax.tick_params(axis="y", which="both", length=0)
    ax.xaxis.set_ticks_position("bottom")


def _stage1_paper_feature_order(collapsed: Mapping[str, pd.DataFrame]) -> list[str]:
    """Fixed morphology order (Fig. 4 panel sequence), not importance-ranked."""
    present = set(collapsed[COHORT_SLUGS_DECK[0]].columns)
    return [c for c in STAGE1_FEATURE_TYPE_ORDER if c in present]


def _mlp_paper_feature_order(shap_df: pd.DataFrame) -> list[str]:
    """Fixed Stage-2 feature order: context inputs then ABR block (Fig. 4 sequence)."""
    present = set(shap_df.columns)
    return [c for c in MLP_PAPER_FEATURE_ORDER if c in present]


def _mlp_abr_feature_order(shap_df: pd.DataFrame) -> list[str]:
    present = set(shap_df.columns)
    return [c for c in RAW_EDA_FEATURES if c in present]


def export_figure_06_stage1(
    *,
    stage1_cache_dir: Path | str | None = None,
    out_dir: Path | str = FIG_EDA_DIR,
) -> Path:
    """Figure 6: Stage-1 noise-colored SHAP beeswarm (1×2 cohorts)."""
    cache = Path(stage1_cache_dir or STAGE1_SHAP_CACHE_DIR)
    feature_table_path = cache / "feature_names_stage1.parquet"
    if not feature_table_path.is_file():
        raise FileNotFoundError(
            f"Missing {feature_table_path}; run abr_stage1_rf_shap.ipynb first."
        )
    feature_table = pd.read_parquet(feature_table_path)
    cohort_outputs = load_stage1_cohort_outputs_from_cache(cache)

    collapsed: dict[str, pd.DataFrame] = {}
    for slug in COHORT_SLUGS_DECK:
        collapsed[slug] = collapse_stage1_shap_by_type(
            cohort_outputs[slug]["shap"],
            feature_table,
            include_frequency=False,
        )

    feature_order = _stage1_paper_feature_order(collapsed)
    n_feat = len(feature_order)
    y_pos = [_y_pos_top_first(n_feat, rank) for rank in range(n_feat)]
    y_labels = [FEATURE_PAPER_LABELS.get(c, c) for c in feature_order]

    with plt.rc_context(PRINT_RC):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(PAPER_FIG_WIDTH_IN, FIGURE_06_HEIGHT_IN),
        )
        fig.subplots_adjust(wspace=0.28)

        for col, slug in enumerate(COHORT_SLUGS_DECK):
            _draw_beeswarm_panel(
                axes[col],
                collapsed[slug],
                cohort_outputs[slug]["meta"],
                feature_order=feature_order,
                label_map=FEATURE_PAPER_LABELS,
                show_yticklabels=(col == 0),
            )
            axes[col].set_ylim(-1, n_feat)

        _finalize_paper_1x2_figure(
            fig,
            axes,
            left_yticks=y_pos,
            left_yticklabels=y_labels,
        )

    base = _save_paper_figure(fig, FIGURE_06_STEM, out_dir=out_dir, tight=False)
    _write_caption(6, CAPTION_06, out_dir=out_dir)
    return base


def _load_mlp_cohort_animal_freq(
    cache_dir: Path | str,
    cohort_slug: str,
) -> dict[str, pd.DataFrame]:
    payload = load_mlp_cohort_from_export_parquets(cache_dir, cohort_slug)
    shap_af, meta_af = aggregate_mlp_export_animal_freq(payload["shap"], payload["meta"])
    return {"shap": shap_af, "meta": meta_af}


def _pooled_mlp_feature_order(
    cohort_payloads: Mapping[str, Mapping[str, pd.DataFrame]],
) -> list[str]:
    shap_df = next(iter(cohort_payloads.values()))["shap"]
    return _mlp_paper_feature_order(shap_df)


def export_figure_07_mlp_importance(
    *,
    mlp_cache_dir: Path | str | None = None,
    out_dir: Path | str = FIG_EDA_DIR,
) -> Path:
    """Figure 7: Stage-2 MLP mean |SHAP| horizontal bars (1×2 cohorts)."""
    cache = Path(mlp_cache_dir or MLP_SHAP_CACHE_DIR)
    cohort_payloads = {
        slug: _load_mlp_cohort_animal_freq(cache, slug) for slug in COHORT_SLUGS_DECK
    }
    feature_order = _pooled_mlp_feature_order(cohort_payloads)

    tables = {slug: mean_abs_shap_table(cohort_payloads[slug]["shap"]) for slug in COHORT_SLUGS_DECK}
    xmax = max(
        tables[slug].set_index("feature")["mean_abs_shap"].reindex(feature_order).max()
        for slug in COHORT_SLUGS_DECK
    )
    xmax *= 1.05

    with plt.rc_context(PRINT_RC):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(PAPER_FIG_WIDTH_IN, FIGURE_07_HEIGHT_IN),
        )
        fig.subplots_adjust(wspace=0.35, left=0.28, right=0.98)

        panels = ("A", "B")
        n_feat = len(feature_order)
        y_pos = np.array([_y_pos_top_first(n_feat, rank) for rank in range(n_feat)])
        labels = [PAPER_MLP_LABELS.get(f, f) for f in feature_order]

        for col, slug in enumerate(COHORT_SLUGS_DECK):
            ax = axes[col]
            tbl = tables[slug].set_index("feature")
            vals = tbl["mean_abs_shap"].reindex(feature_order).fillna(0.0).values
            ax.barh(y_pos, vals, color="#4a4a4a", height=0.7, edgecolor="none")
            ax.set_xlim(0, xmax)
            ax.set_ylim(-0.5, n_feat - 0.5)
            ax.set_xlabel("Mean |SHAP|")
            ax.tick_params(axis="x", labelsize=9)
            for spine in ("right", "top"):
                ax.spines[spine].set_visible(False)
            _paper_panel_letter(fig, ax, panels[col])

        axes[0].set_yticks(y_pos, labels)
        axes[0].tick_params(axis="y", labelsize=7)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels([])
        axes[1].tick_params(axis="y", left=False, labelleft=False)

    base = _save_paper_figure(fig, FIGURE_07_STEM, out_dir=out_dir)
    _write_caption(7, CAPTION_07, out_dir=out_dir)
    return base


def export_figure_08_mlp_abr_noise_colored(
    *,
    mlp_cache_dir: Path | str | None = None,
    out_dir: Path | str = FIG_EDA_DIR,
) -> Path:
    """Figure 8: Stage-2 MLP ABR-only SHAP beeswarm, noise-colored (1×2)."""
    cache = Path(mlp_cache_dir or MLP_SHAP_CACHE_DIR)
    cohort_payloads = {
        slug: _load_mlp_cohort_animal_freq(cache, slug) for slug in COHORT_SLUGS_DECK
    }
    feature_order = _mlp_abr_feature_order(cohort_payloads[COHORT_SLUGS_DECK[0]]["shap"])
    n_feat = len(feature_order)
    y_pos = [_y_pos_top_first(n_feat, rank) for rank in range(n_feat)]
    y_labels = [PAPER_MLP_LABELS.get(c, c) for c in feature_order]

    with plt.rc_context(PRINT_RC):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(PAPER_FIG_WIDTH_IN, FIGURE_08_HEIGHT_IN),
        )
        fig.subplots_adjust(wspace=0.28)

        for col, slug in enumerate(COHORT_SLUGS_DECK):
            _draw_beeswarm_panel(
                axes[col],
                cohort_payloads[slug]["shap"],
                cohort_payloads[slug]["meta"],
                feature_order=feature_order,
                label_map=PAPER_MLP_LABELS,
                show_yticklabels=(col == 0),
            )
            axes[col].set_ylim(-1, n_feat)

        _finalize_paper_1x2_figure(
            fig,
            axes,
            left_yticks=y_pos,
            left_yticklabels=y_labels,
        )

    base = _save_paper_figure(fig, FIGURE_08_STEM, out_dir=out_dir, tight=False)
    _write_caption(8, CAPTION_08, out_dir=out_dir)
    return base


def export_all_paper_shap_figures(
    *,
    stage1_cache_dir: Path | str | None = None,
    mlp_cache_dir: Path | str | None = None,
    out_dir: Path | str = FIG_EDA_DIR,
) -> dict[str, Path]:
    """Export Figures 6–8 and caption files."""
    paths = {
        "figure_06": export_figure_06_stage1(stage1_cache_dir=stage1_cache_dir, out_dir=out_dir),
        "figure_07": export_figure_07_mlp_importance(mlp_cache_dir=mlp_cache_dir, out_dir=out_dir),
        "figure_08": export_figure_08_mlp_abr_noise_colored(
            mlp_cache_dir=mlp_cache_dir, out_dir=out_dir
        ),
    }
    return paths


def paper_shap_digest(
    *,
    stage1_cache_dir: Path | str | None = None,
    mlp_cache_dir: Path | str | None = None,
) -> str:
    """Short text summary for CLI (row counts, top features)."""
    lines: list[str] = []
    s1 = load_stage1_cohort_outputs_from_cache(stage1_cache_dir)
    for slug in COHORT_SLUGS_DECK:
        n = len(s1[slug]["meta"])
        lines.append(f"Stage-1 {COHORT_TO_LAB[slug]} ({slug}): {n} rows")

    cache = Path(mlp_cache_dir or MLP_SHAP_CACHE_DIR)
    ft_table = pd.read_parquet(cache / "feature_names_mlp.parquet")
    lines.append(f"MLP features: {len(ft_table)}")

    for slug in COHORT_SLUGS_DECK:
        payload = _load_mlp_cohort_animal_freq(cache, slug)
        top = mean_abs_shap_series(payload["shap"]).sort_values(ascending=False).head(3)
        top_str = ", ".join(f"{k}={v:.4f}" for k, v in top.items())
        lines.append(
            f"Stage-2 {COHORT_TO_LAB[slug]} animal×freq: {len(payload['meta'])} rows; top |SHAP|: {top_str}"
        )
    return "\n".join(lines)

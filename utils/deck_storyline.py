"""Deck-only figure helpers for ``presentation_deck.ipynb`` (Act I + Ib + Stage 1).

Parquet exports: ``liberman_synapses_group_baseline.ipynb`` (Liberman ribbons),
``abr_wide_long_comparison.ipynb`` (Liberman OLS baselines), Brad ribbons built here.
Stage 1 eval: ``abr_nn_stage2.ipynb``.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from utils.benchmark_metrics import (
    BRAD_SYNAPSE_MEAN_STD_BY_GROUP_FREQ_PARQUET,
    LIBERMAN_SYNAPSE_MEAN_STD_BY_GROUP_FREQ_PARQUET,
    NN_ALL_LONG_FACE,
    O1_OLS_LIBERMAN_TWO_MARKS_PARQUET,
    O2_OLS_LIBERMAN_THREE_NOISE_PARQUET,
    O3_OLS_LIBERMAN_FOUR_STACK_PARQUET,
    RMSE_Y_AXIS_MARGIN_HINT,
    STAGE1_WIDE_LR_RF_EVAL_PARQUET,
    apply_slide_rcparams,
    repo_root,
)
from utils.abr_univariate_eda import NOISE_LABELS

# Act I ribbons + Figure 2 ROC palette
_NOISE_LOWER_COLOR = "#0173B2"
_NOISE_HIGHER_COLOR = "#DE8F05"
_RIBBON_LEGEND_LABELS = (NOISE_LABELS[0], NOISE_LABELS[1])
_DECK_AXIS_LABEL_FS = 13
_DECK_TICK_FS = 11
_DECK_LEGEND_FS = 13
_DECK_LINE_LW = 2.0
_RIBBON_GROUP_DODGE_SPREAD = (
    0.18  # data units between two noise groups at one frequency
)
_RIBBON_MARKER_SIZE = 8.5
# Brad Hz→kHz rounding can land on 5.7 / 45.3; align to test-grid kHz labels.
_BRAD_FREQ_KHZ_ALIAS: dict[float, float] = {5.7: 5.6, 45.3: 45.2}
_COHORT_RIBBONS_DPI = 450
_LIBERMAN_STRAIN_RIBBON_PANELS = (
    ("C57BL/6J", 1),  # panel A
    ("CBA/CaJ", 0),  # panel B
)

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - notebook-only

    def display(fig) -> None:  # type: ignore[no-redef]
        plt.close(fig)


def _presentation_out_dir() -> Path:
    p = repo_root() / "figures" / "presentation"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _deck_footer(fig: plt.Figure, msg: str) -> None:
    if not msg:
        return
    fig.text(0.5, 0.02, msg, ha="center", fontsize=9)


def _deck_panel_letter(fig: plt.Figure, ax: plt.Axes, letter: str) -> None:
    """Panel ID (A/B) above subplot — matches Act IV synthesis cohort panels."""
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


def _save_deck_png_svg(fig: plt.Figure, stem: str, *, dpi: int = 300) -> None:
    """PNG + SVG under ``figures/presentation/`` (no PDF)."""
    out = _presentation_out_dir()
    png = out / f"{stem}.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(png.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.12)
    display(fig)
    plt.close(fig)
    svg = png.with_suffix(".svg")
    print("Saved", png.resolve(), "+", svg.resolve())


def _ols_lollipop_figure(
    df: pd.DataFrame,
    *,
    stem: str,
    digest_title: str,
    parquet_path: Path,
) -> None:
    """OLS RMSE marks: **RMSE on y**, categories on **x**; rotated RMSE margin label."""
    apply_slide_rcparams()
    n = len(df)
    fig_w = max(4.8, 2.2 + 0.95 * n)
    fig_h = max(5.8, 4.2 + 0.55 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x = np.arange(n, dtype=float)
    rmse = df["rmse"].astype(float).to_numpy()
    ax.scatter(
        x,
        rmse,
        s=121,
        c=NN_ALL_LONG_FACE,
        edgecolors="0.25",
        linewidths=0.8,
        zorder=3,
    )
    rmin = float(np.nanmin(rmse)) if len(rmse) else 0.0
    rmax = float(np.nanmax(rmse)) if len(rmse) else 1.0
    span = rmax - rmin
    pad = 0.1 * span if span > 1e-9 else 0.05 * max(abs(rmax), 0.1)
    ax.set_ylim(max(0.0, rmin - pad), rmax + pad)
    ax.set_xlim(-0.55, max(n - 1, 0) + 0.55)
    x_lbls = [
        re.sub("exposure", "Noise", str(lb), flags=re.IGNORECASE)
        for lb in df["strip_label"].tolist()
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(x_lbls, fontsize=10, rotation=22, ha="right")
    ax.set_xlabel("")
    ax.set_axisbelow(True)
    ax.grid(
        True, axis="y", alpha=0.38, linestyle="-", linewidth=0.75, color="0.82"
    )
    sns.despine(ax=ax, top=True, right=True)
    margin_lbl = "RMSE\n" + RMSE_Y_AXIS_MARGIN_HINT
    ax.text(
        -0.22,
        0.5,
        margin_lbl,
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
        linespacing=1.85,
        clip_on=False,
    )
    fig.subplots_adjust(left=0.20, right=0.97, top=0.90, bottom=0.30)
    print(digest_title)
    print("parquet:", parquet_path.resolve())
    print(df.to_string(index=False))
    print("=== end OLS digest ===")
    _save_deck_png_svg(fig, stem)


def _synapse_mean_std_by_group_frequency(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD synapses per ``noise_cat`` × ``frequency`` (``Group`` = ``noise_cat``)."""
    return (
        wide_df.groupby(["noise_cat", "frequency"], dropna=False)["synapses"]
        .agg(mean_synapses="mean", std_synapses="std", n_animal_freq="count")
        .reset_index()
        .rename(columns={"noise_cat": "Group"})
        .sort_values(["Group", "frequency"])
        .reset_index(drop=True)
    )


def _canonicalize_ribbon_frequencies(gf: pd.DataFrame) -> pd.DataFrame:
    """Snap near-miss test frequencies (e.g. Brad 45.3 → 45.2 kHz) and merge duplicates."""
    if gf.empty:
        return gf
    out = gf.copy()
    freq = out["frequency"].astype(float)
    for src, dst in _BRAD_FREQ_KHZ_ALIAS.items():
        freq = freq.mask(freq == src, dst)
    out["frequency"] = freq
    if not out.duplicated(subset=["Group", "frequency"]).any():
        return out.sort_values(["Group", "frequency"]).reset_index(drop=True)

    merged: list[dict] = []
    for (grp, f_hz), part in out.groupby(["Group", "frequency"], dropna=False):
        n = part["n_animal_freq"].astype(float)
        mu = part["mean_synapses"].astype(float)
        sd = part["std_synapses"].fillna(0.0).astype(float)
        n_tot = float(n.sum())
        if n_tot <= 0:
            continue
        mu_p = float((n * mu).sum() / n_tot)
        if n_tot <= 1:
            sd_p = float(sd.iloc[0]) if len(sd) else float("nan")
        else:
            ss = float(((n - 1) * sd**2 + n * (mu - mu_p) ** 2).sum())
            sd_p = float(np.sqrt(max(ss / (n_tot - 1.0), 0.0)))
        merged.append(
            {
                "Group": grp,
                "frequency": float(f_hz),
                "mean_synapses": mu_p,
                "std_synapses": sd_p,
                "n_animal_freq": int(round(n_tot)),
            }
        )
    return (
        pd.DataFrame(merged)
        .sort_values(["Group", "frequency"])
        .reset_index(drop=True)
    )


def _frequency_khz_tick_label(freq: float) -> str:
    f = float(freq)
    for src, dst in _BRAD_FREQ_KHZ_ALIAS.items():
        if abs(f - src) < 1e-6:
            f = dst
            break
    if abs(f - round(f)) < 1e-6:
        return str(int(round(f)))
    return f"{f:.1f}"


def _noise_group_offsets(groups: list) -> dict:
    n_groups = len(groups)
    if n_groups <= 1:
        spread = 0.0
    elif n_groups == 2:
        spread = _RIBBON_GROUP_DODGE_SPREAD
    else:
        spread = min(0.92 / n_groups, 0.52)
    offs = (
        np.linspace(-spread / 2, spread / 2, n_groups)
        if n_groups > 1
        else np.array([0.0])
    )
    return {g: offs[gi] for gi, g in enumerate(groups)}


def _noise_group_colors(groups: list) -> dict:
    out: dict = {}
    for gi, gname in enumerate(groups):
        gs = str(gname).strip()
        if gs in ("0", "0.0"):
            out[gname] = _NOISE_LOWER_COLOR
        elif gs in ("1", "1.0"):
            out[gname] = _NOISE_HIGHER_COLOR
        else:
            out[gname] = plt.cm.tab10(gi % 10)
    return out


def _ribbon_ylim(
    gf: pd.DataFrame, *, pad_frac: float = 0.06
) -> tuple[float, float]:
    lows: list[float] = []
    highs: list[float] = []
    for _, r in gf.iterrows():
        m = float(r["mean_synapses"])
        s = float(r["std_synapses"]) if pd.notna(r["std_synapses"]) else 0.0
        lows.append(m - s)
        highs.append(m + s)
    if not lows:
        return 0.0, 1.0
    lo, hi = min(lows), max(highs)
    span = hi - lo
    pad = pad_frac * span if span > 1e-9 else 0.5
    return lo - pad, hi + pad


def ensure_brad_synapse_mean_std_by_group_frequency() -> pd.DataFrame:
    """Build/cache Brad (Cohort B) stratum ribbons table from wide train+test."""
    p = BRAD_SYNAPSE_MEAN_STD_BY_GROUP_FREQ_PARQUET
    if p.is_file():
        gf = pd.read_parquet(p)
    else:
        from utils.nn_stage2_data import (
            load_nn_stage2_data,
            splits_for_long_stage2,
        )

        data = load_nn_stage2_data()
        sp = splits_for_long_stage2(data)
        bb = pd.concat(
            [sp["bb_wide_train"], sp["bb_wide_test"]], ignore_index=True
        )
        if "noise_cat" not in bb.columns:
            msg = "Brad wide table missing noise_cat — check data_loader / tx mapping."
            raise KeyError(msg)
        bb = bb.copy()
        bb["frequency"] = (
            bb["frequency"].astype(float).replace(_BRAD_FREQ_KHZ_ALIAS)
        )
        gf = _synapse_mean_std_by_group_frequency(bb)
    gf = _canonicalize_ribbon_frequencies(gf)
    p.parent.mkdir(parents=True, exist_ok=True)
    gf.to_parquet(p, index=False)
    return gf


def _plot_synapse_ribbons_panel(
    ax: plt.Axes,
    gf: pd.DataFrame,
    *,
    g_to_color: dict,
    g_to_off: dict,
) -> None:
    freq_order = sorted(gf["frequency"].unique())
    f_to_i = {f: i for i, f in enumerate(freq_order)}
    groups = sorted(gf["Group"].astype(str).unique(), key=lambda x: float(x))
    for gname in groups:
        color = g_to_color[gname]
        sub = gf.loc[gf["Group"].astype(str).eq(str(gname))]
        for _, r in sub.iterrows():
            bi = float(f_to_i[r["frequency"]])
            x = bi + g_to_off[gname]
            m = float(r["mean_synapses"])
            s = float(r["std_synapses"]) if pd.notna(r["std_synapses"]) else 0.0
            ax.errorbar(
                [x],
                [m],
                yerr=[s],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=_DECK_LINE_LW,
                capsize=4.0,
                markersize=_RIBBON_MARKER_SIZE,
                alpha=0.92,
                zorder=3,
            )
    n_f = len(freq_order)
    ax.set_xticks(np.arange(n_f))
    ax.set_xticklabels(
        [_frequency_khz_tick_label(float(f)) for f in freq_order],
        rotation=22,
        ha="right",
        fontsize=_DECK_TICK_FS,
    )
    ax.margins(x=0.02)
    ax.set_axisbelow(True)
    ax.grid(True, which="both", alpha=0.35)
    sns.despine(ax=ax)


def _liberman_synapse_ribbons_for_strain(
    wide: pd.DataFrame, strain_binary: int
) -> pd.DataFrame:
    sub = wide.loc[wide["strain_binary"].eq(strain_binary)]
    return _canonicalize_ribbon_frequencies(
        _synapse_mean_std_by_group_frequency(sub)
    )


def _act1_cohort_ribbons_ylim() -> tuple[float, float] | None:
    """Y limits for Act I cohort + Liberman-strain ribbon figures (Liberman | Brad)."""
    lib_path = LIBERMAN_SYNAPSE_MEAN_STD_BY_GROUP_FREQ_PARQUET
    if not lib_path.is_file():
        return None
    gf_lib = pd.read_parquet(lib_path)
    gf_brad = ensure_brad_synapse_mean_std_by_group_frequency()
    return _ribbon_ylim(pd.concat([gf_lib, gf_brad], ignore_index=True))


def deck_act1_synapse_ribbons_cohorts() -> None:
    """Act I — 1×2 synapse ribbons (Cohort A Liberman | Cohort B Brad), by noise group."""
    lib_path = LIBERMAN_SYNAPSE_MEAN_STD_BY_GROUP_FREQ_PARQUET
    if not lib_path.is_file():
        print(
            "Skip Act I cohort ribbons — missing",
            lib_path.resolve(),
            "(run liberman_synapses_group_baseline.ipynb export).",
        )
        return

    apply_slide_rcparams()
    gf_lib = pd.read_parquet(lib_path)
    gf_brad = ensure_brad_synapse_mean_std_by_group_frequency()

    groups = sorted(
        gf_lib["Group"].astype(str).unique(), key=lambda x: float(x)
    )
    g_to_color = _noise_group_colors(groups)
    g_to_off = _noise_group_offsets(groups)

    y0, y1 = _ribbon_ylim(pd.concat([gf_lib, gf_brad], ignore_index=True))

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.16, 1.0],
        width_ratios=[1, 1],
        wspace=0.12,
        hspace=0.10,
    )
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.set_axis_off()
    ax_leg.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=_NOISE_LOWER_COLOR,
                marker="o",
                linestyle="None",
                markersize=8,
                label=_RIBBON_LEGEND_LABELS[0],
            ),
            Line2D(
                [0],
                [0],
                color=_NOISE_HIGHER_COLOR,
                marker="o",
                linestyle="None",
                markersize=8,
                label=_RIBBON_LEGEND_LABELS[1],
            ),
        ],
        ncol=2,
        loc="center",
        bbox_to_anchor=(0.5, 0.85),
        frameon=True,
        fancybox=False,
        edgecolor="0.82",
        facecolor="white",
        framealpha=1.0,
        fontsize=_DECK_LEGEND_FS,
    )

    ax_a = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1], sharey=ax_a)
    _plot_synapse_ribbons_panel(
        ax_a, gf_lib, g_to_color=g_to_color, g_to_off=g_to_off
    )
    _plot_synapse_ribbons_panel(
        ax_b, gf_brad, g_to_color=g_to_color, g_to_off=g_to_off
    )

    for ax in (ax_a, ax_b):
        ax.set_ylim(y0, y1)

    ax_a.set_ylabel(
        "Synapses / Inner Hair Cell (IHC)", fontsize=_DECK_AXIS_LABEL_FS
    )
    ax_a.tick_params(axis="both", labelsize=_DECK_TICK_FS)
    ax_b.tick_params(axis="x", labelsize=_DECK_TICK_FS)
    ax_b.tick_params(axis="y", left=False, labelleft=False)
    plt.setp(ax_b.get_yticklabels(), visible=False)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.22)
    fig.supxlabel("Frequency (kHz)", fontsize=_DECK_AXIS_LABEL_FS, y=0.08)
    _deck_panel_letter(fig, ax_a, "A")
    _deck_panel_letter(fig, ax_b, "B")
    _deck_footer(fig, "")
    _save_deck_png_svg(
        fig, "deck_act1_synapse_ribbons_cohorts", dpi=_COHORT_RIBBONS_DPI
    )


def deck_act1_synapse_ribbons_liberman_strains() -> None:
    """Act I — 1×2 synapse ribbons (Liberman C57BL/6J | CBA/CaJ), by noise group."""
    from utils.nn_stage2_data import load_nn_stage2_data

    apply_slide_rcparams()
    wide = load_nn_stage2_data().reformatted_orig
    if "strain_binary" not in wide.columns:
        print(
            "Skip Act I Liberman strain ribbons — wide table missing strain_binary "
            "(check load_data / WPZ Mouse groups.xlsx)."
        )
        return

    for label, sb in _LIBERMAN_STRAIN_RIBBON_PANELS:
        n = int(wide.loc[wide["strain_binary"].eq(sb)].shape[0])
        print(f"Liberman strain ribbons — {label} (strain_binary={sb}): {n} wide rows")

    gf_c57 = _liberman_synapse_ribbons_for_strain(wide, 1)
    gf_cba = _liberman_synapse_ribbons_for_strain(wide, 0)
    if gf_c57.empty or gf_cba.empty:
        print(
            "Skip Act I Liberman strain ribbons — empty stratum table for one or both strains."
        )
        return

    gf_all = pd.concat([gf_c57, gf_cba], ignore_index=True)
    groups = sorted(gf_all["Group"].astype(str).unique(), key=lambda x: float(x))
    g_to_color = _noise_group_colors(groups)
    g_to_off = _noise_group_offsets(groups)
    ylim = _act1_cohort_ribbons_ylim()
    if ylim is None:
        print(
            "Skip Act I Liberman strain ribbons — missing",
            LIBERMAN_SYNAPSE_MEAN_STD_BY_GROUP_FREQ_PARQUET.resolve(),
            "(run liberman_synapses_group_baseline.ipynb export; needed for y-axis match).",
        )
        return
    y0, y1 = ylim

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.16, 1.0],
        width_ratios=[1, 1],
        wspace=0.12,
        hspace=0.10,
    )
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.set_axis_off()
    ax_leg.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=_NOISE_LOWER_COLOR,
                marker="o",
                linestyle="None",
                markersize=8,
                label=_RIBBON_LEGEND_LABELS[0],
            ),
            Line2D(
                [0],
                [0],
                color=_NOISE_HIGHER_COLOR,
                marker="o",
                linestyle="None",
                markersize=8,
                label=_RIBBON_LEGEND_LABELS[1],
            ),
        ],
        ncol=2,
        loc="center",
        bbox_to_anchor=(0.5, 0.85),
        frameon=True,
        fancybox=False,
        edgecolor="0.82",
        facecolor="white",
        framealpha=1.0,
        fontsize=_DECK_LEGEND_FS,
    )

    ax_a = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1], sharey=ax_a)
    _plot_synapse_ribbons_panel(
        ax_a, gf_c57, g_to_color=g_to_color, g_to_off=g_to_off
    )
    _plot_synapse_ribbons_panel(
        ax_b, gf_cba, g_to_color=g_to_color, g_to_off=g_to_off
    )

    for ax in (ax_a, ax_b):
        ax.set_ylim(y0, y1)

    ax_a.set_ylabel(
        "Synapses / Inner Hair Cell (IHC)", fontsize=_DECK_AXIS_LABEL_FS
    )
    ax_a.tick_params(axis="both", labelsize=_DECK_TICK_FS)
    ax_b.tick_params(axis="x", labelsize=_DECK_TICK_FS)
    ax_b.tick_params(axis="y", left=False, labelleft=False)
    plt.setp(ax_b.get_yticklabels(), visible=False)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.22)
    fig.supxlabel("Frequency (kHz)", fontsize=_DECK_AXIS_LABEL_FS, y=0.08)
    _deck_panel_letter(fig, ax_a, "A")
    _deck_panel_letter(fig, ax_b, "B")
    _deck_footer(fig, "")
    _save_deck_png_svg(
        fig, "deck_act1_synapse_ribbons_liberman_strains", dpi=_COHORT_RIBBONS_DPI
    )


def deck_act_ib_o1_two_marks_liberman() -> None:
    """O1 — two OLS RMSE marks (Liberman in-library training)."""
    p = O1_OLS_LIBERMAN_TWO_MARKS_PARQUET
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p} — run abr_wide_long_comparison.ipynb Liberman OLS "
            "baselines export (writes deck_o1_ols_two_marks_liberman.parquet)."
        )
    df = pd.read_parquet(p).sort_values("mark")
    _ols_lollipop_figure(
        df,
        stem="deck_act2_ols_two_amp_full_liberman_C_RMSE",
        digest_title="=== O1 Liberman OLS (two marks) ===",
        parquet_path=p,
    )


def deck_act_ib_o2_three_noise_liberman() -> None:
    """O2 — three noise-only OLS marks."""
    p = O2_OLS_LIBERMAN_THREE_NOISE_PARQUET
    if not p.exists():
        msg = f"Missing {p} — run abr_wide_long_comparison.ipynb Liberman OLS export."
        raise FileNotFoundError(msg)
    df = pd.read_parquet(p).sort_values("mark")
    _ols_lollipop_figure(
        df,
        stem="deck_act2_ols_three_noise_liberman_C_RMSE",
        digest_title="=== O2 Liberman OLS (three noise-only marks) ===",
        parquet_path=p,
    )


def deck_act_ib_o3_four_stack_liberman() -> None:
    """O3 — four-mark OLS ladder."""
    p = O3_OLS_LIBERMAN_FOUR_STACK_PARQUET
    if not p.exists():
        msg = f"Missing {p} — run abr_wide_long_comparison.ipynb Liberman OLS export."
        raise FileNotFoundError(msg)
    df = pd.read_parquet(p).sort_values("mark")
    _ols_lollipop_figure(
        df,
        stem="deck_act2_ols_four_stack_liberman_C_RMSE",
        digest_title="=== O3 Liberman OLS (four-mark ladder) ===",
        parquet_path=p,
    )


def deck_act2_stage1_roc_calibration() -> None:
    """IId + IIe — Stage 1 wide LR + RF ROC and calibration (animal-level)."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import auc, roc_curve

    ev_path = STAGE1_WIDE_LR_RF_EVAL_PARQUET
    if not ev_path.is_file():
        print(
            "Skip Stage 1 ROC/calibration — missing",
            ev_path.resolve(),
            "(run abr_nn_stage2.ipynb Stage 1 export cell).",
        )
        return
    apply_slide_rcparams()
    plt.rcParams.update(
        {"legend.fontsize": 9, "axes.titlesize": 12, "axes.labelsize": 11}
    )
    ev = pd.read_parquet(ev_path)
    labs = ("Brad", "Liberman")
    colors = {"LR": "#0173B2", "RF": "#DE8F05"}
    roc_label = {"LR": "Logistic Regression", "RF": "Random Forest"}

    print("=== IId Stage 1 ROC digest (animal-level wide classifier) ===")
    print("parquet:", ev_path.resolve())
    for lab in labs:
        sub = ev[ev["lab"].eq(lab)]
        n_anim0 = int(sub["animal_id"].nunique()) if len(sub) else 0
        print(f"--- {lab} --- rows={len(sub)} animals={n_anim0}")
        if sub.empty:
            continue
        for model in ("LR", "RF"):
            m = sub[sub["model"].eq(model)]
            if len(m) < 2:
                continue
            y = m["y_true"].to_numpy()
            s = m["score"].to_numpy()
            if np.unique(y).size < 2:
                print(f"  {model}: skip (single class)")
                continue
            fpr, tpr, _ = roc_curve(y, s)
            print(f"  {model}: AUC={auc(fpr, tpr):.4f}")
    print("=== end IId ===")

    cohort_panels = (("Liberman", "A"), ("Brad", "B"))
    fig = plt.figure(figsize=(8, 5.5))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.16, 1.0],
        width_ratios=[1, 1],
        wspace=0.12,
        hspace=0.10,
    )
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.set_axis_off()
    ax_leg.legend(
        handles=[
            Line2D([0], [0], color=colors[m], lw=2.0, label=roc_label[m])
            for m in ("LR", "RF")
        ],
        ncol=2,
        loc="center",
        bbox_to_anchor=(0.5, 0.85),
        frameon=True,
        fancybox=False,
        edgecolor="0.82",
        facecolor="white",
        framealpha=1.0,
        fontsize=13,
    )

    ax_lib = fig.add_subplot(gs[1, 0])
    ax_brad = fig.add_subplot(gs[1, 1], sharey=ax_lib)

    for ax, (lab, _letter) in zip((ax_lib, ax_brad), cohort_panels):
        sub = ev[ev["lab"].eq(lab)]
        if sub.empty:
            ax.text(
                0.5,
                0.5,
                f"No rows for {lab}",
                ha="center",
                transform=ax.transAxes,
            )
            continue
        for model in ("LR", "RF"):
            m = sub[sub["model"].eq(model)]
            if len(m) < 2:
                continue
            y = m["y_true"].to_numpy()
            s = m["score"].to_numpy()
            if np.unique(y).size < 2:
                continue
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=colors[model], lw=2.0)
        ax.plot([0, 1], [0, 1], ls="--", color="0.55", lw=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axisbelow(True)
        ax.grid(True, which="both", alpha=0.35)
        sns.despine(ax=ax)

    ax_lib.set_ylabel("True positive rate", fontsize=13)
    ax_lib.tick_params(axis="both", labelsize=11)
    ax_brad.tick_params(axis="x", labelsize=11)
    ax_brad.tick_params(axis="y", left=False, labelleft=False)
    plt.setp(ax_brad.get_yticklabels(), visible=False)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18)
    fig.supxlabel("False positive rate", fontsize=13, y=0.06)
    _deck_panel_letter(fig, ax_lib, "A")
    _deck_panel_letter(fig, ax_brad, "B")
    _deck_footer(fig, "")
    _save_deck_png_svg(fig, "deck_act2_stage1_roc")

    print("=== IIe Stage 1 calibration digest ===")
    for lab in labs:
        sub = ev[ev["lab"].eq(lab)]
        if sub.empty:
            continue
        for model in ("LR", "RF"):
            m = sub[sub["model"].eq(model)]
            if len(m) < 4:
                continue
            y = m["y_true"].to_numpy()
            s = m["score"].to_numpy()
            if np.unique(y).size < 2:
                continue
            n_bins = min(5, max(3, len(m) // 3))
            try:
                calibration_curve(y, s, n_bins=n_bins, strategy="uniform")
            except ValueError as e:
                print(f"  {lab} {model}: calibration skip ({e!s})")
                continue
            nb, nm = n_bins, len(m)
            print(f"  {lab} {model}: n_bins={nb} strategy=uniform n={nm}")
    print("=== end IIe ===")

    fig2, axes2 = plt.subplots(2, 1, figsize=(6.8, 8.8), sharex=True)
    for ax, lab in zip(axes2, labs):
        sub = ev[ev["lab"].eq(lab)]
        ax.plot([0, 1], [0, 1], ls="--", color="0.55", lw=1)
        lines_c = []
        labs_c = []
        for model in ("LR", "RF"):
            m = sub[sub["model"].eq(model)]
            if len(m) < 4:
                continue
            y = m["y_true"].to_numpy()
            s = m["score"].to_numpy()
            if np.unique(y).size < 2:
                continue
            try:
                prob_true, prob_pred = calibration_curve(
                    y, s, n_bins=min(5, max(3, len(m) // 3)), strategy="uniform"
                )
            except ValueError:
                continue
            mk = "o" if model == "LR" else "s"
            (line,) = ax.plot(
                prob_pred,
                prob_true,
                marker=mk,
                label=model,
                color=colors[model],
                lw=2,
            )
            lines_c.append(line)
            labs_c.append(roc_label[model])
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.35)
        sns.despine(ax=ax)
        if lines_c:
            ax.legend(
                lines_c,
                labs_c,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                frameon=True,
                fontsize=9,
            )
    axes2[-1].set_xlabel("Mean predicted probability")
    fig2.subplots_adjust(
        left=0.1, right=0.72, top=0.94, bottom=0.1, hspace=0.35
    )
    _deck_footer(fig2, "One score per test animal · wide features · LR vs RF")
    _save_deck_png_svg(fig2, "deck_act2_stage1_calibration")

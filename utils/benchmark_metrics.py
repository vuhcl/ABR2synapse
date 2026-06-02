"""
Unified presentation benchmark plots: classical (OLS + RF + XGB long) + NN Stage 2.

Reads cached Parquet exported from ``abr_wide_long_comparison.ipynb``
(``rows1`` / optional ``rows2``, ``rows6`` for variant-aware classical metrics)
and ``abr_nn_stage2.ipynb`` (``metrics_all`` + optional ``stage1_wide_noise_lr_rf_eval.parquet``).
See ``presentation_benchmarks.ipynb``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Canonical model order for unified slide figures ─────────────────────────
# Margin hints when the **metric is on the vertical axis** (models / categories on x).
# Label is drawn with rotation=90; use horizontal arrows so after rotation they align with ±y:
# ← → the plot's vertical axis (Matplotlib rotates CCW).
# RMSE: lower y is better → ← points down (better), → points up (worse).
RMSE_Y_AXIS_MARGIN_HINT = r"$\longleftarrow$ better $\mid$ worse $\longrightarrow$"
# R²: higher y is better → ← points down (worse), → points up (better).
R2_Y_AXIS_MARGIN_HINT = r"$\longleftarrow$ worse $\mid$ better $\longrightarrow$"

BENCHMARK_DISPLAY_ORDER: list[str] = [
    "LR baseline",
    "LR full",
    "RF",
    "XGB",
    "MLP",
    "CNN (Wave I)",
    "CNN (full wave)",
]

_CLASSICAL_MODEL_MAP: dict[str, str] = {
    "LR baseline": "OLS base",
    "LR full": "OLS full",
    "RF": "RF",
    "XGB": "XGB",
}

# NN: Liberman resampling grid for Wave I / full CNN (plan default)
NN_WAVE_I_KEY = "CNN_WaveI_LibT"
NN_FULL_KEY = "CNN_full_LibT"
NN_MLP_KEY = "MLP"

_NN_DISPLAY_MAP: dict[str, str] = {
    NN_MLP_KEY: "MLP",
    NN_WAVE_I_KEY: "CNN (Wave I)",
    NN_FULL_KEY: "CNN (full wave)",
}

# Wide/long + SPL variants for classical strip plots (colors shared Brad ↔ Liberman)
VARIANT_ORDER_BRAD: tuple[str, ...] = ("all-long", "all-wide")
VARIANT_ORDER_LIBERMAN: tuple[str, ...] = (
    "all-long",
    "all-wide",
    "even-long",
    "even-wide",
)
VARIANT_PALETTE: dict[str, str] = {
    "all-long": "#1f77b4",
    "all-wide": "#ff7f0e",
    "even-long": "#2ca02c",
    "even-wide": "#d62728",
}
# NN markers use the same blue as classical **all-long** (explicit hex for scatter).
NN_ALL_LONG_FACE = VARIANT_PALETTE["all-long"]


def repo_root() -> Path:
    """Directory that contains the ``utils`` package (repository / project root)."""
    return Path(__file__).resolve().parent.parent


def resolve_cache_file(name: str) -> Path:
    """
    Resolve a single cache filename under ``figures/cache``.

    Checks the canonical root ``<repo>/figures/cache`` first, then
    ``<repo>/ABR2synapse/figures/cache`` so split exports (e.g. classical/NN only
    under ``ABR2synapse``) still load when Jupyter cwd is the outer repo root.

    If the file is missing in both locations, returns the path under the
    canonical directory (for clear ``FileNotFoundError`` messages).
    """
    roots = (
        repo_root() / "figures" / "cache",
        repo_root() / "ABR2synapse" / "figures" / "cache",
    )
    for root in roots:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return roots[0] / name


# Canonical directory for **new** exports (mkdir in notebooks as needed).
CACHE_DIR = repo_root() / "figures" / "cache"

CLASSICAL_BENCHMARK_PARQUET = resolve_cache_file("classical_benchmark_long.parquet")
NN_METRICS_PARQUET = resolve_cache_file("nn_metrics_all.parquet")
BENCHMARK_MERGED_PARQUET = CACHE_DIR / "benchmark_metrics.parquet"
SEED_RUNS_PARQUET = resolve_cache_file("benchmark_metrics_by_seed.parquet")

# Section 5.3 — synthesis CV + HP artifacts (written by stage2 notebooks / scripts)
STAGE2_DATA_DIR = CACHE_DIR / "stage2_data"
STAGE2_BEST_HP_DIR = CACHE_DIR / "stage2_best_hp"
LIBERMAN_T5_SKLEARN_HP_JSON = STAGE2_BEST_HP_DIR / "liberman_t5_sklearn.json"
STAGE2_SYNTHESIS_CV_PROGRESS_JSON = CACHE_DIR / "stage2_synthesis_cv_progress.json"
STAGE2_SYNTHESIS_CV_FOLDS_PARQUET = CACHE_DIR / "stage2_synthesis_cv_folds.parquet"
STAGE2_SYNTHESIS_CV_SUMMARY_PARQUET = CACHE_DIR / "stage2_synthesis_cv_summary.parquet"
STAGE2_SYNTHESIS_CV_POOLED_FOLDS_PARQUET = (
    CACHE_DIR / "stage2_synthesis_cv_pooled_folds.parquet"
)
STAGE2_SYNTHESIS_CV_POOLED_PROGRESS_JSON = (
    CACHE_DIR / "stage2_synthesis_cv_pooled_progress.json"
)
STAGE2_SYNTHESIS_CV_OOF_PARQUET = CACHE_DIR / "stage2_synthesis_cv_oof.parquet"
STAGE2_SYNTHESIS_CV_OOF_PROGRESS_JSON = (
    CACHE_DIR / "stage2_synthesis_cv_oof_progress.json"
)
STAGE2_SYNTHESIS_CV_POOLED_OOF_PARQUET = (
    CACHE_DIR / "stage2_synthesis_cv_pooled_oof.parquet"
)
STAGE2_SYNTHESIS_CV_POOLED_OOF_PROGRESS_JSON = (
    CACHE_DIR / "stage2_synthesis_cv_pooled_oof_progress.json"
)
STAGE2_SYNTHESIS_CV_OOF_R2_SUMMARY_PARQUET = (
    CACHE_DIR / "stage2_synthesis_cv_oof_r2_summary.parquet"
)
STAGE2_SYNTHESIS_STAGE1_TABLE_PARQUET = (
    CACHE_DIR / "stage2_synthesis_stage1_classification_table.parquet"
)


def stage2_synthesis_cv_paths(
    n_folds: int,
) -> tuple[Path, Path, Path]:
    """
    Cache paths for synthesis CV keyed by fold count.

    ``n_folds=10`` uses the legacy unsuffixed filenames (existing runs).
    Other counts use ``stage2_synthesis_cv_{n}fold_*`` under ``figures/cache/``.
    """
    if n_folds == 10:
        return (
            STAGE2_SYNTHESIS_CV_FOLDS_PARQUET,
            STAGE2_SYNTHESIS_CV_PROGRESS_JSON,
            STAGE2_SYNTHESIS_CV_SUMMARY_PARQUET,
        )
    tag = f"{n_folds}fold"
    return (
        CACHE_DIR / f"stage2_synthesis_cv_{tag}_folds.parquet",
        CACHE_DIR / f"stage2_synthesis_cv_{tag}_progress.json",
        CACHE_DIR / f"stage2_synthesis_cv_{tag}_summary.parquet",
    )

LIBERMAN_GROUP_MEAN_BASELINE_PARQUET = resolve_cache_file(
    "liberman_group_mean_baseline_metrics.parquet"
)
LIBERMAN_SYNAPSE_MEAN_STD_BY_GROUP_FREQ_PARQUET = resolve_cache_file(
    "liberman_synapse_mean_std_by_group_frequency.parquet"
)
BRAD_SYNAPSE_MEAN_STD_BY_GROUP_FREQ_PARQUET = resolve_cache_file(
    "brad_synapse_mean_std_by_group_frequency.parquet"
)
LIBERMAN_GROUP_BASELINE_TEST_BY_GROUP_PARQUET = resolve_cache_file(
    "liberman_group_baseline_metrics_test_by_group.parquet"
)
# Act Ib **O1** — two OLS RMSE marks (Liberman scenario C); written by ``abr_wide_long_comparison.ipynb``.
O1_OLS_LIBERMAN_TWO_MARKS_PARQUET = resolve_cache_file(
    "deck_o1_ols_two_marks_liberman.parquet"
)
# Act Ib **O2** — three OLS RMSE marks (noise-only predictors; Liberman scenario C).
O2_OLS_LIBERMAN_THREE_NOISE_PARQUET = resolve_cache_file(
    "deck_o2_ols_three_noise_marks_liberman.parquet"
)
# Act Ib **O3** — four OLS RMSE marks (ladder to shipped linear stack; Liberman scenario C).
O3_OLS_LIBERMAN_FOUR_STACK_PARQUET = resolve_cache_file(
    "deck_o3_ols_four_stack_marks_liberman.parquet"
)
STAGE1_WIDE_RF_METRICS_JSON = resolve_cache_file("stage1_wide_rf_metrics.json")
STAGE1_WIDE_LR_RF_EVAL_PARQUET = resolve_cache_file(
    "stage1_wide_noise_lr_rf_eval.parquet"
)


def apply_benchmark_grid(ax: plt.Axes, *, n_models: int | None = None) -> None:
    """
    Public alias for deck notebooks: y-grid plus vertical guides at model indices.

    If ``n_models`` is omitted, uses ``len(BENCHMARK_DISPLAY_ORDER)``.
    """
    nm = n_models if n_models is not None else len(BENCHMARK_DISPLAY_ORDER)
    _apply_benchmark_axes_grids(ax, n_models=nm)


def apply_slide_rcparams() -> None:
    """Slide-ready matplotlib + seaborn: ``sns.set_context('talk')`` plus explicit font sizes."""
    sns.set_context("talk")
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 16,
        }
    )


def _wl_lookup_row(
    rows: Sequence[tuple],
    test: str,
    scenario: str,
    model: str,
    *,
    long_fmt: bool,
    metric: Literal["r2", "rmse"],
) -> float:
    """Same logic as ``_wl_lookup`` in ``abr_wide_long_comparison.ipynb``."""
    scen_m = scenario if model in ("RF", "XGB") else "—"
    for r in rows:
        if r[0] == test and r[1] == scen_m and r[2] == model:
            if metric == "r2":
                return float(r[4] if long_fmt else r[3])
            return float(r[6] if long_fmt else r[5])
    return float("nan")


def classical_long_rows_to_dataframe(rows1: Sequence[tuple]) -> pd.DataFrame:
    """
    Convert master ``rows1`` list (all-SPL, no strain) to long format.

    Each ``rows1`` row is
    ``(test, scen, model, r2w, r2l, rmsew, rmsel)`` with ``test`` in
    ``{"Brad","Lib"}``.
    """
    recs: list[dict] = []
    tests = ("Brad", "Lib")
    scenarios_abc = ("A", "B", "C")
    long_fmt = True

    for disp in ("LR baseline", "LR full", "RF", "XGB"):
        mk = _CLASSICAL_MODEL_MAP[disp]
        for te in tests:
            tname = "Brad" if te == "Brad" else "Liberman"
            if mk in ("OLS base", "OLS full"):
                r2 = _wl_lookup_row(rows1, te, "A", mk, long_fmt=long_fmt, metric="r2")
                rmse = _wl_lookup_row(
                    rows1, te, "A", mk, long_fmt=long_fmt, metric="rmse"
                )
                for scen in scenarios_abc:
                    recs.append(
                        {
                            "test_set": tname,
                            "scenario": scen,
                            "model": disp,
                            "variant": pd.NA,
                            "R2": r2,
                            "RMSE": rmse,
                            "source": "classical",
                        }
                    )
            else:
                for scen in scenarios_abc:
                    r2 = _wl_lookup_row(
                        rows1, te, scen, mk, long_fmt=long_fmt, metric="r2"
                    )
                    rmse = _wl_lookup_row(
                        rows1, te, scen, mk, long_fmt=long_fmt, metric="rmse"
                    )
                    recs.append(
                        {
                            "test_set": tname,
                            "scenario": scen,
                            "model": disp,
                            "variant": pd.NA,
                            "R2": r2,
                            "RMSE": rmse,
                            "source": "classical",
                        }
                    )

    return pd.DataFrame(recs)


def matched_train_scenario(test_set: str) -> str:
    """Matched in-domain training: **A** for Brad test, **C** for Liberman test."""
    if test_set == "Brad":
        return "A"
    if test_set == "Liberman":
        return "C"
    raise ValueError(f"Unknown test_set: {test_set!r}")


def classical_variants_to_dataframe(
    rows1: Sequence[tuple],
    rows2: Sequence[tuple] | None = None,
    rows6: Sequence[tuple] | None = None,
) -> pd.DataFrame:
    """
    Long classical table with **variant** column.

    **Brad:** ``all-long`` / ``all-wide`` from ``rows1`` (all SPL, no strain).

    **Liberman:** ``all-long`` / ``all-wide`` from ``rows2`` (all SPL + strain); ``even-long`` /
    ``even-wide`` from ``rows6`` (even SPL + strain). If ``rows2`` or ``rows6`` is missing,
    Liberman falls back to ``rows1`` for **all-long** / **all-wide** only (no even-*).
    """
    recs: list[dict] = []
    scenarios_abc = ("A", "B", "C")
    tests = ("Brad", "Lib")

    brad_specs: list[tuple[str, Sequence[tuple], bool]] = [
        ("all-long", rows1, True),
        ("all-wide", rows1, False),
    ]

    def _lib_specs() -> list[tuple[str, Sequence[tuple], bool]]:
        if rows2 is not None and rows6 is not None:
            return [
                ("all-long", rows2, True),
                ("all-wide", rows2, False),
                ("even-long", rows6, True),
                ("even-wide", rows6, False),
            ]
        if rows2 is not None:
            return [
                ("all-long", rows2, True),
                ("all-wide", rows2, False),
            ]
        return [
            ("all-long", rows1, True),
            ("all-wide", rows1, False),
        ]

    for disp in ("LR baseline", "LR full", "RF", "XGB"):
        mk = _CLASSICAL_MODEL_MAP[disp]
        for te in tests:
            tname = "Brad" if te == "Brad" else "Liberman"
            specs = brad_specs if te == "Brad" else _lib_specs()
            for scen in scenarios_abc:
                for vlabel, bundle, long_fmt in specs:
                    if mk in ("OLS base", "OLS full"):
                        r2 = _wl_lookup_row(
                            bundle, te, "A", mk, long_fmt=long_fmt, metric="r2"
                        )
                        rmse = _wl_lookup_row(
                            bundle, te, "A", mk, long_fmt=long_fmt, metric="rmse"
                        )
                    else:
                        r2 = _wl_lookup_row(
                            bundle, te, scen, mk, long_fmt=long_fmt, metric="r2"
                        )
                        rmse = _wl_lookup_row(
                            bundle, te, scen, mk, long_fmt=long_fmt, metric="rmse"
                        )
                    recs.append(
                        {
                            "test_set": tname,
                            "scenario": scen,
                            "model": disp,
                            "variant": vlabel,
                            "R2": r2,
                            "RMSE": rmse,
                            "source": "classical",
                        }
                    )

    return pd.DataFrame(recs)


def nn_metrics_to_benchmark_df(metrics_all: pd.DataFrame) -> pd.DataFrame:
    """Subset ``metrics_all`` to MLP + LibT CNN pair; rename models for slides."""
    df = metrics_all.copy()
    if "stage2_format" in df.columns:
        df = df[df["stage2_format"].fillna("long").eq("long")]
    keep = [NN_MLP_KEY, NN_WAVE_I_KEY, NN_FULL_KEY]
    sub = df[df["model"].isin(keep)].copy()
    sub = sub.assign(
        model=sub["model"].map(_NN_DISPLAY_MAP),
        source="nn",
        variant=pd.NA,
    )
    return sub


def merge_benchmark(classical_df: pd.DataFrame, nn_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate classical + NN long tables (same columns including ``variant``)."""
    cl = classical_df.copy()
    nn = nn_df.copy()
    if "variant" not in cl.columns:
        cl["variant"] = pd.NA
    if "variant" not in nn.columns:
        nn["variant"] = pd.NA
    cols = ["test_set", "scenario", "model", "variant", "R2", "RMSE", "source"]
    return pd.concat([cl[cols], nn[cols]], ignore_index=True)


def build_benchmark_df(
    classical_df: pd.DataFrame | None = None,
    nn_df: pd.DataFrame | None = None,
    *,
    rows1: Sequence[tuple] | None = None,
    metrics_all: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build merged long ``benchmark_df`` from explicit frames or raw notebook objects.

    Provide either ``classical_df`` or ``rows1``; either ``nn_df`` or ``metrics_all``.
    """
    if classical_df is None:
        if rows1 is None:
            raise ValueError("Need classical_df or rows1")
        classical_df = classical_long_rows_to_dataframe(rows1)
    if nn_df is None:
        if metrics_all is None:
            raise ValueError("Need nn_df or metrics_all")
        nn_df = nn_metrics_to_benchmark_df(metrics_all)
    return merge_benchmark(classical_df, nn_df)


def benchmark_merged_has_classical_variants(merged_path: Path | str) -> bool:
    """
    True if merged Parquet has a **variant** column with at least one non-null classical row.
    Used to avoid loading a stale ``benchmark_metrics.parquet`` built before variant export.
    """
    mp = Path(merged_path)
    if not mp.exists():
        return False
    df = pd.read_parquet(mp)
    if "variant" not in df.columns:
        return False
    cl = df[df["source"] == "classical"]
    if cl.empty:
        return False
    return bool(cl["variant"].notna().any())


def load_benchmark_from_cache(
    classical_path: Path | str | None = None,
    nn_path: Path | str | None = None,
    *,
    seed_path: Path | str | None = None,
    save_merged: Path | str | None = BENCHMARK_MERGED_PARQUET,
) -> pd.DataFrame:
    """
    Load classical + NN Parquet exports (``classical_benchmark_long.parquet``,
    ``nn_metrics_all.parquet``), merge, optionally attach multi-seed SEM, and optionally
    persist merged table for plot-only reruns.
    """
    cpath = Path(classical_path or CLASSICAL_BENCHMARK_PARQUET)
    npath = Path(nn_path or NN_METRICS_PARQUET)
    if not cpath.exists():
        raise FileNotFoundError(f"Missing classical metrics: {cpath.resolve()}")
    if not npath.exists():
        raise FileNotFoundError(f"Missing NN metrics: {npath.resolve()}")
    classical_df = pd.read_parquet(cpath)
    nn_df = nn_metrics_to_benchmark_df(pd.read_parquet(npath))
    out = merge_benchmark(classical_df, nn_df)
    spath = Path(seed_path) if seed_path is not None else SEED_RUNS_PARQUET
    out = attach_seed_sem(out, spath)
    if save_merged:
        mp = Path(save_merged)
        mp.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(mp)
    return out


def attach_seed_sem(
    benchmark_df: pd.DataFrame,
    seed_path: Path | str,
) -> pd.DataFrame:
    """
    Optional Phase 2: merge ``benchmark_metrics_by_seed.parquet`` long format:
    columns ``seed``, ``test_set``, ``scenario``, ``model``, ``R2``, ``RMSE``.
    Adds ``R2_sem``, ``RMSE_sem`` aggregated per group (broadcast to all ``variant`` rows).
    """
    path = Path(seed_path)
    if not path.exists():
        return benchmark_df
    s = pd.read_parquet(path)
    g = s.groupby(["test_set", "scenario", "model"], as_index=False).agg(
        R2=("R2", "mean"),
        RMSE=("RMSE", "mean"),
        R2_sem=(
            "R2",
            lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
        ),
        RMSE_sem=(
            "RMSE",
            lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
        ),
    )
    out = benchmark_df.merge(
        g[["test_set", "scenario", "model", "R2_sem", "RMSE_sem"]],
        on=["test_set", "scenario", "model"],
        how="left",
        suffixes=("", "_seed"),
    )
    return out


def summarize_best_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each **test_set** and **model**, select the row with **maximum test R²**
    across all **scenario** × **variant** rows (NN included). Report **RMSE** from that
    same row (not the minimum RMSE).
    """
    rows: list[dict] = []
    for ts in ("Brad", "Liberman"):
        sub_ts = df[df["test_set"] == ts]
        if sub_ts.empty:
            continue
        for model in BENCHMARK_DISPLAY_ORDER:
            sm = sub_ts[sub_ts["model"] == model]
            if sm.empty:
                continue
            idx = sm["R2"].idxmax()
            r = sm.loc[idx]
            rec: dict = {
                "test_set": ts,
                "model": model,
                "best_R2": float(r["R2"]),
                "RMSE_at_best_R2": float(r["RMSE"]),
                "scenario_at_best": r["scenario"],
            }
            if "variant" in r.index and pd.notna(r["variant"]):
                rec["variant_at_best"] = r["variant"]
            rows.append(rec)
    return pd.DataFrame(rows)


def _scatter_classical_by_variant(
    ax: plt.Axes,
    cls: pd.DataFrame,
    *,
    col: str,
    tname: str,
) -> None:
    """Matplotlib scatter with explicit colors and horizontal dodge (reliable vs. seaborn stripplot)."""
    v_order_full = list(
        VARIANT_ORDER_BRAD if tname == "Brad" else VARIANT_ORDER_LIBERMAN
    )
    present = [
        v
        for v in v_order_full
        if v in set(cls["variant"].dropna().astype(str).unique())
    ]
    n = len(present)
    if n <= 0:
        return
    if n == 1:
        offsets = [0.0]
    else:
        # Tight horizontal dodge so variant points read as one model column.
        span = 0.22
        offsets = list(np.linspace(-span / 2, span / 2, n))
    v_to_dx = {v: offsets[i] for i, v in enumerate(present)}

    for _, row in cls.iterrows():
        m = row["model"]
        if m not in BENCHMARK_DISPLAY_ORDER:
            continue
        mi = BENCHMARK_DISPLAY_ORDER.index(m)
        v = row["variant"]
        if pd.isna(v):
            continue
        v = str(v)
        if v not in v_to_dx:
            continue
        dx = v_to_dx[v]
        c = VARIANT_PALETTE.get(v, "0.45")
        ax.scatter(
            mi + dx,
            float(row[col]),
            s=105,
            c=c,
            edgecolors="0.25",
            linewidths=0.75,
            zorder=5,
        )


def _ylim_tight(
    vals: Sequence[float], *, metric: Literal["R2", "RMSE"]
) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (0.0, 1.0)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    span = hi - lo
    pad = 0.07 * span if span > 1e-12 else 0.04 * max(abs(hi), 1.0)
    lo2, hi2 = lo - pad, hi + pad
    if metric == "RMSE" and lo >= 0:
        lo2 = max(0.0, lo2)
    return (lo2, hi2)


def _apply_benchmark_axes_grids(ax: plt.Axes, *, n_models: int) -> None:
    """
    Y-axis horizontal grid + **vertical** lines at each model index.

    ``axis='x'`` on ``ax.grid`` is unreliable with some seaborn/rc styles; ``axvline`` is not.
    """
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
    for xv in range(n_models):
        ax.axvline(
            xv,
            color="0.70",
            linewidth=0.9,
            linestyle=(0, (1, 2)),
            alpha=0.88,
            zorder=0,
        )


def plot_unified_benchmark(
    df: pd.DataFrame,
    *,
    metric: Literal["R2", "RMSE"],
    out_path: Path | str,
    omit_liberman_scenario_a_r2: bool = True,
    yerr_col: str | None = None,
    figsize: tuple[float, float] = (13.8, 7.6),
) -> None:
    """
    **2×2** strip plots: rows = Brad / Liberman test; columns = train **B (combined)** /
    train **matched** (A for Brad, C for Liberman). **No per-panel titles** (add in slides).

    Classical RF/XGB/OLS: multiple **variants** per model (hue): Brad uses **all-long** /
    **all-wide**; Liberman adds **even-long** / **even-wide** when present in the cache.
    NN models use the same **all-long** dot styling as classical (no separate legend entry).
    Each panel has one rotated left-margin label: **$R^2$** or **RMSE**, newline, then an
    **arrow** worse/better hint for that metric on **y** (``linespacing`` controls gap).
    No figure-level suptitle (add in slides). Y-limits are zoomed to the plotted values.

    ``omit_liberman_scenario_a_r2``: deprecated (ignored); kept for call compatibility.
    ``yerr_col``: reserved; multi-seed SEM is not drawn on variant strips (merge still in ``df``).

    Saves **PNG** (300 dpi) and **SVG** next to ``out_path`` (same basename; **S3** digests).
    """
    if omit_liberman_scenario_a_r2:
        warnings.warn(
            "omit_liberman_scenario_a_r2 is ignored for the 2×2 stripplot layout.",
            UserWarning,
            stacklevel=2,
        )
    del yerr_col  # signature compatibility
    apply_slide_rcparams()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    col = metric
    metric_df = df.copy()
    if "variant" not in metric_df.columns:
        metric_df["variant"] = pd.NA

    _cl = metric_df[metric_df["source"] == "classical"]
    if not _cl.empty and not _cl["variant"].notna().any():
        warnings.warn(
            "Classical rows have no **variant** values. Re-export "
            "`figures/cache/classical_benchmark_long.parquet` from "
            "`abr_wide_long_comparison.ipynb`, then rebuild the merged cache with "
            "`load_benchmark_from_cache(...)` (or delete `benchmark_metrics.parquet`).",
            UserWarning,
            stacklevel=2,
        )

    model_cat = pd.Categorical(
        metric_df["model"],
        categories=BENCHMARK_DISPLAY_ORDER,
        ordered=True,
    )
    metric_df = metric_df.assign(model=model_cat)

    test_order = ["Brad", "Liberman"]
    scen_cols: list[tuple[str, str]] = [
        ("B", "combined"),
        ("__match__", "matched"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=False)
    better_txt = R2_Y_AXIS_MARGIN_HINT if metric == "R2" else RMSE_Y_AXIS_MARGIN_HINT
    _fs_tick = 13
    _fs_better = 12
    _fs_leg = 12

    xtick_labels = [
        "LR base",
        "LR full",
        "RF",
        "XGB",
        "MLP",
        "CNN W I",
        "CNN full",
    ]

    for row, tname in enumerate(test_order):
        matched = matched_train_scenario(tname)
        for col_ix, (scen_key, _kind) in enumerate(scen_cols):
            ax = axes[row, col_ix]
            scen = "B" if scen_key == "B" else matched
            sub = metric_df[
                (metric_df["test_set"] == tname) & (metric_df["scenario"] == scen)
            ].copy()
            if sub.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_xticks([])
                sns.despine(ax=ax)
                continue

            cls = sub[sub["source"] == "classical"].copy()
            nn_df = sub[sub["source"] == "nn"]
            use_hue = not cls.empty and cls["variant"].notna().any()

            if cls.empty and nn_df.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_xticks([])
                sns.despine(ax=ax)
                continue

            if not cls.empty:
                if use_hue:
                    _scatter_classical_by_variant(ax, cls, col=col, tname=tname)
                else:
                    for _, row in cls.iterrows():
                        m = row["model"]
                        if m not in BENCHMARK_DISPLAY_ORDER:
                            continue
                        mi = BENCHMARK_DISPLAY_ORDER.index(m)
                        ax.scatter(
                            mi,
                            float(row[col]),
                            s=95,
                            c="0.35",
                            edgecolors="0.25",
                            linewidths=0.6,
                            zorder=4,
                        )
            else:
                ax.set_xticks(range(len(BENCHMARK_DISPLAY_ORDER)))
                ax.set_xticklabels(xtick_labels, rotation=28, ha="right")

            for mi, mname in enumerate(BENCHMARK_DISPLAY_ORDER):
                hit = nn_df[nn_df["model"] == mname]
                if hit.empty:
                    continue
                yv = float(hit.iloc[0][col])
                ax.scatter(
                    mi,
                    yv,
                    s=105,
                    marker="o",
                    facecolors=NN_ALL_LONG_FACE,
                    edgecolors="0.25",
                    linewidths=0.75,
                    zorder=8,
                )

            yvals: list[float] = []
            if not cls.empty:
                for _, rr in cls.iterrows():
                    yvals.append(float(rr[col]))
            if not nn_df.empty:
                for _, rr in nn_df.iterrows():
                    yvals.append(float(rr[col]))
            if yvals:
                lo, hi = _ylim_tight(yvals, metric=metric)
                ax.set_ylim(lo, hi)

            # Single rotated label: metric line, newline, then worse/better (linespacing separates).
            metric_annot = r"$R^2$" if metric == "R2" else "RMSE"
            margin_lbl = metric_annot + "\n" + better_txt
            ax.text(
                -0.28,
                0.5,
                margin_lbl,
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=_fs_better,
                linespacing=1.85,
                clip_on=False,
            )
            ax.tick_params(axis="both", labelsize=_fs_tick)

            ax.set_xticks(range(len(BENCHMARK_DISPLAY_ORDER)))
            ax.set_xticklabels(xtick_labels, rotation=26, ha="right")
            ax.set_xlabel(None)
            _apply_benchmark_axes_grids(ax, n_models=len(BENCHMARK_DISPLAY_ORDER))
            sns.despine(ax=ax, top=True, right=True)

    # Figure legend: SPL/variant colors only (NN uses same all-long dot as classical).
    from matplotlib.patches import Patch

    leg_handles = [
        Patch(
            facecolor=VARIANT_PALETTE[v],
            edgecolor="0.25",
            label=v.replace("-", " "),
        )
        for v in VARIANT_ORDER_LIBERMAN
    ]

    # No figure suptitle — add main title in slide deck. Left margin for rotated labels.
    fig.subplots_adjust(
        left=0.14,
        right=0.985,
        top=0.96,
        bottom=0.16,
        wspace=0.28,
        hspace=0.36,
    )
    fig.legend(
        handles=leg_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        fontsize=_fs_leg,
        frameon=True,
    )
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.12)
    slug = "r2" if metric == "R2" else "rmse"
    print(f"=== S3 benchmark_{slug} digest ===")
    print(f"  n_rows={len(df)}  out_png={out_path.resolve()}  out_svg={svg_path.resolve()}")
    if col in df.columns:
        print(f"  metric_column={col}  finite_values={int(df[col].notna().sum())}")
    print(f"=== end S3 benchmark_{slug} ===")
    plt.close(fig)


def plot_stage1_bars(
    metrics: Mapping[str, tuple[float, float, float]],
    out_path: Path | str,
    *,
    labels: tuple[str, str] = ("Brad Buran wide", "Liberman wide"),
) -> None:
    """
    Grouped bars for Stage 1 selected model: val row-acc, test animal acc, test AUC.

    ``metrics``: e.g.
    ``{"Brad": (val_acc_pre, test_acc_post, test_auc_post), "Lib": (...)}`` from
    ``stage1_wide_metrics.json`` or ``stage1_wide_rf_metrics.json`` shim.

    Saves **PNG** (300 dpi) and **SVG**; digest **S4** when invoked from
    ``presentation_benchmarks.ipynb``.
    """
    apply_slide_rcparams()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labs = list(labels)
    cats = ["Val acc (row)", "Test acc (animal, Youden)", "Test AUC (animal)"]
    x = np.arange(len(cats))
    w = 0.35
    br = metrics["Brad"]
    li = metrics["Lib"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, br, w, label=labs[0], color="C0")
    ax.bar(x + w / 2, li, w, label=labs[1], color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=13)
    ax.set_ylabel("Score", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.35)
    sns.despine(ax=ax)
    fig.tight_layout()
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print("=== S4 stage1_wide_rf bars digest ===")
    print(f"  metrics_keys={list(metrics.keys())}  Brad={metrics.get('Brad')}  Lib={metrics.get('Lib')}")
    print(f"  out_png={out_path.resolve()}  out_svg={svg_path.resolve()}")
    print("=== end S4 stage1_wide_rf ===")
    plt.close(fig)


def plot_stage1_lr_rf_roc_and_calibration(
    ev_path: Path | str | None = None,
    *,
    out_dir: Path | str | None = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    ROC (2×1: Brad, Liberman) + calibration (2×1) from Appendix D animal-level scores.

    Reads ``stage1_wide_noise_lr_rf_eval.parquet`` (columns ``lab``, ``model``,
    ``animal_id``, ``y_true``, ``score``). Saves **PNG + SVG** under ``out_dir``
    (deck policy; no PDF).
    Returns ``(roc_path, calibration_path)`` or ``(None, None)`` if the eval file is missing.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import auc, roc_curve

    ev_path = Path(ev_path or STAGE1_WIDE_LR_RF_EVAL_PARQUET)
    out_dir = Path(out_dir or (repo_root() / "figures" / "presentation"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ev_path.is_file():
        return (None, None)

    apply_slide_rcparams()
    ev = pd.read_parquet(ev_path)
    labs = ("Brad", "Liberman")
    colors = {"LR": "#0173B2", "RF": "#DE8F05"}

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.2), sharex=True)
    for ax, lab in zip(axes, labs):
        sub = ev[ev["lab"].eq(lab)]
        if sub.empty:
            ax.text(0.5, 0.5, f"No rows for {lab}", ha="center", transform=ax.transAxes)
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
            ax.plot(
                fpr,
                tpr,
                label=f"{model} (AUC={auc(fpr, tpr):.3f})",
                color=colors[model],
                lw=2.2,
            )
        ax.plot([0, 1], [0, 1], ls="--", color="0.55", lw=1)
        ax.set_ylabel("True positive rate")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.35)
        sns.despine(ax=ax)
        n_anim = int(sub["animal_id"].nunique())
        ax.text(0.02, 0.02, f"n={n_anim} test animals", transform=ax.transAxes, fontsize=10)
    axes[-1].set_xlabel("False positive rate")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.10, hspace=0.32)
    roc_base = out_dir / "stage1_wide_lr_rf_roc"
    fig.savefig(roc_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(roc_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.12)
    print("=== S1 stage1_wide_lr_rf_roc digest ===")
    print(f"  parquet={ev_path.resolve()}  n_rows={len(ev)}")
    for lab in labs:
        sub = ev[ev["lab"].eq(lab)]
        if sub.empty:
            print(f"  {lab}: (no rows)")
        else:
            na = int(sub["animal_id"].nunique())
            mods = sorted(sub["model"].dropna().astype(str).unique().tolist())
            print(f"  {lab}: n_test_animals={na}  models={mods}")
    print("=== end S1 ===")
    plt.close(fig)

    fig2, axes2 = plt.subplots(2, 1, figsize=(7.2, 8.2), sharex=True)
    for ax, lab in zip(axes2, labs):
        sub = ev[ev["lab"].eq(lab)]
        ax.plot([0, 1], [0, 1], ls="--", color="0.55", lw=1)
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
            ax.plot(prob_pred, prob_true, marker="o", label=model, color=colors[model], lw=2)
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.35)
        sns.despine(ax=ax)
    axes2[-1].set_xlabel("Mean predicted probability")
    fig2.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.10, hspace=0.32)
    cal_base = out_dir / "stage1_wide_lr_rf_calibration"
    fig2.savefig(cal_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig2.savefig(cal_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.12)
    print("=== S2 stage1_wide_lr_rf_calibration digest ===")
    print(f"  parquet={ev_path.resolve()}  (same eval frame as S1)")
    print("=== end S2 ===")
    plt.close(fig2)

    return (roc_base.with_suffix(".png"), cal_base.with_suffix(".png"))


MULTI_SEED_DOC = """
Phase 2 multi-seed SEM (test R² / RMSE): repeat full fit + evaluation with fixed splits
and seeds 1..n; save long Parquet::

    seed, test_set, scenario, model, R2, RMSE

Merge with ``attach_seed_sem``. See plan document.
"""

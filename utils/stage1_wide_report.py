"""Stage-1 wide RF/LR exports and summary tables for deck/benchmarks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.benchmark_metrics import repo_root

STAGE1_PREAGG_HEATMAP_STEM = "supplementary_figure_stage1_preagg_accuracy_heatmap"
STAGE1_PREAGG_HEATMAP_TEST_STEM = (
    "supplementary_figure_stage1_preagg_accuracy_heatmap_test"
)
STAGE1_ROW_THRESHOLD = 0.5
BRAD_TX_ORDER: tuple[str, ...] = ("young", "age", "noise+acute", "noise+age")
# Brad Hz→kHz rounding; align with Liberman test-grid labels (see deck_storyline).
_HEATMAP_FREQ_ALIAS: dict[float, float] = {5.7: 5.6, 45.3: 45.2}
PRINT_HEATMAP_HEIGHT_IN = 9.0
PRINT_HEATMAP_DPI = 300
PRINT_HEATMAP_CBAR_WIDTH = 0.07
PRINT_HEATMAP_CBAR_WSPACE = 0.06
PRINT_HEATMAP_RIGHT_PAD_IN = 0.38
BRAD_TX_DISPLAY: dict[str, str] = {
    "young": "young",
    "age": "aged",
    "noise+acute": "acute noise exp.",
    "noise+age": "aged noise exp.",
}
_LIBERMAN_GROUP_DISPLAY: dict[str, str] = {
    "101dB 24h post": "101 dB, 24 hours post",
    "14wks ctrl": "14 weeks control",
    "6wks ctrl": "6 weeks control",
    "8wks ctrl": "8 weeks control",
    "90dB 0h post": "90 dB, 0 hours post",
    "90dB 24h post": "90 dB, 24 hours post",
    "90dB 2w post": "90 dB, 2 weeks post",
    "94dB 0h post": "94 dB, 0 hours post",
    "94dB 24h post": "94 dB, 24 hours post",
    "94dB 2w post": "94 dB, 2 weeks post",
    "94dB 8wks post": "94 dB, 8 weeks post",
    "98dB 24h post": "98 dB, 24 hours post",
    "98dB 2w post": "98 dB, 2 weeks post",
    "98dB 8wks post": "98 dB, 8 weeks post",
}
# Y-axis order (top → bottom) for Liberman heatmap panel.
LIBERMAN_GROUP_ORDER: tuple[str, ...] = (
    "6wks ctrl",
    "8wks ctrl",
    "14wks ctrl",
    "90dB 0h post",
    "90dB 24h post",
    "90dB 2w post",
    "94dB 0h post",
    "94dB 24h post",
    "94dB 2w post",
    "94dB 8wks post",
    "98dB 24h post",
    "98dB 2w post",
    "98dB 8wks post",
    "101dB 24h post",
)
STAGE1_PREAGG_CAPTION = (
    "Row-level Stage 1 noise-classifier accuracy (probability ≥ 0.5 vs noise_cat) on wide "
    "animal×frequency strata (SPL 50–80 pivots). Panel A: Liberman experimental groups "
    "(abbreviations spelled out); panel B: Brad/Buran exposure groups (young, aged, "
    "acute noise exp., aged noise exp.). Brad 5.7→5.6 and 45.3→45.2 kHz merged. Selected wide "
    "RF/LR model per cohort. Fit rows are in-sample; Validate and Test included. "
    "Cell labels show accuracy and n strata."
)
STAGE1_PREAGG_CAPTION_TEST = (
    "Row-level Stage 1 noise-classifier accuracy (probability ≥ 0.5 vs noise_cat) on wide "
    "animal×frequency strata (SPL 50–80 pivots), held-out Test animals only. "
    "Panel A: Liberman experimental groups (abbreviations spelled out); panel B: Brad/Buran "
    "exposure groups (young, aged, acute noise exp., aged noise exp.). "
    "Brad 5.7→5.6 and 45.3→45.2 kHz merged. Selected wide RF/LR model per cohort "
    "(within-cohort Train fit). Cell labels show accuracy and n strata."
)
ROW_EVAL_PROTOCOL = (
    "Selected model; row proba>=0.5 vs noise_cat; Fit+Validate+Test wide rows; "
    "noise_group=strip(experimental_group); frequency 5.7→5.6 and 45.3→45.2 kHz merged"
)
ROW_EVAL_PROTOCOL_TEST = (
    "Selected model; row proba>=0.5 vs noise_cat; Test wide rows only (held-out animals); "
    "noise_group=strip(experimental_group); frequency 5.7→5.6 and 45.3→45.2 kHz merged"
)


def _s1_animal_eval_rows(lab: str, model: str, s1_out: dict) -> pd.DataFrame:
    y = s1_out["animal_y_te"]
    s = s1_out["animal_prob_te"]
    t = float(s1_out["stage1_threshold"])
    scores = s.reindex(y.index).values.astype(np.float64)
    return pd.DataFrame(
        {
            "lab": lab,
            "model": model,
            "animal_id": y.index.astype(str),
            "y_true": y.values.astype(np.int8),
            "score": scores,
            "threshold": t,
            "y_pred": (scores > t).astype(np.int8),
        }
    )


def _metrics_block(s1: dict) -> dict:
    return {
        "fit_acc_row": float(s1["s1_fit_acc_row"]),
        "fit_acc_animal": float(s1["s1_fit_acc_animal"]),
        "val_acc_pre": float(s1["s1_val_acc_pre"]),
        "val_acc_animal": float(s1["s1_val_acc_animal"]),
        "val_acc_animal_val_youden": float(s1["s1_val_acc_animal_val_youden"]),
        "non_test_acc_animal": float(s1["s1_non_test_acc_animal"]),
        "non_test_auc_animal": float(s1["s1_non_test_auc_animal"]),
        "non_test_label_protocol": "animal-level (mean row proba → threshold → broadcast)",
        "threshold_youden": float(s1["stage1_threshold"]),
        "threshold_youden_val_only": float(s1["stage1_threshold_val_youden"]),
        "test_acc_post": float(s1["s1_test_acc"]),
        "test_acc_post_0p5": float(s1["s1_test_acc_0p5"]),
        "test_auc_post": float(s1["s1_test_auc"]),
    }


def stage1_summary_table(candidates: Mapping[str, dict], selected: str) -> pd.DataFrame:
    """RF / LR / selected: fit/val/non-test/test metrics at calibrated threshold."""
    rows = []
    for name, s1 in candidates.items():
        rows.append(
            {
                "model": name.upper(),
                "fit_acc_row": float(s1["s1_fit_acc_row"]),
                "fit_acc_animal": float(s1["s1_fit_acc_animal"]),
                "val_acc_row": float(s1["s1_val_acc_pre"]),
                "val_acc_animal": float(s1["s1_val_acc_animal"]),
                "val_acc_animal_val_thr": float(s1["s1_val_acc_animal_val_youden"]),
                "non_test_acc_animal": float(s1["s1_non_test_acc_animal"]),
                "non_test_auc_animal": float(s1["s1_non_test_auc_animal"]),
                "threshold_fit_val_youden": float(s1["stage1_threshold"]),
                "threshold_val_only_youden": float(s1["stage1_threshold_val_youden"]),
                "test_acc_animal": float(s1["s1_test_acc"]),
                "test_acc_animal_0p5": float(s1["s1_test_acc_0p5"]),
                "test_auc_animal": float(s1["s1_test_auc"]),
                "selected": name == selected,
            }
        )
    return pd.DataFrame(rows)


def export_stage1_artifacts(
    bb_best: dict,
    lib_best: dict,
    bb_rf: dict,
    bb_lr: dict,
    lib_rf: dict,
    lib_lr: dict,
    *,
    cache_dir: Path | str | None = None,
    sklearn_version: str = "",
) -> dict[str, Path]:
    cache = Path(cache_dir or repo_root() / "figures" / "cache")
    cache.mkdir(parents=True, exist_ok=True)

    metrics = {
        "Brad": {
            "selected_model": bb_best["stage1_model"],
            **_metrics_block(bb_best),
            "rf": _metrics_block(bb_rf),
            "lr": _metrics_block(bb_lr),
        },
        "Lib": {
            "selected_model": lib_best["stage1_model"],
            **_metrics_block(lib_best),
            "rf": _metrics_block(lib_rf),
            "lr": _metrics_block(lib_lr),
        },
    }
    metrics_path = cache / "stage1_wide_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    shim = {
        "Brad": (
            metrics["Brad"]["val_acc_pre"],
            metrics["Brad"]["test_acc_post"],
            metrics["Brad"]["test_auc_post"],
        ),
        "Lib": (
            metrics["Lib"]["val_acc_pre"],
            metrics["Lib"]["test_acc_post"],
            metrics["Lib"]["test_auc_post"],
        ),
    }
    shim_path = cache / "stage1_wide_rf_metrics.json"
    with shim_path.open("w") as f:
        json.dump({k: list(v) for k, v in shim.items()}, f, indent=2)

    eval_df = pd.concat(
        [
            _s1_animal_eval_rows("Brad", "LR", bb_lr),
            _s1_animal_eval_rows("Brad", "RF", bb_rf),
            _s1_animal_eval_rows("Liberman", "LR", lib_lr),
            _s1_animal_eval_rows("Liberman", "RF", lib_rf),
        ],
        ignore_index=True,
    )
    eval_path = cache / "stage1_wide_noise_lr_rf_eval.parquet"
    eval_df.to_parquet(eval_path, index=False)

    meta = {
        "sklearn_version": sklearn_version,
        "protocol": (
            "Train fit / Validate row-acc HP / "
            "Youden J threshold on Fit+Validate animals (full calibration) / "
            "Test animal acc+AUC at calibrated threshold"
        ),
        "Brad_selected": bb_best["stage1_model"],
        "Lib_selected": lib_best["stage1_model"],
    }
    meta_path = cache / "stage1_wide_noise_lr_rf_meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return {
        "metrics": metrics_path,
        "shim": shim_path,
        "eval_parquet": eval_path,
        "meta": meta_path,
    }


def load_stage1_metrics(
    cache_dir: Path | str | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Return {Brad|Lib: (val_acc_pre, test_acc_post, test_auc_post)} for bar charts."""
    cache = Path(cache_dir or repo_root() / "figures" / "cache")
    path = cache / "stage1_wide_metrics.json"
    if path.is_file():
        raw = json.loads(path.read_text())
        out = {}
        for lab in ("Brad", "Lib"):
            block = raw[lab]
            out[lab] = (
                float(block["val_acc_pre"]),
                float(block["test_acc_post"]),
                float(block["test_auc_post"]),
            )
        return out
    shim = cache / "stage1_wide_rf_metrics.json"
    if shim.is_file():
        raw = json.loads(shim.read_text())
        return {k: tuple(float(x) for x in v) for k, v in raw.items()}
    return {}


def _canonicalize_heatmap_frequency(freq: float) -> float:
    f = float(freq)
    return _HEATMAP_FREQ_ALIAS.get(f, f)


def _with_canonical_frequency(row_df: pd.DataFrame) -> pd.DataFrame:
    out = row_df.copy()
    out["frequency"] = out["frequency"].astype(float).map(_canonicalize_heatmap_frequency)
    return out


def _heatmap_panel_letter(fig: plt.Figure, ax: plt.Axes, letter: str, *, fontsize: float = 12) -> None:
    """Panel ID above subplot top edge (left margin; avoids y-label overlap)."""
    pos = ax.get_position()
    fig.text(
        max(0.018, pos.x0 - 0.018),
        pos.y1 + 0.004,
        letter,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="right",
        clip_on=False,
    )


def _style_heatmap_colorbar(cbar) -> None:
    """Vertical label and ticks on the right; needs PRINT_HEATMAP_RIGHT_PAD_IN."""
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    cbar.set_label("Pre-aggregation accuracy", fontsize=9, labelpad=8)
    cbar.ax.tick_params(labelsize=8, pad=2, labelleft=False, labelright=True)


def _apply_print_heatmap_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
        }
    )


def _print_heatmap_core_width(n_cols: int, y_labels: Sequence[str]) -> float:
    """Heatmap content width before right padding for colorbar ticks."""
    max_label = max((len(str(s)) for s in y_labels), default=24)
    return max(6.8, 0.52 * max(n_cols, 1) + 1.95 + 0.048 * max_label)


def _print_heatmap_figsize(n_cols: int, y_labels: Sequence[str]) -> tuple[float, float, float]:
    """Fixed 9 in height; core width plus right pad for colorbar tick labels."""
    core_w = _print_heatmap_core_width(n_cols, y_labels)
    fig_w = core_w + PRINT_HEATMAP_RIGHT_PAD_IN
    return fig_w, PRINT_HEATMAP_HEIGHT_IN, core_w


def _heatmap_left_margin(y_labels: Sequence[str], *, fig_w: float) -> float:
    """Fraction of figure width for y tick labels (8 pt; longest Liberman labels)."""
    max_label = max((len(str(s)) for s in y_labels), default=20)
    inches = 0.058 * max_label + 0.11
    return float(np.clip(inches / max(fig_w, 1.0), 0.17, 0.24))


def _heatmap_gridspec_right(*, core_w: float, fig_w: float) -> float:
    """Keep heatmap block width stable when adding right-side padding."""
    return min(0.97, (0.97 * core_w) / fig_w)


def _liberman_group_display_label(raw: str) -> str:
    key = str(raw).strip()
    if key in _LIBERMAN_GROUP_DISPLAY:
        return _LIBERMAN_GROUP_DISPLAY[key]
    return key


def _brad_tx_display_label(raw: str) -> str:
    return BRAD_TX_DISPLAY.get(str(raw).strip(), str(raw).strip())


def _heatmap_yticklabels(groups: Sequence[str], *, lab: str) -> list[str]:
    if lab == "Brad":
        return [_brad_tx_display_label(g) for g in groups]
    return [_liberman_group_display_label(g) for g in groups]


def _normalize_experimental_group(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def build_stage1_row_eval_df(
    wide_df: pd.DataFrame,
    s1: dict,
    feat_cols: Sequence[str],
    *,
    lab: str,
) -> pd.DataFrame:
    """Row-level Stage-1 eval (pre-aggregation, threshold 0.5) on evaluable wide rows."""
    if "experimental_group" not in wide_df.columns:
        raise KeyError("wide_df missing experimental_group")
    clf = s1["clf"]
    model = str(s1["stage1_model"]).upper()
    X = wide_df[list(feat_cols)].dropna()
    if len(X) == 0:
        raise ValueError(f"{lab}: no rows with complete Stage-1 features")

    proba = pd.Series(clf.predict_proba(X)[:, 1], index=X.index, dtype=float)
    y_true = wide_df.loc[X.index, "noise_cat"].round().astype(int)
    y_pred = (proba.values >= STAGE1_ROW_THRESHOLD).astype(np.int8)

    meta = wide_df.loc[
        X.index, ["animal_id", "frequency", "experimental_group", "DataGroup"]
    ].copy()
    meta["noise_group"] = _normalize_experimental_group(meta["experimental_group"])
    meta = meta.drop(columns=["experimental_group"])

    out = meta.assign(
        lab=lab,
        y_true=y_true.values,
        proba=proba.values,
        y_pred=y_pred,
        correct=(y_true.values == y_pred).astype(np.int8),
        model=model,
        row_threshold=STAGE1_ROW_THRESHOLD,
    )
    out["animal_id"] = out["animal_id"].astype(str)
    return out.reset_index(drop=True)


def stage1_row_accuracy_grid(
    row_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot row eval to accuracy and count matrices (noise_group × frequency)."""
    df = _with_canonical_frequency(row_df)
    g = (
        df.groupby(["noise_group", "frequency"], dropna=False)["correct"]
        .agg(accuracy="mean", n="count")
        .reset_index()
    )
    acc = g.pivot(index="noise_group", columns="frequency", values="accuracy")
    counts = g.pivot(index="noise_group", columns="frequency", values="n")
    return acc, counts


def _grid_long_from_row_eval(row_df: pd.DataFrame, *, lab: str) -> pd.DataFrame:
    acc, counts = stage1_row_accuracy_grid(row_df)
    rows = []
    for grp in acc.index:
        for freq in acc.columns:
            n = int(counts.loc[grp, freq]) if pd.notna(counts.loc[grp, freq]) else 0
            if n == 0:
                continue
            rows.append(
                {
                    "lab": lab,
                    "noise_group": grp,
                    "frequency": float(freq),
                    "accuracy": float(acc.loc[grp, freq]),
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def _sorted_frequency_columns(cols: pd.Index) -> list[float]:
    return sorted(float(c) for c in cols)


def _order_noise_groups(
    groups: Sequence[str],
    *,
    lab: str,
) -> list[str]:
    uniq = list(dict.fromkeys(str(g) for g in groups))
    if lab == "Brad":
        order = {g: i for i, g in enumerate(BRAD_TX_ORDER)}
        return sorted(uniq, key=lambda g: (order.get(g, 99), g))
    order = {g: i for i, g in enumerate(LIBERMAN_GROUP_ORDER)}
    return sorted(uniq, key=lambda g: (order.get(g, 99), g))


def _heatmap_annot_matrix(
    acc: pd.DataFrame,
    counts: pd.DataFrame,
) -> np.ndarray:
    annot = np.empty(acc.shape, dtype=object)
    for i, grp in enumerate(acc.index):
        for j, freq in enumerate(acc.columns):
            n = counts.loc[grp, freq] if grp in counts.index and freq in counts.columns else np.nan
            a = acc.loc[grp, freq]
            if pd.isna(n) or int(n) == 0 or pd.isna(a):
                annot[i, j] = ""
            else:
                annot[i, j] = f"{float(a):.0%}\n(n={int(n)})"
    return annot


def _save_presentation_png_svg(
    fig: plt.Figure, stem: str, *, dpi: int = PRINT_HEATMAP_DPI
) -> tuple[Path, Path]:
    out_dir = repo_root() / "figures" / "presentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png, svg


def plot_stage1_row_accuracy_heatmap(
    lib_acc: pd.DataFrame,
    lib_counts: pd.DataFrame,
    brad_acc: pd.DataFrame,
    brad_counts: pd.DataFrame,
    *,
    out_stem: str = STAGE1_PREAGG_HEATMAP_STEM,
) -> tuple[Path, Path]:
    """Stacked print heatmap: Liberman (A, top) over Brad (B, bottom); frequency on x."""
    _apply_print_heatmap_rcparams()

    lib_acc = lib_acc.copy()
    brad_acc = brad_acc.copy()
    lib_counts = lib_counts.copy()
    brad_counts = brad_counts.copy()

    lib_order = _order_noise_groups(lib_acc.index, lab="Liberman")
    brad_order = _order_noise_groups(brad_acc.index, lab="Brad")
    lib_acc = lib_acc.reindex(lib_order)
    brad_acc = brad_acc.reindex(brad_order)
    lib_counts = lib_counts.reindex(index=lib_acc.index, columns=lib_acc.columns)
    brad_counts = brad_counts.reindex(index=brad_acc.index, columns=brad_acc.columns)

    all_freqs = _sorted_frequency_columns(lib_acc.columns.union(brad_acc.columns))
    lib_acc = lib_acc.reindex(columns=all_freqs)
    lib_counts = lib_counts.reindex(columns=all_freqs)
    brad_acc = brad_acc.reindex(columns=all_freqs)
    brad_counts = brad_counts.reindex(columns=all_freqs)

    freq_labels = [f"{f:g}" for f in all_freqs]
    n_cols = len(all_freqs)
    n_lib = len(lib_acc)
    n_brad = len(brad_acc)
    lib_y_labels = _heatmap_yticklabels(lib_acc.index, lab="Liberman")
    brad_y_labels = _heatmap_yticklabels(brad_acc.index, lab="Brad")
    fig_w, fig_h, core_w = _print_heatmap_figsize(
        n_cols, list(lib_y_labels) + list(brad_y_labels)
    )
    left = _heatmap_left_margin(lib_y_labels, fig_w=fig_w)
    right = _heatmap_gridspec_right(core_w=core_w, fig_w=fig_w)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[max(n_lib, 1), max(n_brad, 1)],
        width_ratios=[1.0, PRINT_HEATMAP_CBAR_WIDTH],
        hspace=0.12,
        wspace=PRINT_HEATMAP_CBAR_WSPACE,
        left=left,
        right=right,
        top=0.975,
        bottom=0.065,
    )
    ax_lib = fig.add_subplot(gs[0, 0])
    ax_brad = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[:, 1])

    panels = (
        (ax_lib, lib_acc, lib_counts, lib_y_labels, "A", False),
        (ax_brad, brad_acc, brad_counts, brad_y_labels, "B", True),
    )
    annot_kw = {"size": 7.5, "ha": "center", "va": "center"}
    for ax, acc, counts, y_labels, letter, show_xlabel in panels:
        annot = _heatmap_annot_matrix(acc, counts)
        sns.heatmap(
            acc.astype(float),
            ax=ax,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            mask=counts.fillna(0).astype(int) == 0,
            annot=annot,
            fmt="",
            annot_kws=annot_kw,
            linewidths=0.4,
            linecolor="0.90",
            cbar=False,
            xticklabels=freq_labels,
            yticklabels=y_labels,
        )
        ax.set_xlabel("Frequency (kHz)" if show_xlabel else "")
        ax.set_ylabel("")
        yticks = ax.get_yticklabels()
        plt.setp(yticks, ha="right", rotation=0, clip_on=False)
        ax.tick_params(axis="y", pad=2.5, length=0)
        if show_xlabel:
            ax.tick_params(axis="x", pad=4)
            for lbl in ax.get_xticklabels():
                lbl.set_clip_on(False)
        else:
            ax.set_xticklabels([])
        _heatmap_panel_letter(fig, ax, letter, fontsize=12)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    _style_heatmap_colorbar(cbar)

    out_dir = repo_root() / "figures" / "presentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{out_stem}.png"
    svg = out_dir / f"{out_stem}.svg"
    save_kw = dict(dpi=PRINT_HEATMAP_DPI, facecolor="white")
    fig.savefig(png, bbox_inches=None, **save_kw)
    fig.savefig(svg, bbox_inches=None, **save_kw)
    plt.close(fig)
    return png, svg


def print_stage1_preagg_heatmap_digest(
    row_eval: pd.DataFrame,
    *,
    metrics_path: Path | None = None,
    digest_title: str = "=== Stage1 preagg heatmap ===",
    check_fit_sanity: bool = True,
) -> None:
    """Print row-level accuracy summary; optional Train vs fit_acc_row sanity."""
    print(digest_title)
    cache = Path(metrics_path or repo_root() / "figures" / "cache" / "stage1_wide_metrics.json")
    metrics_raw = json.loads(cache.read_text()) if cache.is_file() else {}

    lab_to_metrics_key = {"Brad": "Brad", "Liberman": "Lib"}
    for lab in ("Liberman", "Brad"):
        sub = row_eval[row_eval["lab"].eq(lab)]
        if sub.empty:
            print(f"--- {lab}: no rows ---")
            continue
        print(f"--- {lab} --- rows={len(sub)} model={sub['model'].iloc[0]}")
        print(f"  row acc: {sub['correct'].mean():.4f}")
        for dg in ("Train", "Validate", "Test"):
            part = sub[sub["DataGroup"].eq(dg)]
            if len(part):
                print(f"  {dg} row acc: {part['correct'].mean():.4f} (n={len(part)})")
        mkey = lab_to_metrics_key[lab]
        if check_fit_sanity and mkey in metrics_raw:
            train = sub.loc[sub["DataGroup"].eq("Train"), "correct"]
            if len(train):
                expected = float(metrics_raw[mkey]["fit_acc_row"])
                train_acc = float(train.mean())
                print(f"  Train sanity vs fit_acc_row: {train_acc:.6f} vs {expected:.6f}")
    end = digest_title.replace("=== ", "").replace(" ===", "")
    print(f"=== end {end} ===")


def export_stage1_row_heatmap(
    bb_best: dict,
    lib_best: dict,
    bb_wide_all: pd.DataFrame,
    lib_wide_all: pd.DataFrame,
    bb_feat_cols: Sequence[str],
    lib_feat_cols: Sequence[str],
    *,
    cache_dir: Path | str | None = None,
    sklearn_version: str = "",
    meta_path: Path | str | None = None,
    test_only: bool = False,
) -> dict[str, Path]:
    """Export row-level eval parquet, grid summary, stacked heatmap, and caption."""
    cache = Path(cache_dir or repo_root() / "figures" / "cache")
    cache.mkdir(parents=True, exist_ok=True)

    if test_only:
        bb_pool = bb_wide_all.loc[bb_wide_all["DataGroup"] == "Test"].reset_index(drop=True)
        lib_pool = lib_wide_all.loc[lib_wide_all["DataGroup"] == "Test"].reset_index(drop=True)
        stem = STAGE1_PREAGG_HEATMAP_TEST_STEM
        caption_text = STAGE1_PREAGG_CAPTION_TEST
        row_name = "stage1_wide_row_eval_test.parquet"
        grid_name = "stage1_wide_row_eval_grid_test.parquet"
        protocol_key = "row_eval_protocol_test"
        protocol_val = ROW_EVAL_PROTOCOL_TEST
        digest_title = "=== Stage1 preagg heatmap (Test only) ==="
        check_fit_sanity = False
    else:
        bb_pool = bb_wide_all
        lib_pool = lib_wide_all
        stem = STAGE1_PREAGG_HEATMAP_STEM
        caption_text = STAGE1_PREAGG_CAPTION
        row_name = "stage1_wide_row_eval.parquet"
        grid_name = "stage1_wide_row_eval_grid.parquet"
        protocol_key = "row_eval_protocol"
        protocol_val = ROW_EVAL_PROTOCOL
        digest_title = "=== Stage1 preagg heatmap ==="
        check_fit_sanity = True

    bb_rows = build_stage1_row_eval_df(bb_pool, bb_best, bb_feat_cols, lab="Brad")
    lib_rows = build_stage1_row_eval_df(lib_pool, lib_best, lib_feat_cols, lab="Liberman")
    row_eval = pd.concat([lib_rows, bb_rows], ignore_index=True)

    row_path = cache / row_name
    row_eval.to_parquet(row_path, index=False)

    grid = pd.concat(
        [
            _grid_long_from_row_eval(lib_rows, lab="Liberman"),
            _grid_long_from_row_eval(bb_rows, lab="Brad"),
        ],
        ignore_index=True,
    )
    grid_path = cache / grid_name
    grid.to_parquet(grid_path, index=False)

    lib_acc, lib_counts = stage1_row_accuracy_grid(lib_rows)
    brad_acc, brad_counts = stage1_row_accuracy_grid(bb_rows)
    png, svg = plot_stage1_row_accuracy_heatmap(
        lib_acc, lib_counts, brad_acc, brad_counts, out_stem=stem
    )

    caption_path = png.parent / f"{stem}_caption.txt"
    caption_path.write_text(caption_text + "\n")

    meta_file = Path(meta_path or cache / "stage1_wide_noise_lr_rf_meta.json")
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text())
    else:
        meta = {}
    meta[protocol_key] = protocol_val
    if sklearn_version:
        meta["sklearn_version"] = sklearn_version
    with meta_file.open("w") as f:
        json.dump(meta, f, indent=2)

    print_stage1_preagg_heatmap_digest(
        row_eval,
        check_fit_sanity=check_fit_sanity,
        digest_title=digest_title,
    )
    print("Saved", png.resolve(), "+", svg.resolve())
    print("Saved", row_path.resolve())
    print("Saved", grid_path.resolve())
    print("Saved", caption_path.resolve())

    prefix = "heatmap_test" if test_only else "heatmap"
    return {
        f"row_eval_{prefix}" if test_only else "row_eval": row_path,
        f"row_eval_grid_{prefix}" if test_only else "row_eval_grid": grid_path,
        f"{prefix}_png": png,
        f"{prefix}_svg": svg,
        f"caption_{prefix}" if test_only else "caption": caption_path,
    }

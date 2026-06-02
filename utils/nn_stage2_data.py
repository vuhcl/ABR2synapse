"""
Build Brad / Liberman long and wide tables for the NN Stage-2 pipeline.

Mirrors ``abr_wide_long_comparison_newfeats.ipynb`` without even-SPL pivots; by default
``load_nn_stage2_data()`` omits the newfeats-only extras in Stage 1 and Stage 2 (see
``include_extra_wide_features``).
Liberman long rows keep ``WaveI`` (30-sample window) and ``full_waveform`` (full trace,
0–17 ms timebase). The ``cnn_full`` branch uses 0–8 ms with a common resampled length.
"""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.model_selection import train_test_split

from utils.data_loader import load_data
from utils.feature_extraction import (amplitude_vs_spl_slopes,
                                      slope_from_amplitude_distance)

warnings.filterwarnings("ignore")

WAVE_I_LEN = 30

# Full-wave CNN: Liberman trace spans 0–17 ms; both cohorts use 0–8 ms window, then resample.
LIB_FULL_WAVEFORM_MS = 17.0
FULL_WAVE_WINDOW_MS = (0.0, 8.0)

MOUSE_SPLIT_RANDOM_STATE = 22
NN_TRAIN_RANDOM_STATE = 1
STAGE_SPL_LEVELS: tuple[int, ...] = (50, 60, 70, 80)

_WIDE_SPL_LEVEL_SUFFIX_RE = re.compile(r"_(\d+(?:\.\d+)?)$")


# Long-format Stage 2 tabular numeric columns
LONG_NUM_BASE = [
    "amplitude",
    "slope",
    "distance",
    "level",
]
# With ``include_extra_wide_features=True``, append the same three fields used in Stage 1 extras:
LONG_NUM = LONG_NUM_BASE + ["p1_latency", "Slope_all", "Slope_high4"]
LONG_LOG = [
    "frequency",
    "total_variance",
    "PeakIEarlyCurvature",
    "PeakICentralCurvature",
    "PeakILateCurvature",
    "TroughIEarlyCurvature",
    "TroughICentralCurvature",
    "TroughILateCurvature",
]
LONG_CAT = ["noise_preds"]

_PIVOT_COLS = [
    "amplitude",
    "slope",
    "distance",
    "total_variance",
    "PeakIEarlyCurvature",
    "PeakICentralCurvature",
    "PeakILateCurvature",
    "TroughIEarlyCurvature",
    "TroughICentralCurvature",
    "TroughILateCurvature",
]

_EXTRA_WIDE_FEATURES = ("Slope_all", "Slope_high4", "p1_latency")

_WIDE_S2_LOG_BASES: tuple[str, ...] = (
    "total_variance",
    "PeakIEarlyCurvature",
    "PeakICentralCurvature",
    "PeakILateCurvature",
    "TroughIEarlyCurvature",
    "TroughICentralCurvature",
    "TroughILateCurvature",
)

_PIVOT_NUM_BASES: tuple[str, ...] = ("amplitude", "slope", "distance")


def parse_wide_spl_level(col: str) -> float | None:
    """Parse trailing ``_50.0`` style level from a pivoted wide column name."""
    m = _WIDE_SPL_LEVEL_SUFFIX_RE.search(col)
    if not m:
        return None
    return float(m.group(1))


def wide_columns_at_stage_spl(
    columns,
    bases: tuple[str, ...] = _PIVOT_NUM_BASES + _WIDE_S2_LOG_BASES,
    *,
    levels: tuple[int, ...] = STAGE_SPL_LEVELS,
) -> list[str]:
    """Pivoted columns whose parsed dB level is in ``levels``."""
    allowed = {float(x) for x in levels}
    out = []
    for c in columns:
        for base in bases:
            if not c.startswith(f"{base}_"):
                continue
            lvl = parse_wide_spl_level(c)
            if lvl is not None and lvl in allowed:
                out.append(c)
                break
    return sorted(set(out))


def wide_stage1_fit(wide_df: pd.DataFrame) -> pd.DataFrame:
    return wide_df.loc[wide_df["DataGroup"] == "Train"].reset_index(drop=True)


def wide_stage1_val(wide_df: pd.DataFrame) -> pd.DataFrame:
    return wide_df.loc[wide_df["DataGroup"] == "Validate"].reset_index(drop=True)


def assert_wide_feats_stage_spl(
    columns: list[str] | tuple[str, ...],
    *,
    levels: tuple[int, ...] = STAGE_SPL_LEVELS,
) -> None:
    """Raise if any pivoted wide feature column parses to an SPL outside ``levels``."""
    allowed = {float(x) for x in levels}
    for col in columns:
        lvl = parse_wide_spl_level(col)
        if lvl is not None and lvl not in allowed:
            raise ValueError(
                f"Tree wide feature {col!r} uses SPL {lvl}; allowed {levels}"
            )


def syn_feats_from_wide_common(
    common_cols: list | tuple,
    *,
    has_strain: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """
    Stage-2 **wide** tabular features (same logic as ``abr_wide_long_comparison``):
    pivoted ``amplitude`` / ``slope`` / ``distance`` (numeric) and curvature / variance
    columns (log-scaled) on the 50/60/70/80 dB grid, plus ``frequency`` and ``noise_preds``.
    Optional extras present in both labs' wide tables: ``Slope_all``, ``Slope_high4``,
    ``p1_latency``; optional ``strain_binary``.
    """
    common = list(common_cols)
    spl_cols = set(wide_columns_at_stage_spl(common))
    syn_num: list[str] = []
    syn_log: list[str] = ["frequency"]
    syn_cat: list[str] = ["noise_preds"]
    for base in _WIDE_S2_LOG_BASES:
        syn_log += sorted(c for c in common if c in spl_cols and c.startswith(f"{base}_"))
    for base in _PIVOT_NUM_BASES:
        syn_num += sorted(c for c in common if c in spl_cols and c.startswith(f"{base}_"))
    for extra in _EXTRA_WIDE_FEATURES:
        if extra in common:
            syn_num.append(extra)
    if has_strain and "strain_binary" in common:
        syn_num.append("strain_binary")
    syn_num = sorted(set(syn_num))
    return syn_num, syn_log, syn_cat


def split_by_mouse(
    data,
    split_on="animal_id",
    stratify_on="tx",
    test_size=0.2,
    val_size=0.18,
    random_state=MOUSE_SPLIT_RANDOM_STATE,
    return_idx=False,
):
    mice = data[[split_on, stratify_on]].drop_duplicates().set_index(split_on)

    train, test = train_test_split(
        mice.index,
        test_size=test_size,
        shuffle=True,
        stratify=mice[stratify_on],
        random_state=random_state,
    )

    train_indices = data[split_on].isin(train)
    test_indices = data[split_on].isin(test)

    train2, val = train_test_split(
        train,
        test_size=val_size,
        shuffle=True,
        stratify=mice.loc[train][stratify_on],
        random_state=random_state,
    )

    train2_indices = data[split_on].isin(train2)
    val_indices = data[split_on].isin(val)

    data["DataGroup"] = ""
    data.loc[train2_indices, "DataGroup"] = "Train"
    data.loc[val_indices, "DataGroup"] = "Validate"
    data.loc[test_indices, "DataGroup"] = "Test"
    if return_idx:
        return data, train_indices, train2_indices, val_indices, test_indices
    return data


def _noise_feats_from_wide(columns, *, include_extra_wide: bool = False):
    """
    Wide columns used by the Stage 1 noise RF.

    ``include_extra_wide``: when True, add animal×frequency aggregates
    ``Slope_all``, ``Slope_high4``, ``p1_latency`` (same as ``newfeats`` notebook).
    """
    cols = list(columns)
    spl_cols = set(wide_columns_at_stage_spl(cols))
    num, log = [], ["frequency"]
    for col in _WIDE_S2_LOG_BASES:
        log += sorted(c for c in cols if c in spl_cols and c.startswith(f"{col}_"))
    for col in _PIVOT_NUM_BASES:
        num += sorted(c for c in cols if c in spl_cols and c.startswith(f"{col}_"))
    if include_extra_wide:
        for c in _EXTRA_WIDE_FEATURES:
            if c in cols:
                num.append(c)
    return num, log


def brad_wave_i_fixed(waveform, p1_index) -> np.ndarray:
    """P1-centered window ``p1-10 : p1+20``, zero-pad to 30 (matches Liberman ``WaveI``)."""
    wf = np.asarray(waveform, dtype=np.float64).ravel()
    p1 = int(np.clip(int(p1_index), 0, max(len(wf) - 1, 0)))
    lo, hi = p1 - 10, p1 + 20
    out = np.zeros(WAVE_I_LEN, dtype=np.float64)
    wlo, whi = max(0, lo), min(len(wf), hi)
    if wlo < whi:
        olo = wlo - lo
        seg = olo + (whi - wlo)
        out[olo:seg] = wf[wlo:whi]
    return out


def liberman_wave_i_fixed(row) -> np.ndarray:
    w = np.asarray(row["WaveI"], dtype=np.float64).ravel()
    out = np.zeros(WAVE_I_LEN, dtype=np.float64)
    n = min(WAVE_I_LEN, w.size)
    out[:n] = w[:n]
    return out


def _wave_i_value_usable(val) -> bool:
    """True if Liberman-style ``WaveI`` is present and has numeric samples."""
    if val is None:
        return False
    try:
        if val is pd.NA or pd.isna(val):
            return False
    except (ValueError, TypeError):
        pass
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return False
    arr = np.asarray(val, dtype=np.float64).ravel()
    if arr.size == 0:
        return False
    return bool(np.any(np.isfinite(arr)))


def _waveform_usable(wf) -> bool:
    """True if Brad-style full waveform is present (not a concat placeholder NaN)."""
    if wf is None:
        return False
    try:
        if wf is pd.NA or pd.isna(wf):
            return False
    except (ValueError, TypeError):
        pass
    if isinstance(wf, (float, np.floating)) and np.isnan(wf):
        return False
    arr = np.asarray(wf, dtype=np.float64).ravel()
    return arr.size > 0


def wave_i_for_long_row(row: pd.Series) -> np.ndarray:
    """
    Liberman long rows use ``WaveI``; Brad uses ``waveform`` + ``p1_index``.
    After ``pd.concat``, Brad rows often have ``WaveI`` = NaN (use waveform). Lib rows
    may have ``waveform`` = NaN only (use ``WaveI`` or zeros if missing).
    """
    if "WaveI" in row.index and _wave_i_value_usable(row["WaveI"]):
        return liberman_wave_i_fixed(row)
    if "waveform" in row.index and _waveform_usable(row["waveform"]):
        p1 = row["p1_index"] if "p1_index" in row.index else 0
        return brad_wave_i_fixed(row["waveform"], p1)
    return np.zeros(WAVE_I_LEN, dtype=np.float64)


def wave_i_matrix(df: pd.DataFrame, index: pd.Index) -> np.ndarray:
    """Shape (n, 30) aligned with ``df.loc[index]`` row order."""
    rows = df.loc[index]
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T
    return np.stack([wave_i_for_long_row(rows.loc[i]) for i in index], axis=0).astype(
        np.float32
    )


def _brad_x_ms_array(row: pd.Series) -> np.ndarray:
    """Time axis in ms; Brad stores ``x`` as a list whose first element is the grid."""
    x = row["x"]
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return np.asarray(x[0], dtype=np.float64).ravel()
    return np.asarray(x, dtype=np.float64).ravel()


def _crop_waveform_ms(t_ms: np.ndarray, y: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Return samples whose times fall in ``[t0, t1]`` ms (inclusive)."""
    t_ms = np.asarray(t_ms, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(t_ms.size, y.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    t_ms, y = t_ms[:n], y[:n]
    m = (t_ms >= t0) & (t_ms <= t1)
    return y[m]


def _resample_1d_to_len(y: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample amplitude along normalized arclength to ``target_len`` points."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if y.size == 0:
        return np.zeros(target_len, dtype=np.float64)
    if y.size == target_len:
        return y.astype(np.float64)
    x_old = np.linspace(0.0, 1.0, y.size)
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, y).astype(np.float64)


def crop_brad_waveform_0_8ms(row: pd.Series) -> np.ndarray:
    """Brad: ``x`` in ms (may start below 0); keep only ``[0, 8]`` ms."""
    if "waveform" not in row.index or not _waveform_usable(row.get("waveform")):
        return np.zeros(0, dtype=np.float64)
    t_ms = _brad_x_ms_array(row)
    y = np.asarray(row["waveform"], dtype=np.float64).ravel()
    t0, t1 = FULL_WAVE_WINDOW_MS
    return _crop_waveform_ms(t_ms, y, t0, t1)


def crop_lib_waveform_0_8ms(row: pd.Series) -> np.ndarray:
    """Liberman: timebase ``0 … 17`` ms over the full ``full_waveform``; keep ``[0, 8]`` ms."""
    col = "full_waveform"
    if col not in row.index or not _waveform_usable(row.get(col)):
        return np.zeros(0, dtype=np.float64)
    y = np.asarray(row[col], dtype=np.float64).ravel()
    t_ms = np.linspace(0.0, LIB_FULL_WAVEFORM_MS, len(y), dtype=np.float64)
    t0, t1 = FULL_WAVE_WINDOW_MS
    return _crop_waveform_ms(t_ms, y, t0, t1)


def compute_full_wave_target_len(
    brad_df: pd.DataFrame,
    lib_df: pd.DataFrame,
    *,
    max_sample_rows: int = 4000,
    prefer_liberman_len: bool = True,
) -> tuple[int, int, int]:
    """
    Median sample counts after 0–8 ms crop for each cohort. If they differ, return
    Liberman's median as ``target_len`` when ``prefer_liberman_len`` (else Brad's).
    All rows are then resampled to ``target_len`` so CNN inputs match.
    """
    bb_lens, ll_lens = [], []
    n_bb = min(max_sample_rows, len(brad_df))
    n_ll = min(max_sample_rows, len(lib_df))
    for _, row in brad_df.iloc[:n_bb].iterrows():
        seg = crop_brad_waveform_0_8ms(row)
        if seg.size > 0:
            bb_lens.append(seg.size)
    for _, row in lib_df.iloc[:n_ll].iterrows():
        seg = crop_lib_waveform_0_8ms(row)
        if seg.size > 0:
            ll_lens.append(seg.size)

    med_bb = int(np.median(bb_lens)) if bb_lens else 1
    med_lib = int(np.median(ll_lens)) if ll_lens else 1
    med_bb = max(1, med_bb)
    med_lib = max(1, med_lib)

    if med_bb == med_lib:
        target = med_bb
    elif prefer_liberman_len:
        target = med_lib
    else:
        target = med_bb

    return target, med_bb, med_lib


def full_wave_0_8_for_row(row: pd.Series, target_len: int) -> np.ndarray:
    """
    0–8 ms segment, then resample to ``target_len``. Brad uses ``x`` + ``waveform``;
    Liberman uses ``full_waveform`` with ``0–17`` ms timebase.
    """
    if "full_waveform" in row.index and _waveform_usable(row.get("full_waveform")):
        seg = crop_lib_waveform_0_8ms(row)
    elif "waveform" in row.index and _waveform_usable(row.get("waveform")):
        seg = crop_brad_waveform_0_8ms(row)
    else:
        seg = np.zeros(0, dtype=np.float64)
    return _resample_1d_to_len(seg, target_len)


def full_wave_matrix(df: pd.DataFrame, index: pd.Index, target_len: int) -> np.ndarray:
    """Shape (n, target_len) for the 0–8 ms full-wave CNN branch.

    ``index`` must be row labels that belong to this ``df`` (e.g. ``sub.index`` where
    ``sub = df.iloc[:k]``), not labels from another frame after ``reset_index`` —
    otherwise ``df.loc`` selects the wrong rows.
    """
    rows = df.loc[index]
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T
    return np.stack(
        [full_wave_0_8_for_row(rows.loc[i], target_len) for i in index], axis=0
    ).astype(np.float32)


def unified_grid_wave_i_for_row(row: pd.Series, target_len: int) -> np.ndarray:
    """
    **Wave I CNN input on a common sampling grid:** take the 0–8 ms segment, resample to
    ``target_len``, then slice **30 samples** centred at ``p1_latency`` (ms) on the
    uniform ``0 … 8`` ms grid (same logic as native Wave I span, but after cross-lab
    alignment). Edge-padded with zeros.
    """
    y = full_wave_0_8_for_row(row, target_len)
    if target_len < 1:
        return np.zeros(WAVE_I_LEN, dtype=np.float64)
    t0, t1 = FULL_WAVE_WINDOW_MS
    t = np.linspace(t0, t1, target_len, dtype=np.float64)
    p1 = row["p1_latency"] if "p1_latency" in row.index else np.nan
    try:
        p1f = float(p1)
    except (TypeError, ValueError):
        p1f = np.nan
    if not np.isfinite(p1f):
        p1f = 0.5 * (t0 + t1)
    p1c = float(np.clip(p1f, t0, t1))
    ci = int(np.argmin(np.abs(t - p1c)))
    lo, hi = ci - 10, ci + 20
    out = np.zeros(WAVE_I_LEN, dtype=np.float64)
    for j in range(WAVE_I_LEN):
        k = lo + j
        if 0 <= k < target_len:
            out[j] = y[k]
    return out


def unified_grid_wave_i_matrix(
    df: pd.DataFrame, index: pd.Index, target_len: int
) -> np.ndarray:
    """Shape (n, 30) — Wave I window on the unified 0–8 ms resampled grid.

    Same indexing rule as ``full_wave_matrix``: ``index`` labels must exist on ``df``.
    """
    rows = df.loc[index]
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T
    return np.stack(
        [unified_grid_wave_i_for_row(rows.loc[i], target_len) for i in index], axis=0
    ).astype(np.float32)


def _calculate_variance(waveI):
    return np.sum((waveI - np.mean(waveI)) ** 2)


def _calculate_curvature(wave, point, x_axis):
    idx = range(point - 10, point + 11)
    x = x_axis[idx]
    y = wave[idx]
    cs = CubicSpline(x, y)
    values = x_axis[point - 3:point + 4:3]
    return np.abs(cs(values, 2)) / (1 + cs(values, 1) ** 2) ** (3 / 2)


def _io_slope_lookup(df, level_col="level", amp_col="amplitude"):
    rows = []
    for (animal_id, frequency), g in df.groupby(["animal_id", "frequency"]):
        g = g.sort_values(level_col)
        sa, sh = amplitude_vs_spl_slopes(g[level_col].values, g[amp_col].values)
        rows.append(
            {
                "animal_id": animal_id,
                "frequency": frequency,
                "Slope_all": sa,
                "Slope_high4": sh,
            }
        )
    return pd.DataFrame(rows)


def _build_reformatted(
    brad_buran_df: pd.DataFrame, *, join_io_features: bool = False
) -> pd.DataFrame:
    cols = list(_PIVOT_COLS)
    parts = []
    for col in cols:
        part = (
            brad_buran_df.pivot_table(
                index=["animal_id", "frequency"], columns="level", values=col
            )
            .rename_axis(columns="level_dB")
            .add_prefix(f"{col}_")
        )
        parts.append(part)
    reformatted = pd.concat(parts, axis=1)

    noise = brad_buran_df.groupby(["animal_id", "frequency"])["noise_cat"].mean()
    syn = brad_buran_df.groupby(["animal_id", "frequency"])["synapses"].mean()
    groups = brad_buran_df.groupby(["animal_id", "frequency"])[["DataGroup"]].first()
    exp_group = brad_buran_df.groupby(["animal_id", "frequency"])["tx"].first().rename(
        "experimental_group"
    )

    reformatted = reformatted.join(noise).join(syn).join(groups).join(exp_group)
    if "strain_binary" in brad_buran_df.columns:
        _sb = brad_buran_df.groupby(["animal_id", "frequency"])["strain_binary"].first()
        reformatted = reformatted.join(_sb)

    if join_io_features:
        _io_extra = brad_buran_df.groupby(["animal_id", "frequency"])[
            ["Slope_all", "Slope_high4", "p1_latency"]
        ].first()
        reformatted = reformatted.join(_io_extra)

    for col in cols:
        amp_cols = [c for c in reformatted.columns if c.startswith(col)]
        reformatted[amp_cols] = reformatted[amp_cols].interpolate(
            axis=1, limit_direction="both"
        )
    return reformatted.reset_index()


def _build_reformatted_orig(
    _orig: pd.DataFrame, *, join_io_features: bool = False
) -> pd.DataFrame:
    _dfs = [
        _orig.pivot_table(index=["animal_id", "frequency"], columns="level", values=col)
        .rename_axis(columns="level_dB")
        .add_prefix(f"{col}_")
        for col in _PIVOT_COLS
    ]
    reformatted_orig = pd.concat(_dfs, axis=1)

    reformatted_orig = reformatted_orig.join(
        _orig.groupby(["animal_id", "frequency"])["noise_cat"].mean()
    ).join(_orig.groupby(["animal_id", "frequency"])["synapses"].mean())
    if "strain_binary" in _orig.columns:
        reformatted_orig = reformatted_orig.join(
            _orig.groupby(["animal_id", "frequency"])["strain_binary"].first()
        )

    reformatted_orig = reformatted_orig.join(
        _orig.groupby(["animal_id", "frequency"])[["DataGroup"]].first()
    ).join(
        _orig.groupby(["animal_id", "frequency"])["Group"]
        .first()
        .rename("experimental_group")
    )

    if join_io_features:
        _io_lib = _orig.groupby(["animal_id", "frequency"])[
            ["Slope_all", "Slope_high4", "p1_latency"]
        ].first()
        reformatted_orig = reformatted_orig.join(_io_lib)

    for col in _PIVOT_COLS:
        _cs = [c for c in reformatted_orig.columns if c.startswith(f"{col}_")]
        reformatted_orig[_cs] = reformatted_orig[_cs].interpolate(
            axis=1, limit_direction="both"
        )

    return reformatted_orig.reset_index().dropna()


def _build_brad_prefeature(share_path: Path) -> pd.DataFrame:
    """Brad CSV + IO merge + native waveform indices / waveI slice (before LibT features)."""
    abr_share_path = share_path / "abr_data"

    data_list = []
    for fq in os.listdir(abr_share_path):
        if fq.startswith("ABR") and fq.endswith(".csv"):
            path = os.path.join(abr_share_path, fq)
            parts = path.split("_")
            animal_id = parts[2]
            n_freq = int(parts[3])
            datetime_str = parts[4]
            epochs = pd.read_csv(path, index_col=[0, 1, 2, 3])
            epochs.columns = epochs.columns.astype("f").rename("time")
            epochs_mean = epochs.groupby(["frequency", "level"]).mean()
            x = [(epochs_mean.columns.values * 1000).round(2)]
            data = (
                epochs_mean.apply(lambda x: x.values.tolist(), axis=1)
                .rename("waveform")
                .reset_index()
            )
            data["animal_id"] = animal_id
            data["n_freq"] = n_freq
            data["datetime_str"] = datetime_str
            data["strain"] = "CBA/CaJ"
            data["strain_binary"] = 0
            data["x"] = x * len(data)
            data_list.append(data)
    df = pd.concat(data_list, ignore_index=True)

    df = df[df.frequency != "click"]
    df["frequency"] = df["frequency"].astype("float")

    key = ["animal_id", "datetime_str", "frequency", "level", "n_freq"]
    df = df.set_index(key, verify_integrity=True)

    abr_io = pd.read_csv(share_path / "abr_io.csv")
    synapses_df = pd.read_csv(share_path / "synapses.csv")
    synapse_key = ["animal_id", "frequency"]

    joined_df = abr_io.set_index(key).join(df, on=key).reset_index().dropna()
    joined_df = pd.merge(
        joined_df.set_index(synapse_key),
        synapses_df,
        left_on=synapse_key,
        right_on=synapse_key,
    )

    joined_df["waveform"] = joined_df["waveform"].apply(lambda x: np.array(x))

    joined_df["p1_index"] = joined_df["p1_latency"].apply(
        lambda x: np.where(joined_df["x"][0] == x)[0][0]
    )
    joined_df["n1_index"] = joined_df["n1_latency"].apply(
        lambda x: np.where(joined_df["x"][0] == x)[0][0]
    )

    def _slice_waveI_row(x):
        waveform = np.array(x["waveform"])
        start = x["p1_index"]
        end = x["n1_index"]
        return waveform[start:end]

    joined_df["waveI"] = joined_df.apply(lambda x: _slice_waveI_row(x), axis=1)

    return joined_df


def _curvature_safe_libt(wave: np.ndarray, point: int, x_axis: np.ndarray) -> np.ndarray:
    """``_calculate_curvature`` with index clamping for short / edge rows."""
    wave = np.asarray(wave, dtype=np.float64).ravel()
    x_axis = np.asarray(x_axis, dtype=np.float64).ravel()
    n = min(wave.size, x_axis.size)
    if n < 21:
        return np.full(3, np.nan, dtype=np.float64)
    wave, x_axis = wave[:n], x_axis[:n]
    p = int(np.clip(point, 10, n - 11))
    return np.asarray(_calculate_curvature(wave, p, x_axis), dtype=np.float64)


def _apply_brad_libt_features(joined_df: pd.DataFrame, tw_lib: int) -> pd.DataFrame:
    """
    Compute Brad amplitude, slope, distance, variance, curvature, and waveI on the **LibT**
    grid: 0–8 ms crop linearly resampled to ``tw_lib`` (same target length as Liberman CNN path).
    Native ``waveform`` / ``p1_index`` / ``n1_index`` are kept for legacy indexing.
    """
    t_ms = np.linspace(
        FULL_WAVE_WINDOW_MS[0], FULL_WAVE_WINDOW_MS[1], tw_lib, dtype=np.float64
    )

    amps, dists, slopes, tvars, waveI_segs = [], [], [], [], []
    pk_rows, tr_rows = [], []

    for _, row in joined_df.iterrows():
        y = full_wave_0_8_for_row(row, tw_lib)
        if y.size == 0:
            amps.append(np.nan)
            dists.append(np.nan)
            slopes.append(np.nan)
            tvars.append(np.nan)
            waveI_segs.append(np.array([], dtype=np.float64))
            pk_rows.append(np.full(3, np.nan))
            tr_rows.append(np.full(3, np.nan))
            continue

        p1_ms = float(row["p1_latency"])
        n1_ms = float(row["n1_latency"])
        pk_i = int(np.clip(np.argmin(np.abs(t_ms - p1_ms)), 0, tw_lib - 1))
        tr_i = int(np.clip(np.argmin(np.abs(t_ms - n1_ms)), 0, tw_lib - 1))

        amp_mag = abs(float(y[pk_i]) - float(y[tr_i]))
        dist_ms = abs(float(t_ms[tr_i] - t_ms[pk_i]))

        amps.append(amp_mag)
        dists.append(dist_ms)
        slopes.append(slope_from_amplitude_distance(amp_mag, dist_ms))

        lo, hi = sorted([pk_i, tr_i])
        wseg = y[lo : hi + 1].copy()
        waveI_segs.append(wseg)
        tvars.append(float(_calculate_variance(wseg)) if wseg.size > 0 else np.nan)

        pk_rows.append(_curvature_safe_libt(y, pk_i, t_ms))
        tr_rows.append(_curvature_safe_libt(y, tr_i, t_ms))

    out = joined_df.copy()
    out["amplitude"] = amps
    out["distance"] = dists
    out["slope"] = slopes
    out["total_variance"] = tvars
    out["waveI"] = waveI_segs

    peak_curvs = pd.DataFrame(
        pk_rows,
        columns=[
            "PeakIEarlyCurvature",
            "PeakICentralCurvature",
            "PeakILateCurvature",
        ],
        index=out.index,
    )
    trough_curvs = pd.DataFrame(
        tr_rows,
        columns=[
            "TroughIEarlyCurvature",
            "TroughICentralCurvature",
            "TroughILateCurvature",
        ],
        index=out.index,
    )
    brad_buran_df = pd.concat([out, peak_curvs, trough_curvs], axis=1)

    if "strain_binary" not in brad_buran_df.columns and "strain" in brad_buran_df.columns:
        _sv = brad_buran_df["strain"].astype(str).str.strip().str.lower()
        _first = _sv.dropna().iloc[0] if _sv.notna().any() else ""
        brad_buran_df["strain_binary"] = (_sv != _first).astype(int)
    elif "strain_binary" not in brad_buran_df.columns:
        brad_buran_df["strain_binary"] = 0

    brad_buran_df["noise_cat"] = brad_buran_df["tx"].map(
        {"noise+age": 1, "noise+acute": 1, "young": 0, "age": 0}
    )

    brad_buran_df["frequency"] = brad_buran_df["frequency"].apply(
        lambda x: np.round(x / 1000, 1)
    )

    _brad_sl = _io_slope_lookup(brad_buran_df)
    brad_buran_df = brad_buran_df.dropna()
    brad_buran_df = brad_buran_df.merge(
        _brad_sl, on=["animal_id", "frequency"], how="left"
    )

    return split_by_mouse(brad_buran_df, stratify_on="tx")


def _build_liberman_orig() -> pd.DataFrame:
    liberman_df = load_data(db_level=0)
    _orig = liberman_df.copy()

    _pk = _orig["PeakI Curvature"].apply(pd.Series)
    _pk.columns = [
        "PeakI Early Curvature",
        "PeakI Central Curvature",
        "PeakI Late Curvature",
    ]
    _tr = _orig["TroughI Curvature"].apply(pd.Series)
    _tr.columns = [
        "TroughI Early Curvature",
        "TroughI Central Curvature",
        "TroughI Late Curvature",
    ]
    _orig = pd.concat([_orig, _pk, _tr], axis=1)

    _orig["noise_cat"] = [0 if x else 1 for x in (_orig["Noise"] <= 0.91)]

    _orig = _orig.rename(
        columns={
            "Subject": "animal_id",
            "Frequency(kHz)": "frequency",
            "Level(dB)": "level",
            "Amplitude": "amplitude",
            "Slope": "slope",
            "Total Variance": "total_variance",
            "Distance": "distance",
            "SynapsesPerIHC": "synapses",
            "Strain (binary)": "strain_binary",
            "PeakI Early Curvature": "PeakIEarlyCurvature",
            "PeakI Central Curvature": "PeakICentralCurvature",
            "PeakI Late Curvature": "PeakILateCurvature",
            "TroughI Early Curvature": "TroughIEarlyCurvature",
            "TroughI Central Curvature": "TroughICentralCurvature",
            "TroughI Late Curvature": "TroughILateCurvature",
            "P1_Latency_ms": "p1_latency",
            "Waveform": "full_waveform",
        }
    )
    _orig["level"] = _orig["level"].astype(float)
    _orig["frequency"] = _orig["frequency"].astype(float)

    _keep = [
        "animal_id",
        "frequency",
        "level",
        "amplitude",
        "slope",
        "total_variance",
        "distance",
        "p1_latency",
        "Slope_all",
        "Slope_high4",
        "synapses",
        "noise_cat",
        "Group",
        "PeakIEarlyCurvature",
        "PeakICentralCurvature",
        "PeakILateCurvature",
        "TroughIEarlyCurvature",
        "TroughICentralCurvature",
        "TroughILateCurvature",
        "Peaks",
        "Troughs",
        "WaveI",
        "full_waveform",
    ]
    if "strain_binary" in _orig.columns:
        _i = _keep.index("noise_cat") + 1
        _keep = _keep[:_i] + ["strain_binary"] + _keep[_i:]

    _orig = _orig[_keep].dropna(
        subset=[c for c in _keep if c not in ("p1_latency", "Slope_all", "Slope_high4")]
    )

    return split_by_mouse(_orig, split_on="animal_id", stratify_on="Group")


def _common_wide_cols(
    reformatted, reformatted_orig, *, include_extra_wide: bool = False
):
    _feat_joined = {
        c
        for c in reformatted.columns
        if any(c.startswith(f"{col}_") for col in _PIVOT_COLS)
    }
    _feat_orig = {
        c
        for c in reformatted_orig.columns
        if any(c.startswith(f"{col}_") for col in _PIVOT_COLS)
    }
    if include_extra_wide:
        _feat_joined |= {c for c in _EXTRA_WIDE_FEATURES if c in reformatted.columns}
        _feat_orig |= {
            c for c in _EXTRA_WIDE_FEATURES if c in reformatted_orig.columns
        }
    return sorted(_feat_joined & _feat_orig)


@dataclass
class NNStage2Data:
    brad_buran_df: pd.DataFrame
    reformatted: pd.DataFrame
    reformatted_orig: pd.DataFrame
    orig_lib: pd.DataFrame
    common_cols: list
    noise_num_bb: list
    noise_log_bb: list
    noise_num_lib: list
    noise_log_lib: list
    noise_num_common: list
    noise_log_common: list
    has_strain: bool
    long_num: list
    long_log: list
    long_cat: list
    full_wave_target_len: int
    full_wave_target_len_lib: int
    full_wave_target_len_brad: int
    full_wave_median_len_bb: int
    full_wave_median_len_lib: int


def load_nn_stage2_data(
    share_path: Path | None = None,
    *,
    full_wave_prefer_liberman_len: bool = True,
    include_extra_wide_features: bool = False,
    join_io_features: bool = False,
) -> NNStage2Data:
    """
    ``share_path`` is the project directory that contains ``abr_data/``, ``abr_io.csv``,
    and ``synapses.csv`` (the repository root next to the ``utils`` package).

    ``load_data`` resolves ``liberman_wpz`` from the repo root (see ``utils/data_loader.py``),
    so it does not depend on the process working directory.

    ``full_wave_prefer_liberman_len``: when Brad and Liberman median 0–8 ms crop
    lengths differ, use Liberman's median as the fixed CNN length if True, else Brad's.

    Brad long-form features (amplitude, slope, distance, variance, curvatures, ``WaveI``)
    are computed after resampling each 0–8 ms crop to LibT (``tw_lib`` from
    ``compute_full_wave_target_len(..., prefer_liberman_len=True)``).

    ``include_extra_wide_features``: when True, **Stage 1** wide RF and **Stage 2**
    tabular numeric inputs both include ``Slope_all``, ``Slope_high4``, and ``p1_latency``
    (matches ``abr_wide_long_comparison_newfeats``). Default False: Stage 1 uses only
    pivoted level-wise wide columns; Stage 2 uses ``LONG_NUM_BASE`` (amplitude, slope,
    distance, level) plus optional ``strain_binary``.

    ``join_io_features``: when True, join ``Slope_all``, ``Slope_high4``, ``p1_latency``
    onto wide tables. Default False matches ``abr_wide_long_comparison`` Liberman pivot.
    """
    abr2_root = Path(__file__).resolve().parent.parent
    if share_path is None:
        share_path = abr2_root

    brad_prefeature = _build_brad_prefeature(share_path)
    orig_lib = _build_liberman_orig()
    reformatted_orig = _build_reformatted_orig(orig_lib, join_io_features=join_io_features)

    tw_lib, mb, ml = compute_full_wave_target_len(
        brad_prefeature,
        orig_lib,
        prefer_liberman_len=True,
    )
    brad_buran_df = _apply_brad_libt_features(brad_prefeature, tw_lib)
    reformatted = _build_reformatted(brad_buran_df, join_io_features=join_io_features)

    common = _common_wide_cols(
        reformatted,
        reformatted_orig,
        include_extra_wide=include_extra_wide_features,
    )
    noise_num_bb, noise_log_bb = _noise_feats_from_wide(
        reformatted.columns, include_extra_wide=include_extra_wide_features
    )
    noise_num_lib, noise_log_lib = _noise_feats_from_wide(
        reformatted_orig.columns, include_extra_wide=include_extra_wide_features
    )
    noise_num_common, noise_log_common = _noise_feats_from_wide(
        common, include_extra_wide=include_extra_wide_features
    )

    has_strain = (
        "strain_binary" in reformatted.columns
        and "strain_binary" in reformatted_orig.columns
        and "strain_binary" in orig_lib.columns
    )
    long_num = (
        (LONG_NUM if include_extra_wide_features else LONG_NUM_BASE)
        + (["strain_binary"] if has_strain else [])
    )

    tw_bb, _, _ = compute_full_wave_target_len(
        brad_buran_df,
        orig_lib,
        prefer_liberman_len=False,
    )
    tw_primary = tw_lib if full_wave_prefer_liberman_len else tw_bb

    return NNStage2Data(
        brad_buran_df=brad_buran_df,
        reformatted=reformatted,
        reformatted_orig=reformatted_orig,
        orig_lib=orig_lib,
        common_cols=common,
        noise_num_bb=noise_num_bb,
        noise_log_bb=noise_log_bb,
        noise_num_lib=noise_num_lib,
        noise_log_lib=noise_log_lib,
        noise_num_common=noise_num_common,
        noise_log_common=noise_log_common,
        has_strain=has_strain,
        long_num=long_num,
        long_log=list(LONG_LOG),
        long_cat=list(LONG_CAT),
        full_wave_target_len=tw_primary,
        full_wave_target_len_lib=tw_lib,
        full_wave_target_len_brad=tw_bb,
        full_wave_median_len_bb=mb,
        full_wave_median_len_lib=ml,
    )


def splits_for_long_stage2(data: NNStage2Data):
    """Return the same train/test long and wide frames as the comparison notebook."""
    ref = data.reformatted
    ref_o = data.reformatted_orig
    bb = data.brad_buran_df
    lib = data.orig_lib

    bb_wide_train = ref[ref.DataGroup != "Test"].reset_index(drop=True)
    bb_wide_test = ref[ref.DataGroup == "Test"].reset_index(drop=True)
    bb_test_animals = set(bb_wide_test["animal_id"])

    bb_long_train = bb[~bb["animal_id"].isin(bb_test_animals)].reset_index(drop=True)
    bb_long_test = bb[bb["animal_id"].isin(bb_test_animals)].reset_index(drop=True)

    lib_long_train = lib[lib.DataGroup != "Test"].reset_index(drop=True)
    lib_long_test = lib[lib.DataGroup == "Test"].reset_index(drop=True)

    lib_train = ref_o[ref_o.DataGroup != "Test"].reset_index(drop=True)
    lib_test = ref_o[ref_o.DataGroup == "Test"].reset_index(drop=True)

    bb_long_all = bb.reset_index(drop=True)
    lib_long_all = lib.reset_index(drop=True)

    return {
        "bb_wide_train": bb_wide_train,
        "bb_wide_test": bb_wide_test,
        "bb_long_train": bb_long_train,
        "bb_long_test": bb_long_test,
        "lib_train": lib_train,
        "lib_test": lib_test,
        "lib_long_train": lib_long_train,
        "lib_long_test": lib_long_test,
        "bb_long_all": bb_long_all,
        "lib_long_all": lib_long_all,
    }

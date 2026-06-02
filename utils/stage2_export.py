"""
Export wide/long Parquet snapshots for Stage-2 HP tuning (Section 5.3).

Fits official Stage-1 noise model + sklearn preprocessors on **Train** rows only;
writes float ``x_*`` columns for tree feature matrices. **Not** used for synthesis CV.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.nn_stage2 import attach_noise_preds_long, fit_stage1_wide_best
from utils.nn_stage2_data import (
    NNStage2Data,
    load_nn_stage2_data,
    splits_for_long_stage2,
    syn_feats_from_wide_common,
    wide_stage1_fit,
    wide_stage1_val,
)
from utils.stage2_sklearn import _syn_prep
from utils.subject_cv import train_only_frame


def _x_cols(n: int) -> List[str]:
    return [f"x_{i}" for i in range(n)]

def _safe_tabular_long(df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
    """
    Select tabular-only columns for Parquet export.

    The raw long tables can contain waveform columns (e.g. ``WaveI``) holding
    Python objects (Series/arrays), which pyarrow cannot serialize.
    """
    # De-duplicate requested columns while preserving order.
    dedup_keep = list(dict.fromkeys(keep_cols))
    cols = [c for c in dedup_keep if c in df.columns]
    out = df.loc[:, cols].copy()
    # Defensive: drop any remaining object columns (typically waveform blobs).
    # Use select_dtypes so duplicate column names don't break dtype access.
    obj_cols = list(out.select_dtypes(include=["object"]).columns)
    if obj_cols:
        out = out.drop(columns=obj_cols)
    return out


def _concat_animal_preds(s1_out: Dict[str, Any]) -> pd.Series:
    tr = s1_out["animal_pred_non_test"]
    te = s1_out["animal_pred_te"]
    parts = [s for s in (tr, te) if s is not None and len(s) > 0]
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts)
    return s[~s.index.duplicated(keep="last")]


def _export_cohort_wide(
    wide_df: pd.DataFrame,
    noise_num: List[str],
    noise_log: List[str],
    common_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    wide_test_rows = wide_df[wide_df["DataGroup"].eq("Test")].reset_index(drop=True)
    s1_out = fit_stage1_wide_best(
        wide_stage1_fit(wide_df),
        wide_stage1_val(wide_df),
        wide_test_rows,
        noise_num,
        noise_log,
        random_state=1,
        verbose=False,
    )
    pred_all = _concat_animal_preds(s1_out)
    aug = attach_noise_preds_long(
        wide_df.reset_index(drop=True),
        pred_all,
        fallback_noise_cat=False,
        require_full_coverage=True,
    )
    # syn_feats_from_wide_common returns (syn_num, syn_log, syn_cat)
    syn_num, syn_log, syn_cat = syn_feats_from_wide_common(common_cols, has_strain=False)
    prep = _syn_prep(syn_num, syn_cat, syn_log)
    feat_cols = syn_num + syn_cat + syn_log
    tr_fit = train_only_frame(aug)
    X_fit = tr_fit[feat_cols].dropna()
    prep.fit(X_fit)

    aug = aug.copy()
    aug["cohort"] = aug.get("cohort", None)
    idx_ok = aug[feat_cols].dropna().index
    Xt = prep.transform(aug.loc[idx_ok, feat_cols])
    colnames = _x_cols(Xt.shape[1])
    for c in colnames:
        aug[c] = np.nan
    aug.loc[idx_ok, colnames] = Xt.astype(np.float32)

    manifest: Dict[str, Any] = {
        "stage1_model": s1_out.get("stage1_model"),
        "stage1_threshold": float(s1_out.get("stage1_threshold", 0.5)),
        "wide_syn_num": syn_num,
        "wide_syn_cat": syn_cat,
        "wide_syn_log": syn_log,
        "x_columns": colnames,
    }
    return aug, manifest


def _export_cohort_long(
    long_df: pd.DataFrame,
    wide_for_s1: pd.DataFrame,
    noise_num: List[str],
    noise_log: List[str],
    *,
    long_num: List[str],
    long_cat: List[str],
    long_log: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    wide_test_rows = wide_for_s1[wide_for_s1["DataGroup"].eq("Test")].reset_index(
        drop=True
    )
    s1_out = fit_stage1_wide_best(
        wide_stage1_fit(wide_for_s1),
        wide_stage1_val(wide_for_s1),
        wide_test_rows,
        noise_num,
        noise_log,
        random_state=1,
        verbose=False,
    )
    pred_all = _concat_animal_preds(s1_out)
    out = attach_noise_preds_long(
        long_df.reset_index(drop=True),
        pred_all,
        fallback_noise_cat=False,
        require_full_coverage=True,
    )
    keep = (
        ["animal_id", "frequency", "synapses", "noise_cat", "noise_preds", "DataGroup"]
        + list(long_num)
        + list(long_cat)
        + list(long_log)
    )
    out = _safe_tabular_long(out, keep)
    return out, {"stage1_model": s1_out.get("stage1_model")}


def export_stage2_data(
    *,
    out_dir: Path | None = None,
    data: NNStage2Data | None = None,
) -> Dict[str, Path]:
    """
    Write ``buran_wide/long``, ``liberman_wide/long`` parquet + ``manifest.json``
    under ``figures/cache/stage2_data``.
    """
    if out_dir is None:
        out_dir = (
            Path(__file__).resolve().parent.parent / "figures" / "cache" / "stage2_data"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = data or load_nn_stage2_data()
    sp = splits_for_long_stage2(data)

    common = list(data.common_cols)
    buran_wide, m_bb = _export_cohort_wide(
        data.reformatted, list(data.noise_num_bb), list(data.noise_log_bb), common
    )
    buran_wide["cohort"] = "buran"
    m_bb["cohort"] = "buran"
    m_bb["long_num"] = list(data.long_num)
    m_bb["long_cat"] = list(data.long_cat)
    m_bb["long_log"] = list(data.long_log)
    buran_wide.to_parquet(out_dir / "buran_wide.parquet", index=False)

    buran_long, lm_bb = _export_cohort_long(
        data.brad_buran_df,
        data.reformatted,
        list(data.noise_num_bb),
        list(data.noise_log_bb),
        long_num=list(data.long_num),
        long_cat=list(data.long_cat),
        long_log=list(data.long_log),
    )
    buran_long["cohort"] = "buran"
    m_bb.update(lm_bb)

    buran_long.to_parquet(out_dir / "buran_long.parquet", index=False)

    common_lib = list(
        sorted(set(data.common_cols) & set(data.reformatted_orig.columns))
    )
    liberman_wide, m_lib = _export_cohort_wide(
        data.reformatted_orig,
        list(data.noise_num_lib),
        list(data.noise_log_lib),
        common_lib,
    )
    liberman_wide["cohort"] = "liberman"
    m_lib["cohort"] = "liberman"
    m_lib["long_num"] = list(data.long_num)
    m_lib["long_cat"] = list(data.long_cat)
    m_lib["long_log"] = list(data.long_log)
    liberman_wide.to_parquet(out_dir / "liberman_wide.parquet", index=False)

    liberman_long, lm_lib = _export_cohort_long(
        data.orig_lib,
        data.reformatted_orig,
        list(data.noise_num_lib),
        list(data.noise_log_lib),
        long_num=list(data.long_num),
        long_cat=list(data.long_cat),
        long_log=list(data.long_log),
    )
    liberman_long["cohort"] = "liberman"
    m_lib.update(lm_lib)

    liberman_long.to_parquet(out_dir / "liberman_long.parquet", index=False)

    full_manifest = {
        "buran": m_bb,
        "liberman": m_lib,
        "split_keys": sorted(sp.keys()),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(full_manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "buran_wide": out_dir / "buran_wide.parquet",
        "buran_long": out_dir / "buran_long.parquet",
        "liberman_wide": out_dir / "liberman_wide.parquet",
        "liberman_long": out_dir / "liberman_long.parquet",
        "manifest": manifest_path,
    }

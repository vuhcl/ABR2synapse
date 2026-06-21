#!/usr/bin/env python3
"""Verify MLP SHAP parquet exports and deck figures."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.stage2_mlp_shap import (
    DECK_FIGURE_STEMS,
    EXPECTED_ROW_COUNTS,
    MLP_SHAP_CACHE_DIR,
    MLP_SHAP_FIG_DIR,
    SHAP_EXPORT_RENAMES,
    COHORT_A_SLUG,
    COHORT_B_SLUG,
)

def _check_values(cohort: str, cache_dir: Path, expected_n_features: int) -> None:
    path = cache_dir / f"shap_values_mlp_{cohort}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    expected_n = EXPECTED_ROW_COUNTS[cohort]
    if abs(len(df) - expected_n) > max(50, int(0.02 * expected_n)):
        raise ValueError(f"{cohort}: expected ~{expected_n} rows, got {len(df)}")
    shap_cols = [c for c in df.columns if c not in ("animal_id", "frequency", "level")]
    if len(shap_cols) != expected_n_features:
        raise ValueError(
            f"{cohort}: expected {expected_n_features} SHAP cols, got {len(shap_cols)}"
        )
    dup = df.duplicated(subset=["animal_id", "frequency", "level"])
    if dup.any():
        raise ValueError(f"{cohort}: {int(dup.sum())} duplicate animal×frequency×level rows")


def _check_metadata(cohort: str, cache_dir: Path) -> None:
    path = cache_dir / f"shap_metadata_mlp_{cohort}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    required = {
        "animal_id",
        "frequency",
        "stimulus_level",
        "noise_cat",
        "noise_preds",
        "synapses",
        "fold_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{cohort} metadata missing columns: {sorted(missing)}")
    if len(df) != len(pd.read_parquet(cache_dir / f"shap_values_mlp_{cohort}.parquet")):
        raise ValueError(f"{cohort}: metadata row count != values row count")


def _check_figures(fig_dir: Path) -> None:
    for stem in DECK_FIGURE_STEMS:
        for ext in (".png", ".svg"):
            p = fig_dir / f"{stem}{ext}"
            if not p.is_file():
                raise FileNotFoundError(p)


def main() -> None:
    cache_dir = MLP_SHAP_CACHE_DIR
    fig_dir = MLP_SHAP_FIG_DIR
    fn_path = cache_dir / "feature_names_mlp.parquet"
    if not fn_path.is_file():
        raise FileNotFoundError(fn_path)
    expected_n_features = len(pd.read_parquet(fn_path))

    for cohort in (COHORT_A_SLUG, COHORT_B_SLUG):
        _check_values(cohort, cache_dir, expected_n_features)
        _check_metadata(cohort, cache_dir)
        print(f"OK parquet {cohort}")

    _check_figures(fig_dir)
    print("OK figures", fig_dir)
    print("verify_mlp_shap_exports: all checks passed")


if __name__ == "__main__":
    main()

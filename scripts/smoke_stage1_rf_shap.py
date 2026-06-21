#!/usr/bin/env python3
"""Smoke Stage-1 RF TreeExplainer SHAP (subsampled eval rows)."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.nn_stage2_data import load_nn_stage2_data, splits_for_long_stage2
from utils.stage1_rf_shap import (
    N_STAGE1_WIDE_FEATURES,
    compute_cohort_stage1_rf_shap,
    export_stage1_shap_parquet,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", choices=("liberman", "buran"), default="liberman")
    p.add_argument("--max-eval-rows", type=int, default=200)
    args = p.parse_args()

    data = load_nn_stage2_data()
    splits = splits_for_long_stage2(data)

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        out = compute_cohort_stage1_rf_shap(
            args.cohort,
            data,
            splits,
            max_eval_rows=args.max_eval_rows,
        )
        rf = out["s1"]["clf"].named_steps["rf"]
        assert isinstance(rf, RandomForestClassifier), type(rf)

        shap_df = out["shap"]
        assert shap_df.shape[1] == N_STAGE1_WIDE_FEATURES
        assert shap_df.shape[0] <= args.max_eval_rows
        assert shap_df.shape[0] > 0
        assert not shap_df.isna().any().any()

        paths = export_stage1_shap_parquet(
            args.cohort,
            shap_df,
            out["meta"],
            feature_table=out["feature_table"],
            cache_dir=cache,
        )
        for key in ("shap", "meta", "feature_names"):
            assert paths[key].is_file(), paths[key]

        print(
            f"ok cohort={args.cohort} rows={len(shap_df)} "
            f"features={shap_df.shape[1]} cache={cache}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Smoke-run Liberman classical config panel (12 rows) or true-noise mirror (10 rows)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.nn_stage2_data import load_nn_stage2_data, splits_for_long_stage2
from utils.liberman_classical import (
    COMPARISON_PAIRS_TRUE,
    TRUE_NOISE_TREE_CONFIGS,
    animal_noise_series,
    attach_animal_noise_cat,
    derive_comparisons,
    export_artifacts,
    liberman_feature_lists,
    run_config_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--true-noise",
        action="store_true",
        help="Run true-noise mirror panel (10 rows, skip Stage 1)",
    )
    args = parser.parse_args()

    data = load_nn_stage2_data(join_io_features=False)
    sp = splits_for_long_stage2(data)
    an = animal_noise_series(data.orig_lib)
    lib_tr = attach_animal_noise_cat(sp["lib_train"].copy(), an)
    lib_te = attach_animal_noise_cat(sp["lib_test"].copy(), an)
    lib_long_tr = attach_animal_noise_cat(sp["lib_long_train"].copy(), an)
    lib_long_te = attach_animal_noise_cat(sp["lib_long_test"].copy(), an)
    feats = liberman_feature_lists(data.reformatted_orig, data.common_cols)

    if args.true_noise:
        results = run_config_panel(
            lib_tr,
            lib_te,
            lib_long_tr,
            lib_long_te,
            feats,
            tree_configs=TRUE_NOISE_TREE_CONFIGS,
            skip_stage1=True,
            ols_noise_mode="true",
            verbose=True,
        )
        expected = 10
        stem = "liberman_classical_true_noise"
        comp_kw = dict(
            comparison_pairs=COMPARISON_PAIRS_TRUE,
            tree_configs=TRUE_NOISE_TREE_CONFIGS,
            skip_stage1=True,
        )
    else:
        results = run_config_panel(
            lib_tr, lib_te, lib_long_tr, lib_long_te, feats, verbose=True
        )
        expected = 12
        stem = None
        comp_kw = {}

    assert len(results) == expected, f"got {len(results)} rows"
    if data.has_strain:
        assert "strain_binary" in feats["syn_num"]
        assert "strain_binary" not in feats["noise_num"]
        assert "strain_binary" in feats["long_num"]
    comps = derive_comparisons(
        results,
        wide_train=lib_tr,
        wide_test=lib_te,
        long_train=lib_long_tr,
        long_test=lib_long_te,
        feats=feats,
        **comp_kw,
    )
    assert "strain" not in comps["question"].values
    assert "t_pvalue" in comps.columns
    export_kw = {"stem": stem} if stem else {}
    pq, js, cp = export_artifacts(results, comps, Path("figures/cache"), **export_kw)
    print("wrote", pq, js, cp)
    print(results.to_string())


if __name__ == "__main__":
    main()

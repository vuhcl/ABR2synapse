#!/usr/bin/env python3
"""Smoke-run Liberman classical config panel (12 rows)."""
from pathlib import Path

from utils.nn_stage2_data import load_nn_stage2_data, splits_for_long_stage2
from utils.liberman_classical import (
    animal_noise_series,
    attach_animal_noise_cat,
    derive_comparisons,
    export_artifacts,
    liberman_feature_lists,
    run_config_panel,
)


def main() -> None:
    data = load_nn_stage2_data(join_io_features=False)
    sp = splits_for_long_stage2(data)
    an = animal_noise_series(data.orig_lib)
    lib_tr = attach_animal_noise_cat(sp["lib_train"].copy(), an)
    lib_te = attach_animal_noise_cat(sp["lib_test"].copy(), an)
    lib_long_tr = attach_animal_noise_cat(sp["lib_long_train"].copy(), an)
    lib_long_te = attach_animal_noise_cat(sp["lib_long_test"].copy(), an)
    feats = liberman_feature_lists(data.reformatted_orig, data.common_cols)
    results = run_config_panel(
        lib_tr, lib_te, lib_long_tr, lib_long_te, feats, verbose=True
    )
    assert len(results) == 12, f"got {len(results)} rows"
    assert "strain_binary" not in feats["syn_num"]
    comps = derive_comparisons(
        results,
        wide_train=lib_tr,
        wide_test=lib_te,
        long_train=lib_long_tr,
        long_test=lib_long_te,
        feats=feats,
    )
    assert "strain" not in comps["question"].values
    assert "t_pvalue" in comps.columns
    pq, js, cp = export_artifacts(results, comps, Path("figures/cache"))
    print("wrote", pq, js, cp)
    print(results.to_string())


if __name__ == "__main__":
    main()

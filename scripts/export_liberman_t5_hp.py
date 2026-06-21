#!/usr/bin/env python3
"""Export T5 wide RF/XGB best_params for scenario C (Liberman 5.2)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.liberman_classical import export_t5_stage2_hp, liberman_feature_lists
from utils.nn_stage2_data import load_nn_stage2_data


def main() -> None:
    out = Path("figures/cache/stage2_best_hp/liberman_t5_sklearn.json")
    if out.is_file():
        print(f"Already exists: {out} (delete to re-export)")
        return

    data = load_nn_stage2_data()
    feats = liberman_feature_lists(data.reformatted_orig, data.common_cols)
    wide_train = data.reformatted_orig.loc[
        data.reformatted_orig["DataGroup"] != "Test"
    ].reset_index(drop=True)
    wide_test = data.reformatted_orig.loc[
        data.reformatted_orig["DataGroup"] == "Test"
    ].reset_index(drop=True)
    export_t5_stage2_hp(wide_train, wide_test, feats, out_path=out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

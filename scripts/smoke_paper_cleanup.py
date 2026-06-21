#!/usr/bin/env python3
"""Smoke checks for paper notebook cleanup (utils + splits)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.nn_stage2_data import (
    STAGE_SPL_LEVELS,
    load_nn_stage2_data,
    parse_wide_spl_level,
    splits_for_long_stage2,
    wide_columns_at_stage_spl,
    wide_stage1_fit,
    wide_stage1_val,
    _noise_feats_from_wide,
)


def main() -> int:
    errors: list[str] = []
    data = load_nn_stage2_data()
    sp = splits_for_long_stage2(data)

    for name, df in [("Brad long", data.brad_buran_df), ("Lib long", data.orig_lib)]:
        if "DataGroup" not in df.columns:
            errors.append(f"{name}: missing DataGroup")
            continue
        for g in ("Train", "Validate", "Test"):
            if (df["DataGroup"] == g).sum() == 0:
                errors.append(f"{name}: empty DataGroup {g}")
        animals = df.groupby("animal_id")["DataGroup"].nunique()
        if (animals > 1).any():
            errors.append(f"{name}: animal in multiple DataGroups")

    bb_test = set(
        data.brad_buran_df.loc[data.brad_buran_df["DataGroup"] == "Test", "animal_id"]
    )
    bb_test2 = set(sp["bb_wide_test"]["animal_id"])
    if bb_test != bb_test2:
        errors.append(f"Brad test animals mismatch: {bb_test ^ bb_test2}")

    cols = list(data.reformatted.columns) + ["amplitude_45.0", "amplitude_500.0"]
    spl = wide_columns_at_stage_spl(cols)
    if "amplitude_50.0" not in spl:
        errors.append("amplitude_50.0 missing from SPL filter")
    if "amplitude_45.0" in spl or "amplitude_500.0" in spl:
        errors.append("forbidden SPL columns in filter")

    num, _ = _noise_feats_from_wide(data.reformatted.columns)
    for c in num:
        lvl = parse_wide_spl_level(c)
        if lvl is not None and int(lvl) not in STAGE_SPL_LEVELS:
            errors.append(f"noise_num column outside SPL: {c}")

    fit = wide_stage1_fit(sp["bb_wide_train"])
    val = wide_stage1_val(sp["bb_wide_train"])
    if len(fit) >= len(sp["bb_wide_train"]):
        errors.append("s1_fit should be smaller than non-Test wide when Validate exists")
    if set(fit["animal_id"]) & set(val["animal_id"]):
        errors.append("s1_fit and s1_val animals overlap")
    if len(val) == 0:
        errors.append("s1_val empty")

    if data.has_strain:
        for lst in (data.noise_num_bb, data.noise_num_lib, data.noise_num_common):
            if "strain_binary" in lst:
                errors.append("strain_binary must not be in Stage-1 noise features")
        if "strain_binary" not in data.long_num:
            errors.append("strain_binary missing from long_num")
        if not data.reformatted["strain_binary"].eq(0).all():
            errors.append("Brad strain_binary must be all 0")
        if not data.orig_lib["strain_binary"].isin([0, 1]).all():
            errors.append("Liberman strain_binary must be in {0, 1}")

    baseline_path = ROOT / "figures/cache/_baseline_probe.json"
    if baseline_path.exists():
        base = json.loads(baseline_path.read_text())
        if base.get("len_noise_num_common") != len(data.noise_num_common):
            print(
                "note: len(noise_num_common) changed",
                base.get("len_noise_num_common"),
                "->",
                len(data.noise_num_common),
            )
    else:
        probe = {
            "bb_test_animals": sorted(bb_test),
            "len_noise_num_common": len(data.noise_num_common),
            "noise_num_sample": data.noise_num_common[:5],
        }
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(probe, indent=2))
        print("wrote", baseline_path)

    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1
    print("smoke_paper_cleanup: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

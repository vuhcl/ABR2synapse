#!/usr/bin/env python3
"""Smoke dual-grain ABR univariate EDA (tables only, no figures)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.abr_univariate_eda import (
    COHORT_A_LABEL,
    COHORT_B_LABEL,
    GRAIN_ANIMAL_FREQ,
    GRAIN_LONG,
    run_univariate_eda,
)
from utils.nn_stage2_data import load_nn_stage2_data


def main() -> None:
    data = load_nn_stage2_data()
    out = run_univariate_eda(data, write_figures=False)

    for grain, tbl in [(GRAIN_LONG, out["long_tbl"]), (GRAIN_ANIMAL_FREQ, out["af_tbl"])]:
        for cohort in (COHORT_A_LABEL, COHORT_B_LABEL):
            for ng in ("lower", "higher"):
                sub = tbl[(tbl["cohort"] == cohort) & (tbl["noise_group"] == ng)]
                n = int(sub["n"].iloc[0]) if len(sub) else 0
                if n <= 0:
                    raise SystemExit(f"Empty stratum: {grain} {cohort} {ng}")

    ds = out["flags"]["dedup_stats"]
    skew_path = out.get("skew_path")
    if skew_path is None or not Path(skew_path).is_file():
        raise SystemExit("Missing feature_skewness_long.csv")
    print("OK: spearman + skew tables written (no figures)")
    print("n_long:", out["flags"]["n_long"])
    print("n_animal_freq:", out["flags"]["n_animal_freq"])
    print("dedup_stats:", ds)


if __name__ == "__main__":
    main()

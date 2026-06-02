#!/usr/bin/env python3
"""Report CV splitter and per-cohort experimental_group balance."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from utils.nn_stage2_data import load_nn_stage2_data
from utils.subject_cv import (
    DEFAULT_CEILING_CV_SPLITS,
    EXPERIMENTAL_GROUP_COL,
    build_subject_cv,
    prepare_strat_groups,
)
from utils.stage2_synthesis_cv import CV_SEED


def main() -> None:
    data = load_nn_stage2_data()
    for name, wide in [("Brad", data.reformatted), ("Liberman", data.reformatted_orig)]:
        work = wide.reset_index(drop=True)
        groups, strat, _ = prepare_strat_groups(work, EXPERIMENTAL_GROUP_COL, "animal_id")
        per_animal = work.groupby("animal_id")[EXPERIMENTAL_GROUP_COL].first()
        counts = per_animal.value_counts()
        cv = build_subject_cv(
            strat,
            groups,
            n_splits=DEFAULT_CEILING_CV_SPLITS,
            random_state=CV_SEED,
            prefer_stratified=True,
        )
        print(f"\n{name}: splitter={type(cv).__name__}, n_splits={DEFAULT_CEILING_CV_SPLITS}")
        print(
            f"  animals={work['animal_id'].nunique()}, "
            f"exp_groups={per_animal.nunique()}, "
            f"min/max animals per group: {counts.min()}/{counts.max()}"
        )
        split_x = np.zeros(len(work))
        folds = list(cv.split(split_x, strat, groups))
        te_sizes = [work.iloc[te_idx]["animal_id"].nunique() for _, te_idx in folds]
        print(f"  held-out animals per fold: {te_sizes}")


if __name__ == "__main__":
    main()

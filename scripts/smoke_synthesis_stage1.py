"""Smoke: global Stage-1 labels + fold attach (Brad + Liberman, A/B/C)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.nn_stage2_data import load_nn_stage2_data, splits_for_long_stage2
from utils.stage2_synthesis_cv import (
    _attach_global_stage1,
    build_cv_scenario_frames,
    fit_global_stage1_for_scenario,
    generate_cv_folds,
)

data = load_nn_stage2_data()
assert data.has_strain, "expected strain on both cohorts"
assert "strain_binary" not in data.noise_num_bb
assert "strain_binary" not in data.noise_num_lib
assert "strain_binary" in data.long_num
assert data.reformatted["strain_binary"].eq(0).all()

splits = splits_for_long_stage2(data)
for cohort, wide in (
    ("Brad", data.reformatted.reset_index(drop=True)),
    ("Liberman", data.reformatted_orig.reset_index(drop=True)),
):
    gs1 = {
        scen: fit_global_stage1_for_scenario(
            scen, data, splits, eval_cohort=cohort
        )
        for scen in ("A", "B", "C")
    }
    _, _, tr_anim, te_anim = generate_cv_folds(wide)[0]
    for scen in "ABC":
        w_tr, w_ev, l_tr, l_ev, _, _ = build_cv_scenario_frames(
            cohort, scen, tr_anim, te_anim, data, splits
        )
        _attach_global_stage1(
            w_tr, w_ev, l_tr, l_ev, gs1[scen]["animal_preds"]
        )
        print(f"{cohort} {scen} OK")

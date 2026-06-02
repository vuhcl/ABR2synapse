"""Smoke: global Stage-1 labels + fold attach (Brad + Liberman, A/B/C)."""
from utils.nn_stage2_data import load_nn_stage2_data, splits_for_long_stage2
from utils.stage2_synthesis_cv import (
    _attach_global_stage1,
    build_cv_scenario_frames,
    fit_global_stage1_for_scenario,
    generate_cv_folds,
)

data = load_nn_stage2_data()
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

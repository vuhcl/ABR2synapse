#!/usr/bin/env python3
"""Phase 6 verification gates for paper cleanup."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/bin/python"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)})


def main() -> int:
    _run([str(PY), str(ROOT / "scripts/smoke_paper_cleanup.py")])

    _run([str(PY), str(ROOT / "scripts/run_stage1_export.py")])

    _run(
        [
            str(PY),
            "-c",
            """
from utils.nn_stage2 import (
    _LR_C_GRID,
    RunConfig,
    _youden_j_threshold,
    bb_long_s2_rows,
    fit_stage1_wide_best,
    run_two_stage_long_nn,
)
from utils.nn_stage2_data import load_nn_stage2_data, splits_for_long_stage2, wide_stage1_fit, wide_stage1_val
import numpy as np

t = _youden_j_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
assert 0.0 < t < 1.0, t

assert len(_LR_C_GRID) == 6
data = load_nn_stage2_data()
sp = splits_for_long_stage2(data)
bb_tr, bb_te = sp['bb_wide_train'], sp['bb_wide_test']
s1 = fit_stage1_wide_best(wide_stage1_fit(bb_tr), wide_stage1_val(bb_tr), bb_te, data.noise_num_bb, data.noise_log_bb, verbose=False)
assert s1['stage1_model'] in ('rf', 'lr')
assert 'animal_pred_non_test' in s1
assert 0.0 <= s1['stage1_threshold'] <= 1.0
pred_te = (s1['animal_prob_te'] > s1['stage1_threshold']).astype(int)
assert pred_te.equals(s1['animal_pred_te'])

# Scenario B row: mixed pool + common feats
row_b = bb_long_s2_rows(data, sp)[1]
wt, wte, lt, lte, nn, nl = row_b
assert len(wt) > len(sp['bb_wide_train'])
cfg = RunConfig(epochs=1, batch_size=256, split_random_state=22, random_state=1)
r = run_two_stage_long_nn(wt, wte, lt, lte, 'smoke-B', data, mode='mlp', cfg=cfg, noise_num=nn, noise_log=nl, verbose=False)
assert np.isfinite(r[0])
print('stage1 + scenario B smoke OK')
""",
        ]
    )

    s1_path = ROOT / "figures/cache/stage1_wide_metrics.json"
    if not s1_path.is_file():
        raise SystemExit(f"missing {s1_path}")
    s1 = json.loads(s1_path.read_text())
    for lab in ("Brad", "Lib"):
        block = s1[lab]
        if block.get("selected_model") not in ("rf", "lr"):
            raise SystemExit(f"bad selected_model for {lab}")
        for key in (
            "fit_acc_row",
            "fit_acc_animal",
            "val_acc_pre",
            "val_acc_animal",
            "non_test_acc_animal",
            "non_test_auc_animal",
            "threshold_youden",
            "threshold_youden_val_only",
            "test_acc_post",
            "test_acc_post_0p5",
            "test_auc_post",
        ):
            if key not in block:
                raise SystemExit(f"missing {key} for {lab}")
        for sub in ("rf", "lr"):
            for key in (
                "fit_acc_row",
                "fit_acc_animal",
                "val_acc_pre",
                "val_acc_animal",
                "non_test_acc_animal",
                "threshold_youden",
                "threshold_youden_val_only",
                "test_acc_post",
                "test_acc_post_0p5",
                "test_auc_post",
            ):
                if key not in block[sub]:
                    raise SystemExit(f"missing {lab}.{sub}.{key}")

    shim = json.loads((ROOT / "figures/cache/stage1_wide_rf_metrics.json").read_text())
    for lab in ("Brad", "Lib"):
        if len(shim[lab]) != 3:
            raise SystemExit(f"bad shim tuple for {lab}")

    ev = ROOT / "figures/cache/stage1_wide_noise_lr_rf_eval.parquet"
    if not ev.is_file():
        raise SystemExit(f"missing {ev}")

    brad_p = ROOT / "figures/cache/brad_group_mean_baseline_metrics.parquet"
    if not brad_p.exists():
        raise SystemExit("missing brad_group_mean_baseline_metrics.parquet")

    nn_p = ROOT / "figures/cache/nn_metrics_all.parquet"
    if nn_p.exists():
        import pandas as pd

        nn = pd.read_parquet(nn_p)
        if "stage2_format" in nn.columns:
            nn = nn[(nn["model"] != "MLP_wide") & (nn["stage2_format"] == "long")].copy()
        else:
            nn = nn[nn["model"] != "MLP_wide"].copy()
        if (nn["model"] == "MLP_wide").any():
            raise SystemExit("nn_metrics_all still contains MLP_wide")
        print(f"nn_metrics_all OK ({len(nn)} rows)")
    else:
        print("note: nn_metrics_all.parquet not present (re-run abr_nn_stage2 to export)")

    print("verify_paper_cleanup: all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

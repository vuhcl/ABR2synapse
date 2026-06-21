"""Smoke: Stage 1 classification table pooled overall acc/AUC."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.stage2_synthesis_cv import build_stage1_classification_table

EXPECTED = {
    1: {"overall_test_acc": 0.939, "overall_test_auc": 0.842},
    2: {"overall_test_acc": 0.576, "overall_test_auc": 0.662},
    3: {"overall_test_acc": 0.788, "overall_test_auc": 0.805},
}
TOL = 1e-3

table = build_stage1_classification_table(verbose=False)
assert set(table["overall_n_test"]) == {33}, table["overall_n_test"].tolist()

for _, row in table.iterrows():
    scen = int(row["scenario"])
    exp = EXPECTED[scen]
    for col, target in exp.items():
        got = float(row[col])
        assert abs(got - target) < TOL, f"scenario {scen} {col}: got {got}, want {target}"

print("smoke_stage1_pooled OK")
for _, row in table.iterrows():
    print(
        f"  scenario {int(row['scenario'])}: "
        f"overall acc={row['overall_test_acc']:.3f} "
        f"auc={row['overall_test_auc']:.3f}"
    )

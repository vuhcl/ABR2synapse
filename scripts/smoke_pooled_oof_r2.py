"""Smoke: OOF R² summarizer + minimal Figure S2 render (synthetic data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.stage2_synthesis_cv import (
    pooled_oof_r2_report_table,
    summarize_cohort_oof_r2,
    summarize_pooled_oof_r2,
    summarize_pooled_panel_oof_r2,
)
from utils.stage2_synthesis_plot import synthesis_cohort_panels_3cv_r2


def _synthetic_cohort_oof(n_folds: int = 10) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    for fold in range(n_folds):
        for eval_cohort in ("Brad", "Liberman"):
            for scen in ("A", "B", "C", "all"):
                model = "ceiling" if scen == "all" else "L7"
                for i in range(5):
                    y = float(rng.normal(100, 10))
                    pred = y + float(rng.normal(0, 5 if model == "ceiling" else 3))
                    rows.append(
                        {
                            "eval_cohort": eval_cohort,
                            "fold": fold,
                            "train_scenario": scen,
                            "model": model,
                            "animal_id": f"{eval_cohort}_{fold}_{i}",
                            "frequency": float(i + 1),
                            "y_true": y,
                            "y_pred": pred,
                        }
                    )
            for scen in ("A", "B", "C"):
                for model in ("L7", "RF", "XGB", "MLP"):
                    for i in range(5):
                        y = float(rng.normal(100, 10))
                        pred = y + float(rng.normal(0, 4))
                        rows.append(
                            {
                                "eval_cohort": eval_cohort,
                                "fold": fold,
                                "train_scenario": scen,
                                "model": model,
                                "animal_id": f"{eval_cohort}_{scen}_{fold}_{i}",
                                "frequency": float(i + 1),
                                "y_true": y,
                                "y_pred": pred,
                            }
                        )
    return pd.DataFrame(rows)


def _synthetic_pooled_oof(n_folds: int = 10) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(7)
    for fold in range(n_folds):
        for display in (0, 1, 2, 3):
            model = "ceiling" if display == 0 else "MLP"
            for i in range(8):
                y = float(rng.normal(100, 10))
                pred = y + float(rng.normal(0, 6 if display == 2 else 4))
                rows.append(
                    {
                        "fold": fold,
                        "display": display,
                        "model": model,
                        "animal_id": f"p_{display}_{fold}_{i}",
                        "frequency": float(i + 1),
                        "y_true": y,
                        "y_pred": pred,
                    }
                )
        for display in (1, 2, 3):
            for model in ("L7", "RF", "XGB", "MLP"):
                for i in range(8):
                    y = float(rng.normal(100, 10))
                    pred = y + float(rng.normal(0, 5))
                    rows.append(
                        {
                            "fold": fold,
                            "display": display,
                            "model": model,
                            "animal_id": f"p_{display}_{model}_{fold}_{i}",
                            "frequency": float(i + 1),
                            "y_true": y,
                            "y_pred": pred,
                        }
                    )
    return pd.DataFrame(rows)


def smoke_summarizer() -> pd.DataFrame:
    cohort = _synthetic_cohort_oof()
    pooled = _synthetic_pooled_oof()
    r2_cohort = summarize_cohort_oof_r2(cohort)
    r2_pooled = summarize_pooled_panel_oof_r2(pooled)
    r2_full = summarize_pooled_oof_r2(cohort, pooled)
    assert set(r2_full["test_set"]) == {"Brad", "Liberman", "Pooled"}
    assert len(r2_cohort) == 26  # 2 ceiling + 2*3*4 models
    assert len(r2_pooled) == 13  # 1 ceiling + 3*4 models
    tbl = pooled_oof_r2_report_table(r2_full)
    assert "Cohort A (Liberman)" in tbl.columns
    assert len(tbl) == 5  # 4 models + ceiling
    print("summarize_pooled_oof_r2 OK", r2_full.shape, "table", tbl.shape)
    return r2_full


def smoke_figure(r2_summary: pd.DataFrame) -> Path:
    out = ROOT / "figures" / "presentation" / "_smoke_S2_pooled_oof_r2.png"
    png = synthesis_cohort_panels_3cv_r2(r2_summary, out_dir=out.parent, fname=out.name)
    assert png.is_file(), png
    print("synthesis_cohort_panels_3cv_r2 OK", png)
    return png


if __name__ == "__main__":
    r2 = smoke_summarizer()
    smoke_figure(r2)
    print("smoke_pooled_oof_r2 OK")

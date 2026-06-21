"""Rebuild true-noise synthesis OOF caches (after OOF schema v2 fix)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.benchmark_metrics import STAGE2_BEST_HP_TRUE_DIR, synthesis_cv_cache_paths
from utils.stage2_hp import resolve_torch_device
from utils.stage2_synthesis_cv import (
    compute_pooled_synthesis_folds,
    run_synthesis_cv,
    summarize_pooled_oof_r2,
)

N_FOLDS = 10


def main() -> None:
    hp = {
        s: json.loads((STAGE2_BEST_HP_TRUE_DIR / f"{s}.json").read_text())
        for s in ("A", "B", "C")
    }
    paths = synthesis_cv_cache_paths(N_FOLDS, variant="true")
    folds_path, progress_path, summary_path = paths[:3]
    pooled_folds, pooled_progress = paths[3], paths[4]
    oof_path, _, pooled_oof_path, _, oof_r2_summary_path = paths[5:10]
    device = resolve_torch_device()

    run_synthesis_cv(
        hp,
        noise_label="true",
        n_splits=N_FOLDS,
        folds_path=folds_path,
        progress_path=progress_path,
        summary_path=summary_path,
        oof_path=oof_path,
        oof_progress_path=paths[6],
        device=device,
        verbose=True,
        force_rerun=False,
    )
    compute_pooled_synthesis_folds(
        hp,
        noise_label="true",
        n_splits=N_FOLDS,
        folds_path=pooled_folds,
        progress_path=pooled_progress,
        oof_path=pooled_oof_path,
        oof_progress_path=paths[8],
        device=device,
        verbose=True,
        force_rerun=False,
    )

    folds = pd.read_parquet(folds_path)
    cohort_oof = pd.read_parquet(oof_path)
    pooled_oof = pd.read_parquet(pooled_oof_path)
    r2 = summarize_pooled_oof_r2(cohort_oof, pooled_oof)
    r2.to_parquet(oof_r2_summary_path, index=False)

    # Parity: fold RMSE vs OOF RMSE (MLP, fold 0, Brad A)
    target = float(
        folds.loc[
            (folds.fold == 0)
            & (folds.eval_cohort == "Brad")
            & (folds.train_scenario == "A")
            & (folds.model == "MLP"),
            "rmse",
        ].iloc[0]
    )
    sub = cohort_oof.loc[
        (cohort_oof.fold == 0)
        & (cohort_oof.eval_cohort == "Brad")
        & (cohort_oof.train_scenario == "A")
        & (cohort_oof.model == "MLP")
    ]
    oof_rmse = float(np.sqrt(np.mean((sub.y_true - sub.y_pred) ** 2)))
    print(f"MLP fold0 Brad-A: fold_rmse={target:.4f} oof_rmse={oof_rmse:.4f}")
    if abs(target - oof_rmse) >= 0.15:
        raise AssertionError(
            f"OOF RMSE drift vs fold cache: fold={target:.4f} oof={oof_rmse:.4f}"
        )

    mlp_r2 = r2.loc[r2.model.eq("MLP")]
    print("\nMLP R2 summary:\n", mlp_r2.to_string(index=False))
    pooled_mlp = mlp_r2.loc[mlp_r2.test_set.eq("Pooled") & mlp_r2.scenario.eq("3")]
    assert float(pooled_mlp.R2.iloc[0]) > 0.1
    print("rebuild_true_noise_oof OK")


if __name__ == "__main__":
    main()

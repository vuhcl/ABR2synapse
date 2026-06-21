#!/usr/bin/env python3
"""Smoke: 1-fold synthesis CV under oracle noise_cat + true HP dir."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.benchmark_metrics import STAGE2_BEST_HP_TRUE_DIR, synthesis_cv_cache_paths
from utils.stage2_hp import resolve_torch_device
from utils.stage2_synthesis_cv import (
    compute_pooled_synthesis_folds,
    run_synthesis_cv,
    summarize_cv_folds,
    summarize_pooled_folds,
)


def _load_hp() -> dict:
    hp = {}
    for scen in ("A", "B", "C"):
        p = STAGE2_BEST_HP_TRUE_DIR / f"{scen}.json"
        if not p.is_file():
            raise SystemExit(
                f"Missing {p}; run: uv run python scripts/run_stage2_hp_true_noise.py --smoke"
            )
        payload = json.loads(p.read_text(encoding="utf-8"))
        assert payload.get("noise_label") == "true", scen
        for block in ("RF", "XGB", "MLP"):
            assert block in payload, f"{scen} missing {block}"
        hp[scen] = payload
    return hp


def main() -> None:
    hp = _load_hp()
    device = resolve_torch_device()
    n_splits = 2  # subject_cv effective minimum is 2
    smoke_dir = Path("figures/cache/_smoke_synthesis_true")
    smoke_dir.mkdir(parents=True, exist_ok=True)
    paths = synthesis_cv_cache_paths(n_splits, variant="true")
    folds_path = smoke_dir / paths[0].name
    progress_path = smoke_dir / paths[1].name
    summary_path = smoke_dir / paths[2].name
    pooled_folds = smoke_dir / paths[3].name
    pooled_progress = smoke_dir / paths[4].name

    folds_df, summary = run_synthesis_cv(
        hp,
        noise_label="true",
        n_splits=n_splits,
        folds_path=folds_path,
        progress_path=progress_path,
        summary_path=summary_path,
        device=device,
        force_rerun=True,
        verbose=True,
    )
    assert not folds_df.empty
    assert summary["n_folds"].max() == n_splits

    pooled_df = compute_pooled_synthesis_folds(
        hp,
        noise_label="true",
        n_splits=n_splits,
        folds_path=pooled_folds,
        progress_path=pooled_progress,
        device=device,
        force_rerun=True,
        verbose=True,
    )
    assert not pooled_df.empty
    summarize_cv_folds(folds_df)
    summarize_pooled_folds(pooled_df)
    print("true-noise synthesis CV smoke OK")


if __name__ == "__main__":
    main()

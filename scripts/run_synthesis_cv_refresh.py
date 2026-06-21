#!/usr/bin/env python3
"""Re-run synthesis CV with force_rerun for 10- and 5-fold caches."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.benchmark_metrics import STAGE2_BEST_HP_DIR, stage2_synthesis_cv_paths
from utils.stage2_hp import resolve_torch_device
from utils.stage2_synthesis_cv import run_synthesis_cv


def main() -> None:
    hp = {
        scen: json.loads((STAGE2_BEST_HP_DIR / f"{scen}.json").read_text(encoding="utf-8"))
        for scen in ("A", "B", "C")
    }
    device = resolve_torch_device()
    for n_folds in (10, 5):
        _, summary = run_synthesis_cv(
            hp,
            device=device,
            n_splits=n_folds,
            force_rerun=True,
            verbose=True,
        )
        print(n_folds, "fold summary keys:", list(summary.keys()))
        print(stage2_synthesis_cv_paths(n_folds)[2])


if __name__ == "__main__":
    main()

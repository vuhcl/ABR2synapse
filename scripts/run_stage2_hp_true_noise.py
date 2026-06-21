#!/usr/bin/env python3
"""Tune scenario A/B/C HP JSONs under oracle noise_cat (true-noise mirror)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.benchmark_metrics import STAGE2_BEST_HP_TRUE_DIR, STAGE2_DATA_DIR
from utils.nn_stage2_data import load_nn_stage2_data
from utils.stage2_export import export_stage2_data
from utils.stage2_hp import resolve_torch_device, tune_scenario, write_scenario_hp_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Minimal RF/MLP search (rf_n_iter=2, mlp max_trials=2) for all scenarios",
    )
    args = parser.parse_args()

    if not (STAGE2_DATA_DIR / "manifest.json").is_file():
        export_stage2_data(out_dir=STAGE2_DATA_DIR)
    data = load_nn_stage2_data()
    device = resolve_torch_device()
    STAGE2_BEST_HP_TRUE_DIR.mkdir(parents=True, exist_ok=True)

    rf_n_iter = 2 if args.smoke else 24
    mlp_max_trials = 2 if args.smoke else None
    logs: list[pd.DataFrame] = []
    for scen in ("A", "B", "C"):
        payload, mlp_log = tune_scenario(
            data,
            scen,  # type: ignore[arg-type]
            noise_label="true",
            rf_n_iter=rf_n_iter,
            mlp_max_trials=mlp_max_trials,
            device=device,
            verbose=True,
        )
        assert payload.get("noise_label") == "true"
        out = STAGE2_BEST_HP_TRUE_DIR / f"{scen}.json"
        write_scenario_hp_json(payload, out)
        print(f"Wrote {out}")
        logs.append(mlp_log)

    log_path = STAGE2_BEST_HP_TRUE_DIR / "hp_search_log_true.parquet"
    pd.concat(logs, ignore_index=True).to_parquet(log_path, index=False)
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()

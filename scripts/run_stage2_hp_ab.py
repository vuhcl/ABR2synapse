#!/usr/bin/env python3
"""Tune scenario A/B HP JSONs and write hp_search_log.parquet."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.benchmark_metrics import STAGE2_BEST_HP_DIR, STAGE2_DATA_DIR
from utils.nn_stage2_data import load_nn_stage2_data
from utils.stage2_export import export_stage2_data
from utils.stage2_hp import resolve_torch_device, tune_scenario_ab, write_scenario_hp_json


def main() -> None:
    if not (STAGE2_DATA_DIR / "manifest.json").is_file():
        export_stage2_data(out_dir=STAGE2_DATA_DIR)
    data = load_nn_stage2_data()
    device = resolve_torch_device()
    logs: list[pd.DataFrame] = []
    for scen in ("A", "B"):
        payload, mlp_log = tune_scenario_ab(data, scen, device=device, verbose=True)
        out = STAGE2_BEST_HP_DIR / f"{scen}.json"
        write_scenario_hp_json(payload, out)
        print(f"Wrote {out}")
        logs.append(mlp_log)
    log_path = STAGE2_BEST_HP_DIR / "hp_search_log.parquet"
    pd.concat(logs, ignore_index=True).to_parquet(log_path, index=False)
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Smoke checks for Stage 1 row-level pre-aggregation heatmap exports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.benchmark_metrics import repo_root
from utils.stage1_wide_report import (
    STAGE1_PREAGG_HEATMAP_STEM,
    STAGE1_PREAGG_HEATMAP_TEST_STEM,
    _with_canonical_frequency,
)

REQUIRED_ROW_COLS = {
    "lab",
    "animal_id",
    "frequency",
    "noise_group",
    "DataGroup",
    "y_true",
    "proba",
    "y_pred",
    "correct",
    "model",
    "row_threshold",
}
REQUIRED_GRID_COLS = {"lab", "noise_group", "frequency", "accuracy", "n"}


def _check_variant(
    errors: list[str],
    *,
    row_path: Path,
    grid_path: Path,
    png: Path,
    svg: Path,
    caption: Path,
    test_only: bool,
    metrics: dict | None = None,
) -> tuple[int, int, int, int] | None:
    for p in (row_path, grid_path, png, svg, caption):
        if not p.is_file():
            errors.append(f"missing artifact: {p}")
            return None

    row = pd.read_parquet(row_path)
    grid = pd.read_parquet(grid_path)
    missing = REQUIRED_ROW_COLS - set(row.columns)
    if missing:
        errors.append(f"{row_path.name} missing columns: {sorted(missing)}")
    missing_g = REQUIRED_GRID_COLS - set(grid.columns)
    if missing_g:
        errors.append(f"{grid_path.name} missing columns: {sorted(missing_g)}")

    if test_only and not row["DataGroup"].eq("Test").all():
        errors.append(f"{row_path.name}: expected all Test rows")

    labs = set(row["lab"].unique())
    if labs != {"Brad", "Liberman"}:
        errors.append(f"{row_path.name} unexpected labs: {labs}")

    brad_groups = set(row.loc[row["lab"].eq("Brad"), "noise_group"].unique())
    lib_groups = set(row.loc[row["lab"].eq("Liberman"), "noise_group"].unique())

    if (grid["accuracy"] < 0).any() or (grid["accuracy"] > 1).any():
        errors.append(f"{grid_path.name} accuracy out of [0,1]")

    if grid["frequency"].astype(float).eq(45.3).any():
        errors.append(f"{grid_path.name} still contains 45.3 kHz")
    if grid["frequency"].astype(float).eq(5.7).any():
        errors.append(f"{grid_path.name} still contains 5.7 kHz")

    expected_cells = 0
    for lab in ("Liberman", "Brad"):
        sub = _with_canonical_frequency(row[row["lab"].eq(lab)])
        cell_n = sub.groupby(["noise_group", "frequency"]).size()
        expected_cells += int((cell_n > 0).sum())
    if len(grid) != expected_cells:
        errors.append(
            f"{grid_path.name} row count {len(grid)} != cells {expected_cells}"
        )

    if not test_only and metrics is not None:
        tol = 1e-4
        for lab, mkey in (("Brad", "Brad"), ("Liberman", "Lib")):
            train = row[(row["lab"].eq(lab)) & (row["DataGroup"].eq("Train"))]
            if train.empty:
                errors.append(f"{lab}: no Train rows in {row_path.name}")
                continue
            got = float(train["correct"].mean())
            exp = float(metrics[mkey]["fit_acc_row"])
            if abs(got - exp) > tol:
                errors.append(
                    f"{lab} Train row acc {got:.6f} != fit_acc_row {exp:.6f}"
                )

    return len(row), len(grid), len(brad_groups), len(lib_groups)


def main() -> int:
    errors: list[str] = []
    cache = repo_root() / "figures" / "cache"
    pres = repo_root() / "figures" / "presentation"
    metrics_path = cache / "stage1_wide_metrics.json"

    if not metrics_path.is_file():
        errors.append(f"missing artifact: {metrics_path}")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}

    all_stats = _check_variant(
        errors,
        row_path=cache / "stage1_wide_row_eval.parquet",
        grid_path=cache / "stage1_wide_row_eval_grid.parquet",
        png=pres / f"{STAGE1_PREAGG_HEATMAP_STEM}.png",
        svg=pres / f"{STAGE1_PREAGG_HEATMAP_STEM}.svg",
        caption=pres / f"{STAGE1_PREAGG_HEATMAP_STEM}_caption.txt",
        test_only=False,
        metrics=metrics,
    )
    test_stats = _check_variant(
        errors,
        row_path=cache / "stage1_wide_row_eval_test.parquet",
        grid_path=cache / "stage1_wide_row_eval_grid_test.parquet",
        png=pres / f"{STAGE1_PREAGG_HEATMAP_TEST_STEM}.png",
        svg=pres / f"{STAGE1_PREAGG_HEATMAP_TEST_STEM}.svg",
        caption=pres / f"{STAGE1_PREAGG_HEATMAP_TEST_STEM}_caption.txt",
        test_only=True,
    )

    meta_path = cache / "stage1_wide_noise_lr_rf_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if "row_eval_protocol" not in meta:
            errors.append("meta missing row_eval_protocol")
        if "row_eval_protocol_test" not in meta:
            errors.append("meta missing row_eval_protocol_test")
    else:
        errors.append(f"missing artifact: {meta_path}")

    if all_stats and all_stats[2] != 4:
        errors.append(f"Brad expected 4 noise_group, got {all_stats[2]}")
    if all_stats and all_stats[3] != 14:
        errors.append(f"Liberman expected 14 noise_group, got {all_stats[3]}")

    if errors:
        for e in errors:
            print("ERROR:", e)
        print("Run: uv run python scripts/run_stage1_export.py")
        return 1

    print("OK smoke_stage1_heatmap")
    if all_stats:
        print(f"  all rows: {all_stats[0]} | grid cells: {all_stats[1]}")
    if test_stats:
        print(f"  test rows: {test_stats[0]} | grid cells: {test_stats[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage-1 wide RF/LR exports and summary tables for deck/benchmarks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from utils.benchmark_metrics import repo_root


def _s1_animal_eval_rows(lab: str, model: str, s1_out: dict) -> pd.DataFrame:
    y = s1_out["animal_y_te"]
    s = s1_out["animal_prob_te"]
    t = float(s1_out["stage1_threshold"])
    scores = s.reindex(y.index).values.astype(np.float64)
    return pd.DataFrame(
        {
            "lab": lab,
            "model": model,
            "animal_id": y.index.astype(str),
            "y_true": y.values.astype(np.int8),
            "score": scores,
            "threshold": t,
            "y_pred": (scores > t).astype(np.int8),
        }
    )


def _metrics_block(s1: dict) -> dict:
    return {
        "fit_acc_row": float(s1["s1_fit_acc_row"]),
        "fit_acc_animal": float(s1["s1_fit_acc_animal"]),
        "val_acc_pre": float(s1["s1_val_acc_pre"]),
        "val_acc_animal": float(s1["s1_val_acc_animal"]),
        "val_acc_animal_val_youden": float(s1["s1_val_acc_animal_val_youden"]),
        "non_test_acc_animal": float(s1["s1_non_test_acc_animal"]),
        "non_test_auc_animal": float(s1["s1_non_test_auc_animal"]),
        "non_test_label_protocol": "animal-level (mean row proba → threshold → broadcast)",
        "threshold_youden": float(s1["stage1_threshold"]),
        "threshold_youden_val_only": float(s1["stage1_threshold_val_youden"]),
        "test_acc_post": float(s1["s1_test_acc"]),
        "test_acc_post_0p5": float(s1["s1_test_acc_0p5"]),
        "test_auc_post": float(s1["s1_test_auc"]),
    }


def stage1_summary_table(candidates: Mapping[str, dict], selected: str) -> pd.DataFrame:
    """RF / LR / selected: fit/val/non-test/test metrics at calibrated threshold."""
    rows = []
    for name, s1 in candidates.items():
        rows.append(
            {
                "model": name.upper(),
                "fit_acc_row": float(s1["s1_fit_acc_row"]),
                "fit_acc_animal": float(s1["s1_fit_acc_animal"]),
                "val_acc_row": float(s1["s1_val_acc_pre"]),
                "val_acc_animal": float(s1["s1_val_acc_animal"]),
                "val_acc_animal_val_thr": float(s1["s1_val_acc_animal_val_youden"]),
                "non_test_acc_animal": float(s1["s1_non_test_acc_animal"]),
                "non_test_auc_animal": float(s1["s1_non_test_auc_animal"]),
                "threshold_fit_val_youden": float(s1["stage1_threshold"]),
                "threshold_val_only_youden": float(s1["stage1_threshold_val_youden"]),
                "test_acc_animal": float(s1["s1_test_acc"]),
                "test_acc_animal_0p5": float(s1["s1_test_acc_0p5"]),
                "test_auc_animal": float(s1["s1_test_auc"]),
                "selected": name == selected,
            }
        )
    return pd.DataFrame(rows)


def export_stage1_artifacts(
    bb_best: dict,
    lib_best: dict,
    bb_rf: dict,
    bb_lr: dict,
    lib_rf: dict,
    lib_lr: dict,
    *,
    cache_dir: Path | str | None = None,
    sklearn_version: str = "",
) -> dict[str, Path]:
    cache = Path(cache_dir or repo_root() / "figures" / "cache")
    cache.mkdir(parents=True, exist_ok=True)

    metrics = {
        "Brad": {
            "selected_model": bb_best["stage1_model"],
            **_metrics_block(bb_best),
            "rf": _metrics_block(bb_rf),
            "lr": _metrics_block(bb_lr),
        },
        "Lib": {
            "selected_model": lib_best["stage1_model"],
            **_metrics_block(lib_best),
            "rf": _metrics_block(lib_rf),
            "lr": _metrics_block(lib_lr),
        },
    }
    metrics_path = cache / "stage1_wide_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    shim = {
        "Brad": (
            metrics["Brad"]["val_acc_pre"],
            metrics["Brad"]["test_acc_post"],
            metrics["Brad"]["test_auc_post"],
        ),
        "Lib": (
            metrics["Lib"]["val_acc_pre"],
            metrics["Lib"]["test_acc_post"],
            metrics["Lib"]["test_auc_post"],
        ),
    }
    shim_path = cache / "stage1_wide_rf_metrics.json"
    with shim_path.open("w") as f:
        json.dump({k: list(v) for k, v in shim.items()}, f, indent=2)

    eval_df = pd.concat(
        [
            _s1_animal_eval_rows("Brad", "LR", bb_lr),
            _s1_animal_eval_rows("Brad", "RF", bb_rf),
            _s1_animal_eval_rows("Liberman", "LR", lib_lr),
            _s1_animal_eval_rows("Liberman", "RF", lib_rf),
        ],
        ignore_index=True,
    )
    eval_path = cache / "stage1_wide_noise_lr_rf_eval.parquet"
    eval_df.to_parquet(eval_path, index=False)

    meta = {
        "sklearn_version": sklearn_version,
        "protocol": (
            "Train fit / Validate row-acc HP / "
            "Youden J threshold on Fit+Validate animals (full calibration) / "
            "Test animal acc+AUC at calibrated threshold"
        ),
        "Brad_selected": bb_best["stage1_model"],
        "Lib_selected": lib_best["stage1_model"],
    }
    meta_path = cache / "stage1_wide_noise_lr_rf_meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return {
        "metrics": metrics_path,
        "shim": shim_path,
        "eval_parquet": eval_path,
        "meta": meta_path,
    }


def load_stage1_metrics(
    cache_dir: Path | str | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Return {Brad|Lib: (val_acc_pre, test_acc_post, test_auc_post)} for bar charts."""
    cache = Path(cache_dir or repo_root() / "figures" / "cache")
    path = cache / "stage1_wide_metrics.json"
    if path.is_file():
        raw = json.loads(path.read_text())
        out = {}
        for lab in ("Brad", "Lib"):
            block = raw[lab]
            out[lab] = (
                float(block["val_acc_pre"]),
                float(block["test_acc_post"]),
                float(block["test_auc_post"]),
            )
        return out
    shim = cache / "stage1_wide_rf_metrics.json"
    if shim.is_file():
        raw = json.loads(shim.read_text())
        return {k: tuple(float(x) for x in v) for k, v in raw.items()}
    return {}

#!/usr/bin/env python3
"""Run canonical Stage 1 fits and write figures/cache artifacts."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import sklearn

from utils.nn_stage2 import fit_stage1_wide_best, fit_stage1_wide_logistic, fit_stage1_wide_rf
from utils.nn_stage2_data import load_nn_stage2_data, splits_for_long_stage2, wide_stage1_fit, wide_stage1_val
from utils.stage1_wide_report import export_stage1_artifacts, export_stage1_row_heatmap

data = load_nn_stage2_data()
sp = splits_for_long_stage2(data)
bb_tr, bb_te = sp["bb_wide_train"], sp["bb_wide_test"]
lib_tr, lib_te = sp["lib_train"], sp["lib_test"]

bb_fit, bb_val = wide_stage1_fit(bb_tr), wide_stage1_val(bb_tr)
lib_fit, lib_val = wide_stage1_fit(lib_tr), wide_stage1_val(lib_tr)

s1_bb_rf = fit_stage1_wide_rf(bb_fit, bb_val, bb_te, data.noise_num_bb, data.noise_log_bb, verbose=True)
s1_bb_lr = fit_stage1_wide_logistic(bb_fit, bb_val, bb_te, data.noise_num_bb, data.noise_log_bb, verbose=True)
s1_bb_best = fit_stage1_wide_best(bb_fit, bb_val, bb_te, data.noise_num_bb, data.noise_log_bb, verbose=True)

s1_lib_rf = fit_stage1_wide_rf(lib_fit, lib_val, lib_te, data.noise_num_lib, data.noise_log_lib, verbose=True)
s1_lib_lr = fit_stage1_wide_logistic(lib_fit, lib_val, lib_te, data.noise_num_lib, data.noise_log_lib, verbose=True)
s1_lib_best = fit_stage1_wide_best(lib_fit, lib_val, lib_te, data.noise_num_lib, data.noise_log_lib, verbose=True)

paths = export_stage1_artifacts(
    s1_bb_best, s1_lib_best, s1_bb_rf, s1_bb_lr, s1_lib_rf, s1_lib_lr,
    sklearn_version=sklearn.__version__,
)
heatmap_paths = export_stage1_row_heatmap(
    s1_bb_best,
    s1_lib_best,
    pd.concat([bb_tr, bb_te], ignore_index=True),
    pd.concat([lib_tr, lib_te], ignore_index=True),
    list(data.noise_num_bb) + list(data.noise_log_bb),
    list(data.noise_num_lib) + list(data.noise_log_lib),
    sklearn_version=sklearn.__version__,
    meta_path=paths["meta"],
)
heatmap_test_paths = export_stage1_row_heatmap(
    s1_bb_best,
    s1_lib_best,
    pd.concat([bb_tr, bb_te], ignore_index=True),
    pd.concat([lib_tr, lib_te], ignore_index=True),
    list(data.noise_num_bb) + list(data.noise_log_bb),
    list(data.noise_num_lib) + list(data.noise_log_lib),
    sklearn_version=sklearn.__version__,
    meta_path=paths["meta"],
    test_only=True,
)
for k, p in {**paths, **heatmap_paths, **heatmap_test_paths}.items():
    print(k, p)

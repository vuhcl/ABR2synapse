"""
Export Liberman long-format Stage-2 NN tensors for Colab GPU training.

Writes preprocessed tabular features (fitted on Train rows only), Wave-I windows,
full 0–8 ms waveforms, targets, and row metadata under ``figures/cache/nn_colab_liberman/``.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.nn_stage2 import (
    RunConfig,
    attach_noise_preds_long,
    fit_stage1_wide_best,
    prepare_long_xy,
    syn_prep_transformer,
)
from utils.nn_stage2_data import (
    NN_TRAIN_RANDOM_STATE,
    NNStage2Data,
    full_wave_matrix,
    load_nn_stage2_data,
    splits_for_long_stage2,
    unified_grid_wave_i_matrix,
    wide_stage1_fit,
    wide_stage1_val,
)

from utils.liberman_classical import (
    NoiseLabel,
    animal_noise_series,
    attach_animal_noise_cat,
)
from utils.nn_colab_train import (
    DROPOUT_GRID,
    LR_GRID,
    MLP_GRID_SIZE,
    MLP_HIDDEN_GRID,
    WEIGHT_DECAY_GRID,
)

DEFAULT_OUT = Path("figures/cache/nn_colab_liberman")
DEFAULT_TRUE_OUT = Path("figures/cache/nn_colab_liberman_true")


def _stage2_long_frames(
    data: NNStage2Data,
    *,
    exclude_strain: bool = False,
    noise_label: NoiseLabel = "predicted",
):
    sp = splits_for_long_stage2(data)
    lib_tr = sp["lib_train"].copy()
    lib_te = sp["lib_test"].copy()
    long_tr = sp["lib_long_train"].copy()
    long_te = sp["lib_long_test"].copy()

    nn_data = data
    if exclude_strain:
        nn_data = replace(data, long_num=[c for c in data.long_num if c != "strain_binary"])

    if noise_label == "true":
        animal_noise = animal_noise_series(data.orig_lib)
        long_tr = attach_animal_noise_cat(long_tr, animal_noise)
        long_te = attach_animal_noise_cat(long_te, animal_noise)
        long_cat = ["noise_cat"]
    else:
        s1_fit = wide_stage1_fit(lib_tr)
        s1_val = wide_stage1_val(lib_tr)
        s1 = fit_stage1_wide_best(
            s1_fit,
            s1_val,
            lib_te,
            data.noise_num_lib,
            data.noise_log_lib,
            random_state=NN_TRAIN_RANDOM_STATE,
            verbose=True,
        )
        long_tr = attach_noise_preds_long(
            long_tr,
            s1["animal_pred_non_test"],
            require_full_coverage=True,
        )
        long_te = attach_noise_preds_long(long_te, s1["animal_pred_te"])
        long_cat = list(nn_data.long_cat)

    long_fit = long_tr[long_tr["DataGroup"] == "Train"].reset_index(drop=True)
    long_val = long_tr[long_tr["DataGroup"] == "Validate"].reset_index(drop=True)
    long_te = long_te.reset_index(drop=True)

    fit_animals = set(long_fit["animal_id"].unique())
    val_animals = set(long_val["animal_id"].unique())
    if fit_animals & val_animals:
        raise ValueError("Train and Validate animals must be disjoint")

    return nn_data, long_fit, long_val, long_te, long_cat


def _pack_split(
    long_df: pd.DataFrame,
    idx: pd.Index,
    X_tab: np.ndarray,
    wave_len: int,
    *,
    row_offset: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta = long_df.loc[idx, ["animal_id", "frequency", "synapses", "DataGroup"]].copy()
    meta = meta.reset_index(drop=True)
    meta.insert(0, "row_id", np.arange(row_offset, row_offset + len(meta), dtype=np.int64))
    wave_i = unified_grid_wave_i_matrix(long_df, idx, wave_len)
    wave_full = full_wave_matrix(long_df, idx, wave_len)
    return meta, wave_i, wave_full


def export_liberman_nn_colab_pack(
    out_dir: Path | str | None = None,
    *,
    data: NNStage2Data | None = None,
    cfg: RunConfig | None = None,
    exclude_strain: bool = False,
    noise_label: NoiseLabel = "predicted",
) -> Path:
    """Build Colab artifact directory; returns ``out_dir``."""
    if out_dir is None:
        out_dir = DEFAULT_TRUE_OUT if noise_label == "true" else DEFAULT_OUT
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = data or load_nn_stage2_data(join_io_features=False)
    cfg = cfg or RunConfig(full_wave_ref="lib")
    nn_data, long_fit, long_val, long_te, long_cat = _stage2_long_frames(
        data, exclude_strain=exclude_strain, noise_label=noise_label
    )

    long_num = nn_data.long_num
    long_log = nn_data.long_log
    prep = syn_prep_transformer(long_num, long_cat, long_log)

    tr_idx, X_tr, y_tr = prepare_long_xy(long_fit, long_num, long_cat, long_log, prep, fit=True)
    va_idx, X_va, y_va = prepare_long_xy(long_val, long_num, long_cat, long_log, prep, fit=False)
    te_idx, X_te, y_te = prepare_long_xy(long_te, long_num, long_cat, long_log, prep, fit=False)

    wave_len = nn_data.full_wave_target_len_lib
    meta_tr, wi_tr, wf_tr = _pack_split(long_fit, tr_idx, X_tr, wave_len, row_offset=0)
    meta_va, wi_va, wf_va = _pack_split(
        long_val, va_idx, X_va, wave_len, row_offset=len(meta_tr)
    )
    meta_te, wi_te, wf_te = _pack_split(
        long_te, te_idx, X_te, wave_len, row_offset=len(meta_tr) + len(meta_va)
    )

    tab_cols = [f"x_{i:03d}" for i in range(X_tr.shape[1])]
    for meta, X in ((meta_tr, X_tr), (meta_va, X_va), (meta_te, X_te)):
        for j, c in enumerate(tab_cols):
            meta[c] = X[:, j].astype(np.float32)

    meta_tr.to_parquet(out_dir / "train.parquet", index=False)
    meta_va.to_parquet(out_dir / "validate.parquet", index=False)
    meta_te.to_parquet(out_dir / "test.parquet", index=False)

    np.save(out_dir / "wave_i_train.npy", wi_tr)
    np.save(out_dir / "wave_i_validate.npy", wi_va)
    np.save(out_dir / "wave_i_test.npy", wi_te)
    np.save(out_dir / "wave_full_train.npy", wf_tr)
    np.save(out_dir / "wave_full_validate.npy", wf_va)
    np.save(out_dir / "wave_full_test.npy", wf_te)

    manifest: dict[str, Any] = {
        "cohort": "liberman",
        "noise_label": noise_label,
        "exclude_strain": exclude_strain,
        "n_tabular": int(X_tr.shape[1]),
        "tabular_columns": tab_cols,
        "wave_i_len": 30,
        "wave_full_len": int(wave_len),
        "random_state": int(cfg.random_state),
        "hp_holdout_frac": 0.1,
        "counts": {
            "train_rows": int(len(meta_tr)),
            "validate_rows": int(len(meta_va)),
            "test_rows": int(len(meta_te)),
            "train_animals": int(meta_tr["animal_id"].nunique()),
            "validate_animals": int(meta_va["animal_id"].nunique()),
            "test_animals": int(meta_te["animal_id"].nunique()),
        },
        "training": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "huber_delta": 1.0,
            "early_stop_patience": 15,
            "optimizer": "AdamW",
        },
        "hp_grids": {
            "mlp": {
                "hidden": [list(h) for h in MLP_HIDDEN_GRID],
                "dropout": DROPOUT_GRID,
                "lr": LR_GRID,
                "weight_decay": WEIGHT_DECAY_GRID,
                "search": "full_grid",
                "n_configs": MLP_GRID_SIZE,
            },
            "cnn": {
                "dropout": DROPOUT_GRID,
                "lr": LR_GRID,
                "weight_decay": WEIGHT_DECAY_GRID,
            },
        },
        "files": {
            "train": "train.parquet",
            "validate": "validate.parquet",
            "test": "test.parquet",
            "wave_i_train": "wave_i_train.npy",
            "wave_i_validate": "wave_i_validate.npy",
            "wave_i_test": "wave_i_test.npy",
            "wave_full_train": "wave_full_train.npy",
            "wave_full_validate": "wave_full_validate.npy",
            "wave_full_test": "wave_full_test.npy",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    train_py = Path(__file__).resolve().parent / "nn_colab_train.py"
    if train_py.is_file():
        shutil.copy2(train_py, out_dir / "nn_colab_train.py")

    print(f"Exported Colab pack → {out_dir.resolve()}")
    print(
        f"  train {manifest['counts']['train_rows']} rows, "
        f"{manifest['counts']['train_animals']} animals | "
        f"validate {manifest['counts']['validate_rows']} | "
        f"test {manifest['counts']['test_rows']}"
    )
    print(f"  tabular dim={manifest['n_tabular']}, wave_full_len={wave_len}")
    return out_dir

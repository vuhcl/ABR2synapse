"""Liberman classical synapse comparison panel (linear + RF/XGB trees)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import r2_score

from utils.nn_stage2 import attach_noise_preds_long, fit_stage1_wide_best
from utils.nn_stage2_data import (
    LONG_CAT,
    LONG_LOG,
    LONG_NUM_BASE,
    STAGE_SPL_LEVELS,
    assert_wide_feats_stage_spl,
    wide_stage1_fit,
    wide_stage1_val,
    _noise_feats_from_wide,
    syn_feats_from_wide_common,
)
from utils.stage2_sklearn import (
    _fit_stage2_rf,
    _fit_stage2_xgb,
    long_stage2_test_agg,
    metrics_from_animal_frequency_agg,
    run_two_stage_long_s2,
    run_two_stage_long_s2_xgb,
    run_two_stage_model,
    run_two_stage_model_xgb_wide,
    wide_stage2_test_agg,
)

EXCLUDE_S2_EXTRA = frozenset({"p1_latency", "Slope_all", "Slope_high4"})

NoiseLabel = Literal["none", "predicted", "true"]
FormatLabel = Literal["long", "wide"]


@dataclass(frozen=True)
class TreeConfig:
    config_id: str
    format: FormatLabel
    noise_label: NoiseLabel


TREE_CONFIGS: Tuple[TreeConfig, ...] = (
    TreeConfig("T1", "long", "none"),
    TreeConfig("T3", "long", "predicted"),
    TreeConfig("T4", "wide", "none"),
    TreeConfig("T5", "wide", "predicted"),
    TreeConfig("T6", "wide", "true"),
)


def attach_animal_noise_cat(df: pd.DataFrame, animal_noise: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["noise_cat"] = out["animal_id"].map(animal_noise).astype(int)
    return out


def animal_noise_series(orig_long: pd.DataFrame) -> pd.Series:
    return orig_long.groupby("animal_id")["noise_cat"].first().round().astype(int)


def filter_syn_num(syn_num: List[str]) -> List[str]:
    return [c for c in syn_num if c not in EXCLUDE_S2_EXTRA]


def liberman_feature_lists(
    reformatted_orig: pd.DataFrame,
    common_cols: list | tuple,
) -> Dict[str, Any]:
    syn_num, syn_log, syn_cat = syn_feats_from_wide_common(
        common_cols, has_strain=False
    )
    syn_num = filter_syn_num(syn_num)
    noise_num, noise_log = _noise_feats_from_wide(
        reformatted_orig.columns, include_extra_wide=False
    )
    assert_wide_feats_stage_spl(syn_num + syn_log + noise_num + noise_log)
    return {
        "syn_num": syn_num,
        "syn_log": syn_log,
        "syn_cat_default": syn_cat,
        "noise_num": noise_num,
        "noise_log": noise_log,
        "long_num_base": list(LONG_NUM_BASE),
        "long_log": list(LONG_LOG),
        "long_cat_default": list(LONG_CAT),
        "stage_spl_levels": list(STAGE_SPL_LEVELS),
    }


def stage2_feature_lists(
    feats: Dict[str, Any], *, noise_label: NoiseLabel
) -> Tuple[List[str], List[str], List[str], List[str]]:
    syn_num = list(feats["syn_num"])
    long_num = list(feats["long_num_base"])

    if noise_label == "none":
        syn_cat: List[str] = []
        long_cat: List[str] = []
    elif noise_label == "predicted":
        syn_cat = ["noise_preds"]
        long_cat = ["noise_preds"]
    else:
        syn_cat = ["noise_cat"]
        long_cat = ["noise_cat"]

    return syn_num, list(feats["syn_log"]), syn_cat, long_num, long_cat


def resolve_amp80_column(columns: pd.Index) -> Optional[str]:
    if "amplitude_80.0" in columns:
        return "amplitude_80.0"
    cands = [c for c in columns if str(c).startswith("amplitude_8")]
    return sorted(cands)[-1] if cands else None


def ols_wide_amp80(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[float, float]:
    amp = resolve_amp80_column(train_df.columns)
    if amp is None:
        return float("nan"), float("nan")
    model = smf.ols(f"synapses ~ Q('{amp}')", data=train_df).fit()
    pred = model.predict(test_df)
    r2 = float(r2_score(test_df["synapses"], pred))
    rmse = float(np.sqrt(np.mean((test_df["synapses"].values - pred.values) ** 2)))
    return r2, rmse


def ols_wide_full_noise_pred(
    wide_train: pd.DataFrame,
    wide_test: pd.DataFrame,
    syn_num: List[str],
    syn_log: List[str],
    noise_num: List[str],
    noise_log: List[str],
) -> Tuple[float, float]:
    s1 = fit_stage1_wide_best(
        wide_stage1_fit(wide_train),
        wide_stage1_val(wide_train),
        wide_test,
        noise_num,
        noise_log,
        verbose=False,
    )
    tr = attach_noise_preds_long(
        wide_train, s1["animal_pred_non_test"], require_full_coverage=True
    )
    te = attach_noise_preds_long(wide_test, s1["animal_pred_te"])

    num_cols = list(syn_num)
    formula = "synapses ~ C(noise_preds)"
    for col in num_cols + syn_log:
        formula += f" + Q('{col}')"
    dropna_cols = ["noise_preds"] + num_cols + syn_log
    tr_clean = tr.dropna(subset=dropna_cols + ["synapses"])
    te_clean = te.dropna(subset=dropna_cols + ["synapses"])
    model = smf.ols(formula, data=tr_clean).fit()
    pred = model.predict(te_clean)
    r2 = float(r2_score(te_clean["synapses"], pred))
    rmse = float(np.sqrt(np.mean((te_clean["synapses"].values - pred.values) ** 2)))
    return r2, rmse


def run_tree_config(
    cfg: TreeConfig,
    *,
    model: Literal["RF", "XGB"],
    wide_train: pd.DataFrame,
    wide_test: pd.DataFrame,
    long_train: pd.DataFrame,
    long_test: pd.DataFrame,
    feats: Dict[str, Any],
    verbose: bool = False,
) -> Tuple[float, float]:
    syn_num, syn_log, syn_cat, long_num, long_cat = stage2_feature_lists(
        feats, noise_label=cfg.noise_label
    )
    long_log = list(feats["long_log"])
    noise_num = feats["noise_num"]
    noise_log = feats["noise_log"]
    label = f"{cfg.config_id}-{model}"

    if cfg.format == "long":
        if verbose:
            runner = run_two_stage_long_s2 if model == "RF" else run_two_stage_long_s2_xgb
            return runner(
                wide_train,
                wide_test,
                long_train,
                long_test,
                label,
                noise_num=noise_num,
                noise_log=noise_log,
                long_num=long_num,
                long_cat=long_cat,
                long_log=long_log,
            )
        agg = long_stage2_test_agg(
            wide_train,
            wide_test,
            long_train,
            long_test,
            noise_num=noise_num,
            noise_log=noise_log,
            long_num=long_num,
            long_cat=long_cat,
            long_log=long_log,
            model=model,
        )
        return metrics_from_animal_frequency_agg(agg)

    agg = wide_stage2_test_agg(
        wide_train,
        wide_test,
        noise_num=noise_num,
        noise_log=noise_log,
        syn_num=syn_num,
        syn_cat=syn_cat,
        syn_log=syn_log,
        model=model,
    )
    return metrics_from_animal_frequency_agg(agg)


def append_result(
    rows: List[Dict[str, Any]],
    *,
    config_id: str,
    model: str,
    format: str,
    noise_label: str,
    r2_test: float,
    rmse_test: float,
) -> None:
    rows.append(
        {
            "config_id": config_id,
            "model": model,
            "format": format,
            "noise_label": noise_label,
            "r2_test": r2_test,
            "rmse_test": rmse_test,
        }
    )


def _json_numpy_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_numpy_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_numpy_sanitize(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def export_t5_stage2_hp(
    wide_train: pd.DataFrame,
    wide_test: pd.DataFrame,
    feats: Dict[str, Any],
    *,
    out_path: Optional[Path] = None,
    rf_random_state: int = 1,
) -> Dict[str, Any]:
    """
    Run T5 wide RF + XGB HP search (Stage 1 + predicted noise) and write best params.

    Default ``out_path``: ``figures/cache/stage2_best_hp/liberman_t5_sklearn.json``.
    """
    cfg = TreeConfig("T5", "wide", "predicted")
    syn_num, syn_log, syn_cat, _, _ = stage2_feature_lists(feats, noise_label=cfg.noise_label)
    noise_num = list(feats["noise_num"])
    noise_log = list(feats["noise_log"])

    s1_out = fit_stage1_wide_best(
        wide_stage1_fit(wide_train),
        wide_stage1_val(wide_train),
        wide_test,
        noise_num,
        noise_log,
        random_state=1,
        verbose=False,
    )
    tr_aug = attach_noise_preds_long(
        wide_train,
        s1_out["animal_pred_non_test"],
        fallback_noise_cat=False,
        require_full_coverage=True,
    )
    _, rf_cv, rf_bp = _fit_stage2_rf(
        tr_aug, syn_num, syn_cat, syn_log, random_state=rf_random_state
    )
    _, xgb_cv, xgb_bp = _fit_stage2_xgb(tr_aug, syn_num, syn_cat, syn_log)

    payload: Dict[str, Any] = {
        "config_id": cfg.config_id,
        "format": cfg.format,
        "noise_label": cfg.noise_label,
        "RF": {"best_params": _json_numpy_sanitize(rf_bp), "cv_score": rf_cv},
        "XGB": {"best_params": _json_numpy_sanitize(xgb_bp), "cv_score": xgb_cv},
    }
    if out_path is None:
        out_path = (
            Path(__file__).resolve().parent.parent
            / "figures"
            / "cache"
            / "stage2_best_hp"
            / "liberman_t5_sklearn.json"
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_json_numpy_sanitize(payload), indent=2) + "\n", encoding="utf-8"
    )
    return payload


def run_config_panel(
    wide_train: pd.DataFrame,
    wide_test: pd.DataFrame,
    long_train: pd.DataFrame,
    long_test: pd.DataFrame,
    feats: Dict[str, Any],
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for cfg in TREE_CONFIGS:
        for model in ("RF", "XGB"):
            if verbose:
                print(f"\n=== {cfg.config_id} {model} ({cfg.format}, noise={cfg.noise_label}) ===")
            r2, rmse = run_tree_config(
                cfg,
                model=model,
                wide_train=wide_train,
                wide_test=wide_test,
                long_train=long_train,
                long_test=long_test,
                feats=feats,
                verbose=verbose,
            )
            append_result(
                rows,
                config_id=cfg.config_id,
                model=model,
                format=cfg.format,
                noise_label=cfg.noise_label,
                r2_test=r2,
                rmse_test=rmse,
            )

    syn_num = list(feats["syn_num"])
    if verbose:
        print("\n=== L7 OLS amp80 (wide) ===")
    r2_l7, rmse_l7 = ols_wide_amp80(wide_train, wide_test)
    append_result(
        rows,
        config_id="L7",
        model="OLS",
        format="wide",
        noise_label="none",
        r2_test=r2_l7,
        rmse_test=rmse_l7,
    )

    if verbose:
        print("\n=== L8 OLS full + noise_preds (wide) ===")
    r2_l8, rmse_l8 = ols_wide_full_noise_pred(
        wide_train,
        wide_test,
        syn_num,
        feats["syn_log"],
        feats["noise_num"],
        feats["noise_log"],
    )
    append_result(
        rows,
        config_id="L8",
        model="OLS",
        format="wide",
        noise_label="predicted",
        r2_test=r2_l8,
        rmse_test=rmse_l8,
    )

    return pd.DataFrame(rows)


def _row(
    df: pd.DataFrame, config_id: str, model: str
) -> Optional[pd.Series]:
    sub = df[(df["config_id"] == config_id) & (df["model"] == model)]
    if sub.empty:
        return None
    return sub.iloc[0]


COMPARISON_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("Q3_format", "T1", "T4"),
    ("Q1_noise", "T4", "T5"),
    ("Q1_noise_long", "T1", "T3"),
    ("Q2_oracle", "T5", "T6"),
)


def _unit_index(df: pd.DataFrame) -> pd.Index:
    return (
        df["animal_id"].astype(str) + "|" + df["frequency"].astype(float).astype(str)
    )


def tree_config_test_sq_errors(
    cfg: TreeConfig,
    *,
    model: Literal["RF", "XGB"],
    wide_train: pd.DataFrame,
    wide_test: pd.DataFrame,
    long_train: pd.DataFrame,
    long_test: pd.DataFrame,
    feats: Dict[str, Any],
) -> pd.Series:
    """Per animal×frequency squared test error (aligned unit index)."""
    syn_num, syn_log, syn_cat, long_num, long_cat = stage2_feature_lists(
        feats, noise_label=cfg.noise_label
    )
    noise_num = list(feats["noise_num"])
    noise_log = list(feats["noise_log"])
    long_log = list(feats["long_log"])

    if cfg.format == "long":
        agg = long_stage2_test_agg(
            wide_train,
            wide_test,
            long_train,
            long_test,
            noise_num=noise_num,
            noise_log=noise_log,
            long_num=long_num,
            long_cat=long_cat,
            long_log=long_log,
            model=model,
        )
    else:
        agg = wide_stage2_test_agg(
            wide_train,
            wide_test,
            noise_num=noise_num,
            noise_log=noise_log,
            syn_num=syn_num,
            syn_cat=syn_cat,
            syn_log=syn_log,
            model=model,
        )
    sq = (agg["y_true"].astype(float) - agg["y_pred"].astype(float)) ** 2
    return pd.Series(sq.values, index=_unit_index(agg), name="sq_err")


def build_sq_error_cache(
    wide_train: pd.DataFrame,
    wide_test: pd.DataFrame,
    long_train: pd.DataFrame,
    long_test: pd.DataFrame,
    feats: Dict[str, Any],
    *,
    models: Tuple[str, ...] = ("RF", "XGB"),
) -> Dict[Tuple[str, str], pd.Series]:
    cache: Dict[Tuple[str, str], pd.Series] = {}
    for cfg in TREE_CONFIGS:
        for model in models:
            key = (cfg.config_id, model)
            cache[key] = tree_config_test_sq_errors(
                cfg,
                model=model,  # type: ignore[arg-type]
                wide_train=wide_train,
                wide_test=wide_test,
                long_train=long_train,
                long_test=long_test,
                feats=feats,
            )
    return cache


def _f_test_equal_variance(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Two-sided F-test for equal variances (ddof=1)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    if v1 == 0 and v2 == 0:
        return 1.0, 1.0
    if v1 >= v2:
        f_stat, dfn, dfd = v1 / v2, n1 - 1, n2 - 1
    else:
        f_stat, dfn, dfd = v2 / v1, n2 - 1, n1 - 1
    cdf = float(stats.f.cdf(f_stat, dfn, dfd))
    p_two = float(np.clip(2 * min(cdf, 1 - cdf), 0.0, 1.0))
    return float(f_stat), p_two


def paired_sq_error_inference(
    sq_a: np.ndarray,
    sq_b: np.ndarray,
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compare paired squared errors (config_a vs config_b).

    F-test on variances; Welch t-test if unequal, else paired t-test on the two vectors.
    Unit of analysis: animal×frequency (same pairing for long vs wide).
    """
    sq_a = np.asarray(sq_a, dtype=float)
    sq_b = np.asarray(sq_b, dtype=float)
    if len(sq_a) != len(sq_b):
        raise ValueError(f"paired length mismatch: {len(sq_a)} vs {len(sq_b)}")
    if len(sq_a) < 2:
        return {
            "n_units": len(sq_a),
            "mean_sq_err_a": float("nan"),
            "mean_sq_err_b": float("nan"),
            "mean_sq_err_diff_a_minus_b": float("nan"),
            "f_stat": float("nan"),
            "f_pvalue": float("nan"),
            "variances_unequal": False,
            "t_test": "insufficient_n",
            "t_stat": float("nan"),
            "t_pvalue": float("nan"),
            "significant_at_alpha": False,
        }

    f_stat, f_p = _f_test_equal_variance(sq_a, sq_b)
    unequal = bool(np.isfinite(f_p) and f_p < alpha)
    if unequal:
        t_res = stats.ttest_ind(sq_a, sq_b, equal_var=False)
        test_name = "welch"
    else:
        t_res = stats.ttest_rel(sq_a, sq_b)
        test_name = "paired"

    t_p = float(t_res.pvalue)
    return {
        "n_units": int(len(sq_a)),
        "mean_sq_err_a": float(np.mean(sq_a)),
        "mean_sq_err_b": float(np.mean(sq_b)),
        "mean_sq_err_diff_a_minus_b": float(np.mean(sq_a - sq_b)),
        "f_stat": f_stat,
        "f_pvalue": f_p,
        "variances_unequal": unequal,
        "t_test": test_name,
        "t_stat": float(t_res.statistic),
        "t_pvalue": t_p,
        "significant_at_alpha": bool(np.isfinite(t_p) and t_p < alpha),
    }


def pairwise_delta(
    df: pd.DataFrame,
    config_a: str,
    config_b: str,
    model: str,
    *,
    question: str,
) -> Optional[Dict[str, Any]]:
    ra = _row(df, config_a, model)
    rb = _row(df, config_b, model)
    if ra is None or rb is None:
        return None
    return {
        "question": question,
        "model": model,
        "config_a": config_a,
        "config_b": config_b,
        "r2_a": ra["r2_test"],
        "r2_b": rb["r2_test"],
        "delta_r2": float(rb["r2_test"] - ra["r2_test"]),
        "rmse_a": ra["rmse_test"],
        "rmse_b": rb["rmse_test"],
        "delta_rmse": float(rb["rmse_test"] - ra["rmse_test"]),
    }


def derive_comparisons(
    results: pd.DataFrame,
    *,
    wide_train: Optional[pd.DataFrame] = None,
    wide_test: Optional[pd.DataFrame] = None,
    long_train: Optional[pd.DataFrame] = None,
    long_test: Optional[pd.DataFrame] = None,
    feats: Optional[Dict[str, Any]] = None,
    alpha: float = 0.05,
    sq_error_cache: Optional[Dict[Tuple[str, str], pd.Series]] = None,
) -> pd.DataFrame:
    """
    Q1–Q3 (+ Q2 oracle) pairwise tables for RF and XGB.

    When train/test frames and ``feats`` are supplied, adds F-test (equal variances on
    paired squared errors) and Welch / paired t-test (if ``alpha`` exceeded on F-test).
    Units: animal×frequency test cells.
    """
    run_tests = all(
        x is not None
        for x in (wide_train, wide_test, long_train, long_test, feats)
    )
    cache = sq_error_cache
    if run_tests and cache is None:
        cache = build_sq_error_cache(
            wide_train, wide_test, long_train, long_test, feats  # type: ignore[arg-type]
        )

    comps: List[Dict[str, Any]] = []
    for model in ("RF", "XGB"):
        for q, a, b in COMPARISON_PAIRS:
            row = pairwise_delta(results, a, b, model, question=q)
            if row is None:
                continue
            if run_tests and cache is not None:
                sa = cache.get((a, model))
                sb = cache.get((b, model))
                if sa is not None and sb is not None:
                    common = sa.index.intersection(sb.index)
                    infer = paired_sq_error_inference(
                        sa.loc[common].values,
                        sb.loc[common].values,
                        alpha=alpha,
                    )
                    row = {**row, **infer, "alpha": alpha}
            comps.append(row)
    return pd.DataFrame(comps)


def best_tree_format(results: pd.DataFrame, model: str) -> str:
    row = pairwise_delta(results, "T1", "T4", model, question="Q3_format")
    if row is None:
        return "wide"
    return "wide" if row["delta_r2"] > 0 else "long"


def pick_best_tree_config(results: pd.DataFrame) -> Dict[str, Any]:
    trees = results[results["config_id"].str.startswith("T")].copy()
    pred = trees[trees["noise_label"] == "predicted"]
    if pred.empty:
        pred = trees
    best_idx = pred["r2_test"].idxmax()
    best = pred.loc[best_idx]
    return {
        "cohort": "liberman",
        "config_id": str(best["config_id"]),
        "format": str(best["format"]),
        "tree_model": str(best["model"]),
        "noise_label": str(best["noise_label"]),
        "r2_test": float(best["r2_test"]),
        "rmse_test": float(best["rmse_test"]),
    }


def export_artifacts(
    results: pd.DataFrame,
    comparisons: pd.DataFrame,
    cache_dir: Path,
) -> Tuple[Path, Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / "liberman_classical_comparison.parquet"
    comp_path = cache_dir / "liberman_classical_comparisons.parquet"
    json_path = cache_dir / "liberman_best_tree_config.json"

    results.to_parquet(parquet_path, index=False)
    comparisons.to_parquet(comp_path, index=False)

    best = pick_best_tree_config(results)
    best["comparisons"] = comparisons.to_dict(orient="records")
    json_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    return parquet_path, json_path, comp_path


def _r2_mask(y_true: pd.Series, y_pred: np.ndarray, mask: pd.Series) -> Optional[float]:
    if mask.sum() < 2:
        return None
    yt = y_true.loc[mask].values
    yp = y_pred[mask.values]
    return float(r2_score(yt, yp))


def _stage2_noise_importance(
    reg: Any,
    syn_num: List[str],
    syn_cat: List[str],
    syn_log: List[str],
) -> Dict[str, Any]:
    est = reg.steps[-1][1]
    imps = np.asarray(est.feature_importances_, dtype=float)
    prep = reg.named_steps["prep"]
    names: List[str] = list(syn_num)
    if syn_cat:
        ohe = prep.named_transformers_["cat"]
        names.extend(list(ohe.get_feature_names_out(syn_cat)))
    names.extend([f"log1p({c})" for c in syn_log])
    if len(names) != len(imps):
        names = [f"f{i}" for i in range(len(imps))]
    order = np.argsort(-imps)
    ranked = [{"feature": names[i], "importance": float(imps[i])} for i in order[:15]]
    noise_idx = [
        i
        for i, n in enumerate(names)
        if any(tok in str(n) for tok in ("noise_preds", "noise_cat"))
    ]
    noise_imp = float(imps[noise_idx].sum()) if noise_idx else 0.0
    total = float(imps.sum()) or 1.0
    return {
        "noise_importance_share": noise_imp / total,
        "noise_importance_rank": int(order.tolist().index(noise_idx[0])) + 1
        if noise_idx
        else None,
        "top_features": ranked,
    }


def diagnose_wide_noise_lift(
    wide_train: pd.DataFrame,
    wide_test: pd.DataFrame,
    feats: Dict[str, Any],
    *,
    model: Literal["RF", "XGB"] = "RF",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Decompose wide T4/T5/T6 lift: stage-1 errors vs RF under-using ``noise_preds``.

    Returns (test-animal table, train-animal table, summary dict).
    """
    syn_num = list(feats["syn_num"])
    syn_log = list(feats["syn_log"])
    noise_num = list(feats["noise_num"])
    noise_log = list(feats["noise_log"])
    fit_fn = _fit_stage2_rf if model == "RF" else _fit_stage2_xgb

    s1 = fit_stage1_wide_best(
        wide_stage1_fit(wide_train),
        wide_stage1_val(wide_train),
        wide_test,
        noise_num,
        noise_log,
        random_state=1,
        verbose=False,
    )
    animal_pred = s1["animal_pred_te"].astype(int)
    animal_prob = s1["animal_prob_te"].astype(float)
    animal_gt = wide_test.groupby("animal_id")["noise_cat"].first().round().astype(int)
    animal_gt = animal_gt.reindex(animal_pred.index).astype(int)

    tr_aug = attach_noise_preds_long(
        wide_train,
        s1["animal_pred_non_test"],
        fallback_noise_cat=False,
        require_full_coverage=True,
    )
    te_aug = attach_noise_preds_long(
        wide_test, s1["animal_pred_te"], fallback_noise_cat=False
    )

    train_gt = (
        wide_train.groupby("animal_id")["noise_cat"].first().round().astype(int)
    )
    train_pred = s1["animal_pred_non_test"].astype(int)
    train_pred = train_pred.reindex(train_gt.index).astype(int)
    train_row_match = (
        tr_aug["noise_cat"].astype(int) == tr_aug["noise_preds"].astype(int)
    )
    train_animal_tbl = (
        tr_aug.groupby("animal_id", as_index=False)
        .agg(
            noise_cat=("noise_cat", "first"),
            noise_pred=("noise_preds", "first"),
            n_rows=("noise_preds", "size"),
        )
        .astype({"noise_cat": int, "noise_pred": int})
    )
    train_animal_tbl["pred_correct"] = (
        train_animal_tbl["noise_cat"] == train_animal_tbl["noise_pred"]
    )
    tr_oracle = tr_aug.copy()
    tr_oracle["noise_preds"] = tr_oracle["noise_cat"].astype(int)

    reg_t4, _, _ = fit_fn(wide_train, syn_num, [], syn_log)
    reg_t5, _, _ = fit_fn(tr_aug, syn_num, ["noise_preds"], syn_log)
    reg_t5_oracle_train, _, _ = fit_fn(tr_oracle, syn_num, ["noise_preds"], syn_log)
    reg_t6, _, _ = fit_fn(wide_train, syn_num, ["noise_cat"], syn_log)

    feat_t4 = syn_num + syn_log
    feat_t5 = syn_num + ["noise_preds"] + syn_log
    feat_t6 = syn_num + ["noise_cat"] + syn_log
    idx = wide_test[feat_t6 + ["synapses"]].dropna().index

    y = wide_test.loc[idx, "synapses"]
    pred_t4 = reg_t4.predict(wide_test.loc[idx, feat_t4])
    pred_t5 = reg_t5.predict(te_aug.loc[idx, feat_t5])
    pred_t5_oracle_train = reg_t5_oracle_train.predict(te_aug.loc[idx, feat_t5])
    pred_t6 = reg_t6.predict(wide_test.loc[idx, feat_t6])

    # Counterfactual: T5 weights but force true noise label on test (only changes wrong animals).
    te_oracle = te_aug.loc[idx].copy()
    te_oracle["noise_preds"] = te_oracle["noise_cat"].astype(int)
    pred_t5_oracle_test = reg_t5.predict(te_oracle[feat_t5])

    row_animal = wide_test.loc[idx, "animal_id"].astype(str)
    correct_ids = set(animal_pred.index[animal_pred.values == animal_gt.values].astype(str))
    wrong_ids = set(animal_pred.index[animal_pred.values != animal_gt.values].astype(str))
    mask_correct = row_animal.isin(correct_ids)
    mask_wrong = row_animal.isin(wrong_ids)

    r2_all = {
        "T4": float(r2_score(y, pred_t4)),
        "T5": float(r2_score(y, pred_t5)),
        "T5_oracle_train_labels": float(r2_score(y, pred_t5_oracle_train)),
        "T6": float(r2_score(y, pred_t6)),
        "T5_oracle_test_labels": float(r2_score(y, pred_t5_oracle_test)),
    }
    r2_correct = {
        "T4": _r2_mask(y, pred_t4, mask_correct),
        "T5": _r2_mask(y, pred_t5, mask_correct),
        "T6": _r2_mask(y, pred_t6, mask_correct),
        "T5_oracle_test_labels": _r2_mask(y, pred_t5_oracle_test, mask_correct),
    }
    r2_wrong = {
        "T4": _r2_mask(y, pred_t4, mask_wrong),
        "T5": _r2_mask(y, pred_t5, mask_wrong),
        "T6": _r2_mask(y, pred_t6, mask_wrong),
        "T5_oracle_test_labels": _r2_mask(y, pred_t5_oracle_test, mask_wrong),
    }

    sq_t4 = (y.values - pred_t4) ** 2
    sq_t5 = (y.values - pred_t5) ** 2
    sq_t6 = (y.values - pred_t6) ** 2
    eval_rows = te_aug.loc[idx, ["animal_id", "frequency", "synapses"]].copy()
    eval_rows["sq_err_T4"] = sq_t4
    eval_rows["sq_err_T5"] = sq_t5
    eval_rows["sq_err_T6"] = sq_t6
    animal_eval = (
        eval_rows.groupby("animal_id", as_index=False)
        .agg(
            n_rows=("synapses", "size"),
            synapses=("synapses", "first"),
            mse_T4=("sq_err_T4", "mean"),
            mse_T5=("sq_err_T5", "mean"),
            mse_T6=("sq_err_T6", "mean"),
        )
    )
    animal_tbl = pd.DataFrame(
        {
            "animal_id": animal_pred.index.astype(str),
            "noise_cat": animal_gt.values,
            "noise_pred": animal_pred.values,
            "noise_prob": animal_prob.reindex(animal_pred.index).values,
            "pred_correct": (animal_pred.values == animal_gt.values),
        }
    )
    animal_tbl = animal_tbl.merge(animal_eval, on="animal_id", how="left")
    animal_tbl["delta_mse_T5_minus_T4"] = animal_tbl["mse_T5"] - animal_tbl["mse_T4"]
    animal_tbl["delta_mse_T6_minus_T5"] = animal_tbl["mse_T6"] - animal_tbl["mse_T5"]

    n_test = len(animal_tbl)
    n_wrong = int((~animal_tbl["pred_correct"]).sum())
    n_correct = n_test - n_wrong

    summary: Dict[str, Any] = {
        "model": model,
        "stage1": {
            "n_test_animals": n_test,
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "accuracy": float(animal_tbl["pred_correct"].mean()),
            "wrong_animal_ids": animal_tbl.loc[~animal_tbl["pred_correct"], "animal_id"]
            .astype(str)
            .tolist(),
            "threshold": float(s1["stage1_threshold"]),
            "test_auc": float(s1["s1_test_auc"]),
            "non_test_train_animal_accuracy": float(
                (train_gt.values == train_pred.values).mean()
            ),
        },
        "train_noise_labels": {
            "n_train_animals": int(len(train_animal_tbl)),
            "n_train_wide_rows": int(len(tr_aug)),
            "noise_preds_nan_rows": int(tr_aug["noise_preds"].isna().sum()),
            "row_match_rate": float(train_row_match.mean()),
            "animal_match_rate": float(train_animal_tbl["pred_correct"].mean()),
            "n_mismatched_animals": int((~train_animal_tbl["pred_correct"]).sum()),
            "mismatched_animal_ids": train_animal_tbl.loc[
                ~train_animal_tbl["pred_correct"], "animal_id"
            ]
            .astype(str)
            .tolist(),
        },
        "n_test_wide_rows": int(len(idx)),
        "n_rows_correct_animals": int(mask_correct.sum()),
        "n_rows_wrong_animals": int(mask_wrong.sum()),
        "r2_all_rows": r2_all,
        "r2_correct_animals_only": r2_correct,
        "r2_wrong_animals_only": r2_wrong,
        "delta_r2": {
            "T5_minus_T4_all": r2_all["T5"] - r2_all["T4"],
            "T5_minus_T4_correct_only": (r2_correct["T5"] or float("nan"))
            - (r2_correct["T4"] or float("nan")),
            "T6_minus_T5_all": r2_all["T6"] - r2_all["T5"],
            "T6_minus_T5_correct_only": (r2_correct["T6"] or float("nan"))
            - (r2_correct["T5"] or float("nan")),
            "T5_oracle_test_minus_T5_all": r2_all["T5_oracle_test_labels"] - r2_all["T5"],
            "T5_oracle_train_minus_T5_all": r2_all["T5_oracle_train_labels"] - r2_all["T5"],
            "T6_minus_T5_oracle_train": r2_all["T6"] - r2_all["T5_oracle_train_labels"],
        },
        "feature_importance_T5": _stage2_noise_importance(
            reg_t5, syn_num, ["noise_preds"], syn_log
        ),
        "feature_importance_T6": _stage2_noise_importance(
            reg_t6, syn_num, ["noise_cat"], syn_log
        ),
        "interpretation": (
            "Train wide rows get noise_preds via animal_pred_non_test (100% coverage). "
            "Stage-1 non-test animal accuracy is much lower than test accuracy; noisy train "
            "labels prevent T5 from learning the noise split. T5 with oracle train labels "
            "should approach T6."
        ),
    }
    return animal_tbl, train_animal_tbl, summary


def export_noise_diagnostic(
    animal_tbl: pd.DataFrame,
    summary: Dict[str, Any],
    cache_dir: Path,
    *,
    model: str = "RF",
    train_animal_tbl: Optional[pd.DataFrame] = None,
) -> Tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pq = cache_dir / f"liberman_noise_pred_diagnostic_{model.lower()}.parquet"
    js = cache_dir / f"liberman_noise_pred_diagnostic_{model.lower()}.json"
    animal_tbl.to_parquet(pq, index=False)
    if train_animal_tbl is not None:
        train_pq = cache_dir / f"liberman_noise_train_audit_{model.lower()}.parquet"
        train_animal_tbl.to_parquet(train_pq, index=False)
    js.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return pq, js

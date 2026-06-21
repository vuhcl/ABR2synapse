"""
Stage-2 XGBoost SHAP summary tables (wide format, level-concatenated features).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from utils.nn_stage2_data import STRAIN_BINARY_COL

COHORT_A_SLUG = "liberman"
COHORT_B_SLUG = "buran"
COHORT_SLUGS_DECK = (COHORT_A_SLUG, COHORT_B_SLUG)
XGB_SHAP_CACHE_DIR = Path("figures/cache/shap")

FEATURE_TYPE_ORDER: tuple[str, ...] = (
    "amplitude",
    "distance",
    "slope",
    "total_variance",
    "PeakIEarlyCurvature",
    "PeakICentralCurvature",
    "PeakILateCurvature",
    "TroughIEarlyCurvature",
    "TroughICentralCurvature",
    "TroughILateCurvature",
)

AGGREGATED_TYPE_ORDER: tuple[str, ...] = FEATURE_TYPE_ORDER + (
    "frequency",
    STRAIN_BINARY_COL,
    "noise_preds",
)

FEATURE_PAPER_LABELS: dict[str, str] = {
    "amplitude": "Amplitude",
    "distance": "Peak-to-trough latency",
    "slope": "Wave I slope",
    "total_variance": "Total variance",
    "PeakIEarlyCurvature": "Peak I curvature (early)",
    "PeakICentralCurvature": "Peak I curvature (central)",
    "PeakILateCurvature": "Peak I curvature (late)",
    "TroughIEarlyCurvature": "Trough I curvature (early)",
    "TroughICentralCurvature": "Trough I curvature (central)",
    "TroughILateCurvature": "Trough I curvature (late)",
    "noise_preds": "Predicted noise",
    "frequency": "Frequency (kHz)",
    STRAIN_BINARY_COL: "Strain (CBA/CaJ vs C57BL/6J)",
}

SHAP_EXPORT_FREQ_RENAME = {"frequency_feature": "frequency"}
SHAP_EXPORT_META_COLS = ("animal_id", "frequency")


def feature_annotations(raw_features: Sequence[str]) -> pd.DataFrame:
    """Annotate wide Stage-2 feature names with type and SPL level."""
    rows = []
    for f in raw_features:
        level_db = None
        feature_type = f
        is_noise_pred = f == "noise_preds"
        is_frequency = f == "frequency"
        if not (is_noise_pred or is_frequency):
            if "_" in f:
                base, suffix = f.rsplit("_", 1)
                try:
                    level_db = int(float(suffix))
                    feature_type = base
                except ValueError:
                    feature_type = f
            else:
                feature_type = f
        rows.append(
            {
                "feature_name": f,
                "feature_type": feature_type,
                "level_db": level_db,
                "is_stage1_noise_pred": bool(is_noise_pred),
                "is_frequency": bool(is_frequency),
            }
        )
    return pd.DataFrame(rows)


def paper_label_from_raw_feature(raw_name: str) -> str:
    if raw_name in FEATURE_PAPER_LABELS:
        return FEATURE_PAPER_LABELS[raw_name]
    if "_" in raw_name:
        base, suffix = raw_name.rsplit("_", 1)
        try:
            level_db = int(float(suffix))
            base_label = FEATURE_PAPER_LABELS.get(base, base)
            return f"{base_label} ({level_db} dB)"
        except ValueError:
            return FEATURE_PAPER_LABELS.get(raw_name, raw_name)
    return FEATURE_PAPER_LABELS.get(raw_name, raw_name)


def mean_abs_shap_series(shap_df: pd.DataFrame, features: Sequence[str]) -> pd.Series:
    cols = [c for c in features if c in shap_df.columns]
    if not cols:
        raise ValueError("No requested features found in SHAP frame")
    return shap_df[cols].abs().mean(axis=0)


def pooled_mean_abs_order(
    shap_a: pd.DataFrame,
    shap_b: pd.DataFrame,
    features: Sequence[str],
) -> list[str]:
    pooled = pd.concat([shap_a, shap_b], ignore_index=True)
    means = mean_abs_shap_series(pooled, features).sort_values(ascending=False)
    return means.index.tolist()


def _rank_descending(scores: pd.Series) -> pd.Series:
    return scores.rank(method="first", ascending=False).astype(int)


def _type_mean_abs_score(shap_df: pd.DataFrame, type_cols: Sequence[str]) -> float:
    return float(mean_abs_shap_series(shap_df, type_cols).sum())


def _cohort_shap(
    cohort_outputs: Mapping[str, Mapping[str, pd.DataFrame]],
    slug: str,
) -> pd.DataFrame:
    if slug not in cohort_outputs:
        raise KeyError(f"Missing cohort_outputs entry for {slug!r}")
    return cohort_outputs[slug]["shap"]


def _shared_features(
    cohort_outputs: Mapping[str, Mapping[str, pd.DataFrame]],
    feature_table: pd.DataFrame,
) -> list[str]:
    shap_a = _cohort_shap(cohort_outputs, COHORT_A_SLUG)
    shap_b = _cohort_shap(cohort_outputs, COHORT_B_SLUG)
    candidates = feature_table["feature_name"].tolist()
    features = [f for f in candidates if f in shap_a.columns and f in shap_b.columns]
    if not features:
        raise ValueError("No shared SHAP features between cohorts")
    return features


def full_feature_comparison_table(
    cohort_outputs: Mapping[str, Mapping[str, pd.DataFrame]],
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """Mean |SHAP| per wide feature; cohort A/B columns; pooled sort."""
    features = _shared_features(cohort_outputs, feature_table)
    shap_a = _cohort_shap(cohort_outputs, COHORT_A_SLUG)
    shap_b = _cohort_shap(cohort_outputs, COHORT_B_SLUG)

    mean_a = mean_abs_shap_series(shap_a, features)
    mean_b = mean_abs_shap_series(shap_b, features)
    order = pooled_mean_abs_order(shap_a, shap_b, features)

    out = pd.DataFrame(
        {
            "feature": order,
            "feature_label": [paper_label_from_raw_feature(f) for f in order],
            "cohort_a_mean_abs_shap": [float(mean_a[f]) for f in order],
            "cohort_b_mean_abs_shap": [float(mean_b[f]) for f in order],
        }
    )
    out["cohort_a_rank"] = _rank_descending(mean_a).loc[order].astype(int).values
    out["cohort_b_rank"] = _rank_descending(mean_b).loc[order].astype(int).values
    return out.reset_index(drop=True)


def aggregated_feature_type_comparison_table(
    cohort_outputs: Mapping[str, Mapping[str, pd.DataFrame]],
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """Sum of per-column mean |SHAP| within each feature type; pooled sort."""
    shap_a = _cohort_shap(cohort_outputs, COHORT_A_SLUG)
    shap_b = _cohort_shap(cohort_outputs, COHORT_B_SLUG)
    pooled = pd.concat([shap_a, shap_b], ignore_index=True)

    by_type = feature_table.groupby("feature_type")["feature_name"].apply(list)

    rows: list[dict[str, object]] = []
    pooled_scores: dict[str, float] = {}
    mean_a: dict[str, float] = {}
    mean_b: dict[str, float] = {}

    for ftype in AGGREGATED_TYPE_ORDER:
        if ftype == STRAIN_BINARY_COL and ftype not in shap_a.columns:
            continue
        type_cols = [c for c in by_type.get(ftype, []) if c in shap_a.columns]
        if not type_cols:
            continue
        score_a = _type_mean_abs_score(shap_a, type_cols)
        score_b = _type_mean_abs_score(shap_b, type_cols)
        pooled_scores[ftype] = _type_mean_abs_score(pooled, type_cols)
        mean_a[ftype] = score_a
        mean_b[ftype] = score_b
        rows.append(
            {
                "feature_type": ftype,
                "feature_type_label": FEATURE_PAPER_LABELS.get(ftype, ftype),
                "cohort_a_mean_abs_shap": score_a,
                "cohort_b_mean_abs_shap": score_b,
            }
        )

    if not rows:
        raise ValueError("No aggregated feature types with SHAP columns")

    order = sorted(pooled_scores, key=lambda k: pooled_scores[k], reverse=True)
    out = pd.DataFrame(rows).set_index("feature_type").loc[order].reset_index()

    rank_a = _rank_descending(pd.Series(mean_a))
    rank_b = _rank_descending(pd.Series(mean_b))
    out["cohort_a_rank"] = [int(rank_a[ftype]) for ftype in order]
    out["cohort_b_rank"] = [int(rank_b[ftype]) for ftype in order]
    return out.reset_index(drop=True)


def load_cohort_outputs_from_cache(
    cache_dir: Path | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Reload OOF SHAP values from exported parquets."""
    cache_dir = Path(cache_dir or XGB_SHAP_CACHE_DIR)
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for slug in COHORT_SLUGS_DECK:
        path = cache_dir / f"shap_values_{slug}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        drop_cols = [c for c in SHAP_EXPORT_META_COLS if c in df.columns]
        shap_df = df.drop(columns=drop_cols).rename(columns=SHAP_EXPORT_FREQ_RENAME)
        outputs[slug] = {"shap": shap_df}
    return outputs

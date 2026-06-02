"""Long-format OLS with animal×frequency aggregation (comparison notebook parity)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score


def eval_ols_long(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    formula: str,
    dropna_cols: List[str],
) -> Tuple[float, float]:
    """Fit OLS on long train rows; score test with mean pred per animal×frequency."""
    train_clean = train_df.dropna(subset=dropna_cols + ["synapses"])
    test_clean = test_df.dropna(subset=dropna_cols + ["synapses"])
    model = smf.ols(formula, data=train_clean).fit()
    y_pred = model.predict(test_clean)
    _eval = test_clean[["animal_id", "frequency", "synapses"]].copy()
    _eval["y_pred"] = y_pred
    _agg = _eval.groupby(["animal_id", "frequency"]).agg(
        y_true=("synapses", "first"),
        y_pred=("y_pred", "mean"),
    )
    r2 = float(r2_score(_agg["y_true"], _agg["y_pred"]))
    rmse = float(np.sqrt(np.mean((_agg["y_true"] - _agg["y_pred"]) ** 2)))
    return r2, rmse

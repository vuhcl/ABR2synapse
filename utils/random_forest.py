import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.subject_cv import (
    DEFAULT_CV_RANDOM_STATE,
    build_subject_cv,
    prepare_xy_groups,
    resolve_subject_col,
    train_only_frame,
)


def RF_data_prep(
    data, num_features, cat_features, target="SynapsesPerIHC"
):  # features='default'):
    X = data[num_features + cat_features]
    y = data[target]

    X_train, X_test, y_train, y_test = (
        X[data.DataGroup != "Test"],
        X[data.DataGroup == "Test"],
        y[data.DataGroup != "Test"],
        y[data.DataGroup == "Test"],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features),
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor


def RF_cross_validation(
    data,
    num_features=None,
    cat_features=None,
    model="default",
    folds=5,
    random_state=DEFAULT_CV_RANDOM_STATE,
):
    X_train, X_test, y_train, y_test, preprocessor = RF_data_prep(
        data, num_features=num_features, cat_features=cat_features
    )

    target_col = "SynapsesPerIHC"
    tune_frame = train_only_frame(data)
    id_col = resolve_subject_col(tune_frame)
    feat_cols = list(num_features or []) + list(cat_features or [])
    X_tune, y_tune, groups, _ = prepare_xy_groups(
        tune_frame, feat_cols, target_col, id_col
    )

    cv = build_subject_cv(
        y_tune.to_numpy(),
        groups,
        n_splits=folds,
        random_state=random_state,
        prefer_stratified=False,
    )
    cv_scores = []
    rmse_scores = []

    if model == "default":
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            max_depth=10,
            min_samples_leaf=5,
        )

    X_arr = X_tune.to_numpy() if hasattr(X_tune, "to_numpy") else np.asarray(X_tune)
    y_arr = y_tune.to_numpy()

    for train_idx, val_idx in cv.split(X_arr, y_arr, groups):
        X_fold_train = X_tune.iloc[train_idx]
        X_fold_val = X_tune.iloc[val_idx]
        y_fold_train = y_tune.iloc[train_idx]
        y_fold_val = y_tune.iloc[val_idx]

        rf_pipeline = Pipeline([("preprocessing", preprocessor), ("rf", model)])
        rf_pipeline.fit(X_fold_train, y_fold_train)
        y_pred = rf_pipeline.predict(X_fold_val)

        rmse = np.sqrt(np.mean((y_fold_val - y_pred) ** 2))
        rmse_scores.append(rmse)

    print("Cross-Validation RMSE Scores:", rmse_scores)
    print("Average CV RMSE:", np.mean(rmse_scores))

    return rmse_scores

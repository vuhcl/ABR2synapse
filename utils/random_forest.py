import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def RF_data_prep(
    data, num_features, cat_features, target="SynapsesPerIHC"
):  # features='default'):
    # if num_features is None:
    #     num_features = [
    #         "Frequency(kHz)",
    #         "Level(dB)",
    #         "Amplitude",
    #         "Noise",
    #         "Time (hrs)",
    #     ]
    # if cat_features is None:
    #     cat_features = [
    #         "Strain (binary)",
    #     ]
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
    data, num_features=None, cat_features=None, model="default", folds=5, random_state=1
):

    X_train, X_test, y_train, y_test, preprocessor = RF_data_prep(
        data, num_features=num_features, cat_features=cat_features
    )

    k_folds = KFold(n_splits=folds)
    cv_scores = []
    rmse_scores = []

    if model == "default":
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            max_depth=10,
            min_samples_leaf=5,
        )

    for fold_idx, (train_idx, val_idx) in enumerate(k_folds.split(X_train)):
        # Split the training data into training and validation sets for this fold
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Create Random Forest Pipeline
        rf_pipeline = Pipeline([("preprocessing", preprocessor), ("rf", model)])

        rf_pipeline.fit(X_fold_train, y_fold_train)
        y_pred = rf_pipeline.predict(X_fold_val)

        rmse = np.sqrt(np.mean((y_fold_val - y_pred) ** 2))
        rmse_scores.append(rmse)

    print("Cross-Validation RMSE Scores:", rmse_scores)
    print("Average CV RMSE:", np.mean(rmse_scores))

    return rmse_scores

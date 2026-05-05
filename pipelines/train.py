"""
Model Training Pipeline (Ensemble + LOOCV Selection)

This module:
- Loads processed dataset from CLEAN_DATA_PATH
- Applies feature encoding (Label + OneHot)
- Trains multiple ML models (linear + tree-based)
- Evaluates models using Leave-One-Out Cross Validation (LOOCV)
- Selects top models based on MSE
- Trains final ensemble models on full dataset
- Saves trained models + metadata for inference

CLI usage (from repo root or src/):
    python -m pipelines.train
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
import joblib

from config import (
    CLEAN_DATA_PATH,
    MODELS_DIR,
    MODEL_ARTIFACT_DIR,
    TOP_MODELS_PATH,
    TRAINED_MODELS_PATH,
    ENSEMBLE_NUM
)
from models.models import LINEAR_MODELS, TREE_MODELS

def Cat_LabelEncoding(df, cols):
    modified_df = df.copy()

    le = LabelEncoder()

    for col in cols:
        modified_df[col] = le.fit_transform(modified_df[col])

    modified_df.drop(columns=cols)

    return modified_df

def Cat_OneHotEncoding(df, cols):
    modified_df = pd.get_dummies(df, columns=["category", "day_of_week"])

    return modified_df

def train():
    category_cols = ["category", "day_of_week"]
    target_col = "late_duration_min"

    feature_df = pd.read_parquet(CLEAN_DATA_PATH)

    # Split features and target
    y = feature_df[target_col]

    X_label = Cat_LabelEncoding(feature_df.drop(columns=[target_col]), category_cols)
    X_onehot = Cat_OneHotEncoding(feature_df.drop(columns=[target_col]), category_cols)

    def loocv_mse(model, X, y):
        loo = LeaveOneOut()
        errors = []

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            pred = model_clone.predict(X_test)

            errors.append(mean_squared_error(y_test, pred))

        return np.mean(errors)

    results = {}

    # Linear models (one-hot encoding)
    for name, model in LINEAR_MODELS:
        mse = loocv_mse(model, X_onehot, y)
        results[name] = {
            "mse": mse,
            "model": model,
            "type": "linear"
        }

    # Tree models (label encoding)
    for name, model in TREE_MODELS:
        mse = loocv_mse(model, X_label, y)
        results[name] = {
            "mse": mse,
            "model": model,
            "type": "tree"
        }

    # Rank models
    sorted_results = sorted(results.items(), key=lambda x: x[1]["mse"])

    # Select top N models based on LOOCV results
    top_models = [name for name, _ in sorted_results[:ENSEMBLE_NUM]]

    # Retrain each selected model using ALL available data
    trained_models = {}

    for name, info in results.items():
        # Create a fresh copy of the model
        model = clone(info["model"])

        # Choose correct feature representation
        # - Linear models → one-hot encoded features
        # - Tree models   → label encoded features
        if info["type"] == "linear":
            model.fit(X_onehot, y)
        else:
            model.fit(X_label, y)

        # Store trained model + metadata
        trained_models[name] = {
            "model": model,
            "type": info["type"],
            "mse": info["mse"]
        }

    def ensemble_predict(trained_models, model_names, X_df):
        preds = []

        for name in model_names:
            model_info = trained_models[name]

            if model_info["type"] == "linear":
                preds.append(
                    model_info["model"].predict(
                        Cat_OneHotEncoding(X_df.drop(columns=[target_col]), category_cols)
                    )
                )
            else:
                preds.append(
                    model_info["model"].predict(
                        Cat_LabelEncoding(X_df.drop(columns=[target_col]), category_cols)
                    )
                )

        return np.mean(preds, axis=0)

    final_pred = ensemble_predict(
        trained_models,
        top_models,
        feature_df
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(MODEL_ARTIFACT_DIR, exist_ok=True)

    joblib.dump(trained_models, TRAINED_MODELS_PATH)
    joblib.dump(top_models, TOP_MODELS_PATH)

if __name__ == "__main__":
    train()
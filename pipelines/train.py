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
import logging
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def Cat_LabelEncoding(df, cols):
    logger.info(f"[Encoding] LabelEncoding columns: {cols}")

    modified_df = df.copy()
    le = LabelEncoder()

    for col in cols:
        before_nunique = modified_df[col].nunique()
        modified_df[col] = le.fit_transform(modified_df[col])

        logger.info(f"[LabelEncoding] {col}: unique values {before_nunique} → {modified_df[col].nunique()}")

    logger.info(f"[LabelEncoding] Final shape: {modified_df.shape}")

    return modified_df


def Cat_OneHotEncoding(df, cols):
    logger.info(f"[Encoding] OneHotEncoding columns: {cols}")

    before_shape = df.shape
    modified_df = pd.get_dummies(df, columns=cols)

    logger.info(f"[OneHotEncoding] Shape {before_shape} → {modified_df.shape}")
    return modified_df


def train():
    logger.info("Starting training pipeline")

    category_cols = ["category", "day_of_week"]
    target_col = "late_duration_min"

    feature_df = pd.read_parquet(CLEAN_DATA_PATH)
    logger.info(f"Loaded dataset: shape={feature_df.shape}")

    # Basic checks
    missing_target = feature_df[target_col].isna().sum()
    logger.info(f"Missing target values: {missing_target}")

    # Split features and target
    y = feature_df[target_col]
    X_raw = feature_df.drop(columns=[target_col])

    logger.info(f"Feature columns: {X_raw.columns.tolist()}")

    # Encoding
    X_label = Cat_LabelEncoding(X_raw, category_cols)
    X_onehot = Cat_OneHotEncoding(X_raw, category_cols)

    logger.info(f"Encoded datasets ready | Label: {X_label.shape}, OneHot: {X_onehot.shape}")

    def loocv_mse(model, X, y):
        loo = LeaveOneOut()
        errors = []

        n = len(X)
        logger.info(f"[LOOCV] Starting | samples={n}")

        for i, (train_idx, test_idx) in enumerate(loo.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            pred = model_clone.predict(X_test)
            errors.append(mean_squared_error(y_test, pred))

            if i % 50 == 0:
                logger.debug(f"[LOOCV] Progress {i}/{n}")

        mse = np.mean(errors)
        logger.info(f"[LOOCV COMPLETE] MSE={mse:.6f}")

        return mse

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

    logger.info("========== MODEL RANKING ==========")
    for rank, (name, info) in enumerate(sorted_results):
        logger.info(f"{rank+1}. {name} | MSE={info['mse']:.6f} | type={info['type']}")

    # Select top N models based on LOOCV results
    top_models = [name for name, _ in sorted_results[:ENSEMBLE_NUM]]

    logger.info(f"Selected top models: {top_models}")

    # Retrain each selected model using ALL available data
    trained_models = {}

    for name, info in results.items():
        # Create a fresh copy of the model
        model = clone(info["model"])

        # Choose correct feature representation
        # - Linear models → one-hot encoded features
        # - Tree models   → label encoded features
        if info["type"] == "linear":
            logger.info(f"Training {name} on OneHot features")
            model.fit(X_onehot, y)
        else:
            logger.info(f"Training {name} on LabelEncoded features")
            model.fit(X_label, y)

        # Store trained model + metadata
        trained_models[name] = {
            "model": model,
            "type": info["type"],
            "mse": info["mse"]
        }

        logger.info(f"Trained {name} complete")


    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(MODEL_ARTIFACT_DIR, exist_ok=True)

    joblib.dump(trained_models, TRAINED_MODELS_PATH)
    joblib.dump(top_models, TOP_MODELS_PATH)

    logger.info(f"Saved models → {TRAINED_MODELS_PATH}")
    logger.info(f"Saved top models → {TOP_MODELS_PATH}")


if __name__ == "__main__":
    train()
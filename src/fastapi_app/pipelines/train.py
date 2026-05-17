"""
Model Training Pipeline (Ensemble + LOOCV Selection)

This module:
- Loads dataset containing features
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
import numpy as np
import logging
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
import joblib
from . import preprocess
from utils import cat_encoding
from utils.logger import setup_logging
from config import (
    FASTAPI_MODELS_DIR,
    FASTAPI_MODEL_ARTIFACT_DIR,
    TOP_MODELS_PATH,
    TRAINED_MODELS_PATH,
    ONEHOT_COL_PATH,
    ENSEMBLE_NUM
)
from ..models.models import LINEAR_MODELS, TREE_MODELS

setup_logging()
logger = logging.getLogger(__name__)

def loocv_mse(model, X, y):
    loo = LeaveOneOut()
    errors = []

    n = len(X)
    logger.info(f"[LOOCV with {model}] Starting | samples={n}")

    for i, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)

        pred = model_clone.predict(X_test)
        errors.append(mean_squared_error(y_test, pred))

        if i % 50 == 0:
            logger.debug(f"\n[LOOCV with {model}] Progress {i}/{n}\n")

    mse = np.mean(errors)
    logger.info(f"[LOOCV with {model} COMPLETE] MSE={mse:.6f}\n")

    return mse


def train():
    X_raw, y, category_cols = preprocess.train_preprocess()

    # Encoding
    X_label = cat_encoding.Cat_LabelEncoding(X_raw, category_cols)
    X_onehot = cat_encoding.Cat_OneHotEncoding(X_raw, category_cols)

    logger.info(f"Encoded datasets ready | Label: {X_label.shape}, OneHot: {X_onehot.shape}\n")

    results = {}

    # Linear models (one-hot encoding)
    for model_name, model in LINEAR_MODELS:
        mse = loocv_mse(model, X_onehot, y)
        results[model_name] = {
            "mse": mse,
            "model": model,
            "type": "linear"
        }

    # Tree models (label encoding)
    for model_name, model in TREE_MODELS:
        mse = loocv_mse(model, X_label, y)
        results[model_name] = {
            "mse": mse,
            "model": model,
            "type": "tree"
        }

    # Rank models
    sorted_results = sorted(results.items(), key=lambda x: x[1]["mse"])

    logger.info("\n========== MODEL RANKING ==========")
    for rank, (model_name, model_result) in enumerate(sorted_results):
        logger.info(f"{rank+1}. {model_name} | MSE={model_result['mse']:.6f} | type={model_result['type']}")

    # Select top N models based on LOOCV results
    top_models = [name for name, _ in sorted_results[:ENSEMBLE_NUM]]

    logger.info(f"\n\nSelected top models: {top_models}\n")

    # Retrain each selected model using ALL available data
    trained_models = {}

    logger.info("Retraining top models...\n")

    for model_name, model_result in results.items():
        # Create a fresh copy of the model
        model = clone(model_result["model"])

        # Choose correct feature representation
        # - Linear models → one-hot encoded features
        # - Tree models   → label encoded features
        if model_result["type"] == "linear":
            logger.info(f"Training {model_name} on OneHot features")
            model.fit(X_onehot, y)
        else:
            logger.info(f"Training {model_name} on LabelEncoded features")
            model.fit(X_label, y)

        # Store trained model + metadata
        trained_models[model_name] = {
            "model": model,
            "type": model_result["type"],
            "mse": model_result["mse"]
        }

        logger.info(f"Training of {model_name} complete\n\n")

    os.makedirs(FASTAPI_MODELS_DIR, exist_ok=True)
    os.makedirs(FASTAPI_MODEL_ARTIFACT_DIR, exist_ok=True)

    joblib.dump(trained_models, TRAINED_MODELS_PATH)
    joblib.dump(top_models, TOP_MODELS_PATH)
    joblib.dump(X_onehot.columns, ONEHOT_COL_PATH)

    logger.info(f"Saved models → {TRAINED_MODELS_PATH}\n")
    logger.info(f"Saved top models → {TOP_MODELS_PATH}\n")

if __name__ == "__main__":
    train()
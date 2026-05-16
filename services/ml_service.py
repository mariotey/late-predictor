import os
import joblib
import logging
from pipelines.train import train
from pipelines.preprocess import predict_preprocess
from pipelines.predict import run_ensemble_prediction
from utils.logger import setup_logging
from config import (
    TRAINED_MODELS_PATH,
    TOP_MODELS_PATH
)

setup_logging()
logger = logging.getLogger(__name__)


class MLService:
    def __init__(self):
        self.trained_models = None
        self.top_models = None

    def load_models(self):
        if not os.path.exists(TRAINED_MODELS_PATH):
            logger.info("⚠️ Model file missing. Run /train first.")
            raise FileNotFoundError("Trained model not found")

        self.trained_models = joblib.load(TRAINED_MODELS_PATH)

        if not os.path.exists(TOP_MODELS_PATH):
            logger.info("⚠️ Top Model file missing. Run /train first.")
            raise FileNotFoundError("Top models not found")

        self.top_models = joblib.load(TOP_MODELS_PATH)

        logger.info("✅ Models loaded")

    def retrain(self):
        train()
        self.load_models()

    def predict(self, payload):
        X_df, _, category_cols = predict_preprocess(payload)

        try:
            pred = run_ensemble_prediction(
                X_df,
                category_cols,
                self.trained_models,
                self.top_models
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError("Model not trained!")

        result = {"pred_min": float(pred)}

        logger.info(f"Output:\n{result}")

        return result

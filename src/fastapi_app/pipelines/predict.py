import numpy as np
import logging
import joblib
from typing import Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from utils import cat_encoding
from utils.logger import setup_logging
from config import ONEHOT_COL_PATH

setup_logging()
logger = logging.getLogger(__name__)

class PredictRequest(BaseModel):
    datetime_val: datetime = Field(..., description="ISO 8601 timestamp of the event")
    init_latlon: Tuple[float, float] = Field(..., description="(latitude, longitude) of origin")
    dest_latlon: Tuple[float, float] = Field(..., description="(latitude, longitude) of destination")
    category: str = Field(..., description="Activity category (e.g. dinner/drinks)")


def run_ensemble_prediction(X_df, category_cols, trained_models, top_models):
    preds = []

    logger.info(f"Starting ensemble prediction | models={top_models}")
    logger.info(f"Input shape: {X_df.shape}")
    logger.info(f"Input preview:\n{X_df.head()}")

    ONEHOT_COLUMNS = joblib.load(ONEHOT_COL_PATH)

    X_label = cat_encoding.Cat_LabelEncoding(X_df, category_cols)
    X_onehot = cat_encoding.Cat_OneHotEncoding(X_df, category_cols)
    X_onehot = X_onehot.reindex(columns=ONEHOT_COLUMNS, fill_value=0)

    for name in top_models:
        model_info = trained_models[name]

        logger.info(f"Running model: {name} | type={model_info['type']}")

        if model_info["type"] == "linear":
            pred = model_info["model"].predict(X_onehot)
        else:
            pred = model_info["model"].predict(X_label)

        logger.info(
            f"[{name}] prediction stats -> shape={pred.shape}, "
            f"mean={np.mean(pred):.4f}, std={np.std(pred):.4f}"
        )

        preds.append(pred)

    logger.info("Inference complete")

    return np.mean(preds, axis=0)[0]
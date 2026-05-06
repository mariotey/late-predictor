import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    day_of_week: int
    distance_km: float
    category: str


def Cat_LabelEncoding(df, cols):
    modified_df = df.copy()

    logger.info(f"Applying Label Encoding on columns: {cols}")

    le = LabelEncoder()

    for col in cols:
        unique_before = modified_df[col].nunique()
        modified_df[col] = le.fit_transform(modified_df[col])

        logger.info(
            f"Encoded '{col}' | unique values before={unique_before}, "
            f"after={modified_df[col].nunique()}"
        )

    return modified_df


def Cat_OneHotEncoding(df, cols):
    logger.info(f"Applying OneHot Encoding on columns: {cols}")

    before_shape = df.shape
    modified_df = pd.get_dummies(df, columns=cols)

    logger.info(
        f"OneHot encoding complete | before_shape={before_shape}, "
        f"after_shape={modified_df.shape}, "
        f"new_columns_added={modified_df.shape[1] - before_shape[1]}"
    )

    return modified_df


def run_ensemble_prediction(X_df, trained_models, top_models):
    category_cols = ["category", "day_of_week"]
    preds = []

    logger.info(f"Starting ensemble prediction | models={top_models}")
    logger.info(f"Input shape: {X_df.shape}")
    logger.info(f"Input preview:\n{X_df.head()}")

    for name in top_models:
        model_info = trained_models[name]

        logger.info(f"Running model: {name} | type={model_info['type']}")

        if model_info["type"] == "linear":
            encoded_X = Cat_OneHotEncoding(X_df, category_cols)
        else:
            encoded_X = Cat_LabelEncoding(X_df, category_cols)

        logger.info(f"[{name}] Encoded X shape: {encoded_X.shape}")

        pred = model_info["model"].predict(encoded_X)

        logger.info(
            f"[{name}] prediction stats -> shape={pred.shape}, "
            f"mean={np.mean(pred):.4f}, std={np.std(pred):.4f}"
        )

        preds.append(pred)

    logger.info("Inference complete")

    return np.mean(preds, axis=0)[0]
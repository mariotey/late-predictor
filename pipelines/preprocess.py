import logging
import pandas as pd
from haversine import haversine, Unit
from . import load_data
from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def train_preprocess():
    feature_registry_dict = load_data.extract_cached_registry()
    logger.info("Loaded feature registry\n")

    X_col = [
        col
        for col_types in feature_registry_dict["feature_col"].values()
        for col in col_types
    ]

    y_col = feature_registry_dict["target_col"]

    category_col = feature_registry_dict["feature_col"]["categorical"]

    train_df = load_data.extract_feature_store()
    logger.info(f"Loaded dataset: shape={train_df.shape}\n")

    # Basic checks
    missing_target = train_df[y_col].isna().sum()
    logger.info(f"Missing target values: {missing_target}\n")

    logger.info(f"Feature columns: {train_df[X_col].columns.tolist()}\n")

    return train_df[X_col], train_df[y_col], category_col

def predict_preprocess(payload):
    # Derive features
    datetime_val = payload.datetime_val
    day_of_week = datetime_val.weekday()
    hour = datetime_val.hour

    if hour >= 3 and hour < 12:
        time_of_day = "morning"
    elif hour >= 12 and hour < 18:
        time_of_day = "afternoon"
    else:
        time_of_day = "evening"

    distance_km = haversine(
        payload.init_latlon,
        payload.dest_latlon,
        unit=Unit.KILOMETERS
    )

    X_df = pd.DataFrame([{
        "day_of_week": day_of_week,
        "distance_km": round(distance_km, 2),
        "time_of_day": time_of_day,
        "category": payload.category
    }])

    feature_registry_dict = load_data.extract_cached_registry()
    logger.info("Loaded feature registry\n")

    X_col = [
        col
        for col_types in feature_registry_dict["feature_col"].values()
        for col in col_types
    ]

    category_col = feature_registry_dict["feature_col"]["categorical"]

    X_df = X_df[X_col]

    logger.info("📊 Preprocessed X_df: %s", X_df.to_dict(orient="records")[0])

    return X_df, category_col

import logging
import pandas as pd
from haversine import haversine, Unit
from ..services.feature_registry import load_feature_registry
import utils.supabase_client as supabase_client
from utils.logger import setup_logging
from config import CATEGORY_ID_COL

setup_logging()
logger = logging.getLogger(__name__)

def train_preprocess():
    feature_registry_dict = load_feature_registry()
    logger.info("Loaded feature registry\n")

    X_col = [
        col
        for col_types in feature_registry_dict["feature_col"].values()
        for col in col_types
    ]

    y_col = feature_registry_dict["target_col"]

    category_col = feature_registry_dict["feature_col"]["categorical"]

    train_df = supabase_client.get_feature_store()
    logger.info(f"Loaded dataset: shape={train_df.shape}\n")

    # Basic checks
    missing_target = train_df[y_col].isna().sum()
    logger.info(f"Missing target values: {missing_target}\n")

    logger.info(f"Feature columns: {train_df[X_col].columns.tolist()}\n")

    return train_df[X_col], train_df[y_col], category_col

def predict_preprocess(payload):
    # Derive features
    meeting_datetime = payload.datetime_val
    day_of_week = meeting_datetime.weekday()
    hour = meeting_datetime.hour

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

    feature_registry_dict = load_feature_registry()
    logger.info("Loaded feature registry\n")

    X_col = [
        col
        for col_types in feature_registry_dict["feature_col"].values()
        for col in col_types
    ]

    category_col = feature_registry_dict["feature_col"]["categorical"]
    target_col = feature_registry_dict["target_col"]

    X_df = X_df[X_col]

    logger.info("📊 Preprocessed X_df: %s", X_df.to_dict(orient="records")[0])

    return X_df, target_col, category_col

def feedback_preprocess(payload):
    if payload.arrived_datetime is None:
        raise ValueError("arrived_datetime is invalid!")

    if payload.pred_min is None:
        raise ValueError("est_min is invalid!")

    feedback_df = pd.DataFrame([{
        "meeting_location": payload.meeting_location,
        "date": payload.meeting_datetime.date(),
        "meeting_time": payload.meeting_datetime,
        "arrived_time": payload.arrived_datetime,
        "meeting_lat": payload.meeting_latlon[0],
        "meeting_lon": payload.meeting_latlon[1],
        CATEGORY_ID_COL: payload.category_id,
        "pred_min": payload.pred_min
    }])

    feedback_df["date"] = feedback_df["date"].astype(str)

    for col in ["meeting_time", "arrived_time"]:
        feedback_df[col] = feedback_df[col].apply(
            lambda x: x.isoformat() if pd.notna(x) else None
        )

    return feedback_df
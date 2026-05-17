import logging
from typing import Tuple
import pandas as pd
from datetime import datetime
from pydantic import BaseModel, Field
from .preprocess import feedback_preprocess
from utils.supabase_client import load_into_supabase, get_latest_registry
from utils.logger import setup_logging
from config import FEATURE_REGISTRY_ID_COL

setup_logging()
logger = logging.getLogger(__name__)

class DataFeedbackRequest(BaseModel):
    meeting_location: str = Field(..., description="Address of the meeting location")
    meeting_datetime: datetime = Field(..., description="ISO 8601 timestamp of meeting")
    init_latlon: Tuple[float, float] = Field(..., description="(latitude, longitude) of origin")
    meeting_latlon: Tuple[float, float] = Field(..., description="(latitude, longitude) of meetup")
    category_id: str = Field(..., description="The categorical ID of the activity")
    pred_min: float = Field(..., description="Predicted duration in minutes")
    arrived_datetime: datetime = Field(..., description="ISO 8601 timestamp of actual arrival")

def feedback_data(payload, top_models):
    feedback_df = feedback_preprocess(payload)

    logger.info(feedback_df, "\n")

    registry_dict = get_latest_registry()

    feedback_df["models_used"] = ", ".join(map(str, top_models))
    feedback_df[FEATURE_REGISTRY_ID_COL] = registry_dict[FEATURE_REGISTRY_ID_COL]

    load_into_supabase(feedback_df)

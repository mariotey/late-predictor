import logging
from typing import Tuple
import pandas as pd
from datetime import datetime
from pydantic import BaseModel, Field
import supabase_client
from pipelines.preprocess import feedback_preprocess
from supabase_client import load_into_supabase
from utils.latlon_parser import parse_latlon
from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class DataFeedbackRequest(BaseModel):
    meeting_location: str = Field(..., description="Address of the meeting location")
    meeting_datetime: datetime = Field(..., description="ISO 8601 timestamp of meeting")
    init_latlon: Tuple[float, float] = Field(..., description="(latitude, longitude) of origin")
    meeting_latlon: Tuple[float, float] = Field(..., description="(latitude, longitude) of meetup")
    category: str = Field(..., description="Activity category (e.g. dinner/drinks)")
    pred_min: float = Field(..., description="Predicted duration in minutes")
    arrived_datetime: datetime = Field(..., description="ISO 8601 timestamp of actual arrival")

def feedback_data(payload, top_models):
    feedback_df = feedback_preprocess(payload)

    logger.info(feedback_df, "\n")

    registry_dict = supabase_client.get_latest_registry()

    feedback_df["models_used"] = ", ".join(map(str, top_models))
    feedback_df["feature_registry_id"] = registry_dict["id"]

    load_into_supabase(feedback_df)

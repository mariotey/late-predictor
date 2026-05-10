import logging
from typing import Tuple
from datetime import datetime
from pydantic import BaseModel, Field
import supabase_client
from pipelines.preprocess import feedback_preprocess
from supabase_client import load_into_supabase
from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class DataFeedbackRequest(BaseModel):
    datetime_val: datetime = Field(..., description="ISO 8601 timestamp of the event")
    init_latlon: Tuple[float, float] = Field(..., description="(latitude, longitude) of origin")
    dest_latlon: Tuple[float, float] = Field(..., description="(latitude, longitude) of destination")
    category: str = Field(..., description="Activity category (e.g. dinner/drinks)")
    est_min: float = Field(..., description="Predicted duration in minutes")
    act_min: float = Field(..., description="Actual observed duration in minutes")

def feedback_data(payload, top_models):
    feedback_df = feedback_preprocess(payload)

    registry_dict = supabase_client.get_latest_registry()

    feedback_df["models_used"] = ", ".join(map(str, top_models))
    feedback_df["feature_registry_id"] = registry_dict["id"]

    load_into_supabase(feedback_df)

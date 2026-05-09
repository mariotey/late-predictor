import pandas as pd
import json
import os
import joblib
import logging
from supabase import create_client
from utils.supbase_info import get_info
from utils.logger import setup_logging
from config import (
    FEATURE_STORE_NAME,
    FEATURE_REGISTRY_NAME,
    FEATURE_REGISTRY_PATH,
    TRAINED_MODELS_PATH,
    TOP_MODELS_PATH,
)

setup_logging()
logger = logging.getLogger(__name__)

SUPABASE_URL, SUPABASE_KEY = get_info()

# Create client once
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_feature_store():
    res = supabase_client.table(FEATURE_STORE_NAME).select("*").execute()
    return pd.DataFrame(res.data)

def get_latest_registry():
    res = (
        supabase_client
        .table(FEATURE_REGISTRY_NAME)
        .select("*")
        .order("id", desc=True)
        .limit(1)
        .execute()
    )

    data = res.data
    return data[0]["config"] if data else None

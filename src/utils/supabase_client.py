import pandas as pd
import json
import os
import joblib
import logging
from dotenv import load_dotenv
from supabase import create_client
from .logger import setup_logging
from config import (
    FEATURES_NAME,
    FEATURE_REGISTRY_NAME,
    FEEDBACK_NAME,
    FEATURE_REGISTRY_ID_COL
)

setup_logging()
logger = logging.getLogger(__name__)

def get_info():
    # Load environment variables
    load_dotenv()

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")

    # Check if the values obtained are valid
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY")

    return SUPABASE_URL, SUPABASE_KEY

SUPABASE_URL, SUPABASE_KEY = get_info()

# Create client once
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_feature_store(table_name = FEATURES_NAME):
    res = supabase_client.table(table_name).select("*").execute()
    return pd.DataFrame(res.data)

def get_latest_registry(table_name = FEATURE_REGISTRY_NAME):
    res = (
        supabase_client
        .table(table_name)
        .select("*")
        .order(FEATURE_REGISTRY_ID_COL, desc=True)
        .limit(1)
        .execute()
    )

    data = res.data
    return data[0] if data else None

def load_into_supabase(df, table_name = FEEDBACK_NAME):
    records = df.to_dict("records")

    # Clean NaNs, although should not be present at this stage
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None

    # Insert data
    supabase_client.table(table_name).insert(records).execute()

    print("✅ Loaded into Supabase successfully\n")

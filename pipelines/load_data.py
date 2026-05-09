import pandas as pd
import json
from supabase import create_client
from utils.supbase_info import get_info
from config import (
    FEATURE_STORE_NAME,
    FEATURE_REGISTRY_NAME,
    FEATURE_REGISTRY_PATH
)

SUPABASE_URL, SUPABASE_KEY = get_info()

# Create client once
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

def extract_feature_store():
    res = supabase_client.table(FEATURE_STORE_NAME).select("*").execute()
    return pd.DataFrame(res.data)

def extract_latest_registry():
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

def extract_cached_registry():
    with open(FEATURE_REGISTRY_PATH, "r") as f:
        feature_registry_dict = json.load(f)

    return feature_registry_dict
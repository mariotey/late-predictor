import pandas as pd
from supabase import create_client
from utils.supabase_client import get_info
from config import APPT_NAME, FEATURE_REGISTRY_NAME, FEATURE_REGISTRY_ID_COL

SUPABASE_URL, SUPABASE_KEY = get_info()

# Create client once
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

def extract_all_data(table_name):
    res = supabase_client.table(table_name).select("*").execute()

    return pd.DataFrame(res.data)

def extract_latest_registry():
    res = (
        supabase_client
        .table(FEATURE_REGISTRY_NAME)
        .select("*")
        .order(FEATURE_REGISTRY_ID_COL, desc=True)
        .limit(1)
        .execute()
    )

    data = res.data
    return data[0]["config"] if data else None
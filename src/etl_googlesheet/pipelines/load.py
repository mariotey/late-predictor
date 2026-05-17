import pandas as pd
from supabase import create_client
import datetime
import math
from utils.supabase_client import get_info
from config import APPT_NAME

SUPABASE_URL, SUPABASE_KEY = get_info()

# Create client once
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

def clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v

def load_to_supabase(df, table_name):
    # Wipe old data first
    supabase_client.rpc(f"truncate_{table_name.lower()}").execute()

    records = [
        {k: clean(v) for k, v in row.items()}
        for row in df.to_dict("records")
    ]

    # Save fresh data
    supabase_client.table(table_name).insert(records).execute()

    print("\n✅ Loaded into Supabase successfully\n")
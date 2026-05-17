import pandas as pd
import json
from supabase import create_client
from .extract import extract_latest_registry
from utils.supabase_client import get_info
from config import FEATURES_NAME, FEATURE_REGISTRY_NAME, FEATURES_ID_COL

SUPABASE_URL, SUPABASE_KEY = get_info()

# Create client once
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

def load_features_into_supabase(df):
    # Wipe old data first
    supabase_client.rpc(f"truncate_{FEATURES_NAME.lower()}").execute()

    # Insert fresh data
    records = [
        {k: v for k, v in row.items()}
        for row in df.to_dict("records")
    ]

    supabase_client.table(FEATURES_NAME).upsert(
        records,
        on_conflict=FEATURES_ID_COL
    ).execute()

    print("✅ Loaded into Supabase successfully\n")


def load_registry_into_supabase(registry_dict):
    # Extract the latest feature registry
    latest = extract_latest_registry()

    # No previous record → insert
    if latest is None:
        supabase_client.table(FEATURE_REGISTRY_NAME).insert({
            "config": registry_dict
        }).execute()

        print("✅ Inserted first registry version\n")
        return

    def normalize(d):
        return json.dumps(d, sort_keys=True)

    # Compare
    if normalize(latest) == normalize(registry_dict):
        print("🟡 Registry unchanged — skipping insert\n")
        return

    # Insert new version
    supabase_client.table(FEATURE_REGISTRY_NAME).insert({
        "config": registry_dict
    }).execute()

    print("✅ Loaded registry into Supabase successfully\n")
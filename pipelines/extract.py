import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
from config import FEATURE_STORE_NAME

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Create client once
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

def extract_feature_store():
    res = supabase_client.table(FEATURE_STORE_NAME).select("*").execute()
    return pd.DataFrame(res.data)

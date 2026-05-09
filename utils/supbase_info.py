import os
from dotenv import load_dotenv

def get_info():
    # Load environment variables
    load_dotenv()

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")

    # Check if the values obtained are valid
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY")

    return SUPABASE_URL, SUPABASE_KEY
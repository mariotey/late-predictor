import pandas as pd

def int_to_datetime(val, date):
    if pd.isna(val):
        return pd.NaT
    try:
        val = int(val)
        hours = val // 100
        minutes = val % 100
        return pd.Timestamp(f"{date} {hours:02d}:{minutes:02d}")
    except:
        return pd.NaT
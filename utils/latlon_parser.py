import pandas as pd

def parse_latlon(x):
    if not x or pd.isna(x):
        return None, None

    try:
        lat, lon = map(float, x.split(","))
        return lat, lon
    except:
        return None, None
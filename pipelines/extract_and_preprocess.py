"""
Pipeline for Data Extraction & Feature Engineering.

It pulls raw data from Google Sheets, cleans and transforms meeting data, engineers features for ML
training and saves processed dataset as a parquet.

CLI usage (from repo root or src/):
    python -m pipelines.extract_and_preprocess
"""
import pandas as pd
from haversine import haversine, Unit
import ast

from config import DATA_URL, CLEAN_DATA_PATH

LONLAT_MAPPING_DICT = {
    "uncle kkm (bukit panjang)": (1.3770731732807227, 103.76455875389925),
    "bbdc": (1.367094085362527, 103.75025701156822),
    "raffles place": (1.2844344943505839, 103.84979077988179),
    "cck": (1.3849417640590647, 103.74457971156812),
    "suntec": (1.2952635056725392, 103.85829414863679),
    "jurong east mrt": (1.33305589084698, 103.7424011999689)
}

def int_to_datetime(val, date):
    if pd.isna(val):
        return pd.NaT
    val = int(val)
    hours = val // 100
    minutes = val % 100
    return pd.Timestamp(f"{date.date()} {hours:02d}:{minutes:02d}")

def calc_distance(row, origin_col, destination_col):
    try:
        origin = ast.literal_eval(row[origin_col]) if isinstance(row[origin_col], str) else row[origin_col]
        destination = ast.literal_eval(row[destination_col]) if isinstance(row[destination_col], str) else row[destination_col]
        return haversine(origin, destination, unit=Unit.KILOMETERS)
    except (ValueError, TypeError):
        return None

def extract_data():
    # Read data from Google Excel Sheet
    df = pd.read_csv(DATA_URL)

    # Drop late flag column since it is not accurate and rename columns
    modified_df = (
        df[["Date", "Location", "Category", "Time to meet", "Actual time arrived", "home_latlon"]]
        .rename(
            columns={
                "Date": "date",
                "Time to meet": "meeting_time",
                "Actual time arrived": "arrived_time",
                "Location": "meeting_location",
                "Category": "category"
            }
        )
    )

    # Convert date to date values
    modified_df["date"] = pd.to_datetime(modified_df["date"])
    modified_df = modified_df.sort_values(by="date")

    # Acquire date features
    modified_df["month"] = modified_df["date"].dt.month
    modified_df["day_of_week"] = modified_df["date"].dt.dayofweek

    # Convert meeting and arrival times to datetime values
    modified_df["meeting_time"] = modified_df.apply(lambda row: int_to_datetime(row["meeting_time"], row["date"]), axis=1)
    modified_df["arrived_time"] = modified_df.apply(lambda row: int_to_datetime(row["arrived_time"], row["date"]), axis=1)

    # Lowercase and clear white spacing
    modified_df["meeting_location"] = modified_df["meeting_location"].str.strip().str.lower()
    modified_df["category"] = modified_df["category"].str.strip().str.lower()

    # Calculate how late in minutes
    modified_df["late_duration_min"] = (modified_df["arrived_time"] - modified_df["meeting_time"]).dt.total_seconds() / 60

    # Drop rows where late_duration_min is NaN
    modified_df = modified_df.dropna(subset="late_duration_min")

    # Drop rows that have virtual meeting since only interested in physical
    modified_df = modified_df[~modified_df["meeting_location"].isin(["virtual", "her house"])]

    # Convert meeting location into latlon
    modified_df["meeting_latlon"] = modified_df["meeting_location"].str.lower().str.strip().map(LONLAT_MAPPING_DICT)

    # Calculate the distance away from home to meeting location
    modified_df["distance_km"] = (
        modified_df.apply(
            calc_distance,
            axis=1,
            origin_col="home_latlon",
            destination_col="meeting_latlon"
        )
    )

    # Filter for feature columns
    feature_df = (
        modified_df[["day_of_week", "distance_km", "category", "late_duration_min"]]
        .reset_index(drop=True)
    )

    # Export Dataset
    feature_df.to_parquet(CLEAN_DATA_PATH)

if __name__ == "__main__":
    extract_data()
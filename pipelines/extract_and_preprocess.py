"""
Pipeline for Data Extraction & Feature Engineering.

It pulls raw data from Google Sheets, cleans and transforms meeting data, engineers features for ML
training and saves processed dataset as a parquet.

CLI usage (from repo root or src/):
    python -m pipelines.extract_and_preprocess
"""
import pandas as pd
import logging
from haversine import haversine, Unit
import ast
from config import DATA_URL, CLEAN_DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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
    try:
        val = int(val)
        hours = val // 100
        minutes = val % 100
        return pd.Timestamp(f"{date.date()} {hours:02d}:{minutes:02d}")

    except Exception as e:
        logger.warning(f"Failed to parse time value={val} on date={date}: {e}")
        return pd.NaT

def calc_distance(row, origin_col, destination_col):
    try:
        origin = ast.literal_eval(row[origin_col]) if isinstance(row[origin_col], str) else row[origin_col]
        destination = ast.literal_eval(row[destination_col]) if isinstance(row[destination_col], str) else row[destination_col]
        return haversine(origin, destination, unit=Unit.KILOMETERS)

    except Exception as e:
        logger.warning(f"{e}: Distance calc failed for: \n {row}")
        return None

def extract_data():
    logger.info("Starting data extraction pipeline")

    # Read data from Google Excel Sheet
    df = pd.read_csv(DATA_URL)

    expected_cols = ["Date", "Location", "Category", "Time to meet", "Actual time arrived", "home_latlon"]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing expected columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")

    # Drop late flag column since it is not accurate and rename columns
    modified_df = (
        df[expected_cols]
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
    null_dates = modified_df["date"].isna().sum()
    logger.info(f"Date parsing complete. Null dates: {null_dates}")

    modified_df = modified_df.sort_values(by="date")

    # Date features
    modified_df["month"] = modified_df["date"].dt.month
    modified_df["day_of_week"] = modified_df["date"].dt.dayofweek

    # Convert meeting and arrival times to datetime values
    modified_df["meeting_time"] = modified_df.apply(
        lambda row: int_to_datetime(row["meeting_time"], row["date"]),
        axis=1
    )
    modified_df["arrived_time"] = modified_df.apply(
        lambda row: int_to_datetime(row["arrived_time"], row["date"]),
        axis=1
    )

    logger.info(
        f"Time conversion done. "
        f"meeting_time NaT: {modified_df['meeting_time'].isna().sum()}, "
        f"arrived_time NaT: {modified_df['arrived_time'].isna().sum()}"
    )

    # Lowercase and clear white spacing
    modified_df["meeting_location"] = modified_df["meeting_location"].str.strip().str.lower()
    modified_df["category"] = modified_df["category"].str.strip().str.lower()

    # Calculate how late in minutes
    modified_df["late_duration_min"] = (
        modified_df["arrived_time"] - modified_df["meeting_time"]
    ).dt.total_seconds() / 60

    logger.info(f"Late duration computed. Null values: {modified_df['late_duration_min'].isna().sum()}")

    # Drop rows where late_duration_min is NaN
    modified_df = modified_df.dropna(subset="late_duration_min")

    # Drop rows that have virtual meeting since only interested in physical
    modified_df = modified_df[~modified_df["meeting_location"].isin(["virtual", "her house"])]

    # Convert meeting location into latlon
    modified_df["meeting_latlon"] = modified_df["meeting_location"].str.lower().str.strip().map(LONLAT_MAPPING_DICT)

    unmapped = modified_df["meeting_latlon"].isna().sum()
    logger.info(f"Unmapped meeting locations: {unmapped}")

    # Calculate the distance away from home to meeting location
    modified_df["distance_km"] = (
        modified_df.apply(
            calc_distance,
            axis=1,
            origin_col="home_latlon",
            destination_col="meeting_latlon"
        )
    )

    logger.info(f"Distance calc complete. Missing distances: {modified_df['distance_km'].isna().sum()}")

    # Filter for feature columns
    feature_df = (
        modified_df[["day_of_week", "distance_km", "category", "late_duration_min"]]
        .reset_index(drop=True)
    )

    # Export Dataset
    feature_df.to_parquet(CLEAN_DATA_PATH)

    logger.info(f"Saved cleaned dataset to {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    extract_data()
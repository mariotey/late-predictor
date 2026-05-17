import numpy as np
import pandas as pd
from utils.distance_cal import get_distance_km

SG_TZ = "Asia/Singapore"
HOME_LATLON = (1.3824797878551964, 103.75444675699774)
HOME_LATLON_COL = "home_latlon"

def get_features(df):
    modified_df = df.copy()

    # Drop rows that have virtual meeting since only interested in physical
    modified_df = modified_df[~modified_df["meeting_location"].isin(["virtual", "her house"])]

    # Date features
    modified_df["date"] = pd.to_datetime(modified_df["date"])
    modified_df["month"] = modified_df["date"].dt.month
    modified_df["day_of_week"] = modified_df["date"].dt.dayofweek
    modified_df = modified_df.sort_values(by="date")

    # Calculate how late in minutes
    modified_df["arrived_time"] = (
        pd.to_datetime(modified_df["arrived_time"], utc=True)
        .dt.tz_convert(SG_TZ)
    )

    modified_df["meeting_time"] = (
        pd.to_datetime(modified_df["meeting_time"], utc=True)
        .dt.tz_convert(SG_TZ)
    )

    modified_df["late_duration_min"] = (
        modified_df["arrived_time"] - modified_df["meeting_time"]
    ).dt.total_seconds() / 60

    # time-of-day bucketing
    modified_df["hour"] = modified_df["meeting_time"].dt.hour
    modified_df["time_of_day"] = np.select(
        [
            (modified_df["hour"] >= 3) & (modified_df["hour"] < 12),
            (modified_df["hour"] >= 12) & (modified_df["hour"] < 18),
            (modified_df["hour"] >= 18) | (modified_df["hour"] < 3),
        ],
        [
            "morning",
            "afternoon",
            "evening"
        ],
        default=None
    )

    # Drop rows where late_duration_min is NaN
    modified_df = modified_df.dropna(subset=["late_duration_min"])

    # Transform meeting latlon into a tuple
    modified_df["meeting_latlon"] = list(zip(
        modified_df["meeting_lat"],
        modified_df["meeting_lon"]
    ))

    # Calculate the distance away from home to meeting location
    modified_df[HOME_LATLON_COL] = [HOME_LATLON] * len(modified_df)

    modified_df["distance_km"] = (
        modified_df.apply(
            get_distance_km,
            axis=1,
            origin_col=HOME_LATLON_COL,
            destination_col="meeting_latlon"
        )
    )

    return modified_df
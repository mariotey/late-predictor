import pandas as pd
import uuid
from utils.time_parser import int_to_datetime
from utils.latlon_parser import parse_latlon

SG_TZ = "Asia/Singapore"

def appt_data(
    df: pd.DataFrame
) -> pd.DataFrame:

    modified_df = df.copy()

    # Convert date to date values
    modified_df["date"] = (
        pd.to_datetime(
            modified_df["date"]
        )
        .dt.date
    )

    # Convert meeting times to datetime values
    modified_df["meeting_time"] = modified_df.apply(
        lambda row: int_to_datetime(row["meeting_time"], row["date"]),
        axis=1
    )

    # Convert arrival times to datetime values
    modified_df["arrived_time"] = modified_df.apply(
        lambda row: int_to_datetime(row["arrived_time"], row["date"]),
        axis=1
    )

    # Enforce Singapore timezone
    modified_df["meeting_time"] = (
        pd.to_datetime(
            modified_df["meeting_time"]
        )
        .dt.tz_localize(SG_TZ)
    )

    modified_df["arrived_time"] = (
        pd.to_datetime(
            modified_df["arrived_time"]
        )
        .dt.tz_localize(SG_TZ)
    )

    # Lowercase and clear white spacing of string columns
    modified_df["meeting_location"] = modified_df["meeting_location"].fillna("").str.strip().str.lower()
    modified_df["category"] = modified_df["category"].fillna("").str.strip().str.lower()

    # Parsing of GeoCoordinates
    modified_df[["meeting_lat", "meeting_lon"]] = (
        modified_df["meeting_latlon"]
        .apply(lambda x: pd.Series(parse_latlon(x)))
    )

    modified_df["meeting_lat"] = pd.to_numeric(modified_df["meeting_lat"], errors="coerce")
    modified_df["meeting_lon"] = pd.to_numeric(modified_df["meeting_lon"], errors="coerce")
    modified_df = modified_df.drop(columns=["meeting_latlon"]).sort_values(by="date")

    # Enforce Data Types
    modified_df["date"] = modified_df["date"].astype(str)
    modified_df["meeting_time"] = modified_df["meeting_time"].astype(str)
    modified_df["arrived_time"] = modified_df["arrived_time"].astype(str)

    return modified_df

def category_data(
    df: pd.DataFrame
) -> pd.DataFrame:
    # Create category lookup table
    category_dim = (
        df[["category"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Add Others
    others_row = pd.DataFrame({
        "category": ["others"]
    })

    category_dim = pd.concat([category_dim, others_row], ignore_index=True)

    # Assign IDs
    category_dim["category_id"] = category_dim.index + 1

    category_dim["category_id"] = [
        str(uuid.uuid4()) for _ in range(len(category_dim))
    ]

    return category_dim
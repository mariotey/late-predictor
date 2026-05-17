import yaml
from .pipelines import extract, transform, load
from config import (
    APPT_NAME,
    CATEGORY_NAME,
    FEATURE_REGISTRY_INPUT_PATH,
    APPT_ID_COL,
    CATEGORY_ID_COL
)


def run_pipeline():
    # Extract
    appt_df = extract.extract_all_data(APPT_NAME)
    print(f"\nExtracted rows: {len(appt_df)}\n")
    print(f"\n {appt_df.dtypes} \n")

    category = extract.extract_all_data(CATEGORY_NAME)
    print(f"\nExtracted rows: {len(appt_df)}\n")
    print(f"\n {category.dtypes} \n")

    # Features
    merged_df = (
        appt_df
        .merge(
            category,
            how="left",
            on=CATEGORY_ID_COL
        )
        .drop(columns=CATEGORY_ID_COL)
    )

    print(merged_df, "\n")
    features_df = transform.get_features(merged_df)

    print(f"Transformed rows: {len(features_df)}")
    print(f"\n {features_df.dtypes} \n")

    # Load feature registry
    with open(FEATURE_REGISTRY_INPUT_PATH, "r") as f:
        feature_registry_dict = yaml.safe_load(f)

    selected_cols = [APPT_ID_COL] + [
        col
        for col_types in feature_registry_dict["feature_col"].values()
        for col in col_types
    ] + [feature_registry_dict["target_col"]]

    print(f"Selected Cols: {selected_cols}\n")

    # Load
    features_df = features_df[selected_cols].reset_index(drop=True)

    print(features_df, "\n")
    load.load_features_into_supabase(features_df)

    print(feature_registry_dict, "\n")
    load.load_registry_into_supabase(feature_registry_dict)


if __name__ == "__main__":
    run_pipeline()

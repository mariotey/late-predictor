from .pipelines import extract, transform, load
from config import APPT_NAME, CATEGORY_NAME

def run_pipeline():
    # Extract
    df = extract.extract_data()

    # Transform
    appt_df = transform.appt_data(df)

    print(appt_df, "\n")

    # Create category lookup table
    category_dim = transform.category_data(appt_df)

    # Join back to Main DataFrame
    appt_df = (
        appt_df
        .merge(
            category_dim,
            on="category",
            how="left"
        )
        .drop(columns=["category"])
    )

    # Load
    print(category_dim, "\n")
    load.load_to_supabase(category_dim, CATEGORY_NAME)

    print(appt_df, "\n")
    load.load_to_supabase(appt_df, APPT_NAME)


if __name__ == "__main__":
    run_pipeline()
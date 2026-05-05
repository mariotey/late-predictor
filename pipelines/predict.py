import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel

class PredictRequest(BaseModel):
    day_of_week: int
    distance_km: float
    category: str

def Cat_LabelEncoding(df, cols):
    modified_df = df.copy()

    le = LabelEncoder()

    for col in cols:
        modified_df[col] = le.fit_transform(modified_df[col])

    return modified_df

def Cat_OneHotEncoding(df, cols):
    modified_df = pd.get_dummies(df, columns=["category", "day_of_week"])

    return modified_df

def run_ensemble_prediction(X_df, trained_models, top_models):
    category_cols = ["category", "day_of_week"]
    target_col = "late_duration_min"
    preds = []

    for name in top_models:
        model_info = trained_models[name]

        if model_info["type"] == "linear":
            preds.append(
                model_info["model"].predict(
                    Cat_OneHotEncoding(X_df, category_cols)
                )
            )
        else:
            preds.append(
                model_info["model"].predict(
                    Cat_LabelEncoding(X_df, category_cols)
                )
            )

    return np.mean(preds, axis=0)[0]
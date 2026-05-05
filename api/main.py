"""
FastAPI ML Service: Training + Model Serving API

This service provides:
- Model training via background job
- Model loading from persisted artifacts
- Foundation for future inference endpoints

CLI usage (from repo root or src/):
    python -m api.main
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/train" -Method POST
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST `
        -ContentType "application/json" `
        -Body '{"day_of_week": <day_of_week>,"distance_km": <distance in km>,"category":<category of activity>}'
"""
from fastapi import FastAPI, BackgroundTasks
import joblib
import os
import pandas as pd
from pipelines.train import train
from pipelines.predict import PredictRequest, run_ensemble_prediction

from config import (
    TRAINED_MODELS_PATH,
    TOP_MODELS_PATH
)

app = FastAPI()

trained_models, top_models = None, None

def load_models():
    global trained_models, top_models

    if not os.path.exists(TRAINED_MODELS_PATH):
        return

    trained_models = joblib.load(TRAINED_MODELS_PATH)
    top_models = joblib.load(TOP_MODELS_PATH)

def retrain_and_reload():
    train()
    load_models()

@app.on_event("startup")
def startup_event():
    load_models()

@app.get("/")
def root():
    return {"message": "API running"}

@app.post("/train")
def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_and_reload)
    return {"status": "Model training started"}

@app.post("/predict")
def predict(payload: PredictRequest):
    target_col = "late_duration_min"

    print("🔥 Received payload:", payload)

    if trained_models is None or top_models is None:
        return {
            "error": "Model not trained yet. Call /train first."
        }

    print("🧠 Models loaded:", trained_models is not None)
    print("🏆 Top models:", top_models)

    X_df = pd.DataFrame([payload.dict()])

    print("📊 Input DataFrame:\n", X_df)

    result_df = X_df.copy()
    result_df[f"pred_{target_col}"] = run_ensemble_prediction(
        X_df,
        trained_models,
        top_models
    )

    result_df["models_used"] = ", ".join(top_models)

    # Convert to JSON-friendly format
    return result_df.to_dict(orient="records")[0]

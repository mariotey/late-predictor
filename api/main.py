"""
FastAPI ML Service: Training + Model Serving API

This service provides:
- Model training via background job
- Model loading from persisted artifacts
- Foundation for future inference endpoints

----------------------------------------------------------------------

HOW TO RUN (CLI)

From project root (LatePredictor/),

uvicorn api.main:app --reload

OR:

python -m api.main

----------------------------------------------------------------------

HOW TO RUN (CLI / POWERSHELL)

curl -X POST "http://127.0.0.1:8000/predict" ^
-H "Content-Type: application/json" ^
-d "{\"datetime_val\":\"2026-05-06T15:30:00Z\",\"init_lonlat\":[103.8,1.3],\"dest_lonlat\":[103.9,1.35],\"category\":\"dinner\"}"

OR:

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body '{"datetime_val":"2026-05-06T15:30:00Z","init_lonlat":[103.8,1.3],"dest_lonlat":[103.9,1.35],"category":"dinner"}'

----------------------------------------------------------------------

HOW TRAINING WORKS

POST /train:
    - Runs train() in background thread
    - Saves model artifacts
    - Reloads models into memory

Note:
    Training is asynchronous and non-blocking.

----------------------------------------------------------------------

"""
from fastapi import FastAPI, BackgroundTasks, HTTPException
import joblib
import os
import numpy as np
import pandas as pd
import logging
from pipelines.train import train
from pipelines.predict import PredictRequest, run_ensemble_prediction
from contextlib import asynccontextmanager

from config import (
    TRAINED_MODELS_PATH,
    TOP_MODELS_PATH
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

trained_models, top_models = None, None


def load_models():
    global trained_models, top_models

    if not os.path.exists(TRAINED_MODELS_PATH):
        logger.info("⚠️ Model file missing. Run /train first.")
        return

    trained_models = joblib.load(TRAINED_MODELS_PATH)
    top_models = joblib.load(TOP_MODELS_PATH)

    logger.info("✅ Models loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("🚀 Loading models...")
        load_models()
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
    yield
    logger.info("🧹 Shutdown complete")


app = FastAPI(lifespan=lifespan)


def retrain_and_reload():
    train()
    load_models()


@app.get("/")
def root():
    return {"message": "API running"}


@app.post("/train")
def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_and_reload)

    return {
        "status": "Model training started",
        "note": "Model may take a few seconds/minutes to become available"
    }


def haversine(coord1, coord2):
    lon1, lat1 = coord1
    lon2, lat2 = coord2

    R = 6371  # Earth radius in km

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


@app.post("/predict")
def predict(payload: PredictRequest):
    logger.info(f"🔥 Received payload: {payload}")

    if trained_models is None or top_models is None:
        return {
            "error": "Model not trained yet. Call /train first."
        }

    logger.info(f"🧠 Models loaded: {trained_models is not None}")
    logger.info(f"🏆 Top models: {top_models}")

    # Derive features
    day_of_week = payload.datetime_val.weekday()

    distance_km = haversine(
        payload.init_lonlat,
        payload.dest_lonlat
    )

    X_df = pd.DataFrame([{
        "day_of_week": day_of_week,
        "distance_km": round(distance_km, 2),
        "category": payload.category
    }])

    logger.info("📊 Input DataFrame: %s", X_df.to_dict(orient="records")[0])

    try:
        pred = run_ensemble_prediction(
            X_df,
            trained_models,
            top_models
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail="Model not trained")

    result = {
        "est_min": float(pred),
        "models_used": list(top_models) if top_models is not None else []
    }

    logger.info(f"Output:\n{result}")

    return result
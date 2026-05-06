"""
FastAPI ML Service: Training + Model Serving API

This service provides:
- Model training via background job
- Model loading from persisted artifacts
- Foundation for future inference endpoints

----------------------------------------------------------------------
HOW TO RUN (CLI)

From project root (LatePredictor/):

    uvicorn api.main:app --reload

OR:

    python -m api.main

----------------------------------------------------------------------
HOW TO TEST INFERENCE (CLI / CURL)

1. Start the API:

    uvicorn api.main:app --reload

2. Send a prediction request:

    curl -X POST "http://127.0.0.1:8000/predict" ^
    -H "Content-Type: application/json" ^
    -d "{\"day_of_week\": 4, \"distance_km\": 10.5, \"category\": \"dinner/drinks\"}"

OR using PowerShell:

    Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
        -Method POST `
        -ContentType "application/json" `
        -Body '{"day_of_week":4,"distance_km":10.5,"category":"dinner/drinks"}'

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


@app.post("/predict")
def predict(payload: PredictRequest):
    logger.info(f"🔥 Received payload: {payload}")

    if trained_models is None or top_models is None:
        return {
            "error": "Model not trained yet. Call /train first."
        }

    logger.info(f"🧠 Models loaded: {trained_models is not None}")
    logger.info(f"🏆 Top models: {top_models}")


    X_df = pd.DataFrame([payload.model_dump()])

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

    return {
        "prediction": float(pred),
        "models_used": list(top_models) if top_models is not None else []
    }
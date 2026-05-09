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
-d "{\"datetime_val\":\"2026-05-06T15:30:00Z\",\"init_lonlat\":[1.3, 103.8],\"dest_lonlat\":[1.35, 103.9],\"category\":\"dinner\"}"

OR:

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body '{"datetime_val":"2026-05-06T15:30:00Z","init_latlon":[1.3, 103.8],"dest_latlon":[1.35, 103.9],"category":"dinner"}'

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
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging

from utils.logger import setup_logging
from pipelines.predict import PredictRequest

from services.ml_service import MLService
from services.feature_registry import refresh_feature_registry

setup_logging()
logger = logging.getLogger(__name__)

ml_service = MLService()

# App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lifespan
@app.on_event("startup")
def startup():
    logger.info("🚀 Starting API...")

    refresh_feature_registry()
    ml_service.load_models()

# Routes
@app.get("/")
def root():
    return {"message": "API running"}


@app.post("/train")
def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(ml_service.retrain)

    return {
        "status": "training started"
    }


@app.post("/predict")
def predict(payload: PredictRequest):

    if ml_service.trained_models is None or ml_service.top_models is None:
        return {"error": "Model not trained yet"}

    return ml_service.predict(payload)
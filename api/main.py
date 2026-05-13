"""
FastAPI ML Service: Training + Model Serving API

This service provides:
- Model training via background job
- Model loading from persisted artifacts
- Foundation for future inference endpoints

----------------------------------------------------------------------

HOW TO RUN (CLI)

From project root (latepredictor/ml),

python -m uvicorn api.main:app --reload

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

Invoke-RestMethod -Uri "http://127.0.0.1:8000/feedback" `
    -Method POST `
    -ContentType "application/json" `
    -Body '{"meeting_location": "Bukit Panjang Plaza", "meeting_datetime":"2026-05-06T15:30:00Z","init_latlon":[1.3, 103.8],"meeting_latlon":[1.35, 103.9],"category_id":"4ea1b39c-3be4-4cb8-9279-0befb9c030a8","pred_min":19,"arrived_datetime":"2026-05-06T18:30:00Z"}'

"""
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import logging

from utils.logger import setup_logging
from pipelines.predict import PredictRequest
from pipelines.data_feedback import DataFeedbackRequest, feedback_data

from services.ml_service import MLService
from services.feature_registry import refresh_feature_registry

# Logging setup
setup_logging()
logger = logging.getLogger(__name__)

# ML service singleton, holds trained models in memory
ml_service = MLService()

# FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup hook
@app.on_event("startup")
def startup():
    logger.info("🚀 Starting API...")

    refresh_feature_registry()
    ml_service.load_models()

# Global validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    logger.error(f"Validation error: {exc.errors()}")

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors()
        }
    )

# Health check endpoint
@app.get("/")
def root():
    return {"message": "API running"}

# Train endpoint
@app.post("/train")
def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(ml_service.retrain)

    return {
        "status": "Training started"
    }

# Prediction endpoint
@app.post("/predict")
def predict(payload: PredictRequest):
    logger.info(f"Received payload: {payload}")

    if ml_service.trained_models is None or ml_service.top_models is None:
        return {"error": "Model not trained yet"}

    return ml_service.predict(payload)

# Feedback endpoint
@app.post("/feedback")
def feedback(payload: DataFeedbackRequest):
    logger.info(f"Received payload: {payload}")

    feedback_data(
        payload,
        ml_service.top_models
    )

    return {
        "status": "Feedback received"
    }
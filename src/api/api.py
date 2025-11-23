# app.py
from fastapi import FastAPI, HTTPException, Body
import pandas as pd
import os
import json
from models.Logistic_Regression import SentimentModel
from utils.paths import CHECKPOINT_DIR, RESULT_DIR

# ------------------------------
# FastAPI setup
# ------------------------------
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment using trained Logistic Regression model (single input)",
    version="1.0.0"
)

# ------------------------------
# Load trained model on startup
# ------------------------------
MODEL_FILE = os.path.join(CHECKPOINT_DIR, "logreg_model.pkl")
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Trained model not found at {MODEL_FILE}")

model = SentimentModel()
model.load(MODEL_FILE)

# ------------------------------
# Load evaluation metrics (optional)
# ------------------------------
METRICS_FILE = os.path.join(RESULT_DIR, "eval_results.json")
if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)
else:
    metrics = None

# ------------------------------
# Endpoints
# ------------------------------
@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running."}


@app.post("/predict")
def predict(text: str = Body(..., embed=True)):
    """
    Returns a single prediction.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    try:
        X_input = [text]
        pred = model.pipeline.predict(X_input)[0]
        return {"prediction": pred}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    """
    Returns evaluation metrics from test.py
    """
    if metrics is None:
        raise HTTPException(status_code=404, detail="Evaluation metrics not found")
    return metrics

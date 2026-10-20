from fastapi.responses import HTMLResponse
from prometheus_client import REGISTRY
import logging
import time
from fastapi import Request
from prometheus_fastapi_instrumentator import Instrumentator
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from src.preprocess import FEATURES


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "heart_disease_model.joblib"

app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Assignment - Heart Disease Prediction",
    version="1.0.0",
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-disease-api")

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def log_api_requests(request: Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - start_time
        logger.exception(
            "API request failed: method=%s path=%s duration=%.3fs",
            request.method,
            request.url.path,
            duration,
        )
        raise

    duration = time.perf_counter() - start_time

    logger.info(
        "API request: method=%s path=%s status=%s duration=%.3fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )

    return response


class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int
    trestbps: float
    chol: float
    fbs: int = Field(ge=0, le=1)
    restecg: int
    thalach: float
    exang: int = Field(ge=0, le=1)
    oldpeak: float
    slope: int
    ca: float
    thal: float


@app.on_event("startup")
def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    app.state.model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, "model"),
    }


@app.post("/predict")
def predict(patient: PatientInput) -> dict[str, Any]:
    if not hasattr(app.state, "model"):
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    patient_data = patient.model_dump()
    input_df = pd.DataFrame([patient_data])

    # Ensure feature order matches model training
    input_df = input_df[FEATURES]

    prediction = int(
        app.state.model.predict(input_df)[0]
    )

    probability = float(
        app.state.model.predict_proba(input_df)[0][1]
    )

    return {
        "prediction": prediction,
        "heart_disease_probability": round(probability, 4),
        "interpretation": (
            "Heart disease predicted"
            if prediction == 1
            else "No heart disease predicted"
        ),
    }
@app.get("/monitor", response_class=HTMLResponse)
def monitoring_dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Heart Disease API Monitoring</title>
        <style>
            body { font-family: Arial; background: #f4f6f9; padding: 30px; }
            h1 { color: #263547; }
            .card {
                background: white;
                padding: 20px;
                margin: 15px 0;
                border-radius: 10px;
                box-shadow: 0 2px 6px #ddd;
            }
            pre { white-space: pre-wrap; overflow-wrap: anywhere; }
        </style>
    </head>
    <body>
        <h1>Heart Disease API Monitoring Dashboard</h1>
        <div class="card">
            <h3>Live API Metrics</h3>
            <pre id="metrics">Loading...</pre>
        </div>
        <script>
            async function refresh() {
                try {
                    const response = await fetch('/metrics');
                    if (!response.ok) throw new Error('HTTP ' + response.status);
                    const text = await response.text();
                    const lines = text.split('\\n').filter(line =>
                        line.startsWith('http_requests_total{') ||
                        line.startsWith('http_request_duration_seconds_sum{') ||
                        line.startsWith('http_request_duration_seconds_count{')
                    );
                    document.getElementById('metrics').textContent =
                        lines.join('\\n') || 'No HTTP metrics yet. Generate API requests.';
                } catch (error) {
                    document.getElementById('metrics').textContent =
                        'Unable to load metrics: ' + error.message;
                }
            }
            refresh();
            setInterval(refresh, 5000);
        </script>
    </body>
    </html>
    """
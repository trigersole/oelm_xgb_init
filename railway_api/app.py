"""
FastAPI inference server for the Open Emotional Learner Model.
Loads four XGBoost classifiers (one per emotion label) and exposes a
single POST /predict endpoint that the browser frontend calls.
"""

import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb

# ── Label configuration ──────────────────────────────────────────
LABEL_COLS = ["Boredom", "Engagement", "Confusion", "Frustration"]

_THIS_DIR = os.path.dirname(__file__)
FEATURE_ORDER_CANDIDATES = [
    os.path.join(_THIS_DIR, "feature_order.pkl"),
    os.path.join(_THIS_DIR, "..", "feature_order.pkl"),
]

_feature_order_path = next((p for p in FEATURE_ORDER_CANDIDATES if os.path.exists(p)), None)
if _feature_order_path is None:
    raise FileNotFoundError(
        "feature_order.pkl not found. Expected in railway_api/ or project root."
    )

FEATURE_ORDER = joblib.load(_feature_order_path)

# Which model files to load (relative to this file).
# Switch to "model_tuned_{label}.ubj" or "model_{label}.ubj" if preferred.
MODEL_FILES = {
    label: os.path.join(os.path.dirname(__file__), "models", f"model_tuned_smote_{label}.ubj")
    for label in LABEL_COLS
}

# ── Load models once at startup ──────────────────────────────────
models: dict[str, xgb.Booster] = {}

def load_models() -> None:
    for label, path in MODEL_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Copy your .ubj model files into railway_api/models/"
            )
        booster = xgb.Booster()
        booster.load_model(path)
        models[label] = booster
    print(f"[OELM] Loaded {len(models)} models: {list(models)}")

# ── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(title="OELM XGBoost API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your frontend domain in production
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event() -> None:
    load_models()

# ── Request / Response schemas ───────────────────────────────────
class PredictRequest(BaseModel):
    # 364-key dict of aggregated blendshape statistics
    agg_features: dict[str, float]

class LabelPrediction(BaseModel):
    label: int                      # 0=Very Low, 1=Low, 2=High, 3=Very High
    probabilities: dict[int, float] # {0: p0, 1: p1, 2: p2, 3: p3}

class PredictResponse(BaseModel):
    Boredom:     LabelPrediction
    Engagement:  LabelPrediction
    Confusion:   LabelPrediction
    Frustration: LabelPrediction

# ── Health check ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}

# ── Predict endpoint ─────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    agg = body.agg_features
    missing = [k for k in FEATURE_ORDER if k not in agg]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing {len(missing)} features. First 5: {missing[:5]}",
        )

    x_row = np.array([[agg[k] for k in FEATURE_ORDER]], dtype=np.float32)
    dmat = xgb.DMatrix(x_row, feature_names=FEATURE_ORDER)

    result: dict = {}

    for label, booster in models.items():
        probs = booster.predict(dmat)
        if isinstance(probs, np.ndarray) and probs.ndim > 1:
            probs = probs[0]
        probs = np.array(probs, dtype=float)

        if probs.ndim == 0:
            scalar = int(probs.item())
            probs = np.eye(4)[scalar]

        predicted_class = int(np.argmax(probs))

        result[label] = {
            "label": predicted_class,
            "probabilities": {i: round(float(p), 5) for i, p in enumerate(probs)},
        }

    return result

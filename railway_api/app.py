"""
FastAPI inference server for the Open Emotional Learner Model.
Loads four XGBoost classifiers (one per emotion label) and exposes a
single POST /predict endpoint that the browser frontend calls.
"""

import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from xgboost import XGBClassifier

# ── Label configuration ──────────────────────────────────────────
LABEL_COLS = ["Boredom", "Engagement", "Confusion", "Frustration"]

# Which model files to load (relative to this file).
# Switch to "model_tuned_{label}.ubj" or "model_{label}.ubj" if preferred.
MODEL_FILES = {
    label: os.path.join(os.path.dirname(__file__), "models", f"model_tuned_smote_{label}.ubj")
    for label in LABEL_COLS
}

# ── Load models once at startup ──────────────────────────────────
models: dict[str, XGBClassifier] = {}

def load_models() -> None:
    for label, path in MODEL_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Copy your .ubj model files into railway_api/models/"
            )
        clf = XGBClassifier()
        clf.load_model(path)
        models[label] = clf
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
    result: dict = {}

    for label, clf in models.items():
        # Respect the feature order stored inside the booster
        booster_features = clf.get_booster().feature_names
        if booster_features:
            x_row = np.array([[agg.get(f, 0.0) for f in booster_features]], dtype=np.float32)
        else:
            x_row = np.array([list(agg.values())], dtype=np.float32)

        predicted_class = int(clf.predict(x_row)[0])
        proba = clf.predict_proba(x_row)[0]   # shape (n_classes,)

        result[label] = {
            "label": predicted_class,
            "probabilities": {i: round(float(p), 5) for i, p in enumerate(proba)},
        }

    return result

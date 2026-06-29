"""
Open Emotional Learner — HuggingFace Space
==========================================
Hosts 4 XGBoost classifiers (one per label) as a single Gradio API.
Labels:  Boredom | Engagement | Confusion | Frustration
Input:   dict of 364 aggregated blendshape features
         key format: "<blendshape_name>_<stat>"
         e.g. {"browInnerUp_mean": 0.12, "browInnerUp_std": 0.03, ...}
         52 blendshapes × 7 stats = 364 features total
Output:  {
           "Boredom":     {"label": 0, "probabilities": [0.8, 0.1, 0.05, 0.05]},
           "Engagement":  {"label": 2, "probabilities": [0.1, 0.2, 0.6,  0.1]},
           "Confusion":   {"label": 1, "probabilities": [0.3, 0.5, 0.1,  0.1]},
           "Frustration": {"label": 0, "probabilities": [0.7, 0.2, 0.05, 0.05]},
           "window_seconds": 10
         }
         label int = ordinal level (0 = very low ... 3 = very high)
Deploy:
  1. Create HuggingFace Space (SDK: Gradio)
  2. Upload this file + requirements.txt + model files:
       model_Boredom.ubj  model_Engagement.ubj
       model_Confusion.ubj  model_Frustration.ubj
  3. /api/predict is auto-exposed by Gradio
Swap model later: replace .ubj files, set MODEL_BACKEND env var, redeploy.
Streamlit side never needs to change.
"""

import os
import numpy as np
import gradio as gr
import joblib

# ── Label & stat config ───────────────────────────────────────
LABEL_COLS   = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
STAT_SUFFIXES = ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurt']

BLENDSHAPE_NAMES = [
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]  # 52 names

# 364-feature order — must exactly match training column order
FEATURE_ORDER = joblib.load("feature_order.pkl")
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "xgboost")


# ── Model loading ─────────────────────────────────────────────
def _load_models() -> dict:
    models = {}
    for label in LABEL_COLS:
        if MODEL_BACKEND == "xgboost":
            import xgboost as xgb
            b = xgb.Booster()
            b.load_model(f"model_{label}.ubj")
            models[label] = b

        elif MODEL_BACKEND == "sklearn":
            import joblib
            models[label] = joblib.load(f"model_{label}.joblib")

        elif MODEL_BACKEND == "onnx":
            import onnxruntime as ort
            models[label] = ort.InferenceSession(f"model_{label}.onnx")

        else:
            raise ValueError(f"Unknown MODEL_BACKEND: {MODEL_BACKEND}")
    return models

_models = _load_models()


# ── Single-label inference ────────────────────────────────────
def _predict_one(label: str, x: np.ndarray) -> dict:
    """Return {"label": int, "probabilities": [float, ...]} for one label."""
    
    model = _models[label]

    if MODEL_BACKEND == "xgboost":
        import xgboost as xgb

        dmat = xgb.DMatrix(x, feature_names=FEATURE_ORDER)

        probs = model.predict(dmat)

        # Convert batch output -> single prediction vector
        if isinstance(probs, np.ndarray) and probs.ndim > 1:
            probs = probs[0]

    elif MODEL_BACKEND == "sklearn":
        probs = model.predict_proba(x)[0]

    elif MODEL_BACKEND == "onnx":
        inp = model.get_inputs()[0].name
        probs = model.run(None, {inp: x.astype(np.float32)})[0][0]

    probs = np.array(probs, dtype=float)

    # Handle scalar prediction edge case
    if probs.ndim == 0:
        scalar = int(probs.item())
        probs = np.eye(4)[scalar]

    return {
        "label": int(np.argmax(probs)),
        "probabilities": [round(float(p), 4) for p in probs],
    }


# ── Main API endpoint ─────────────────────────────────────────
def predict(agg_features: dict) -> dict:
    """
    Input:  dict of 364 keys  →  "<blendshape>_<stat>": float
    Output: dict with one entry per label + window_seconds
    Example input key: "browInnerUp_mean"
    Example output:
        {
          "Boredom":     {"label": 0, "probabilities": [0.8, 0.1, 0.05, 0.05]},
          "Engagement":  {"label": 2, "probabilities": [0.1, 0.2, 0.6,  0.1]},
          "Confusion":   {"label": 1, "probabilities": [0.3, 0.5, 0.1,  0.1]},
          "Frustration": {"label": 0, "probabilities": [0.7, 0.2, 0.05, 0.05]},
          "window_seconds": 10
        }
    """
    # Validate
    missing = [k for k in FEATURE_ORDER if k not in agg_features]
    if missing:
        return {"error": f"Missing {len(missing)} features. First 5: {missing[:5]}"}

    # Build feature vector in training column order
    x = np.array(
        [[agg_features[k] for k in FEATURE_ORDER]],
        dtype=np.float32
    )

    result = {label: _predict_one(label, x) for label in LABEL_COLS}
    result["window_seconds"] = 10
    return result


# ── Health check ──────────────────────────────────────────────
def health() -> dict:
    return {
        "status":        "ok",
        "model_backend": MODEL_BACKEND,
        "labels":        LABEL_COLS,
        "n_features":    len(FEATURE_ORDER),      # 364
        "n_blendshapes": len(BLENDSHAPE_NAMES),   # 52
        "n_stats":       len(STAT_SUFFIXES),       # 7
    }


# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(title="Engagement State Classifier") as demo:
    gr.Markdown("## Engagement State Classifier — Blendshape Aggregation API")
    gr.Markdown(
        "**Input**: 364 aggregated features (`<blendshape>_<stat>` keys)  \n"
        "**Output**: Boredom / Engagement / Confusion / Frustration (ordinal 0–3)"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.JSON(label="Aggregated features (364 keys)")
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            out = gr.JSON(label="4-label prediction output")

    btn.click(fn=predict, inputs=inp, outputs=out, api_name="predict")

    gr.Markdown("---")
    health_btn = gr.Button("Health check")
    health_out = gr.JSON()
    health_btn.click(fn=health, outputs=health_out, api_name="health")

if __name__ == "__main__":
    demo.launch(show_error=True)


from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"


@lru_cache(maxsize=1)
def load_artifacts():
    model = joblib.load(MODEL_DIR / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    imputer = joblib.load(MODEL_DIR / "imputer.pkl")
    features = joblib.load(MODEL_DIR / "features.pkl")
    return model, scaler, imputer, features


def predict_heart_disease(values):
    model, scaler, imputer, features = load_artifacts()

    # Keep input columns in the exact training order.
    row = np.array(
        [[float(values[name]) for name in features]],
        dtype=float
    )

    row = np.where(np.isfinite(row), row, np.nan)
    row = imputer.transform(row)
    row = scaler.transform(row)

    prediction = int(model.predict(row)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(row)[0][1])

    return {
        "prediction": prediction,
        "result": (
            "Higher predicted likelihood of heart disease"
            if prediction == 1
            else "Lower predicted likelihood of heart disease"
        ),
        "probability": probability
    }

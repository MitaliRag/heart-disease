
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.predict import predict_heart_disease

BASE_DIR = Path(__file__).resolve().parents[1]
PUBLIC_DIR = BASE_DIR / "public"

app = FastAPI(title="Heart Disease Prediction API")

app.mount(
    "/static",
    StaticFiles(directory=str(PUBLIC_DIR)),
    name="static"
)


class HeartInput(BaseModel):
    age: float = Field(gt=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(ge=0, le=1000)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-10, le=20)
    slope: int = Field(ge=0, le=3)
    ca: float = Field(ge=0, le=4)
    thal: float = Field(ge=0, le=7)


@app.get("/")
def home():
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "message": "Prediction API is running"}


@app.post("/api/predict")
def predict(data: HeartInput):
    try:
        return predict_heart_disease(data.model_dump())
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model files are missing. Run train_model.py first."
        )
    except Exception as exc:
        print("Prediction error:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the model and dataset."
        )

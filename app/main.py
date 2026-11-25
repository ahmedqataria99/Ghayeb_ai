from contextlib import asynccontextmanager
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "material_demand_model.pkl"

FEATURE_COLUMNS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_mean_7", "sales_mean_14", "sales_mean_28",
    "sales_std_7", "sales_std_14", "sales_std_28",
    "price_lag_1", "price_mean_7",
    "forecast_day_of_week", "forecast_month",
]


class HistoricalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: date
    units_sold: float = Field(ge=0)
    unit_price: float = Field(gt=0)

    @model_validator(mode="after")
    def finite_values(self):
        if not np.isfinite(self.units_sold):
            raise ValueError("units_sold must be finite")
        if not np.isfinite(self.unit_price):
            raise ValueError("unit_price must be finite")
        return self


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forecast_date: date
    history: List[HistoricalRecord] = Field(min_length=29)

    @model_validator(mode="after")
    def validate_history(self):
        validate_history_records(self.history, self.forecast_date)
        return self


class EngineeredPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    predicted_demand_next_7_days: float


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requests: List[PredictionRequest] = Field(min_length=1, max_length=100)


class MaterialDemandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forecast_date: date
    forecast_horizon_days: Literal[7] = 7
    current_stock: float = Field(ge=0)
    history: List[HistoricalRecord] = Field(min_length=29)

    @model_validator(mode="after")
    def validate_history(self):
        validate_history_records(self.history, self.forecast_date)
        return self


class MaterialDemandResponse(BaseModel):
    success: bool
    forecast_horizon_days: int
    predicted_demand_units: float
    current_stock: float
    suggested_restock_quantity: float
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    feature_count: int
    feature_names: List[str]
    feature_order: List[str]
    model_sha256: str
    sklearn_version: str


model = None
model_version = "unavailable"
model_sha256 = "unavailable"


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Ghayeb AI Material Demand API",
    version="1.0.0",
    description="7-day material demand forecasts using the exported HistGradientBoosting pipeline.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


def validate_history_records(history: List[HistoricalRecord], forecast_date: date) -> None:
    ordered = sorted(history, key=lambda record: record.date)
    dates = [record.date for record in ordered]

    if dates != [record.date for record in history]:
        raise ValueError("history must be ordered chronologically from oldest to newest")
    if len(set(dates)) != len(dates):
        raise ValueError("history contains duplicate dates")

    for previous, current in zip(dates, dates[1:]):
        if current != previous + timedelta(days=1):
            raise ValueError("history must contain every consecutive day without gaps")

    if forecast_date != dates[-1] + timedelta(days=1):
        raise ValueError("forecast_date must be exactly one day after the newest history date")


def load_model() -> None:
    global model, model_version, model_sha256
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")

    model_sha256 = sha256(MODEL_PATH.read_bytes()).hexdigest()
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifact: {exc}") from exc

    count = getattr(model, "n_features_in_", None)
    if count != len(FEATURE_COLUMNS):
        raise RuntimeError(f"Loaded model expects {count} features; expected {len(FEATURE_COLUMNS)}")

    names = list(getattr(model, "feature_names_in_", []))
    if names:
        if names != FEATURE_COLUMNS:
            raise RuntimeError("Loaded model feature names/order do not match notebook FEATURE_COLUMNS")

    model_version = f"{sklearn.__version__}-{model_sha256[:12]}"


def build_features(request: PredictionRequest) -> pd.DataFrame:
    sales = pd.Series([record.units_sold for record in request.history], dtype="float64")
    prices = pd.Series([record.unit_price for record in request.history], dtype="float64")

    values = {
        "sales_lag_1": sales.iloc[-2],
        "sales_lag_7": sales.iloc[-8],
        "sales_lag_14": sales.iloc[-15],
        "sales_lag_28": sales.iloc[-29],
        "sales_mean_7": sales.iloc[-8:-1].mean(),
        "sales_mean_14": sales.iloc[-15:-1].mean(),
        "sales_mean_28": sales.iloc[-29:-1].mean(),
        "sales_std_7": sales.iloc[-8:-1].std(ddof=1),
        "sales_std_14": sales.iloc[-15:-1].std(ddof=1),
        "sales_std_28": sales.iloc[-29:-1].std(ddof=1),
        "price_lag_1": prices.iloc[-2],
        "price_mean_7": prices.iloc[-8:-1].mean(),
        "forecast_day_of_week": request.forecast_date.weekday(),
        "forecast_month": request.forecast_date.month,
    }

    return pd.DataFrame([[values[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def validate_prediction_frame(features: pd.DataFrame) -> None:
    if list(features.columns) != FEATURE_COLUMNS:
        raise HTTPException(status_code=500, detail="Prediction features are in the wrong order")
    if features.shape[1] != len(FEATURE_COLUMNS):
        raise HTTPException(status_code=500, detail="Prediction feature count is invalid")
    if not np.all(np.isfinite(features.to_numpy(dtype="float64"))):
        raise HTTPException(status_code=500, detail="Prediction features contain non-finite values")


def predict_frame(features: pd.DataFrame) -> float:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    validate_prediction_frame(features)
    raw_predictions = np.asarray(model.predict(features), dtype="float64").reshape(-1)
    if raw_predictions.size != 1:
        raise HTTPException(status_code=500, detail="Model returned an unexpected number of predictions")
    raw_prediction = float(raw_predictions[0])

    if not np.isfinite(raw_prediction):
        raise HTTPException(status_code=500, detail="Model returned a non-finite prediction")
    return max(raw_prediction, 0.0)


def predict_batch_frames(frames: List[pd.DataFrame]) -> List[float]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    features = pd.concat(frames, ignore_index=True)
    validate_prediction_frame(features)
    raw_predictions = np.asarray(model.predict(features), dtype="float64").reshape(-1)

    if raw_predictions.size != len(frames):
        raise HTTPException(status_code=500, detail="Model returned an unexpected number of predictions")
    if not np.all(np.isfinite(raw_predictions)):
        raise HTTPException(status_code=500, detail="Model returned a non-finite prediction")
    return [float(max(prediction, 0.0)) for prediction in raw_predictions]


@app.get("/", tags=["General"])
def root() -> Dict[str, str]:
    return {"name": "Ghayeb AI Material Demand API", "status": "ready"}


@app.get("/health", tags=["General"])
def health() -> Dict[str, str]:
    return {"status": "healthy", "model": "loaded" if model is not None else "not_loaded"}


@app.get("/v1/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info() -> ModelInfoResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return ModelInfoResponse(
        model_version=model_version,
        model_type=f"{type(model).__module__}.{type(model).__name__}",
        feature_count=len(FEATURE_COLUMNS),
        feature_names=list(getattr(model, "feature_names_in_", FEATURE_COLUMNS)),
        feature_order=FEATURE_COLUMNS,
        model_sha256=model_sha256,
        sklearn_version=sklearn.__version__,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(request: PredictionRequest) -> PredictionResponse:
    return PredictionResponse(predicted_demand_next_7_days=predict_frame(build_features(request)))


@app.post("/predict/engineered", response_model=PredictionResponse, tags=["Predictions"])
def predict_engineered(request: EngineeredPredictionRequest) -> PredictionResponse:
    missing = [column for column in FEATURE_COLUMNS if column not in request.features]
    extra = [column for column in request.features if column not in FEATURE_COLUMNS]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})
    if extra:
        raise HTTPException(status_code=422, detail={"unexpected_features": extra})

    values = {column: float(request.features[column]) for column in FEATURE_COLUMNS}
    if not np.all(np.isfinite(list(values.values()))):
        raise HTTPException(status_code=422, detail="Engineered features must be finite")

    frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    return PredictionResponse(predicted_demand_next_7_days=predict_frame(frame))


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Predictions"])
def predict_batch(request: BatchPredictionRequest) -> List[PredictionResponse]:
    predictions = predict_batch_frames([build_features(item) for item in request.requests])
    return [PredictionResponse(predicted_demand_next_7_days=value) for value in predictions]


@app.post("/v1/materials/demand-forecast", response_model=MaterialDemandResponse, tags=["Materials"])
def demand_forecast(request: MaterialDemandRequest) -> MaterialDemandResponse:
    prediction_request = PredictionRequest(forecast_date=request.forecast_date, history=request.history)
    predicted = predict_frame(build_features(prediction_request))
    restock = max(0.0, predicted - request.current_stock)

    return MaterialDemandResponse(
        success=True,
        forecast_horizon_days=7,
        predicted_demand_units=predicted,
        current_stock=request.current_stock,
        suggested_restock_quantity=restock,
        model_version=model_version,
    )

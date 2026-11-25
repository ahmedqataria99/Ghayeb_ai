import math
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import FEATURE_COLUMNS, MODEL_PATH, PredictionRequest, app, build_features


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def dated_history(start: date = date(2025, 1, 1), sales_start: float = 100.0):
    return [
        {
            "date": (start + timedelta(days=index)).isoformat(),
            "units_sold": sales_start + (index % 5),
            "unit_price": 50.0 + (index % 3),
        }
        for index in range(29)
    ]


def valid_payload():
    history = dated_history()
    return {"forecast_date": "2025-01-30", "history": history}


def test_health_loads_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model": "loaded"}


def test_feature_order_matches_model(client):
    expected = [
        "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
        "sales_mean_7", "sales_mean_14", "sales_mean_28",
        "sales_std_7", "sales_std_14", "sales_std_28",
        "price_lag_1", "price_mean_7",
        "forecast_day_of_week", "forecast_month",
    ]
    assert FEATURE_COLUMNS == expected

    response = client.get("/v1/model-info")
    assert response.status_code == 200
    assert response.json()["feature_order"] == expected
    assert response.json()["feature_names"] == expected


def test_notebook_api_feature_engineering_parity():
    source = pd.read_csv("sales_data.csv")
    source.columns = source.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    sample = source[(source.store_id == "S001") & (source.product_id == "P0001")].copy()
    sample["date"] = pd.to_datetime(sample["date"])
    sample = sample.sort_values("date").head(29)

    history = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "units_sold": float(row.units_sold),
            "unit_price": float(row.price),
        }
        for row in sample.itertuples()
    ]
    request = PredictionRequest(forecast_date=sample.iloc[-1].date.date() + timedelta(days=1), history=history)
    api_features = build_features(request).iloc[0]

    sales = pd.Series(sample.units_sold.astype("float64").tolist())
    prices = pd.Series(sample.price.astype("float64").tolist())

    for lag in (1, 7, 14, 28):
        expected = sales.shift(1).iloc[-lag]
        assert math.isclose(api_features[f"sales_lag_{lag}"], expected, rel_tol=1e-9, abs_tol=1e-9)

    for window in (7, 14, 28):
        expected_mean = sales.shift(1).rolling(window=window, min_periods=window).mean().iloc[-1]
        expected_std = sales.shift(1).rolling(window=window, min_periods=window).std().iloc[-1]
        assert math.isclose(api_features[f"sales_mean_{window}"], expected_mean, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(api_features[f"sales_std_{window}"], expected_std, rel_tol=1e-9, abs_tol=1e-9)

    assert math.isclose(api_features.price_lag_1, prices.shift(1).iloc[-1], rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(api_features.price_mean_7, prices.shift(1).rolling(window=7, min_periods=7).mean().iloc[-1], rel_tol=1e-9, abs_tol=1e-9)

    forecast_start = pd.Timestamp(request.forecast_date)
    assert api_features.forecast_day_of_week == forecast_start.day_of_week
    assert api_features.forecast_month == forecast_start.month


def test_raw_prediction_is_valid(client):
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    value = response.json()["predicted_demand_next_7_days"]
    assert math.isfinite(value)
    assert value >= 0


def test_engineered_prediction_is_valid(client):
    response = client.post("/predict/engineered", json={"features": {column: 10.0 for column in FEATURE_COLUMNS}})
    assert response.status_code == 200
    value = response.json()["predicted_demand_next_7_days"]
    assert math.isfinite(value)
    assert value >= 0


def test_batch_prediction_returns_one_result_per_request(client):
    item = valid_payload()
    response = client.post("/predict/batch", json={"requests": [item, item]})
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert all(math.isfinite(row["predicted_demand_next_7_days"]) for row in response.json())


def test_duplicate_dates_are_rejected(client):
    payload = valid_payload()
    payload["history"][28] = dict(payload["history"][27])
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_dates_are_rejected(client):
    payload = valid_payload()
    payload["history"][28]["date"] = "2025-02-02"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_forecast_date_must_follow_history(client):
    payload = valid_payload()
    payload["forecast_date"] = "2025-02-01"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_negative_sales_are_rejected(client):
    payload = valid_payload()
    payload["history"][0]["units_sold"] = -1
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_price_is_rejected(client):
    payload = valid_payload()
    payload["history"][0]["unit_price"] = 0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_short_history_is_rejected(client):
    payload = valid_payload()
    payload["history"] = payload["history"][:28]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_model_info(client):
    response = client.get("/v1/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["feature_count"] == 14
    assert body["model_sha256"] == sha256(MODEL_PATH.read_bytes()).hexdigest()


def test_restock_calculation(client):
    payload = valid_payload()
    payload["current_stock"] = 100
    payload["forecast_horizon_days"] = 7
    response = client.post("/v1/materials/demand-forecast", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["forecast_horizon_days"] == 7
    assert body["suggested_restock_quantity"] == max(0.0, body["predicted_demand_units"] - 100)


def test_unsupported_forecast_horizon(client):
    payload = valid_payload()
    payload["current_stock"] = 100
    payload["forecast_horizon_days"] = 14
    response = client.post("/v1/materials/demand-forecast", json=payload)
    assert response.status_code == 422


def test_exported_pkl_is_consistent():
    original = Path("material_demand_model.pkl")
    exported = MODEL_PATH
    assert sha256(original.read_bytes()).hexdigest() == sha256(exported.read_bytes()).hexdigest()

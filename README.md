# Ghayeb AI Material Demand API

This FastAPI backend serves the existing `models/material_demand_model.pkl` artifact. The model is not retrained or replaced.

## Setup

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The application requires scikit-learn `1.6.1`, matching the environment used by the exported pipeline.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Date alignment

For a feature row on historical date `D`, the notebook predicts demand for `D+1` through `D+7`.

- Supply at least 29 consecutive daily records through date `D`.
- `forecast_date` must equal `D+1`.
- `sales_lag_1` and `price_lag_1` use `D-1`, not `D`.
- Lag features use `D-1`, `D-7`, `D-14`, and `D-28`.
- Rolling sales and price features use shifted windows ending at `D-1`.
- Rolling standard deviations use the pandas notebook default `ddof=1`.
- Missing dates and duplicate dates are rejected. Missing sales are never fabricated.
- Zero-sales days are valid.

## Endpoints

### `GET /`

Returns basic service status.

### `GET /health`

Returns application and model-loaded status.

### `GET /v1/model-info`

Returns model version, type, SHA-256, scikit-learn version, and exact feature names/order.

### `POST /predict`

Returns seven-day demand for the legacy prediction route using dated daily history.

Request schema:

```json
{
  "forecast_date": "2025-01-30",
  "history": [
    {"date":"2025-01-01","units_sold":100,"unit_price":50.0},
    {"date":"2025-01-02","units_sold":101,"unit_price":50.1},
    {"date":"2025-01-03","units_sold":102,"unit_price":50.2},
    {"date":"2025-01-04","units_sold":103,"unit_price":50.3},
    {"date":"2025-01-05","units_sold":104,"unit_price":50.4},
    {"date":"2025-01-06","units_sold":105,"unit_price":50.5},
    {"date":"2025-01-07","units_sold":106,"unit_price":50.6},
    {"date":"2025-01-08","units_sold":107,"unit_price":50.7},
    {"date":"2025-01-09","units_sold":108,"unit_price":50.8},
    {"date":"2025-01-10","units_sold":109,"unit_price":50.9},
    {"date":"2025-01-11","units_sold":110,"unit_price":51.0},
    {"date":"2025-01-12","units_sold":111,"unit_price":51.1},
    {"date":"2025-01-13","units_sold":112,"unit_price":51.2},
    {"date":"2025-01-14","units_sold":113,"unit_price":51.3},
    {"date":"2025-01-15","units_sold":114,"unit_price":51.4},
    {"date":"2025-01-16","units_sold":115,"unit_price":51.5},
    {"date":"2025-01-17","units_sold":116,"unit_price":51.6},
    {"date":"2025-01-18","units_sold":117,"unit_price":51.7},
    {"date":"2025-01-19","units_sold":118,"unit_price":51.8},
    {"date":"2025-01-20","units_sold":119,"unit_price":51.9},
    {"date":"2025-01-21","units_sold":120,"unit_price":52.0},
    {"date":"2025-01-22","units_sold":121,"unit_price":52.1},
    {"date":"2025-01-23","units_sold":122,"unit_price":52.2},
    {"date":"2025-01-24","units_sold":123,"unit_price":52.3},
    {"date":"2025-01-25","units_sold":124,"unit_price":52.4},
    {"date":"2025-01-26","units_sold":125,"unit_price":52.5},
    {"date":"2025-01-27","units_sold":126,"unit_price":52.6},
    {"date":"2025-01-28","units_sold":127,"unit_price":52.7},
    {"date":"2025-01-29","units_sold":128,"unit_price":52.8}
  ]
}
```

Real requests require 29 or more records with no missing or duplicate dates.

Response:

```json
{"predicted_demand_next_7_days": 867.4861792747859}
```

### `POST /predict/engineered`

Accepts all 14 already-engineered feature values. Feature names must match exactly:

```json
{
  "features": {
    "sales_lag_1": 100,
    "sales_lag_7": 98,
    "sales_lag_14": 101,
    "sales_lag_28": 95,
    "sales_mean_7": 99,
    "sales_mean_14": 100,
    "sales_mean_28": 101,
    "sales_std_7": 4.2,
    "sales_std_14": 5.1,
    "sales_std_28": 6.3,
    "price_lag_1": 50,
    "price_mean_7": 49.8,
    "forecast_day_of_week": 3,
    "forecast_month": 1
  }
}
```

### `POST /predict/batch`

Request:

```json
{
  "requests": [
    {
      "forecast_date": "2025-01-30",
      "history": [
        {"date":"2025-01-01","units_sold":100,"unit_price":50.0},
        {"date":"2025-01-02","units_sold":101,"unit_price":50.1},
        {"date":"2025-01-03","units_sold":102,"unit_price":50.2},
        {"date":"2025-01-04","units_sold":103,"unit_price":50.3},
        {"date":"2025-01-05","units_sold":104,"unit_price":50.4},
        {"date":"2025-01-06","units_sold":105,"unit_price":50.5},
        {"date":"2025-01-07","units_sold":106,"unit_price":50.6},
        {"date":"2025-01-08","units_sold":107,"unit_price":50.7},
        {"date":"2025-01-09","units_sold":108,"unit_price":50.8},
        {"date":"2025-01-10","units_sold":109,"unit_price":50.9},
        {"date":"2025-01-11","units_sold":110,"unit_price":51.0},
        {"date":"2025-01-12","units_sold":111,"unit_price":51.1},
        {"date":"2025-01-13","units_sold":112,"unit_price":51.2},
        {"date":"2025-01-14","units_sold":113,"unit_price":51.3},
        {"date":"2025-01-15","units_sold":114,"unit_price":51.4},
        {"date":"2025-01-16","units_sold":115,"unit_price":51.5},
        {"date":"2025-01-17","units_sold":116,"unit_price":51.6},
        {"date":"2025-01-18","units_sold":117,"unit_price":51.7},
        {"date":"2025-01-19","units_sold":118,"unit_price":51.8},
        {"date":"2025-01-20","units_sold":119,"unit_price":51.9},
        {"date":"2025-01-21","units_sold":120,"unit_price":52.0},
        {"date":"2025-01-22","units_sold":121,"unit_price":52.1},
        {"date":"2025-01-23","units_sold":122,"unit_price":52.2},
        {"date":"2025-01-24","units_sold":123,"unit_price":52.3},
        {"date":"2025-01-25","units_sold":124,"unit_price":52.4},
        {"date":"2025-01-26","units_sold":125,"unit_price":52.5},
        {"date":"2025-01-27","units_sold":126,"unit_price":52.6},
        {"date":"2025-01-28","units_sold":127,"unit_price":52.7},
        {"date":"2025-01-29","units_sold":128,"unit_price":52.8}
      ]
    }
  ]
}
```

Each item requires 29 or more consecutive history records. The response is an array of `/predict` response objects, with one result per request.

### `POST /v1/materials/demand-forecast`

Request:

```json
{
  "forecast_date": "2025-01-30",
  "forecast_horizon_days": 7,
  "current_stock": 500,
  "history": [
    {"date":"2025-01-01","units_sold":100,"unit_price":50.0},
    {"date":"2025-01-02","units_sold":101,"unit_price":50.1},
    {"date":"2025-01-03","units_sold":102,"unit_price":50.2},
    {"date":"2025-01-04","units_sold":103,"unit_price":50.3},
    {"date":"2025-01-05","units_sold":104,"unit_price":50.4},
    {"date":"2025-01-06","units_sold":105,"unit_price":50.5},
    {"date":"2025-01-07","units_sold":106,"unit_price":50.6},
    {"date":"2025-01-08","units_sold":107,"unit_price":50.7},
    {"date":"2025-01-09","units_sold":108,"unit_price":50.8},
    {"date":"2025-01-10","units_sold":109,"unit_price":50.9},
    {"date":"2025-01-11","units_sold":110,"unit_price":51.0},
    {"date":"2025-01-12","units_sold":111,"unit_price":51.1},
    {"date":"2025-01-13","units_sold":112,"unit_price":51.2},
    {"date":"2025-01-14","units_sold":113,"unit_price":51.3},
    {"date":"2025-01-15","units_sold":114,"unit_price":51.4},
    {"date":"2025-01-16","units_sold":115,"unit_price":51.5},
    {"date":"2025-01-17","units_sold":116,"unit_price":51.6},
    {"date":"2025-01-18","units_sold":117,"unit_price":51.7},
    {"date":"2025-01-19","units_sold":118,"unit_price":51.8},
    {"date":"2025-01-20","units_sold":119,"unit_price":51.9},
    {"date":"2025-01-21","units_sold":120,"unit_price":52.0},
    {"date":"2025-01-22","units_sold":121,"unit_price":52.1},
    {"date":"2025-01-23","units_sold":122,"unit_price":52.2},
    {"date":"2025-01-24","units_sold":123,"unit_price":52.3},
    {"date":"2025-01-25","units_sold":124,"unit_price":52.4},
    {"date":"2025-01-26","units_sold":125,"unit_price":52.5},
    {"date":"2025-01-27","units_sold":126,"unit_price":52.6},
    {"date":"2025-01-28","units_sold":127,"unit_price":52.7},
    {"date":"2025-01-29","units_sold":128,"unit_price":52.8}
  ]
}
```

The history must contain 29 or more complete consecutive days. Response:

```json
{
  "success": true,
  "forecast_horizon_days": 7,
  "predicted_demand_units": 867.4861792747859,
  "current_stock": 500,
  "suggested_restock_quantity": 367.4861792747859,
  "model_version": "1.6.1-<model-sha-prefix>"
}
```

Restock quantity is calculated only as `max(0, predicted_demand_units - current_stock)`. Inventory and purchasing are not updated.

## Error codes

- `422 Unprocessable Entity`: Invalid date, chronological order, missing/duplicate dates, negative sales, invalid prices, feature mismatch, or unsupported horizon.
- `503 Service Unavailable`: The model is not loaded.
- `500 Internal Server Error`: Feature layout or model output is invalid/non-finite.

## GitHub and Vercel deployment

The repository root is the backend project root. The ASGI entry point is `app/main.py`, and `app` is the FastAPI application instance.

```powershell
git init
git add .
git commit -m "Prepare Ghayeb AI FastAPI for production deployment"
git branch -M main
git remote add origin https://github.com/ahmedqataria99/Ghayeb_ai.git
git push -u origin main
```

`vercel.json` routes all requests to `app/main.py`. The copied PKL at `models/material_demand_model.pkl` is loaded with a path derived from the source file, so it does not depend on the local working directory. The original notebook and raw datasets are excluded from deployment.

Before deploying, run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
npx vercel@latest --prod
```

Vercel authentication must be completed through the official CLI or dashboard when prompted. Do not commit credentials or tokens.

## Production requirements remaining

The local API still requires production infrastructure decisions that are outside this implementation: HTTPS/TLS, authentication and authorization, persistent request logging, rate limiting, secret management, monitored artifact storage, container/runtime configuration, health probes, and deployment secrets. No Vercel deployment or GitHub push has been performed.

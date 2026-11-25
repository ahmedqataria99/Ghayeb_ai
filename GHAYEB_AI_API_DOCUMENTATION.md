# Ghayeb AI API Documentation

**Base URL:** `https://ghayeb-plivdzhdb-ahmed-qataria.vercel.app`  
**API version:** `1.0.0`  
**Production deployment:** Vercel  
**Forecast horizon:** 7 days  
**Model:** HistGradientBoosting Regressor in a fitted scikit-learn Pipeline

## Authentication Status

Authentication is not implemented by the API. Ghayeb currently has no Firebase Authentication integration, and no API key or client-supplied `organizationId` is trusted for authorization.

The API is currently publicly reachable after Vercel deployment protection was disabled. This is suitable for local integration testing only. Secure server-verifiable authentication, authorization, organization scoping, API abuse controls, and secrets management remain required for production.

Do not embed a secret API key in the Flutter executable.

## Model Information

- Model name: `HistGradientBoostingRegressor`
- Model type: `sklearn.pipeline.Pipeline`
- Model version: `1.6.1-a43c16964824`
- scikit-learn: `1.6.1`
- Feature count: 14
- PKL SHA-256: `a43c16964824ae9ae1f575df092d1c049e7051edc9f7c547da1a4aede62aaefa`
- Evaluation metrics: Experimental notebook results only

Validation metrics: MAE `124.89`, RMSE `163.67`, WMAPE `18.92`. Test metrics: MAE `131.23`, RMSE `173.14`, WMAPE `18.19`. These values are not production SLAs.

## Feature Order

1. `sales_lag_1`
2. `sales_lag_7`
3. `sales_lag_14`
4. `sales_lag_28`
5. `sales_mean_7`
6. `sales_mean_14`
7. `sales_mean_28`
8. `sales_std_7`
9. `sales_std_14`
10. `sales_std_28`
11. `price_lag_1`
12. `price_mean_7`
13. `forecast_day_of_week`
14. `forecast_month`

## Common Headers

```http
Content-Type: application/json
Accept: application/json
```

## Date and History Rules

The API expects at least 29 ordered, consecutive daily history records.

- `forecast_date` must be exactly one day after the newest history date.
- Dates must use ISO `YYYY-MM-DD` format.
- Dates must be in chronological order.
- Duplicate dates are rejected.
- Missing calendar dates are rejected.
- Missing historical values are never fabricated.
- `units_sold` must be finite and at least `0`.
- `unit_price` must be finite and greater than `0`.
- The last supplied history date is `D`; the forecast covers `D+1` through `D+7`.

## Endpoints

### `GET /`

Basic service status.

Response `200`:

```json
{"name":"Ghayeb AI Material Demand API","status":"ready"}
```

### `GET /health`

Model loading status.

Response `200`:

```json
{"status":"healthy","model":"loaded"}
```

### `GET /v1/model-info`

Returns deployed model metadata and exact feature contract.

Response `200`:

```json
{
  "model_name":"HistGradientBoostingRegressor",
  "model_version":"1.6.1-a43c16964824",
  "model_type":"sklearn.pipeline.Pipeline",
  "forecast_horizon_days":7,
  "feature_count":14,
  "feature_names":["sales_lag_1","sales_lag_7","sales_lag_14","sales_lag_28","sales_mean_7","sales_mean_14","sales_mean_28","sales_std_7","sales_std_14","sales_std_28","price_lag_1","price_mean_7","forecast_day_of_week","forecast_month"],
  "feature_order":["sales_lag_1","sales_lag_7","sales_lag_14","sales_lag_28","sales_mean_7","sales_mean_14","sales_mean_28","sales_std_7","sales_std_14","sales_std_28","price_lag_1","price_mean_7","forecast_day_of_week","forecast_month"],
  "model_sha256":"a43c16964824ae9ae1f575df092d1c049e7051edc9f7c547da1a4aede62aaefa",
  "sklearn_version":"1.6.1",
  "evaluation_metrics":{"status":"experimental","validation":{"mae":124.89,"rmse":163.67,"wmape":18.92},"test":{"mae":131.23,"rmse":173.14,"wmape":18.19}}
}
```

### `POST /v1/materials/demand-forecast`

Request fields:

- `forecast_date`: date, required
- `forecast_horizon_days`: integer, must be `7`
- `current_stock`: non-negative finite number
- `history`: array of at least 29 `HistoricalRecord` objects

`HistoricalRecord`:

- `date`: ISO date
- `units_sold`: non-negative finite number; zero is allowed
- `unit_price`: finite number greater than zero

Request example:

```json
{
  "forecast_date":"2025-01-30",
  "forecast_horizon_days":7,
  "current_stock":500,
  "history":[
    {"date":"2025-01-01","units_sold":100,"unit_price":50.0},
    {"date":"2025-01-02","units_sold":101,"unit_price":50.1},
    {"date":"2025-01-29","units_sold":128,"unit_price":52.8}
  ]
}
```

The abbreviated example is illustrative. Real requests require all 29 consecutive dates.

Response `200`:

```json
{
  "success":true,
  "forecast_horizon_days":7,
  "predicted_demand_units":867.4861792747859,
  "current_stock":500.0,
  "suggested_restock_quantity":367.48617927478585,
  "model_version":"1.6.1-a43c16964824"
}
```

`suggested_restock_quantity = max(0, predicted_demand_units - current_stock)`.

### `POST /predict`

Legacy seven-day prediction endpoint.

Request:

```json
{
  "forecast_date":"2025-01-30",
  "history":[
    {"date":"2025-01-01","units_sold":100,"unit_price":50.0},
    {"date":"2025-01-29","units_sold":128,"unit_price":52.8}
  ]
}
```

A real request requires all 29 consecutive records.

Response `200`:

```json
{"predicted_demand_next_7_days":867.4861792747859}
```

### `POST /predict/engineered`

Accepts the exact 14 engineered features.

```json
{
  "features":{
    "sales_lag_1":100,
    "sales_lag_7":98,
    "sales_lag_14":101,
    "sales_lag_28":95,
    "sales_mean_7":99,
    "sales_mean_14":100,
    "sales_mean_28":101,
    "sales_std_7":4.2,
    "sales_std_14":5.1,
    "sales_std_28":6.3,
    "price_lag_1":50,
    "price_mean_7":49.8,
    "forecast_day_of_week":3,
    "forecast_month":1
  }
}
```

Response `200`:

```json
{"predicted_demand_next_7_days":867.4861792747859}
```

### `POST /predict/batch`

Request:

```json
{
  "requests":[
    {
      "forecast_date":"2025-01-30",
      "history":[
        {"date":"2025-01-01","units_sold":100,"unit_price":50.0},
        {"date":"2025-01-29","units_sold":128,"unit_price":52.8}
      ]
    }
  ]
}
```

Each item requires all 29 consecutive records. Maximum batch size: 100.

Response `200`:

```json
[{"predicted_demand_next_7_days":867.4861792747859}]
```

## cURL Examples

```bash
curl "https://ghayeb-plivdzhdb-ahmed-qataria.vercel.app/health"

curl "https://ghayeb-plivdzhdb-ahmed-qataria.vercel.app/v1/model-info"

curl -X POST "https://ghayeb-plivdzhdb-ahmed-qataria.vercel.app/v1/materials/demand-forecast" \
  -H "Content-Type: application/json" \
  -d '{"forecast_date":"2025-01-30","forecast_horizon_days":7,"current_stock":500,"history":[<29 consecutive HistoricalRecord objects>]}'
```

## Dart HTTP Example

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> forecast(Map<String, dynamic> body) async {
  final response = await http
      .post(
        Uri.parse('https://ghayeb-plivdzhdb-ahmed-qataria.vercel.app/v1/materials/demand-forecast'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(body),
      )
      .timeout(const Duration(seconds: 60));

  if (response.statusCode != 200) {
    throw Exception('Forecast failed: ${response.statusCode} ${response.body}');
  }
  return jsonDecode(response.body) as Map<String, dynamic>;
}
```

## Dart Dio Example

```dart
import 'package:dio/dio.dart';

final dio = Dio(BaseOptions(
  baseUrl: 'https://ghayeb-plivdzhdb-ahmed-qataria.vercel.app',
  connectTimeout: const Duration(seconds: 30),
  receiveTimeout: const Duration(seconds: 60),
));

Future<Map<String, dynamic>> forecast(Map<String, dynamic> body) async {
  final response = await dio.post('/v1/materials/demand-forecast', data: body);
  return Map<String, dynamic>.from(response.data);
}
```

## Retry and Timeout Guidance

- Set connect timeout to 30 seconds and receive timeout to 60 seconds.
- Retry only transport failures and HTTP `408`, `429`, or `5xx` responses.
- Do not blindly retry `422` validation responses.
- Use exponential backoff with jitter, for a maximum of 3 attempts.
- Keep the same request payload for a retry; do not mutate history or inventory.
- Requests can experience serverless cold-start latency.

## Insufficient History

If fewer than 29 records are available, do not fabricate or extrapolate missing sales. Return an application-level validation message and collect/restore the missing source records. Duplicate and missing dates must be repaired before retrying.

## Error Codes

- `422`: Invalid schema, missing/duplicate/gapped dates, negative sales, invalid price, unsupported horizon, or feature mismatch.
- `503`: Model is not loaded.
- `500`: Invalid feature layout or non-finite model output.
- `408/429/5xx`: Retry according to the timeout policy.

Example `422`:

```json
{"detail":[{"type":"literal_error","loc":["body","forecast_horizon_days"],"msg":"Input should be 7","input":14}]}
```

## Model Version Handling

Read `model_version` and `model_sha256` from `/v1/model-info`. Persist the forecast response's `model_version` alongside the forecast result if auditing or traceability is required. A model-version change should trigger application monitoring and compatibility review.

## Postman

Create an environment variable:

```text
base_url = https://ghayeb-plivdzhdb-ahmed-qataria.vercel.app
```

Set `GET {{base_url}}/health`, `GET {{base_url}}/v1/model-info`, and `POST {{base_url}}/v1/materials/demand-forecast` with the JSON body format documented above.

## Production Test Results

Tested on `2026-11-25`:

- `GET /health`: `200`, loaded model
- `GET /v1/model-info`: `200`, 14 features, seven-day horizon, experimental metrics
- `POST /v1/materials/demand-forecast`: `200`, finite prediction `867.4861792747859`, stock `500`, restock `367.48617927478585`
- `POST /predict/batch`: `200`, one result returned
- Unsupported horizon `14`: `422`
- 28-history records: `422`
- Negative sales: `422`

## Remaining Production Requirements

- Secure server-verifiable authentication and organization authorization
- Remove reliance on public unauthenticated access
- CORS restrictions matched to the Flutter/Desktop deployment model
- Rate limiting and abuse protection
- Persistent structured logging without sensitive data
- Monitoring, alerting, and uptime checks
- Secret and environment-variable management
- Serverless cold-start and request-size monitoring
- Model artifact integrity verification during startup/deployment

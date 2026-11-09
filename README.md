# FuelNet — Fuel Consumption & CO2 Emission Prediction Platform

Predict passenger-vehicle CO2 emissions (g/km) and derived fuel consumption (L/100 km) from a
CatBoost model trained on Canadian, US, EPA and Indian laboratory/road-cycle data.

The flagship deliverable is a working end-to-end stack: **trained model → FastAPI → React dashboard**.

> Status: complete and runnable locally. Automated tests, Docker packaging and documentation are included.

---

## Canonical model: CatBoost v3

Two model families live in this repository. The API and frontend serve **CatBoost v3** because it has the
best accuracy and the broadest training data.

| Property | CatBoost v3 (served) |
|---|---|
| Artifact | `catboost_v3_bundle/catboost.pt` (also `.cbm`) |
| Target | `co2_g_per_km`, trained on `log1p` and inverted with `expm1` |
| Features | 57 engineered features (year, engine, cylinders, transmission, target encodings, aggregates/flags) |
| Training data | India + Global + Canada (1995–2024) + EPA — 87,369 rows in the training distribution |
| Trees / depth | 24,998 iterations, depth 8 |
| Overall test R² | **0.887** (MAE 22.2, RMSE 44.9, MedAE 10.4) |
| Validation R² | 0.891 |

### Per-source test performance

| Source | n | R² | MAE | RMSE |
|---|---:|---:|---:|---:|
| Kaggle_Canada | 4,511 | 0.957 | 9.08 | 12.40 |
| Canada_1995_2014 | 3,570 | 0.956 | 10.16 | 13.89 |
| Canada_2015_2024 | 2,012 | 0.956 | 8.98 | 12.53 |
| EPA_FuelEconomy | 9,432 | 0.952 | 17.85 | 25.79 |
| Vehicle_Specs_2026 | 2,779 | 0.902 | 13.75 | 20.79 |
| India_Vehicle_Emission | 2,000 | **-0.193** | 119.11 | 141.12 |

**India predictions are experimental and flagged in the UI.** Canada/US figures are the reliable ones.

### Derived fuel values

The served model predicts **CO₂ only**. Fuel is derived with `fuel_combined = CO2 × 0.04310` (~2,320 g CO₂/L),
and the city/highway split uses median ratios measured from `data/Fuel_Consumption_2000-2022.csv`
(`city ≈ 1.157 × combined`, `highway ≈ 0.808 × combined`), with constants as fallback.

### Uncertainty

`/predict` returns residual-based standard deviations computed from the held-out
`catboost_v3_bundle/test_predictions.csv`: a per-source value when known, otherwise India / non-India / global.
Unseen vehicles are inflated by 10%.

---

## Legacy model: FuelNet (PyTorch)

The original 4-target residual MLP is still present for reference:

- `src/models/fuelnet.py`, `src/models/train.py`, `src/models/export.py`
- `models_artifacts/` (`fuelnet_best.pt`, `fuelnet_ts.pt`, `preprocessor.pkl`, `target_scaler.pkl`, `model_card.json`)
- Canada-only (2000–2022), test R² 0.88 (CO₂), 0.89 (combined fuel), 0.90 (city), 0.83 (highway)
- Legacy adapters: `src/api/predictor.py`, `src/api/main_fuelnet_backup.py`, `src/evaluation/explain.py`

These are **not wired into the running API**. They are kept as a documented alternative and for the SHAP
analysis (`models_artifacts/shap_summary.png`).

---

## Architecture

```
                 ┌──────────────────────────────┐
                 │  raw vehicle / trip inputs   │
                 └──────────────┬───────────────┘
                                │
                   ┌────────────▼────────────┐
                   │ src/api/schemas.py      │  Pydantic validation
                   └────────────┬────────────┘
                                │
             ┌──────────────────▼──────────────────┐
             │ src/api/catboost_predictor.py       │
             │ 57-feature engineering, TE lookups, │
             │ median imputation, SHAP, uncertainty│
             └──────────────────┬──────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
   ┌──────▼──────┐      ┌───────▼───────┐     ┌───────▼───────┐
   │ /predict    │      │ /batch-predict│     │ /explain      │
   │ CO2 + fuel  │      │  ≤ 500 rows   │     │ CatBoost SHAP │
   └──────┬──────┘      └───────┬───────┘     └───────┬───────┘
          └─────────────────────┼─────────────────────┘
                                │  FastAPI (src/api/main.py)
                   ┌────────────▼────────────┐
                   │ frontend/ React + Vite  │
                   │ Single / Compare / Batch│
                   │ / Road-check simulator  │
                   └─────────────────────────┘
```

---

## Project structure

```
fuelnet/
├── data/                      # Fuel_Consumption_2000-2022.csv (gitignored)
├── notebooks/                 # 01_eda.ipynb (+ executed copy)
├── src/
│   ├── data/                  # preprocess.py, dataset.py (FuelNet pipeline)
│   ├── models/                # fuelnet.py, train.py, baselines.py, export.py (legacy)
│   ├── api/                   # main.py, schemas.py, catboost_predictor.py (+ legacy predictor.py)
│   └── evaluation/            # explain.py (FuelNet SHAP, legacy)
├── frontend/                  # React 19 + Vite 8 + TypeScript + recharts
│   ├── nginx.conf             # production proxy to the API
│   └── Dockerfile
├── catboost_v3_bundle/        # catboost.pt/.cbm, metadata, TE mappings, metrics, test predictions
├── prepared_dataset/prepared_data/  # feature_cols.json, feature_median.pkl, y_train_orig.npy, ...
├── models_artifacts/          # legacy FuelNet artifacts (gitignored)
├── configs/                   # train.yaml, model.yaml
├── tests/                     # pytest suite (schemas, predictor, API)
├── scripts/                   # run_local.bat, run_tests.bat
├── Dockerfile                 # API image
├── docker-compose.yml         # api + web
├── requirements.txt           # full dev/training environment
├── requirements-api.txt       # slim serving environment
└── pytest.ini
```

---

## Setup

### Prerequisites
- Python 3.11+ (tested on 3.13)
- Node.js 18+ (tested on 24)
- Optional: Docker Desktop / Docker Engine + Compose

### 1. Python environment

```bat
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Environment variables

```bat
copy .env.example .env
```

| Variable | Default | Meaning |
|---|---|---|
| `FUEL_PRICE` | `1.50` | USD/L used for annual cost math |
| `CATBOOST_BUNDLE_DIR` | `catboost_v3_bundle` | model/artifact directory |
| `PREPARED_DATA_DIR` | `prepared_dataset/prepared_data` | medians + feature list |
| `EXPLAIN_SHAP_TREES` | `3000` | trees used for SHAP; `0` = exact over all 24,999 (slow) |

### 3. Frontend

```bat
cd frontend
npm install
```

---

## Run locally

One command (opens two terminals):

```bat
scripts\run_local.bat
```

Manual:

```bat
:: terminal 1 — API
venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

:: terminal 2 — web UI
cd frontend
npm run dev
```

- API docs: http://127.0.0.1:8000/docs
- Web UI: http://127.0.0.1:5173
- Vite proxies `/predict`, `/batch-predict`, `/explain`, `/options`, `/model`, `/health` to port 8000.

---

## Tests

```bat
venv\Scripts\python.exe -m pytest -q
:: or
scripts\run_tests.bat
```

Coverage: request schema validation, 57-feature preprocessing + median fill, unseen-vehicle flag,
prediction sanity, fuel-split ratios, uncertainty, batch consistency, SHAP output, and every API endpoint
(including 422 cases). Current result: **29 passed**.

---

## Docker

```bat
docker compose up --build
```

- API: http://127.0.0.1:8000
- Web: http://127.0.0.1:8080 (nginx proxies API calls to the `api` service)

The API image uses the slim `requirements-api.txt`; the full `requirements.txt` is for training/notebooks.
`catboost.cbm` is ~100 MB, so keep `.dockerignore` in place to avoid bloating the build context.

> Docker was not installed on the development machine used to build this project, so the images were
> authored and configured but not built here. Compose file and Dockerfiles are ready to run where Docker exists.

---

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | health check |
| GET | `/model/info` | model card, metrics, per-source table, limitations |
| GET | `/options` | makes, models, classes, fuel types, transmissions, regions |
| POST | `/predict` | single vehicle prediction |
| POST | `/batch-predict` | batch prediction (≤ 500; empty rejected) |
| POST | `/explain` | CatBoost TreeSHAP contributions + base value |

### Example request

```json
{
  "year": 2020,
  "make": "Toyota",
  "model": "Corolla",
  "vehicle_class": "Compact",
  "engine_size_l": 1.8,
  "cylinders": 4,
  "transmission": "A6",
  "fuel_type": "X",
  "region": "Canada"
}
```

### Example response (abridged)

```json
{
  "fuel_city_l_100km": 10.51,
  "fuel_hwy_l_100km": 7.34,
  "fuel_comb_l_100km": 9.09,
  "fuel_comb_mpg": 25.9,
  "co2_g_km": 210.8,
  "co2_g_per_km": 210.8,
  "uncertainty": { "fuel_comb_std": 0.87, "co2_std": 20.24, "...": "..." },
  "annual_cost_usd": 2043.2,
  "eco_score": 78,
  "unseen_vehicle": false,
  "model_version": "catboost-v3-24998",
  "inference_ms": 50.6,
  "note": ""
}
```

`/explain` returns the top-15 contributions (positive = raises CO₂), a `base_value` and an `approximate`
flag. By default SHAP runs over the first 3,000 trees (~3–5 s); set `EXPLAIN_SHAP_TREES=0` for exact SHAP
over all trees (~20 s).

---

## Dataset

- Source: Kaggle `ahmettyilmazz/fuel-consumption` — Environment and Climate Change Canada / NRCan
- Years 2000–2022, Canadian laboratory cycle; the CatBoost bundle additionally uses EPA, India and global data
- FuelNet targets: `fuel_city_l_100km`, `fuel_hwy_l_100km`, `fuel_comb_l_100km`, `co2_g_km`

Place the Canadian CSV at `data/Fuel_Consumption_2000-2022.csv` to enable measured fuel-split ratios
(optional; constants are used otherwise).

---

## Known limitations

- CO₂ is the only directly modeled target; fuel figures are derived and should be treated as estimates.
- India accuracy is poor (R² −0.19) and clearly flagged; use Canada/US results for decisions.
- Laboratory-cycle predictions, not real-world driving.
- Unseen makes/models fall back to smoothed global target encodings and are flagged.
- Explanations use a tree-limited SHAP approximation by default (see `EXPLAIN_SHAP_TREES`).

## License

MIT

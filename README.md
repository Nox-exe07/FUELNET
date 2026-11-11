# FuelNet — Fuel Consumption & CO2 Emission Prediction Platform

[![CI](https://github.com/Nox-exe07/FUELNET/actions/workflows/ci.yml/badge.svg)](https://github.com/Nox-exe07/FUELNET/actions/workflows/ci.yml)
[![Model release](https://img.shields.io/github/v/release/Nox-exe07/FUELNET?label=model)](https://github.com/Nox-exe07/FUELNET/releases/tag/v1.0-demo)

Predict passenger-vehicle CO2 emissions (g/km) and derived fuel consumption (L/100 km) from a
CatBoost model trained on Canadian, US, EPA and Indian laboratory/road-cycle data.

The deliverable is a working end-to-end stack: **trained model → FastAPI → React dashboard**.

> The repository is code-only. The 100.8 MiB CatBoost model is a GitHub Release asset and is
> downloaded automatically on first run. That keeps `git clone` small while making the app fully
> runnable from a fresh checkout.

---

## Quick start (Windows)

```bat
scripts\run_demo.bat
```

This will:

1. create `venv` if needed,
2. install the slim serving dependencies,
3. download and verify `catboost.cbm` if it is missing,
4. start the API and the prebuilt React UI on **http://127.0.0.1:8000**.

The browser opens automatically. Close the `FuelNet Server` window to stop it.

### Manual start

```bat
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements-api.txt
venv\Scripts\python.exe scripts\fetch_model.py
venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 for the dashboard and http://127.0.0.1:8000/docs for the API docs.

On Linux/macOS:

```bash
scripts/run_demo.sh
```

Or manually:

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements-api.txt
venv/bin/python scripts/fetch_model.py
venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

---

## Model download

The model is not stored in normal git history because GitHub rejects files above 100 MiB.

| Item | Value |
|---|---|
| Release | `v1.0-demo` |
| Asset | `catboost.cbm` |
| Size | 100.8 MiB (105,714,880 bytes) |
| SHA-256 | `b9f91f3879a7d087ef4ae516e5e9c60a5fcef11a59f5233c38ef97a5cb055fa6` |
| URL | https://github.com/Nox-exe07/FUELNET/releases/download/v1.0-demo/catboost.cbm |

`scripts/fetch_model.py` downloads to `catboost_v3_bundle/catboost.cbm` and verifies the checksum.
If the download is unavailable, download the asset manually from the release page and place it at that
path.

---

## Canonical model: CatBoost v3

| Property | CatBoost v3 (served) |
|---|---|
| Artifact | `catboost_v3_bundle/catboost.cbm` |
| Target | `co2_g_per_km`, trained on `log1p` and inverted with `expm1` |
| Features | 57 engineered features (year, engine, cylinders, transmission, target encodings, aggregates/flags) |
| Training data | India + Global + Canada (1995–2024) + EPA — 87,369 training rows |
| Trees / depth | 24,999 stored trees (`best_iteration` 24,998), depth 8 |
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

The served model predicts **CO₂ only**. Fuel is derived with `fuel_combined = CO2 × 0.04310`
(~2,320 g CO₂/L), and the city/highway split uses median ratios measured from
`data/Fuel_Consumption_2000-2022.csv` (`city ≈ 1.157 × combined`, `highway ≈ 0.808 × combined`),
with constants as fallback.

### Uncertainty

`/predict` returns residual-based standard deviations computed from the held-out
`catboost_v3_bundle/test_predictions.csv`: a per-source value when known, otherwise India / non-India /
global. Unseen vehicles are inflated by 10%.

---

## Architecture

```text
raw vehicle / trip inputs
        │
        ▼
src/api/schemas.py          Pydantic validation
        │
        ▼
src/api/catboost_predictor.py
57-feature engineering, target encodings,
median imputation, SHAP, uncertainty
        │
        ├── /predict          CO2 + derived fuel
        ├── /batch-predict    ≤ 500 rows
        └── /explain          CatBoost TreeSHAP
        │
        ▼
FastAPI (src/api/main.py) ── serves frontend/dist (React 19 + Vite)
```

---

## Project structure

```text
fuelnet/
├── data/                      # Fuel_Consumption_2000-2022.csv (tracked)
├── notebooks/                 # EDA notebooks
├── src/
│   ├── data/                  # preprocess.py, dataset.py
│   ├── models/                # FuelNet legacy PyTorch model/training code
│   ├── api/                   # main.py, schemas.py, catboost_predictor.py
│   └── evaluation/            # legacy explain.py
├── frontend/                  # React 19 + Vite 8 + TypeScript + Recharts
│   ├── dist/                  # prebuilt static UI (committed for one-click demo)
│   ├── nginx.conf             # optional two-service Docker proxy
│   └── Dockerfile
├── catboost_v3_bundle/        # small artifacts; catboost.cbm downloaded separately
├── prepared_dataset/prepared_data/  # feature_cols.json, feature_median.pkl, y_train_orig.npy
├── models_artifacts/          # legacy FuelNet artifacts (not tracked)
├── configs/                   # train.yaml, model.yaml
├── tests/                     # pytest suite
├── scripts/
│   ├── fetch_model.py         # download + SHA-256 verification
│   ├── run_demo.bat           # one-click Windows demo
│   ├── run_demo.sh            # one-click Linux/macOS demo
│   ├── run_local.bat          # two-terminal development mode
│   └── run_tests.bat
├── Dockerfile                 # single-container API + built UI
├── docker-compose.yml         # one service on :8000
├── requirements.txt           # full dev/training environment
├── requirements-api.txt       # slim serving environment
└── pytest.ini
```

### What is committed vs downloaded

| Committed | Downloaded at first run |
|---|---|
| `metadata.json`, `te_mappings.json`, metrics CSVs | `catboost_v3_bundle/catboost.cbm` |
| `feature_cols.json`, `feature_median.pkl`, `y_train_orig.npy` | |
| `test_predictions.csv`, `per_source.csv` | |
| `data/Fuel_Consumption_2000-2022.csv` | |
| `frontend/dist/` | |

Large training arrays (`X_train_*.pkl`, `X_test_raw.pkl`, `full_dataset.pkl`) are not needed for serving
and are not shipped.

---

## Environment variables

Copy `.env.example` to `.env` if you want to override defaults. Do not commit `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `FUEL_PRICE` | `1.50` | USD/L used for annual cost math |
| `CATBOOST_BUNDLE_DIR` | `catboost_v3_bundle` | model/artifact directory |
| `PREPARED_DATA_DIR` | `prepared_dataset/prepared_data` | medians + feature list |
| `FRONTEND_DIST` | `frontend/dist` | built React UI served by FastAPI |
| `EXPLAIN_SHAP_TREES` | `3000` | trees used for SHAP; `0` = exact over all 24,999 (slow) |
| `MODEL_URL` | release asset URL | override the model download URL |
| `MODEL_SHA256` | hash in `model.sha256` | expected model checksum |

---

## Development mode

For live frontend editing, use the two-terminal launcher:

```bat
scripts\run_local.bat
```

- API: http://127.0.0.1:8000
- Vite dev UI: http://127.0.0.1:5173
- Vite proxies `/predict`, `/batch-predict`, `/explain`, `/options`, `/model`, `/health` to port 8000.

To rebuild the committed UI:

```bat
cd frontend
npm install
npm run build
```

---

## Tests

```bat
venv\Scripts\python.exe -m pip install -r requirements-test.txt
venv\Scripts\python.exe scripts\fetch_model.py
venv\Scripts\python.exe -m pytest -q
```

or:

```bat
scripts\run_tests.bat
```

Coverage: request schema validation, 57-feature preprocessing + median fill, unseen-vehicle flag,
prediction sanity, fuel-split ratios, uncertainty, batch consistency, SHAP output, and every API endpoint
(including 422 cases). Current result: **29 passed**.

---

## Docker

The root `Dockerfile` is a single container: it builds the React UI, installs the slim API dependencies,
downloads the model from the GitHub Release, and serves everything on port 8000.

```bat
docker compose up --build
```

Open http://127.0.0.1:8000.

To use a different model mirror:

```bat
set MODEL_URL=https://example.com/catboost.cbm
docker compose up --build
```

---

## Public demo

Live deployment: **https://nagavarundaddanala--fuelnet-web.us-east.modal.run**

The Modal deployment is defined in
[`deploy/modal/modal_app.py`](deploy/modal/modal_app.py). It builds the API
image, downloads the verified model from the GitHub Release, and serves the
prebuilt frontend and FastAPI endpoints.

```bat
modal deploy deploy/modal/modal_app.py
```

Hugging Face Docker Spaces require a PRO subscription on the free `cpu-basic`
hardware, so the prepared `deploy/hf-space` files are kept only as an
alternative for accounts with PRO.

For a temporary public URL from a local machine, a tunnel can be used:

```bat
cloudflared tunnel --url http://127.0.0.1:8000
```

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

The Canadian CSV at `data/Fuel_Consumption_2000-2022.csv` enables measured fuel-split ratios.

---

## Known limitations

- CO₂ is the only directly modeled target; fuel figures are derived and should be treated as estimates.
- India accuracy is poor (R² −0.19) and clearly flagged; use Canada/US results for decisions.
- Laboratory-cycle predictions, not real-world driving.
- Unseen makes/models fall back to smoothed global target encodings and are flagged.
- Explanations use a tree-limited SHAP approximation by default (see `EXPLAIN_SHAP_TREES`).

## Legacy model: FuelNet (PyTorch)

The original 4-target residual MLP is still present for reference in `src/models/` and
`models_artifacts/` (not tracked). It is **not wired into the running API**. It remains a documented
alternative and the source of the legacy SHAP analysis.

## License

MIT

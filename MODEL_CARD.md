# FuelNet CatBoost v3 — Model Card

## Model details

| Item | Value |
|---|---|
| Model | CatBoostRegressor |
| Version | `catboost-v3-24998` |
| Target | `co2_g_per_km`, trained on `log1p(CO2)` and inverted with `expm1` |
| Weights | `catboost_v3_bundle/catboost.cbm` (100.8 MiB release asset) |
| Trees | 24,999 stored (`best_iteration` 24,998) |
| Depth | 8 |
| Learning rate | 0.02 |
| Loss | RMSE |
| L2 regularization | 3 |
| Random seed | 42 |
| Grow policy | SymmetricTree |
| Border count | 128 |
| Features | 57 engineered numeric features |

## Intended use

Estimate passenger-vehicle tailpipe CO2 and derived fuel consumption for dashboard exploration,
comparison, batch scoring, and classroom demonstrations.

## Training data

The bundle combines Canadian, US EPA, Indian, and global vehicle records. The metadata dataset
summary reports 128,485 total rows: 87,369 train, 15,419 validation, and 25,697 test rows.

## Metrics (reported validation/test split)

| Metric | Value |
|---|---|
| Test R² | 0.887 |
| Test MAE | 22.2 g/km |
| Test RMSE | 44.9 g/km |
| Test MedAE | 10.4 g/km |
| Test MAPE | 8.87% |
| Validation R² | 0.891 |

### Per-source test performance

| Source | R² | MAE | RMSE |
|---|---:|---:|---:|
| Kaggle_Canada | 0.957 | 9.08 | 12.40 |
| Canada_1995_2014 | 0.956 | 10.16 | 13.89 |
| Canada_2015_2024 | 0.956 | 8.98 | 12.53 |
| EPA_FuelEconomy | 0.952 | 17.85 | 25.79 |
| Vehicle_Specs_2026 | 0.902 | 13.75 | 20.79 |
| India_Vehicle_Emission | -0.193 | 119.11 | 141.12 |

## Limitations

- CO2 is the only directly modeled target. Fuel values are derived using `CO2 × 0.04310`
  (approximately 2,320 g CO2/L); city/highway splits are estimated from Canadian median ratios.
- India performance is poor and flagged in the API/UI. Use Canada and US results for decisions.
- Predictions describe laboratory/road-cycle data, not real-world driving.
- Unseen makes/models fall back to smoothed global target encodings and are flagged.
- SHAP explanations use the first 3,000 trees by default for speed; set `EXPLAIN_SHAP_TREES=0`
  for exact SHAP over all trees.

## Distribution

The model is distributed as the `catboost.cbm` asset on the GitHub Release `v1.0-demo`:

https://github.com/Nox-exe07/FUELNET/releases/download/v1.0-demo/catboost.cbm

SHA-256: `b9f91f3879a7d087ef4ae516e5e9c60a5fcef11a59f5233c38ef97a5cb055fa6`

## License

MIT

"""
FuelNet Baselines — Phase 3
- Linear Regression (multi-output)
- XGBoost per target
- Evaluated on 2021-2022 temporal test
"""
import pathlib
import pickle
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "models_artifacts"
PROCESSED_PATH = ARTIFACTS_DIR / "processed_data.pkl"
METRICS_PATH = ARTIFACTS_DIR / "baseline_metrics.json"
XGB_PATH = ARTIFACTS_DIR / "xgb_baseline.pkl"

TARGET_NAMES = ['fuel_city_l_100km', 'fuel_hwy_l_100km', 'fuel_comb_l_100km', 'co2_g_km']

def load_data():
    with open(PROCESSED_PATH, 'rb') as f:
        data = pickle.load(f)
    train = data['train_pool']
    test = data['test']
    # Feature matrix: numeric_scaled (5) + ohe (39) + make_idx + model_idx = 46
    # Use processed arrays directly — this uses the same preprocessing as Phase 2 (fit only on 2000-2020)
    X_train = np.concatenate([
        train['numeric_scaled'],
        train['ohe'],
        train['make_idx'].reshape(-1, 1),
        train['model_idx'].reshape(-1, 1)
    ], axis=1)
    X_test = np.concatenate([
        test['numeric_scaled'],
        test['ohe'],
        test['make_idx'].reshape(-1, 1),
        test['model_idx'].reshape(-1, 1)
    ], axis=1)
    y_train = train['targets']  # (20660, 4)
    y_test = test['targets']    # (1896, 4)
    return X_train, y_train, X_test, y_test, data

def compute_metrics(y_true, y_pred):
    """Returns dict with r2, mae, mape, rmse per target."""
    metrics = {}
    for i, name in enumerate(TARGET_NAMES):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        mape = mean_absolute_percentage_error(yt, yp) * 100  # percent
        rmse = np.sqrt(mean_squared_error(yt, yp))
        metrics[name] = {
            "r2": float(r2),
            "mae": float(mae),
            "mape": float(mape),
            "rmse": float(rmse)
        }
    return metrics

def print_comparison(metrics_lr, metrics_xgb):
    print("\n" + "="*110)
    print(f"{'Target':<22} | {'Model':<14} | {'R2':>7} | {'MAE':>8} | {'MAPE%':>8} | {'RMSE':>8}")
    print("-"*110)
    for name in TARGET_NAMES:
        for model_name, metrics in [("LinearReg", metrics_lr), ("XGBoost", metrics_xgb)]:
            m = metrics[name]
            print(f"{name:<22} | {model_name:<14} | {m['r2']:7.4f} | {m['mae']:8.3f} | {m['mape']:8.2f} | {m['rmse']:8.3f}")
        print("-"*110)

def main():
    print("=== FuelNet Baselines — Training ===")
    print(f"Loading processed data from {PROCESSED_PATH}")
    X_train, y_train, X_test, y_test, data = load_data()
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test  shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Train pool (2000-2020) rows: {X_train.shape[0]}, Test (2021-2022) rows: {X_test.shape[0]}")
    print(f"Features: 5 numeric_scaled + 39 ohe + 2 make/model = {X_train.shape[1]}")

    # Also show GroupKFold info (from Phase 2)
    if 'splits' in data:
        print(f"GroupKFold splits (from Phase 2): {len(data['splits'])} splits, each ~{len(data['splits'][0][0])} train / {len(data['splits'][0][1])} val (grouped by make+model)")

    # 1. Linear Regression (multi-output)
    print("\n--- Training Linear Regression (multi-output) ---")
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    metrics_lr = compute_metrics(y_test, y_pred_lr)
    print("Linear Regression metrics computed.")

    # 2. XGBoost per target
    print("\n--- Training XGBoost (one model per target) ---")
    xgb_models = {}
    y_pred_xgb = np.zeros_like(y_test, dtype=float)
    for i, name in enumerate(TARGET_NAMES):
        print(f"  Training XGB for {name} ({i+1}/4)...")
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            tree_method='hist'
        )
        model.fit(X_train, y_train[:, i])
        xgb_models[name] = model
        y_pred_xgb[:, i] = model.predict(X_test)
        print(f"    Done. Train R2: {model.score(X_train, y_train[:, i]):.4f}")

    metrics_xgb = compute_metrics(y_test, y_pred_xgb)
    print("XGBoost metrics computed.")

    # Save metrics
    all_metrics = {
        "linear_regression": metrics_lr,
        "xgboost": metrics_xgb
    }
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved baseline metrics to {METRICS_PATH}")

    # Save XGB models
    with open(XGB_PATH, 'wb') as f:
        pickle.dump(xgb_models, f)
    print(f"Saved XGBoost models to {XGB_PATH}")

    # Print comparison table
    print_comparison(metrics_lr, metrics_xgb)

    # Simple explanation helper
    print("\n=== Metrics Explanation ===")
    print("R2 (R-squared): 0 to 1, 1 = perfect. How much variance the model explains. Higher is better.")
    print("MAE (Mean Absolute Error): Average absolute difference between predicted and true. Lower is better. In same units as target (L/100km or g/km).")
    print("MAPE (Mean Absolute Percentage Error): Average % error. Lower is better. 5% means on average 5% off.")
    print("RMSE (Root Mean Squared Error): Like MAE but penalizes large errors more. Lower is better.")

    # Quick verdict
    print("\n=== Verdict ===")
    for name in TARGET_NAMES:
        lr_r2 = metrics_lr[name]['r2']
        xgb_r2 = metrics_xgb[name]['r2']
        winner = "XGBoost" if xgb_r2 > lr_r2 else "LinearReg"
        diff = abs(xgb_r2 - lr_r2)
        print(f"{name}: XGB R2 {xgb_r2:.4f} vs LR R2 {lr_r2:.4f} -> winner {winner} by {diff:.4f}")

    return all_metrics

if __name__ == "__main__":
    main()

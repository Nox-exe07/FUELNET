"""
FuelNet Explainability — Phase 5
- Loads FuelNet best checkpoint + test data (2021-2022)
- SHAP PermutationExplainer for CO2 (beeswarm)
- explain_single() per-car contributions
"""
import pathlib
import pickle
import json
import re
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "models_artifacts"
PROCESSED_PATH = ARTIFACTS_DIR / "processed_data.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
TARGET_SCALER_PATH = ARTIFACTS_DIR / "target_scaler.pkl"
BEST_CHECKPOINT_PATH = ARTIFACTS_DIR / "fuelnet_best.pt"
CATEGORY_MAPPINGS_PATH = ARTIFACTS_DIR / "category_mappings.json"
SHAP_PLOT_PATH = ARTIFACTS_DIR / "shap_summary.png"

from src.models.fuelnet import FuelNet

# Global cache for explainer
_explainer = None
_background = None
_feature_names = None
_model = None
_target_scaler = None
_device = None

def load_model_and_data():
    global _model, _target_scaler, _device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _device = device
    # Load mappings
    with open(CATEGORY_MAPPINGS_PATH, 'r') as f:
        mappings = json.load(f)
    make_vocab = mappings['make_vocab_size']
    model_vocab = mappings['model_vocab_size']
    # Load preprocessor for feature names
    with open(PREPROCESSOR_PATH, 'rb') as f:
        prep = pickle.load(f)
    # Load target scaler
    with open(TARGET_SCALER_PATH, 'rb') as f:
        target_scaler = pickle.load(f)
    _target_scaler = target_scaler

    # Load processed test data
    with open(PROCESSED_PATH, 'rb') as f:
        data = pickle.load(f)
    test = data['test']

    # Build FuelNet and load checkpoint
    numeric_dim = test['numeric_scaled'].shape[1] + test['ohe'].shape[1]  # 44
    model = FuelNet(make_vocab_size=make_vocab, model_vocab_size=model_vocab, numeric_dim=numeric_dim)
    state = torch.load(BEST_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    _model = model

    # Feature names: numeric_cols (5) + ohe (39) + make/model idx (2) = 46
    numeric_cols = prep['numeric_cols']  # 5
    ohe_names = prep['ohe_feature_names']  # 39
    feature_names = numeric_cols + ohe_names + ['make_idx', 'model_idx']
    # For SHAP we will use 46-dim array: [numeric_scaled (5) + ohe (39) + make_idx + model_idx]
    # Prepare test arrays
    X_numeric = np.concatenate([test['numeric_scaled'], test['ohe']], axis=1)  # (1896,44)
    X_full = np.concatenate([X_numeric, test['make_idx'].reshape(-1,1), test['model_idx'].reshape(-1,1)], axis=1)  # (1896,46)
    y_test = test['targets']  # raw (1896,4)
    return X_full, y_test, feature_names, model, target_scaler, device

def predict_co2_wrapper(X_batch):
    """
    Model-agnostic wrapper for SHAP: takes numpy array (N, 46) and returns CO2 predictions (N,) in raw g/km.
    X_batch columns: [numeric_scaled(5) + ohe(39) + make_idx + model_idx]
    """
    global _model, _target_scaler, _device
    if _model is None:
        load_model_and_data()
    X_batch = np.asarray(X_batch)
    if X_batch.ndim == 1:
        X_batch = X_batch.reshape(1, -1)
    # Split
    numeric = X_batch[:, :44]  # first 44 are numeric_scaled+ohe
    make_idx = X_batch[:, 44].astype(int)
    model_idx = X_batch[:, 45].astype(int)

    # Clip indices to valid range (in case permutation creates out-of-range)
    with open(CATEGORY_MAPPINGS_PATH, 'r') as f:
        m = json.load(f)
    make_vocab = m['make_vocab_size']
    model_vocab = m['model_vocab_size']
    make_idx = np.clip(make_idx, 0, make_vocab-1)
    model_idx = np.clip(model_idx, 0, model_vocab-1)

    numeric_t = torch.from_numpy(numeric.astype(np.float32)).to(_device)
    make_t = torch.from_numpy(make_idx.astype(np.int64)).to(_device)
    model_t = torch.from_numpy(model_idx.astype(np.int64)).to(_device)

    _model.eval()
    with torch.no_grad():
        pred_scaled = _model(numeric_t, make_t, model_t)  # (N,4) scaled
        pred_scaled_np = pred_scaled.cpu().numpy()
        # inverse transform to raw
        pred_raw = _target_scaler.inverse_transform(pred_scaled_np)  # (N,4)
        co2 = pred_raw[:, 3]  # CO2 is 4th target
    return co2

def compute_shap(sample_size=50, max_evals=500):
    """
    Compute SHAP values for CO2 using PermutationExplainer on sample of test rows.
    Returns shap_values, X_sample, feature_names
    """
    global _explainer, _background, _feature_names
    X_full, y_test, feature_names, model, target_scaler, device = load_model_and_data()
    _feature_names = feature_names

    # Sample
    n = X_full.shape[0]
    sample_size = min(sample_size, n)
    # Use 50 background samples for speed, and 50 explanation samples
    # Background should be representative (e.g., random 50 from test)
    np.random.seed(42)
    bg_idx = np.random.choice(n, size=min(50, n), replace=False)
    background = X_full[bg_idx]

    # Sample to explain
    exp_idx = np.random.choice(n, size=sample_size, replace=False)
    X_sample = X_full[exp_idx]

    print(f"Computing SHAP with background {background.shape}, sample {X_sample.shape}, max_evals={max_evals}")
    # PermutationExplainer
    # For newer shap, use shap.Explainer with masker. We'll use shap.PermutationExplainer if available
    try:
        explainer = shap.PermutationExplainer(predict_co2_wrapper, background, max_evals=max_evals)
        shap_values = explainer(X_sample)
        # shap_values is Explanation object with .values shape (N, F)
        values = shap_values.values
    except Exception as e:
        print(f"PermutationExplainer failed ({e}), trying shap.Explainer with Permutation masker")
        # Fallback to shap.Explainer
        masker = shap.maskers.Independent(background)
        explainer = shap.Explainer(predict_co2_wrapper, masker)
        # limit to fewer evaluations via max_evals? shap.Explainer doesn't have max_evals param for Permutation
        # Use shap.Permutation via shap.explainers.Permutation
        values = explainer(X_sample).values if hasattr(explainer, '__call__') else explainer(X_sample)

    _explainer = explainer
    _background = background

    # Save summary plot (beeswarm)
    try:
        plt.figure(figsize=(10,6))
        # Use shap.summary_plot with show=False
        # shap.summary_plot expects shap_values and features
        # We need to handle shape (N,F)
        shap.summary_plot(values, X_sample, feature_names=feature_names, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(SHAP_PLOT_PATH, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP summary plot to {SHAP_PLOT_PATH}")
    except Exception as e:
        print(f"Failed to create SHAP summary plot: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: bar plot of mean abs shap
        try:
            mean_abs = np.abs(values).mean(axis=0)
            order = np.argsort(mean_abs)[::-1][:15]
            plt.figure(figsize=(10,6))
            plt.barh([feature_names[i] for i in order[::-1]], mean_abs[order[::-1]], color='skyblue')
            plt.xlabel("mean |SHAP value| (g/km)")
            plt.title("SHAP feature importance for CO2 (mean |SHAP|)")
            plt.tight_layout()
            plt.savefig(SHAP_PLOT_PATH, dpi=150)
            plt.close()
            print(f"Saved fallback SHAP bar plot to {SHAP_PLOT_PATH}")
        except Exception as e2:
            print(f"Fallback also failed: {e2}")

    # Print top 5 features by mean abs SHAP
    mean_abs = np.abs(values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:5]
    print("\nTop 5 features by mean |SHAP| for CO2:")
    for i in top_idx:
        print(f"  {feature_names[i]:<30} mean|SHAP| {mean_abs[i]:.3f} g/km")

    return values, X_sample, feature_names

def explain_single(car_features_dict):
    """
    Returns per-feature SHAP contributions for one car as list of {"feature": name, "contribution": value} sorted by abs.
    car_features_dict: dict with keys like year, make, model, vehicle_class, engine_size_l, cylinders, transmission, fuel_type
    Example: {"year": 2020, "make": "toyota", "model": "corolla", "vehicle_class": "Compact", "engine_size_l": 1.8, "cylinders": 4, "transmission": "A6", "fuel_type": "X"}
    """
    global _explainer, _background, _feature_names, _model, _target_scaler
    if _explainer is None:
        compute_shap(sample_size=50, max_evals=500)

    # Need to convert car_features_dict to feature vector of 46 dims
    # Use preprocessing pipeline from src.data.preprocess + dataset
    # Simplified: load preprocessor and manually transform
    with open(PREPROCESSOR_PATH, 'rb') as f:
        prep = pickle.load(f)
    with open(CATEGORY_MAPPINGS_PATH, 'r') as f:
        mappings = json.load(f)

    # Cleaning helpers (same as preprocess)
    def clean_string(s):
        if pd.isna(s):
            return s
        s = str(s).lower().strip()
        s = re.sub(r'[^a-z0-9\s]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    # Build a temporary DataFrame with one row
    # Expected raw fields: year, make, model, vehicle_class, engine_size_l, cylinders, transmission, fuel_type
    # Fill missing with defaults from preprocessor modes/medians
    row = {}
    # Use median/mode defaults where not provided
    numeric_medians = prep['numeric_medians']
    categorical_modes = prep['categorical_modes']
    # year default
    row['year'] = car_features_dict.get('year', categorical_modes.get('year', 2015) if 'year' in categorical_modes else 2015)
    row['make'] = clean_string(car_features_dict.get('make', categorical_modes.get('make', 'unknown')))
    row['model'] = clean_string(car_features_dict.get('model', categorical_modes.get('model', 'unknown')))
    row['vehicle_class'] = str(car_features_dict.get('vehicle_class', categorical_modes.get('vehicle_class', 'Compact'))).strip()
    row['engine_size_l'] = float(car_features_dict.get('engine_size_l', numeric_medians.get('engine_size_l', 3.0)))
    row['cylinders'] = int(car_features_dict.get('cylinders', numeric_medians.get('cylinders', 4)))
    row['transmission'] = str(car_features_dict.get('transmission', 'A6')).strip()
    row['fuel_type'] = str(car_features_dict.get('fuel_type', categorical_modes.get('fuel_type', 'X'))).strip()

    # Need to derive transmission_type, gears, power_proxy
    trans = row['transmission']
    if trans.startswith('M'):
        t_type = 'manual'
    else:
        t_type = 'automatic'
    m = re.search(r'(\d+)', trans)
    gears = int(m.group(1)) if m else numeric_medians.get('transmission_gears', 6)
    row['transmission_type'] = t_type
    row['transmission_gears'] = gears
    row['power_proxy'] = row['engine_size_l'] * row['cylinders']

    df = pd.DataFrame([row])

    # Scale numerics
    numeric_cols = prep['numeric_cols']
    # Ensure all numeric cols present
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = numeric_medians.get(col, 0)
    df[numeric_cols] = df[numeric_cols].fillna(value=numeric_medians)
    scaler = prep['scaler']
    numeric_scaled = scaler.transform(df[numeric_cols])  # (1,5)

    ohe_cols = prep['ohe_cols']
    for col in ohe_cols:
        if col not in df.columns:
            df[col] = categorical_modes.get(col, 'unknown')
    df[ohe_cols] = df[ohe_cols].astype(str)
    ohe = prep['ohe']
    ohe_arr = ohe.transform(df[ohe_cols])  # (1,39)

    make_mapping = mappings['make']
    model_mapping = mappings['model']
    make_idx = np.array([make_mapping.get(row['make'], make_mapping['unknown'])])
    model_idx = np.array([model_mapping.get(row['model'], model_mapping['unknown'])])

    X_full = np.concatenate([numeric_scaled, ohe_arr, make_idx.reshape(-1,1), model_idx.reshape(-1,1)], axis=1)  # (1,46)

    # Compute SHAP for this single instance using explainer
    try:
        # Use background from earlier
        # PermutationExplainer expects 2D array
        shap_values = _explainer(X_full)
        # shap_values.values shape (1,46)
        values = shap_values.values[0] if hasattr(shap_values, 'values') else shap_values[0]
        if isinstance(values, np.ndarray) and values.ndim == 2:
            values = values[0]
    except Exception as e:
        print(f"explain_single SHAP failed: {e}, returning fallback with zero contributions")
        values = np.zeros(len(_feature_names))

    # Build sorted list
    contributions = []
    for fname, val in zip(_feature_names, values):
        contributions.append({"feature": fname, "contribution": float(val)})
    contributions = sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions

if __name__ == "__main__":
    print("=== SHAP Explainability ===")
    values, X_sample, feature_names = compute_shap(sample_size=50, max_evals=500)
    print(f"SHAP values shape: {values.shape}")
    # Example explain_single
    example_car = {
        "year": 2020,
        "make": "toyota",
        "model": "corolla",
        "vehicle_class": "Compact",
        "engine_size_l": 1.8,
        "cylinders": 4,
        "transmission": "A6",
        "fuel_type": "X"
    }
    print("\nExample explain_single for 2020 Toyota Corolla:")
    contribs = explain_single(example_car)
    for c in contribs[:10]:
        print(f"  {c['feature']:<30} {c['contribution']:+.3f}")
    print(f"\nOverall SHAP summary saved to {SHAP_PLOT_PATH}")

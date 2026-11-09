"""
FuelNet Preprocessing Pipeline — Phase 2
- Cleaning, encoding, scaling, honest splitting with GroupKFold
- Saves category_mappings.json and preprocessor.pkl to models_artifacts/
"""
import pathlib
import re
import json
import pickle
import warnings
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupKFold

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_CSV = PROJECT_ROOT / "data" / "Fuel_Consumption_2000-2022.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "models_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_MAPPINGS_PATH = ARTIFACTS_DIR / "category_mappings.json"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"

# Column rename map (exact raw -> snake_case) — must match EDA notebook
RENAME_MAP = {
    'YEAR': 'year',
    'MAKE': 'make',
    'MODEL': 'model',
    'VEHICLE CLASS': 'vehicle_class',
    'ENGINE SIZE': 'engine_size_l',
    'CYLINDERS': 'cylinders',
    'TRANSMISSION': 'transmission',
    'FUEL': 'fuel_type',
    'FUEL CONSUMPTION': 'fuel_city_l_100km',  # CITY consumption (9.2 L/100km)
    'HWY (L/100 km)': 'fuel_hwy_l_100km',
    'COMB (L/100 km)': 'fuel_comb_l_100km',
    'COMB (mpg)': 'fuel_comb_mpg',
    'EMISSIONS': 'co2_g_km',
}

TARGET_COLS = ['fuel_city_l_100km', 'fuel_hwy_l_100km', 'fuel_comb_l_100km', 'co2_g_km']
NUMERIC_COLS = ['year', 'engine_size_l', 'cylinders', 'power_proxy', 'transmission_gears']
OHE_COLS = ['vehicle_class', 'transmission_type', 'fuel_type']

# ------------------------------------------------------------------
# Cleaning helpers
# ------------------------------------------------------------------
def clean_string(s):
    """Lowercase, remove special characters, strip, collapse whitespace."""
    if pd.isna(s):
        return s
    s = str(s).lower().strip()
    # keep alphanumeric and space only — hyphens, dots, slashes removed so F-150 -> f150
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def parse_transmission(trans):
    """Returns (transmission_type, transmission_gears)."""
    if pd.isna(trans):
        return pd.Series(['unknown', np.nan])
    trans_str = str(trans).strip()
    # type: M* -> manual, otherwise automatic (covers A, AS, AM, AV)
    if trans_str.startswith('M'):
        t_type = 'manual'
    else:
        t_type = 'automatic'
    # gears: trailing digits
    m = re.search(r'(\d+)', trans_str)
    gears = int(m.group(1)) if m else np.nan
    return pd.Series([t_type, gears])

def load_and_rename(csv_path=DATA_CSV):
    df = pd.read_csv(csv_path)
    # verify rename covers all columns
    missing = set(df.columns) - set(RENAME_MAP.keys())
    if missing:
        warnings.warn(f"Unexpected columns not in rename map: {missing}")
    df = df.rename(columns=RENAME_MAP)
    return df

def clean_dataframe(df):
    """Apply cleaning steps to a already-renamed df."""
    df = df.copy()
    # 1. Standardize make and model
    df['make'] = df['make'].apply(clean_string)
    df['model'] = df['model'].apply(clean_string)
    # also clean vehicle_class lightly? Keep raw for one-hot but we could also lower for consistency
    # Spec says only make/model standardized, so leave vehicle_class as is (but strip)
    df['vehicle_class'] = df['vehicle_class'].astype(str).str.strip()
    df['fuel_type'] = df['fuel_type'].astype(str).str.strip()
    df['transmission'] = df['transmission'].astype(str).str.strip()

    # 2. Extract transmission_type and gears
    df[['transmission_type', 'transmission_gears']] = df['transmission'].apply(parse_transmission)
    # ensure transmission_gears numeric
    df['transmission_gears'] = pd.to_numeric(df['transmission_gears'], errors='coerce')

    # 3. Create power_proxy
    df['power_proxy'] = df['engine_size_l'] * df['cylinders']

    return df

# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def run_pipeline(csv_path=DATA_CSV, artifacts_dir=ARTIFACTS_DIR, n_splits=5, random_state=42):
    print("=== FuelNet Preprocessing Pipeline ===")
    print(f"Loading CSV: {csv_path}")
    if not pathlib.Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}. Ensure data/Fuel_Consumption_2000-2022.csv exists.")
    # First print exact raw columns BEFORE renaming
    df_raw_exact = pd.read_csv(csv_path, nrows=0)
    print(f"Exact raw columns: {df_raw_exact.columns.tolist()}")
    for c in df_raw_exact.columns:
        print(f"  {repr(c)}")
    df_raw = load_and_rename(csv_path)
    print(f"Renamed columns: {list(df_raw.columns)}")
    print(f"Raw shape: {df_raw.shape}")

    # Cleaning
    df = clean_dataframe(df_raw)
    print(f"After cleaning shape: {df.shape}")
    print(f"Columns after cleaning: {list(df.columns)}")

    # Handle missing values — compute medians/modes from train_pool only (2000-2020)
    # First split temporal hold-out
    temporal_test = df[df['year'].isin([2021, 2022])].copy()
    train_pool = df[~df['year'].isin([2021, 2022])].copy()
    print(f"\nTemporal split: train_pool (2000-2020) = {train_pool.shape}, temporal_test (2021-2022) = {temporal_test.shape}")
    print(f"Train pool year range: {train_pool['year'].min()} - {train_pool['year'].max()}")
    print(f"Test year range: {temporal_test['year'].min()} - {temporal_test['year'].max()}" if len(temporal_test) else "Test empty!")

    # Compute imputation values from train_pool
    numeric_medians = {}
    for col in NUMERIC_COLS + TARGET_COLS:  # targets also need handling but they are not features? ensure no NaN
        if col in df.columns:
            numeric_medians[col] = train_pool[col].median()

    # For categorical OHE cols, mode
    categorical_modes = {}
    for col in OHE_COLS + ['make', 'model']:
        if col in df.columns:
            mode_vals = train_pool[col].mode()
            categorical_modes[col] = mode_vals.iloc[0] if len(mode_vals) else 'unknown'

    # Apply imputation to both splits
    def impute(df_in):
        df_out = df_in.copy()
        for col, med in numeric_medians.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(med)
        for col, mode in categorical_modes.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(mode)
                # also replace empty strings
                df_out[col] = df_out[col].replace('', mode)
        return df_out

    train_pool = impute(train_pool)
    temporal_test = impute(temporal_test)

    # Build category-to-index mappings for make and model from train_pool ONLY
    make_uniques = sorted(train_pool['make'].dropna().unique().tolist())
    model_uniques = sorted(train_pool['model'].dropna().unique().tolist())
    make_mapping = {k: i for i, k in enumerate(make_uniques)}
    model_mapping = {k: i for i, k in enumerate(model_uniques)}
    # add unknown index at the end
    make_mapping['unknown'] = len(make_mapping)
    model_mapping['unknown'] = len(model_mapping)
    print(f"\nCategory mappings: make vocab={len(make_mapping)} (incl unknown), model vocab={len(model_mapping)} (incl unknown)")
    print(f"Example make mapping: {list(make_mapping.items())[:5]} ... unknown={make_mapping['unknown']}")
    print(f"Example model mapping: {list(model_mapping.items())[:5]} ... unknown={model_mapping['unknown']}")

    # Save mappings to JSON
    mappings = {
        "make": make_mapping,
        "model": model_mapping,
        "make_vocab_size": len(make_mapping),
        "model_vocab_size": len(model_mapping)
    }
    with open(CATEGORY_MAPPINGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2)
    print(f"Saved category mappings to {CATEGORY_MAPPINGS_PATH}")

    # One-hot encode and scale — fit only on train_pool
    # Ensure OHE cols are string
    for col in OHE_COLS:
        train_pool[col] = train_pool[col].astype(str)
        temporal_test[col] = temporal_test[col].astype(str)

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(train_pool[OHE_COLS])
    ohe_feature_names = ohe.get_feature_names_out(OHE_COLS).tolist()
    print(f"OHE fitted: {len(ohe_feature_names)} features -> {ohe_feature_names[:10]}...")

    scaler = StandardScaler()
    scaler.fit(train_pool[NUMERIC_COLS])
    print(f"Scaler fitted on {NUMERIC_COLS}: mean {scaler.mean_.round(2).tolist()}, var {scaler.var_.round(2).tolist()}")

    # Save preprocessor objects
    preprocessor = {
        "scaler": scaler,
        "ohe": ohe,
        "numeric_cols": NUMERIC_COLS,
        "ohe_cols": OHE_COLS,
        "target_cols": TARGET_COLS,
        "numeric_medians": numeric_medians,
        "categorical_modes": categorical_modes,
        "make_mapping": make_mapping,
        "model_mapping": model_mapping,
        "ohe_feature_names": ohe_feature_names,
    }
    with open(PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Saved preprocessor to {PREPROCESSOR_PATH}")

    # Helper to transform a dataframe using fitted objects
    def transform(df_in):
        df_t = df_in.copy()
        # numeric scaled
        num_scaled = scaler.transform(df_t[NUMERIC_COLS])
        # ohe
        ohe_arr = ohe.transform(df_t[OHE_COLS])
        # make/model indices
        make_idx = df_t['make'].apply(lambda x: make_mapping.get(x, make_mapping['unknown'])).values
        model_idx = df_t['model'].apply(lambda x: model_mapping.get(x, model_mapping['unknown'])).values
        # targets
        targets = df_t[TARGET_COLS].values if all(c in df_t.columns for c in TARGET_COLS) else None
        # groups for leak check
        groups = (df_t['make'] + " " + df_t['model']).values
        return {
            "numeric_scaled": num_scaled,
            "ohe": ohe_arr,
            "make_idx": make_idx,
            "model_idx": model_idx,
            "targets": targets,
            "groups": groups,
            "df": df_t
        }

    train_pool_transformed = transform(train_pool)
    test_transformed = transform(temporal_test)

    print(f"\nTransformed shapes:")
    print(f"  train_pool numeric_scaled: {train_pool_transformed['numeric_scaled'].shape}")
    print(f"  train_pool ohe: {train_pool_transformed['ohe'].shape}")
    print(f"  test numeric_scaled: {test_transformed['numeric_scaled'].shape}")
    print(f"  test ohe: {test_transformed['ohe'].shape}")

    # GroupKFold splitting on train_pool (2000-2020) grouped by make+model
    groups = train_pool_transformed['groups']
    print(f"\nUnique groups (make+model) in train_pool: {len(np.unique(groups))}")
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(train_pool_transformed['numeric_scaled'], groups=groups))

    print(f"\n=== GroupKFold with {n_splits} splits, grouping by make+model ===")
    leak_passed = True
    for i, (train_idx, val_idx) in enumerate(splits):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = train_groups.intersection(val_groups)
        status = "LEAK DETECTED" if overlap else "LEAK CHECK PASSED"
        if overlap:
            leak_passed = False
        print(f"Fold {i+1}: train {len(train_idx)} rows, val {len(val_idx)} rows, overlap {len(overlap)} -> {status}")
        if overlap:
            print(f"  Example overlapping groups: {list(overlap)[:5]}")

    overall_status = "LEAK CHECK PASSED" if leak_passed else "LEAK DETECTED"
    print(f"\n=== Overall: {overall_status} ===")
    if not leak_passed:
        print("WARNING: Leakage detected — same make+model appears in both train and val!")

    # Also save processed arrays for convenience (train_pool and test)
    processed_path = artifacts_dir / "processed_data.pkl"
    with open(processed_path, 'wb') as f:
        pickle.dump({
            "train_pool": train_pool_transformed,
            "test": test_transformed,
            "splits": splits,
            "groups": groups,
        }, f)
    print(f"Saved processed data to {processed_path}")

    return {
        "train_pool": train_pool_transformed,
        "test": test_transformed,
        "splits": splits,
        "leak_passed": leak_passed,
        "preprocessor": preprocessor,
        "mappings": mappings,
    }

if __name__ == "__main__":
    result = run_pipeline()
    # Ensure leak check passes before proceeding
    if not result["leak_passed"]:
        print("Pipeline finished but leak check FAILED — fix before continuing.")
    else:
        print("Pipeline finished successfully — LEAK CHECK PASSED.")

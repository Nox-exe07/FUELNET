"""
FuelNet PyTorch Dataset — Phase 2
Loads processed data and returns (numeric_features_tensor, categorical_indices_dict, targets_tensor)
Targets: 4 values [fuel_city_l_100km, fuel_hwy_l_100km, fuel_comb_l_100km, co2_g_km]
"""
import pathlib
import pickle
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "models_artifacts"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
CATEGORY_MAPPINGS_PATH = ARTIFACTS_DIR / "category_mappings.json"


class FuelConsumptionDataset(Dataset):
    """
    PyTorch Dataset that returns:
        numeric_features_tensor: FloatTensor shape (numeric_dim + ohe_dim)
        categorical_indices_dict: dict {'make': LongTensor, 'model': LongTensor}
        targets_tensor: FloatTensor shape (4) or None
    """
    def __init__(self, numeric_features, make_indices, model_indices, targets=None, ohe_features=None, concat_ohe=True):
        """
        Args:
            numeric_features: np.ndarray (N, numeric_dim) — already scaled
            make_indices: np.ndarray (N,) — integer indices for make embedding
            model_indices: np.ndarray (N,) — integer indices for model embedding
            targets: np.ndarray (N, 4) or None
            ohe_features: np.ndarray (N, ohe_dim) or None
            concat_ohe: if True and ohe_features provided, concat with numeric_features
        """
        # Combine numeric + ohe into single numeric tensor as per Fusion spec
        if ohe_features is not None and concat_ohe:
            self.numeric = np.concatenate([numeric_features, ohe_features], axis=1)
        else:
            self.numeric = numeric_features

        self.make_indices = make_indices
        self.model_indices = model_indices
        self.targets = targets

        assert len(self.numeric) == len(self.make_indices) == len(self.model_indices), "Length mismatch"
        if self.targets is not None:
            assert len(self.targets) == len(self.numeric), "Targets length mismatch"

        # Store as float32/long
        self.numeric = self.numeric.astype(np.float32)
        self.make_indices = self.make_indices.astype(np.int64)
        self.model_indices = self.model_indices.astype(np.int64)
        if self.targets is not None:
            self.targets = self.targets.astype(np.float32)

    def __len__(self):
        return len(self.numeric)

    def __getitem__(self, idx):
        numeric = torch.from_numpy(self.numeric[idx])  # (F,)
        cat_dict = {
            'make': torch.tensor(self.make_indices[idx], dtype=torch.long),
            'model': torch.tensor(self.model_indices[idx], dtype=torch.long),
        }
        if self.targets is not None:
            target = torch.from_numpy(self.targets[idx])  # (4,)
        else:
            target = torch.zeros(4, dtype=torch.float32)  # placeholder if no target
        return numeric, cat_dict, target

    @classmethod
    def from_processed_pickle(cls, split="train_pool", fold=0, artifacts_dir=ARTIFACTS_DIR):
        """
        Load processed_data.pkl saved by preprocess.py and return dataset for a split.
        split: 'train_pool' (all 2000-2020), 'test' (2021-2022), or f'fold_{i}_train' / f'fold_{i}_val'
        """
        processed_path = pathlib.Path(artifacts_dir) / "processed_data.pkl"
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed data not found at {processed_path}. Run preprocess.py first.")
        with open(processed_path, 'rb') as f:
            data = pickle.load(f)

        if split == "train_pool":
            d = data["train_pool"]
            return cls(d["numeric_scaled"], d["make_idx"], d["model_idx"], d["targets"], d["ohe"])
        elif split == "test":
            d = data["test"]
            return cls(d["numeric_scaled"], d["make_idx"], d["model_idx"], d["targets"], d["ohe"])
        elif split.startswith("fold_"):
            # e.g., fold_0_train, fold_1_val
            # parse fold idx and train/val
            import re
            m = re.match(r'fold_(\d+)_(train|val)', split)
            if not m:
                raise ValueError(f"Invalid split name {split}")
            fold_idx = int(m.group(1))
            which = m.group(2)
            all_data = data["train_pool"]
            splits = data["splits"]
            train_idx, val_idx = splits[fold_idx]
            idx = train_idx if which == "train" else val_idx
            return cls(
                all_data["numeric_scaled"][idx],
                all_data["make_idx"][idx],
                all_data["model_idx"][idx],
                all_data["targets"][idx],
                all_data["ohe"][idx],
            )
        else:
            raise ValueError(f"Unknown split {split}")

    @classmethod
    def from_csv_with_preprocessor(cls, csv_path, preprocessor_path=PREPROCESSOR_PATH):
        """
        Transform a raw CSV (or DataFrame) using saved preprocessor.pkl.
        Useful for inference or creating datasets from new data.
        Returns dataset with targets if targets present, else zeros.
        """
        import re
        # Load preprocessor
        with open(preprocessor_path, 'rb') as f:
            prep = pickle.load(f)
        scaler = prep["scaler"]
        ohe = prep["ohe"]
        numeric_cols = prep["numeric_cols"]
        ohe_cols = prep["ohe_cols"]
        target_cols = prep["target_cols"]
        numeric_medians = prep["numeric_medians"]
        categorical_modes = prep["categorical_modes"]
        make_mapping = prep["make_mapping"]
        model_mapping = prep["model_mapping"]

        # Load csv if path provided
        if isinstance(csv_path, (str, pathlib.Path)):
            df = pd.read_csv(csv_path)
            # rename if raw
            rename_map = {
                'YEAR': 'year', 'MAKE': 'make', 'MODEL': 'model',
                'VEHICLE CLASS': 'vehicle_class', 'ENGINE SIZE': 'engine_size_l',
                'CYLINDERS': 'cylinders', 'TRANSMISSION': 'transmission',
                'FUEL': 'fuel_type', 'FUEL CONSUMPTION': 'fuel_city_l_100km',
                'HWY (L/100 km)': 'fuel_hwy_l_100km', 'COMB (L/100 km)': 'fuel_comb_l_100km',
                'COMB (mpg)': 'fuel_comb_mpg', 'EMISSIONS': 'co2_g_km',
            }
            if any(c in df.columns for c in rename_map.keys()):
                df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        else:
            df = csv_path  # assume DataFrame

        # Cleaning helpers (same as preprocess.py)
        def clean_string(s):
            if pd.isna(s):
                return s
            s = str(s).lower().strip()
            s = re.sub(r'[^a-z0-9\s]', '', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        def parse_transmission(trans):
            if pd.isna(trans):
                return pd.Series(['unknown', np.nan])
            trans_str = str(trans).strip()
            t_type = 'manual' if trans_str.startswith('M') else 'automatic'
            m = re.search(r'(\d+)', trans_str)
            gears = int(m.group(1)) if m else np.nan
            return pd.Series([t_type, gears])

        df = df.copy()
        if 'make' in df.columns:
            df['make'] = df['make'].apply(clean_string)
        if 'model' in df.columns:
            df['model'] = df['model'].apply(clean_string)
        if 'transmission' in df.columns:
            df[['transmission_type', 'transmission_gears']] = df['transmission'].apply(parse_transmission)
            df['transmission_gears'] = pd.to_numeric(df['transmission_gears'], errors='coerce')
        if all(c in df.columns for c in ['engine_size_l', 'cylinders']):
            df['power_proxy'] = df['engine_size_l'] * df['cylinders']

        # Impute
        for col, med in numeric_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(med)
        for col, mode in categorical_modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode)

        for col in ohe_cols:
            df[col] = df[col].astype(str)

        numeric_scaled = scaler.transform(df[numeric_cols])
        ohe_arr = ohe.transform(df[ohe_cols])
        make_idx = df['make'].apply(lambda x: make_mapping.get(x, make_mapping['unknown'])).values
        model_idx = df['model'].apply(lambda x: model_mapping.get(x, model_mapping['unknown'])).values
        targets = df[target_cols].values if all(c in df.columns for c in target_cols) else None

        return cls(numeric_scaled, make_idx, model_idx, targets, ohe_arr)


def get_temporal_splits(artifacts_dir=ARTIFACTS_DIR):
    """Helper to return train_pool and test datasets for quick use."""
    train_ds = FuelConsumptionDataset.from_processed_pickle("train_pool", artifacts_dir=artifacts_dir)
    test_ds = FuelConsumptionDataset.from_processed_pickle("test", artifacts_dir=artifacts_dir)
    return train_ds, test_ds


if __name__ == "__main__":
    # Quick sanity check
    import sys
    try:
        train_ds, test_ds = get_temporal_splits()
        print(f"Train pool dataset: {len(train_ds)} samples")
        print(f"Test dataset: {len(test_ds)} samples")
        num, cat, tgt = train_ds[0]
        print(f"Sample numeric shape: {num.shape}, cat keys: {list(cat.keys())}, target shape: {tgt.shape}")
        print(f"Numeric sample: {num[:5]}")
        print(f"Cat make idx: {cat['make']}, model idx: {cat['model']}")
        print(f"Target: {tgt}")
    except Exception as e:
        print(f"Dataset sanity check failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

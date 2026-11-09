"""ModelLoader for FuelNet API — preprocessing + inference + uncertainty + eco score"""
import pathlib
import pickle
import json
import re
import os
import time
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "models_artifacts"
# Support env override
load_dotenv(PROJECT_ROOT / ".env")
FUEL_PRICE = float(os.getenv("FUEL_PRICE", "1.50"))
MODEL_ARTIFACTS_DIR = os.getenv("MODEL_ARTIFACTS_DIR", "models_artifacts")
ARTIFACTS_DIR = PROJECT_ROOT / MODEL_ARTIFACTS_DIR if not pathlib.Path(MODEL_ARTIFACTS_DIR).is_absolute() else pathlib.Path(MODEL_ARTIFACTS_DIR)

from src.models.fuelnet import FuelNet

def clean_string(s):
    if pd.isna(s):
        return s
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def parse_transmission(trans):
    if pd.isna(trans):
        return ('unknown', np.nan)
    trans_str = str(trans).strip()
    t_type = 'manual' if trans_str.startswith('M') else 'automatic'
    m = re.search(r'(\d+)', trans_str)
    gears = int(m.group(1)) if m else np.nan
    return (t_type, gears)

class ModelLoader:
    def __init__(self, artifacts_dir=ARTIFACTS_DIR):
        self.artifacts_dir = pathlib.Path(artifacts_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ModelLoader] Device: {self.device}")
        # Paths
        self.ts_path = self.artifacts_dir / "fuelnet_ts.pt"
        self.best_path = self.artifacts_dir / "fuelnet_best.pt"
        self.preprocessor_path = self.artifacts_dir / "preprocessor.pkl"
        self.mappings_path = self.artifacts_dir / "category_mappings.json"
        self.target_scaler_path = self.artifacts_dir / "target_scaler.pkl"
        self.model_card_path = self.artifacts_dir / "model_card.json"
        self.processed_path = self.artifacts_dir / "processed_data.pkl"

        # Load artifacts
        with open(self.mappings_path, 'r', encoding='utf-8') as f:
            self.mappings = json.load(f)
        self.make_mapping = self.mappings['make']
        self.model_mapping = self.mappings['model']
        self.make_vocab_size = self.mappings['make_vocab_size']
        self.model_vocab_size = self.mappings['model_vocab_size']

        with open(self.preprocessor_path, 'rb') as f:
            self.prep = pickle.load(f)
        self.scaler = self.prep['scaler']
        self.ohe = self.prep['ohe']
        self.numeric_cols = self.prep['numeric_cols']
        self.ohe_cols = self.prep['ohe_cols']
        self.numeric_medians = self.prep['numeric_medians']
        self.categorical_modes = self.prep['categorical_modes']
        self.ohe_feature_names = self.prep['ohe_feature_names']

        with open(self.target_scaler_path, 'rb') as f:
            self.target_scaler = pickle.load(f)

        with open(self.model_card_path, 'r', encoding='utf-8') as f:
            self.model_card = json.load(f)
        self.model_version = self.model_card.get("model_version", "fuelnet-t-v1.0.0")

        # Training CO2 distribution for eco_score
        try:
            with open(self.processed_path, 'rb') as f:
                data = pickle.load(f)
            train_co2 = data['train_pool']['targets'][:, 3]  # raw CO2
            self.train_co2_sorted = np.sort(train_co2)
        except Exception as e:
            print(f"[ModelLoader] Warning: could not load train CO2 distribution: {e}")
            self.train_co2_sorted = np.array([250.0])  # fallback

        # Load models
        numeric_dim = len(self.numeric_cols) + len(self.ohe_feature_names)  # 44
        self.numeric_dim = numeric_dim

        # Primary model for fast inference — try TorchScript first
        self.model_ts = None
        self.model = None
        try:
            if self.ts_path.exists():
                self.model_ts = torch.jit.load(str(self.ts_path), map_location=self.device)
                self.model_ts.eval()
                print(f"[ModelLoader] Loaded TorchScript from {self.ts_path}")
            else:
                print(f"[ModelLoader] TorchScript not found at {self.ts_path}, using regular model")
        except Exception as e:
            print(f"[ModelLoader] Failed to load TorchScript: {e}")
            self.model_ts = None

        # Regular model for dropout uncertainty (and fallback)
        try:
            regular = FuelNet(make_vocab_size=self.make_vocab_size, model_vocab_size=self.model_vocab_size, numeric_dim=numeric_dim)
            state = torch.load(str(self.best_path), map_location=self.device)
            regular.load_state_dict(state)
            regular.to(self.device)
            regular.eval()
            self.model = regular
            print(f"[ModelLoader] Loaded regular FuelNet from {self.best_path} ({sum(p.numel() for p in regular.parameters()):,} params)")
            if self.model_ts is None:
                self.model_ts = self.model
        except Exception as e:
            print(f"[ModelLoader] Failed to load regular model: {e}")
            if self.model_ts is None:
                raise RuntimeError(f"Could not load any model: {e}")

        # Fuel price
        self.fuel_price = FUEL_PRICE
        print(f"[ModelLoader] Fuel price: {self.fuel_price} USD/L")
        print(f"[ModelLoader] Ready — numeric_dim {numeric_dim}, ohe {len(self.ohe_feature_names)}, vocab make {self.make_vocab_size} model {self.model_vocab_size}")

    def _preprocess_single(self, req_dict):
        """Preprocess one PredictRequest dict -> (numeric_tensor, make_idx, model_idx, unseen_flag)"""
        # Extract fields with defaults
        year = int(req_dict.get('year', 2020))
        make_raw = str(req_dict.get('make', 'unknown'))
        model_raw = str(req_dict.get('model', 'unknown'))
        vehicle_class = str(req_dict.get('vehicle_class', self.categorical_modes.get('vehicle_class', 'Compact')))
        engine_size_l = float(req_dict.get('engine_size_l', self.numeric_medians.get('engine_size_l', 3.0)))
        cylinders = int(req_dict.get('cylinders', self.numeric_medians.get('cylinders', 4)))
        transmission = str(req_dict.get('transmission', 'A6'))
        fuel_type = str(req_dict.get('fuel_type', self.categorical_modes.get('fuel_type', 'X')))

        # Clean make/model
        make_clean = clean_string(make_raw)
        model_clean = clean_string(model_raw)

        # Check unseen
        unseen_make = make_clean not in self.make_mapping or make_clean == 'unknown'
        unseen_model = model_clean not in self.model_mapping or model_clean == 'unknown'
        unseen_vehicle = bool(unseen_make or unseen_model)
        make_idx = self.make_mapping.get(make_clean, self.make_mapping['unknown'])
        model_idx = self.model_mapping.get(model_clean, self.model_mapping['unknown'])

        # Transmission parse
        t_type, gears = parse_transmission(transmission)
        if pd.isna(gears):
            gears = self.numeric_medians.get('transmission_gears', 6)

        power_proxy = engine_size_l * cylinders

        # Build DF for scaler/ohe
        row = {
            'year': year,
            'engine_size_l': engine_size_l,
            'cylinders': cylinders,
            'power_proxy': power_proxy,
            'transmission_gears': gears,
            'vehicle_class': vehicle_class,
            'transmission_type': t_type,
            'fuel_type': fuel_type
        }
        # Ensure all numeric cols present, fill NA with median
        for col in self.numeric_cols:
            if col not in row or pd.isna(row[col]):
                row[col] = self.numeric_medians.get(col, 0)

        df_num = pd.DataFrame([row])
        # Numeric scaled
        numeric_scaled = self.scaler.transform(df_num[self.numeric_cols])  # (1,5)

        # OHE
        df_ohe = pd.DataFrame([{k: str(row[k]) for k in self.ohe_cols}])
        ohe_arr = self.ohe.transform(df_ohe[self.ohe_cols])  # (1,39)

        numeric = np.concatenate([numeric_scaled, ohe_arr], axis=1)  # (1,44)
        # Convert to tensors
        numeric_t = torch.from_numpy(numeric.astype(np.float32)).to(self.device)
        make_t = torch.tensor([make_idx], dtype=torch.long, device=self.device)
        model_t = torch.tensor([model_idx], dtype=torch.long, device=self.device)

        return numeric_t, make_t, model_t, unseen_vehicle, row

    def predict_single(self, req_dict, n_stochastic=20):
        """Returns dict for PredictResponse"""
        start = time.time()
        numeric_t, make_t, model_t, unseen_vehicle, row = self._preprocess_single(req_dict)

        # Use primary model for point prediction (eval mode)
        primary = self.model_ts if self.model_ts is not None else self.model
        primary.eval()
        with torch.no_grad():
            pred_scaled = primary(numeric_t, make_t, model_t)  # (1,4) scaled
            pred_raw = self.target_scaler.inverse_transform(pred_scaled.cpu().numpy())  # (1,4)
            pred_raw = pred_raw[0]

        # Uncertainty via 20 stochastic passes with dropout
        # Use regular model with dropout enabled (train mode for dropout layers)
        uncertainty_model = self.model if self.model is not None else primary
        # Put dropout in train mode, keep Layernorm in eval? FuelNet only has LayerNorm (not BatchNorm) so train/eval both similar except dropout.
        # We'll set the whole model to train() for stochasticity, then eval() after.
        # If using TorchScript, it may not support train() properly, so we fallback to regular.
        preds_stochastic = []
        try:
            uncertainty_model.train()  # enable dropout
            with torch.no_grad():
                for _ in range(n_stochastic):
                    # For stochastic, we need to keep dropout active: no torch.no_grad effect on dropout, but we disabled grad.
                    # Dropout will still be active in train() mode even with no_grad
                    p = uncertainty_model(numeric_t, make_t, model_t)
                    p_raw = self.target_scaler.inverse_transform(p.cpu().numpy())[0]
                    preds_stochastic.append(p_raw)
            uncertainty_model.eval()
        except Exception as e:
            print(f"[ModelLoader] Stochastic passes failed: {e}")
            # Fallback: small uncertainty
            preds_stochastic = [pred_raw for _ in range(n_stochastic)]

        preds_stochastic = np.array(preds_stochastic)  # (20,4)
        stds = preds_stochastic.std(axis=0)  # (4,)

        fuel_city, fuel_hwy, fuel_comb, co2 = pred_raw.tolist()
        # Derived
        fuel_comb_mpg = 235.215 / fuel_comb if fuel_comb > 0 else 0.0
        annual_cost_usd = fuel_comb * 15000/100 * self.fuel_price
        # Eco score: percentile within train CO2
        # percentile = proportion <= co2
        percentile = (self.train_co2_sorted <= co2).mean() * 100
        eco_score = int(round(100 - percentile))  # lower CO2 = higher score
        eco_score = max(0, min(100, eco_score))

        inference_ms = (time.time() - start) * 1000

        return {
            "fuel_city_l_100km": float(fuel_city),
            "fuel_hwy_l_100km": float(fuel_hwy),
            "fuel_comb_l_100km": float(fuel_comb),
            "fuel_comb_mpg": float(fuel_comb_mpg),
            "co2_g_km": float(co2),
            "uncertainty": {
                "fuel_city_std": float(stds[0]),
                "fuel_hwy_std": float(stds[1]),
                "fuel_comb_std": float(stds[2]),
                "co2_std": float(stds[3]),
            },
            "annual_cost_usd": float(annual_cost_usd),
            "eco_score": int(eco_score),
            "unseen_vehicle": bool(unseen_vehicle),
            "model_version": str(self.model_version),
            "inference_ms": float(inference_ms),
        }

    def predict_batch(self, req_list, n_stochastic=5):
        """Batch predict: reduce stochastic passes for speed (5)"""
        results = []
        for req in req_list:
            # req may be dict or Pydantic model
            if hasattr(req, 'model_dump'):
                d = req.model_dump()
            elif hasattr(req, 'dict'):
                d = req.dict()
            else:
                d = dict(req)
            results.append(self.predict_single(d, n_stochastic=n_stochastic))
        return results

    def explain_single(self, req_dict):
        """Wrapper to call src.evaluation.explain.explain_single"""
        try:
            from src.evaluation.explain import explain_single as exp_single
            return exp_single(req_dict)
        except Exception as e:
            print(f"[ModelLoader] explain_single failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return dummy
            return [{"feature": "error", "contribution": 0.0}]

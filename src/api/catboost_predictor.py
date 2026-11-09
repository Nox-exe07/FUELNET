"""CatBoost Loader for FuelNet — 57-feature pipeline + log1p target + expm1 invert"""
import pathlib
import pickle
import json
import re
import os
import copy
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
FUEL_PRICE = float(os.getenv("FUEL_PRICE", "1.50"))
# artifact dirs support env override
CATBOOST_BUNDLE_DIR = os.getenv("CATBOOST_BUNDLE_DIR", "catboost_v3_bundle")
PREPARED_DATA_DIR = os.getenv("PREPARED_DATA_DIR", "prepared_dataset/prepared_data")
CATBOOST_DIR = PROJECT_ROOT / CATBOOST_BUNDLE_DIR if not pathlib.Path(CATBOOST_BUNDLE_DIR).is_absolute() else pathlib.Path(CATBOOST_BUNDLE_DIR)
PREPARED_DIR = PROJECT_ROOT / PREPARED_DATA_DIR if not pathlib.Path(PREPARED_DATA_DIR).is_absolute() else pathlib.Path(PREPARED_DATA_DIR)

# Helpers copied from notebook
def parse_transmission(t):
    if pd.isna(t) or str(t).strip()=='' or str(t).upper()=='UNKNOWN':
        return pd.Series({'gear_count': 0, 'is_automatic': 0, 'is_cvt': 0})
    t = str(t).upper()
    is_cvt = int('CVT' in t or 'VARIABLE' in t)
    is_auto = int('A' in t[:3] or 'AUTO' in t or is_cvt)
    # booleans: original had 1 for automatic if A prefix or AUTO or CVT
    # ensure 0/1 ints
    m = re.search(r'(\d+)', t)
    return pd.Series({'gear_count': int(m.group(1)) if m else 0,
                      'is_automatic': is_auto, 'is_cvt': is_cvt})

def unify_vehicle_class(vc):
    if pd.isna(vc): return 'Unknown'
    v = str(vc).strip().upper()
    if v == 'MOTORCYCLE' or v == 'TWO WHEELER': return 'Motorcycle'
    if v == 'BUS': return 'Bus'
    if v == 'TRUCK' or 'PICKUP' in v: return 'Truck'
    if v == 'CAR': return 'Car'
    if any(k in v for k in ['SUBCOMPACT','COMPACT','MINICOMPACT']): return 'Car'
    if any(k in v for k in ['MID-SIZE','MIDSIZE','FULL-SIZE','LARGE']): return 'Car'
    if any(k in v for k in ['SPORT','LUXURY','STATION WAGON']): return 'Car'
    if any(k in v for k in ['SUV','SPORT UTILITY','CROSSOVER']): return 'SUV'
    if any(k in v for k in ['MINIVAN','VAN']): return 'Van'
    return 'Other'

def unify_fuel_type(ft):
    if pd.isna(ft): return 'Unknown'
    f = str(ft).strip().upper()
    if f == 'X': return 'Petrol'
    if f == 'Z': return 'Premium'
    if f == 'D': return 'Diesel'
    if f == 'E': return 'E85'
    if f == 'N': return 'CNG'
    if 'PETROL' in f or 'REGULAR' in f or 'GASOLINE' in f: return 'Petrol'
    if 'PREMIUM' in f: return 'Premium'
    if 'DIESEL' in f: return 'Diesel'
    if 'ELECTRIC' in f or f == 'EV': return 'Electric'
    if 'HYBRID' in f: return 'Hybrid'
    if 'E85' in f: return 'E85'
    if 'CNG' in f or 'NATURAL GAS' in f: return 'CNG'
    return 'Other'

def parse_drivetrain(row):
    model = str(row.get('model', '') if hasattr(row, 'get') else '')
    drive = str(row.get('drive', '') if hasattr(row, 'get') else '')
    text = (model + ' ' + drive).upper()
    if any(k in text for k in ['AWD','4WD','ALL WHEEL','4MOTION','XDRIVE','QUATTRO']): return 'AWD'
    if any(k in text for k in ['RWD','REAR']): return 'RWD'
    if any(k in text for k in ['FWD','FRONT','2WD']): return 'FWD'
    return 'UNK'

def parse_turbo(row):
    text = str(row.get('model', '') if hasattr(row, 'get') else '').upper()
    return int(any(k in text for k in ['TURBO','TDI','TSI','ECOBOOST','TFSI','GTI','STI','TURBODIESEL']))

class CatBoostLoader:
    def __init__(self, bundle_dir=CATBOOST_DIR, prepared_dir=PREPARED_DIR):
        self.bundle_dir = pathlib.Path(bundle_dir)
        self.prepared_dir = pathlib.Path(prepared_dir)
        print(f"[CatBoostLoader] bundle_dir={self.bundle_dir} prepared_dir={self.prepared_dir}")

        # Load bundle
        self.pt_path = self.bundle_dir / "catboost.pt"
        self.cbm_path = self.bundle_dir / "catboost.cbm"
        self.metadata_path = self.bundle_dir / "metadata.json"
        self.te_mappings_path = self.bundle_dir / "te_mappings.json"
        self.te_mappings_pkl = self.bundle_dir / "te_mappings.pkl"
        self.feature_cols_path = self.prepared_dir / "feature_cols.json"
        # legacy also bundle feature_cols
        self.median_path = self.prepared_dir / "feature_median.pkl"

        # Load catboost bundle
        if self.pt_path.exists():
            with open(self.pt_path, 'rb') as f:
                bundle = pickle.load(f)
            self.model = bundle['model']
            self.feature_cols = bundle['feature_cols']
            self.target_transform = bundle.get('target_transform', 'log1p')
            self.best_iteration = bundle.get('best_iteration', None)
            print(f"[CatBoostLoader] Loaded CatBoost pt from {self.pt_path} best_iter={self.best_iteration}")
            print(f"[CatBoostLoader] feature_cols {len(self.feature_cols)}: {self.feature_cols[:6]}...")
        else:
            # try cbm
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor()
            self.model.load_model(str(self.cbm_path))
            with open(self.feature_cols_path, 'r') as f:
                self.feature_cols = json.load(f)
            print(f"[CatBoostLoader] Loaded CatBoost cbm from {self.cbm_path}")

        # Load median
        with open(self.median_path, 'rb') as f:
            self.median = pickle.load(f)  # pandas Series
        # Ensure median is Series with index feature_cols
        if not isinstance(self.median, pd.Series):
            self.median = pd.Series(self.median)

        # Load TE mappings
        self.te_mappings = {}
        if self.te_mappings_path.exists():
            with open(self.te_mappings_path, 'r', encoding='utf-8') as f:
                self.te_mappings = json.load(f)
            print(f"[CatBoostLoader] Loaded TE mappings {len(self.te_mappings)} keys from {self.te_mappings_path}")
        elif self.te_mappings_pkl.exists():
            with open(self.te_mappings_pkl, 'rb') as f:
                self.te_mappings = pickle.load(f)
            print(f"[CatBoostLoader] Loaded TE pkl")
        else:
            print(f"[CatBoostLoader] WARNING: TE mappings not found, fallback to median only")

        # Load metadata
        try:
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        except:
            self.metadata = {"model":"CatBoost","R²":0.887}
        self.model_version = f"catboost-v3-{self.metadata.get('best_iteration',24998)}"

        # --- Residual-based uncertainty (from held-out test predictions) ---
        self.residual_std_global = 44.9
        self.residual_std_india = 119.0
        self.residual_std_by_source = {}
        self._load_residual_stats()

        # --- Data-driven fuel split ratios (city/combined, hwy/combined) ---
        self.city_ratio = 1.1566265060240963
        self.hwy_ratio = 0.8080808080808081
        self._compute_fuel_ratios()

        # --- SHAP explainability config ---
        # EXPLAIN_SHAP_TREES=0 -> exact SHAP over all trees (slow, ~20s)
        self.explain_tree_limit = int(os.getenv("EXPLAIN_SHAP_TREES", "3000"))
        self._shap_model = None
        self._explain_cache = {}

        # Training CO2 distribution for eco score (try to load processed npy)
        try:
            # try full dataset y_log?
            # use median distribution fallback
            self.train_co2_sorted = np.array([150, 200, 250, 300, 400])
            # try loading y_train_orig if available
            import pathlib as pl
            y_path = self.prepared_dir / "y_train_orig.npy"
            if y_path.exists():
                arr = np.load(str(y_path))
                self.train_co2_sorted = np.sort(arr)
                print(f"[CatBoostLoader] Loaded train CO2 dist {len(arr)} rows median {np.median(arr):.1f}")
        except Exception as e:
            print(f"[CatBoostLoader] train CO2 fallback {e}")

        self.fuel_price = FUEL_PRICE
        print(f"[CatBoostLoader] Ready — median len {len(self.median)}, TE keys {len(self.te_mappings)}, fuel_price {self.fuel_price}")
        print(f"[CatBoostLoader] residual std global(non-India)={self.residual_std_global:.2f} "
              f"india={self.residual_std_india:.2f} sources={len(self.residual_std_by_source)} | "
              f"fuel ratios city={self.city_ratio:.3f} hwy={self.hwy_ratio:.3f}")

    def _load_residual_stats(self):
        """Derive uncertainty from held-out residuals in test_predictions.csv."""
        path = self.bundle_dir / "test_predictions.csv"
        try:
            if not path.exists():
                print(f"[CatBoostLoader] {path} missing — using fallback residual std")
                return
            df = pd.read_csv(path)
            if "Residual" not in df.columns:
                return
            resid = pd.to_numeric(df["Residual"], errors="coerce")
            india_mask = pd.Series(False, index=df.index)
            if "Is_India" in df.columns:
                india_mask = pd.to_numeric(df["Is_India"], errors="coerce").fillna(0) == 1
            non_india = resid[~india_mask].dropna()
            india = resid[india_mask].dropna()
            if len(non_india) > 1:
                self.residual_std_global = float(non_india.std())
            if len(india) > 1:
                self.residual_std_india = float(india.std())
            if "Source" in df.columns:
                for src, grp in df.groupby("Source"):
                    r = pd.to_numeric(grp["Residual"], errors="coerce").dropna()
                    if len(r) > 1:
                        self.residual_std_by_source[str(src)] = float(r.std())
        except Exception as e:
            print(f"[CatBoostLoader] residual stats load failed: {e}")

    def _compute_fuel_ratios(self):
        """City/highway split ratios measured from the Canadian reference dataset."""
        csv_path = PROJECT_ROOT / "data" / "Fuel_Consumption_2000-2022.csv"
        try:
            if not csv_path.exists():
                print(f"[CatBoostLoader] {csv_path} missing — using fallback fuel split ratios")
                return
            d = pd.read_csv(csv_path)
            comb = pd.to_numeric(d["COMB (L/100 km)"], errors="coerce")
            city = pd.to_numeric(d["FUEL CONSUMPTION"], errors="coerce")
            hwy = pd.to_numeric(d["HWY (L/100 km)"], errors="coerce")
            mask = (comb > 0) & city.notna() & hwy.notna()
            if int(mask.sum()) > 100:
                self.city_ratio = float((city[mask] / comb[mask]).median())
                self.hwy_ratio = float((hwy[mask] / comb[mask]).median())
        except Exception as e:
            print(f"[CatBoostLoader] fuel ratio compute failed: {e}")

    def _select_residual_std(self, raw_row):
        """Pick the most specific residual std available for this request."""
        source = str(raw_row.get("data_source") or "")
        if source in self.residual_std_by_source:
            return self.residual_std_by_source[source]
        if int(raw_row.get("is_india", 0) or 0) == 1:
            return self.residual_std_india
        return self.residual_std_global

    def _get_shap_model(self):
        """Lazily build the (optionally tree-limited) model used for SHAP."""
        if self._shap_model is not None:
            return self._shap_model
        from catboost import CatBoostRegressor
        try:
            m = CatBoostRegressor()
            m.load_model(str(self.cbm_path))
            total = int(getattr(m, "tree_count_", 0) or 0)
            limit = self.explain_tree_limit
            if limit and limit > 0 and total and limit < total:
                m.shrink(limit)
                print(f"[CatBoostLoader] SHAP model limited to {limit}/{total} trees")
            else:
                print(f"[CatBoostLoader] SHAP model uses all {total} trees (exact)")
            self._shap_model = m
        except Exception as e:
            print(f"[CatBoostLoader] could not build SHAP model ({e}); using served model")
            self._shap_model = self.model
        return self._shap_model

    def _te_lookup(self, col, raw_value):
        # col is like 'make', 'model' etc — prioritize case-insensitive lower mapping (aggregated) for robustness
        if raw_value is None or (isinstance(raw_value,float) and np.isnan(raw_value)) or str(raw_value).strip()=='' or str(raw_value).upper()=='UNKNOWN':
            gm_key = f"{col}_global_mean"
            return float(self.te_mappings.get(gm_key, self.median.get(f"{col}_te", 5.66)))
        raw_str = str(raw_value).strip()
        lk = raw_str.lower()
        d_low = self.te_mappings.get(f"{col}_lower", {})
        # prioritize lower (aggregated across case variants) for consistency
        if lk in d_low:
            return float(d_low[lk])
        # fallback exact
        d = self.te_mappings.get(col, {})
        if raw_str in d:
            return float(d[raw_str])
        if raw_str.upper() in d:
            return float(d[raw_str.upper()])
        # fallback to global mean
        gm_key = f"{col}_global_mean"
        gm = self.te_mappings.get(gm_key, None)
        if gm is not None:
            return float(gm)
        med_key = f"{col}_te"
        if med_key in self.median.index:
            return float(self.median[med_key])
        return 5.65

    def _drivetrain_enc_lookup(self, drv_str):
        if drv_str is None or str(drv_str).strip()=='':
            return int(self.te_mappings.get('drivetrain_global_enc', int(self.median.get('drivetrain_enc', 2))))
        s = str(drv_str).strip()
        d = self.te_mappings.get('drivetrain', {})
        if s in d:
            return int(d[s])
        dl = self.te_mappings.get('drivetrain_lower', {})
        lk = s.lower()
        if lk in dl:
            return int(dl[lk])
        # try upper
        uk = s.upper()
        if uk in d:
            return int(d[uk])
        return int(self.te_mappings.get('drivetrain_global_enc', 2))

    def _preprocess_single(self, req_dict):
        # Raw inputs with defaults - prefer provided, else NaN for median fill later
        def get(req, key, default=np.nan):
            v = req.get(key, default)
            if v is None or (isinstance(v,str) and v.strip()==''):
                return np.nan
            return v

        year = int(req_dict.get('year', 2020)) if not pd.isna(req_dict.get('year', 2020)) else 2020
        make_raw = str(req_dict.get('make', 'UNKNOWN'))
        model_raw = str(req_dict.get('model', 'UNKNOWN'))
        vehicle_class = str(req_dict.get('vehicle_class', 'Car'))
        engine_size_l = float(req_dict.get('engine_size_l', np.nan)) if not pd.isna(req_dict.get('engine_size_l', np.nan)) else np.nan
        cylinders = float(req_dict.get('cylinders', np.nan)) if not pd.isna(req_dict.get('cylinders', np.nan)) else np.nan
        transmission = str(req_dict.get('transmission', 'UNKNOWN'))
        fuel_type = str(req_dict.get('fuel_type', 'UNKNOWN'))

        # Advanced optional fields - if not provided, NaN -> median fill
        mileage_km = get(req_dict, 'mileage_km', np.nan)
        power_hp = get(req_dict, 'power_hp', np.nan)
        speed_kmh = get(req_dict, 'speed_kmh', np.nan)
        acceleration_ms2 = get(req_dict, 'acceleration_ms2', np.nan)
        temperature_c = get(req_dict, 'temperature_c', np.nan)
        humidity_pct = get(req_dict, 'humidity_pct', np.nan)
        wind_speed_ms = get(req_dict, 'wind_speed_ms', np.nan)
        air_pressure_hpa = get(req_dict, 'air_pressure_hpa', np.nan)
        vehicle_age_raw = get(req_dict, 'vehicle_age', np.nan)
        # vehicle_age derived if not provided: use 2026 - year
        if pd.isna(vehicle_age_raw):
            try:
                vehicle_age = 2026 - int(year)
            except:
                vehicle_age = np.nan
        else:
            vehicle_age = float(vehicle_age_raw) if not pd.isna(vehicle_age_raw) else np.nan

        road_type = str(req_dict.get('road_type', 'UNKNOWN')) if req_dict.get('road_type', None) is not None and str(req_dict.get('road_type')).strip()!='' else 'UNKNOWN'
        traffic_condition = str(req_dict.get('traffic_condition', 'UNKNOWN')) if req_dict.get('traffic_condition', None) is not None and str(req_dict.get('traffic_condition')).strip()!='' else 'UNKNOWN'
        region = str(req_dict.get('region', 'UNKNOWN')) if req_dict.get('region', None) is not None and str(req_dict.get('region')).strip()!='' else 'UNKNOWN'
        regulatory_cycle = str(req_dict.get('regulatory_cycle', 'UNKNOWN')) if req_dict.get('regulatory_cycle', None) is not None and str(req_dict.get('regulatory_cycle')).strip()!='' else 'UNKNOWN'
        data_source = str(req_dict.get('data_source', 'UNKNOWN')) if req_dict.get('data_source', None) is not None and str(req_dict.get('data_source')).strip()!='' else 'UNKNOWN'
        drive_raw = str(req_dict.get('drive', 'UNKNOWN')) if req_dict.get('drive', None) is not None and str(req_dict.get('drive')).strip()!='' else 'UNKNOWN'

        # is_india: explicit or inferred from region/data_source
        is_india_val = req_dict.get('is_india', None)
        if is_india_val is not None:
            try:
                is_india = int(float(is_india_val))
            except:
                is_india = 1 if str(is_india_val).lower() in ['1','true','india'] else 0
        else:
            if str(region).lower()=='india' or str(data_source)=='India_Vehicle_Emission':
                is_india = 1
            else:
                is_india = 0

        # has flags - before median fill, determine if speed/mileage etc were provided
        has_speed = 0 if pd.isna(speed_kmh) else 1
        has_acceleration = 0 if pd.isna(acceleration_ms2) else 1
        has_weather = 0 if (pd.isna(temperature_c) or pd.isna(humidity_pct)) else 1
        has_road = 0 if road_type=='UNKNOWN' else 1
        has_age_mileage = 0 if (pd.isna(vehicle_age) or pd.isna(mileage_km)) else 1

        # Unified categorical helpers
        vehicle_class_unified = unify_vehicle_class(vehicle_class)
        fuel_type_unified = unify_fuel_type(fuel_type)

        # Ensure engine_size and cylinders defaults for derived if NaN -> will be filled median later, but need for derived calc we keep NaN
        # Use float for math, NaN propagates
        # Derived features
        # engine_per_cylinder
        try:
            if pd.notna(engine_size_l) and pd.notna(cylinders) and float(cylinders)!=0:
                engine_per_cylinder = float(engine_size_l) / float(cylinders)
            else:
                engine_per_cylinder = np.nan
        except:
            engine_per_cylinder = np.nan
        try:
            engine_squared = float(engine_size_l)**2 if pd.notna(engine_size_l) else np.nan
        except:
            engine_squared = np.nan
        try:
            log_engine = float(np.log1p(max(float(engine_size_l),0))) if pd.notna(engine_size_l) else np.nan
        except:
            log_engine = np.nan
        try:
            log_power = float(np.log1p(max(float(power_hp),0))) if pd.notna(power_hp) else np.nan
        except:
            log_power = np.nan

        # age
        try:
            age = 2026 - int(year) if pd.notna(year) else np.nan
        except:
            age = np.nan
        try:
            age_decay = float(np.exp(-max(float(age),0)/10)) if pd.notna(age) else np.nan
        except:
            age_decay = np.nan
        try:
            age_x_engine = float(age)*float(engine_size_l) if pd.notna(age) and pd.notna(engine_size_l) else np.nan
        except:
            age_x_engine = np.nan
        try:
            cyl_x_engine = float(cylinders)*float(engine_size_l) if pd.notna(cylinders) and pd.notna(engine_size_l) else np.nan
        except:
            cyl_x_engine = np.nan
        try:
            engine_x_year = float(engine_size_l)*float(year) if pd.notna(engine_size_l) and pd.notna(year) else np.nan
        except:
            engine_x_year = np.nan
        try:
            is_4cyl = 1 if float(cylinders)==4 else 0 if pd.notna(cylinders) else 0
            is_6cyl = 1 if float(cylinders)==6 else 0 if pd.notna(cylinders) else 0
            is_8cyl_plus = 1 if float(cylinders)>=8 else 0 if pd.notna(cylinders) else 0
        except:
            is_4cyl=is_6cyl=is_8cyl_plus=0
        try:
            year_x_cyl = float(year)*float(cylinders) if pd.notna(year) and pd.notna(cylinders) else np.nan
        except:
            year_x_cyl = np.nan
        try:
            speed_x_accel = float(speed_kmh)*float(acceleration_ms2) if pd.notna(speed_kmh) and pd.notna(acceleration_ms2) else np.nan
        except:
            speed_x_accel = np.nan
        try:
            speed_x_engine = float(speed_kmh)*float(engine_size_l) if pd.notna(speed_kmh) and pd.notna(engine_size_l) else np.nan
        except:
            speed_x_engine = np.nan
        try:
            age_x_mileage = float(vehicle_age)*float(mileage_km) if pd.notna(vehicle_age) and pd.notna(mileage_km) else np.nan
        except:
            age_x_mileage = np.nan
        try:
            temp_x_humidity = float(temperature_c)*float(humidity_pct) if pd.notna(temperature_c) and pd.notna(humidity_pct) else np.nan
        except:
            temp_x_humidity = np.nan
        try:
            speed_per_engine = float(speed_kmh)/(float(engine_size_l)+0.1) if pd.notna(speed_kmh) and pd.notna(engine_size_l) else np.nan
        except:
            speed_per_engine = np.nan

        # Transmission parse
        trans_parsed = parse_transmission(transmission)
        gear_count = int(trans_parsed['gear_count']) if pd.notna(trans_parsed['gear_count']) else 0
        is_automatic = int(trans_parsed['is_automatic']) if pd.notna(trans_parsed['is_automatic']) else 0
        is_cvt = int(trans_parsed['is_cvt']) if pd.notna(trans_parsed['is_cvt']) else 0

        # Drivetrain & turbo
        # need model+drive for drivetrain
        drv_row = {'model': model_raw, 'drive': drive_raw}
        drivetrain_str = parse_drivetrain(drv_row)
        turbo_flag = parse_turbo({'model': model_raw})
        drivetrain_enc = self._drivetrain_enc_lookup(drivetrain_str)

        try:
            power_to_weight = float(power_hp)/(float(engine_size_l)*100+1) if pd.notna(power_hp) and pd.notna(engine_size_l) else np.nan
            if np.isinf(power_to_weight): power_to_weight=np.nan
        except:
            power_to_weight=np.nan
        try:
            power_per_cylinder = float(power_hp)/float(cylinders) if pd.notna(power_hp) and pd.notna(cylinders) and float(cylinders)!=0 else np.nan
            if np.isinf(power_per_cylinder): power_per_cylinder=np.nan
        except:
            power_per_cylinder=np.nan
        try:
            hp_x_disp = float(power_hp)*float(engine_size_l) if pd.notna(power_hp) and pd.notna(engine_size_l) else np.nan
        except:
            hp_x_disp=np.nan

        make_model = f"{str(make_raw).strip()}_{str(model_raw).strip()}"

        # TE lookups (need raw values; if raw is UNKNOWN -> global mean)
        make_te = self._te_lookup('make', make_raw)
        model_te = self._te_lookup('model', model_raw)
        make_model_te = self._te_lookup('make_model', make_model)
        vehicle_class_te = self._te_lookup('vehicle_class', vehicle_class)
        vehicle_class_unified_te = self._te_lookup('vehicle_class_unified', vehicle_class_unified)
        fuel_type_te = self._te_lookup('fuel_type', fuel_type)
        fuel_type_unified_te = self._te_lookup('fuel_type_unified', fuel_type_unified)
        transmission_te = self._te_lookup('transmission', transmission)
        road_type_te = self._te_lookup('road_type', road_type)
        traffic_condition_te = self._te_lookup('traffic_condition', traffic_condition)
        region_te = self._te_lookup('region', region)
        regulatory_cycle_te = self._te_lookup('regulatory_cycle', regulatory_cycle)
        data_source_te = self._te_lookup('data_source', data_source)

        # Build row dict with 57 cols in bundle order
        row = {
            'year': float(year) if pd.notna(year) else np.nan,
            'engine_size_l': float(engine_size_l) if pd.notna(engine_size_l) else np.nan,
            'cylinders': float(cylinders) if pd.notna(cylinders) else np.nan,
            'vehicle_age': float(vehicle_age) if pd.notna(vehicle_age) else np.nan,
            'mileage_km': float(mileage_km) if pd.notna(mileage_km) else np.nan,
            'speed_kmh': float(speed_kmh) if pd.notna(speed_kmh) else np.nan,
            'acceleration_ms2': float(acceleration_ms2) if pd.notna(acceleration_ms2) else np.nan,
            'temperature_c': float(temperature_c) if pd.notna(temperature_c) else np.nan,
            'humidity_pct': float(humidity_pct) if pd.notna(humidity_pct) else np.nan,
            'wind_speed_ms': float(wind_speed_ms) if pd.notna(wind_speed_ms) else np.nan,
            'air_pressure_hpa': float(air_pressure_hpa) if pd.notna(air_pressure_hpa) else np.nan,
            'power_hp': float(power_hp) if pd.notna(power_hp) else np.nan,
            'is_india': float(is_india),
            'has_speed': int(has_speed),
            'has_acceleration': int(has_acceleration),
            'has_weather': int(has_weather),
            'has_road': int(has_road),
            'has_age_mileage': int(has_age_mileage),
            'engine_per_cylinder': float(engine_per_cylinder) if pd.notna(engine_per_cylinder) else np.nan,
            'engine_squared': float(engine_squared) if pd.notna(engine_squared) else np.nan,
            'log_engine': float(log_engine) if pd.notna(log_engine) else np.nan,
            'log_power': float(log_power) if pd.notna(log_power) else np.nan,
            'age': float(age) if pd.notna(age) else np.nan,
            'age_decay': float(age_decay) if pd.notna(age_decay) else np.nan,
            'age_x_engine': float(age_x_engine) if pd.notna(age_x_engine) else np.nan,
            'cyl_x_engine': float(cyl_x_engine) if pd.notna(cyl_x_engine) else np.nan,
            'engine_x_year': float(engine_x_year) if pd.notna(engine_x_year) else np.nan,
            'is_4cyl': int(is_4cyl),
            'is_6cyl': int(is_6cyl),
            'is_8cyl_plus': int(is_8cyl_plus),
            'year_x_cyl': float(year_x_cyl) if pd.notna(year_x_cyl) else np.nan,
            'speed_x_accel': float(speed_x_accel) if pd.notna(speed_x_accel) else np.nan,
            'speed_x_engine': float(speed_x_engine) if pd.notna(speed_x_engine) else np.nan,
            'age_x_mileage': float(age_x_mileage) if pd.notna(age_x_mileage) else np.nan,
            'temp_x_humidity': float(temp_x_humidity) if pd.notna(temp_x_humidity) else np.nan,
            'speed_per_engine': float(speed_per_engine) if pd.notna(speed_per_engine) else np.nan,
            'gear_count': int(gear_count),
            'is_automatic': int(is_automatic),
            'is_cvt': int(is_cvt),
            'turbo_flag': int(turbo_flag),
            'power_to_weight': float(power_to_weight) if pd.notna(power_to_weight) else np.nan,
            'power_per_cylinder': float(power_per_cylinder) if pd.notna(power_per_cylinder) else np.nan,
            'hp_x_disp': float(hp_x_disp) if pd.notna(hp_x_disp) else np.nan,
            'make_te': float(make_te),
            'model_te': float(model_te),
            'make_model_te': float(make_model_te),
            'vehicle_class_te': float(vehicle_class_te),
            'vehicle_class_unified_te': float(vehicle_class_unified_te),
            'fuel_type_te': float(fuel_type_te),
            'fuel_type_unified_te': float(fuel_type_unified_te),
            'transmission_te': float(transmission_te),
            'road_type_te': float(road_type_te),
            'traffic_condition_te': float(traffic_condition_te),
            'region_te': float(region_te),
            'regulatory_cycle_te': float(regulatory_cycle_te),
            'data_source_te': float(data_source_te),
            'drivetrain_enc': int(drivetrain_enc),
        }

        # Build DataFrame in feature_cols order and fill median
        df = pd.DataFrame([row])
        # Ensure all feature_cols present
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_cols]
        # Fill NaN with median, inf handling
        df = df.replace([np.inf, -np.inf], np.nan)
        # fillna with median Series (median indexed by col)
        for col in df.columns:
            if col in self.median.index and pd.isna(df[col].iloc[0]):
                df[col] = self.median[col]
            # still NaN -> 0
        df = df.fillna(0)
        # Determine unseen flag: if any TE used global mean (i.e., raw not in mapping)
        # Simple: check if make/model not in mappings exact
        raw_make_in = str(make_raw).strip() in self.te_mappings.get('make', {})
        raw_make_low = str(make_raw).strip().lower() in self.te_mappings.get('make_lower', {})
        is_unseen_make = not (raw_make_in or raw_make_low)
        raw_model_in = str(model_raw).strip() in self.te_mappings.get('model', {})
        raw_model_low = str(model_raw).strip().lower() in self.te_mappings.get('model_lower', {})
        is_unseen_model = not (raw_model_in or raw_model_low)
        unseen_vehicle = bool(is_unseen_make or is_unseen_model)
        return df, unseen_vehicle, row

    def predict_single(self, req_dict):
        start = time.time()
        df, unseen_vehicle, raw_row = self._preprocess_single(req_dict)
        # Predict log, invert
        try:
            pred_log = self.model.predict(df)
            # catboost predict returns array shape (1,) or float
            if isinstance(pred_log, np.ndarray):
                pred_log = pred_log[0] if pred_log.size==1 else pred_log
            else:
                pred_log = float(pred_log)
                pred_log = np.array([pred_log])
                pred_log = pred_log[0]
        except Exception as e:
            import traceback; traceback.print_exc()
            raise RuntimeError(f"CatBoost predict failed: {e}")
        # Ensure scalar
        if isinstance(pred_log, np.ndarray):
            pred_log_scalar = float(pred_log[0]) if pred_log.ndim>0 else float(pred_log)
        else:
            pred_log_scalar = float(pred_log)
        co2 = float(np.expm1(pred_log_scalar))
        co2 = max(0.0, float(co2))
        # Derived fuel estimate (L/100km); CO2->fuel uses ~2320 g CO2 per litre
        fuel_comb = co2 * 0.04310344827586207  # 100/2320
        if fuel_comb < 0.5:
            fuel_comb = 0.5
        # Split city/hwy using ratios measured from the Canadian reference data
        fuel_city = fuel_comb * self.city_ratio
        fuel_hwy = fuel_comb * self.hwy_ratio
        fuel_comb_mpg = 235.215 / fuel_comb if fuel_comb>0 else 0
        annual_cost_usd = fuel_comb * 15000/100 * self.fuel_price
        # eco score percentile vs train
        try:
            percentile = (self.train_co2_sorted <= co2).mean()*100
            eco_score = int(round(100 - percentile))
            eco_score = max(0, min(100, eco_score))
        except:
            eco_score = 50
        inference_ms = (time.time()-start)*1000
        # Uncertainty from held-out residual spread (per data source / India / global)
        co2_std = float(self._select_residual_std(raw_row))
        if unseen_vehicle:
            co2_std *= 1.1
        fuel_std = float(co2_std * 0.04310344827586207)

        return {
            "fuel_city_l_100km": float(fuel_city),
            "fuel_hwy_l_100km": float(fuel_hwy),
            "fuel_comb_l_100km": float(fuel_comb),
            "fuel_comb_mpg": float(fuel_comb_mpg),
            "co2_g_km": float(co2),
            "co2_g_per_km": float(co2),
            "uncertainty": {
                "fuel_city_std": float(fuel_std*1.1),
                "fuel_hwy_std": float(fuel_std*0.9),
                "fuel_comb_std": float(fuel_std),
                "co2_std": float(co2_std),
            },
            "annual_cost_usd": float(annual_cost_usd),
            "eco_score": int(eco_score),
            "unseen_vehicle": bool(unseen_vehicle),
            "model_version": str(self.model_version),
            "inference_ms": float(inference_ms),
            "is_india": bool(raw_row.get('is_india',0)==1),
            "note": "CatBoost v3 — India R2 -0.19, use with caution for India" if raw_row.get('is_india',0)==1 else ""
        }

    def predict_batch(self, req_list):
        results=[]
        for req in req_list:
            if hasattr(req,'model_dump'):
                d=req.model_dump()
            elif hasattr(req,'dict'):
                d=req.dict()
            else:
                d=dict(req)
            results.append(self.predict_single(d))
        return results

    def _explain_fallback(self, df):
        """Fast heuristic fallback: global importance x normalised deviation."""
        importances = self.model.get_feature_importance()
        contribs = []
        for fname, imp in zip(self.feature_cols, importances):
            try:
                val = float(df[fname].iloc[0])
                med = float(self.median.get(fname, val))
                dev = (val - med) / (abs(med) + 1.0)
                contrib = float(imp) * dev * 0.1
            except Exception:
                contrib = float(imp) * 0.01
            contribs.append({"feature": fname, "contribution": float(contrib)})
        if all(abs(c["contribution"]) < 1e-6 for c in contribs):
            contribs = [{"feature": f, "contribution": float(i)} for f, i in zip(self.feature_cols, importances)]
        contribs = sorted(contribs, key=lambda x: abs(x["contribution"]), reverse=True)[:15]
        return {"contributions": contribs, "base_value": None, "approximate": True}

    def explain_single(self, req_dict):
        """Real CatBoost TreeSHAP contributions for CO2, expressed on a g/km scale.

        Uses the first EXPLAIN_SHAP_TREES boosting trees by default (~3 s).
        Set EXPLAIN_SHAP_TREES=0 for exact SHAP over all trees (~20 s).
        """
        try:
            df, _, _ = self._preprocess_single(req_dict)
            cache_key = json.dumps(
                {col: float(df[col].iloc[0]) for col in self.feature_cols}, sort_keys=True
            )
            cached = self._explain_cache.get(cache_key)
            if cached is not None:
                return copy.deepcopy(cached)
            try:
                from catboost import Pool
                shap_model = self._get_shap_model()
                pool = Pool(df, feature_names=self.feature_cols)
                sv = np.asarray(
                    shap_model.get_feature_importance(pool, type="ShapValues", thread_count=-1)
                )
                contribs_log = sv[0, :-1]
                base_log = float(sv[0, -1])
                pred_log = float(base_log + contribs_log.sum())
                scale = float(np.exp(np.clip(pred_log, -50.0, 50.0)))
                contribs = [
                    {"feature": fname, "contribution": float(c * scale)}
                    for fname, c in zip(self.feature_cols, contribs_log)
                ]
                contribs = sorted(contribs, key=lambda x: abs(x["contribution"]), reverse=True)[:15]
                result = {
                    "contributions": contribs,
                    "base_value": float(np.expm1(base_log)),
                    "approximate": bool(self.explain_tree_limit and self.explain_tree_limit > 0),
                }
            except Exception as e:
                print(f"[CatBoostLoader] SHAP failed ({e}); using heuristic fallback")
                result = self._explain_fallback(df)
            if len(self._explain_cache) >= 64:
                self._explain_cache.pop(next(iter(self._explain_cache)))
            self._explain_cache[cache_key] = copy.deepcopy(result)
            return result
        except Exception as e:
            import traceback; traceback.print_exc()
            return {
                "contributions": [{"feature": "co2_g_per_km", "contribution": 0.0}],
                "base_value": None,
                "approximate": True,
            }

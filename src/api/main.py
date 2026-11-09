"""FuelNet CatBoost API — wired to catboost_v3_bundle/catboost.pt (57 features)"""
import pathlib
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PredictRequest, PredictResponse, HealthResponse, ExplainResponse
from src.api.catboost_predictor import CatBoostLoader

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

loader: CatBoostLoader = None


def get_loader():
    global loader
    if loader is None:
        loader = CatBoostLoader()
    return loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_loader()
    print("CatBoost API ready")
    yield


app = FastAPI(
    title="FuelNet API — CatBoost v3",
    description="CO2 g/km Prediction — CatBoost depth8 25k iter (India + Global + Canada + EPA) | Single target log1p→expm1",
    version="catboost-v3-24998",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    return {"status": "ok"}

@app.get("/model/info", tags=["model"])
def model_info():
    ldr = get_loader()
    info = dict(ldr.metadata)
    info["feature_cols"] = ldr.feature_cols
    info["model_version"] = ldr.model_version
    info["target"] = "co2_g_per_km (log1p -> expm1)"
    info["n_features"] = len(ldr.feature_cols)
    info["known_limitations"] = [
        "India R2 -0.19 (MAE 119) — experimental for India; Canada/US R2 0.97",
        "Single target CO2 only — fuel values are estimates via co2*0.0431 (≈ 2320g/L)",
        "Target encodings with smoothed global-mean fallback for unseen makes/models",
        "Retrain v4 recommended for better India generalization"
    ]
    try:
        per_source = ldr.bundle_dir / "per_source.csv"
        if per_source.exists():
            import csv
            with open(per_source, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                info["per_source"] = list(reader)
        # add metrics from metadata
        info["test_R2"] = ldr.metadata.get("R²") or ldr.metadata.get("test_R2") or 0.887
        info["val_R2"] = ldr.metadata.get("val_R2", 0.89)
        info["india_R2"] = ldr.metadata.get("India_R2", ldr.metadata.get("india_R2", -0.19))
    except Exception as e:
        print(f"per_source load fail {e}")
    return info

@app.get("/options", tags=["options"])
def options():
    ldr = get_loader()
    m = ldr.te_mappings
    makes = m.get("makes", [])
    models_by_make = m.get("models_by_make", {})
    vehicle_classes = m.get("vehicle_class_options", ["Car","SUV","Truck","Van","Bus","Motorcycle","Other"])
    fuel_types = m.get("fuel_type_options", ["Petrol","Diesel","Premium","E85","CNG","Electric","Hybrid","Other"])
    transmissions = m.get("transmission_options", ["A3","A4","A6","A8","A10","M5","M6","AS6","AS8","AV","CVT"])
    # Compact list for legacy frontend fallback
    if not transmissions:
        transmissions = ["A4","A6","A8","M5","M6","AS6","AS8","AM6","AV","AV6","CVT"]
    return {
        "makes": sorted(makes, key=lambda x: x.lower()) if makes else [],
        "models_by_make": models_by_make,
        "models_by_make_lower": m.get("models_by_make_lower", {}),
        "vehicle_classes": vehicle_classes,
        "vehicle_class_unified_options": m.get("vehicle_class_unified_options", []),
        "fuel_types": fuel_types,
        "fuel_type_unified_options": m.get("fuel_type_unified_options", []),
        "transmissions": transmissions,
        "road_types": m.get("road_type_options", ["City","Highway","Rural","Unknown"]),
        "traffic_conditions": m.get("traffic_condition_options", ["Free flow","Moderate","Heavy","Unknown"]),
        "regions": m.get("region_options", ["India","Canada","US","Global","Unknown"]),
        "regulatory_cycles": m.get("regulatory_cycle_options", ["ARAI","EPA","NEDC","Mixed","Unknown"]),
        "data_sources": m.get("data_source_options", ["India_Vehicle_Emission","EPA_FuelEconomy","Kaggle_Canada","Canada_1995_2014"]),
        "feature_cols": ldr.feature_cols,
    }

@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(req: PredictRequest):
    ldr = get_loader()
    try:
        d = req.model_dump()
        result = ldr.predict_single(d)
        return result
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict", tags=["predict"])
def batch_predict(requests: List[PredictRequest]):
    ldr = get_loader()
    if len(requests) > 500:
        raise HTTPException(status_code=422, detail="Batch size must be <= 500")
    if len(requests) == 0:
        raise HTTPException(status_code=422, detail="Batch must not be empty")
    try:
        results = ldr.predict_batch(requests)
        return results
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain", response_model=ExplainResponse, tags=["explain"])
def explain(req: PredictRequest):
    ldr = get_loader()
    try:
        d = req.model_dump()
        return ldr.explain_single(d)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain_single", include_in_schema=False)
def explain_alias(req: PredictRequest):
    return explain(req)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

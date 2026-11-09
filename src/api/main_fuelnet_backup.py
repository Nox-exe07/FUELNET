"""FuelNet FastAPI — Phase 6"""
import pathlib
import json
import time
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from src.api.schemas import PredictRequest, PredictResponse, Uncertainty, HealthResponse
from src.api.predictor import ModelLoader

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

app = FastAPI(
    title="FuelNet API",
    description="Fuel Consumption & CO2 Prediction Platform — Canadian NRCan",
    version="fuelnet-t-v1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global loader
loader: ModelLoader = None

def get_loader():
    global loader
    if loader is None:
        loader = ModelLoader()
        print("FuelNet API ready")
    return loader

@app.on_event("startup")
def startup_event():
    global loader
    loader = ModelLoader()
    print("FuelNet API ready")

@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    return {"status": "ok"}

@app.get("/model/info", tags=["model"])
def model_info():
    ldr = get_loader()
    return ldr.model_card

@app.get("/options", tags=["options"])
def options():
    """Return makes/models/vehicle_classes for frontend cascading dropdowns"""
    ldr = get_loader()
    # Provide curated options: use mappings and preprocessor categories
    makes = sorted(ldr.make_mapping.keys())
    # Remove unknown for dropdown
    makes_clean = [m for m in makes if m != 'unknown']
    # For models_by_make, we need to infer from training data? We'll provide top models per make from processed_data
    # Load processed_data to build make->model mapping (cleaned names)
    try:
        import pickle
        with open(ldr.processed_path, 'rb') as f:
            data = pickle.load(f)
        # train_pool df has cleaned make/model, use groups
        # Reconstruct make->model via unique groups in train_pool
        groups = data['groups']  # array of "make model"
        import collections
        make_to_models = collections.defaultdict(set)
        for g in groups:
            parts = g.split(" ", 1)
            if len(parts) == 2:
                mk, md = parts
                make_to_models[mk].add(md)
        # Convert to sorted lists
        models_by_make = {k: sorted(list(v))[:50] for k, v in make_to_models.items()}
    except Exception as e:
        models_by_make = {}
        import traceback
        print(f"Failed to build models_by_make: {e}")

    # Vehicle classes and fuel types from preprocessor
    try:
        # Use ohe feature names to extract categories? But easier to read from preprocessor categorical modes? Use original unique?
        # Get vehicle_class categories from ohe categories
        vehicle_classes = []
        fuel_types = []
        transmissions = []
        # ohe.categories_ is list per column in order OHE_COLS
        cats = ldr.ohe.categories_
        # Order is ['vehicle_class','transmission_type','fuel_type']
        if len(cats) >= 3:
            vehicle_classes = sorted(cats[0].tolist())
            # transmission_type is automatic/manual, but we want transmission codes like A6, M5 for input
            # Provide typical transmission codes from original data unique
            transmissions = ["A4","A6","A8","M5","M6","AS6","AS8","AM6","AV","AV6"]
            fuel_types = sorted(cats[2].tolist())
    except Exception as e:
        vehicle_classes = ["Compact","SUV","Mid-size","Subcompact","Pickup truck: Small"]
        fuel_types = ["X","Z","D","E"]
        transmissions = ["A4","A6","M5"]

    return {
        "makes": makes_clean,
        "models_by_make": models_by_make,
        "vehicle_classes": vehicle_classes,
        "fuel_types": fuel_types,
        "transmissions": transmissions
    }

@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(req: PredictRequest):
    ldr = get_loader()
    try:
        d = req.model_dump()
        result = ldr.predict_single(d, n_stochastic=20)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict", tags=["predict"])
def batch_predict(requests: List[PredictRequest]):
    ldr = get_loader()
    if len(requests) > 500:
        raise HTTPException(status_code=422, detail="Batch size must be <= 500")
    if len(requests) == 0:
        raise HTTPException(status_code=422, detail="Batch must not be empty")
    try:
        results = ldr.predict_batch(requests, n_stochastic=5)
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain", tags=["explain"])
def explain(req: PredictRequest):
    ldr = get_loader()
    try:
        d = req.model_dump()
        contribs = ldr.explain_single(d)
        return {"contributions": contribs}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# For compatibility with spec's POST /explain expecting ExplainRequest
@app.post("/explain_single", include_in_schema=False)
def explain_alias(req: PredictRequest):
    return explain(req)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

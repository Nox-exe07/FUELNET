"""Pydantic v2 schemas for FuelNet API — CatBoost v3 (CO2 g/km single target)"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PredictRequest(BaseModel):
    year: int = Field(..., ge=1995, le=2026, description="Model year 1995-2026", examples=[2020])
    make: str = Field(..., min_length=1, max_length=50, description="Vehicle make", examples=["Toyota"])
    model: str = Field(..., min_length=1, max_length=50, description="Vehicle model", examples=["Corolla"])
    vehicle_class: str = Field(..., min_length=1, description="Vehicle class", examples=["Compact"])
    engine_size_l: float = Field(..., ge=0.5, le=8.0, description="Engine size in litres", examples=[1.8])
    cylinders: int = Field(..., ge=2, le=16, description="Number of cylinders", examples=[4])
    transmission: str = Field(..., min_length=1, description="Transmission code (e.g., A6, M5, AS6)", examples=["A6"])
    fuel_type: str = Field(..., min_length=1, description="Fuel type code (D,E,N,X,Z)", examples=["X"])
    # CatBoost advanced optional fields (median imputed if omitted)
    power_hp: Optional[float] = Field(None, ge=0, le=2000, description="Power HP")
    mileage_km: Optional[float] = Field(None, ge=0, description="Mileage km")
    speed_kmh: Optional[float] = Field(None, ge=0, le=300, description="Avg speed km/h")
    acceleration_ms2: Optional[float] = Field(None, description="Acceleration m/s2")
    temperature_c: Optional[float] = Field(None, description="Ambient temp C")
    humidity_pct: Optional[float] = Field(None, ge=0, le=100, description="Humidity %")
    wind_speed_ms: Optional[float] = Field(None, ge=0, description="Wind m/s")
    air_pressure_hpa: Optional[float] = Field(None, description="Air pressure hPa")
    vehicle_age: Optional[float] = Field(None, ge=0, description="Vehicle age years (auto 2026-year if omitted)")
    road_type: Optional[str] = Field(None, description="Road type (City/Highway/Rural)")
    traffic_condition: Optional[str] = Field(None, description="Traffic (Free flow/Moderate/Heavy)")
    region: Optional[str] = Field(None, description="Region (India/Canada/US/Global)")
    regulatory_cycle: Optional[str] = Field(None, description="Regulatory cycle (ARAI/EPA/NEDC/Mixed)")
    data_source: Optional[str] = Field(None, description="Data source tag")
    drive: Optional[str] = Field(None, description="Drive (FWD/RWD/AWD/UNK)")
    is_india: Optional[int] = Field(None, ge=0, le=1, description="1=India, 0=non-India (auto from region if omitted)")

class Uncertainty(BaseModel):
    fuel_city_std: float
    fuel_hwy_std: float
    fuel_comb_std: float
    co2_std: float

class PredictResponse(BaseModel):
    fuel_city_l_100km: float
    fuel_hwy_l_100km: float
    fuel_comb_l_100km: float
    fuel_comb_mpg: float
    co2_g_km: float
    # alias for catboost single-target
    co2_g_per_km: Optional[float] = None
    uncertainty: Uncertainty
    annual_cost_usd: float
    eco_score: int = Field(..., ge=0, le=100, description="0-100, higher = greener")
    unseen_vehicle: bool
    model_version: str
    inference_ms: float
    is_india: Optional[bool] = None
    note: Optional[str] = None

class BatchPredictRequest(BaseModel):
    vehicles: List[PredictRequest] = Field(..., max_length=500)

# For /explain endpoint — same request but returns SHAP contributions
class ExplainRequest(PredictRequest):
    pass

class SHAPContribution(BaseModel):
    feature: str
    contribution: float

class ExplainResponse(BaseModel):
    contributions: List[SHAPContribution]
    base_value: Optional[float] = None
    approximate: bool = False

class OptionsResponse(BaseModel):
    makes: List[str]
    models_by_make: Dict[str, List[str]]
    vehicle_classes: List[str]
    fuel_types: List[str]
    transmissions: List[str]

class HealthResponse(BaseModel):
    status: str

class ModelInfoResponse(BaseModel):
    model_version: str
    training_date: str
    dataset: dict
    architecture: dict
    test_metrics: dict
    known_limitations: List[str]

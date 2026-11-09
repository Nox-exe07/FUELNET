import pytest
from pydantic import ValidationError

from src.api.schemas import PredictRequest


def valid_payload(**overrides):
    payload = {
        "year": 2020,
        "make": "Toyota",
        "model": "Corolla",
        "vehicle_class": "Compact",
        "engine_size_l": 1.8,
        "cylinders": 4,
        "transmission": "A6",
        "fuel_type": "X",
    }
    payload.update(overrides)
    return payload


def test_valid_request_defaults():
    req = PredictRequest(**valid_payload())
    assert req.year == 2020
    assert req.make == "Toyota"
    assert req.power_hp is None
    assert req.is_india is None


def test_optional_fields_accepted():
    req = PredictRequest(
        **valid_payload(
            power_hp=180,
            mileage_km=50000,
            speed_kmh=80,
            temperature_c=22,
            humidity_pct=55,
            road_type="Highway",
            region="India",
            is_india=1,
        )
    )
    assert req.power_hp == 180
    assert req.is_india == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("year", 1900),
        ("year", 2050),
        ("engine_size_l", 0.1),
        ("engine_size_l", 9.0),
        ("cylinders", 1),
        ("cylinders", 20),
        ("humidity_pct", 101),
        ("speed_kmh", 400),
    ],
)
def test_invalid_ranges_rejected(field, value):
    with pytest.raises(ValidationError):
        PredictRequest(**valid_payload(**{field: value}))


def test_blank_make_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(**valid_payload(make=""))

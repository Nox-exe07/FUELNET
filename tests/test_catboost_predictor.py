import math

import pytest


def test_feature_columns_and_row_shape(loader, sample_request):
    df, unseen, _ = loader._preprocess_single(sample_request)
    assert len(loader.feature_cols) == 57
    assert list(df.columns) == loader.feature_cols
    assert df.shape == (1, 57)
    assert not df.isna().any().any()
    assert unseen is False


def test_unseen_vehicle_flagged(loader):
    df, unseen, _ = loader._preprocess_single(
        {
            "year": 2021,
            "make": "ZZZ-Not-A-Brand",
            "model": "ZZZ-Not-A-Model",
            "vehicle_class": "Car",
            "engine_size_l": 2.0,
            "cylinders": 4,
            "transmission": "A6",
            "fuel_type": "X",
        }
    )
    assert df.shape == (1, 57)
    assert unseen is True


def test_prediction_is_plausible(loader, sample_request):
    r = loader.predict_single(sample_request)
    for key in ("fuel_city_l_100km", "fuel_hwy_l_100km", "fuel_comb_l_100km", "co2_g_km"):
        assert math.isfinite(r[key])
        assert r[key] > 0
    assert 0 < r["co2_g_km"] < 800
    assert 0 <= r["eco_score"] <= 100
    assert r["fuel_comb_mpg"] > 0
    assert r["annual_cost_usd"] > 0
    assert r["inference_ms"] >= 0


def test_uncertainty_is_positive(loader, sample_request):
    u = loader.predict_single(sample_request)["uncertainty"]
    assert u["co2_std"] > 0
    assert u["fuel_comb_std"] > 0


def test_fuel_split_uses_measured_ratios(loader, sample_request):
    r = loader.predict_single(sample_request)
    assert r["fuel_city_l_100km"] == pytest.approx(r["fuel_comb_l_100km"] * loader.city_ratio, rel=1e-6)
    assert r["fuel_hwy_l_100km"] == pytest.approx(r["fuel_comb_l_100km"] * loader.hwy_ratio, rel=1e-6)


def test_larger_vehicle_emits_more(loader):
    base = {
        "year": 2020,
        "vehicle_class": "Car",
        "transmission": "A6",
        "fuel_type": "X",
        "region": "Canada",
    }
    small = loader.predict_single(
        {**base, "make": "Toyota", "model": "Corolla", "engine_size_l": 1.8, "cylinders": 4}
    )
    large = loader.predict_single(
        {
            **base,
            "make": "Ford",
            "model": "F150",
            "vehicle_class": "Truck",
            "engine_size_l": 3.5,
            "cylinders": 6,
            "transmission": "A10",
        }
    )
    assert large["co2_g_km"] > small["co2_g_km"]


def test_batch_matches_single(loader, sample_request):
    results = loader.predict_batch([sample_request, sample_request])
    assert len(results) == 2
    assert results[0]["co2_g_km"] == results[1]["co2_g_km"]


def test_explain_returns_shap_contributions(loader, sample_request):
    result = loader.explain_single(sample_request)
    assert set(result) >= {"contributions", "base_value", "approximate"}
    contribs = result["contributions"]
    assert isinstance(contribs, list) and contribs
    assert len(contribs) <= 15
    for item in contribs:
        assert "feature" in item and "contribution" in item
        assert math.isfinite(item["contribution"])
    assert result["base_value"] is not None
    assert math.isfinite(result["base_value"])


def test_explain_cached_is_stable(loader, sample_request):
    first = loader.explain_single(sample_request)
    second = loader.explain_single(sample_request)
    assert first["contributions"] == second["contributions"]

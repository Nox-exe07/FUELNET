def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_options(client):
    r = client.get("/options")
    assert r.status_code == 200
    body = r.json()
    assert body["makes"]
    assert body["models_by_make"]
    assert body["vehicle_classes"]
    assert body["fuel_types"]
    assert body["transmissions"]


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["n_features"] == 57
    assert body["target"].startswith("co2_g_per_km")
    assert body["model_version"].startswith("catboost-v3")
    assert "known_limitations" in body


def test_predict_valid(client, sample_request):
    r = client.post("/predict", json=sample_request)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["co2_g_km"] > 0
    assert body["co2_g_per_km"] == body["co2_g_km"]
    assert body["fuel_comb_l_100km"] > 0
    assert 0 <= body["eco_score"] <= 100
    assert "uncertainty" in body


def test_predict_invalid_returns_422(client, sample_request):
    bad = dict(sample_request, year=1900)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_batch_predict_valid(client, sample_request):
    r = client.post("/batch-predict", json=[sample_request, sample_request])
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 2
    assert all(item["co2_g_km"] > 0 for item in body)


def test_batch_predict_empty_returns_422(client):
    r = client.post("/batch-predict", json=[])
    assert r.status_code == 422


def test_batch_predict_oversized_returns_422(client, sample_request):
    r = client.post("/batch-predict", json=[sample_request] * 501)
    assert r.status_code == 422


def test_explain_valid(client, sample_request):
    r = client.post("/explain", json=sample_request)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["contributions"]
    assert isinstance(body["contributions"][0]["feature"], str)
    assert isinstance(body["contributions"][0]["contribution"], (int, float))
    assert body["base_value"] is not None
    assert body["approximate"] is True

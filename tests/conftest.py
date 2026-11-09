import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE = {
    "year": 2020,
    "make": "Toyota",
    "model": "Corolla",
    "vehicle_class": "Compact",
    "engine_size_l": 1.8,
    "cylinders": 4,
    "transmission": "A6",
    "fuel_type": "X",
    "region": "Canada",
}


@pytest.fixture(scope="session")
def sample_request():
    return dict(SAMPLE)


@pytest.fixture(scope="session")
def loader():
    """Shared CatBoost loader (model loads once for the whole session)."""
    import src.api.main as main

    if main.loader is None:
        from src.api.catboost_predictor import CatBoostLoader

        main.loader = CatBoostLoader()
    return main.loader


@pytest.fixture(scope="session")
def client(loader):
    """FastAPI TestClient with lifespan/startup executed once."""
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as c:
        yield c

"""Modal deployment for the FuelNet FastAPI + CatBoost web app.

Deploy with:
    modal deploy deploy/modal/modal_app.py

The model is downloaded from the GitHub Release during image build so it is
not uploaded from the local workspace.
"""
from __future__ import annotations

import pathlib

import modal

ROOT = (
    pathlib.Path(__file__).resolve().parents[2]
    if modal.is_local()
    else pathlib.Path("/root")
)

app = modal.App("fuelnet")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ca-certificates")
    .pip_install_from_requirements(str(ROOT / "requirements-api.txt"))
    .add_local_dir(
        str(ROOT / "src"),
        remote_path="/root/src",
        ignore=["**/__pycache__", "**/*.pyc"],
        copy=True,
    )
    .add_local_dir(str(ROOT / "configs"), remote_path="/root/configs", copy=True)
    .add_local_dir(
        str(ROOT / "catboost_v3_bundle"),
        remote_path="/root/catboost_v3_bundle",
        ignore=["catboost.cbm", "catboost.pt", "*.zip"],
        copy=True,
    )
    .add_local_file(
        str(ROOT / "prepared_dataset" / "prepared_data" / "feature_cols.json"),
        remote_path="/root/prepared_dataset/prepared_data/feature_cols.json",
        copy=True,
    )
    .add_local_file(
        str(ROOT / "prepared_dataset" / "prepared_data" / "feature_median.pkl"),
        remote_path="/root/prepared_dataset/prepared_data/feature_median.pkl",
        copy=True,
    )
    .add_local_file(
        str(ROOT / "prepared_dataset" / "prepared_data" / "y_train_orig.npy"),
        remote_path="/root/prepared_dataset/prepared_data/y_train_orig.npy",
        copy=True,
    )
    .add_local_file(
        str(ROOT / "prepared_dataset" / "prepared_data" / "metadata.json"),
        remote_path="/root/prepared_dataset/prepared_data/metadata.json",
        copy=True,
    )
    .add_local_file(
        str(ROOT / "data" / "Fuel_Consumption_2000-2022.csv"),
        remote_path="/root/data/Fuel_Consumption_2000-2022.csv",
        copy=True,
    )
    .add_local_dir(
        str(ROOT / "frontend" / "dist"),
        remote_path="/root/frontend/dist",
        copy=True,
    )
    .add_local_file(
        str(ROOT / "scripts" / "fetch_model.py"),
        remote_path="/root/scripts/fetch_model.py",
        copy=True,
    )
    .env(
        {
            "CATBOOST_BUNDLE_DIR": "catboost_v3_bundle",
            "PREPARED_DATA_DIR": "prepared_dataset/prepared_data",
            "FRONTEND_DIST": "frontend/dist",
            "EXPLAIN_SHAP_TREES": "3000",
            "FUEL_PRICE": "1.50",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .run_commands("cd /root && python scripts/fetch_model.py")
) if modal.is_local() else modal.Image.debian_slim()


@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=600,
    startup_timeout=900,
    scaledown_window=900,
    max_containers=1,
)
@modal.asgi_app()
def web():
    from src.api.main import app as fastapi_app

    return fastapi_app

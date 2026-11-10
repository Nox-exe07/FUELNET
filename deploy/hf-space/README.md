---
title: FuelNet
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Vehicle CO2 and fuel consumption prediction demo
---

# FuelNet — Hugging Face Space

This Space runs the same FuelNet container as the local demo:

- FastAPI backend
- React dashboard served from `frontend/dist`
- CatBoost v3 model downloaded from the GitHub Release during the Docker build
- Public API docs at `/docs`

## Deploy from the repository

From the repository root:

```bat
venv\Scripts\python.exe -m pip install huggingface_hub
set HF_TOKEN=hf_...
venv\Scripts\python.exe scripts\deploy_hf_space.py --space FuelNet
```

The script creates the Docker Space if needed, assembles a clean deployment folder, and uploads it.
It does not include the 100 MB model in the Space repository; the Docker build downloads it from
the GitHub Release.

If the Space build cannot reach GitHub, upload `catboost.cbm` to the Space files and set the Docker
build to copy it instead of downloading it.

## Local equivalent

```bat
scripts\run_demo.bat
```

Open http://127.0.0.1:8000.

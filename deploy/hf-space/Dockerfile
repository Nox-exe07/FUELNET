FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CATBOOST_BUNDLE_DIR=catboost_v3_bundle \
    PREPARED_DATA_DIR=prepared_dataset/prepared_data \
    FRONTEND_DIST=frontend/dist \
    EXPLAIN_SHAP_TREES=3000 \
    PORT=8000

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY catboost_v3_bundle ./catboost_v3_bundle
COPY prepared_dataset/prepared_data ./prepared_dataset/prepared_data
COPY data/Fuel_Consumption_2000-2022.csv ./data/Fuel_Consumption_2000-2022.csv
COPY frontend/dist ./frontend/dist
COPY .env.example .env.example

ARG MODEL_URL=https://github.com/Nox-exe07/FUELNET/releases/download/v1.0-demo/catboost.cbm
RUN python scripts/fetch_model.py --url "$MODEL_URL"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/health', timeout=3)"

CMD ["sh", "-c", "python -m uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

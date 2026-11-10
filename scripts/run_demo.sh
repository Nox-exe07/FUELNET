#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[FuelNet] python3 not found. Install Python 3.11+ first." >&2
  exit 1
fi

if [ ! -x venv/bin/python ]; then
  echo "[FuelNet] Creating virtual environment..."
  python3 -m venv venv
fi

VPY="venv/bin/python"
if ! "$VPY" -c "import catboost, fastapi, uvicorn, pandas, numpy, dotenv" >/dev/null 2>&1; then
  echo "[FuelNet] Installing API dependencies..."
  "$VPY" -m pip install --upgrade pip
  "$VPY" -m pip install -r requirements-api.txt
fi

echo "[FuelNet] Checking model weights..."
"$VPY" scripts/fetch_model.py

if [ ! -f frontend/dist/index.html ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "[FuelNet] Building frontend..."
    (cd frontend && npm ci && npm run build)
  else
    echo "[FuelNet] frontend/dist is missing and npm is not installed." >&2
    exit 1
  fi
fi

export FRONTEND_DIST="${FRONTEND_DIST:-frontend/dist}"
export EXPLAIN_SHAP_TREES="${EXPLAIN_SHAP_TREES:-3000}"
export PYTHONIOENCODING=utf-8

echo "[FuelNet] Starting at http://127.0.0.1:8000 ..."
exec "$VPY" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

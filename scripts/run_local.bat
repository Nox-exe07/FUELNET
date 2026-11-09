@echo off
setlocal
set "ROOT=%~dp0.."
pushd "%ROOT%"

if not exist "venv\Scripts\python.exe" (
  echo [FuelNet] venv not found. Create it and install deps first:
  echo   python -m venv venv
  echo   venv\Scripts\python.exe -m pip install -r requirements.txt
  popd
  exit /b 1
)

if not exist "frontend\node_modules" (
  echo [FuelNet] Installing frontend dependencies...
  pushd frontend
  call npm install
  popd
)

echo [FuelNet] Starting API on http://127.0.0.1:8000 ...
start "FuelNet API" cmd /k "venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload"

echo [FuelNet] Starting web UI on http://127.0.0.1:5173 ...
start "FuelNet Web" cmd /k "cd frontend && npm run dev"

popd
endlocal

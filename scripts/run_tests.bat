@echo off
setlocal
set "ROOT=%~dp0.."
pushd "%ROOT%"

if not exist "venv\Scripts\python.exe" (
  echo [FuelNet] venv not found. Run scripts\run_demo.bat first.
  popd
  endlocal
  exit /b 1
)

venv\Scripts\python.exe -c "import pytest, httpx" >nul 2>nul
if errorlevel 1 (
  echo [FuelNet] Installing test dependencies...
  venv\Scripts\python.exe -m pip install -r requirements-test.txt
  if errorlevel 1 (
    popd
    endlocal
    exit /b 1
  )
)

venv\Scripts\python.exe scripts\fetch_model.py
if errorlevel 1 (
  popd
  endlocal
  exit /b 1
)

venv\Scripts\python.exe -m pytest -q %*
popd
endlocal

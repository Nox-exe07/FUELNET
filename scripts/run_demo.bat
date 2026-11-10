@echo off
setlocal
set "ROOT=%~dp0.."
pushd "%ROOT%"

set "PYTHON="
where python >nul 2>nul
if not errorlevel 1 set "PYTHON=python"
if not defined PYTHON (
  where py >nul 2>nul
  if not errorlevel 1 set "PYTHON=py -3"
)
if not defined PYTHON (
  echo [FuelNet] Python 3 was not found. Install Python 3.11+ from https://www.python.org/downloads/
  goto :error
)

if not exist "venv\Scripts\python.exe" (
  echo [FuelNet] Creating virtual environment...
  %PYTHON% -m venv venv
  if errorlevel 1 goto :error
)

set "VPY=venv\Scripts\python.exe"

%VPY% -c "import catboost, fastapi, uvicorn, pandas, numpy, dotenv" >nul 2>nul
if errorlevel 1 (
  echo [FuelNet] Installing API dependencies...
  %VPY% -m pip install --upgrade pip
  %VPY% -m pip install -r requirements-api.txt
  if errorlevel 1 goto :error
)

echo [FuelNet] Checking model weights...
%VPY% scripts\fetch_model.py
if errorlevel 1 (
  echo [FuelNet] Could not obtain the model. See the message above.
  goto :error
)

if not exist "frontend\dist\index.html" (
  echo [FuelNet] Built frontend not found. Trying to build it with npm...
  where npm >nul 2>nul
  if errorlevel 1 (
    echo [FuelNet] npm is not installed and frontend\dist is missing.
    goto :error
  )
  pushd frontend
  call npm ci
  if errorlevel 1 (
    call npm install
    if errorlevel 1 (
      popd
      goto :error
    )
  )
  call npm run build
  if errorlevel 1 (
    popd
    goto :error
  )
  popd
)

if not defined EXPLAIN_SHAP_TREES set "EXPLAIN_SHAP_TREES=3000"
set "FRONTEND_DIST=frontend\dist"
set "PYTHONIOENCODING=utf-8"

netstat -ano | findstr /r /c:":8000 .*LISTENING" >nul 2>nul
if not errorlevel 1 (
  echo [FuelNet] Port 8000 is already listening. Opening http://127.0.0.1:8000 ...
  start "" http://127.0.0.1:8000
  goto :done
)

echo [FuelNet] Starting FuelNet at http://127.0.0.1:8000 ...
start "FuelNet Server" cmd /k "%VPY% -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000"

echo [FuelNet] Waiting for the server to start...
timeout /t 5 /nobreak >nul
start "" http://127.0.0.1:8000
echo [FuelNet] The browser should open shortly. Close the "FuelNet Server" window to stop.
goto :done

:error
echo.
echo [FuelNet] Setup failed. Read the error above.
pause
popd
endlocal
exit /b 1

:done
popd
endlocal

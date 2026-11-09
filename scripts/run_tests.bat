@echo off
setlocal
set "ROOT=%~dp0.."
pushd "%ROOT%"
venv\Scripts\python.exe -m pytest -q %*
popd
endlocal

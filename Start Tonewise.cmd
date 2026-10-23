@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set "TONEWISE_PYTHON=.venv\Scripts\python.exe"
) else (
  set "TONEWISE_PYTHON=python"
)
echo Starting Tonewise at http://127.0.0.1:8000
"%TONEWISE_PYTHON%" scripts\serve.py
pause

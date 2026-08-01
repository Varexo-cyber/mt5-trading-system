@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-live\Scripts\python.exe" (
  echo Missing .venv-live. Install with: py -m venv .venv-live
  echo Then: .venv-live\Scripts\python.exe -m pip install -e ".[live,dashboard]"
  pause
  exit /b 1
)
".venv-live\Scripts\python.exe" scripts\launch_dashboard.py
endlocal

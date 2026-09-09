@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-live\Scripts\python.exe" (
  echo ERROR: .venv-live\Scripts\python.exe not found.
  exit /b 1
)
.venv-live\Scripts\python.exe -m scripts.check_market_data
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%

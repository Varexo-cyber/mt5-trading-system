@echo off
setlocal
cd /d "%~dp0"
set "DAYS=%~1"
if not defined DAYS set "DAYS=180"
if not exist ".venv-live\Scripts\python.exe" (
  echo ERROR: Run update.cmd first. Python runtime is missing.
  pause
  exit /b 1
)
echo US30: impulse_retest and order_block_fast, M1 and M5 separately.
echo Historical Jarvis replay only. MT5 must be running and logged in.
".venv-live\Scripts\python.exe" -u -m scripts.replay_requested_markets us30 "%DAYS%"
if errorlevel 1 (
  echo ERROR: Replay failed. Read the error above.
  pause
  exit /b 1
)
echo Results: runtime\us30-m1-%DAYS%.csv and runtime\us30-m5-%DAYS%.csv
pause

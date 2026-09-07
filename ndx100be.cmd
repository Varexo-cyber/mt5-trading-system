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
echo NDX100 S5 M5: fixed stop versus configured break-even.
echo Two independent historical runs. Live configuration is unchanged.
".venv-live\Scripts\python.exe" -u -m scripts.replay_requested_markets s5 "%DAYS%"
if errorlevel 1 (
  echo ERROR: Replay failed. Read the error above.
  pause
  exit /b 1
)
echo Results: runtime\ndx100-s5-fixed-%DAYS%.csv and runtime\ndx100-s5-break-even-%DAYS%.csv
pause

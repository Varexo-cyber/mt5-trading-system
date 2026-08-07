@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Have these five theories ever worked?
echo  ============================================
echo.
echo  Replays every playbook over real MT5 bar history, one closed M5 bar
echo  at a time, and reports what each would have returned on its own.
echo.
echo  No orders. No Claude API. Nothing written anywhere. It only costs time,
echo  and on 90 days of four symbols it takes a while - leave it running.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

if "%~1"=="" (
  ".venv-live\Scripts\python.exe" scripts\backtest_playbooks.py --days 90
) else (
  ".venv-live\Scripts\python.exe" scripts\backtest_playbooks.py %*
)

echo.
pause
endlocal

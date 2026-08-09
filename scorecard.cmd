@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Scorecard - where this account makes money
echo   and where it loses it
echo  ============================================
echo.
echo  Journal read-only. No orders, no Claude API, nothing written.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

if "%~1"=="" (
  ".venv-live\Scripts\python.exe" scripts\scorecard.py --days 30
) else (
  ".venv-live\Scripts\python.exe" scripts\scorecard.py %*
)

echo.
pause
endlocal

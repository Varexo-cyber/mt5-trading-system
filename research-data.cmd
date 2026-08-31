@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ==========================================================
echo   Build the reusable MT5 research database
echo  ==========================================================
echo.
echo  One SQLite file containing closed OHLCV bars, historical spread,
echo  broker contract specifications and the exact configuration snapshot.
echo  Safe and read-only against MT5: no orders, no Claude, no live changes.
echo.
echo  Default: 180 evaluation days, EUR 203 research capital,
echo  16 core markets, M1 M5 M15 M30 H1 H4.
echo.
echo  The command is resumable. If MT5 or Windows interrupts it, run it again.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

set DAYS=%~1
if "%DAYS%"=="" set DAYS=180

".venv-live\Scripts\python.exe" scripts\capture_research_data.py --days %DAYS% --equity 203

if errorlevel 1 (
  echo.
  echo  Capture is incomplete. Read the ERROR lines above, fix MT5 history
  echo  availability if needed, and run this same command again to resume.
) else (
  echo.
  echo  Ready: runtime\research\market-history.sqlite3
)

echo.
pause
endlocal

@echo off
setlocal

if not defined SEPTEMBER_ROOT set SEPTEMBER_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\september2025-goud-actief.cmd" >nul
  call "%TEMP%\september2025-goud-actief.cmd" __uittemp
  exit /b
)

cd /d "%SEPTEMBER_ROOT%"

echo.
echo  ========================================================================
echo   HELE KALENDERMAAND SEPTEMBER 2025 - ALLEEN S6 EN S10 GOUD
echo  ========================================================================
echo.
echo   Exact 2025-09-01 00:00 t/m 2025-09-30 23:59:59 UTC.
echo   Geen NDX100, BTC of failed breakout. Niets live wordt veranderd.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe ontbreekt. Draai eerst update.cmd.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections ^
  --start-date 2025-09-01 ^
  --end-date 2025-09-30 ^
  --only section_six_gold_m5,section_ten_gold_m1 ^
  --section-markets ^
  --jarvis-replay ^
  --csv runtime\september-2025-goud.csv

echo.
echo  Uitvoer: runtime\september-2025-goud.csv
echo.
pause

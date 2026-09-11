@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ========================================================================
echo   WAT HEBBEN WINNAARS EN VERLIEZERS GEMEEN? - GEEN NIEUWE REPLAY
echo  ========================================================================
echo.

if not exist "runtime\hoeveel-goud-360.csv" (
  echo  FOUT: runtime\hoeveel-goud-360.csv ontbreekt.
  echo  Draai eerst goud360.cmd.
  pause
  exit /b 1
)

.venv-live\Scripts\python.exe -m scripts.trade_outcome_analysis runtime\hoeveel-goud-360.csv

echo.
echo  Dit las alleen de bestaande 360-daagse goudmeting; niets live gewijzigd.
echo.
pause

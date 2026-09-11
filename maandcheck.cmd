@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ========================================================================
echo   WELKE SECTIE MAAKTE DE RODE MAANDEN? - GEEN NIEUWE REPLAY
echo  ========================================================================
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe ontbreekt. Draai eerst update.cmd.
  pause
  exit /b 1
)

set BESTANDEN=
if exist "runtime\hoeveel.csv" set BESTANDEN=%BESTANDEN% "runtime\hoeveel.csv"
if exist "runtime\hoeveel-stapel2.csv" set BESTANDEN=%BESTANDEN% "runtime\hoeveel-stapel2.csv"
if exist "runtime\hoeveel-goud-360.csv" set BESTANDEN=%BESTANDEN% "runtime\hoeveel-goud-360.csv"

if not defined BESTANDEN (
  echo  Geen hoeveel-CSV gevonden in runtime.
  echo  Draai eerst hoeveel.cmd; bestaande bestanden worden direct hergebruikt.
  pause
  exit /b 1
)

.venv-live\Scripts\python.exe -m scripts.month_attribution %BESTANDEN%

echo.
echo  Dit veranderde NIETS live en haalde GEEN nieuwe MT5-bars op.
echo.
pause

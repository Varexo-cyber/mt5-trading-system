@echo off
setlocal
cd /d "%~dp0"

set DAGEN=180
if not "%~1"=="" set DAGEN=%~1

echo.
echo  S15-S17 BTCUSD -- TWEEDE METING MET JARVIS ACCOUNTREGELS
echo  Alleen S15 M1, S16 M5 en S17 M15. Geen echt geld.
echo  %DAGEN% dagen. Nieuws wordt in deze historische replay overgeslagen.
echo  De bestaande research-parity meting blijft onaangeraakt.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo FOUT: draai eerst update.cmd
  pause
  exit /b 1
)
if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections ^
  --days %DAGEN% ^
  --only section_fifteen_btc_m1,section_sixteen_btc_m5,section_seventeen_btc_m15 ^
  --symbols BTCUSD ^
  --btc-jarvis-replay ^
  --csv runtime\secties15-17-jarvis.csv

if errorlevel 1 (
  echo.
  echo De replay stopte. Lees de eerste ERROR-regel hierboven.
  pause
  exit /b 1
)

echo.
echo Resultaat: runtime\secties15-17-jarvis.csv
echo Dit is shadow-only en zet niets live.
pause

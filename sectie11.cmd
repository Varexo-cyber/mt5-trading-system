@echo off
setlocal
cd /d "%~dp0"

set DAGEN=180
if not "%~1"=="" set DAGEN=%~1

echo.
echo  S11-S14 GOUDKRUISEN -- ECHTE BROKERBARS, GEEN ECHT GELD
echo  XAUEUR M1, XAUGBP M1, XAUAUD M5, XAUJPY M1
echo  %DAGEN% dagen. MT5 moet draaien en ingelogd zijn.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo FOUT: draai eerst update.cmd
  pause
  exit /b 1
)
if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections ^
  --days %DAGEN% --only section_eleven_xaueur_m1,section_twelve_xaugbp_m1,section_thirteen_xauaud_m5,section_fourteen_xaujpy_m1 ^
  --symbols XAUEUR,XAUGBP,XAUAUD,XAUJPY ^
  --csv runtime\secties11-14.csv

echo.
echo Resultaat: runtime\secties11-14.csv
echo Deze vier staan bewust NIET live. Deze broker-replay beslist promotie.
pause

@echo off
setlocal
cd /d "%~dp0"

set DAGEN=180
if not "%~1"=="" set DAGEN=%~1

echo.
echo  S15-S17 BTCUSD BROKER-REPLAY -- GEEN ECHT GELD
echo  S15 M1, S16 M5, S17 M15; alleen BTCUSD.
echo  %DAGEN% dagen, RESEARCH-PARITY SHADOW met brokerkosten en positiebeheer.
echo  Alleen vooraf geldige setups binnen de gemeten spread/risico-envelope.
echo  Dit wijzigt Jarvis live-risico NIET en plaatst S15-S17 NIET live.
echo  MT5 moet draaien en ingelogd zijn.
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
  --btc-research-parity ^
  --manage-grid ^
  --csv runtime\secties15-17.csv

if errorlevel 1 (
  echo.
  echo De replay stopte. Lees de eerste ERROR-regel hierboven.
  pause
  exit /b 1
)

echo.
echo Resultaat: runtime\secties15-17.csv
echo Deze drie staan shadow-only. Deze volledige broker-replay beslist niet automatisch live.
pause

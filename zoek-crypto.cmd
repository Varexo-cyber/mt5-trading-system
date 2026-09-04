@echo off
setlocal
cd /d "%~dp0"

set DAGEN=180
if not "%~1"=="" set DAGEN=%~1
set DB=runtime\research\btcusd-actual.sqlite3

echo.
echo  BTCUSD ONDERZOEK OP ECHTE EIGHTCAP-BARS
echo  %DAGEN% dagen, M1 M5 M15 M30 H1. Geen orders en geen AI-kosten.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo FOUT: draai eerst update.cmd
  pause
  exit /b 1
)

.venv-live\Scripts\python.exe scripts\capture_research_data.py ^
  --days %DAGEN% --equity 203 --output %DB% ^
  --symbols BTCUSD --timeframes M1,M5,M15,M30,H1
if errorlevel 1 goto fout

.venv-live\Scripts\python.exe -m scripts.research_btcusd --database %DB%
if errorlevel 1 goto fout

echo.
echo BTCUSD heeft alle onafhankelijke controles gehaald; laat het resultaat reviewen.
pause
exit /b 0

:fout
echo.
echo Geen promotie. BTCUSD blijft shadow; lees de eerste afgevinkte fout hierboven.
pause
exit /b 1

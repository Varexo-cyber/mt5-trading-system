@echo off
setlocal
cd /d "%~dp0"

set DAGEN=180
if not "%~1"=="" set DAGEN=%~1
set DB=runtime\research\metals-actual.sqlite3

echo.
echo  ZOEK OPNIEUW OP ECHTE EIGHTCAP-BARS -- ALLE GOUDKRUISEN EN CLOCKS
echo  %DAGEN% dagen, M1 M5 M15 M30 H1. Geen orders en geen Claude-kosten.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo FOUT: draai eerst update.cmd
  pause
  exit /b 1
)

.venv-live\Scripts\python.exe scripts\capture_research_data.py ^
  --days %DAGEN% --equity 203 --output %DB% ^
  --symbols XAUUSD,EURUSD,GBPUSD,AUDUSD,USDJPY,XAUEUR,XAUGBP,XAUAUD,XAUJPY ^
  --timeframes M1,M5,M15,M30,H1
if errorlevel 1 goto fout

.venv-live\Scripts\python.exe -m scripts.research_section_eleven_crosses --database %DB%
if errorlevel 1 goto fout

echo.
echo Klaar. Controleer bovenaan dat elke markt actual=M1,M5,M15,M30,H1 zegt.
echo Een SYNTHETIC winnaar krijgt geen live toestemming.
pause
exit /b 0

:fout
echo.
echo De zoekrun stopte. Lees de eerste ERROR-regel hierboven.
pause
exit /b 1

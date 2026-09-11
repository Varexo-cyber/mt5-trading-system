@echo off
setlocal

if not defined SECTIE7_ROOT set SECTIE7_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\sectie7-actief.cmd" >nul
  call "%TEMP%\sectie7-actief.cmd" __uittemp
  exit /b
)
cd /d "%SECTIE7_ROOT%"

echo.
echo  ========================================================================
echo   SECTIE 7 SMC GOUD - 180 DAGEN SHADOW - KLOK EN SESSIE ZOEKEN
echo  ========================================================================
echo.
echo   M5, M15, M30 en H1 worden apart vergeleken. Elke variant leest daarnaast
echo   M15/M30/H1/H4-context. M1 wordt alleen geladen voor de gedeelde volume-
echo   spike gate en is GEEN S7-entryklok. Maximaal een setup per UTC-dag.
echo   Nieuws is niet historisch reproduceerbaar en wordt daarom eerlijk als
echo   ontbrekend gemeld. Live zou de bestaande nieuwsblokkade wel gelden.
echo   NIETS wordt aan de live allowlist toegevoegd.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe ontbreekt. Draai eerst update.cmd.
  pause
  exit /b 1
)
if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections ^
  --days 180 ^
  --only section_seven_gold_smc ^
  --section-markets ^
  --sweep M5 M15 M30 H1 ^
  --jarvis-replay ^
  --csv runtime\sectie7-smc.csv

echo.
echo  Lees de kloktabel, UTC-uren en vroege/late helft; hoogste R alleen is
echo  geen winnaar. Uitvoer: runtime\sectie7-smc.csv
echo.
pause

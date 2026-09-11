@echo off
setlocal

rem Draai vanuit een tijdelijke kopie zodat update.cmd een nachtelijke run niet
rem halverwege kan veranderen.
if not defined HOLDOUT_ROOT set HOLDOUT_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\jarvis-holdout-2024-2025-actief.cmd" >nul
  call "%TEMP%\jarvis-holdout-2024-2025-actief.cmd" __uittemp
  exit /b
)
cd /d "%HOLDOUT_ROOT%"

echo.
echo  ========================================================================
echo   JARVIS HOLDOUT - 1 SEP 2024 T/M 31 AUG 2025 UTC - SHADOW/REPLAY
echo  ========================================================================
echo.
echo   Onaangeraakte, niet-overlappende periode direct voor de 360-dagenrun.
echo   Meet de huidige bevroren S6- en S10-configuratie op hun eigen markten.
echo   De vooraf gekozen S10-kandidaat part 50%% op +1R staat in de exit-grid.
echo   Beoordeel alleen die vooraf gekozen rij; kies morgen geen nieuwe winnaar.
echo   Dit wijzigt niets live en kan meerdere uren duren.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe ontbreekt. Draai eerst update.cmd.
  pause
  exit /b 1
)
if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections ^
  --start-date 2024-09-01 ^
  --end-date 2025-08-31 ^
  --only section_six_gold_m5,section_ten_gold_m1 ^
  --section-markets ^
  --jarvis-replay ^
  --exit-grid kern ^
  --csv runtime\jarvis-holdout-2024-2025.csv

if errorlevel 1 (
  echo.
  echo  HOLDOUT GESTOPT. Er is niets live veranderd.
  pause
  exit /b 1
)

echo.
echo  ========================================================================
echo   WINST EN VERLIES PER PERIODE
echo  ========================================================================
echo.
.venv-live\Scripts\python.exe -m scripts.trade_outcome_analysis ^
  runtime\jarvis-holdout-2024-2025.csv

echo.
echo  Klaar: runtime\jarvis-holdout-2024-2025.csv
echo  Dit is holdoutbewijs; deze run verandert geen enkele live regel.
echo.
pause

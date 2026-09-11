@echo off
setlocal

rem Werk vanuit een tijdelijke kopie zodat update.cmd deze lange run niet
rem halverwege kan veranderen.
if not defined S6S10_ROOT set S6S10_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\s6s10-meting-actief.cmd" >nul
  call "%TEMP%\s6s10-meting-actief.cmd" __uittemp
  exit /b
)
cd /d "%S6S10_ROOT%"

echo.
echo  ========================================================================
echo   S6 + S10 VERLIESMETING - 360 DAGEN - ALLEEN SHADOW/REPLAY
echo  ========================================================================
echo.
echo   Meet uitsluitend section_six_gold_m5 en section_ten_gold_m1.
echo   De actieve S10-tijdblokkade blijft onderdeel van de configuratie.
echo.
echo   Op exact dezelfde toegelaten entries worden vergeleken:
echo     - vaste SL/TP tegenover huidig live break-even
echo     - meerdere robuuste break-even-niveaus en stopplaatsen
echo     - M5-, M15- en gecombineerde trendrichting bij entry
echo     - vroege exit bij aantoonbare trend- plus structuurbreuk
echo     - richting, UTC-uur, weekdag en herhaalde verliesuren
echo.
echo   Dit wijzigt NIETS live en kan langer duren dan goud360.cmd.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe ontbreekt. Draai eerst update.cmd.
  pause
  exit /b 1
)
if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections ^
  --days 360 ^
  --only section_six_gold_m5,section_ten_gold_m1 ^
  --section-markets ^
  --jarvis-replay ^
  --exit-grid kern ^
  --trend-grid ^
  --fault-exit-grid ^
  --csv runtime\s6-s10-meting-360.csv

if errorlevel 1 (
  echo.
  echo  METING GESTOPT. Er is niets live veranderd.
  pause
  exit /b 1
)

echo.
echo  ========================================================================
echo   BESTAANDE TRADES UITSPLITSEN
echo  ========================================================================
echo.
.venv-live\Scripts\python.exe -m scripts.trade_outcome_analysis ^
  runtime\s6-s10-meting-360.csv

echo.
echo  Klaar: runtime\s6-s10-meting-360.csv
echo  Dit is diagnose; geen gevonden kandidaat wordt automatisch live gezet.
echo.
pause

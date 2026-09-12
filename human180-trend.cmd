@echo off
setlocal

if not defined HUMAN180_ROOT set HUMAN180_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\human180-trend-actief.cmd" >nul
  call "%TEMP%\human180-trend-actief.cmd" __uittemp
  exit /b
)
cd /d "%HUMAN180_ROOT%"

echo.
echo  ========================================================================
echo   HUMAN TREND CONTINUATION - 180 DAGEN - ALLEEN SHADOW/REPLAY
echo  ========================================================================
echo.
echo   Alleen setups die BIJ ENTRY al trend_continuation waren.
echo   XAUUSD, gesloten M5/M15/H1/H4, structurele stop, vaste 2R-baseline.
echo   Volledige exit-, fault-, trend-, risico- en contextmeting in een run.
echo   Dit verandert niets live.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe ontbreekt. Draai eerst update.cmd.
  pause
  exit /b 1
)
if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections ^
  --days 180 ^
  --only human_context_decision ^
  --human-story trend_continuation ^
  --section-markets ^
  --jarvis-replay ^
  --fixed-exits ^
  --exit-grid alles ^
  --trend-grid ^
  --fault-exit-grid ^
  --csv runtime\human-trend-continuation-180.csv

if errorlevel 1 (
  echo.
  echo  METING GESTOPT. Er is niets live veranderd.
  pause
  exit /b 1
)

echo.
.venv-live\Scripts\python.exe -m scripts.trade_outcome_analysis ^
  runtime\human-trend-continuation-180.csv

echo.
echo  Klaar: runtime\human-trend-continuation-180.csv
echo  Dit is shadowonderzoek en zet niets automatisch live.
echo.
pause

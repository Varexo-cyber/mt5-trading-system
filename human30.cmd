@echo off
setlocal

if not defined HUMAN30_ROOT set HUMAN30_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\human30-actief.cmd" >nul
  call "%TEMP%\human30-actief.cmd" __uittemp
  exit /b
)
cd /d "%HUMAN30_ROOT%"

echo.
echo  ========================================================================
echo   HUMAN-CONTEXT BESLISSER - 30 DAGEN - ALLEEN SHADOW/REPLAY
echo  ========================================================================
echo.
echo   Een centrale beslisser, drie marktverhalen:
echo     - trend continuation
echo     - failed auction / liquidity reclaim
echo     - break en verdedigde retest
echo.
echo   Alleen gesloten M5, M15, H1 en H4 candles. XAUUSD, structurele stop,
echo   vast 2R-doel en de bestaande accountpoorten. Dit verandert niets live.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe ontbreekt. Draai eerst update.cmd.
  pause
  exit /b 1
)
if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections ^
  --days 30 ^
  --only human_context_decision ^
  --section-markets ^
  --jarvis-replay ^
  --fixed-exits ^
  --exit-grid alles ^
  --trend-grid ^
  --fault-exit-grid ^
  --csv runtime\human-context-30.csv

if errorlevel 1 (
  echo.
  echo  METING GESTOPT. Er is niets live veranderd.
  pause
  exit /b 1
)

echo.
.venv-live\Scripts\python.exe -m scripts.trade_outcome_analysis ^
  runtime\human-context-30.csv

echo.
echo  Klaar: runtime\human-context-30.csv
echo  Geen uitkomst uit deze eerste 30 dagen wordt automatisch live gezet.
echo.
pause

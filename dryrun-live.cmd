@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Sections 2 and 3, EXACTLY AS CONFIGURED
echo  ============================================
echo.
echo  impulse_retest on M15 and order_block on M30 -- their own clocks, nothing
echo  else. One position per symbol. The account's own concurrent-position cap.
echo.
echo  This is the number that answers "would I have made money". dryrun.cmd
echo  answers a different question: which CLOCK is best on this broker's feed,
echo  and to do that it runs ten combinations that will never run together.
echo.
echo  One fifth of the work, so a month over the whole catalogue finishes.
echo.
echo  MT5 must be running and logged in.
echo.
echo  USAGE -- one plain number
echo    dryrun-live.cmd          7 days
echo    dryrun-live.cmd 30       30 days
echo.

rem PLAIN NUMBERS ONLY. cmd splits arguments on commas, so a timeframe list
rem typed at the prompt loses the rest of the command line before the script
rem is ever called. Nothing here is a list.

set DAYS=%1
if "%DAYS%"=="" set DAYS=7

echo  Window: last %DAYS% days, every market in the scan universe
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe not found. Run update.cmd first.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections --days %DAYS% --live-only --csv runtime\dryrun-live.csv

if errorlevel 1 (
  echo.
  echo  The run failed. The traceback above says why -- most often MT5 is not
  echo  logged in, or the symbols have no history that far back.
)

echo.
echo  Every single decision, one row each, is in runtime\dryrun-live.csv
echo.
pause

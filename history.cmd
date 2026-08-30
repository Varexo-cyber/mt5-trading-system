@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ==================================================
echo   Is this real? — the long history test
echo  ==================================================
echo.
echo  Sections 2 and 3 exactly as configured, over MONTHS of this broker's own
echo  data, with the sample judging itself:
echo.
echo    - the 95%% interval around the win rate, so 10 trades cannot look
echo      like a finding
echo    - sigma measured on DAILY totals, not per trade. Sixteen markets
echo      breaking on the same morning are one observation, not sixteen --
echo      that correction was the largest single one in the research
echo    - every month listed separately. One good week inside a bad month
echo      is not an edge, it is where you happened to look
echo    - a verdict with tick boxes: 200+ trades, +2 sigma, 3+ months,
echo      every month positive. All four, or it says NOT ENOUGH.
echo.
echo  Resolved on M5 rather than M1 so that half a year finishes. An M30 trade
echo  on M5 bars is a 6:1 ratio, which is enough to tell which barrier came
echo  first; M1 over 180 days is a quarter of a million bars per market.
echo.
echo  MT5 must be running and logged in. THIS TAKES A WHILE -- let it run.
echo.
echo  USAGE
echo    history.cmd          90 days
echo    history.cmd 180      180 days
echo    history.cmd 365      a full year, if your broker has the bars
echo.

set DAYS=%1
if "%DAYS%"=="" set DAYS=90

echo  Window: last %DAYS% days, core markets, live clocks only
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe not found. Run update.cmd first.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections --days %DAYS% --core --live-only --no-m1 --csv runtime\history.csv

if errorlevel 1 (
  echo.
  echo  The run failed. The traceback above says why -- most often MT5 is not
  echo  logged in, or the symbols have no history that far back. Try a shorter
  echo  window.
)

echo.
echo  Every decision, one row each, is in runtime\history.csv
echo.
pause

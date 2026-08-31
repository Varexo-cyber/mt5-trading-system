@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ==================================================
echo   Which clock? Both sections, every clock, long window
echo  ==================================================
echo.
echo  The table you already have was seven days and read the FIXED stop --
echo  four trades in some rows, and a row of losses for a configuration that
echo  was actually making money. This is the same table done properly:
echo.
echo    - the BREAK-EVEN exit, which is what the account runs
echo    - the fixed stop beside it, because the gap between them IS what the
echo      stop rule is worth on that clock
echo    - a "too few to judge" marker on any row under 200 trades
echo.
echo  M5 IS NOT SWEPT. An M5 trade needs M1 bars to tell which barrier came
echo  first, and M1 over 180 days is a quarter of a million bars per market --
echo  it triples the run. M5 was also the worst row for both sections in every
echo  measurement so far.
echo.
echo  HOW LONG: roughly 1.5 hours for 180 days on this machine. Measured, not
echo  guessed: 0.72M evaluations at 2.4ms each. Start it and leave it.
echo.
echo  MT5 must be running and logged in.
echo.
echo  USAGE
echo    sweep.cmd            180 days   (about 90 min)
echo    sweep.cmd 90         90 days    (about 45 min)
echo    sweep.cmd 30         30 days    (about 15 min, still thin)
echo.

set DAYS=%1
if "%DAYS%"=="" set DAYS=180

echo  Window: last %DAYS% days, core markets, clocks M15 M30 H1 H4
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe not found. Run update.cmd first.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections --days %DAYS% --core --no-m1 --sweep M15 M30 H1 H4 --csv runtime\sweep.csv

if errorlevel 1 (
  echo.
  echo  The run failed. The traceback above says why.
)

echo.
echo  Every decision is in runtime\sweep.csv
echo.
pause

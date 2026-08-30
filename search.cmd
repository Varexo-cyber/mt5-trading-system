@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ==================================================
echo   Section four: is there an edge on YOUR bars?
echo  ==================================================
echo.
echo  Twelve candidate mechanisms, three clocks, on this broker's own data --
echo  not on HistData. That distinction is the point.
echo.
echo  The old research measured 94 detectors over ten years of HistData and
echo  shipped the two that survived. On Eightcap those two came back at 49.9%%
echo  over 1,610 trades: a coin flip. The problem was not too little research,
echo  it was research on data that could not price this account.
echo.
echo  WHAT STOPS IT FINDING NOISE:
echo    - a Bonferroni bar that RISES with the size of the grid. Keeping the
echo      best of forty cells finds 2 sigma on pure noise most of the time.
echo    - a holdout split by date; the newer 40%% must pass on its own
echo    - sigma measured on DAILY totals, not per trade
echo    - a random-entry control on the same bars, subtracted. A coin flip in
echo      this harness does NOT read zero, and the research learned that the
echo      expensive way.
echo    - this broker's real commission and slippage, per asset class
echo.
echo  IT IS BUILT TO COME BACK EMPTY. If nothing survives, that is the honest
echo  answer and not a broken run.
echo.
echo  MT5 must be running and logged in.
echo.
echo  USAGE
echo    search.cmd           365 days
echo    search.cmd 180       180 days, quicker
echo.

set DAYS=%1
if "%DAYS%"=="" set DAYS=365

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe not found. Run update.cmd first.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.search_section_four --days %DAYS% --clocks M15 M30 H1 --csv runtime\search.csv

if errorlevel 1 (
  echo.
  echo  The run failed. The traceback above says why.
)

echo.
echo  Every cell, one row each, is in runtime\search.csv
echo.
pause

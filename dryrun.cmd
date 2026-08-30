@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   What would sections 2 and 3 have done?
echo  ============================================
echo.
echo  The research measured these two on ten years of HistData bid bars. That
echo  answered "does this entry have an edge". It could not answer the question
echo  that decides whether the account makes money:
echo.
echo    on THIS broker, THESE symbols, THIS equity, what came out?
echo.
echo  The one number the research had to assume was the real Eightcap spread,
echo  and that is exactly the number that has killed every previous detector on
echo  this account. This charges the spread each bar actually carried.
echo.
echo  It also runs the REAL position sizer, so TRADE_SKIPPED_UNDERCAPITALIZED
echo  is counted rather than modelled -- and every refusal is attributed by
echo  name. A week with four trades and three hundred refusals is a diagnosis.
echo  A week with four trades is a number.
echo.
echo  WHAT IT CANNOT SEE, so read the result with these in mind:
echo    - slippage beyond the recorded spread, so fills are optimistic
echo    - the 4-slot cap is not enforced, only the trades are counted
echo    - the news blackout is not replayed, so this is an UPPER BOUND
echo.
echo  MT5 must be running and logged in.
echo.

set DAYS=%1
if "%DAYS%"=="" set DAYS=7
set SWEEP=%2
if "%SWEEP%"=="" set SWEEP=M5,M15,M30,H1,H4

echo  Window: last %DAYS% days.   (dryrun.cmd 30  for a month)
echo  Clocks: %SWEEP%
echo.
echo  Each section is run on EVERY one of those timeframes, separately. The
echo  shipped clocks (M15 for section 2, M30 for section 3) were chosen on
echo  HistData, which could not price your spread. This can, so if another
echo  clock wins here it wins for a reason that matters.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe not found. Run update.cmd first.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections --days %DAYS% --sweep %SWEEP% --csv runtime\dryrun.csv
if errorlevel 1 (
  echo.
  echo  The run failed. The traceback above says why -- most often MT5 is not
  echo  logged in, or the symbol has no history that far back.
)

echo.
echo  Every single decision, one row each, is in runtime\dryrun.csv --
echo  open it in Excel to see each trade with its lots, its risk in euros and
echo  its result.
echo.
pause

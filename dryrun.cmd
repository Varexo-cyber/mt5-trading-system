@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   What would sections 2 and 3 have done?
echo  ============================================
echo.
echo  Runs the live scan universe -- the whole broker catalogue, same asset
echo  class filter the scanner uses -- through the real modules, the real
echo  gates and the real position sizer, over real Eightcap history.
echo.
echo  Every decision is reported, not just the trades, so a quiet week is a
echo  diagnosis instead of a number. TRADE_SKIPPED_UNDERCAPITALIZED is counted
echo  for real rather than modelled.
echo.
echo  Each section runs on EVERY clock separately (M5 M15 M30 H1 H4). The
echo  shipped ones were chosen on HistData, and that data could not price your
echo  spread. This can.
echo.
echo  WHAT IT CANNOT SEE:
echo    - slippage beyond the recorded spread, so fills are optimistic
echo    - the 4-slot cap is not enforced, only the trades are counted
echo    - the news blackout is not replayed, so this is an UPPER BOUND
echo.
echo  MT5 must be running and logged in.
echo.
echo  THIS ONE IS SLOW ON PURPOSE. Ten section/clock combinations over the
echo  whole catalogue is hours of work. If you want "would I have made money",
echo  run dryrun-live.cmd instead -- sixteen markets, two passes, minutes.
echo.
echo  USAGE -- plain words, no commas
echo    dryrun.cmd            every market, 7 days     (hours)
echo    dryrun.cmd 7 core     core markets, 7 days     (minutes)
echo    dryrun.cmd 30 core    core markets, 30 days
echo    dryrun.cmd 7 40       first 40 markets only
echo.

rem ARGUMENTS ARE PLAIN NUMBERS, and that is deliberate.
rem
rem `dryrun.cmd 7 M15,M30 --limit 40` failed because CMD SPLITS ARGUMENTS ON
rem COMMAS. The shell handed the script "M15" and "M30" as separate words, so
rem --limit collected "M30" and argparse rejected the remainder. No validation
rem inside the script could have caught it: the damage is done before it runs.
rem
rem So the timeframe list never passes through the shell. It is written out
rem below and the only things anyone types are two numbers.

set DAYS=%1
if "%DAYS%"=="" set DAYS=7
set LIMIT=%2
set SCOPE=
if /i "%2"=="core" (set SCOPE=--core& set LIMIT=0)
if "%LIMIT%"=="" set LIMIT=0

echo  Window: last %DAYS% days
if defined SCOPE (
  echo  Markets: the core set -- the eleven majors, gold, four indices
) else (
  if "%LIMIT%"=="0" (echo  Markets: the whole scan universe) else (echo  Markets: first %LIMIT% only)
)
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe not found. Run update.cmd first.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections --days %DAYS% --limit %LIMIT% %SCOPE% --sweep M5 M15 M30 H1 H4 --csv runtime\dryrun.csv

if errorlevel 1 (
  echo.
  echo  The run failed. The traceback above says why -- most often MT5 is not
  echo  logged in, or the symbols have no history that far back.
)

echo.
echo  Every single decision, one row each, is in runtime\dryrun.csv --
echo  open it in Excel to see each trade with its lots, its risk in euros and
echo  its result.
echo.
pause

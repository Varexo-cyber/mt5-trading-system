@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ==================================================
echo   Which clock? Both sections, the clocks you name
echo  ==================================================
echo.
echo  Every row is measured on:
echo.
echo    - the BREAK-EVEN exit, which is what the account runs
echo    - the fixed stop beside it, because the gap between them IS what the
echo      stop rule is worth on that clock
echo    - a "too few to judge" marker on any row under 200 trades
echo.
echo  M1 AND M5 ARE ALLOWED NOW. They were not before, and the reason was
echo  real: a trade has to be walked out on bars FINER than the one that
echo  produced it, or you cannot tell whether the stop or the target came
echo  first inside a bar. M5 needs M1 bars. M1 has nothing beneath it.
echo.
echo  So an M1 trade is walked out on its own bars, and a bar holding BOTH
echo  barriers is booked as a LOSS -- the order is unknowable and guessing in
echo  your own favour is how a backtest lies. That makes the M1 row biased
echo  AGAINST M1. Positive there means something. Negative there may be the
echo  measurement rather than the strategy, and only tick data would settle it.
echo.
echo  ASKING FOR M1 OR M5 TURNS THE M1 FETCH BACK ON automatically, because
echo  without it there is nothing to resolve them against.
echo.
echo  HOW LONG. Measured, not guessed: 100 days on M15+M30+H1+H4 over the 16
echo  core markets took 45 minutes, all of it compute. Cost scales with the
echo  number of BARS, so:
echo.
echo      clock   bars per market per day   ~16 markets, 14 days
echo      H4                  2                    1 min
echo      H1                 17                    1 min
echo      M30                34                    3 min
echo      M15                69                    5 min
echo      M5                206                   15 min
echo      M1               1029                   55 min
echo.
echo  M1 over 180 days is six hours. Do not. Use a SHORT window for M1 -- it
echo  fires so much more often that two weeks there is more trades than six
echo  months on H4.
echo.
echo  EXPECT MARKETS TO BE SKIPPED ON M1. The stop is one M1 ATR wide and the
echo  round trip is the same spread it always was, so the cost share rises as
echo  the clock falls. That refusal is itself part of the answer to "should we
echo  trade M1", and the run prints it per market.
echo.
echo  MT5 must be running and logged in.
echo.
echo  USAGE
echo    sweep.cmd                    180 days, M15 M30 H1 H4
echo    sweep.cmd 90                 90 days,  M15 M30 H1 H4
echo    sweep.cmd 14 M1 M5           14 days,  M1 and M5 only
echo    sweep.cmd 30 M5 M15 M30      30 days,  those three
echo.

set DAYS=%1
if "%DAYS%"=="" set DAYS=180
shift

set CLOCKS=
:collect
if "%1"=="" goto collected
set CLOCKS=%CLOCKS% %1
shift
goto collect
:collected
if "%CLOCKS%"=="" set CLOCKS=M15 M30 H1 H4

REM M1 history is the expensive fetch, so it stays off unless a clock needs
REM it. M5 needs it to resolve against; M1 needs it to exist at all. Getting
REM this wrong is not a slow run, it is a MISSING ROW -- the script refuses
REM --no-m1 together with an M1 clock rather than dropping it quietly.
set FINE=--no-m1
for %%C in (%CLOCKS%) do (
  if /I "%%C"=="M1" set FINE=
  if /I "%%C"=="M5" set FINE=
)

echo  Window: last %DAYS% days, core markets, clocks %CLOCKS%
if "%FINE%"=="" echo  M1 history is being fetched, because those clocks need it to resolve.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe not found. Run update.cmd first.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections --days %DAYS% --core %FINE% --sweep %CLOCKS% --csv runtime\sweep.csv

if errorlevel 1 (
  echo.
  echo  The run failed. The traceback above says why.
)

echo.
echo  Every decision is in runtime\sweep.csv
echo    verdict.cmd runtime\sweep.csv                        every combination
echo    verdict.cmd runtime\sweep.csv --only order_block:M5  just one
echo.
pause

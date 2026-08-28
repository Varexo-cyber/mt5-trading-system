@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Does section six actually work?
echo  ============================================
echo.
echo  modules.cmd CANNOT answer this and never could. The general replay
echo  fetches D1, H4, H1, M15 and M5 - no M1 - and section six triggers on
echo  M1, so it returned a neutral signal in every backtest ever run and
echo  appeared in no table. Not graded badly. Never graded.
echo.
echo  This walks every closed M1 bar with the detector at live settings and
echo  the lane's own stop and target, and charges the broker's own recorded
echo  spread on the trigger bar.
echo.
echo  NOT modelled: the per-second claim and cut, the profit lock, the news
echo  blackout, the two-position cap. This is the ENTRY and the plan. If that
echo  loses, no exit rule rescues it. If it wins, the exit work has something
echo  real to improve.
echo.
echo  No orders. No Claude API. Nothing written anywhere.
echo.
echo  THE LANE IS OFF since 28 August, and the sweep is what switched it off.
echo.
echo  This banner keeps saying so because the last time it drifted out of step
echo  with the config it said "switched OFF" for a day while the lane was
echo  trading real money. Whatever else is wrong on this screen, that line has
echo  to be true.
echo.
echo  WHAT THE SWEEP FOUND, over 30 days and five gold markets:
echo.
echo    gate     cost   trades   win   per trade
echo       5    28.0%%      853   42%%    -0.332R
echo      14    10.0%%      564   42%%    -0.229R   ^<- what was live
echo      40     3.5%%      111   43%%    -0.064R
echo      56     2.5%%       39   49%%    +0.001R
echo.
echo  THE WIN RATE DOES NOT MOVE. 42%% at every gate. If the entry read a
echo  direction, a stricter gate would keep better trades and that column would
echo  climb. What moves instead is the cost band, 28%% of R down to 2.5%% - so
echo  the whole distance from -0.332R to +0.001R is the spread, not selection.
echo  This lane goes to exactly zero as costs go to zero.
echo.
echo  And 42%% is what CHANCE pays here: a 1.4:1 payoff needs 41.7%%. The entry
echo  beat a coin by three tenths of a percentage point, worth +0.03R, against
echo  a round trip that costs 0.235R.
echo.
echo  Live said the same from a different direction: candle_momentum over seven
echo  days, 32 trades, 20 won, +0.00R, -13.49 EUR.
echo.
echo  WHAT COMES BACK ON, AND WHEN. Not a gate and not a target - the ENTRY has
echo  to change. `--payoff` is the run that says whether a change worked: it
echo  moves the target instead of the gate and prints the achieved rate beside
echo  the rate a coin pays. The bar is a corrected sigma, not a percentage,
echo  because a random walk through that table prints +4%% to +8%% edges.
echo.
echo  The detector itself stays live in section one, where it can confirm what
echo  something else already saw. What it lost is the right to trade alone.
echo.
echo  section6.cmd --sweep              IS there a gate where this pays?
echo  section6.cmd --payoff             DOES THE ENTRY READ ANYTHING AT ALL?
echo  section6.cmd --days 60            longer history
echo  section6.cmd --symbols XAUUSD     one market
echo  section6.cmd --stride 2           faster, coarser
echo.
echo  --sweep is the run that decides whether the lane comes back on. It
echo  raises minimum_target_spreads step by step and reports what survives at
echo  each. Cost is 1.4/gate of R, so the gate and the cost are one number
echo  seen from two sides.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

REM  A bare number means days. See the note in modules.cmd: `--days30` fails
REM  as "unrecognized arguments", which is a missing space reported as a
REM  missing feature.
echo %~1| findstr /r "^[0-9][0-9]*$" >nul
if not errorlevel 1 (
  echo  Reading "%~1" as --days %~1.
  echo.
  ".venv-live\Scripts\python.exe" scripts\backtest_section_six.py --days %~1
  goto :done
)

if "%~1"=="" (
  ".venv-live\Scripts\python.exe" scripts\backtest_section_six.py --days 30
) else (
  ".venv-live\Scripts\python.exe" scripts\backtest_section_six.py %*
)

:done

echo.
pause
endlocal

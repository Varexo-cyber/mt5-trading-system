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
echo  MEASURED 26 AUGUST: 1,681 trades over 30 days, -0.304R each, -511.73R.
echo  The entry is not the problem - on gold it reads the minute correctly,
echo  43%% against the 41.7%% that a 1.4:1 payoff needs. It is worth +0.03R.
echo  The round trip costs 0.235R.
echo.
echo  THE LANE IS ON. This banner said "switched OFF" until 27 August and the
echo  config said `enabled: true` the whole time - the line was written when it
echo  was off, the lane came back after the entry was reworked, and the banner
echo  did not follow. That is a launcher lying to its operator about what is
echo  trading his money, which is worse than any number on this screen.
echo.
echo  It is on as a BOUNDED experiment and here is the whole bound:
echo    stake 1.0%% per trade, two at a time, counted in the book cap, and a
echo    section breaker that switches the lane off by itself after six losers
echo    in a row. Six losers at 1%% is 6%% of the account. That is the most this
echo    trial can cost before it stops without anyone watching.
echo.
echo  section6.cmd --sweep              IS there a gate where this pays?
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

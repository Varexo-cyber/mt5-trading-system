@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Which detector actually makes money?
echo  ============================================
echo.
echo  backtest.cmd grades the five PLAYBOOKS, and all five were already
echo  killed. This grades the CONFLUENCE MODULES - the eight detectors the
echo  live account actually trades on, which have never been graded at all.
echo.
echo  Read the second table first. AGAINST CHANCE is a coin flip taking the
echo  same moments with the same stops and the same targets. A module that
echo  cannot beat it is not analysis - it is a way of choosing when to pay
echo  the spread.
echo.
echo  No orders. No Claude API. Nothing written anywhere. It only costs time,
echo  and 120 days of six symbols takes a while - leave it running.
echo.
echo  The default run now also sweeps EXIT RULES. These trades win 45-57%% of
echo  the time and still lose money, so the winners are smaller than the
echo  losers - and the backtest models no profit lock and no trailing stop at
echo  all. The sweep says whether banking earlier would have paid, measured
echo  against doing nothing rather than assumed.
echo.
echo  modules.cmd --days 240              longer history, smaller error bars
echo  modules.cmd --no-baseline           skip the coin flip, faster
echo  modules.cmd --by-regime             split every detector by regime
echo  modules.cmd --stride 2              faster, coarser
echo  modules.cmd 90                      a bare number means --days
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

REM  A BARE NUMBER MEANS DAYS.
REM
REM  `modules.cmd --days90` fails with "unrecognized arguments: --days90",
REM  which is a missing space reported as a missing feature. The number is the
REM  only thing anybody types here, so accept it on its own: `modules.cmd 90`.
REM  Everything else is passed through untouched.
echo %~1| findstr /r "^[0-9][0-9]*$" >nul
if not errorlevel 1 (
  echo  Reading "%~1" as --days %~1.
  echo.
  ".venv-live\Scripts\python.exe" scripts\backtest_modules.py --days %~1 --exits
  goto :done
)

if "%~1"=="" (
  ".venv-live\Scripts\python.exe" scripts\backtest_modules.py --days 240 --exits
) else (
  ".venv-live\Scripts\python.exe" scripts\backtest_modules.py %*
)

:done

echo.
pause
endlocal

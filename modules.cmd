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
echo  modules.cmd --days 240              longer history, smaller error bars
echo  modules.cmd --by-regime             split every detector by regime
echo  modules.cmd --stride 2              faster, coarser
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

if "%~1"=="" (
  ".venv-live\Scripts\python.exe" scripts\backtest_modules.py --days 120
) else (
  ".venv-live\Scripts\python.exe" scripts\backtest_modules.py %*
)

echo.
pause
endlocal

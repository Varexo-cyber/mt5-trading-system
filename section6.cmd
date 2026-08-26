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
echo  section6.cmd --days 60            longer history
echo  section6.cmd --symbols XAUUSD     one market
echo  section6.cmd --stride 2           faster, coarser
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

if "%~1"=="" (
  ".venv-live\Scripts\python.exe" scripts\backtest_section_six.py --days 30
) else (
  ".venv-live\Scripts\python.exe" scripts\backtest_section_six.py %*
)

echo.
pause
endlocal

@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Quick look — about a minute
echo  ============================================
echo.
echo  Sections 2 and 3 on their own clocks, core markets, 30 days.
echo  Same report as history.cmd, smaller window.
echo.
echo  WHAT IT IS FOR: checking that a change did what it was supposed to.
echo  Thirty days is NOT enough to conclude anything about profitability --
echo  the verdict block at the bottom will say so itself. Use history.cmd 180
echo  when you want an answer rather than a check.
echo.
echo  MT5 must be running and logged in.
echo.
echo  USAGE
echo    quick.cmd          30 days
echo    quick.cmd 14       14 days, faster still
echo.

set DAYS=%1
if "%DAYS%"=="" set DAYS=30

echo  Window: last %DAYS% days, core markets, live clocks only
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe not found. Run update.cmd first.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections --days %DAYS% --core --live-only --no-m1 --csv runtime\quick.csv

if errorlevel 1 (
  echo.
  echo  The run failed. The traceback above says why.
)

echo.
echo  Every decision, one row each, is in runtime\quick.csv
echo.
pause

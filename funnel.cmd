@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Why does nothing fire on FX?
echo  ============================================
echo.
echo  The core run took 42 trades and NOT ONE was on an FX pair -- the eleven
echo  majors the strategy was actually measured on. 95.9%% of everything read
echo  "no weighted directional evidence", which only means "the detector
echo  returned nothing", and it has five separate reasons to do that.
echo.
echo  This counts them, per market, in the order the detector applies them:
echo.
echo    a bar closed beyond its 20-bar channel
echo    ...by at least 1.0 ATR          the impulse filter
echo    ...the level was not given up
echo    ...price is on the retest side
echo    ...and within 0.15 ATR AT THE CLOSE
echo.
echo  Plus how often the bar TOUCHED that band without closing in it. The
echo  research bought these with a resting limit order, which fills on a touch.
echo  A once-per-bar check cannot see one.
echo.
echo  MT5 must be running and logged in.
echo.
echo  USAGE
echo    funnel.cmd        7 days, core markets
echo    funnel.cmd 30     30 days
echo.

set DAYS=%1
if "%DAYS%"=="" set DAYS=7

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe not found. Run update.cmd first.
  pause
  exit /b 1
)

.venv-live\Scripts\python.exe -m scripts.signal_funnel --days %DAYS%

if errorlevel 1 (
  echo.
  echo  The run failed. The traceback above says why.
)

echo.
pause

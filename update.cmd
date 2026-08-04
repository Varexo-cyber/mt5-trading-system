@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Updating the trading system from GitHub
echo  ============================================
echo.

git pull
if errorlevel 1 (
  echo.
  echo  UPDATE FAILED. Nothing was changed.
  echo  Most common cause: local edits that conflict.
  echo  Run "git status" to see what.
  echo.
  pause
  exit /b 1
)

if not exist ".venv-live\Scripts\python.exe" (
  echo.
  echo  Missing .venv-live. Create it once with:
  echo    py -m venv .venv-live
  echo    .venv-live\Scripts\python.exe -m pip install -e ".[live,dashboard]"
  echo.
  pause
  exit /b 1
)

echo.
echo  Installing any new dependencies...
".venv-live\Scripts\python.exe" -m pip install -q -e ".[live,dashboard]"

echo.
echo  Checking the configuration loads...
".venv-live\Scripts\python.exe" main.py --overlay config\eightcap.yaml --check-config
if errorlevel 1 (
  echo.
  echo  CONFIG IS BROKEN. Do not start trading. Send the error above to Claude.
  echo.
  pause
  exit /b 1
)

echo.
echo  --------------------------------------------
echo   Why it has or has not been trading
echo  --------------------------------------------
".venv-live\Scripts\python.exe" scripts\why_no_trades.py --hours 12

echo.
echo  --------------------------------------------
echo   Account and experimental-live approval
echo  --------------------------------------------
".venv-live\Scripts\python.exe" scripts\show_account.py
if errorlevel 1 (
  echo  MT5 is not reachable. Fine if the terminal is closed - open it
  echo  and run this again before starting live trading.
)

echo.
echo  ============================================
echo   Update complete.
echo  ============================================
echo.
echo  If the section above says re-arming is required, run:
echo    rearm_experimental_live.cmd
echo.
pause
endlocal

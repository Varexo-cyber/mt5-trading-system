@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Updating the trading system from GitHub
echo  ============================================
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo.
  echo  Missing .venv-live. Create it once with:
  echo    py -m venv .venv-live
  echo    .venv-live\Scripts\python.exe -m pip install -e ".[live,dashboard,brain]"
  echo.
  pause
  exit /b 1
)

REM Not a bare "git pull". That reports "Already up to date." whenever the
REM current branch's upstream has nothing new - which is true, and badly
REM misleading, when the fetch it just ran did update a different branch. It
REM printed exactly that under a line showing new commits arriving, so an
REM update that changed nothing announced success and everything afterwards
REM looked broken for the wrong reason.
".venv-live\Scripts\python.exe" scripts\update_repo.py
if errorlevel 1 (
  echo.
  echo  UPDATE FAILED. Nothing was changed. The reason is above.
  echo.
  pause
  exit /b 1
)

echo.
echo  Installing any new dependencies...
REM `brain` is in here so the Postgres driver arrives with every update. Left
REM out, `build_brain` quietly returns the null object and the long-term memory
REM stays off after each pull - with nothing failing and nothing saying so.
".venv-live\Scripts\python.exe" -m pip install -q -e ".[live,dashboard,brain]"

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

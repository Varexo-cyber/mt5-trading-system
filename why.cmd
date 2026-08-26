@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Why was that trade taken, and what happened
echo  ============================================
echo.
echo  Everything the journal holds about a trade, in the order it happened:
echo  the modules that argued for it and their scores, the stop and target it
echo  was given, every management action, what it was worth at its best, and
echo  what it actually returned.
echo.
echo  Journal read-only. No orders, no API, nothing written.
echo.
echo  why.cmd                     the last closed trade
echo  why.cmd --list 20           the last twenty, one line each
echo  why.cmd --hours 12          everything from the last 12 hours
echo  why.cmd XAUAUD              that market's last trade
echo  why.cmd --ticket 135035654  one exact broker ticket
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

if "%~1"=="" (
  ".venv-live\Scripts\python.exe" scripts\postmortem.py --hours 6
) else (
  ".venv-live\Scripts\python.exe" scripts\postmortem.py %*
)

echo.
pause
endlocal

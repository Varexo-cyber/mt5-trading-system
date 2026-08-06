@echo off
setlocal
cd /d "%~dp0"

REM Everything on this machine runs out of .venv-live. Typing a bare "python"
REM picks up the system interpreter, which has none of the dependencies, and
REM the failure that produces ("No module named numpy") reads like a broken
REM repository rather than the wrong interpreter. Hence this file.

echo.
echo  ============================================
echo   Management replay - what the rules would
echo   have done to the trades we already took
echo  ============================================
echo.
echo  Reads bars from MT5 and the journal read-only. It sends no orders,
echo  spends nothing on the Claude API and writes nothing anywhere.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

REM %* passes anything you type through, so "replay.cmd GBPAUD" and
REM "replay.cmd --ticket 123" work. With no arguments it does the last
REM twenty closed trades, which is the number worth looking at: one trade
REM improving proves nothing, twenty getting worse is a finding.
if "%~1"=="" (
  ".venv-live\Scripts\python.exe" scripts\replay_trade.py --all 20
) else (
  ".venv-live\Scripts\python.exe" scripts\replay_trade.py %*
)

echo.
pause
endlocal

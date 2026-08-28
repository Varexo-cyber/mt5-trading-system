@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Why has it not traded?
echo  ============================================
echo.
echo  `scripts/why_no_trades.py` has answered this since the day it was
echo  written and had NO LAUNCHER, so the one question an operator asks most
echo  was the one question he could not ask. That is why this file exists.
echo.
echo  The cycle line says "91 analysed, 0 opened" and stops. That invites
echo  exactly one question and answers none of it, and a night that is working
echo  correctly looks identical from the outside to one that is silently
echo  broken. Those two need opposite responses.
echo.
echo  Every refusal is already in the journal with its reason and its numbers.
echo  This reads them back and names the gate that is actually stopping
echo  everything - per reason, with examples, so you can see whether one gate
echo  refused the whole night or thirty gates each refused a little.
echo.
echo  Journal read-only. No orders, no API, nothing written.
echo.
echo  whynot.cmd                  the last 6 hours
echo  whynot.cmd 12               the last 12 hours ^(a bare number means hours^)
echo  whynot.cmd --hours 24       a full day
echo  whynot.cmd --symbol XAUUSD  one market
echo  whynot.cmd --examples 8     more detail per reason
echo.
echo  WORTH KNOWING BEFORE YOU READ IT. Overnight, two things move at once:
echo  the spread widens and the ATR shrinks, so `spread / stop` climbs from
echo  both ends. The gate on that went from 0.08 to 0.03 on 27 August. If the
echo  night was refused on cost, that is the first place to look and it is one
echo  line to loosen.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

REM  A bare number means hours. Same convention as modules.cmd and section6.cmd,
REM  and for the same reason: `--hours12` fails as "unrecognized arguments",
REM  which is a missing space reported as a missing feature.
echo %~1| findstr /r "^[0-9][0-9]*$" >nul
if not errorlevel 1 (
  echo  Reading "%~1" as --hours %~1.
  echo.
  ".venv-live\Scripts\python.exe" scripts\why_no_trades.py --hours %~1
  goto :done
)

if "%~1"=="" (
  ".venv-live\Scripts\python.exe" scripts\why_no_trades.py
) else (
  ".venv-live\Scripts\python.exe" scripts\why_no_trades.py %*
)

:done

echo.
pause
endlocal

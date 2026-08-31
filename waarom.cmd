@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ==================================================
echo   Why has it not traded?
echo  ==================================================
echo.
echo  Reads the LIVE journal and says which gate is actually stopping
echo  everything, with the numbers behind it.
echo.
echo  The console line says "68 analysed, 0 opened" and stops there. That
echo  invites exactly one question and answers none of it -- a night with no
echo  trades because nothing set up and a night with no trades because a gate
echo  is broken look identical from the outside and need opposite responses.
echo.
echo  WHAT THE MEASUREMENT EXPECTS, so you can tell the two apart:
echo.
echo    order_block    M30   about 7 trades a day   (measured over 180 days)
echo    impulse_retest M15   about 2 a day
echo.
echo  Those were measured on 16 markets. Live scans 232, so more is normal
echo  and ZERO over a full session is not.
echo.
echo  USAGE
echo    waarom.cmd            last 6 hours
echo    waarom.cmd 24         last 24 hours
echo    waarom.cmd 24 XAUUSD  last 24 hours, one market
echo.

set HOURS=%1
if "%HOURS%"=="" set HOURS=6
set ONLY=
if not "%2"=="" set ONLY=--symbol %2

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe not found. Run update.cmd first.
  pause
  exit /b 1
)

.venv-live\Scripts\python.exe -m scripts.why_no_trades --hours %HOURS% %ONLY%

if errorlevel 1 (
  echo.
  echo  The run failed. The traceback above says why -- most often the journal
  echo  path differs from the one Jarvis is writing to.
)

echo.
pause

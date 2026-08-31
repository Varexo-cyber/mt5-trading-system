@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ==================================================
echo   Download the bars ONCE and keep them
echo  ==================================================
echo.
echo  Every measurement pulls its own history out of MT5 and pays for it
echo  again. Bars do not change - a closed M15 candle from June is the same
echo  object in September - so they get fetched once and stored.
echo.
echo  180 days over the 16 core markets:
echo.
echo      M1    ~186,000 bars a market     2,970,000 total
echo      M5     ~37,000                     594,000
echo      M15    ~12,400                     198,000
echo      M30     ~6,200                      99,000
echo      H1      ~3,100                      50,000
echo      H4        ~780                      12,000
echo                                      ----------
echo                                      ~3,900,000 bars, 60-150 MB
echo.
echo  RESUMABLE. Each market and clock is written as it arrives and skipped
echo  next time, so if this is interrupted, run it again and it only fetches
echo  what was missing. Nothing is lost.
echo.
echo  Expect the M1 pass to be by far the longest - it is three quarters of
echo  the bars. Leave it running.
echo.
echo  MT5 must be running and logged in FOR THIS ONE. After it, the measuring
echo  commands do not need MT5 at all.
echo.
echo  USAGE
echo    ophalen.cmd                    180 days, all six clocks
echo    ophalen.cmd 365                a year
echo    ophalen.cmd 180 M15 M30 H1 H4  skip M1 and M5 (much faster)
echo.

set DAYS=%1
if "%DAYS%"=="" set DAYS=180
shift

set CLOCKS=
:collect
if "%1"=="" goto collected
set CLOCKS=%CLOCKS% %1
shift
goto collect
:collected
if "%CLOCKS%"=="" set CLOCKS=M1 M5 M15 M30 H1 H4

echo  Window: last %DAYS% days, clocks %CLOCKS%
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe not found. Run update.cmd first.
  pause
  exit /b 1
)

.venv-live\Scripts\python.exe -m scripts.fetch_history --days %DAYS% --timeframes %CLOCKS% --out data\history

if errorlevel 1 (
  echo.
  echo  Some frames failed. Run this again - what succeeded is kept.
)

echo.
pause

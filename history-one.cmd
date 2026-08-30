@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ==================================================
echo   One section, long history
echo  ==================================================
echo.
echo  Same measurement as history.cmd, on ONE section instead of both. Half
echo  the work when the other one has already been measured.
echo.
echo  A section that is switched off is still measured -- that is the whole
echo  point of switching it off rather than deleting it, and this script did
echo  not honour it until 30 August. `impulse_retest` came off the live list
echo  and silently vanished from the 180-day run with it.
echo.
echo  USAGE
echo    history-one.cmd impulse_retest        180 days
echo    history-one.cmd order_block 365       365 days
echo.

set SECTION=%1
if "%SECTION%"=="" set SECTION=impulse_retest
set DAYS=%2
if "%DAYS%"=="" set DAYS=180

echo  Section: %SECTION%
echo  Window : last %DAYS% days, core markets
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe not found. Run update.cmd first.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections --days %DAYS% --core --live-only --no-m1 --only %SECTION% --csv runtime\history-%SECTION%.csv

if errorlevel 1 (
  echo.
  echo  The run failed. The traceback above says why.
)

echo.
echo  Every decision is in runtime\history-%SECTION%.csv
echo.
pause

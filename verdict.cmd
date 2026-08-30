@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ==================================================
echo   The verdict for a run you already did
echo  ==================================================
echo.
echo  Reads a CSV that dry_run_sections already wrote and prints the judgement
echo  block: sigma on daily totals, the monthly table, the concentration check
echo  and the tick boxes.
echo.
echo  WHY IT EXISTS: the report skips all of that when the section measured is
echo  not on the live list, so a shadow run gives its trades and no way to tell
echo  whether they mean anything. Re-running twenty minutes to get arithmetic
echo  that can be done on the file it already wrote is not reasonable.
echo.
echo  USAGE
echo    verdict.cmd runtime\history-impulse_retest.csv
echo    verdict.cmd runtime\history.csv
echo.

set FILE=%1
if "%FILE%"=="" set FILE=runtime\history.csv

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe not found. Run update.cmd first.
  pause
  exit /b 1
)

.venv-live\Scripts\python.exe -m scripts.verdict_from_csv %FILE%

echo.
pause

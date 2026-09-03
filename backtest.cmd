@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Have these five theories ever worked?
echo  ============================================
echo.
echo  Replays every playbook over real MT5 bar history, one closed M5 bar
echo  at a time, and reports what each would have returned on its own.
echo.
echo  No orders. No Claude API. Nothing written anywhere. It only costs time,
echo  and on 90 days of four symbols it takes a while - leave it running.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

rem EEN KAAL GETAL IS EEN AANTAL DAGEN, dezelfde scheefstand als in
rem scorecard.cmd: `backtest.cmd 90` gaf een argparse-usage terug omdat het
rem script alleen --days kent, terwijl elke andere launcher hier een kaal
rem getal aanneemt.
set ARGS=%*
if "%~1"=="" set ARGS=--days 90
set ISGETAL=0
echo %~1| findstr /r /c:"^[0-9][0-9]*$" >nul 2>&1 && set ISGETAL=1
if "%ISGETAL%"=="1" set ARGS=--days %*

".venv-live\Scripts\python.exe" scripts\backtest_playbooks.py %ARGS%

echo.
pause
endlocal

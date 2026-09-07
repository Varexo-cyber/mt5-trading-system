@echo off
setlocal

rem Draait uit een kopie in %TEMP%: cmd leest een batchbestand van schijf
rem TERWIJL het draait, dus een git pull halverwege laat het hervatten midden
rem in een woord. Dat heeft hier al een keer een compleet `set`-blok opgegeten
rem zonder enige melding.
if not defined US30RUW_ROOT set US30RUW_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\us30ruw-actief.cmd" >nul
  call "%TEMP%\us30ruw-actief.cmd" __uittemp %*
  exit /b
)
shift

cd /d "%US30RUW_ROOT%"

set "DAYS=%~1"
if not defined DAYS set "DAYS=180"
if not exist ".venv-live\Scripts\python.exe" (
  echo ERROR: Run update.cmd first. Python runtime is missing.
  pause
  exit /b 1
)
echo.
echo  US30 -- DE BESTAANDE DETECTORS, RAUW. Dit is NIET us30.cmd.
echo    us30ruw.cmd  impulse_retest + order_block_fast, zoals ze nu draaien,
echo                 los op M1 en los op M5. Geen nieuwe secties.
echo    us30.cmd     de VIER nieuwe experimentele secties op US30.
echo  Twee verschillende metingen. Haal de uitkomsten niet door elkaar.
echo.
echo US30: impulse_retest and order_block_fast, M1 and M5 separately.
echo Historical Jarvis replay only. MT5 must be running and logged in.
".venv-live\Scripts\python.exe" -u -m scripts.replay_requested_markets us30 "%DAYS%"
if errorlevel 1 (
  echo ERROR: Replay failed. Read the error above.
  pause
  exit /b 1
)
echo Results: runtime\us30-m1-%DAYS%.csv and runtime\us30-m5-%DAYS%.csv
pause

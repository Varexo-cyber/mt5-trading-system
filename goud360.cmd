@echo off
setlocal

rem Maak een tijdelijke kopie: zo kan update.cmd dit bestand niet halverwege
rem een lange replay veranderen terwijl cmd.exe het nog aan het lezen is.
if not defined GOUD360_ROOT set GOUD360_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\goud360-actief.cmd" >nul
  call "%TEMP%\goud360-actief.cmd" __uittemp
  exit /b
)

cd /d "%GOUD360_ROOT%"

echo.
echo  ========================================================================
echo   360 DAGEN - ALLEEN DE TWEE LIVE GOUDSECTIES - GEEN NDX100
echo  ========================================================================
echo.
echo   Gemeten:
echo     section_six_gold_m5
echo     section_ten_gold_m1
echo.
echo   Niet gemeten: NDX100, BTC en failed-session-breakout.
echo   De twee goudsecties delen hetzelfde positieboek, accountplafond,
echo   dagelijkse verliesstop en dezelfde historische Jarvis-poorten als
echo   hoeveel.cmd. Dit verandert helemaal niets aan live.
echo.
echo   Dit kan circa 30-60 minuten duren. Laat het venster openstaan.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe ontbreekt. Draai eerst update.cmd.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections ^
  --days 360 ^
  --only section_six_gold_m5,section_ten_gold_m1 ^
  --section-markets ^
  --jarvis-replay ^
  --csv runtime\hoeveel-goud-360.csv

if errorlevel 1 (
  echo.
  echo  De run is gestopt. Lees de fout hierboven; er is niets live veranderd.
) else (
  echo.
  echo  Klaar. Alle beslissingen staan in runtime\hoeveel-goud-360.csv
)

echo.
pause

@echo off
cd /d "%~dp0"

rem ============================================================
rem   SETUP.cmd  --  eenmalig. Zet Python en de pakketten klaar.
rem
rem   Draai dit als ALLES.cmd zegt dat er geen werkende Python is.
rem ============================================================

echo.
echo   ============================================================
echo     SETUP  --  Python en pakketten klaarzetten
echo   ============================================================
echo.

rem EERST KIJKEN OF ER EEN ECHTE PYTHON IS.
rem
rem `where python` slaagt op Windows ook zonder Python: er ligt een
rem nep-python.exe van de Microsoft Store in WindowsApps. Die print alleen
rem een reclameregel. Dus we DRAAIEN hem.
set BASIS=
py -3 -c "pass" >nul 2>&1 && set BASIS=py -3
if not defined BASIS ( python -c "pass" >nul 2>&1 && set BASIS=python )
if not defined BASIS ( python3 -c "pass" >nul 2>&1 && set BASIS=python3 )

if not defined BASIS (
  echo   Er staat geen echte Python op deze computer.
  echo.
  echo   Haal hem op bij:   https://www.python.org/downloads/windows/
  echo.
  echo   BELANGRIJK bij het installeren: zet het vinkje aan bij
  echo      "Add python.exe to PATH"
  echo   Dat vinkje staat onderaan het eerste scherm en staat standaard UIT.
  echo.
  echo   Daarna dit bestand opnieuw draaien.
  echo.
  pause
  exit /b 1
)

echo   Python gevonden: %BASIS%
%BASIS% --version
echo.

if not exist ".venv\Scripts\python.exe" (
  echo   Eigen omgeving aanmaken in .venv ...
  %BASIS% -m venv .venv
  if errorlevel 1 (
    echo   Dat is mislukt. Stuur dit scherm door.
    pause
    exit /b 1
  )
)

echo   Pakketten installeren ^(dit duurt een paar minuten^)...
echo.
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"

echo.
echo   MetaTrader5 erbij ^(alleen op Windows te installeren^)...
.venv\Scripts\python.exe -m pip install MetaTrader5

echo.
echo   Controleren of alles laadt...
.venv\Scripts\python.exe -c "import pandas, numpy, pydantic, yaml; print('  kern: OK')"
.venv\Scripts\python.exe -c "import MetaTrader5; print('  MetaTrader5: OK')"

echo.
echo   ============================================================
echo     KLAAR. Draai nu ALLES.cmd
echo   ============================================================
echo.
pause

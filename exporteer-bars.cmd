@echo off
cd /d "%~dp0"

rem DE BARS UIT MT5 NAAR EEN BESTAND.
rem
rem MetaTrader5 draait alleen op Windows, dus alleen jouw PC kan bars ophalen.
rem Dit script haalt ze EEN KEER op en schrijft ze naar een CSV. Daarna kan
rem iedereen -- ook Claude in de cloud -- exact dezelfde meting draaien.
rem
rem En het is herhaalbaar: twee runs rechtstreeks tegen de terminal meten twee
rem verschillende reeksen, want er is een bar bijgekomen. Twee runs tegen
rem hetzelfde bestand meten dezelfde bars.
rem
rem 180 dagen M1 is ongeveer 190.000 regels en een dikke 10 MB. ZIP HEM voor je
rem hem verstuurt.

set DAGEN=%1
if "%DAGEN%"=="" set DAGEN=180

set PY=
if exist ".venv-live\Scripts\python.exe" set PY=.venv-live\Scripts\python.exe
if not defined PY if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if not defined PY (where py >nul 2>&1 && set PY=py -3)
if not defined PY (where python >nul 2>&1 && set PY=python)
if not defined PY (
  echo   Geen Python gevonden.
  pause
  exit /b 1
)

%PY% -m scripts.exporteer_bars --days %DAGEN%

echo.
echo   Controleer hierboven of "gaten ^>12u" laag is. Een gat van drie dagen
echo   midden in de week betekent dat je terminal zijn historie niet compleet
echo   had -- dan meet je een markt die er niet was.
echo.
pause

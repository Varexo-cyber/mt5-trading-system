@echo off
cd /d "%~dp0"

rem SECTIE 20 -- de terugval-ladder meten. Handelt niets.
rem
rem Dit is een GRID MET AVERAGING DOWN. `risk_manager.assert_not_forbidden`
rem weigert dat vandaag met zoveel woorden, op verzoek van de eigenaar zelf.
rem Deze meting haalt die regel niet weg; ze beantwoordt alleen de vraag of hij
rem er terecht staat.
rem
rem LEES DE UITSLAG VAN LINKS NAAR RECHTS. Ruine, diepste stand en meeste benen
rem staan VOOR trefkans en winst, en dat is geen opmaak. Een grid haalt bijna
rem altijd een trefkans boven de 90% met een positief totaal -- tot de dag dat
rem de rekening weg is. Staat er ergens RUINE, dan is de rest van die regel niet
rem meer interessant.
rem
rem --balans is verplicht en moet je ECHTE balans zijn. De ruinekans hangt
rem eraan: dezelfde ladder is op 5000 euro een schrammetje en op 400 euro het
rem einde.

set BALANS=%1
if "%BALANS%"=="" (
  echo.
  echo   Gebruik: ladder.cmd ^<balans-in-euro^> [dagen]
  echo.
  echo   Bijvoorbeeld:  ladder.cmd 400
  echo                  ladder.cmd 400 365
  echo.
  echo   De balans is verplicht omdat de ruinekans eraan hangt.
  echo.
  pause
  exit /b 1
)

set DAGEN=%2
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

echo.
echo   Eerst de gevraagde regel zelf, daarna de uitsplitsing per jaar,
echo   per sessie en per uur.
echo.
%PY% -m scripts.section_twenty_pullback_ladder --balans %BALANS% --days %DAGEN%

echo.
echo   En nu alle 72 configuraties. De beste van 72 is iets anders dan een
echo   ontdekking, dus dat getal hoort bij de uitslag te blijven staan.
echo.
%PY% -m scripts.section_twenty_pullback_ladder --balans %BALANS% --days %DAGEN% --alle-configs

echo.
pause

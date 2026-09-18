@echo off
cd /d "%~dp0"

rem SECTIE 21 -- de straddle meten. Handelt niets.
rem
rem 0,01 buy EN 0,01 sell tegelijk is geen hedge maar een STRADDLE: je dekt
rem niets af, je betaalt om te ontdekken welke kant het op gaat.
rem `risk_manager.assert_not_forbidden` weigert het vandaag, en die regel blijft
rem staan. Deze meting beantwoordt alleen of hij er terecht staat.
rem
rem WAAR JE NAAR KIJKT, en het is maar EEN kolom:
rem
rem     straddle   een_richting   altijd_long
rem
rem Verslaat `straddle` de kolom `een_richting` niet -- zelfde momenten,
rem zelfde doel, zelfde stop, maar een been -- dan is dat tweede been ZUIVERE
rem KOSTENPOST en is de vraag beantwoord zonder dat er iets live hoeft.
rem `altijd_long` staat er omdat goud steeg: dan verdient long in elk venster
rem geld, en zonder die kolom lees je de trend als een strategie.
rem
rem EN ONDERAAN STAAT "BESTE UUR" MET EEN p-WAARDE. Het beste van 24 uren is
rem het maximum van 24 ruizige getallen. Staat daar p boven 0,05, dan is dat
rem uur geen vondst maar een selectie -- ook al ziet het er nog zo goed uit.

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

echo.
echo   Eerst de gevraagde regel, met de uitsplitsing per jaar, sessie en uur.
echo.
%PY% -m scripts.section_twentyone_straddle --days %DAGEN%

echo.
echo   En nu alle 48 configuraties. "STRADDLE WINT" staat er alleen als hij
echo   ook echt boven de eenzijdige controle uitkomt.
echo.
%PY% -m scripts.section_twentyone_straddle --days %DAGEN% --alle-configs

echo.
pause

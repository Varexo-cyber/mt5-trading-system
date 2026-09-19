@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem ============================================================
rem   WELK UUR EN WELKE SESSIE WERKEN HET BEST.
rem
rem   Dubbelklik dit bestand. Meer niet.
rem
rem   Twee stappen: eerst de shadow-meting over 730 dagen, die
rem   elke beslissing naar een CSV schrijft. Daarna de
rem   uitsplitsing per uur, per sessie en per weekdag, met de
rem   permutatietest erbij die zegt of het beste uur echt is.
rem
rem   WAAROM DIT NAAST sectie7.cmd STAAT EN HEM NIET VERVANGT.
rem   `sectie7.cmd` eist `.venv-live\Scripts\python.exe` en stopt
rem   als die er niet is. Op deze machine staat `.venv`, en dan
rem   slaat hij af met "draai eerst update.cmd" terwijl er niets
rem   mis is. Hieronder wordt Python gezocht en ECHT GEPROBEERD,
rem   net als in ALLES.cmd.
rem ============================================================

set DAGEN=%1
if "%DAGEN%"=="" set DAGEN=730

set UIT=runtime\uurmeting.txt
set BESLISSINGEN=runtime\sectie7-smc.csv
if not exist runtime mkdir runtime

echo.
echo   ==================================================
echo     UURMETING  --  %DAGEN% dagen, dit duurt even
echo   ==================================================
echo.
echo   Zorg dat MetaTrader 5 OPEN staat en ingelogd is.
echo.
pause

rem PYTHON ZOEKEN EN HEM ECHT DRAAIEN. `where python` slaagt op Windows ook
rem als er alleen een Microsoft Store-stub staat die een reclameregel print.
rem De enige betrouwbare test is de exitcode van een echte aanroep.
set PY=
if exist ".venv-live\Scripts\python.exe" (
  .venv-live\Scripts\python.exe -c "pass" >nul 2>&1 && set PY=.venv-live\Scripts\python.exe
)
if not defined PY if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe -c "pass" >nul 2>&1 && set PY=.venv\Scripts\python.exe
)
if not defined PY ( py -3 -c "pass" >nul 2>&1 && set PY=py -3 )
if not defined PY ( python -c "pass" >nul 2>&1 && set PY=python )

if not defined PY (
  echo.
  echo   GEEN WERKENDE PYTHON GEVONDEN. Draai eerst SETUP.cmd
  echo.
  pause
  exit /b 1
)
echo   Python gevonden: %PY%
echo.

rem DE VOORTGANG HOORT OP HET SCHERM.
rem
rem Hier stond `> "%UIT%" 2>&1` achter deze aanroep, en dat maakte het scherm
rem leeg terwijl er een meting van uren liep. `dry_run_sections` print onderweg
rem wel degelijk waar hij is, maar dat verdween allemaal in het bestand -- en
rem een stilstaand scherm is niet te onderscheiden van een vastgelopen
rem programma. Dat is precies het bezwaar waar `eindresultaat.py` al een
rem `melding()` voor heeft.
rem
rem De CSV is hier het resultaat dat telt, niet de tekst, dus die tekst mag
rem gewoon meelopen op het scherm.
echo   [1/2] shadow-meting over %DAGEN% dagen ...
echo   (elke beslissing gaat naar %BESLISSINGEN%)
echo   Dit is de lange stap. 730 dagen x vijf klokken is veel werk; laat hem staan.
echo.
%PY% -m scripts.dry_run_sections ^
  --days %DAGEN% ^
  --only section_seven_gold_smc ^
  --section-markets ^
  --sweep M1 M5 M15 M30 H1 ^
  --jarvis-replay ^
  --fixed-exits ^
  --strict-risk ^
  --csv "%BESLISSINGEN%"

if not exist "%BESLISSINGEN%" (
  echo.
  echo   !! De meting heeft geen CSV geschreven. De reden staat hierboven
  echo      op het scherm -- scroll terug naar de eerste foutregel.
  pause
  exit /b 1
)

echo.
echo   [2/2] uitsplitsen per uur, sessie en weekdag ...
%PY% -m scripts.per_uur_en_sessie --csv "%BESLISSINGEN%" --uit "%UIT%"

echo.
echo   Klaar. Alles staat in %UIT%
echo.
notepad "%UIT%"

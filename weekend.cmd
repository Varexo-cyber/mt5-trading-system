@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Alles van deze week in een bestand
echo  ============================================
echo.
echo  Vier rapporten achter elkaar, allemaal uit het journaal, allemaal
echo  read-only. Geen orders, geen API, geen marktdata nodig - dit draait dus
echo  gewoon met de markten dicht.
echo.
echo    health      leeft elke laag nog, en heeft een detector zichzelf uitgezet
echo    whynot      de trechter: hoeveel beslissingen, hoeveel setups, welke
echo                poort ze doodmaakte, met voorbeelden per reden
echo    why         elke gesloten trade: welke modules, welk plan, elke
echo                management-actie, wat hij op zijn best was en wat hij werd
echo    scorecard   per detector en per regime over zeven dagen
echo.
echo  WAAROM DIT EEN COMMANDO IS. Ze bestonden alle vier al los en er moest
echo  precies onthouden worden welke, in welke volgorde en met welke vlaggen.
echo  Dat is het soort ding dat een keer gebeurt en daarna niet meer.
echo.
echo  Het gaat naar een tekstbestand in `reports\`, met de datum in de naam.
echo  Dat bestand is wat je doorstuurt.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

if not exist "reports" mkdir "reports"

REM  Datum en tijd in de bestandsnaam, en de streepjes eruit zodat Windows er
REM  geen bezwaar tegen maakt. %date% is landinstelling-afhankelijk, dus dit
REM  komt uit Python en niet uit cmd - hetzelfde script dat de rapporten
REM  schrijft bepaalt ook hoe ze heten.
for /f %%s in ('.venv-live\Scripts\python.exe -c "from datetime import datetime,UTC;print(datetime.now(UTC).strftime('%%Y%%m%%d-%%H%%M'))"') do set STAMP=%%s
set OUT=reports\weekend-%STAMP%.txt

echo  Schrijft naar %OUT%
echo.

> "%OUT%" echo ================ HEALTH ================
>>"%OUT%" ".venv-live\Scripts\python.exe" scripts\selfcheck.py 2>&1

>>"%OUT%" echo.
>>"%OUT%" echo ================ WHYNOT 24H ================
>>"%OUT%" ".venv-live\Scripts\python.exe" scripts\why_no_trades.py --hours 24 --examples 5 2>&1

>>"%OUT%" echo.
>>"%OUT%" echo ================ TRADES ================
>>"%OUT%" ".venv-live\Scripts\python.exe" scripts\postmortem.py --list 40 2>&1

>>"%OUT%" echo.
>>"%OUT%" echo ================ SCORECARD 7D ================
>>"%OUT%" ".venv-live\Scripts\python.exe" scripts\scorecard.py --days 7 2>&1

echo.
echo  ============================================
echo  Klaar. Het bestand staat hier:
echo.
echo     %CD%\%OUT%
echo.
echo  Stuur dat bestand door. Hieronder alvast de health-regels, zodat je
echo  meteen ziet of er iets stuk is:
echo  ============================================
echo.

".venv-live\Scripts\python.exe" scripts\selfcheck.py

echo.
pause
endlocal

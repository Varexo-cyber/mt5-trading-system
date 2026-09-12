@echo off
setlocal

rem Uit een kopie in %TEMP%, net als de andere launchers: cmd leest een
rem batchbestand van schijf TERWIJL het draait, dus een git pull halverwege
rem laat het hervatten midden in een woord.
if not defined S18_ROOT set S18_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\sectie18-actief.cmd" >nul
  call "%TEMP%\sectie18-actief.cmd" __uittemp %*
  exit /b
)
shift

cd /d "%S18_ROOT%"

rem STANDAARD DE ONTWIKKELPERIODE, niet de holdout. Die holdout mag precies EEN
rem keer bekeken worden en dat hoort een bewuste handeling te zijn, geen
rem bijvangst van een run die je toch al deed.
set VENSTER=--days 360
set TAG=ontwikkel
:lees
if "%~1"=="" goto klaar
if /i "%~1"=="holdout" set VENSTER=--start-date 2024-09-01 --end-date 2025-08-31
if /i "%~1"=="holdout" set TAG=HOLDOUT
shift
goto lees
:klaar

echo.
echo  ==========================================================================
echo   SECTIE ACHTTIEN -- MEEDOEN MET DE GOUDTREND  (%TAG%)
echo  ==========================================================================
echo.
echo  WAAROM DEZE SECTIE BESTAAT. domtest.cmd holdout gaf over het onaangeraakte
echo  jaar 1 sep 2024 - 31 aug 2025:
echo.
echo      de hele dag long goud, 256 dagen        +1250,93 R
echo      sectie zes over dezelfde periode           +4,21 R
echo.
echo  Goud maakte een van zijn grootste bewegingen ooit. Sectie zes is LONG-ONLY
echo  goud. Hij stond de goede kant op en ving er drie tiende procent van.
echo.
echo  Dat is geen kostenprobleem en geen modelprobleem. Het is een
echo  DEELNAMEprobleem: alle 44 bestaande modules zijn intraday en dus per
echo  definitie afwezig terwijl een beweging van maanden zich voltrekt.
echo.
echo  WAT DEZE SECTIE DOET. Long zolang de dagslotkoers boven zijn 50-daags
echo  gemiddelde staat. Stop op 2 x de dagelijkse ATR. Geen target. Uitstappen
echo  als het gemiddelde breekt of de stop raakt. Dat is alles.
echo.
echo  TWEE VRIJE PARAMETERS, en dat is het ontwerp. `trend_bars` en `stop_atr`.
echo  De hypothese legt zes toegestane combinaties vast en de Bonferroni-lat
echo  voor zes is t^>2,64. Sectie zes had veertig knoppen en heeft nooit een
echo  eerlijke lat gehaald omdat niemand ooit telde hoeveel er geprobeerd was.
echo.
echo  DE LAT IS NIET DE WINST MAAR DE WINST PER TERUGVAL. Kopen-en-vasthouden
echo  gaf 1250,93 / 295,08 = 4,24, en is onbruikbaar omdat 295 R een veelvoud
echo  van deze rekening is. Deze sectie moet boven 2,12 uitkomen of hij is een
echo  dure manier om goud te kopen.
echo.
echo  De volledige vooraf vastgelegde hypothese, met de voorspellingen EN de
echo  faalvoorwaarden, staat in:
echo      docs\hypotheses\section_eighteen_gold_trend_d1.md
echo.
echo  Dat is het eerste document in dit project dat in die volgorde geschreven
echo  is -- eerst de voorspelling, dan pas de meting.
echo.
echo  GEBRUIK
echo    sectie18.cmd           de ontwikkelperiode, 360 dagen
echo    sectie18.cmd holdout   HET ONAANGERAAKTE JAAR -- precies EEN keer
echo.
echo  DIE HOLDOUT IS EENMALIG. Draai hem als de ontwikkelperiode iets laat zien
echo  dat de moeite waard is, en daarna niet meer. Twee keer kijken en de
echo  gunstigste houden is exact de fout die sectie vijf drie keer van teken
echo  deed wisselen.
echo.
echo  DIT KAN GEEN GELD UITGEVEN. De sectie staat niet op live_enabled_modules
echo  en draagt gewicht nul.
echo.
echo  D1-bars zijn verplicht. MT5 moet draaien en ingelogd zijn.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe ontbreekt. Draai eerst update.cmd.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections ^
  %VENSTER% ^
  --only section_eighteen_gold_trend_d1 ^
  --section-markets ^
  --jarvis-replay ^
  --csv runtime\sectie18-%TAG%.csv

if errorlevel 1 (
  echo.
  echo  De run is gestopt. De traceback hierboven zegt waarom -- meestal is MT5
  echo  niet ingelogd, of er is geen D1-historie zo ver terug.
  pause
  exit /b 1
)

echo.
echo  Elke beslissing staat in runtime\sectie18-%TAG%.csv
echo.
echo  DAARNA: `kosten.cmd runtime\sectie18-%TAG%.csv` geeft bruto tegen netto,
echo  zodat je ziet of de instap iets ziet en wat het handelen ervan kost.
echo.
echo  WAT JE MOET AFVINKEN, uit de hypothese en niet uit je onderbuik:
echo    * minstens 30 trades
echo    * trefkans tussen 30%% en 55%%  (hoger betekent dat hij iets anders doet)
echo    * winst per terugval boven 2,12
echo    * positief op de holdout
echo.
pause

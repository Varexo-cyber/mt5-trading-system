@echo off
setlocal

rem Draait uit een kopie in %TEMP%: cmd leest een batchbestand van schijf
rem TERWIJL het draait, dus een git pull halverwege laat het hervatten midden
rem in een woord. Dat heeft hier al een keer een compleet `set`-blok opgegeten
rem zonder enige melding.
if not defined US30_ROOT set US30_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\us30-actief.cmd" >nul
  call "%TEMP%\us30-actief.cmd" __uittemp %*
  exit /b
)
shift

cd /d "%US30_ROOT%"

rem Elke variabele hier, bovenaan, voordat iets hem leest. Een niet-gezette
rem %VAR% in cmd is een lege string en dat is stil.
set DAGEN=180
set SECTIES="section_us30_impulse_m1,section_us30_impulse_m5,section_us30_orderblock_m1,section_us30_orderblock_m5"
rem LEEG = zoals geconfigureerd. `vast` zet er --fixed-exits achter: de
rem trade loopt naar zijn instapstop of zijn target en niets grijpt in.
set EXITS=
set EXITNAAM=zoals geconfigureerd
rem EIGEN BESTANDSNAAM PER STAND. Zonder dit schrijven beide standen naar
rem runtime\us30.csv en overschrijft de laatste run de vorige, waarna twee
rem verschillende metingen een naam delen.
set CSVTAG=

:lees
if "%~1"=="" goto klaar
echo %~1| findstr /r "^[0-9][0-9]*$" >nul && set DAGEN=%~1
if /i "%~1"=="impuls" set SECTIES="section_us30_impulse_m1,section_us30_impulse_m5"
if /i "%~1"=="block" set SECTIES="section_us30_orderblock_m1,section_us30_orderblock_m5"
if /i "%~1"=="m1" set SECTIES="section_us30_impulse_m1,section_us30_orderblock_m1"
if /i "%~1"=="m5" set SECTIES="section_us30_impulse_m5,section_us30_orderblock_m5"
if /i "%~1"=="vast" set EXITS=--fixed-exits
if /i "%~1"=="vast" set EXITNAAM=VAST -- stop en target, verder niets
if /i "%~1"=="vast" set CSVTAG=-vast
shift
goto lees
:klaar

echo.
echo  ==========================================================================
echo   US30 -- VIER EXPERIMENTELE SECTIES, %DAGEN% DAGEN
echo  ==========================================================================
echo.
echo  WAT DIT IS. Vier nieuwe secties op US30, op de twee mechanismen die dit
echo  account al gemeten heeft:
echo.
echo    section_us30_impulse_m1      impulse-retest   M1
echo    section_us30_impulse_m5      impulse-retest   M5
echo    section_us30_orderblock_m1   order-block      M1
echo    section_us30_orderblock_m5   order-block      M5
echo.
echo  ALLE VIER SCHADUW. Ze staan NIET op `live_enabled_modules`, hun gewicht
echo  is nul, en er is geen bestaande live sectie voor aangepast. Deze run
echo  kost geen geld en kan er geen uitgeven.
echo.
echo  DEZELFDE REPLAYVORM ALS hoeveel.cmd -- dat is het punt, anders zijn de
echo  uitkomsten niet vergelijkbaar met de rest van het boek:
echo    * echte Eightcap-bars, echte spread per bar, echte minimum lot
echo    * dezelfde ConfluenceEngine, PositionSizer en TradeManagement
echo    * spread tegenover stopafstand      SPREAD_EATS_THE_STOP
echo    * markt-te-stil / liveliness        MARKET_TOO_QUIET
echo    * M1 volumepiek                     VOLUME_SPIKE
echo    * target-reach en richtingsvoordeel TARGET_RARELY_REACHED
echo    * EEN gedeeld positieplafond over alle secties, op tijdvolgorde
echo    * break-even zoals geconfigureerd
echo.
echo  WAT ER NIET IN ZIT: het nieuwsblok (geen kalenderarchief van dit venster),
echo  de AI-review, en slippage voorbij de geboekte spread. Alle drie HALEN
echo  alleen trades weg, dus de uitkomst is een bovengrens.
echo.
echo  DE PARAMETERS ZIJN DE BESTAANDE DEFAULTS, veld voor veld -- ATR 14,
echo  impuls 1,0 / 1,5 ATR, span 1,5, tolerantie 0,15 / 0,25, stop 0,85 / 1,0,
echo  lookback 96, vijf bars terug. Dat is opzet: de vraag is wat de GEMETEN
echo  mechanismen doen op een markt en twee klokken waar ze nooit op gemeten
echo  zijn.
echo.
echo  EN DAT IS METEEN HET VOORBEHOUD. Die getallen komen van M15 en M30, op
echo  FX en index-CFDs. De kosten van een trade tegen een stop van een ATR
echo  groeien naarmate de klok krimpt -- precies wat sectie elf van +0,047 R
echo  per trade naar -0,18 bracht. Verwacht dat M1 het zwaarst valt.
echo.
echo  GEBRUIK
echo    us30.cmd 180          alle vier, 180 dagen
echo    us30.cmd 90           alle vier, 90 dagen
echo    us30.cmd 180 impuls   alleen de twee impulse-retest secties
echo    us30.cmd 180 block    alleen de twee order-block secties
echo    us30.cmd 180 m1       alleen de twee M1-secties
echo    us30.cmd 180 m5       alleen de twee M5-secties
echo    us30.cmd 180 vast     zonder beheer: alleen instapstop en target
echo.
echo  UITSTAP: %EXITNAAM%
echo.
echo  `vast` HAALT AL HET BEHEER WEG -- geen break-even, geen partial, geen
echo  trailing, geen profit lock, geen peak-stall, geen tijdsexit, geen
echo  avondsluiting. Alleen de instapstop en het target. Dat is de eerlijke
echo  vraag voor een nieuw mechanisme: verdient de INSTAP geld, of komt het
echo  van een regel die er achteraf overheen ligt. Alleen de replay verandert;
echo  aan de live rekening wordt niets aangeraakt.
echo.
echo  MT5 moet draaien en ingelogd zijn, met US30 zichtbaar in Market Watch.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe ontbreekt. Draai eerst update.cmd.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections --days %DAGEN% --section-markets --only %SECTIES% --jarvis-replay %EXITS% --csv runtime\us30%CSVTAG%.csv

if errorlevel 1 (
  echo.
  echo  De run is gestopt. De traceback hierboven zegt waarom -- meestal is MT5
  echo  niet ingelogd, of er is geen historie zo ver terug.
)

echo.
echo  Elke afzonderlijke beslissing staat in runtime\us30%CSVTAG%.csv
echo  Opnieuw lezen zonder opnieuw te rekenen:  lees.cmd runtime\us30%CSVTAG%.csv
echo.
echo  WAT JE MOET LEZEN, in deze volgorde:
echo    1. BY SECTION -- trades, R en EUR per sectie afzonderlijk.
echo    2. IS THIS REAL -- sigma, de maandtabel, en of een maand het hele
echo       resultaat draagt. Vier secties op een markt is vier keer zoeken,
echo       dus een enkele positieve uitkomst zegt hier minder dan gewoonlijk.
echo    3. WHY NOTHING HAPPENED -- elke weigering met naam en aantal.
echo    4. THE COST WALL -- op M1 is dit de regel die beslist.
echo    5. GATE COUNTERFACTUALS -- wat de geweigerde setups gedaan zouden
echo       hebben. Een sectie die alleen leeft doordat een poort hem redt is
echo       geen sectie.
echo.
pause

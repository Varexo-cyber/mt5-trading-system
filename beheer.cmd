@echo off
setlocal

rem Draait uit een kopie in %TEMP%: cmd leest een batchbestand van schijf
rem TERWIJL het draait, dus een git pull halverwege laat het hervatten midden
rem in een woord. Dat heeft hier al een keer een compleet `set`-blok opgegeten
rem zonder enige melding.
if not defined BEHEER_ROOT set BEHEER_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\beheer-actief.cmd" >nul
  call "%TEMP%\beheer-actief.cmd" __uittemp %*
  exit /b
)
shift

cd /d "%BEHEER_ROOT%"

rem Elke variabele hier, bovenaan, voordat iets hem leest. Een niet-gezette
rem %VAR% in cmd is een lege string en dat is stil.
set DAGEN=180
set MARKTEN=--section-markets
set BOEK=--live-only
set GRID=kern
rem TWEE LOSSE STUKJES NAAM, en dat is geen netheid maar een bug die er al in
rem zat: `goud alles` liet de tweede keuze de naam van de eerste overschrijven,
rem dus een run over twee secties met 78 regels heette hetzelfde als een run
rem over alle secties met 78 regels. Aparte stukjes kunnen dat niet.
set SECTAG=
set GRIDTAG=
rem DE KOMMALIJST STAAT TUSSEN AANHALINGSTEKENS, en dat is niet cosmetisch.
rem Zonder quotes hangt het van cmd af of `--only a,b` als een argument bij
rem python aankomt of als twee, en in het tweede geval stopt argparse met
rem "unrecognized arguments" NA de vraag om historie. Met quotes kan het niet
rem splitsen, welk gedrag ook waar is.
rem LEEG = alle live secties. Een keuze hieronder beperkt het tot een of twee
rem en dat scheelt enorm: sectie zes en tien handelen allebei alleen goud, dus
rem `goud` loopt EEN markt in plaats van de vijf die de hele lijst nodig heeft.
set SECTIES=

:lees
if "%~1"=="" goto klaar
echo %~1| findstr /r "^[0-9][0-9]*$" >nul && set DAGEN=%~1
if /i "%~1"=="alles" set GRID=alles
if /i "%~1"=="alles" set GRIDTAG=-alles
if /i "%~1"=="kern" set GRID=kern
if /i "%~1"=="schaduw" set BOEK=
if /i "%~1"=="markten" set MARKTEN=--core
if /i "%~1"=="zes" set SECTIES=--only section_six_gold_m5
if /i "%~1"=="zes" set SECTAG=-s6
if /i "%~1"=="tien" set SECTIES=--only section_ten_gold_m1
if /i "%~1"=="tien" set SECTAG=-s10
if /i "%~1"=="goud" set SECTIES=--only "section_six_gold_m5,section_ten_gold_m1"
if /i "%~1"=="goud" set SECTAG=-goud
if /i "%~1"=="vijf" set SECTIES=--only section_five_ndx100_m5
if /i "%~1"=="vijf" set SECTAG=-s5
shift
goto lees
:klaar
set CSVTAG=%SECTAG%%GRIDTAG%

echo.
echo  ==========================================================================
echo   WELKE UITSTAP IS DE BESTE, PER SECTIE -- %DAGEN% DAGEN
echo  ==========================================================================
echo.
echo  DIT IS hoeveel.cmd MET EEN VRAAG ERBIJ. Dezelfde secties, dezelfde
echo  markten, dezelfde klokken, dezelfde poorten, hetzelfde gedeelde
echo  positieboek. Het verschil: elke genomen trade wordt daarna NOG EEN KEER
echo  uitgelopen onder elke manier van beheren, en de uitkomsten staan naast
echo  elkaar.
echo.
echo  OP IDENTIEKE INSTAPS, en dat is het hele punt. Een regel die eerder
echo  uitstapt maakt het symbool eerder vrij, een vrij symbool pakt de
echo  volgende setup, en dan vergelijk je twee STRATEGIEEN in plaats van twee
echo  uitstappen. Hier ziet elke regel exact dezelfde trades.
echo.
echo  WAT ER VERGELEKEN WORDT
echo    * vaste SL/TP                 -- geen beheer, de instapstop en target
echo    * break-even op 0,10 / 0,15 / 0,20 / 0,25 / 0,35 / 0,50 / 0,75 /
echo      1,00 / 1,25 / 1,50 / 2,00 R
echo      en per trigger: stop naar entry, naar +0,1R, of naar +0,10 x H1-ATR
echo    * trailing stop op meerdere ATR-afstanden, vanaf meerdere R
echo    * deelafname (partial) op meerdere R en meerdere fracties
echo    * profit lock vanaf meerdere R
echo.
echo  R EN ATR ALLEBEI, met opzet. De live regel is `break_even_offset_atr`
echo  maal de H1-ATR, en dat is GEEN vaste fractie van de stop: op sectie zes
echo  is diezelfde 0,10 ongeveer 0,44 R, op een H1-stop eerder 0,10 R. Alleen
echo  in R meten beantwoordt dus een vraag die deze rekening niet stelt.
echo.
echo  DE UITKOMST IS EEN OORDEEL, GEEN LIJSTJE. Per sectie staat er onderaan
echo  of er een winnaar is. Een regel telt alleen als winnaar als hij:
echo    1. de vaste uitstap verslaat over de hele periode, EN
echo    2. hem OOK verslaat in de vroege helft en in de late helft apart, EN
echo    3. een gepaarde t haalt die is opgehoogd voor het aantal regels.
echo.
echo  DAT DERDE PUNT IS HET BELANGRIJKSTE EN HET WORDT NOOIT GEDAAN. Dertig
echo  regels proberen en de beste houden is net zoveel kansen om voor de gek
echo  gehouden te worden. Op pure ruis haalt de beste van dertig moeiteloos
echo  t=2,1 -- wat er "significant" uitziet en niets is. De lat gaat daarom
echo  mee omhoog met het aantal regels. Verwacht vaak: GEEN winnaar. Dat is
echo  dan het antwoord, niet een mislukte run.
echo.
echo  EN OOK ALS ER EEN WINNAAR IS, is dat geen groen licht. Hij heeft een
echo  volledige replay verdiend waarin het positieboek hem WEL volgt. Pas daar
echo  blijkt of hij overeind blijft als hij ook andere trades gaat pakken.
echo.
echo  WAT ER NIET IN ZIT: het nieuwsblok (geen kalenderarchief van dit
echo  venster), de AI-review, en slippage voorbij de geboekte spread. Alle
echo  drie halen alleen trades weg, dus de uitkomst is een bovengrens.
echo.
echo  GEBRUIK
echo    beheer.cmd 180           alle live secties, 38 regels per stuk
echo    beheer.cmd 180 goud      ALLEEN sectie 6 en 10 -- veruit het snelst
echo    beheer.cmd 180 zes       alleen sectie 6  (goud M5)
echo    beheer.cmd 180 tien      alleen sectie 10 (goud M1)
echo    beheer.cmd 180 vijf      alleen sectie 5  (NDX100 M5)
echo    beheer.cmd 180 alles     78 regels per sectie -- fors langer
echo    beheer.cmd 90            90 dagen
echo    beheer.cmd 180 schaduw   ook de secties die geen geld mogen gebruiken
echo    beheer.cmd 180 markten   de zestien kernmarkten erbij
echo.
echo  De keuzes zijn te combineren: `beheer.cmd 180 goud alles` meet sectie 6
echo  en 10 tegen alle 78 regels. Elke keuze schrijft naar zijn eigen CSV, dus
echo  een run overschrijft een vorige niet.
echo.
echo  DIT DUURT LANG. hoeveel.cmd loopt de bars een keer per trade; deze loopt
echo  ze nog 37 keer extra (of 77 met `alles`). Zet hem aan en laat hem staan.
echo  `goud` scheelt het meest: een markt in plaats van vijf.
echo.
echo  M1-bars zijn verplicht: de volumepoort leest M1 en niets anders.
echo  MT5 moet draaien en ingelogd zijn.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe ontbreekt. Draai eerst update.cmd.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections --days %DAGEN% %MARKTEN% %BOEK% --jarvis-replay --exit-grid %GRID% %SECTIES% --csv runtime\beheer%CSVTAG%.csv

if errorlevel 1 (
  echo.
  echo  De run is gestopt. De traceback hierboven zegt waarom -- meestal is MT5
  echo  niet ingelogd, of er is geen historie zo ver terug.
)

echo.
echo  Elke afzonderlijke beslissing staat in runtime\beheer%CSVTAG%.csv
echo.
echo  WAT JE MOET LEZEN: het blok EXIT GRID. Per sectie een tabel, en onder
echo  elke tabel een regel die begint met VERDICT. Die regel is het antwoord.
echo  Staat er "keep the exit as configured", dan heeft geen enkele regel de
echo  lat gehaald en blijft alles zoals het staat.
echo.
pause

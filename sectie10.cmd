@echo off
setlocal

rem Draait uit een kopie in %TEMP%, net als zoek11.cmd: cmd leest een
rem batchbestand van schijf terwijl het draait, dus een git pull halverwege
rem laat het hervatten midden in een woord. Zie zoek11.cmd voor het hele
rem verhaal -- dat kostte een run van drie uur.
rem De aanroep geeft %* door zodat een langere aanroep niet stil afkapt.
if not defined S10_ROOT set S10_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\sectie10-actief.cmd" >nul
  call "%TEMP%\sectie10-actief.cmd" __uittemp %*
  exit /b
)
shift

cd /d "%S10_ROOT%"

echo.
echo  ==================================================================
echo   SECTIE TIEN -- MEER TRADES ZONDER DE KWALITEIT TE VERNEUKEN
echo  ==================================================================
echo.
echo  WAAROM JUIST DEZE SECTIE. Scorecard 30 dagen, echte rekening:
echo.
echo     section_ten_gold_m1    16 trades   +2,49 R    +0,156 per trade
echo     section_six_gold_m5     3 trades   -0,72 R
echo     drift_continuation    108 trades  -38,43 R
echo     het hele boek         361 trades  -29,49 R
echo.
echo  Sectie tien is de enige live component met een positieve gemeten edge
echo  per trade, en veruit de beste. Hij maakt maar een halve trade per dag.
echo  Dat is dus de plek waar meer trades iets waard zijn -- overal anders
echo  betekent meer trades meer verlies.
echo.
echo  WAT HEM NU AFREMT, in volgorde van grootte:
echo    1. EEN markt. allowed_symbols stond op [XAUUSD]; sinds 3 september
echo       staan er zes in: XAUUSD, XAUEUR, XAUGBP, XAUAUD, XAUJPY, XAGUSD.
echo    2. DE UREN. Stond op 03:00-19:00 UTC met 07:00-13:00 dicht. Dat blok
echo       is er per 3 september AF: die zes uur zijn gekozen door dezelfde
echo       180 dagen op te delen waarop de sectie gekalibreerd is, en de
echo       slechtste zes eruit knippen vindt op ELKE reeks een slecht blok.
echo       Het zijn bovendien de Londense ochtenduren, de drukste die goud
echo       heeft. Deze run print een uitsplitsing PER UUR zodat je ziet of
echo       07:00-13:00 opnieuw negatief is of dat het toeval was.
echo    3. Een positie tegelijk.
echo.
echo  WAT DEZE RUN DOET
echo    - sectie tien precies zoals hij NU in de config staat: zes metalen,
echo      uren open. Dit meet wat er live gaat, niet iets ernaast.
echo    - elke genomen trade OOK opnieuw opgelost bij zes break-even-
echo      niveaus, op dezelfde instap, met dezelfde kosten en dezelfde
echo      20:50-flatten. Alleen de uitgang verschilt.
echo    - beide helften van de periode apart, want de beste van zeven
echo      kolommen kiezen op een steekproef is hoe dit project de meeste
echo      teleurstellingen heeft geproduceerd.
echo.
echo  WAT HET NIET DOET. Het beslist niet of een break-evenniveau live mag.
echo  Break-even laat een symbool eerder los, een vrij symbool pakt de
echo  volgende setup, en een sectie die andere trades neemt is een andere
echo  sectie. Wint een niveau hier, dan verdient het een volledige replay
echo  met het positieboek erachteraan -- geen promotie.
echo.
echo  HOELANG DIT DUURT, gemeten en niet geschat. De run van 3 september
echo  deed XAUUSD alleen in 22,6 minuten: 235.000 M1-bars over 180 dagen.
echo.
echo     180 dagen x 6 metalen  ~ 2 tot 2,5 uur
echo     180 dagen x 1 metaal   ~ 25 min
echo      90 dagen x 6 metalen  ~ 1 uur       <- begin hier als je wacht
echo.
echo  Elke 20.000 bars komt er een regel met hoeveel er te gaan is, zodat
echo  stilte niet meer op vastlopen lijkt. De regel PER MARKT verschijnt pas
echo  als die markt helemaal klaar is.
echo.
echo  MT5 moet draaien en ingelogd zijn.
echo.
echo  GEBRUIK
echo    sectie10.cmd              180 dagen, de zes metalen uit de config
echo    sectie10.cmd 90           korter
echo    sectie10.cmd 180 goud     alleen XAUUSD, om het break-evenrooster
echo                              los te zien van de verbreding
echo    sectie10.cmd 180 alle     alle dertien metalen die de broker heeft,
echo                              inclusief platina, zink en koper -- die
echo                              staan NIET live en dit meet alleen of ze
echo                              er ooit bij zouden moeten
echo.
echo  ALLEEN DE ZES METALEN WORDEN GELOPEN. EURUSD, GBPUSD en de indices
echo  staan niet in sectie tiens allowed_symbols, dus daar kan hij geen trade
echo  nemen -- ze aflopen kost minuten per markt om een symboolfilter nee te
echo  horen zeggen. De andere secties komen in deze run dus niet aan bod; daar
echo  is dryrun-live.cmd voor.
echo.

set DAGEN=%1
if "%DAGEN%"=="" set DAGEN=180

rem STANDAARD MEET DIT DE LIVE CONFIG en niets ernaast: de zes metalen die
rem er nu in staan, met de uren open. Een run die iets ANDERS meet dan wat
rem er gaat draaien beantwoordt de vraag niet.
rem
rem `goud` zet hem terug op alleen XAUUSD, zodat het break-evenrooster los
rem van de verbreding te lezen is -- twee dingen tegelijk veranderen en dan
rem een beter cijfer krijgen zegt niet welke van de twee het deed.
rem `alle` verbreedt naar alle dertien metalen die de broker noteert, alleen
rem in het geheugen; dat is een verkenning, geen voorstel.
set METALEN=
if /i "%2"=="alle" set METALEN=--section-ten-symbols metals
if /i "%2"=="goud" set METALEN=--section-ten-symbols XAUUSD

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe niet gevonden. Doe eerst update.cmd.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

rem --section-ten-only: loop ALLEEN de markten die sectie tien mag handelen.
rem De eerste run stond 252 seconden op EURUSD en 505 op GBPUSD voordat hij
rem het eerste metaal zag, met nog een uur te gaan -- en sectie tien kan in
rem geen van beide een trade nemen, `allowed_symbols` weigert ze op de eerste
rem bar. Zestien markten lopen om er zes te meten is geen grondigheid, dat is
rem een uur bars aflopen om een symboolfilter twaalfduizend keer nee te horen
rem zeggen.
.venv-live\Scripts\python.exe -m scripts.dry_run_sections ^
  --days %DAGEN% --core --live-only --sections-five-to-ten ^
  %METALEN% --section-ten-only --manage-grid ^
  --csv runtime\sectie10.csv

if errorlevel 1 (
  echo.
  echo  De run is gestopt. De regel hierboven zegt waarom -- meestal is MT5
  echo  niet ingelogd, of er is geen historie zo diep terug.
)

echo.
echo  Elke beslissing staat in runtime\sectie10.csv
echo.
echo  WAT JE MOET LEZEN
echo    1. PER MARKET. Hoeveel trades nam sectie tien per metaal? Staat er
echo       een metaal op nul, dan vuurde de detector daar niet -- dat is een
echo       antwoord, geen fout.
echo    2. BY SECTION. Blijft section_ten_gold_m1 positief met zes metalen
echo       erin, of verdunt het? Meer trades die samen minder opleveren is
echo       geen verbetering. De lat: +0,156 R per trade was het live cijfer
echo       op een markt. Zakt het onder +0,05 R of wordt het negatief, dan
echo       gaat allowed_symbols terug naar [XAUUSD].
echo    3. SECTION TEN BY UTC HOUR. Is 07:00-13:00 opnieuw negatief, nu op
echo       zes markten? Dan gaat dat blok terug. Staat het gelijk met de
echo       rest of positief, dan waren het de slechtste zes uur van EEN
echo       steekproef en heeft dichthouden trades gekost voor niets.
echo    4. BREAK-EVEN GRID. Verslaat een niveau de regel `fixed SL/TP` in
echo       ZOWEL de early- als de late-kolom? Alleen dan is het wat waard.
echo       Wint niets, dan is geen positiebeheer het juiste antwoord en
echo       blijft sectie tien zoals hij is.
echo.
pause

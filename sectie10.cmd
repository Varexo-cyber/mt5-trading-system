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
echo    1. EEN markt. allowed_symbols staat op [XAUUSD] terwijl deze broker
echo       dertien metalen noteert.
echo    2. TIEN UUR van de drieentwintig. 03:00-19:00 UTC met 07:00-13:00
echo       geblokkeerd. Die uurselectie komt uit dezelfde 180 dagen waarop
echo       de sectie gekalibreerd is -- het is dus kalibratie en geen
echo       onafhankelijke holdout. Dat staat ook zo in de config.
echo    3. Een positie tegelijk.
echo.
echo  WAT DEZE RUN DOET
echo    - sectie tien op ALLE metalen, alleen in het geheugen. Het bestand
echo      op schijf verandert niet, dus dit zet niets live.
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
echo  MT5 moet draaien en ingelogd zijn. 180 dagen M1 over dertien metalen
echo  duurt lang. Laat hem staan.
echo.
echo  GEBRUIK
echo    sectie10.cmd              180 dagen, alle metalen, break-evenrooster
echo    sectie10.cmd 90           korter
echo    sectie10.cmd 180 goud     alleen XAUUSD, om het rooster los te zien
echo                              van de verbreding
echo.

set DAGEN=%1
if "%DAGEN%"=="" set DAGEN=180

rem TWEE VRAGEN IN EEN RUN IS EEN RUN DIE GEEN VAN BEIDE BEANTWOORDT. Meer
rem markten EN een andere uitgang tegelijk veranderen betekent dat een beter
rem resultaat aan allebei kan liggen. Vandaar het woord `goud`: dan blijft
rem de sectie op XAUUSD en meet deze run alleen het break-evenrooster.
set METALEN=--section-ten-symbols metals
if /i "%2"=="goud" set METALEN=

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe niet gevonden. Doe eerst update.cmd.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections ^
  --days %DAGEN% --core --live-only --sections-five-to-ten ^
  %METALEN% --manage-grid ^
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
echo    2. BY SECTION. Blijft section_ten_gold_m1 positief met dertien
echo       metalen erin, of verdunt het? Meer trades die samen minder
echo       opleveren is geen verbetering.
echo    3. BREAK-EVEN GRID. Verslaat een niveau de regel `fixed SL/TP` in
echo       ZOWEL de early- als de late-kolom? Alleen dan is het wat waard.
echo       Wint niets, dan is geen positiebeheer het juiste antwoord en
echo       blijft sectie tien zoals hij is.
echo.
pause

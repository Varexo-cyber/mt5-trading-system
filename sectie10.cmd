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
echo  WAAROM JUIST DEZE SECTIE. Sectie tien is de enige live component met
echo  een positieve gemeten edge per trade. Op de echte rekening deed hij in
echo  30 dagen 16 trades voor +2,49 R, een halve trade per dag -- terwijl het
echo  hele boek 361 trades deed voor -29,49 R. Dit is dus de plek waar meer
echo  trades iets waard zijn; overal anders betekent meer trades meer verlies.
echo.
echo  WAT ER TOT NU TOE GEMETEN IS
echo.
echo   3 sep, 180 dagen, alleen XAUUSD, uren open:
echo     989 trades, +7,07 R. Het blok 07:00-13:00 UTC bleek WEL echt:
echo     -65,96 R over 422 trades daarbinnen tegen +73,03 R over 567 erbuiten,
echo     zes van de zes uren negatief. Blok staat weer aan.
echo.
echo   3 sep, 30 dagen, zes metalen, uren dicht:
echo     sectie tien 429 trades, +19,66 R, EUR +91,83  (+0,046 per trade)
echo     XAUUSD 228   XAUAUD 99   XAUGBP 88   XAUEUR 86   XAUJPY 71
echo     XAGUSD 150 setups en NUL trades -- boven 20%% van de stop, door de
echo     kostenpoort geweigerd. Zilver staat er sindsdien af.
echo.
echo  DE STAND NU: VIJF metalen, uren 03:00-19:00 UTC met 07:00-13:00 dicht.
echo  Deze run meet precies dat, niets ernaast.
echo.
echo  WAT ER NOG OPEN STAAT, en dat is waarvoor je dit draait:
echo    1. Dragen die vier nieuwe kruisen hun eigen gewicht, of leunen ze op
echo       XAUUSD? De tabel PER SECTION, PER MARKET zegt het per metaal.
echo    2. Verslaat een break-evenniveau de vaste SL/TP? Op 180 dagen goud won
echo       +0,1R @ 1,00R; op 30 dagen zes metalen won +0,1R @ 0,50R. Andere
echo       winnaar op andere data, dus nog geen regel -- wel steeds de
echo       +0,1R-familie, en die verslaat elke keer de vaste uitgang.
echo    3. Houdt het stand buiten augustus? Beide runs tot nu toe hingen aan
echo       een maand die 130%% van het totaal was.
echo.
echo  WAT DEZE RUN NIET DOET. Hij beslist niet of een break-evenniveau live
echo  mag. Break-even laat een symbool eerder los, een vrij symbool pakt de
echo  volgende setup, en een sectie die andere trades neemt is een andere
echo  sectie. Wint een niveau hier, dan verdient het een volledige replay met
echo  het positieboek erachteraan -- geen promotie.
echo.
echo  EN SECTIE ZES GAAT NIET MEE. `SectionSixGoldM5.symbol` staat hardcoded
echo  op XAUUSD, dus die vuurt op geen enkel ander metaal, hoe goed hij hier
echo  ook scoort. Zijn model is een bevroren lineair model met op goud gefitte
echo  coefficienten; die op XAUJPY loslaten levert getallen op die nergens
echo  over gaan. Verbreden vraagt een nieuw model per markt.
echo.
echo  HOELANG DIT DUURT, gemeten en niet geschat. XAUUSD alleen kostte
echo  22,6 minuten: 235.000 M1-bars over 180 dagen.
echo.
echo     180 dagen x 5 metalen  ~ 2 uur
echo     180 dagen x 1 metaal   ~ 25 min
echo      90 dagen x 5 metalen  ~ 50 min      ^<- begin hier als je wacht
echo.
echo  Per markt komt er eerst "fetching history..." en daarna ongeveer acht
echo  voortgangsregels per klok, zodat stilte niet op vastlopen lijkt. De
echo  regel met het aantal trades verschijnt pas als die markt klaar is.
echo.
echo  MT5 moet draaien en ingelogd zijn.
echo.
echo  GEBRUIK
echo    sectie10.cmd              180 dagen, de vijf metalen uit de config
echo    sectie10.cmd 90           korter
echo    sectie10.cmd 180 goud     alleen XAUUSD, om het break-evenrooster
echo                              los te zien van de verbreding
echo    sectie10.cmd 180 alle     alle dertien metalen die de broker heeft,
echo                              inclusief platina, zink en koper -- die
echo                              staan NIET live en dit kijkt alleen of ze
echo                              er ooit bij zouden moeten
echo.
echo  ALLEEN DE VIJF METALEN WORDEN GELOPEN. EURUSD, GBPUSD en de indices
echo  staan niet in sectie tiens allowed_symbols, dus daar kan hij geen trade
echo  nemen -- ze aflopen kost minuten per markt om een symboolfilter nee te
echo  horen zeggen. De andere secties komen in deze run dus niet aan bod; daar
echo  is dryrun-live.cmd voor.
echo.

set DAGEN=%1
if "%DAGEN%"=="" set DAGEN=180

rem STANDAARD MEET DIT DE LIVE CONFIG en niets ernaast: de vijf metalen die
rem er nu in staan, met 07:00-13:00 UTC dicht. Een run die iets ANDERS meet
rem dan wat er gaat draaien beantwoordt de vraag niet.
rem
rem `goud` zet hem terug op alleen XAUUSD, zodat het break-evenrooster los
rem van de verbreding te lezen is -- twee dingen tegelijk veranderen en dan
rem een beter cijfer krijgen zegt niet welke van de twee het deed.
rem `alle` verbreedt naar alle dertien metalen die de broker noteert, alleen
rem in het geheugen; dat is een verkenning, geen voorstel. Zilver zit daar
rem weer bij en zal opnieuw nul trades nemen -- dat is de bevestiging, niet
rem een fout.
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
echo  WAT JE MOET LEZEN, in deze volgorde:
echo    1. PER SECTION, PER MARKET. Dit is de vraag. Staan XAUEUR, XAUGBP,
echo       XAUAUD en XAUJPY positief, of draagt XAUUSD ze? Een sectie is maar
echo       zo verbreed als zijn slechtste markt.
echo    2. BY SECTION. Blijft sectie tien positief per trade met vijf metalen
echo       erin? Meer trades die samen minder opleveren is geen verbetering.
echo    3. IS THIS REAL. Hoeveel maanden positief, en hoeveel procent van het
echo       resultaat hangt aan de beste maand? Beide vorige runs hingen aan
echo       augustus. Als dat hier weer zo is, is het cijfer een maand en geen
echo       edge.
echo    4. SECTION TEN BY UTC HOUR. Nu alleen ter controle: het blok staat
echo       dicht, dus 07:00-13:00 hoort vrijwel leeg te zijn. Staat daar toch
echo       volume, dan lekt er iets en wil ik dat weten.
echo    5. BREAK-EVEN GRID. Verslaat een niveau `fixed SL/TP` in ZOWEL early
echo       als late? Wint dezelfde +0,1R-variant als op 180 dagen goud, dan
echo       zet ik hem aan. Wint niets, dan is geen positiebeheer het juiste
echo       antwoord en blijft sectie tien zoals hij is.
echo.
pause

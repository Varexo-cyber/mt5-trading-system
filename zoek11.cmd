@echo off
setlocal

rem ==================================================================
rem  DRAAI VANUIT EEN KOPIE IN %TEMP%, EN DAT IS GEEN NETHEID.
rem
rem  cmd.exe leest een batchbestand NIET in een keer in. Het onthoudt een
rem  byte-positie en gaat na elk commando terug naar de schijf voor de
rem  volgende regel. Verandert het bestand ondertussen -- een git pull,
rem  een update.cmd, een editor die opslaat -- dan hervat cmd op dezelfde
rem  byte-positie in het NIEUWE bestand. Dat komt midden in een woord uit.
rem
rem  Precies dat gebeurde tijdens de `mean`-run van 3 september: de zoek-
rem  tocht was klaar, en toen kwam er
rem
rem      'e' is not recognized as an internal or external command
rem
rem  gevolgd door DRAAIT: ... en de hele zoektocht NOG EEN KEER. Die `e`
rem  is een halve `echo`. Er is niets mis met de zoektocht; het bestand
rem  verschoof onder de uitvoering vandaan en cmd viel terug in het blok
rem  erboven.
rem
rem  Een kopie in %TEMP% wordt door niets aangeraakt, dus die kan niet
rem  verschuiven. ZOEK11_ROOT houdt de map van de repo vast, want in de
rem  kopie wijst %~dp0 naar %TEMP% en dan vindt hij .venv-live niet.
rem ==================================================================
rem  De aanroep geeft %* door en niet %1 %2 %3 %4: `zoek11.cmd trend index
rem  M30 H1 360` is vijf woorden en het vijfde viel er stil af. Nu de
rem  volgorde niet meer telt kan een aanroep langer worden dan de plekken
rem  die ik vooraf had uitgeschreven.
if not defined ZOEK11_ROOT set ZOEK11_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\zoek11-actief.cmd" >nul
  call "%TEMP%\zoek11-actief.cmd" __uittemp %*
  exit /b
)
shift

cd /d "%ZOEK11_ROOT%"

echo.
echo  ==================================================================
echo   ZOEKEN NAAR SECTIE ELF -- GOUD, INTRADAY, ZESTIEN MECHANISMEN
echo  ==================================================================
echo.
echo  WAAROM DIT DRAAIT EN ER NOG GEEN SECTIE ELF IS.
echo.
echo  Sectie 6 ging live op een cijfer dat later -71,65 R bleek. Dat gebeurde
echo  niet omdat de code fout was maar omdat de parameters gekozen waren
echo  VOORDAT er gemeten was. Een sectie bouwen en daarna hopen dat de meting
echo  meevalt is die fout nog een keer maken met een nieuw nummer erop.
echo.
echo  Wat hier doorheen komt WORDT sectie elf; komt er niets door, dan is
echo  dat het antwoord en geen mislukking van de run.
echo.
echo  ------------------------------------------------------------------
echo   WAT DE EERSTE RUN AL HEEFT UITGEWEZEN  (360 dagen, 13 metalen)
echo  ------------------------------------------------------------------
echo.
echo   KOSTEN ZIJN NIET HET PROBLEEM. metal M5 6,2%% van de stop, M15 3,4%%.
echo   De muur staat op 25%%. Goud is ruim betaalbaar -- als er niets uitkomt
echo   ligt dat aan de instap, niet aan de spread. Dat was vooraf de meest
echo   waarschijnlijke doodsoorzaak en die is nu uitgesloten.
echo.
echo   NIETS HAALDE DE LAT. 32 cellen, beste 0,81 sigma tegen een lat van
echo   3,25. Alles struikelde op de TRAINhelft, niet op de holdout -- dat is
echo   een schoner nee dan overfitting: er was niets om te overfitten.
echo.
echo   EN ER ZAT EEN MEETFOUT IN DIE JUIST DE FADERS TROF. De resolver nam
echo   elke signaalbar. Een breakout vuurt een keer; een mean-reverter blijft
echo   aan zolang prijs uitgerekt is, dus een gebeurtenis werd tien trades --
echo   de eerste aan de rand van de beweging en negen steeds dieper erin.
echo   stretch_fade meldde 150.055 trades, opening_range_break 1.796. Dat is
echo   nu gerepareerd: een positie tegelijk, zoals het account zelf.
echo.
echo  ------------------------------------------------------------------
echo   WAT ER GEMETEN WORDT -- acht mechanismen, elk in twee richtingen
echo  ------------------------------------------------------------------
echo    stretch            prijs 1,8 ATR van zijn eigen 2-uursgemiddelde
echo    quiet_stretch      hetzelfde, alleen 00:00-06:00 UTC (dunne boek)
echo    london             07:00 UTC, de eerste uurbeweging
echo    comex              13:00 UTC, US-data en de COMEX-opening
echo    pm_fix             het 15:00-Londen-fixingvenster
echo    round_number       de tien-dollarniveaus waar het boek op klit
echo    opening_range      het eerste uur van de gouddag, daarna de break
echo    day_range          de dag heeft zijn normale range al gelopen
echo.
echo  Elk mechanisme wordt zowel MEEGENOMEN als GEFADED. Alleen de helft
echo  publiceren die toevallig betaalde is de sample fitten.
echo.
echo  DE LAT STAAT HOGER DAN 2,0 SIGMA EN DAT IS OPZET. De beste van 32
echo  hypotheses haalt 2 sigma op pure ruis in ruim de helft van de gevallen,
echo  dus de Bonferroni-lat ligt boven 3,2 -- en die stijgt met elke run,
echo  want eerdere zoektochten tellen mee. De holdout, de nieuwste 40%% waar
echo  niet in gezocht is, moet daarnaast op eigen kracht 2,0 halen. Een
echo  random instap op dezelfde bars wordt eraf getrokken; die leest NIET
echo  nul (hier -0,025 R op M15, -0,037 R op M5).
echo.
echo  MT5 moet draaien en ingelogd zijn.
echo.
echo  GEBRUIK -- DE VOLGORDE DOET ER NIET TOE. Een getal is dagen, M5/M15/
echo  M30/H1 is een klok (meerdere mag), forex/metal/index/crypto is een
echo  klasse, en al het andere is een symbool.
echo.
echo    zoek11.cmd forex H1 M30 720   FOREX, en dit is de grote open vraag.
echo                                  De zestien mechanismen hieronder zijn
echo                                  NOOIT ergens anders dan op metaal
echo                                  gericht. London-open en COMEX zijn
echo                                  sessie-effecten, en FX heeft de
echo                                  scherpste sessiestructuur die er is.
echo    zoek11.cmd index H1 M30 720   dezelfde vraag op de indices
echo    zoek11.cmd                    metaal, 1:1, stop 1,0 ATR
echo    zoek11.cmd mean               0,5:1, stop 1,5 ATR (fadergeometrie)
echo    zoek11.cmd trend              2:1, stop 1,0 ATR  (volgergeometrie)
echo    zoek11.cmd mean forex H1      combineer gerust
echo    zoek11.cmd 360 M15 XAUUSD     alleen goud
echo.
echo  WAAROM FOREX EN NIET NOG EEN GOUDPASS. Vier passes op metaal, 96
echo  hypotheses, beste cel 1,61 sigma tegen een lat van 3,25. Kosten waren
echo  nooit de rem -- 1,6%% van de stop op H1. Nog een geometrie op hetzelfde
echo  metaal is de vijfde manier om dezelfde vraag te stellen. De mechanismen
echo  zijn nooit op een andere ACTIVAKLASSE losgelaten, en dat is geen
echo  variant maar een andere vraag.
echo.
echo  WAAROM DIE GEOMETRIE EEN APARTE PASS IS. De eerste run testte zestien
echo  instaps op EEN uitgang: stop 1 ATR, doel 1:1. Dit archief weet dat de
echo  uitgang zwaarder weegt dan de instap -- dezelfde entries maten +0,279 R
echo  op 1:1 en +0,016 R op 3:1. Een mean-reverter wil klein doel en ruime
echo  stop (vaak raak, weinig per keer); een volger het omgekeerde. Er is dus
echo  een reele kans dat de mechanismen kloppen en de geometrie niet.
echo.
echo  DIT IS GEEN GRATIS TWEEDE KANS. Elke pass is 32 hypotheses erbij en de
echo  lat gaat mee omhoog. Draai er twee, geen tien.
echo.

rem ------------------------------------------------------------------
rem GEOMETRIE. Plain woorden, geen komma's -- cmd splitst op komma's en
rem dan valt de rest van de regel weg voordat het script hem ziet.
rem
rem `TOT` is het aantal hypotheses uit EERDERE runs dat bij de Bonferroni-
rem telling opgeteld wordt: 12 index-cellen plus 32 per gouddpass die al
rem gedraaid is. Twee keer voor dezelfde 32 betalen is hoe een zoektocht
rem zichzelf witwast tot een ontdekking.
rem ------------------------------------------------------------------
set VORM=%1
set STOP=1.0
set DOEL=1.0
set TOT=12

rem ALLEEN EEN BEKEND VORMWOORD SCHUIFT DE ARGUMENTEN OP. De eerste versie
rem schoof zodra argument 1 niet leeg was, dus `zoek11.cmd 720` las 720 als
rem een vormwoord, vond geen dagen op plek 2 en draaide stilletjes 360 dagen
rem terwijl er 720 op het scherm stond. Precies de fout die dit hele project
rem blijft maken: het draait, het zegt niets, en het meet iets anders.
set SCHUIF=0
if /i "%VORM%"=="mean"  (set STOP=1.5& set DOEL=0.5& set TOT=44& set SCHUIF=1)
if /i "%VORM%"=="trend" (set STOP=1.0& set DOEL=2.0& set TOT=44& set SCHUIF=1)
if /i "%VORM%"=="base"  (set SCHUIF=1)
if "%SCHUIF%"=="0" set VORM=
if "%SCHUIF%"=="1" shift

rem ------------------------------------------------------------------
rem ELK WOORD WORDT HERKEND AAN ZIJN VORM, NIET AAN ZIJN PLAATS.
rem
rem De positieversie is hier twee keer op stukgegaan. Eerst las
rem `zoek11.cmd 720` de 720 als vormwoord en draaide stil 360 dagen. Toen
rem zette `zoek11.cmd base 360 M30 H1` de H1 op de symboolplek en werd dat
rem `--symbols H1` -- geen enkele markt heet zo, dus elke markt zou zijn
rem overgeslagen met "no spec" en de run kwam terug met nul cellen, wat er
rem in de output uitziet als "gemeten en niets gevonden".
rem
rem Aanhalingstekens helpen niet en komma's ook niet, want cmd knipt de
rem regel op allebei. Wat wel werkt is niet meer op volgorde vertrouwen:
rem een getal is dagen, M5/H1/... is een klok, forex/metal/index is een
rem klasse, en al het andere is een symbool. De volgorde doet er niet toe
rem en meerdere klokken kunnen gewoon achter elkaar.
rem ------------------------------------------------------------------
set DAGEN=
set KLOKKEN=
set MARKTEN=

:volgendwoord
if "%~1"=="" goto woordenklaar
set STUK=%~1
set SOORT=symbool
echo %STUK%| findstr /r /c:"^[0-9][0-9]*$" >nul 2>&1 && set SOORT=dagen
for %%T in (M1 M5 M15 M30 H1 H4 D1) do if /i "%STUK%"=="%%T" set SOORT=klok
for %%C in (forex metal index crypto commodity) do if /i "%STUK%"=="%%C" set SOORT=klasse
if "%SOORT%"=="dagen"   set DAGEN=%STUK%
if "%SOORT%"=="klok"    set KLOKKEN=%KLOKKEN% %STUK%
if "%SOORT%"=="klasse"  set MARKTEN=--asset-class %STUK%
if "%SOORT%"=="symbool" set MARKTEN=--symbols %STUK%
shift
goto volgendwoord
:woordenklaar

if "%DAGEN%"=="" set DAGEN=360
if "%KLOKKEN%"=="" set KLOKKEN=M5 M15

rem STANDAARD ALLE METALEN, en de scanner-classificatie vindt hoe deze
rem broker ze spelt -- goud tegen USD, EUR, GBP, AUD en JPY, zilver,
rem platina, palladium en de industriemetalen, dertien in totaal. Ik ga
rem niet gokken naar de namen.
rem
rem DIT IS GEEN ONAFHANKELIJK BEWIJS. XAUUSD en XAUEUR zijn hetzelfde goud
rem door een valutabril: breekt goud 's ochtends, dan breekt het in alle
rem vier. Ze staan er wel in omdat de KOSTEN per metaal flink verschillen
rem en dat verschil beslist of iets betaalbaar is. Daarom wordt de sigma
rem per DAG geclusterd en niet per trade -- dertien metalen die samen
rem bewegen zijn een observatie, geen dertien.
if "%MARKTEN%"=="" set MARKTEN=--asset-class metal

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe niet gevonden. Doe eerst update.cmd.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime
rem DE KLASSE HOORT IN DE BESTANDSNAAM. Zonder dat schrijft een forex-run
rem over de metaalrun heen en is het bewijs voor de vorige conclusie weg.
set KLASSE=metal
for %%C in (forex index crypto commodity) do if "%MARKTEN%"=="--asset-class %%C" set KLASSE=%%C
set UIT=runtime\zoek11-%KLASSE%.csv
if not "%VORM%"=="" set UIT=runtime\zoek11-%KLASSE%-%VORM%.csv

echo  DRAAIT: %DAGEN% dagen, klokken %KLOKKEN%, stop %STOP% ATR, doel %DOEL%:1
echo          Bonferroni telt %TOT% eerdere hypotheses mee.
echo.

.venv-live\Scripts\python.exe -m scripts.search_section_four ^
  --days %DAGEN% --family gold --clocks %KLOKKEN% %MARKTEN% ^
  --stop-atr %STOP% --ratio %DOEL% --horizon 48 ^
  --cells-already-tried %TOT% ^
  --csv %UIT%

if errorlevel 1 (
  echo.
  echo  De run is gestopt. De regel hierboven zegt waarom -- meestal is MT5
  echo  niet ingelogd, of er is geen historie zo diep terug.
)

echo.
echo  Elke cel met zijn cijfers staat in %UIT%
echo.
echo  WAT JE MOET LEZEN, in deze volgorde:
echo    1. WHAT A ROUND TRIP COSTS. Boven ~25%% van de stop betaalt niets op
echo       die klok, wat de instap ook doet, en is de rest ruis.
echo    2. NEVER FIRED. Een kandidaat die nooit vuurde is mijn drempel die
echo       niet bij deze feed past, GEEN gemeten mislukking. Zeg het me.
echo    3. SURVIVED EVERY BAR. Wat hier staat wordt sectie elf.
echo    4. CLOSEST MISSES. Drie kolommen, en ze zeggen verschillende dingen:
echo       raw R is per trade met kosten, vs coin is dat minus de random
echo       instap, sigma is over DAGtotalen. Een cel kan de muntjes verslaan
echo       en toch niet van nul te onderscheiden zijn -- dat is geen tegen-
echo       spraak, dat zijn twee vragen.
echo.
pause

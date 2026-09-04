@echo off
setlocal

rem Draait uit een kopie in %TEMP%, net als sectie10.cmd en zoek11.cmd: cmd
rem leest een batchbestand van schijf TERWIJL het draait, dus een git pull
rem halverwege laat het hervatten midden in een woord. Dat kostte ooit een run
rem van drie uur.
if not defined S11_ROOT set S11_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\sectie11-actief.cmd" >nul
  call "%TEMP%\sectie11-actief.cmd" __uittemp %*
  exit /b
)
shift

cd /d "%S11_ROOT%"

echo.
echo  ==================================================================
echo   SECTIE ELF ALLEEN -- DE VIER GOUDKRUISEN, M5
echo  ==================================================================
echo.
echo  Alleen sectie elf. Geen sectie 6, 7, 8 of 10 ernaast. Alleen de vier
echo  markten waar sectie elf een getraind model voor heeft, alleen op M5.
echo.
echo  DAAROM IS DIT SNEL. Geen M1-historie (sectie elf is een M5-strategie,
echo  dus M1 levert geen extra antwoord op), vier markten in plaats van
echo  zestien, en een sectie in plaats van vijf. Minuten, geen uren.
echo.
echo  VIER EN GEEN VIJF. XAUUSD hoort er niet bij: daar heeft sectie elf geen
echo  model voor, dat is sectie zes zijn markt. Vijf markten meten met vier
echo  modellen zou een lege markt als een nulresultaat laten lezen.
echo.
echo  WAAROM JE DIT DRAAIT. De trainer gaf twee tegengestelde antwoorden:
echo    walk-forward  +0,005 tot +0,030 R per trade, alle 16 cellen positief
echo    HOLDOUT       -0,031 tot -0,052 R per trade, vier markten van de vier
echo  Deze replay is een DERDE, onafhankelijke stem: andere resolver, echte
echo  brokerspread, echte positielimiet, de live poorten ervoor. Hij kan het
echo  met allebei oneens zijn.
echo.
echo  Sectie elf draait hier zoals hij gemeten is: vaste stop en target, geen
echo  break-even. Dat is geen keuze van deze run maar van de fit -- elk cijfer
echo  hierboven is op die route gemeten, dus er beheer overheen leggen zou een
echo  andere strategie meten dan degene waar de vraag over gaat.
echo.
echo  MT5 moet draaien en ingelogd zijn.
echo.
echo  GEBRUIK -- HET GETAL IS HET AANTAL DAGEN
echo    sectie11.cmd 90         90 dagen   ^<-- DEZE
echo    sectie11.cmd 180        180 dagen
echo    sectie11.cmd            90 dagen  (hetzelfde)
echo    sectie11.cmd 90 beheer  90 dagen, EN het break-evenrooster erbij,
echo                            zodat je ziet wat positiebeheer eraan zou
echo                            veranderen. Het beslist niets, het meet.
echo    sectie11.cmd 90 fijn    90 dagen, maar uitgelopen op M1-bars. Trager.
echo.
echo  WAT `fijn` DOET, EN WAAROM HET ERTOE DOET. Zonder dat wordt een M5-trade
echo  uitgelopen op M5-bars, en een bar die zowel de stop als het target raakt
echo  is dan onbeslisbaar -- die boekt dit script als VERLIES, want de volgorde
echo  is niet te weten. Dat is de veilige keuze en het is ook een rem: deze run
echo  leest daardoor pessimistischer dan de trainer. Komt sectie elf hier net
echo  negatief uit, draai hem dan nog een keer met `fijn` voordat je iets
echo  besluit. Komt hij er ruim positief uit, dan is dat cijfer eerder te laag
echo  dan te hoog.
echo.

rem VORMHERKENNING, geen positie. `sectie11.cmd 90 beheer` en
rem `sectie11.cmd beheer 90` doen hetzelfde. Een getal is dagen, het woord
rem beheer zet het rooster aan. Dit is de reparatie van `zoek11.cmd 720`, dat
rem 720 als vormwoord las en er stilletjes 360 van maakte.
set DAGEN=90
set BEHEER=
set FIJN=--no-m1

:lees
if "%~1"=="" goto klaar
echo %~1| findstr /r "^[0-9][0-9]*$" >nul && set DAGEN=%~1
if /i "%~1"=="beheer" set BEHEER=--manage-grid
if /i "%~1"=="fijn" set FIJN=
shift
goto lees
:klaar

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe bestaat niet. Draai eerst update.cmd.
  pause
  exit /b 1
)

if not exist "models\section_eleven\XAUJPY.json" (
  echo.
  echo  ER ZIJN GEEN MODELBESTANDEN. Sectie elf kan zonder model geen enkele
  echo  trade nemen, en dan komt deze run terug met nul trades -- wat eruitziet
  echo  als "de strategie vond niets" terwijl er nooit iets gefit is.
  echo.
  echo  Draai eerst:  train11.cmd 720 forceer
  echo.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

echo  %DAGEN% dagen, vier markten, alleen sectie elf. Even geduld.
echo.

.venv-live\Scripts\python.exe -m scripts.dry_run_sections ^
  --days %DAGEN% %FIJN% --live-only --only section_eleven_metals ^
  --symbols XAUEUR,XAUGBP,XAUAUD,XAUJPY %BEHEER% ^
  --csv runtime\sectie11.csv

if errorlevel 1 (
  echo.
  echo  De run is gestopt. De regel hierboven zegt waarom -- meestal is MT5
  echo  niet ingelogd, of er is geen historie zo diep terug.
)

echo.
echo  Elke beslissing staat in runtime\sectie11.csv
echo  Opnieuw lezen zonder opnieuw te rekenen:  lees.cmd runtime\sectie11.csv
echo.
echo  WAT JE MOET LEZEN, in deze volgorde:
echo    1. PER SECTION, PER MARKET. Staan alle vier positief, of draagt er een
echo       de rest? Een sectie is maar zo goed als zijn slechtste markt.
echo    2. De VROEG/LAAT-splitsing. De holdout die negatief was, was de
echo       NIEUWSTE historie. Als de late helft hier ook negatief is, zeggen
echo       twee onafhankelijke metingen hetzelfde en is het klaar.
echo    3. R PER TRADE, niet de totale R. De fit verwacht +0,03 per trade; dat
echo       is dun, dus een paar honderd trades zeggen nog niks.
echo.
pause

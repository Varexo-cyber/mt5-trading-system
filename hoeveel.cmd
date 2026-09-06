@echo off
setlocal

rem Draait uit een kopie in %TEMP%: cmd leest een batchbestand van schijf
rem TERWIJL het draait, dus een git pull halverwege laat het hervatten midden
rem in een woord. Dat heeft hier al een keer een compleet `set`-blok opgegeten
rem zonder enige melding.
if not defined HOEVEEL_ROOT set HOEVEEL_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\hoeveel-actief.cmd" >nul
  call "%TEMP%\hoeveel-actief.cmd" __uittemp %*
  exit /b
)
shift

cd /d "%HOEVEEL_ROOT%"

rem Elke variabele hier, bovenaan, voordat iets hem leest. Een niet-gezette
rem %VAR% in cmd is een lege string en dat is stil -- `sectie11.cmd --days`
rem eindigde daardoor op "argument --days: expected one argument".
set DAGEN=180
rem ALLEEN DE MARKTEN WAAR DE GEMETEN SECTIES OP MOGEN HANDELEN, uit de
rem secties zelf gelezen. De 90-daagse run liep elf markten en ZES ervan
rem mag geen enkele sectie aanraken -- GBPUSD, USDCHF, EURJPY, GBPJPY,
rem US30, GER40. Driehonderdduizend bars per stuk, zeven keer per bar
rem beoordeeld, om niets op te leveren. `kern` zet de zestien kernmarkten
rem terug, `alles` de hele catalogus.
set MARKTEN=--section-markets

:lees
if "%~1"=="" goto klaar
echo %~1| findstr /r "^[0-9][0-9]*$" >nul && set DAGEN=%~1
if /i "%~1"=="alles" set MARKTEN=
if /i "%~1"=="all" set MARKTEN=
if /i "%~1"=="kern" set MARKTEN=--core
shift
goto lees
:klaar

echo.
echo  ==========================================================================
echo   ALS IK DIT %DAGEN% DAGEN GELEDEN HAD AANGEZET, WAT ZOU IK NU HEBBEN
echo  ==========================================================================
echo.
echo  Dit is de zwaarste meting op deze rekening en dat is met opzet. De gewone
echo  dryrun is ConfluenceEngine + PositionSizer en verder niets; de rekening
echo  draait daar acht poorten omheen. Deze run zet er VIER van aan -- die
echo  waarvan de input in de bars zelf zit -- plus het gedeelde positieboek:
echo.
echo    * spread tegenover stopafstand      SPREAD_EATS_THE_STOP
echo    * markt-te-stil / liveliness        MARKET_TOO_QUIET
echo    * M1 volumepiek                     VOLUME_SPIKE
echo    * target-reach en richtingsvoordeel TARGET_RARELY_REACHED
echo    * EEN gedeeld positieplafond over ALLE secties, op tijdvolgorde
echo.
echo  WAT ER NIET IN ZIT, en dat staat ook onder de uitkomst:
echo    * het nieuwsblok. Er is geen kalenderarchief van dit venster, en er een
echo      verzinnen is precies het tegenovergestelde van de fail-safe die deze
echo      rekening draait. Nieuws HAALT alleen trades weg, dus de uitkomst is
echo      een bovengrens.
echo    * de AI-review. `ai.provider` staat op local_history en de antwoorden
echo      die Claude gegeven zou hebben zijn nooit gegeven.
echo    * slippage voorbij de geboekte spread, en de volgorde van partial /
echo      trailing / health / peak-stall binnen een bar.
echo.
echo  WELKE SECTIES. Het hele boek, inclusief de secties die GEEN echt geld
echo  mogen gebruiken -- S11 (de benen op M5), S12, S13 en de BTC-secties
echo  15/16/17. Die staan apart in de uitkomst onder "niet op de allowlist",
echo  want hun euro's zijn hypothetisch. De vier die wel live zijn --
echo  failed_session_breakout, sectie 6, sectie 8 en sectie 10 -- zijn het
echo  antwoord op de vraag zoals je hem stelde.
echo.
echo  DE UITKOMST STAAT ER TWEE KEER. Vast inzetbedrag (elke trade op dezelfde
echo  startbalans, exact) en samengesteld (de inzet loopt mee met de balans,
echo  benaderd -- de minimum lot wordt niet opnieuw gecontroleerd). Lees de
echo  vaste regel als degene zonder gat erin.
echo.
echo  M1-bars zijn verplicht: de volumepoort leest M1 en niets anders.
echo  Op %DAGEN% dagen over de kernmarkten is dit een lange run. Zet hem aan en
echo  laat hem staan.
echo.
echo  GEBRUIK
echo    hoeveel.cmd 180        180 dagen, alleen de markten van de secties zelf
echo    hoeveel.cmd 90         90 dagen
echo    hoeveel.cmd 180 kern   de zestien kernmarkten erbij
echo    hoeveel.cmd 180 alles  elke markt die de scanner ziet (veel langer)
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe ontbreekt. Draai eerst update.cmd.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections --days %DAGEN% %MARKTEN% --jarvis-replay --csv runtime\hoeveel.csv

if errorlevel 1 (
  echo.
  echo  De run is gestopt. De traceback hierboven zegt waarom -- meestal is MT5
  echo  niet ingelogd, of er is geen historie zo ver terug.
)

echo.
echo  Elke afzonderlijke beslissing, een regel per stuk, staat in
echo  runtime\hoeveel.csv
echo.
pause

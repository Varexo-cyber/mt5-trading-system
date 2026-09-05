@echo off
setlocal

rem Draait uit een kopie in %TEMP%: cmd leest een batchbestand van schijf
rem TERWIJL het draait, dus een git pull halverwege laat het hervatten midden
rem in een woord. Dat kostte ooit een run van drie uur.
if not defined S11_ROOT set S11_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\sectie11-actief.cmd" >nul
  call "%TEMP%\sectie11-actief.cmd" __uittemp %*
  exit /b
)
shift

cd /d "%S11_ROOT%"

rem ELKE VARIABELE WORDT HIER GEZET, BOVENAAN, VOORDAT IETS HEM LEEST.
rem
rem Deze regels waren weg. Een merge met een tweede sessie voegde twee
rem herschrijvingen van dit bestand samen zonder ook maar een conflict te
rem melden -- de hunks overlapten niet -- en het resultaat las %DAGEN% zonder
rem dat iets hem ooit had gezet. De run stierf op
rem `argument --days: expected one argument` nadat hij zijn hele banner had
rem afgedrukt. Een lege variabele in cmd is stil: hij verdwijnt gewoon uit de
rem commandoregel in plaats van een fout te geven.
set DAGEN=180
set BEHEER=
set GROF=

rem VORMHERKENNING, geen positie, en `--days 90` werkt nu ook. Een getal is
rem dagen, `beheer` zet het break-evenrooster aan, `grof` laat M1 vallen.
rem `--days` is geen getal en wordt dus gewoon overgeslagen; het getal erachter
rem wordt herkend als getal.
:lees
if "%~1"=="" goto klaar
echo %~1| findstr /r "^[0-9][0-9]*$" >nul && set DAGEN=%~1
if /i "%~1"=="beheer" set BEHEER=--manage-grid
if /i "%~1"=="grof" set GROF=--no-m1
shift
goto lees
:klaar

if not defined DAGEN set DAGEN=180

echo.
echo  ==================================================================
echo   XAUJPY ALLEEN -- SECTIE 11 (BENEN, M5), 12 (M5) EN 13 (M15)
echo  ==================================================================
echo.
echo  DE ECHTE REPLAY, met het ECHTE kostenmodel. `zoekjpy.cmd` zocht deze
echo  mechanismen met een kapot kostenmodel: een vaste 0,4%% van de prijs als
echo  stopbreedte op elke klok, terwijl de stop 1,0 x ATR van die klok is. Op
echo  M1 is die ATR een orde kleiner, dus M1 kreeg ongeveer een twintigste van
echo  zijn echte kosten. Deze replay rekent wel goed, en dat is precies waarom
echo  hij het cijfer is dat beslist.
echo.
echo  SECTIE 11 IS IETS ANDERS GEWORDEN. `streak_reversal` op M1 deed -0,18 R
echo  per trade over 855 trades zodra de kosten klopten, en die is weg. Wat er
echo  nu staat is het benen-mechanisme op M5:
echo.
echo    XAUJPY = XAUUSD x USDJPY
echo.
echo  Goud beweegt, de yen niet, het kruis moet volgen, en wie het kruis quote
echo  doet dat met vertraging. De tegenpartij heeft een naam. `benen.cmd 720`
echo  mat op M5 met gap 0,50 ATR en R:R 1,5: 152 trades, +54,80 R, +0,361 per
echo  trade, +2,93 sigma tegen een lat van 2,96. HIJ HAALDE DIE LAT NIET, met
echo  0,03, en die lat gaat niet omlaag.
echo.
echo  WAT ER IN DE CONFIG STAAT:
echo    S11  M5   benen-gap 0,50        0,3/dag  +0,361/trade  +2,93 sigma
echo    S12  M5   stretch_fade         20,6/dag  +0,018/trade  +1,61 sigma
echo    S13  M15  close_position_fade   7,6/dag  +0,041/trade  +2,25 sigma
echo.
echo  MIJN VERWACHTING, zodat je me erop kunt pakken: S11 doet weinig trades en
echo  betaalt per trade het meest; S12 staat uit omdat hij de andere twee
echo  verdrong. Als S11 hier onder +0,20 per trade uitkomt is het mechanisme
echo  in de praktijk kleiner dan de zoektocht zei, en dan is de reden bijna
echo  zeker dat de lag korter is dan vijf minuten.
echo.
echo  DE DRIE KLOKKEN IN EEN RUN. Ze delen de positielimiet, dus ze los meten
echo  en de uitkomsten optellen geeft een ander getal dan ze samen draaien --
echo  precies zoals sectie tien 260 trades van sectie zes opat.
echo.
echo  MT5 moet draaien en ingelogd zijn, met XAUJPY zichtbaar in Market Watch.
echo.
echo  GEBRUIK -- HET GETAL IS HET AANTAL DAGEN
echo    sectie11.cmd 180        180 dagen  ^<-- DEZE
echo    sectie11.cmd 90         korter
echo    sectie11.cmd --days 90  hetzelfde, als je liever zo typt
echo    sectie11.cmd 180 beheer 180 dagen, EN het break-evenrooster erbij
echo    sectie11.cmd 180 grof   sneller, uitgelopen op M5-bars in plaats van M1
echo.
echo  WAT `grof` KOST. Een bar die stop en target allebei raakt is onbeslisbaar
echo  en telt als VERLIES. Op M5-bars gebeurt dat vaker dan op M1, dus `grof`
echo  leest pessimistischer. Zonder `grof` is het eerlijke getal.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe bestaat niet. Draai eerst update.cmd.
  pause
  exit /b 1
)

rem GEEN MECHANISME IS GEEN RUN. Een sectie zonder mechanisme is stil, en een
rem stille sectie komt terug met een nulregel die leest als een strategie die
rem niets vond. Sectie 11 heeft geen `mechanism`-veld meer -- die draait het
rem benen-mechanisme en telt mee zodra hij `enabled` is.
.venv-live\Scripts\python.exe -c "import sys; from config.loader import DEFAULT_CONFIG_PATH, load_settings; s=load_settings(DEFAULT_CONFIG_PATH, overlay='config/eightcap.yaml', env_overrides=False); a=s.analysis; n=[m for m in ('section_twelve_xaujpy_m5','section_thirteen_xaujpy_m15') if getattr(a,m).mechanism]; n=(['section_eleven_xaujpy_legs_m5 (benen)'] if a.section_eleven_xaujpy_legs_m5.enabled else [])+n; print('actief:', ', '.join(n) if n else 'GEEN'); sys.exit(0 if n else 1)"
if errorlevel 1 (
  echo.
  echo  GEEN VAN DE DRIE SECTIES KAN IETS DOEN.
  echo.
  echo  Deze run zou nul trades opleveren en dat leest als "niets gevonden"
  echo  terwijl er nooit iets gezocht is. Draai eerst:
  echo.
  echo      benen.cmd 720        (sectie 11)
  echo      zoekjpy.cmd 720      (sectie 12 en 13)
  echo.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

echo.
echo  %DAGEN% dagen, XAUJPY, secties 11 / 12 / 13. Even geduld.
echo.

.venv-live\Scripts\python.exe -m scripts.dry_run_sections ^
  --days %DAGEN% %GROF% --symbols XAUJPY %BEHEER% ^
  --only section_eleven_xaujpy_legs_m5,section_twelve_xaujpy_m5,section_thirteen_xaujpy_m15 ^
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
echo    1. BY SECTION. Welke van de drie klokken betaalt? Ze delen de
echo       positielimiet, dus twee die elkaar verdringen is een echt resultaat.
echo    2. De VROEG/LAAT-splitsing. Blijft het overeind in de tweede helft?
echo    3. R PER TRADE, niet de totale R. Meer trades die samen minder
echo       opleveren is geen verbetering.
echo    4. WHAT THIS RUN DID NOT MODEL. Ook deze replay draait niet alle acht
echo       live poorten. Wat er ontbreekt staat er met naam bij.
echo.
pause

@echo off
setlocal

rem Draait uit een kopie in %TEMP%: cmd leest een batchbestand van schijf
rem TERWIJL het draait, dus een git pull halverwege laat het hervatten midden
rem in een woord. Dat kostte ooit een run van drie uur.
if not defined ZJ_ROOT set ZJ_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\zoekjpy-actief.cmd" >nul
  call "%TEMP%\zoekjpy-actief.cmd" __uittemp %*
  exit /b
)
shift

cd /d "%ZJ_ROOT%"

echo.
echo  ==================================================================
echo   XAUJPY -- ZOEK EEN MECHANISME, EN ZEG WANNEER HET WERKT
echo  ==================================================================
echo.
echo  WAT DIT DOET. 28 mechanismen x 3 klokken (M1, M5, M15) x 2 R:R-
echo  verhoudingen = 168 cellen op XAUJPY. Per cel: hoeveel trades per dag,
echo  hoeveel R, welke sessie ze verdient, en wat een muntworp op dezelfde
echo  bars had gedaan.
echo.
echo  WAT ER VOOR IN DE PLAATS KOMT. De oude sectie 11 was een gefit model per
echo  metaal. Het haalde de lat niet (+2,56 sigma tegen 2,96) en zijn HOLDOUT
echo  stond negatief in vier markten van de vier. Weg. Dit zoekt een REGEL met
echo  een naam die je hardop kunt zeggen, in plaats van een model.
echo.
echo  DE LAT IS HOGER DAN VORIGE KEER, en dat hoort zo: 168 cellen betekent
echo  Bonferroni 3,62 sigma. Met --uren erbij (sessieselectie) 4,01. Meer
echo  zoeken is meer kans iets te vinden dat er niet is, en dat kost.
echo.
echo  EEN CEL MOET VIER DINGEN TEGELIJK HALEN:
echo    1. sigma op of boven de lat, dag-geclusterd
echo    2. positieve totale R
echo    3. beter dan zijn eigen muntworp op dezelfde bars
echo    4. een POSITIEVE onaangeraakte holdout  ^<-- hier stierf sectie 11
echo.
echo  WAT IK AL WEET, en het is nadrukkelijk GEEN resultaat. Op synthetische
echo  XAUJPY (XAUUSD x USDJPY, 2012-2022, zonder spread) zijn de twaalf beste
echo  cellen alle twaalf FADES en geen enkele een continuation. En: de
echo  mechanismen die 3-5 trades per dag geven hadden juist de kleinste edge
echo  en negatieve holdouts. Als deze run dat tegenspreekt is dat informatie.
echo.
echo  WAT DEZE RUN NIET METT. De acht live poorten staan hier UIT. Wel actief:
echo  het echte kostenmodel, een positie tegelijk, en een bar die stop en
echo  target allebei raakt telt als VERLIES. De rest noemt hij bij naam in zijn
echo  eigen rapport. `sectie11.cmd` is de replay die ze wel aan heeft.
echo.
echo  MT5 moet draaien en ingelogd zijn, met XAUJPY zichtbaar in Market Watch.
echo.
echo  GEBRUIK -- HET GETAL IS HET AANTAL DAGEN
echo    zoekjpy.cmd 720        720 dagen, alle drie de klokken  ^<-- DEZE
echo    zoekjpy.cmd 360        korter, en de helft van de tijd
echo    zoekjpy.cmd 720 m5     alleen M5, als M1 te lang duurt
echo    zoekjpy.cmd 720 uren   SELECTEER ook de beste sessie per cel. De lat
echo                           gaat dan van 3,62 naar 4,01 om dat te betalen.
echo.

rem VORMHERKENNING, geen positie. Een getal is dagen, m1/m5/m15 is een klok,
rem `uren` zet sessieselectie aan. `zoek11.cmd 720` las 720 ooit als vormwoord
rem en maakte er stilletjes 360 van; dit is de reparatie daarvan.
set DAGEN=720
set KLOKKEN=M1 M5 M15
set UREN=

:lees
if "%~1"=="" goto klaar
echo %~1| findstr /r "^[0-9][0-9]*$" >nul && set DAGEN=%~1
if /i "%~1"=="m1" set KLOKKEN=M1
if /i "%~1"=="m5" set KLOKKEN=M5
if /i "%~1"=="m15" set KLOKKEN=M15
if /i "%~1"=="uren" set UREN=--hours
shift
goto lees
:klaar

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe bestaat niet. Draai eerst update.cmd.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

echo  %DAGEN% dagen, klokken %KLOKKEN%. M1 over 720 dagen is ongeveer een
echo  miljoen bars, dus dit duurt even. Per klok komt er eerst een regel met
echo  hoeveel bars er opgehaald zijn.
echo.

.venv-live\Scripts\python.exe -m scripts.search_xaujpy ^
  --days %DAGEN% --clocks %KLOKKEN% %UREN% ^
  --csv runtime\zoekjpy.csv

if errorlevel 1 (
  echo.
  echo  De run is gestopt. De regel hierboven zegt waarom -- meestal is MT5
  echo  niet ingelogd, of XAUJPY staat niet in Market Watch.
)

echo.
echo  Elke cel staat in runtime\zoekjpy.csv
echo.
echo  WAT JE MOET LEZEN, in deze volgorde:
echo    1. WHICH CELLS EARNED A SECTION. Staat daar NONE, dan is dat het
echo       antwoord en gaat er niets in de config. Dat is de normale uitkomst.
echo    2. De holdout-kolom. Een cel met mooie sigma en een negatieve holdout
echo       is precies waar de vorige sectie 11 op stukliep.
echo    3. PER SESSION. Betaalt hij in een sessie of over de hele dag? Een
echo       mechanisme dat alleen in een van de vijf werkt is verdacht tenzij
echo       je kunt zeggen WAAROM juist die.
echo    4. De /day-kolom. Jij wilde 3-5 per dag. Staat de beste cel op 0,5,
echo       dan is de vraag welke van de twee je opgeeft.
echo.
pause

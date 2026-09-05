@echo off
setlocal

rem Draait uit een kopie in %TEMP%: cmd leest een batchbestand van schijf
rem TERWIJL het draait, dus een git pull halverwege laat het hervatten midden
rem in een woord.
if not defined BENEN_ROOT set BENEN_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\benen-actief.cmd" >nul
  call "%TEMP%\benen-actief.cmd" __uittemp %*
  exit /b
)
shift

cd /d "%BENEN_ROOT%"

rem Elke variabele hier, bovenaan, voordat iets hem leest.
set DAGEN=720
set KLOKKEN=M5 M15

:lees
if "%~1"=="" goto klaar
echo %~1| findstr /r "^[0-9][0-9]*$" >nul && set DAGEN=%~1
if /i "%~1"=="m5" set KLOKKEN=M5
if /i "%~1"=="m15" set KLOKKEN=M15
if /i "%~1"=="m1" set KLOKKEN=M1
shift
goto lees
:klaar

if not defined DAGEN set DAGEN=720

echo.
echo  ==================================================================
echo   XAUJPY TEGEN ZIJN EIGEN TWEE BENEN
echo  ==================================================================
echo.
echo  HET MECHANISME, en dit is het eerste met een tegenpartij die je kunt
echo  benoemen:
echo.
echo      XAUJPY = XAUUSD x USDJPY
echo.
echo  Twee benen, twee orderboeken. Beweegt goud en de yen niet, dan MOET het
echo  kruis volgen -- en degene die het kruis quote doet dat met vertraging.
echo  De tegenpartij is die market maker. De trade betaalt als hij bijtrekt.
echo.
echo  WAT ER MIS WAS MET DE VORIGE. Die 28 mechanismen zijn geschreven voor
echo  index-CFD's. Het beste ervan was "vier closes op rij dezelfde kant op,
echo  dan tegengesteld instappen" -- dat vuurt op 10,3%% van alle bars, 149 keer
echo  per dag op M1, en niemand verliest geld omdat vier minuutcandles omhoog
echo  gingen. Er viel daar niets te vinden.
echo.
echo  DE EERSTE VRAAG DIE HIJ STELT, voordat er een enkele trade gerekend
echo  wordt: IS ER WEL EEN GAT? Veel brokers rekenen een kruis gewoon uit zijn
echo  benen. Doet Eightcap dat, dan is het gat per definitie nul -- geen
echo  vertraging, geen tegenpartij, geen trade. Dan zegt hij dat en stopt hij,
echo  in een minuut in plaats van na een replay van negentig dagen.
echo.
echo  Dat is getest: op een kruis dat exact het product van zijn benen is komt
echo  er 0,000000 ATR uit en zegt hij NIETS TE HANDELEN. Op een kruis dat een
echo  bar achterloopt komt er 0,23 ATR uit en vuurt drempel 1,00 ongeveer vier
echo  keer per dag op M15.
echo.
echo  WAT HIJ MEET. Vier gatdrempels x twee R:R x je klokken, met het echte
echo  kostenmodel, dag-geclusterde sigma, een muntworp-controle op dezelfde
echo  bars, een onaangeraakte holdout en een Bonferroni-lat. Een cel moet ze
echo  ALLE VIJF halen plus de kostengrens en de vuurgrens.
echo.
echo  MT5 moet draaien en ingelogd zijn, met XAUJPY, XAUUSD EN USDJPY alle
echo  drie zichtbaar in Market Watch. Zonder alle drie kan hij niets.
echo.
echo  GEBRUIK -- HET GETAL IS HET AANTAL DAGEN
echo    benen.cmd 720        720 dagen, M5 en M15   ^<-- DEZE
echo    benen.cmd 360        korter
echo    benen.cmd 720 m15    alleen M15
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe bestaat niet. Draai eerst update.cmd.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

echo  %DAGEN% dagen, klokken %KLOKKEN%. Even geduld.
echo.

.venv-live\Scripts\python.exe -m scripts.search_xaujpy_legs ^
  --days %DAGEN% --clocks %KLOKKEN% ^
  --csv runtime\benen.csv

if errorlevel 1 (
  echo.
  echo  De run is gestopt. Meestal is MT5 niet ingelogd, of staat een van de
  echo  drie symbolen niet in Market Watch.
)

echo.
echo  Elke cel staat in runtime\benen.csv
echo.
echo  WAT JE MOET LEZEN, in deze volgorde:
echo    1. IS THERE A GAP AT ALL. Staat daar NIETS TE HANDELEN, dan is dat het
echo       antwoord en is er verder niets te bespreken. Dat is geen storing.
echo    2. DID THE LAG EARN A SECTION. NONE is de normale uitkomst.
echo    3. De holdout-kolom. Mooie sigma met een negatieve holdout is precies
echo       waar de vorige sectie 11 op stukliep.
echo    4. De kosten-kolom. Een gat kleiner dan de rondrit is geen edge.
echo.
pause

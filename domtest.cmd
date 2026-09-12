@echo off
setlocal

rem Uit een kopie in %TEMP%, net als de andere launchers: cmd leest een
rem batchbestand van schijf TERWIJL het draait, dus een git pull halverwege
rem laat het hervatten midden in een woord.
if not defined DOM_ROOT set DOM_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\domtest-actief.cmd" >nul
  call "%TEMP%\domtest-actief.cmd" __uittemp %*
  exit /b
)
shift

cd /d "%DOM_ROOT%"

set DAGEN=360
set SWEEP=
rem HOLDOUT = exact het venster van jarvis-holdout-2024-2025.cmd, zodat de
rem uitkomst naast dezelfde CSV gelegd kan worden en niet naast een ander jaar.
set VENSTER=
:lees
if "%~1"=="" goto klaar
echo %~1| findstr /r "^[0-9][0-9]*$" >nul && set DAGEN=%~1
if /i "%~1"=="alles" set SWEEP=--sweep
if /i "%~1"=="holdout" set VENSTER=--start-date 2024-09-01 --end-date 2025-08-31
shift
goto lees
:klaar
if not "%VENSTER%"=="" set DAGEN=0

echo.
echo  ==========================================================================
echo   IS SECTIE ZES EEN MODEL, OF EEN KLOK MET EEN MODEL EROMHEEN
echo  ==========================================================================
echo.
echo  Sectie zes handelt goud tussen 20:00 en 02:00 UTC, ALLEEN LONG, op een
echo  bevroren nonlineair model. Dat venster overlapt de Aziatische sessie,
echo  waarvan al dertig jaar beweerd wordt dat goud er gemiddeld in oploopt.
echo.
echo  Als dat klopt, dan kan sectie zes geen model zijn maar een dure manier om
echo  een sessie-effect te handelen. Dit test dat met de domst mogelijke
echo  tegenpartij: KOOP om 20:00, VERKOOP om 02:00, elke dag. Geen model, geen
echo  drempel, geen filter, geen enkele parameter om op te passen.
echo.
echo    de domme versie doet het NET ZO GOED   het model is versiering, en je
echo                                           houdt iets over met vier in
echo                                           plaats van veertig parameters
echo.
echo    de domme versie doet het SLECHTER      het model verdient zijn plek, en
echo                                           dat is dan voor het eerst
echo                                           aangetoond
echo.
echo  ALLEBEI DIE UITKOMSTEN ZIJN WINST. Daarom staat dit vooraan.
echo.
echo  DIT IS GEEN NIEUWE KANDIDAAT. Er wordt niets gekozen en niets afgesteld;
echo  er is een regel, die stond van tevoren vast en heeft geen knop. Het is de
echo  LAT waar sectie zes overheen moet, geen strategie die live mag.
echo.
echo  DAARNA VERGELIJK JE ZELF. Zet `netto totaal` hieronder naast wat sectie
echo  zes over dezelfde periode deed volgens kosten.cmd. Dat is de hele test.
echo.
echo  GEBRUIK
echo    domtest.cmd              360 dagen, alleen het venster van sectie zes
echo    domtest.cmd holdout      HET ONAANGERAAKTE JAAR: 1 sep 2024 - 31 aug 2025,
echo                             exact het venster van jarvis-holdout-2024-2025.cmd
echo    domtest.cmd 720          twee jaar
echo    domtest.cmd 360 alles    ook alle andere vensters, als DIAGNOSE
echo.
echo  DRAAI ZE ALLEBEI. Het 360-venster is waar de uren en filters van sectie
echo  zes op gekozen zijn; de holdout is het jaar dat het systeem nooit gezien
echo  heeft. Als de domme klok het daar OOK bijhoudt, is de zaak gesloten.
echo.
echo  `alles` is geen zoektocht. Ligt 20:00-02:00 tussen even goede buren, dan
echo  is het effect breed en waarschijnlijk echt. Is het een eenzame piek, dan
echo  is het venster zelf al een keuze uit 24. PLUK ER NIETS UIT -- dat is de
echo  fout die sectie vijf drie keer van teken deed wisselen.
echo.
echo  MT5 moet draaien en ingelogd zijn.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe ontbreekt. Draai eerst update.cmd.
  pause
  exit /b 1
)

.venv-live\Scripts\python.exe -m scripts.dumb_session_baseline --days %DAGEN% %VENSTER% %SWEEP%

if errorlevel 1 (
  echo.
  echo  De run is gestopt. De traceback hierboven zegt waarom -- meestal is MT5
  echo  niet ingelogd, of er is geen historie zo ver terug.
)

echo.
pause

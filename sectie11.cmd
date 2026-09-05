@echo off
setlocal
cd /d "%~dp0"

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

echo.
echo  ===========================================================
if not exist ".venv-live\Scripts\python.exe" (
  echo FOUT: draai eerst update.cmd
  pause
  exit /b 1
)

rem GEEN MECHANISME IS GEEN RUN. Een lege `mechanism` maakt de sectie stil, en
rem een stille sectie komt terug met een nulregel die leest als een strategie
rem die niets vond. Dat onderscheid is de reden dat de vorige sectie 11 zoveel
rem verwarring heeft gekost, dus hier stopt het hardop.
.venv-live\Scripts\python.exe -c "import sys; from config.loader import DEFAULT_CONFIG_PATH, load_settings; s=load_settings(DEFAULT_CONFIG_PATH, overlay='config/eightcap.yaml', env_overrides=False); n=[m for m in ('section_eleven_xaujpy_m1','section_twelve_xaujpy_m5','section_thirteen_xaujpy_m15') if getattr(s.analysis,m).mechanism]; print('mechanismen:', ', '.join(n) if n else 'GEEN'); sys.exit(0 if n else 1)"
if errorlevel 1 (
  echo.
  echo  GEEN VAN DE DRIE SECTIES HEEFT EEN MECHANISME.
  echo.
  echo  Deze run zou nul trades opleveren en dat leest als "niets gevonden"
  echo  terwijl er nooit iets gezocht is. Draai eerst:
  echo.
  echo      zoekjpy.cmd 720
  echo.
  echo  Komt daar een cel uit die alle vier de eisen haalt, dan gaat de naam
  echo  van dat mechanisme in config\eightcap.yaml en werkt dit script.
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
  --only section_eleven_xaujpy_m1,section_twelve_xaujpy_m5,section_thirteen_xaujpy_m15 ^
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

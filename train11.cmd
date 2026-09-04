@echo off
setlocal
if not defined T11_ROOT set T11_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\train11-actief.cmd" >nul
  call "%TEMP%\train11-actief.cmd" __uittemp %*
  exit /b
)
shift
cd /d "%T11_ROOT%"

echo.
echo  ==================================================================
echo   SECTIE ELF -- EEN GETRAIND MODEL PER METAAL
echo  ==================================================================
echo.
echo  WAT DIT IS. Sectie zes is een bevroren model op goud: dertien
echo  schaalvrije aflezingen van de laatste bar, gestandaardiseerd, door een
echo  vaste willekeurige projectie naar 48 verborgen eenheden, dan een
echo  lineaire kop. Hij werkt op XAUUSD en alleen daar -- de symboolnaam
echo  staat hardcoded in de module en de coefficienten zijn op goud gefit.
echo.
echo  Dit is hetzelfde mechanisme met een eigen model per markt.
echo.
echo  WAAROM DIT ANDERS GEBOUWD IS DAN SECTIE ZES, en dat is de enige reden
echo  om het te bouwen. Sectie zes ging live, kwam eraf op -71,65 R over 180
echo  dagen, en zijn eigen config schrijft de vorm van die mislukking op: een
echo  sterke recente periode die regimeconcentratie was en geen edge. Vijf
echo  modellen op dezelfde manier fitten levert vijf keer dat op.
echo.
echo  Dus elk getal waarop dit BESLIST komt uit data die de fit nooit zag:
echo.
echo    - WALK-FORWARD. De periode wordt in blokken gesneden. Het model dat
echo      blok i voorspelt is alleen gefit op bars VOOR blok i en wordt daarna
echo      weggegooid. Elke voorspelling is dus out-of-fold per constructie.
echo    - EEN HOLDOUT WAAR NIET IN GEZOCHT IS. De nieuwste 20%% doet aan geen
echo      enkele keuze mee -- niet de drempel, niet de straf, niet welke markt
echo      een model krijgt. Hij wordt een keer gelezen en kan alleen afwijzen.
echo    - DAG-GECLUSTERDE SIGMA. Vijf metalen die op een ochtend breken zijn
echo      een waarneming, geen vijf.
echo    - EEN RANDOM CONTROLE op dezelfde bars bij hetzelfde vuurtempo. Die
echo      leest NIET nul, en wat hij leest wordt eraf getrokken.
echo    - BONFERRONI. Vier markten maal vier drempels is zestien hypotheses.
echo    - DE ECHTE KOSTEN, via de sizer, op elke genomen trade.
echo.
echo  GEBRUIK
echo    train11.cmd                 720 dagen, de vier goudkruisen, alleen
echo                                rapporteren -- er wordt niets weggeschreven
echo    train11.cmd 360             kortere periode
echo    train11.cmd 720 schrijf     schrijf een modelbestand voor elke markt
echo                                die ELKE lat haalt
echo.
echo  EEN MODEL OP SCHIJF IS GEEN SECTIE OP DE LIVE LIJST. Daarna volgt nog
echo  een replay door dry_run_sections voordat er geld aan te pas komt.
echo.
echo  MT5 moet draaien en ingelogd zijn.
echo.

set DAGEN=%1
if "%DAGEN%"=="" set DAGEN=720
set SCHRIJF=
if /i "%2"=="schrijf" set SCHRIJF=--write

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe niet gevonden. Doe eerst update.cmd.
  pause
  exit /b 1
)

.venv-live\Scripts\python.exe scripts\train_section_eleven.py --days %DAGEN% %SCHRIJF%

echo.
pause

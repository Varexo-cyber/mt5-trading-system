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
echo    train11.cmd 720 hoger       DE VERVOLGVRAAG. Op 4 september haalde
echo                                niets de lat, maar XAUJPY liet als enige
echo                                een STIJGENDE lijn zien: sigma +0,82 bij
echo                                drempel 0,10 en +2,56 bij 0,30. Als dat
echo                                signaal echt is hoort strenger selecteren
echo                                te BLIJVEN helpen; is het ruis, dan piekt
echo                                het en zakt het weer. Dit draait 0,30 tot
echo                                0,80 op XAUJPY en XAUGBP -- de twee met
echo                                die vorm -- en telt de zestien cellen van
echo                                4 september mee in de lat.
echo.
echo  EEN MODEL OP SCHIJF IS GEEN SECTIE OP DE LIVE LIJST. Daarna volgt nog
echo  een replay door dry_run_sections voordat er geld aan te pas komt.
echo.
echo  MT5 moet draaien en ingelogd zijn.
echo.

set DAGEN=%1
if "%DAGEN%"=="" set DAGEN=720
set SCHRIJF=
set EXTRA=
if /i "%2"=="schrijf" set SCHRIJF=--write
rem HOGER IS EEN VOORSPELLING, GEEN TWEEDE ZOEKTOCHT. Vooraf opgeschreven:
rem als de aflezing van XAUJPY informatie draagt, hoort sigma door te stijgen
rem naar 0,40 / 0,50 / 0,60 / 0,80. Piekt hij bij 0,30 en zakt daarna, dan is
rem het de steekproef en gaat sectie elf van tafel. De zestien cellen van de
rem eerste run tellen mee, want een grid twee keer doorzoeken en een keer
rem betalen is hoe een zoektocht zichzelf witwast tot een ontdekking.
if /i "%2"=="hoger" set EXTRA=--symbols XAUJPY,XAUGBP --thresholds 0.30,0.40,0.50,0.60,0.80 --cells-already-tried 16

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe niet gevonden. Doe eerst update.cmd.
  pause
  exit /b 1
)

.venv-live\Scripts\python.exe scripts\train_section_eleven.py --days %DAGEN% %EXTRA% %SCHRIJF%

echo.
pause

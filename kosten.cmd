@echo off
setlocal

rem Uit een kopie in %TEMP%, net als de andere launchers: cmd leest een
rem batchbestand van schijf TERWIJL het draait, dus een git pull halverwege
rem laat het hervatten midden in een woord.
if not defined KOSTEN_ROOT set KOSTEN_ROOT=%~dp0
if /i not "%~1"=="__uittemp" (
  copy /y "%~f0" "%TEMP%\kosten-actief.cmd" >nul
  call "%TEMP%\kosten-actief.cmd" __uittemp %*
  exit /b
)
shift

cd /d "%KOSTEN_ROOT%"

rem LEEG = alle CSV's in runtime\. Een naam meegeven leest alleen die.
set DOEL=%~1

echo.
echo  ==========================================================================
echo   WAT HAD HET OPGELEVERD ALS HANDELEN NIETS KOSTTE
echo  ==========================================================================
echo.
echo  DIT DRAAIT GEEN REPLAY. Het leest de CSV's die er al liggen. Elke replay
echo  schrijft per trade de kolom `cost_r_charged`, en die is AL van de
echo  R-kolommen afgetrokken. Bruto is dus netto plus kosten, per trade. Er
echo  hoeft niets opnieuw gemeten te worden en het is in een seconde klaar.
echo.
echo  WAAROM DIT DE VRAAG IS. Netto negatief betekent twee compleet
echo  verschillende dingen, en ze vragen om het tegenovergestelde besluit:
echo.
echo    bruto POSITIEF, netto negatief  De instap ziet richting. Die richting
echo                                    is kleiner dan de prijs van een rondje
echo                                    handelen. Vraag: kan de trade verder
echo                                    lopen, of goedkoper.
echo.
echo    bruto NEGATIEF                  De instap ziet niets, ook niet gratis.
echo                                    Geen enkele poort repareert dat. Verder
echo                                    afregelen is tijd in een gat gooien.
echo.
echo  Zonder deze splitsing zijn die twee niet uit elkaar te houden.
echo.
echo  EN ER KOMT EEN TWEEDE TABEL ONDER: WAT DE POORTEN WEIGERDEN. Dat is een
echo  ANDERE vraag. De eerste tabel gaat over de trades die WEL genomen zijn,
echo  zonder de kosten die eraf gingen. Deze gaat over de setups die NIET
echo  genomen zijn omdat een poort ze weigerde. De replay heeft die tóch
echo  uitgelopen en het resultaat weggeschreven, dus ook dat staat er al in.
echo.
echo    positieve R  de poort haalde WINST weg
echo    negatieve R  de poort hield VERLIES tegen
echo.
echo  DAT IS GEEN REPLAY MET DE POORTEN UIT. Elke geweigerde setup is los
echo  uitgelopen alsof hij er alleen stond. Met de poorten echt uit hadden ze
echo  slots bezet en andere trades verdrongen. Lees die kolom als richting,
echo  niet als een bedrag dat je misgelopen bent.
echo.
echo  BRUTO IS GEEN HAALBAAR RESULTAAT. Handelen zonder spread bestaat niet.
echo  Het zegt WAAR het verlies zit, niet dat het te vermijden was.
echo.
echo  GEBRUIK
echo    kosten.cmd            de holdout 2024-2025 PLUS de 360-daagse run,
echo                          met een totaal over allebei -- dat is 2024 t/m 2026
echo    kosten.cmd alles      elke CSV in runtime\, los
echo    kosten.cmd runtime\jarvis-holdout-2024-2025.csv   alleen die ene
echo.
echo  OVERLAPPENDE PERIODES WORDEN NIET OPGETELD. `september-2025-goud.csv`
echo  valt middenin `hoeveel-goud-360.csv`; die twee bij elkaar optellen telt
echo  dezelfde trades dubbel. Dan komt er geen totaal en staat erbij waarom.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  FOUT: .venv-live\Scripts\python.exe ontbreekt. Draai eerst update.cmd.
  pause
  exit /b 1
)

rem GEEN WILDCARD DOORGEVEN AAN PYTHON. cmd expandeert `runtime\*.csv` niet
rem zelf en python krijgt dan de letterlijke ster, wat een bestandsnaam is die
rem niet bestaat -- en dat leest als "geen trades" in plaats van als een fout.
if /i "%DOEL%"=="alles" (
  set GEVONDEN=
  for %%f in (runtime\*.csv) do (
    set GEVONDEN=1
    .venv-live\Scripts\python.exe -m scripts.gross_vs_net "%%f"
  )
  if not defined GEVONDEN (
    echo  Geen enkele CSV in runtime\. Draai eerst hoeveel.cmd, beheer.cmd of
    echo  jarvis-holdout-2024-2025.cmd.
  )
  goto einde
)

if not "%DOEL%"=="" (
  .venv-live\Scripts\python.exe -m scripts.gross_vs_net "%DOEL%"
  goto einde
)

rem STANDAARD: DE TWEE BESTANDEN DIE SAMEN 2024 T/M 2026 DEKKEN, in EEN aanroep
rem zodat het script er een totaal over kan trekken. Apart aanroepen geeft twee
rem losse tabellen en geen som, en dat was juist de vraag.
set SAMEN=
if exist "runtime\jarvis-holdout-2024-2025.csv" set SAMEN=%SAMEN% "runtime\jarvis-holdout-2024-2025.csv"
if exist "runtime\hoeveel-goud-360.csv" set SAMEN=%SAMEN% "runtime\hoeveel-goud-360.csv"
if not defined SAMEN (
  echo  Geen van de twee verwachte bestanden gevonden:
  echo    runtime\jarvis-holdout-2024-2025.csv   uit jarvis-holdout-2024-2025.cmd
  echo    runtime\hoeveel-goud-360.csv           uit goud360.cmd
  echo.
  echo  Draai `kosten.cmd alles` om te zien wat er wel ligt.
  goto einde
)
.venv-live\Scripts\python.exe -m scripts.gross_vs_net %SAMEN%

:einde
echo.
pause

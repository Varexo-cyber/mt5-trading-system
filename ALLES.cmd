@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem ============================================================
rem   ALLES IN EEN KEER.
rem
rem   Dubbelklik dit bestand. Meer niet.
rem
rem   Hij haalt de nieuwste code op, exporteert je bars uit MT5,
rem   draait sectie 20 en sectie 21, en schrijft ALLES naar een
rem   tekstbestand dat je zo in de chat kunt slepen.
rem
rem   Je hoeft niets te typen, niets te kiezen en niets te lezen.
rem ============================================================

set BALANS=%1
if "%BALANS%"=="" set BALANS=400

rem TWEE JAAR, want dat is wat er gevraagd is. Let op: MT5 bewaart niet
rem onbeperkt M1-historie -- krijg je veel minder bars dan verwacht, dan zet
rem je in de terminal Extra > Opties > Grafieken > "Max bars in grafiek"
rem op Onbeperkt en scroll je de M1-grafiek een keer ver terug.
set DAGEN=%2
if "%DAGEN%"=="" set DAGEN=730

rem GEEN DATUM UIT %DATE% KNIPPEN. Dat formaat hangt af van je
rem taalinstelling: "Fri 09/18/2026" werd hier "uitslag-2026 0Fr.txt".
rem Een vaste naam kan niet misgaan en is makkelijker terug te vinden.
set UIT=runtime\uitslag.txt
if not exist runtime mkdir runtime

echo.
echo   ===========================================
echo     ALLES.cmd  --  dit duurt een paar minuten
echo   ===========================================
echo.
echo   Zorg dat MetaTrader 5 OPEN staat en ingelogd is.
echo   Verder hoef je niets te doen.
echo.
pause

rem PYTHON ZOEKEN, EN HEM ECHT PROBEREN.
rem
rem `where python` SLAAGT op Windows ook als er geen Python staat: er ligt een
rem nep-python.exe van de Microsoft Store in WindowsApps die alleen een
rem reclameregel print. Daar liep dit script op stuk --
rem "Python was not found; run without arguments to install from the
rem Microsoft Store" -- terwijl het dacht dat het Python had.
rem
rem De enige betrouwbare test is hem DRAAIEN en de exitcode bekijken.
set PY=
if exist ".venv-live\Scripts\python.exe" (
  .venv-live\Scripts\python.exe -c "pass" >nul 2>&1 && set PY=.venv-live\Scripts\python.exe
)
if not defined PY if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe -c "pass" >nul 2>&1 && set PY=.venv\Scripts\python.exe
)
if not defined PY ( py -3 -c "pass" >nul 2>&1 && set PY=py -3 )
if not defined PY ( python -c "pass" >nul 2>&1 && set PY=python )
if not defined PY ( python3 -c "pass" >nul 2>&1 && set PY=python3 )

if not defined PY (
  echo.
  echo   ============================================================
  echo     GEEN WERKENDE PYTHON GEVONDEN
  echo   ============================================================
  echo.
  echo   Dit project heeft Python nodig. Gezocht op:
  echo     .venv-live\Scripts\python.exe
  echo     .venv\Scripts\python.exe
  echo     py -3
  echo     python  /  python3
  echo.
  echo   Krijg je "install from the Microsoft Store"? Dan staat er een
  echo   NEP-python op je pad en is er geen echte geinstalleerd.
  echo.
  echo   Draai eerst:   SETUP.cmd
  echo.
  pause
  exit /b 1
)

echo   Python gevonden: %PY%

rem DE INLOGGEGEVENS, EN DE FOUT DIE DAAROVER GING WAS ENGELS.
rem
rem `core.errors.ConfigError: missing MT5 credential(s)` is een correcte
rem melding uit de kern -- en de kern is met opzet Engels. Maar hij komt hier
rem terecht bij iemand die Nederlands leest, dus we vangen hem hier af.
rem
rem En als het bestand in de andere kopie van dit project wel staat, halen we
rem hem daar op in plaats van erover te klagen. Het is zijn eigen bestand op
rem zijn eigen machine, en config\.env staat in .gitignore.
set OUDEENV=C:\Users\%USERNAME%\Documents\Codex\2026-08-01\build\work\mt5-trading-system\config\.env
if not exist "config\.env" (
  if exist "%OUDEENV%" (
    echo   config\.env ontbrak -- overgenomen uit je andere projectkopie.
    copy /y "%OUDEENV%" "config\.env" >nul
  )
)
rem BESTAAN IS NIET GENOEG, EN DAAR LIEP HET OP STUK.
rem
rem De kopie uit de andere projectmap werd netjes neergezet en bevatte geen
rem inloggegevens -- waarschijnlijk gewoon het voorbeeldbestand. Mijn controle
rem keek of het bestand ER WAS en niet of er IETS IN STOND, dus meldde hij
rem "overgenomen" en viel het daarna alsnog om op dezelfde fout.
rem
rem Nu wordt er gezocht naar een ingevulde MT5_LOGIN. Niet naar de regel --
rem naar een regel met een waarde erachter die niet het voorbeeld is.
set HEEFTLOGIN=
if exist "config\.env" (
  findstr /r /c:"^MT5_LOGIN=[0-9][0-9]*" "config\.env" >nul 2>&1 && set HEEFTLOGIN=1
  findstr /c:"MT5_LOGIN=12345678" "config\.env" >nul 2>&1 && set HEEFTLOGIN=
)
if not defined HEEFTLOGIN (
  echo.
  echo   ============================================================
  echo     JE MT5-INLOGGEGEVENS ONTBREKEN
  echo   ============================================================
  echo.
  echo   In config\.env staat geen ingevuld MT5_LOGIN. Daarin hoort met
  echo   welk account ik bij MetaTrader 5 mag. Zonder dat: geen bars.
  echo.
  echo   Ik open het bestand nu voor je. Vul de drie regels in, sla op,
  echo   sluit Kladblok en draai ALLES.cmd opnieuw.
  echo.
  echo   Vul dan deze drie regels in:
  echo      MT5_LOGIN=je accountnummer
  echo      MT5_PASSWORD=je wachtwoord
  echo      MT5_SERVER=bijvoorbeeld Eightcap-Live
  echo.
  echo   Die staan in MetaTrader 5 onder Extra ^> Opties ^> Server.
  echo   Dit bestand blijft op je eigen computer; het staat in .gitignore.
  echo.
  if not exist "config\.env" copy config\.env.example "config\.env" >nul
  start "" notepad "config\.env"
  pause
  exit /b 1
)

rem EN OF DE PAKKETTEN ER ZIJN. Python hebben is niet hetzelfde als pandas
rem hebben, en dat verschil merk je anders pas drie stappen verderop.
%PY% -c "import pandas, numpy, pydantic, yaml" >nul 2>&1
if errorlevel 1 (
  echo.
  echo   Python werkt, maar de pakketten ontbreken ^(pandas/numpy/pydantic^).
  echo   Draai eerst:   SETUP.cmd
  echo.
  pause
  exit /b 1
)

echo ==================================================== > "%UIT%"
echo   UITSLAG  %DATE% %TIME% >> "%UIT%"
echo ==================================================== >> "%UIT%"

echo.
echo   [1/5] nieuwste code ophalen...
git pull origin claude/mt5-autonomous-trading-system-ujd1sk >> "%UIT%" 2>&1

echo   [2/5] bars uit MT5 exporteren (%DAGEN% dagen M1)...
echo. >> "%UIT%"
echo ---------- BARS ---------- >> "%UIT%"
%PY% -m scripts.exporteer_bars --days %DAGEN% >> "%UIT%" 2>&1

if not exist "runtime\xauusd_m1.csv" (
  echo.
  echo   !! De export is mislukt. Staat MetaTrader 5 open en ingelogd?
  echo      De reden staat in %UIT%
  echo.
  echo   Sectie 22 heeft de bars NIET nodig, die draai ik alvast wel.
  echo. >> "%UIT%"
  echo ---------- SECTIE 22: CLAIM-AUDIT ---------- >> "%UIT%"
  %PY% -m scripts.section_twentytwo_claim_audit >> "%UIT%" 2>&1
  notepad "%UIT%"
  pause
  exit /b 1
)

echo   [3/5] sectie 20 -- de terugval-ladder...
echo. >> "%UIT%"
echo ---------- SECTIE 20: LADDER ---------- >> "%UIT%"
%PY% -m scripts.section_twenty_pullback_ladder --csv runtime\xauusd_m1.csv --balans %BALANS% >> "%UIT%" 2>&1
%PY% -m scripts.section_twenty_pullback_ladder --csv runtime\xauusd_m1.csv --balans %BALANS% --alle-configs >> "%UIT%" 2>&1

echo   [4/5] sectie 21 -- de straddle...
echo. >> "%UIT%"
echo ---------- SECTIE 21: STRADDLE ---------- >> "%UIT%"
%PY% -m scripts.section_twentyone_straddle --csv runtime\xauusd_m1.csv >> "%UIT%" 2>&1
%PY% -m scripts.section_twentyone_straddle --csv runtime\xauusd_m1.csv --alle-configs >> "%UIT%" 2>&1

echo   [5/5] sectie 22 -- trackrecords toetsen aan hun eigen beweringen...
echo. >> "%UIT%"
echo ---------- SECTIE 22: CLAIM-AUDIT ---------- >> "%UIT%"
%PY% -m scripts.section_twentytwo_claim_audit >> "%UIT%" 2>&1

echo.
echo   ===========================================
echo     KLAAR.
echo   ===========================================
echo.
echo   Alles staat in:  %CD%\%UIT%
echo.
echo   Sleep dat bestand in de chat. Meer hoef je niet te doen.
echo.
echo   (Het bestand opent nu vanzelf.)
echo.
start "" notepad "%UIT%"
pause

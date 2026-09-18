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
rem JE ECHTE BALANS. Stond op 400 uit een oud voorbeeld; dat was
rem niet jouw rekening en dus ook niet jouw antwoord.
if "%BALANS%"=="" set BALANS=59.16

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
rem DE POORT DIE ER NIET MEER HOORT TE STAAN.
rem
rem Hier stond een controle die eiste dat config\.env een ingevulde MT5_LOGIN
rem had, anders stoppen. Die poort is gebouwd toen de scripts inloggegevens
rem NODIG hadden. Twee commits later heb ik ze optioneel gemaakt -- de scripts
rem haken sindsdien aan bij een terminal die al draait -- en deze poort bleef
rem staan. Dus weigerde de launcher te starten om iets wat niemand meer vroeg.
rem
rem De eigenaar zei dit drie keer voordat ik het zag. Hij had gelijk.
rem
rem Wat er nu gebeurt: staan er inloggegevens, dan worden ze gebruikt. Staan ze
rem er niet, dan haken we aan. Geen van beide is een reden om te stoppen.
if exist "config\.env" (
  findstr /r /c:"^MT5_LOGIN=[0-9][0-9]*" "config\.env" >nul 2>&1 && (
    echo   Inloggegevens gevonden in config\.env
  ) || (
    echo   Geen MT5-inloggegevens -- ik haak aan bij je draaiende terminal.
  )
) else (
  echo   Geen config\.env -- ik haak aan bij je draaiende terminal.
)

echo   [1/10] nieuwste code ophalen...
git pull origin claude/mt5-autonomous-trading-system-ujd1sk >> "%UIT%" 2>&1

echo   [2/10] de BROKER uitmeten (hefboom, spread, stop-out, marge)...
echo. >> "%UIT%"
echo ---------- DE BROKER, GEMETEN UIT DE TERMINAL ---------- >> "%UIT%"
%PY% -m scripts.uitvoering --symbool XAUUSD >> "%UIT%" 2>&1

rem WELKE BALANS JE NODIG HEBT. Dit is rekenwerk en geen backtest, dus het
rem draait ook als de bars niet opgehaald kunnen worden -- en het is het
rem antwoord op de vraag die je stelde, dus het staat bovenaan.
echo   [3/10] welke balans heb je nodig...
echo. >> "%UIT%"
echo ---------- WELKE BALANS HEB JE NODIG ---------- >> "%UIT%"
%PY% -m scripts.welke_balans --balans %BALANS% >> "%UIT%" 2>&1

echo   [4/10] bars uit MT5 exporteren (%DAGEN% dagen M1)...
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

rem NIET MEER ALLES NAAR HET BESTAND WEGGOOIEN.
rem
rem Hier stond overal ">> %UIT%", dus het scherm bleef twintig minuten leeg en
rem dat is niet te onderscheiden van een vastgelopen programma. Het antwoord op
rem de vraag ("wat zou mijn balans nu zijn?") print nu OP HET SCHERM terwijl hij
rem werkt, en schrijft zichzelf ook naar het bestand.
echo   [5/10] EINDRESULTAAT -- wat je balans na deze periode zou zijn...
echo.
echo. >> "%UIT%"
echo ---------- EINDRESULTAAT ---------- >> "%UIT%"
%PY% -m scripts.eindresultaat --csv runtime\xauusd_m1.csv --balans %BALANS% --rapport "%UIT%"

echo.
echo   [6/10] SECTIE 23 -- uitstap: BE, trailen, deels eruit, tijdstop...
echo.
echo. >> "%UIT%"
echo ---------- SECTIE 23: UITSTAP + BALANSLADDER ---------- >> "%UIT%"
%PY% -m scripts.section_twentythree_uitstap --csv runtime\xauusd_m1.csv --balans %BALANS% --rapport "%UIT%"

echo.
echo   [7/10] DE ZOEKTOCHT -- hij stelt zichzelf bij tot er niets beters komt...
echo.
echo. >> "%UIT%"
echo ---------- DE ZOEKTOCHT ---------- >> "%UIT%"
%PY% -m scripts.zoektocht --csv runtime\xauusd_m1.csv --balans %BALANS% --rapport "%UIT%"

echo.
echo   [8/10] HET DAGBOEK -- elke trade apart, met bar, reden en verloop...
echo.
%PY% -m scripts.dagboek --csv runtime\xauusd_m1.csv --balans %BALANS% --uit runtime\dagboek.txt

echo.
echo   [9/10] alle configuraties doorrekenen ^(dit is de lange^)...
echo. >> "%UIT%"
echo ---------- SECTIE 20: ALLE 72 CONFIGURATIES ---------- >> "%UIT%"
%PY% -m scripts.section_twenty_pullback_ladder --csv runtime\xauusd_m1.csv --balans %BALANS% --alle-configs >> "%UIT%" 2>&1
echo. >> "%UIT%"
echo ---------- SECTIE 21: ALLE 48 CONFIGURATIES ---------- >> "%UIT%"
%PY% -m scripts.section_twentyone_straddle --csv runtime\xauusd_m1.csv --alle-configs >> "%UIT%" 2>&1

echo   [10/10] sectie 22 -- trackrecords toetsen aan hun eigen beweringen...
echo. >> "%UIT%"
echo ---------- SECTIE 22: CLAIM-AUDIT ---------- >> "%UIT%"
%PY% -m scripts.section_twentytwo_claim_audit >> "%UIT%" 2>&1

echo.
echo   ===========================================
echo     KLAAR.
echo   ===========================================
echo.
echo   Alles staat in:  %CD%\%UIT%
echo   Het dagboek in:  %CD%\runtime\dagboek.txt
echo.
echo   Sleep dat bestand in de chat. Meer hoef je niet te doen.
echo.
echo   (Het bestand opent nu vanzelf.)
echo.
start "" notepad "%UIT%"
pause

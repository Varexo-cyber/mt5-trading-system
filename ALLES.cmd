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

set STAMP=%DATE:~-4%%DATE:~3,2%%DATE:~0,2%
set UIT=runtime\uitslag-%STAMP%.txt
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

set PY=
if exist ".venv-live\Scripts\python.exe" set PY=.venv-live\Scripts\python.exe
if not defined PY if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if not defined PY (where py >nul 2>&1 && set PY=py -3)
if not defined PY (where python >nul 2>&1 && set PY=python)
if not defined PY (
  echo   Geen Python gevonden. Draai eerst install-market-feed.cmd.
  pause
  exit /b 1
)

echo ==================================================== > "%UIT%"
echo   UITSLAG  %DATE% %TIME% >> "%UIT%"
echo ==================================================== >> "%UIT%"

echo.
echo   [1/4] nieuwste code ophalen...
git pull origin claude/mt5-autonomous-trading-system-ujd1sk >> "%UIT%" 2>&1

echo   [2/4] bars uit MT5 exporteren (180 dagen M1)...
echo. >> "%UIT%"
echo ---------- BARS ---------- >> "%UIT%"
%PY% -m scripts.exporteer_bars --days 180 >> "%UIT%" 2>&1

if not exist "runtime\xauusd_m1.csv" (
  echo.
  echo   !! De export is mislukt. Staat MetaTrader 5 open en ingelogd?
  echo      De reden staat in %UIT%
  echo.
  notepad "%UIT%"
  pause
  exit /b 1
)

echo   [3/4] sectie 20 -- de terugval-ladder...
echo. >> "%UIT%"
echo ---------- SECTIE 20: LADDER ---------- >> "%UIT%"
%PY% -m scripts.section_twenty_pullback_ladder --csv runtime\xauusd_m1.csv --balans %BALANS% >> "%UIT%" 2>&1
%PY% -m scripts.section_twenty_pullback_ladder --csv runtime\xauusd_m1.csv --balans %BALANS% --alle-configs >> "%UIT%" 2>&1

echo   [4/4] sectie 21 -- de straddle...
echo. >> "%UIT%"
echo ---------- SECTIE 21: STRADDLE ---------- >> "%UIT%"
%PY% -m scripts.section_twentyone_straddle --csv runtime\xauusd_m1.csv >> "%UIT%" 2>&1
%PY% -m scripts.section_twentyone_straddle --csv runtime\xauusd_m1.csv --alle-configs >> "%UIT%" 2>&1

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

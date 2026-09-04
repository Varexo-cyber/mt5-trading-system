@echo off
setlocal
cd /d "%~dp0"
echo.
echo  ==================================================================
echo   DE LAATSTE RUN OPNIEUW LEZEN -- ZONDER HEM OPNIEUW TE DRAAIEN
echo  ==================================================================
echo.
echo  De dry run schrijft elke beslissing naar CSV en gooit daarna alles
echo  weg, dus een vraag die het rapport toevallig niet beantwoordde kostte
echo  een nieuwe run van twee uur op bars die niet veranderd zijn.
echo.
echo  Dit leest dezelfde run terug. Geen MT5, geen fetch, twee seconden.
echo.
echo  GEBRUIK
echo    lees.cmd                    runtime\sectie10.csv
echo    lees.cmd runtime\dryrun-live.csv
echo    lees.cmd runtime\sectie10.csv fixed    de vaste SL/TP in plaats van
echo                                           de break-evenuitgang
echo.

set BESTAND=%1
if "%BESTAND%"=="" set BESTAND=runtime\sectie10.csv
set UITGANG=
if /i "%2"=="fixed" set UITGANG=--exit fixed

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe niet gevonden. Doe eerst update.cmd.
  pause
  exit /b 1
)

.venv-live\Scripts\python.exe scripts\read_dryrun_csv.py "%BESTAND%" %UITGANG%

echo.
pause

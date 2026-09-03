@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Scorecard - where this account makes money
echo   and where it loses it
echo  ============================================
echo.
echo  Journal read-only. No orders, no Claude API, nothing written.
echo.
echo  GEBRUIK
echo    scorecard.cmd          30 dagen
echo    scorecard.cmd 90       90 dagen
echo    scorecard.cmd --days 7 --regime NAAM     alles wat het script kent
echo.
echo  HET BLOK WAAR HET OM GAAT heet
echo    WHAT THE GATES REFUSED, AND WHAT IT WOULD HAVE DONE
echo  Elke poort die een setup weigerde heeft die setup mee laten lopen
echo  tegen de echte koers. "cost us" = die poort gooit winnaars weg.
echo  "saved us" = hij verdient zijn plek.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

rem EEN KAAL GETAL IS EEN AANTAL DAGEN. Elke andere launcher hier neemt er
rem een -- dryrun-live.cmd, zoek11.cmd, waarom.cmd -- en deze gaf op precies
rem hetzelfde `scorecard.cmd 30` een argparse-usage terug. Dat is de launcher
rem die scheef staat, niet degene die hem intypt.
set ARGS=%*
if "%~1"=="" set ARGS=--days 30
set ISGETAL=0
echo %~1| findstr /r /c:"^[0-9][0-9]*$" >nul 2>&1 && set ISGETAL=1
if "%ISGETAL%"=="1" set ARGS=--days %*

".venv-live\Scripts\python.exe" scripts\scorecard.py %ARGS%

echo.
pause
endlocal

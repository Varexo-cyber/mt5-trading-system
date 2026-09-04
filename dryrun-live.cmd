@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   DE HELE LIVE BOEK, EXACTLY AS CONFIGURED
echo  ============================================
echo.
echo  LIVE ALLOWLIST, 4 SEPTEMBER: sectie 6 (goud M5), 7 (failed session
echo  breakout), 8 (trend day H1) en 10 (goud M1). Sectie 5, 9 en 11 blijven
echo  enabled voor shadowmeting maar mogen geen geld gebruiken.
echo  Every section runs on its own configured clock, with the account's
echo  real position cap. No impulse or old order-block module is measured.
echo.
echo  TWEE DINGEN ZIJN OP 4 SEPTEMBER TERUGGEDRAAID, allebei op een cijfer:
echo    SECTIE 10 staat weer op ALLEEN XAUUSD. Met de vier kruisen erbij deed
echo    hij -31,10 R over 180 dagen; goud alleen deed +79,12 R over 595 trades.
echo    De kruisen kostten 110 R en verdrongen bovendien 260 sectie-6-trades
echo    die +40,48 R waard waren.
echo    SECTIE 11 is er weer af. Hij haalde de statistische lat niet (+2,56
echo    sigma tegen 2,96) en zijn holdout stond in vier markten van de vier
echo    negatief. Meten kan nog met sectie11.cmd; geld uitgeven niet.
echo.
echo  POSITIEBEHEER: sectie 6, 7 en 8 krijgen de break-even-stap zoals live.
echo  Sectie 10 niet -- die staat op vaste SL/TP en wordt om 20:50 platgezet,
echo  en de replay doet nu precies dat.
echo.
echo  ZESTIEN KERNMARKTEN PLUS DE MARKTEN VAN DE SECTIES ZELF. De kern is de
echo  elf FX-majors, goud en vier index-CFDs. Handelt een live sectie een markt
echo  die daar niet in zit, dan wordt die erbij gezet en met naam genoemd --
echo  anders komt zo'n sectie terug met een nulregel omdat hij nooit een markt
echo  zag die hij mag handelen, en dat leest als "de strategie vond niets".
echo  Elke andere markt uit de catalogus blijft een extrapolatie.
echo.
echo  Roughly a fifteenth of the work of a full sweep.
echo.
echo  MT5 must be running and logged in.
echo.
echo  USAGE  -- HET EERSTE GETAL IS HET AANTAL DAGEN. Zonder getal is het 7,
echo         en 7 dagen is te weinig om iets over een sectie te zeggen.
echo    dryrun-live.cmd 180        180 days, core markets  ^<-- DEZE
echo    dryrun-live.cmd 90         90 days  (ongeveer de helft van de tijd)
echo    dryrun-live.cmd            7 days, core markets
echo    dryrun-live.cmd 7 all      7 days, EVERY market the scanner sees
echo                               (all ~230: forex, crypto, metals, indices,
echo                                commodities. Stocks are never scanned.)
echo.

rem PLAIN WORDS ONLY, NO COMMAS. cmd splits arguments on commas, so a list
rem typed at the prompt loses the rest of the command line before the script
rem is ever called.

set DAYS=%1
if "%DAYS%"=="" set DAYS=7
set SCOPE=--core
set FINE=
rem THE WHOLE CATALOGUE IS FOURTEEN TIMES THE MARKETS, so it drops to M5
rem resolution. An M30 trade resolved on M5 bars is a 6:1 ratio, which is
rem enough to tell which barrier price reached first; M1 over 230 markets is
rem tens of millions of bars for no extra answer.
if /i "%2"=="all" (set SCOPE=& set FINE=--no-m1)

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe not found. Run update.cmd first.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

.venv-live\Scripts\python.exe -m scripts.dry_run_sections --days %DAYS% %SCOPE% %FINE% --live-only --sections-five-to-ten --csv runtime\dryrun-live.csv

if errorlevel 1 (
  echo.
  echo  The run failed. The traceback above says why -- most often MT5 is not
  echo  logged in, or the symbols have no history that far back.
)

echo.
echo  Every single decision, one row each, is in runtime\dryrun-live.csv
echo.
pause

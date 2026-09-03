@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ==================================================================
echo   ZOEKEN NAAR SECTIE ELF -- GOUD, INTRADAY, ZESTIEN MECHANISMEN
echo  ==================================================================
echo.
echo  WAAROM DIT EERST DRAAIT EN ER NOG GEEN SECTIE ELF IS.
echo.
echo  Sectie 6 ging live op een cijfer dat later -71,65 R bleek. Sectie 5
echo  en 9 gingen er weer af. Dat gebeurde niet omdat de code fout was maar
echo  omdat de parameters gekozen waren VOORDAT er gemeten was. Een sectie
echo  bouwen en daarna hopen dat de meting meevalt is precies die fout nog
echo  een keer maken, met een nieuw nummer erop.
echo.
echo  Dus dit draait eerst. Wat hier doorheen komt WORDT sectie elf; komt
echo  er niets door, dan is dat het antwoord en niet een mislukking.
echo.
echo  WAT ER GEMETEN WORDT -- acht mechanismen, elk in twee richtingen:
echo    stretch            prijs 1,8 ATR van zijn eigen 2-uursgemiddelde
echo    quiet_stretch      hetzelfde, alleen 00:00-06:00 UTC (dunne boek)
echo    london             07:00 UTC, de eerste uurbeweging
echo    comex              13:00 UTC, US-data en de COMEX-opening
echo    pm_fix             het 15:00-Londen-fixingvenster
echo    round_number       de tien-dollarniveaus waar het boek op klit
echo    opening_range      het eerste uur van de gouddag, daarna de break
echo    day_range          de dag heeft zijn normale range al gelopen
echo.
echo  Elk mechanisme wordt zowel MEEGENOMEN als GEFADED. Alleen de helft
echo  publiceren die toevallig betaalde is de sample fitten.
echo.
echo  DE LAT STAAT HOGER DAN 2,0 SIGMA EN DAT IS OPZET. Zestien kandidaten
echo  op twee klokken is 32 hypotheses; de beste van 32 haalt 2 sigma op
echo  pure ruis in ruim de helft van de gevallen. De Bonferroni-lat komt
echo  daardoor rond 3,2 sigma te liggen op de trainhelft, EN de holdout --
echo  de nieuwste 40%% van de periode, waar niet in gezocht is -- moet op
echo  eigen kracht 2,0 halen. Plus: een random-instap op dezelfde bars
echo  wordt eraf getrokken, want die leest NIET nul.
echo.
echo  MT5 moet draaien en ingelogd zijn. Dit duurt lang -- 360 dagen goud
echo  op M5 is ongeveer 100.000 bars per markt, 16 keer doorlopen.
echo.
echo  GEBRUIK
echo    zoek11.cmd                 360 dagen, alle metalen, M5 en M15
echo    zoek11.cmd 720             720 dagen  (meer holdout = hardere test)
echo    zoek11.cmd 360 M1          ook op M1  -- traag, veel bars
echo    zoek11.cmd 360 M5 XAUUSD   alleen goud
echo.

set DAGEN=%1
if "%DAGEN%"=="" set DAGEN=360

set KLOKKEN=%2
if "%KLOKKEN%"=="" set KLOKKEN=M5 M15

rem STANDAARD ALLE METALEN, niet alleen XAUUSD. Eightcap noteert goud ook
rem tegen EUR, GBP en AUD en zilver ernaast, en de scanner-classificatie
rem vindt hoe deze broker ze spelt -- ik ga niet gokken naar de namen.
rem
rem DIT IS GEEN ONAFHANKELIJK BEWIJS. XAUUSD en XAUEUR zijn hetzelfde goud
rem door een valutabril: als goud 's ochtends breekt breekt het in alle
rem vier. Ze staan er wel in omdat de KOSTEN per kruis flink verschillen
rem en dat verschil beslist of iets betaalbaar is. De sigma wordt daarom
rem per DAG geclusterd, niet per trade -- vier kruisen die samen bewegen
rem zijn een observatie.
set MARKTEN=--asset-class metal
if not "%3"=="" set MARKTEN=--symbols %3

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe niet gevonden. Doe eerst update.cmd.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

rem --cells-already-tried 12: de index-familie is eerder over deze bars
rem gezocht. Twee zoektochten van elk 32 cellen zijn 64 hypotheses, en
rem twee keer voor 32 betalen is hoe een zoektocht zichzelf witwast tot
rem een ontdekking.
.venv-live\Scripts\python.exe -m scripts.search_section_four ^
  --days %DAGEN% --family gold --clocks %KLOKKEN% %MARKTEN% ^
  --stop-atr 1.0 --ratio 1.0 --horizon 48 ^
  --cells-already-tried 12 ^
  --csv runtime\zoek11.csv

if errorlevel 1 (
  echo.
  echo  De run is gestopt. De regel hierboven zegt waarom -- meestal is MT5
  echo  niet ingelogd, of er is geen historie zo diep terug.
)

echo.
echo  Elke cel met zijn cijfers staat in runtime\zoek11.csv
echo.
echo  WAT JE MOET LEZEN, in deze volgorde:
echo    1. WHAT A ROUND TRIP COSTS. Staat metal boven ~25%% van de stop?
echo       Dan betaalt geen enkele instap op deze klok en is de rest ruis.
echo    2. NEVER FIRED. Een kandidaat die nooit vuurde is mijn drempel die
echo       niet bij deze feed past, GEEN gemeten mislukking. Zeg het me.
echo    3. SURVIVED EVERY BAR. Wat hier staat wordt sectie elf.
echo    4. CLOSEST MISSES. Staan ze allemaal op "holdout, train only", dan
echo       vindt de zoektocht ruis en helpt verder sleutelen niet.
echo.
pause

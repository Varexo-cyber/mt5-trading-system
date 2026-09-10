@echo off
setlocal
cd /d "%~dp0"

set DAGEN=180
set SECTIES=section_five_ndx100_m5,section_six_gold_m5
if not "%~1"=="" set DAGEN=%~1
if /i "%~2"=="vijf" set SECTIES=section_five_ndx100_m5
if /i "%~2"=="zes" set SECTIES=section_six_gold_m5

echo.
echo  SLIMME FOUT-EXIT METING -- %DAGEN% DAGEN -- SHADOW ONLY
echo  Meet bespaarde verliezen, beschermde winst en weggegooide winst.
echo  Wijzigt NIETS aan live-enabled secties of live exits.
echo.

.venv-live\Scripts\python.exe -m scripts.dry_run_sections --days %DAGEN% --section-markets --live-only --jarvis-replay --fault-exit-grid --only "%SECTIES%"

pause

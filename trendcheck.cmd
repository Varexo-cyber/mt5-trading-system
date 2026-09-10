@echo off
setlocal
cd /d "%~dp0"

set DAGEN=30
set SECTIES=section_five_ndx100_m5,section_six_gold_m5
if not "%~1"=="" set DAGEN=%~1
if /i "%~2"=="vijf" set SECTIES=section_five_ndx100_m5
if /i "%~2"=="zes" set SECTIES=section_six_gold_m5

echo.
echo  KORTE TRENDFILTER-METING -- %DAGEN% DAGEN
echo  Meet M5, M15 en M5+M15 op dezelfde S5/S6 trades.
echo  Wijzigt niets aan live Jarvis en draait geen trage exit-grid.
echo.

.venv-live\Scripts\python.exe -m scripts.dry_run_sections --days %DAGEN% --section-markets --live-only --jarvis-replay --trend-grid --only "%SECTIES%"

pause

@echo off
setlocal
cd /d "%~dp0"

echo.
echo  JARVIS WEEKEND VALIDATION - SECTIONS 1, 2 AND 6
echo.
echo  No orders are sent. No Claude API is called. Live settings are not changed.
echo  Every strategy is measured separately on its own H1, M5 or M1 clock.
echo  Spread, commission and slippage are charged. The final 20%% stays locked.
echo  Use --strategy NAME to run one module instead of the whole suite.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  exit /b 1
)

if "%~1"=="" (
  ".venv-live\Scripts\python.exe" scripts\validate_weekend_sections.py --days 180
) else (
  echo %~1| findstr /r "^[0-9][0-9]*$" >nul
  if not errorlevel 1 (
    ".venv-live\Scripts\python.exe" scripts\validate_weekend_sections.py --days %~1
  ) else (
    ".venv-live\Scripts\python.exe" scripts\validate_weekend_sections.py %*
  )
)

echo.
pause
endlocal

@echo off
setlocal
cd /d "%~dp0"

echo.
echo  GOLD CLOSE MOMENTUM - 2 JAAR - ALLEEN SHADOW
echo  Een vooraf vastgelegde JFE-regel, geen Jarvis-sectie en geen live wijziging.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo FOUT: .venv-live ontbreekt. Draai eerst update.cmd.
  pause
  exit /b 1
)

.venv-live\Scripts\python.exe -m scripts.gold_intraday_close_momentum --days 730
echo.
pause


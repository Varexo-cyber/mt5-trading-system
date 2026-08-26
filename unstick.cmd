@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   What is holding NEW RISK HALTED?
echo  ============================================
echo.
echo  "SYSTEM_HALTED - broker/journal reconciliation" means the journal
echo  believes a trade is still live at the broker, and the broker neither
echo  holds it nor can produce its closing deal. Trading on a book that
echo  cannot be reconciled is refused, which is correct.
echo.
echo  The runner now writes such a row off by itself after fifteen minutes.
echo  This shows you WHICH row it is, and can close it now instead.
echo.
echo  Reports only. Nothing is changed unless you pass --settle.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

".venv-live\Scripts\python.exe" scripts\unstick.py %*

echo.
pause
endlocal

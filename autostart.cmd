@echo off
setlocal
cd /d "%~dp0"

rem Registering a logon task needs administrator rights. Re-launch elevated
rem once, then continue in the elevated copy.
net session >nul 2>&1
if errorlevel 1 (
  echo Asking for administrator rights...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b 0
)

echo.
echo  ============================================
echo   Start the trading system automatically
echo  ============================================
echo.
echo  This installs three Windows scheduled tasks that run at logon:
echo.
echo    1. MetaTrader 5, minimized
echo    2. Jarvis, 60 seconds later (so MT5 is connected first)
echo    3. The dashboard, 90 seconds later
echo.
echo  Which mode should start automatically?
echo.
echo    [1] paper              - no real orders, safe to leave running
echo    [2] experimental_live  - REAL MONEY on the bound account
echo.
set "MODE="
set /p CHOICE="Type 1 or 2 and press Enter: "
if "%CHOICE%"=="1" set "MODE=paper"
if "%CHOICE%"=="2" set "MODE=experimental_live"
if not defined MODE (
  echo.
  echo  Nothing chosen. No tasks installed.
  echo.
  pause
  exit /b 1
)

if "%MODE%"=="experimental_live" (
  if not exist "runtime\EXPERIMENTAL_LIVE.json" (
    echo.
    echo  No experimental-live contract found. Arm it first:
    echo    .venv-live\Scripts\python.exe scripts\arm_experimental_live.py ^
--account YOURACCOUNT --risk-percent 1 --drawdown-percent 15 ^
--confirm "BEVESTIG EXPERIMENTEEL LIVE"
    echo.
    pause
    exit /b 1
  )
  echo.
  echo  This will trade REAL MONEY every time you log in to Windows.
  set "CONFIRM="
  set /p CONFIRM="Type yes to continue: "
  if /i not "%CONFIRM%"=="yes" (
    echo.
    echo  Cancelled. No tasks installed.
    echo.
    pause
    exit /b 1
  )
)

rem The installer refuses to run if it cannot find the terminal, so locate the
rem real one instead of assuming the default MetaTrader path. Eightcap installs
rem under its own name.
echo.
echo  Looking for the MetaTrader 5 terminal...
set "MT5="
for /f "delims=" %%P in ('powershell -NoProfile -Command ^
  "$p = Get-Process terminal64 -ErrorAction SilentlyContinue ^| Select-Object -First 1 -ExpandProperty Path;" ^
  "if (-not $p) { $p = Get-ChildItem 'C:\Program Files','C:\Program Files (x86)' -Filter terminal64.exe -Recurse -ErrorAction SilentlyContinue ^| Select-Object -First 1 -ExpandProperty FullName };" ^
  "if ($p) { $p }"') do set "MT5=%%P"

if not defined MT5 (
  echo.
  echo  Could not find terminal64.exe. Open MT5 once, then run this again.
  echo.
  pause
  exit /b 1
)
echo  Found: %MT5%

echo.
echo  Installing the scheduled tasks...
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\install_autostart_paper.ps1" -TradingOperation "%MODE%" -Mt5Path "%MT5%"
if errorlevel 1 (
  echo.
  echo  INSTALL FAILED. No changes were made that will start anything.
  echo.
  pause
  exit /b 1
)

echo.
echo  ============================================
echo   Done. Mode: %MODE%
echo  ============================================
echo.
echo  It starts by itself the next time you log in to Windows.
echo  Start it right now without rebooting:
echo    schtasks /run /tn JarvisMetaTrader5
echo.
echo  To stop it starting automatically again:
echo    autostart-off.cmd
echo.
echo  To halt trading immediately at any time, create a file named STOP
echo  in this folder. Jarvis closes its positions and exits.
echo.
pause
endlocal

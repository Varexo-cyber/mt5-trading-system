@echo off
setlocal
cd /d "%~dp0"

rem Registering scheduled tasks needs administrator rights. Re-launch elevated
rem once, then continue in the elevated copy -- the same pattern as
rem install_autostart_monitor.cmd.
net session >nul 2>&1
if errorlevel 1 (
  echo Asking for administrator rights...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b 0
)

echo.
echo  ============================================
echo   Watch the system, and update it safely
echo  ============================================
echo.
echo  Installs two Windows scheduled tasks:
echo.
echo    JarvisSelfCheck   every 15 minutes. Asks each layer whether it is
echo                      alive - runner, journal, calendar, database,
echo                      news feeds, disk - and alerts only on a problem.
echo                      Reads only. Sends no orders, spends no API credit.
echo.
echo    JarvisAutoUpdate  Sunday 08:00. Pulls, checks the config loads and
echo                      the tests pass ON THIS MACHINE, and puts the old
echo                      commit back if either fails.
echo.
echo  The updater REFUSES to run while a position is open, while a STOP file
echo  is present, or with uncommitted local changes. Those guards are not
echo  waived by anything.
echo.
pause

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\install_maintenance_tasks.ps1"
echo.
pause
endlocal

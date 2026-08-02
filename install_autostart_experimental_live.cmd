@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\install_autostart_paper.ps1" -TradingOperation experimental_live
pause

@echo off
setlocal
cd /d "%~dp0"
".venv-live\Scripts\python.exe" scripts\setup_rio_bridge.py
pause

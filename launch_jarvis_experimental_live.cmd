@echo off
setlocal
cd /d "%~dp0"
".venv-live\Scripts\python.exe" jarvis.py --operation experimental_live
pause

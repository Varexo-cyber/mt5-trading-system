@echo off
cd /d "%~dp0"
rem SECTIE 22 -- een gepubliceerd trackrecord toetsen aan zijn eigen beweringen.
rem Heeft GEEN koersdata nodig en draait overal.
set WELKE=%1
if "%WELKE%"=="" set WELKE=alle

set PY=
if exist ".venv-live\Scripts\python.exe" set PY=.venv-live\Scripts\python.exe
if not defined PY if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if not defined PY (where py >nul 2>&1 && set PY=py -3)
if not defined PY (where python >nul 2>&1 && set PY=python)
%PY% -m scripts.section_twentytwo_claim_audit --welke %WELKE%
pause

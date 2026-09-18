@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem ============================================================
rem   SETUP.cmd  --  eenmalig. Zoekt Python en zet de pakketten klaar.
rem ============================================================
rem
rem WAAROM DIT ZO OMSLACHTIG IS. Op deze machine staan zes python.exe'en en
rem geen daarvan is een gewone installatie:
rem
rem   * twee in WindowsApps -- NEP. Dat is de Microsoft Store-snelkoppeling
rem     die alleen een reclameregel print. `where python` vindt hem wel, dus
rem     elke naieve detectie trapt erin.
rem   * drie in .cache\codex-runtimes -- geleend van een andere tool, staan in
rem     een cachemap die opgeruimd kan worden.
rem   * een python-3.11.9-embed -- de "embeddable" uitgave. Die kan GEEN venv
rem     maken; er zit geen pip in.
rem
rem Vandaar: kandidaten verzamelen, ELK ECHT DRAAIEN, en de eerste nemen die
rem werkt. Niet vragen of hij bestaat -- vragen of hij iets doet.

echo.
echo   ============================================================
echo     SETUP  --  Python zoeken en pakketten klaarzetten
echo   ============================================================
echo.

rem --- 1. Staat er al een bruikbare omgeving IN dit project? --------------
set KLAAR=
for %%V in (".venv-live" ".venv") do (
  if not defined KLAAR if exist "%%~V\Scripts\python.exe" (
    "%%~V\Scripts\python.exe" -c "import pandas,numpy,pydantic,yaml" >nul 2>&1
    if !errorlevel! equ 0 set KLAAR=%%~V\Scripts\python.exe
  )
)
if defined KLAAR (
  echo   Er staat al een werkende omgeving: !KLAAR!
  "!KLAAR!" -c "import MetaTrader5" >nul 2>&1
  if errorlevel 1 (
    echo   MetaTrader5 ontbreekt nog, die installeer ik erbij...
    "!KLAAR!" -m pip install MetaTrader5
  )
  echo.
  echo   KLAAR. Draai nu ALLES.cmd
  echo.
  pause
  exit /b 0
)

rem --- 2. Een basis-Python zoeken die een venv KAN maken ------------------
rem
rem De volgorde is bewust: eerst de nette manieren, dan de geleende runtimes
rem van andere tools. Een embeddable uitgave valt vanzelf af omdat `-m venv`
rem daar niet in zit -- daarom wordt dat hier ECHT geprobeerd en niet aangenomen.
set BASIS=
set KANDIDATEN="py -3" "python" "python3"
for %%C in (%KANDIDATEN%) do (
  if not defined BASIS (
    %%~C -c "import venv,ensurepip" >nul 2>&1
    if !errorlevel! equ 0 set BASIS=%%~C
  )
)

if not defined BASIS (
  echo   Geen Python op het PAD. Ik kijk of er ergens anders eentje staat...
  for /f "delims=" %%P in ('where /r "%USERPROFILE%" python.exe 2^>nul') do (
    if not defined BASIS (
      echo %%P | find /i "WindowsApps" >nul
      if !errorlevel! neq 0 (
        echo %%P | find /i "\Lib\venv\" >nul
        if !errorlevel! neq 0 (
          "%%P" -c "import venv,ensurepip" >nul 2>&1
          if !errorlevel! equ 0 (
            set BASIS="%%P"
            echo   Gevonden: %%P
          )
        )
      )
    )
  )
)

if not defined BASIS (
  echo.
  echo   ============================================================
  echo     GEEN BRUIKBARE PYTHON GEVONDEN
  echo   ============================================================
  echo.
  echo   Alles wat op deze computer staat is of NEP ^(WindowsApps^), of
  echo   geleend van een andere tool, of een "embeddable" uitgave die
  echo   geen eigen omgeving kan maken.
  echo.
  echo   Installeer hem een keer goed:
  echo     https://www.python.org/downloads/windows/
  echo     -^> Windows installer ^(64-bit^)
  echo.
  echo   ZET HET VINKJE AAN:  "Add python.exe to PATH"
  echo   Onderaan het eerste scherm. Staat standaard UIT.
  echo.
  echo   Daarna een NIEUW cmd-venster openen en dit opnieuw draaien.
  echo.
  pause
  exit /b 1
)

echo   Basis-Python: %BASIS%
%BASIS% --version
echo.

rem --- 3. Eigen omgeving in dit project ----------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo   Eigen omgeving aanmaken in .venv ...
  %BASIS% -m venv .venv
  if errorlevel 1 (
    echo   Dat is mislukt. Stuur dit scherm door.
    pause
    exit /b 1
  )
)

echo   Pakketten installeren ^(dit duurt een paar minuten^)...
echo.
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pip install MetaTrader5

echo.
echo   Controleren of alles laadt...
.venv\Scripts\python.exe -c "import pandas,numpy,pydantic,yaml; print('  kern: OK')"
.venv\Scripts\python.exe -c "import MetaTrader5; print('  MetaTrader5: OK')"

echo.
echo   ============================================================
echo     KLAAR. Draai nu ALLES.cmd
echo   ============================================================
echo.
pause

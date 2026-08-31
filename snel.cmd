@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ==================================================
echo   Sweep from the stored bars - no MT5 needed
echo  ==================================================
echo.
echo  Same measurement as sweep.cmd, reading data\history instead of the
echo  terminal. Run ophalen.cmd once first to fill it.
echo.
echo  The fetch is gone, so this is pure compute. It also runs with MT5
echo  closed, while the account is trading, or on any machine holding the
echo  folder.
echo.
echo  USAGE
echo    snel.cmd                       180 days, M15 M30 H1 H4
echo    snel.cmd 180 M1 M5             180 days on the fast clocks
echo    snel.cmd 90 M5 M15 M30 H1 H4   90 days, five clocks
echo.

set DAYS=%1
if "%DAYS%"=="" set DAYS=180
shift

set CLOCKS=
:collect
if "%1"=="" goto collected
set CLOCKS=%CLOCKS% %1
shift
goto collect
:collected
if "%CLOCKS%"=="" set CLOCKS=M15 M30 H1 H4

set FINE=--no-m1
for %%C in (%CLOCKS%) do (
  if /I "%%C"=="M1" set FINE=
  if /I "%%C"=="M5" set FINE=
)

if not exist "data\history\manifest.json" (
  echo  ERROR: data\history is empty. Run ophalen.cmd first.
  pause
  exit /b 1
)

if not exist "runtime" mkdir runtime

set TAG=%CLOCKS: =-%
if "%TAG:~0,1%"=="-" set TAG=%TAG:~1%
set CSVFILE=runtime\sweep-%DAYS%d-%TAG%.csv

echo  Window: last %DAYS% days, core markets, clocks %CLOCKS%
echo.

.venv-live\Scripts\python.exe -m scripts.dry_run_sections --days %DAYS% --core %FINE% --sweep %CLOCKS% --cache data\history --equity 216 --csv %CSVFILE%

if errorlevel 1 (
  echo.
  echo  The run failed. The traceback above says why.
)

echo.
echo  Every decision is in %CSVFILE%
echo    verdict.cmd %CSVFILE%
echo.
pause

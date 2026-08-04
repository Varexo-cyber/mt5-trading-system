@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Re-arm EXPERIMENTAL LIVE
echo  ============================================
echo.
echo  The approval is bound to one broker account and one risk figure.
echo  When either changes, the system refuses to start until you say
echo  again that you accept it. That refusal is working as intended.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

echo  Reading the account from the connected terminal...
for /f "delims=" %%A in ('".venv-live\Scripts\python.exe" scripts\show_account.py --login-only 2^>nul') do set LOGIN=%%A

if "%LOGIN%"=="" (
  echo.
  echo  Could not read an account from MT5.
  echo  Open the MetaTrader 5 terminal, log in, and run this again.
  echo.
  pause
  exit /b 1
)

echo.
echo  Connected account: %LOGIN%
echo.
".venv-live\Scripts\python.exe" scripts\show_account.py
echo.
echo  ------------------------------------------------------------
echo   Arming binds real money to this account at the risk figure
echo   shown above. Close this window now if that is not what you
echo   want.
echo  ------------------------------------------------------------
echo.
pause

".venv-live\Scripts\python.exe" scripts\arm_experimental_live.py --account %LOGIN% --confirm "BEVESTIG EXPERIMENTEEL LIVE"
if errorlevel 1 (
  echo.
  echo  ARMING FAILED. Nothing was changed. Send the error above to Claude.
  echo.
  pause
  exit /b 1
)

echo.
echo  ============================================
echo   Armed. You can start experimental live now.
echo  ============================================
echo.
pause
endlocal

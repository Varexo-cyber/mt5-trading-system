@echo off
setlocal
cd /d "%~dp0"

net session >nul 2>&1
if errorlevel 1 (
  echo Asking for administrator rights...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b 0
)

echo.
echo  ============================================
echo   Stop the trading system starting at logon
echo  ============================================
echo.

for %%T in (JarvisMetaTrader5 JarvisTradingPaper JarvisTradingExperimentalLive JarvisDashboard JarvisTradingMonitor) do (
  schtasks /query /tn "%%T" >nul 2>&1 && (
    schtasks /end /tn "%%T" >nul 2>&1
    schtasks /delete /tn "%%T" /f >nul 2>&1 && echo   removed %%T
  )
)

echo.
echo  Done. Nothing starts automatically any more.
echo.
echo  This does NOT stop a session that is running right now. For that,
echo  create a file named STOP in this folder, or close its window.
echo.
pause
endlocal

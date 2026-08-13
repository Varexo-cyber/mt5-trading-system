@echo off
setlocal
cd /d "%~dp0"
where cloudflared >nul 2>nul
if errorlevel 1 (
  echo cloudflared is not installed.
  echo Install it once with:
  echo   winget install --id Cloudflare.cloudflared
  echo Then run this file again.
  pause
  exit /b 1
)
echo Keep this window open. Copy the https://...trycloudflare.com URL into MacroDroid.
echo The temporary URL changes whenever this window is restarted.
cloudflared tunnel --url http://127.0.0.1:8765 --no-autoupdate
pause

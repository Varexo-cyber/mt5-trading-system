@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ==================================================
echo   Push the stored bars so they can be read anywhere
echo  ==================================================
echo.
echo  The bars downloaded by ophalen.cmd sit on this VPS only. This puts them
echo  in the repository, so the analysis can happen on any machine - including
echo  in a Claude session - without MT5 and without you running anything.
echo.
echo  ALL SIX CLOCKS, M1 through H4, over the sixteen core markets. About
echo  145 MB, of which M1 alone is three quarters.
echo.
echo  The first push will take a while - minutes, not seconds, and longer on
echo  a slow VPS line. It happens ONCE. After that only new bars go up.
echo.
echo  If the push is interrupted, run this again. Git resumes.
echo.

if not exist "data\history\manifest.json" (
  echo  ERROR: data\history is empty. Run ophalen.cmd first.
  pause
  exit /b 1
)

git add -A -- data/history .gitignore

git commit -m "history: stored bars for the core markets, all six clocks"
if errorlevel 1 (
  echo.
  echo  Nothing new to commit - the bars in the repository are already current.
)

echo.
echo  Pushing. A first push of this size takes a few minutes.
git push -u origin claude/mt5-autonomous-trading-system-ujd1sk

if errorlevel 1 (
  echo.
  echo  Push failed. Try again - git resumes what it can.
) else (
  echo.
  echo  Done. The bars are in the repository.
)

echo.
pause

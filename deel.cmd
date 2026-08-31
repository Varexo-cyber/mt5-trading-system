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
echo  WHAT GOES: M15, M30, H1, H4 over the sixteen core markets. About 25 MB.
echo  WHAT STAYS: M1 and M5. Three quarters of the bars and roughly 120 MB,
echo  on the clocks the cost wall refuses anyway. ophalen.cmd refetches them
echo  in one command, so the repository does not need to carry them forever.
echo.
echo    deel.cmd        M15 and slower  (about 25 MB)
echo    deel.cmd all    everything incl. M1 and M5  (about 145 MB, slow push)
echo.

if not exist "data\history\manifest.json" (
  echo  ERROR: data\history is empty. Run ophalen.cmd first.
  pause
  exit /b 1
)

git add -A -- data/history .gitignore

if /I "%1"=="all" (
  echo  Including M1 and M5 - this push will take a while.
  git add -f -- data/history
)

git commit -m "history: stored bars for the core markets"
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

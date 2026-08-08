@echo off
setlocal

REM ---------------------------------------------------------------------------
REM  Run from a copy, because this file updates itself.
REM
REM  cmd.exe reads a batch file one line at a time and remembers its position as
REM  a BYTE OFFSET into the file on disk. It re-opens the file after every
REM  command. So when the pull below rewrites this very script, cmd resumes at
REM  an offset that now points into the middle of a different line.
REM
REM  That is not theoretical. Adding three comment lines here produced:
REM
REM      'o.py' is not recognized as an internal or external command
REM
REM  which is the tail of `scripts\update_repo.py` -- the pull had succeeded and
REM  the script then fell apart mid-sentence, printing "UPDATE FAILED. Nothing
REM  was changed." on an update that had in fact been applied. Every future
REM  edit to this file would do it again.
REM
REM  Copying to %TEMP% first and running that means git can rewrite the original
REM  freely: the copy cmd is reading is nobody's business but ours. The original
REM  directory is passed through, since %~dp0 in the copy points at %TEMP%.
REM ---------------------------------------------------------------------------
REM  Two Windows traps are avoided deliberately below.
REM
REM  `exit /b` carries no argument. Inside a parenthesised block %errorlevel%
REM  is expanded when the block is PARSED, not when it runs, so `exit /b
REM  %errorlevel%` would return the code from before the call. Bare `exit /b`
REM  passes the live one through.
REM
REM  The directory is handed over as "%~dp0." with a trailing dot. %~dp0 always
REM  ends in a backslash, and "C:\mt5-trading-system\" puts \" at the end of an
REM  argument, where cmd reads it as an escaped quote and swallows the closing
REM  one. The dot makes it a normal path that happens to mean the same folder.
if /i not "%~1"=="--from-copy" (
  copy /y "%~f0" "%TEMP%\jarvis_update_run.cmd" >nul 2>&1
  if exist "%TEMP%\jarvis_update_run.cmd" (
    call "%TEMP%\jarvis_update_run.cmd" --from-copy "%~dp0."
    exit /b
  )
  REM No writable %TEMP%. Carry on in place: the update still works, and the
  REM only cost is that a release which edits this file needs a second run.
  echo  [note] could not stage a copy in %%TEMP%%. If this run ends with an odd
  echo         "not recognized" error, just run update.cmd again.
  cd /d "%~dp0"
  goto :body
)
cd /d "%~2"

:body
echo.
echo  ============================================
echo   Updating the trading system from GitHub
echo  ============================================
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo.
  echo  Missing .venv-live. Create it once with:
  echo    py -m venv .venv-live
  echo    .venv-live\Scripts\python.exe -m pip install -e ".[live,dashboard,brain]"
  echo.
  pause
  exit /b 1
)

REM Not a bare "git pull". That reports "Already up to date." whenever the
REM current branch's upstream has nothing new - which is true, and badly
REM misleading, when the fetch it just ran did update a different branch. It
REM printed exactly that under a line showing new commits arriving, so an
REM update that changed nothing announced success and everything afterwards
REM looked broken for the wrong reason.
".venv-live\Scripts\python.exe" scripts\update_repo.py
if errorlevel 1 (
  echo.
  echo  UPDATE FAILED. Nothing was changed. The reason is above.
  echo.
  pause
  exit /b 1
)

echo.
echo  Installing any new dependencies...
REM `brain` is in here so the Postgres driver arrives with every update. Left
REM out, `build_brain` quietly returns the null object and the long-term memory
REM stays off after each pull - with nothing failing and nothing saying so.
".venv-live\Scripts\python.exe" -m pip install -q -e ".[live,dashboard,brain]"

echo.
echo  Checking the configuration loads...
".venv-live\Scripts\python.exe" main.py --overlay config\eightcap.yaml --check-config
if errorlevel 1 (
  echo.
  echo  CONFIG IS BROKEN. Do not start trading. Send the error above to Claude.
  echo.
  pause
  exit /b 1
)

echo.
echo  --------------------------------------------
echo   Why it has or has not been trading
echo  --------------------------------------------
".venv-live\Scripts\python.exe" scripts\why_no_trades.py --hours 12

echo.
echo  --------------------------------------------
echo   Account and experimental-live approval
echo  --------------------------------------------
".venv-live\Scripts\python.exe" scripts\show_account.py
if errorlevel 1 (
  echo  MT5 is not reachable. Fine if the terminal is closed - open it
  echo  and run this again before starting live trading.
)

echo.
echo  ============================================
echo   Update complete.
echo  ============================================
echo.
echo  If the section above says re-arming is required, run:
echo    rearm_experimental_live.cmd
echo.
pause
endlocal

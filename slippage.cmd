@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ==================================================
echo   What does a stop-out ACTUALLY cost?
echo  ==================================================
echo.
echo  This is the number that decides whether FX can trade at all.
echo.
echo  The cost of a round trip on EURUSD, per lot, as the config prices it:
echo.
echo      commission   $ 5.50    round trip
echo      slippage     $17.00    ^<- stop_slippage_pips.forex = 1.7
echo      spread       $ 2.00
echo                   -------
echo                   $24.50
echo.
echo  Slippage is two thirds of it, and it is NOT a measurement. It came from
echo  one observed fill - a stop at 1.19722 filled at 1.19705 - and is charged
echo  on every trade on every pair.
echo.
echo  Against the 12%% cost limit:
echo.
echo      clock   stop      at 1.7 pip   at 0.3 pip
echo      M15     3.5 pip       70%%          30%%
echo      M30     5   pip       49%%          21%%
echo      H1      8   pip       31%%          13%%
echo      H4      20  pip       12%%           5%%
echo.
echo  So one unchecked number is the difference between "FX never clears the
echo  wall" and "FX clears it on H1". This reads your own closed trades and
echo  says what it really is.
echo.
echo  MT5 does not need to be running - it reads the journal.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  ERROR: .venv-live\Scripts\python.exe not found. Run update.cmd first.
  pause
  exit /b 1
)

.venv-live\Scripts\python.exe -m scripts.measure_slippage %*

echo.
pause

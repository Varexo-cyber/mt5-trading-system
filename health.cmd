@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Is alles gezond?
echo  ============================================
echo.
echo  `scripts/selfcheck.py` beantwoordt dit al sinds de dag dat het geschreven
echo  is en had GEEN LAUNCHER, net als whynot.cmd. Twee van de drie vragen die
echo  een operator werkelijk stelt waren de twee die hij niet kon stellen.
echo.
echo  DE DURE STORING IS NIET EEN CRASH. Een crash is luid: het venster sluit,
echo  het dashboard wordt leeg, iemand merkt het binnen het uur. De dure is de
echo  stille - de runner draait, het log scrollt, en een laag eronder is dood.
echo  Een kalender die niet meer ververst, een MT5 die zichzelf heeft uitgelogd,
echo  een detector die zichzelf heeft uitgezet. Van buiten ziet dat er allemaal
echo  precies zo uit als een rustige markt.
echo.
echo  Elke regel is een laag die rechtstreeks is gevraagd:
echo.
echo    kill switch   staat er een STOP-bestand
echo    runner        wanneer maakte hij zijn laatste cyclus af
echo    journal       leesbaar, en hoeveel beslissingen in 24 uur
echo    trading       en hoeveel daarvan werden TRADES
echo    breakers      heeft een detector zichzelf uitgezet
echo    calendar      is de nieuwskalender vers genoeg om te mogen handelen
echo    brain         geheugen bereikbaar
echo    headlines     nieuwslaag
echo    disk          ruimte
echo.
echo  `trading` en `breakers` zijn nieuw op 28 augustus, en ze bestaan om twee
echo  gaten te dichten die deze week zijn opgevallen:
echo.
echo   - Een systeem kan up, druk en gezond zijn en NIET HANDELEN. 66.140
echo     beslissingen in twaalf uur leverden twee trades op, en elke laag
echo     rapporteerde die hele tijd ok. `trading` vraagt naar het resultaat en
echo     niet naar de drukte.
echo.
echo   - Een detector die zichzelf uitzet na acht verliezers is precies waarom
echo     het aanvaardbaar is om modules te draaien met een dun of ongemeten
echo     verleden. En dat er een is afgegaan zag er van buiten uit als een
echo     stille markt. Nu staat het er.
echo.
echo  Journaal read-only. Geen orders, geen API, er wordt alleen een eigen
echo  statusbestand geschreven.
echo.
echo  health.cmd            het rapport
echo  health.cmd --quiet    alleen printen als er iets mis is
echo.
echo  De afsluitcode is 0 als alles ok is en 1 zodra er iets WARN of FAIL is,
echo  dus dit werkt ook als geplande taak: Windows onthoudt die code en een
echo  taak die twee dagen 1 teruggeeft is zichtbaar zonder dat iemand een log
echo  hoeft te lezen.
echo.

if not exist ".venv-live\Scripts\python.exe" (
  echo  Missing .venv-live. Run update.cmd first.
  echo.
  pause
  exit /b 1
)

".venv-live\Scripts\python.exe" scripts\selfcheck.py %*

echo.
pause
endlocal

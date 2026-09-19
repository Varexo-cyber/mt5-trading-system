"""Een meting mag niet afslaan op inloggegevens die ze niet nodig heeft.

DIT IS DEZE WEEK DRIE KEER GEBEURD en het kostte elke keer een avond.

MetaTrader 5 staat open en is ingelogd. `MT5Connector._initialise` zet login,
password en server alleen in `initialize()` wanneer er inloggegevens ZIJN:

    if self.credentials is not None:
        kwargs.update(login=..., password=..., server=...)

Zonder die gegevens haakt hij dus aan bij de terminal die al draait. Precies
wat een meting nodig heeft. `load_credentials(required=True)` gooit er toch een
`ConfigError` overheen:

    ConfigError: missing MT5 credential(s): MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

Dat is een drempel zonder reden, en hij zit in het pad van een script dat verder
prima werkt. Deze test houdt de scripts die door de LAUNCHERS worden aangeroepen
daarvan vrij.

WAAROM ER EEN LIJST MET UITZONDERINGEN IS. Er staan nog meer scripts met
`required=True`, en die zijn niet allemaal fout -- sommige zijn eenmalige
hulpjes die nooit in een meetketen zitten. Ze staan hieronder met naam, zodat
het een KEUZE is en geen vergeten hoekje. Verdwijnt er eentje uit een launcher,
dan hoort hij van deze lijst af.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: De scripts die door ALLES.cmd en uurmeting.cmd worden gedraaid. Op een
#: machine met een draaiende terminal moeten die alle zeven zonder .env starten.
IN_DE_MEETKETEN = (
    "dry_run_sections.py",
    "per_uur_en_sessie.py",
    "exporteer_bars.py",
    "eindresultaat.py",
    "section_twenty_pullback_ladder.py",
    "section_twentyone_straddle.py",
    "section_twentythree_uitstap.py",
    "zoektocht.py",
    "dagboek.py",
    "welke_balans.py",
    "uitvoering.py",
)

EIST_INLOG = re.compile(r"load_credentials\(\s*required\s*=\s*True")


class TestGeenInlogDrempelInDeMeetketen:
    @pytest.mark.parametrize("naam", IN_DE_MEETKETEN)
    def test_dit_script_start_zonder_env(self, naam: str):
        pad = ROOT / "scripts" / naam
        if not pad.exists():
            pytest.skip(f"{naam} bestaat niet (meer)")
        tekst = pad.read_text(encoding="utf-8")

        # Alleen ECHTE aanroepen tellen; de toelichting erboven noemt de oude
        # regel met opzet, en die mag blijven staan.
        zonder_uitleg = "\n".join(
            regel for regel in tekst.splitlines()
            if not regel.lstrip().startswith("#")
        )
        assert not EIST_INLOG.search(zonder_uitleg), (
            f"{naam} eist MT5_LOGIN/MT5_PASSWORD/MT5_SERVER terwijl de "
            f"terminal al draait. Gebruik load_credentials(required=False); "
            f"MT5Connector haakt dan aan bij de openstaande terminal."
        )


class TestDeLijstZelfKlopt:
    """Zonder dit zou een lege of verkeerd gespelde lijst ook groen zijn."""

    def test_de_genoemde_scripts_bestaan_echt(self):
        ontbreekt = [n for n in IN_DE_MEETKETEN
                     if not (ROOT / "scripts" / n).exists()]
        assert not ontbreekt, f"deze staan in de lijst maar niet op schijf: {ontbreekt}"

    def test_de_regex_herkent_de_drempel_wel(self):
        assert EIST_INLOG.search("credentials = load_credentials(required=True)")
        assert EIST_INLOG.search("    load_credentials( required = True ),")
        assert not EIST_INLOG.search("load_credentials(required=False)")

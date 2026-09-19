"""De uur- en sessie-uitsplitsing, en vooral de rem erop.

WAT HIER BEWAAKT WORDT. Niet of een uur goed is -- dat moet de meting zeggen.
Wel dat de uitsplitsing niet liegt op de twee manieren waarop dit soort
uitsplitsingen altijd liegt:

  * WEIGERINGEN TELLEN MEE ALS NUL. De dry-run-CSV bevat elke beslissing, ook
    de honderden die geweigerd zijn. Die hebben geen uitkomst; ze als 0,00 R
    meenemen verdunt elk gemiddelde naar nul.
  * HET BESTE UUR IS HET MAXIMUM VAN VIERENTWINTIG RUIZIGE GETALLEN. Op pure
    ruis hoort de permutatietest een hoge p te geven; op een echt signaal een
    lage. Doet hij dat niet, dan is hij decoratie.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta

import pytest

from scripts.per_uur_en_sessie import lees, permutatie, rapport, sessie_van

KOP = ["when", "symbol", "managed_r_LIVE", "result_r_fixed_stop"]


def _csv(tmp_path, rijen):
    pad = tmp_path / "beslissingen.csv"
    with pad.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(KOP)
        w.writerows(rijen)
    return pad


class TestDeSessie:
    def test_de_overlap_krijgt_een_eigen_naam(self):
        # Londen 7-16, New York 12-21: 12 tot 16 is allebei.
        assert sessie_van(13) == "overlap"
        assert sessie_van(9) == "londen"
        assert sessie_van(18) == "newyork"

    def test_asia_loopt_over_middernacht(self):
        assert sessie_van(23) == "asia"
        assert sessie_van(2) == "asia"
        assert sessie_van(6) == "asia"


class TestWatErGETELDWordt:
    def test_weigeringen_zonder_uitkomst_tellen_niet_mee(self, tmp_path):
        """DE BELANGRIJKSTE HIER.

        De CSV bevat elke beslissing, en de meeste zijn weigeringen. Die hebben
        een lege uitkomstkolom. Ze als 0,00 R meetellen zou elk gemiddelde naar
        nul verdunnen en precies verbergen wat je wil zien.
        """
        nu = datetime(2026, 1, 5, 9, tzinfo=UTC)
        rijen = [
            [nu.isoformat(), "XAUUSD", "1.5", "1.5"],
            [nu.isoformat(), "XAUUSD", "", ""],        # geweigerd
            [nu.isoformat(), "XAUUSD", "-1.0", "-1.0"],
        ]
        gelezen = lees(_csv(tmp_path, rijen), "managed_r_LIVE")
        assert len(gelezen) == 2, "de weigering is als trade meegeteld"
        assert sum(w for _, w in gelezen) == pytest.approx(0.5)

    def test_de_twee_kolommen_worden_apart_gelezen(self, tmp_path):
        nu = datetime(2026, 1, 5, 9, tzinfo=UTC)
        pad = _csv(tmp_path, [[nu.isoformat(), "XAUUSD", "2.0", "5.0"]])
        assert lees(pad, "managed_r_LIVE")[0][1] == 2.0
        assert lees(pad, "result_r_fixed_stop")[0][1] == 5.0

    def test_onleesbare_rijen_laten_de_rest_staan(self, tmp_path):
        nu = datetime(2026, 1, 5, 9, tzinfo=UTC)
        rijen = [[nu.isoformat(), "XAUUSD", "1.0", "1.0"],
                 ["geen datum", "XAUUSD", "1.0", "1.0"],
                 [nu.isoformat(), "XAUUSD", "niet-een-getal", ""]]
        assert len(lees(_csv(tmp_path, rijen), "managed_r_LIVE")) == 1


class TestDePermutatieRem:
    """Zonder deze rem is elke uur-uitsplitsing een vondst."""

    @staticmethod
    def _rijen(waarden_per_uur):
        start = datetime(2026, 1, 5, tzinfo=UTC)
        uit = []
        for uur, waarden in waarden_per_uur.items():
            for i, w in enumerate(waarden):
                uit.append((start + timedelta(days=i, hours=uur), w))
        return uit

    def test_op_pure_ruis_is_het_beste_uur_niet_bijzonder(self):
        import numpy as np

        rng = np.random.default_rng(7)
        rijen = self._rijen({u: list(rng.normal(0, 1, 40)) for u in range(24)})
        _, p, hokjes = permutatie(rijen, lambda m: m.hour, rondes=300)

        assert hokjes == 24
        assert p > 0.05, (
            "op ruis wordt het beste van 24 uren als een vondst gerapporteerd"
        )

    def test_een_echt_signaal_wordt_wel_opgemerkt(self):
        """Anders zou 'geef altijd p = 1' ook slagen."""

        import numpy as np

        rng = np.random.default_rng(7)
        per_uur = {u: list(rng.normal(0, 1, 40)) for u in range(24)}
        per_uur[14] = list(rng.normal(3.0, 1, 40))     # dit uur is echt anders
        _, p, _ = permutatie(self._rijen(per_uur), lambda m: m.hour, rondes=300)

        assert p < 0.05, "een verschil van drie sigma wordt niet opgemerkt"


class TestHetRapport:
    def test_een_lege_csv_geeft_uitleg_en_geen_crash(self, tmp_path):
        pad = _csv(tmp_path, [])
        regels = rapport(pad, kolom="managed_r_LIVE", minimum=30, rondes=10)
        assert any("Geen enkele rij" in r for r in regels)

    def test_dunne_hokjes_worden_als_dun_gemarkeerd(self, tmp_path):
        nu = datetime(2026, 1, 5, 9, tzinfo=UTC)
        rijen = [[(nu + timedelta(days=i)).isoformat(), "XAUUSD", "1.0", "1.0"]
                 for i in range(3)]
        regels = rapport(_csv(tmp_path, rijen), kolom="managed_r_LIVE",
                         minimum=30, rondes=10)
        assert any("te weinig trades" in r for r in regels), (
            "drie trades in een uur worden als uitslag gepresenteerd"
        )

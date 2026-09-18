"""Sectie 23: de uitstapmeting mag zichzelf niet flatteren.

VIJF KEUZES DIE BEPALEN OF DE UITSLAG IETS WAARD IS:

  * R IS DE EERSTE STOP, altijd. Meet je in de VERSCHOVEN stop, dan verandert
    de meeteenheid mee met de variant en vergelijk je appels met peren.
  * DE STOP WINT DE BAR van het doel en van de breakeven-verschuiving.
  * BREAKEVEN WORDT PAS NA DE STOPCONTROLE GEZET, anders krijgt een trade die
    binnen een bar omhoog en daarna omlaag ging gratis een bescherming die hij
    live niet had.
  * ELKE VARIANT KRIJGT DEZELFDE INGANGEN, anders meet je de ingang.
  * HET MINIMUMLOT IS EEN VLOER in de balansladder -- onder 0,01 bestaat niet,
    dus op een kleine rekening bepaalt de broker het risico en niet de regel.
"""

from __future__ import annotations

from datetime import UTC

import pandas as pd
import pytest

from scripts.section_twentythree_uitstap import (
    Trade, Uitstap, balansladder, hoeveel_verliezen_dodelijk, loop_trade, varianten,
)


def _f(bars, start="2026-02-02 09:00"):
    idx = pd.date_range(start, periods=len(bars), freq="1min", tz=UTC)
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c} for o, h, l, c in bars], index=idx)


class TestRIsAltijdDeEersteStop:
    def test_een_breakeven_stop_telt_als_nul_en_niet_als_min_een(self):
        """Als R op de VERSCHOVEN stop gemeten werd, zou deze trade -1R heten
        terwijl er niets verloren is."""

        m1 = _f([(100.0, 100.0, 100.0, 100.0),
                 (100.0, 101.0, 100.0, 101.0),    # +1R, breakeven gaat aan
                 (101.0, 101.0, 99.0, 99.0)])     # terug naar instap
        t = loop_trade(m1, 0, entry=100.0, stop=99.0, atr=1.0,
                       uitstap=Uitstap("x", be_bij=0.5), kosten_r=0.0, max_bars=10)

        assert t.r == pytest.approx(0.0), "de breakeven-uitstap wordt niet als 0R geteld"
        assert t.reden == "stop verschoven"

    def test_zonder_breakeven_is_dezelfde_reeks_wel_min_een(self):
        """Anders zegt de test hierboven niets over breakeven."""

        m1 = _f([(100.0, 100.0, 100.0, 100.0),
                 (100.0, 101.0, 100.0, 101.0),
                 (101.0, 101.0, 99.0, 99.0)])
        t = loop_trade(m1, 0, entry=100.0, stop=99.0, atr=1.0,
                       uitstap=Uitstap("x"), kosten_r=0.0, max_bars=10)

        assert t.r == pytest.approx(-1.0)
        assert t.reden == "stop"


class TestDeStopWintDeBar:
    def test_een_bar_die_stop_en_doel_raakt_telt_als_stop(self):
        m1 = _f([(100.0, 103.0, 99.0, 100.0)])
        t = loop_trade(m1, 0, entry=100.0, stop=99.0, atr=1.0,
                       uitstap=Uitstap("x", doel_r=2.0), kosten_r=0.0, max_bars=10)

        assert t.r == pytest.approx(-1.0), "het doel is gepakt in een bar die ook de stop raakte"

    def test_breakeven_wordt_pas_na_de_stopcontrole_gezet(self):
        """Een bar die eerst omhoog gaat en daarna door de stop zakt, mag geen
        gratis bescherming krijgen -- binnen die bar weet je de volgorde niet."""

        m1 = _f([(100.0, 100.6, 98.9, 99.0)])    # raakt +0,6R EN de stop
        t = loop_trade(m1, 0, entry=100.0, stop=99.0, atr=1.0,
                       uitstap=Uitstap("x", be_bij=0.5), kosten_r=0.0, max_bars=10)

        assert t.r == pytest.approx(-1.0), (
            "de breakeven is binnen dezelfde bar al gezet; dat is vooruitkijken"
        )


class TestDeVariantenZijnEcht:
    def test_er_zijn_er_minstens_twintig_en_allemaal_uniek(self):
        v = varianten()
        assert len(v) >= 20
        assert len({x.naam for x in v}) == len(v)

    def test_de_eerste_is_de_nulhypothese_zonder_beheer(self):
        """Alles wordt hiertegen afgezet; heeft hij zelf beheer, dan is er geen
        ijkpunt meer."""

        nul = varianten()[0]
        assert nul.be_bij is None and nul.trail_atr is None
        assert nul.deel_bij is None and nul.tijd_bars is None

    def test_trailen_doet_echt_iets(self):
        """Een variant die niets verandert, hoort niet in de lijst."""

        m1 = _f([(100.0, 100.0, 100.0, 100.0)]
                + [(100.0 + i, 101.0 + i, 99.5 + i, 100.5 + i) for i in range(8)]
                + [(108.0, 108.0, 100.0, 100.0)])
        zonder = loop_trade(m1, 0, entry=100.0, stop=99.0, atr=1.0,
                            uitstap=Uitstap("x", doel_r=99.0), kosten_r=0.0, max_bars=20)
        met = loop_trade(m1, 0, entry=100.0, stop=99.0, atr=1.0,
                         uitstap=Uitstap("y", trail_atr=1.0, doel_r=0.0),
                         kosten_r=0.0, max_bars=20)

        assert met.r > zonder.r, "de trailing stop pakt de beweging niet"


class TestHetMinimumlotIsEenVloer:
    def _trades(self, n=200, raak=0.6):
        idx = pd.date_range("2026-01-01", periods=n, freq="h", tz=UTC)
        return [
            Trade(s, 1.0 if i % 10 < raak * 10 else -1.0, 5, 1.0, -1.0, "x")
            for i, s in enumerate(idx)
        ]

    def test_een_kleine_rekening_kan_niet_binnen_twee_procent_blijven(self):
        """DE KERN VAN HET BALANSVERHAAL. Onder 0,01 lot bestaat niet, dus op
        EUR 59 bepaalt de broker je risico."""

        lad = balansladder(self._trades(), stop_punten=10.2)
        klein = lad[lad["start"] < 250]
        groot = lad[lad["start"] >= 467]

        assert not klein["haalbaar"].any(), "een kleine rekening heet ten onrechte haalbaar"
        assert groot["haalbaar"].all()
        assert klein["risico_eerste_trade"].max() > 0.10

    def test_het_grootste_veelvoud_zit_bij_de_kleinste_rekening(self):
        """Contra-intuitief en juist daarom belangrijk: de kleine rekening
        vermenigvuldigt het hardst omdat ze GEDWONGEN wordt te veel te
        riskeren. Dat hoort geen verkoopargument te worden."""

        lad = balansladder(self._trades(), stop_punten=10.2)
        assert lad.iloc[0]["keer"] > lad.iloc[-1]["keer"]

    def test_en_daar_hoort_het_dodelijke_aantal_bij(self):
        d = hoeveel_verliezen_dodelijk(self._trades(), 59.16, stop_punten=10.2)
        groot = hoeveel_verliezen_dodelijk(self._trades(), 5000.0, stop_punten=10.2)

        assert d["dodelijk_na"] < 10, "op EUR 59 is een korte pechreeks al fataal"
        assert groot["dodelijk_na"] > 100
        assert d["kans_op_die_reeks"] > groot["kans_op_die_reeks"] * 1000


class TestDeVloerIsEchtEenVloer:
    """DEZE TEST ONTBRAK EN DAT BLEEK UIT EEN MUTATIE.

    Ik haalde `max(lotstap, ...)` uit de ladder -- dus lotgroottes kleiner dan
    0,01 toegestaan -- en de hele suite bleef groen. Terwijl die vloer het hele
    punt is: op EUR 59 kun je NIET binnen 2% blijven, want kleiner dan een
    minimumlot bestaat niet.

    Mijn bestaande tests keken naar `haalbaar`, en dat veld wordt apart
    uitgerekend uit de stopafstand. Ze raakten de lotberekening nooit.
    """

    def _rechte_reeks(self, n=40):
        """Alleen verliezers, zodat het bedrag per trade exact af te lezen is."""

        idx = pd.date_range("2026-01-01", periods=n, freq="h", tz=UTC)
        return [Trade(s, -1.0, 5, 0.0, -1.0, "stop") for s in idx]

    def test_op_een_kleine_rekening_kost_een_verliezer_het_minimumlot(self):
        """Niet 2% van de balans -- het volle bedrag van een 0,01 lot. Dat is
        wat een vloer betekent en het is het verschil tussen wel en niet
        kunnen handelen."""

        stop_punten, euro_per_punt = 10.2, 0.87
        vol_verlies = stop_punten * euro_per_punt          # EUR 8.87 bij 0,01 lot

        lad = balansladder(self._rechte_reeks(), stop_punten=stop_punten,
                           euro_per_punt_min=euro_per_punt)
        klein = lad[lad["start"] == 59.16].iloc[0]

        # 59,16 / 8,87 = 6,7 -> na 7 verliezers is hij op.
        assert klein["ruine"], "de kleine rekening overleeft zeven volle verliezers"
        assert klein["trades"] == 7, (
            f"hij deed {klein['trades']} trades voor de ruine; bij een echte "
            "vloer zijn dat er zeven"
        )
        # En 2% van 59,16 is EUR 1,18 -- vijf keer minder dan wat het kost.
        assert vol_verlies > 0.02 * 59.16 * 5

    def test_op_een_grote_rekening_geldt_de_twee_procent_wel(self):
        """Anders zou de vloer overal gelden en zou de ladder nergens over gaan."""

        lad = balansladder(self._rechte_reeks(n=5), stop_punten=10.2)
        groot = lad[lad["start"] == 10_000.0].iloc[0]

        verloren = 10_000.0 - groot["eind"]
        # Vijf verliezers van ongeveer 2%, samengesteld omlaag.
        assert 0.07 < verloren / 10_000.0 < 0.11, (
            f"een grote rekening verloor {verloren / 10_000.0:.1%} over vijf "
            "verliezers; bij 2% per trade hoort dat rond de 9,6% te zijn"
        )

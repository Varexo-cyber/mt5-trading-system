"""Sectie 21: de mechaniek van de straddle, zonder MT5.

WAT HIER BEWAAKT WORDT. Niet of de straddle goed is -- dat moet de meting
uitwijzen. Deze tests bewaken de vijf keuzes waarmee dit mechanisme er
stilletjes beter uit gaat zien dan het is:

  * VIER SPREADS PER CYCLUS, niet twee. Twee benen, elk in en uit. Dat is de
    prijs van het mechanisme en de enige reden dat het kan verliezen terwijl
    het "altijd een winnaar heeft".
  * DE WHIPSAW BESTAAT. Raakt een bar allebei de drempels, dan ben je allebei
    de benen kwijt. Dat wegdefinieren maakt de hele meting waardeloos.
  * DE STOP WINT DE BAR. Raakt een bar zowel stop als doel, dan wint de stop --
    wat er binnen die bar eerst gebeurde weet niemand.
  * DE CONTROLE STAAT OP DEZELFDE MOMENTEN. Anders vergelijk je twee
    steekproeven en niet twee regels.
  * HET BESTE UUR WORDT GETOETST. Het beste van 24 uren is het maximum van 24
    ruizige getallen.
"""

from __future__ import annotations

from datetime import UTC

import pandas as pd
import pytest

from scripts.section_twenty_pullback_ladder import CONTRACT
from scripts.section_twentyone_straddle import (
    Cyclus,
    Instelling,
    draai,
    een_richting_cyclus,
    rooster,
    straddle_cyclus,
    uur_permutatie,
)


def _frame(prijzen, start="2026-01-05 09:00"):
    index = pd.date_range(start, periods=len(prijzen), freq="1min", tz=UTC)
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c} for o, h, l, c in prijzen],
        index=index,
    )


class TestDeWhipsawBestaat:
    def test_een_bar_die_allebei_de_drempels_raakt_kost_allebei_de_benen(self):
        """DE DUURSTE UITKOMST VAN DIT MECHANISME, en de verleiding is groot om
        hem weg te definieren. Binnen een bar is de volgorde onbekend; de
        pessimistische lezing is dat je ze allebei kwijt bent."""

        m1 = _frame([(100.0, 110.0, 90.0, 100.0)] + [(100.0, 100.0, 100.0, 100.0)] * 3)
        cyclus = straddle_cyclus(m1, 0, instelling=Instelling(kap=5.0, spread=0.0, lot=0.01))

        assert cyclus.gehouden == 0, "er is een been blijven staan na een dubbele stop"
        verwacht = -2 * 5.0 * 0.01 * CONTRACT
        assert cyclus.netto_euro == pytest.approx(verwacht), (
            "een dubbele stop kost twee keer de kap, niet een keer"
        )

    def test_een_gewone_beweging_laat_wel_een_been_staan(self):
        """Als deze test niet bestond, zou elke cyclus een whipsaw kunnen heten
        en nog steeds groen zijn."""

        m1 = _frame([
            (100.0, 101.0, 100.0, 101.0),
            (101.0, 106.0, 101.0, 106.0),    # short been -6 -> afgekapt
            (106.0, 121.0, 106.0, 121.0),    # long loopt door naar het doel
        ])
        cyclus = straddle_cyclus(
            m1, 0, instelling=Instelling(kap=5.0, doel=20.0, spread=0.0, lot=0.01)
        )

        assert cyclus.gehouden == 1, "het LONG been hoort te blijven staan"


class TestVierSpreadsPerCyclus:
    def test_twee_benen_betalen_elk_twee_keer_de_spread(self):
        """DE PRIJS VAN HET MECHANISME. Bij een gewone trade gaan er twee halve
        spreads af; hier vier. Dat is precies het verschil dat maakt dat een
        straddle kan verliezen terwijl hij altijd een winnaar heeft."""

        bars = [(100.0, 101.0, 100.0, 101.0), (101.0, 106.0, 101.0, 106.0),
                (106.0, 121.0, 106.0, 121.0)]
        m1 = _frame(bars)

        zonder = straddle_cyclus(
            m1, 0, instelling=Instelling(kap=5.0, doel=20.0, spread=0.0, lot=0.01)
        )
        met = straddle_cyclus(
            m1, 0, instelling=Instelling(kap=5.0, doel=20.0, spread=0.16, lot=0.01)
        )

        verschil = zonder.netto_euro - met.netto_euro
        verwacht = 0.16 * 2 * 0.01 * CONTRACT * 2       # twee benen
        assert verschil == pytest.approx(verwacht), (
            "er wordt maar voor een been spread gerekend"
        )

    def test_de_controle_betaalt_maar_de_helft(self):
        """Het hele punt van de vergelijking: een richting kost een been minder.
        Rekende de controle net zoveel kosten, dan zou de straddle er gratis
        uitzien."""

        bars = [(100.0, 101.0, 100.0, 101.0)] + [(101.0, 121.0, 101.0, 121.0)] * 2
        m1 = _frame(bars)
        instelling = Instelling(kap=5.0, doel=20.0, spread=0.16, lot=0.01)

        zonder = een_richting_cyclus(
            m1, 0, instelling=Instelling(kap=5.0, doel=20.0, spread=0.0, lot=0.01),
            richting=1,
        )
        met = een_richting_cyclus(m1, 0, instelling=instelling, richting=1)

        verschil = zonder.netto_euro - met.netto_euro
        assert verschil == pytest.approx(0.16 * 2 * 0.01 * CONTRACT), (
            "de controle rekent niet met precies een been"
        )


class TestDeStopWintDeBar:
    def test_een_bar_met_stop_en_doel_telt_als_stop(self):
        """Wat er binnen die bar eerst gebeurde weet niemand, en de andere
        aanname flatteert elke uitslag."""

        m1 = _frame([
            (100.0, 101.0, 100.0, 101.0),
            (101.0, 106.0, 101.0, 106.0),        # short eruit bij -5
            (106.0, 130.0, 80.0, 100.0),         # doel EN stop in een bar
        ])
        cyclus = straddle_cyclus(
            m1, 0,
            instelling=Instelling(kap=5.0, doel=20.0, winnaar_stop=10.0,
                                  spread=0.0, lot=0.01),
        )

        # -10 op het overgebleven been, -5 op het afgekapte been.
        verwacht = (-10.0 - 5.0) * 0.01 * CONTRACT
        assert cyclus.netto_euro == pytest.approx(verwacht), (
            "het doel is gepakt in een bar die ook de stop raakte"
        )


class TestDeControleStaatOpDezelfdeMomenten:
    def test_alle_drie_de_reeksen_openen_op_dezelfde_bars(self):
        """Anders vergelijk je twee steekproeven in plaats van twee regels, en
        dan zegt "de straddle wint" alleen iets over welke momenten hij kreeg."""

        import numpy as np

        rng = np.random.default_rng(3)
        prijzen = 100 + np.cumsum(rng.normal(0, 0.5, 400))
        m1 = _frame([(p, p + 1, p - 1, p) for p in prijzen])

        uit = draai(m1, instelling=Instelling(kap=5.0, doel=20.0), om_de=60)

        momenten = {soort: [c.geopend for c in cycli] for soort, cycli in uit.items()}
        assert momenten["straddle"] == momenten["een_richting"] == momenten["altijd_long"]
        assert len(momenten["straddle"]) > 3, "er zijn te weinig cycli om iets te zeggen"

    def test_de_drie_reeksen_zijn_echt_verschillend(self):
        """Als `draai` drie keer hetzelfde teruggaf, zou de test hierboven nog
        steeds slagen en zou de vergelijking niets vergelijken."""

        import numpy as np

        rng = np.random.default_rng(5)
        prijzen = 100 + np.cumsum(rng.normal(0, 0.5, 400))
        m1 = _frame([(p, p + 1, p - 1, p) for p in prijzen])

        uit = draai(m1, instelling=Instelling(kap=5.0, doel=20.0), om_de=60)
        totalen = {s: sum(c.netto_euro for c in cycli) for s, cycli in uit.items()}

        assert len(set(round(v, 6) for v in totalen.values())) == 3, (
            f"twee reeksen zijn identiek: {totalen}"
        )


class TestHetBesteUurWordtGetoetst:
    def test_zuivere_ruis_levert_geen_significant_uur_op(self):
        """DE VAL. Het beste van 24 uren is het maximum van 24 ruizige getallen.
        Gooi je willekeurige uitslagen in de toets, dan hoort er niets uit te
        komen -- en die eigenschap is het hele bestaansrecht van deze functie."""

        import numpy as np

        rng = np.random.default_rng(1)
        stamps = pd.date_range("2026-01-01", periods=480, freq="h", tz=UTC)
        cycli = [
            Cyclus(geopend=s, soort="straddle", netto_euro=float(rng.normal()))
            for s in stamps
        ]

        toets = uur_permutatie(cycli, rondes=200)
        assert toets["p"] > 0.05, (
            f"zuivere ruis levert een 'significant' uur op (p={toets['p']:.3f}); "
            "dan keurt deze toets alles goed"
        )

    def test_een_echt_uur_wordt_wel_gevonden(self):
        """En als hij nooit iets vindt, keurt hij alles af en is hij net zo
        nutteloos."""

        import numpy as np

        rng = np.random.default_rng(2)
        stamps = pd.date_range("2026-01-01", periods=480, freq="h", tz=UTC)
        cycli = [
            Cyclus(
                geopend=s, soort="straddle",
                netto_euro=float(rng.normal()) + (8.0 if s.hour == 14 else 0.0),
            )
            for s in stamps
        ]

        toets = uur_permutatie(cycli, rondes=200)
        assert toets["p"] < 0.05, "een uur met een enorm echt effect wordt gemist"


class TestHetAantalConfiguratiesKlopt:
    def test_het_rooster_is_de_achtenveertig_uit_de_hypothese(self):
        configs = rooster()
        assert len(configs) == 48
        assert len({c.naam for c in configs}) == 48

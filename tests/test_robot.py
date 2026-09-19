"""De robot, en de zes manieren waarop hij zichzelf mooier zou kunnen maken.

Elk van deze zes heeft deze week een uitslag opgeblazen. Ze staan hier zodat ze
dat niet nog een keer stilletjes kunnen doen:

  * INSTAPPEN OP EEN PRIJS DIE JE NIET KON KRIJGEN. Het signaal komt uit
    gesloten bars; de vroegste prijs is de opening van de bar erna.
  * DE STOP VERLIEZEN VAN HET DOEL binnen dezelfde bar.
  * DE BROKER NEGEREN. Een rekening gaat dood op margin level, niet op nul.
  * KOSTEN VERGETEN. Spread en slippage horen van elke trade af.
  * DOLLARS ALS EURO'S TELLEN. Goud noteert in USD.
  * OVERLAPPENDE POSITIES. Duizenden signalen tegelijk aanhouden meet de
    hefboom en niet de regel.
"""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from dataclasses import replace

from scripts.robot import (
    CONTRACT, EURUSD, Regels, _nachten, _stopprijs, draai, koop_en_hou,
)
from scripts.uitvoering import Uitvoering

#: Een broker zonder kosten, zodat een test over de MECHANIEK niet struikelt
#: over centen. Waar de kosten zelf getest worden staat hij expliciet aan.
GRATIS = Uitvoering(spread=0.0, slippage=0.0, commissie_per_lot_per_kant=0.0)


def _bars(slot: list[float], start="2026-01-05 00:00") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(slot), freq="1min", tz=UTC)
    reeks = pd.Series(slot, index=index, dtype=float)
    return pd.DataFrame({
        "open": reeks.shift(1).fillna(reeks.iloc[0]),
        "high": reeks + 0.5,
        "low": reeks - 0.5,
        "close": reeks,
    })


def _terugkeermarkt(n: int = 3000, zaad: int = 5) -> pd.DataFrame:
    """Een markt die naar 4000 terugveert -- daar is de robot op gebouwd."""

    rng = np.random.default_rng(zaad)
    prijs, koersen = 4000.0, []
    for _ in range(n):
        prijs = 4000.0 + (prijs - 4000.0) * 0.97 + rng.normal(0, 1.0)
        koersen.append(prijs)
    return _bars(koersen)


class TestEenPositieTegelijk:
    def test_trades_overlappen_nooit(self):
        """Duizenden signalen tegelijk aanhouden meet de hefboom, niet de regel."""

        uit = draai(_terugkeermarkt(), regels=Regels(uitvoering=GRATIS),
                    balans=10_000.0)
        assert uit.trades, "geen enkele trade; dan test dit niets"
        for vorige, volgende in zip(uit.trades, uit.trades[1:]):
            assert vorige.gesloten is not None
            assert volgende.geopend > vorige.gesloten, (
                f"trade van {volgende.geopend} opende terwijl die van "
                f"{vorige.geopend} nog liep")


class TestDeInstapprijs:
    def test_er_wordt_ingestapt_op_de_volgende_bar(self):
        m1 = _terugkeermarkt()
        uit = draai(m1, regels=Regels(uitvoering=GRATIS), balans=10_000.0)

        for t in uit.trades[:20]:
            assert t.entry == pytest.approx(float(m1.loc[t.geopend, "open"])), (
                "de instapprijs is niet de opening van de bar waarop wordt "
                "ingestapt")


class TestDeStopWintDeBar:
    def test_een_trade_die_zijn_stop_raakt_verliest(self):
        """Binnen een bar kent niemand de volgorde. De andere aanname flatteert
        elke uitslag, en dat is deze week vier keer gebeurd.

        EERST RUIS EN DAN PAS DE VAL, want zonder ruis is er geen spreiding en
        dus geen signaal -- mijn eerste testmarkt stond honderd bars stil op
        4000 en opende daardoor nooit een positie voor de val begon.
        """
        rng = np.random.default_rng(3)
        koersen = list(4000.0 + rng.normal(0, 2.0, 300))
        koersen += list(np.linspace(koersen[-1], koersen[-1] - 200.0, 400))
        m1 = _bars(koersen)

        uit = draai(m1, regels=Regels(uitvoering=GRATIS, horizon=120,
                                      lengte=30, atr_lengte=30, sigma=1.0),
                    balans=1_000_000.0)

        gestopt = [t for t in uit.trades if t.reden == "stop"]
        assert gestopt, "geen enkele trade raakte zijn stop in deze val"
        for t in gestopt:
            assert t.r < 0, "een trade die zijn stop raakte levert winst op"
            # En hij is precies OP de stop afgerekend, niet ergens lager.
            assert t.exit == pytest.approx(t.stop)

    def test_iedere_trade_heeft_een_stop(self):
        """Harde projectregel: geen trades zonder stoploss."""

        uit = draai(_terugkeermarkt(), regels=Regels(uitvoering=GRATIS),
                    balans=10_000.0)
        for t in uit.trades:
            assert t.stop != t.entry
            afstand = abs(t.entry - t.stop)
            assert afstand > 0, "trade zonder stop"
            # En hij staat aan de goede kant.
            if t.richting > 0:
                assert t.stop < t.entry
            else:
                assert t.stop > t.entry


class TestDeKostenGaanEraf:
    def test_met_spread_blijft_er_minder_over_dan_zonder(self):
        m1 = _terugkeermarkt()
        gratis = draai(m1, regels=Regels(uitvoering=GRATIS), balans=10_000.0)
        duur = draai(m1, regels=Regels(
            uitvoering=Uitvoering(spread=0.50, slippage=0.25,
                                  commissie_per_lot_per_kant=0.0)),
            balans=10_000.0)

        assert len(gratis.trades) == len(duur.trades), (
            "de kosten mogen het AANTAL trades niet veranderen, alleen de winst")
        assert duur.eind < gratis.eind, "de spread kost niets"

    def test_de_kosten_kloppen_op_de_cent(self):
        """MET EEN VAST LOT, anders vergelijk je appels met peren.

        De lotgrootte schaalt mee met de balans, dus zodra de kosten de balans
        raken lopen de twee runs uiteen en meet je niet meer alleen de kosten.
        `max_lot=0.01` klemt allebei de runs op het minimumlot vast.
        """
        m1 = _terugkeermarkt()
        vast = Regels(uitvoering=GRATIS, max_lot=0.01)
        gratis = draai(m1, regels=vast, balans=10_000.0)
        duur = draai(m1, regels=replace(
            vast, uitvoering=Uitvoering(spread=0.20, slippage=0.10,
                                        commissie_per_lot_per_kant=0.0)),
            balans=10_000.0)

        assert len(gratis.trades) == len(duur.trades)
        assert all(t.lot == 0.01 for t in gratis.trades)
        # 0,20 spread + 0,10 slippage, op 0,01 lot van 100 ounce = $0,30 per trade.
        verwacht_usd = 0.30 * 0.01 * CONTRACT * len(gratis.trades)
        assert gratis.eind - duur.eind == pytest.approx(verwacht_usd / EURUSD,
                                                        rel=0.001)


class TestGoudRekentInDollars:
    def test_een_andere_eurokoers_verandert_het_eindbedrag(self):
        """Codex wees dit aan en het klopte: XAUUSD noteert in USD, de rekening
        staat in euro. Die twee door elkaar halen is een fout van 5 tot 15%."""

        m1 = _terugkeermarkt()
        laag = draai(m1, regels=Regels(uitvoering=GRATIS, eurusd=1.00),
                     balans=10_000.0)
        hoog = draai(m1, regels=Regels(uitvoering=GRATIS, eurusd=1.20),
                     balans=10_000.0)

        assert laag.eind != hoog.eind, (
            "de eurokoers doet niets -- dollars worden als euro's geteld")


class TestDeStopprijs:
    """De regel zelf, los getest.

    Ik zat een kunstmatige markt te kneden om deze ene regel via een hele
    simulatie te raken -- drie pogingen, en elke poging raakte net de andere
    kant op. Dat is de verkeerde volgorde: een regel die een naam verdient,
    verdient ook een eigen test.
    """

    def test_zonder_gat_vul_je_gewoon_op_je_stop(self):
        assert _stopprijs(3990.0, 3995.0, +1) == 3990.0
        assert _stopprijs(4010.0, 4005.0, -1) == 4010.0

    def test_met_een_gat_vul_je_op_de_opening(self):
        # Long met stop op 3990, bar opent al op 3900: je krijgt 3900.
        assert _stopprijs(3990.0, 3900.0, +1) == 3900.0
        # Short met stop op 4010, bar opent al op 4100: je krijgt 4100.
        assert _stopprijs(4010.0, 4100.0, -1) == 4100.0

    def test_het_is_altijd_de_slechtste_van_de_twee(self):
        for stop, opening in ((3990.0, 3900.0), (3990.0, 3995.0),
                              (4010.0, 4100.0), (4010.0, 4005.0)):
            assert _stopprijs(stop, opening, +1) == min(stop, opening)
            assert _stopprijs(stop, opening, -1) == max(stop, opening)


class TestEenGatDoorJeStopHeen:
    """Een stop is een verzoek, geen prijsgarantie.

    Hier stond `exit = stop`, altijd. Dat is de vriendelijke aanname: dan kun je
    per definitie nooit meer verliezen dan je risico en word je dus ook nooit
    geliquideerd. Dat klinkt geruststellend en het is niet waar -- na een
    weekend of bij nieuws opent de bar al voorbij je stop en word je op die
    opening gevuld.
    """

    GAT = 600
    #: Bars aanloop voor het gat. Moet ruimer zijn dan `horizon` (120).
    AANLOOP = 200

    @classmethod
    def _markt_met_gat(cls, omhoog: bool = True):
        """Een ECHT gat, en dat is iets anders dan een grote bar.

        `_bars` zet de opening van elke bar gelijk aan de vorige slotkoers. Dan
        is er nooit een gat: de prijs valt binnen de bar en je stop wordt netjes
        geraakt. Bij een weekendgat springt de OPENING zelf, en precies dat moet
        hier gebeuren -- dus die kolom wordt met de hand gezet.
        """
        rng = np.random.default_rng(8)
        koersen = list(4000.0 + rng.normal(0, 2.0, cls.GAT - cls.AANLOOP))

        # EERST EEN AANLOOP DE KANT VAN HET GAT OP, en dat is de kern.
        #
        # Een terugkeer-regel neemt altijd de TEGENOVERGESTELDE kant: staat de
        # prijs boven zijn gemiddelde, dan verkoopt hij. Dus vlak voor een gat
        # OMHOOG hangt er een short, en vlak voor een gat OMLAAG een long. Precies
        # die posities worden door het gat heen geramd.
        #
        # Mijn eerste versie liet de markt eerst vlak lopen en dan omlaag
        # springen. De positie die op dat moment openstond was een short, en die
        # VERDIENT aan een gat omlaag. Ik testte dus het gunstige geval en
        # noemde het een gattest.
        kant = 1.0 if omhoog else -1.0
        aanloop = koersen[-1]
        # LANGER DAN DE HORIZON, anders staat de vorige positie er nog en
        # opent de goede kant nooit -- een positie tegelijk, en de oude
        # short bleef 120 bars hangen terwijl mijn aanloop er 60 duurde.
        koersen += [aanloop + kant * i * 0.15
                    for i in range(1, cls.AANLOOP + 1)]

        sprong = koersen[-1] + kant * 100.0
        koersen += [sprong] * 300
        m1 = _bars(koersen)
        m1.iloc[cls.GAT, m1.columns.get_loc("open")] = sprong
        m1.iloc[cls.GAT, m1.columns.get_loc(
            "low" if omhoog else "high")] = sprong + (-0.5 if omhoog else 0.5)
        return m1

    def test_in_een_echte_run_valt_een_trade_door_zijn_stop(self):
        omhoog = False
        m1 = self._markt_met_gat(omhoog)
        uit = draai(m1, regels=Regels(uitvoering=GRATIS, lengte=30,
                                      atr_lengte=30, sigma=1.0, horizon=120),
                    balans=1_000_000.0)

        # Slechter dan de stop: voor een long lager, voor een short hoger.
        doorheen = [
            t for t in uit.trades
            if t.reden == "stop" and (
                (t.richting > 0 and t.exit < t.stop - 1e-9)
                or (t.richting < 0 and t.exit > t.stop + 1e-9))
        ]
        assert doorheen, (
            "geen enkele trade viel door zijn stop heen in een gat van honderd "
            "punten -- de stop vult blijkbaar altijd precies")
        for t in doorheen:
            assert t.r < -1.0, (
                "door een gat verlies je MEER dan een R; deze trade niet")


class TestDeBrokerGooitJeEruit:
    def test_een_te_kleine_rekening_gaat_dood_op_margin_level(self):
        """Niet op balans nul: de broker liquideert bij 50% margin level."""

        # Een val die veel groter is dan een kleine rekening aankan.
        koersen = [4000.0] * 200 + list(np.linspace(4000.0, 3000.0, 300))
        m1 = _bars(koersen)
        uit = draai(m1, regels=Regels(uitvoering=GRATIS, lengte=20,
                                      atr_lengte=20, sigma=0.5, horizon=600,
                                      stop_atr=500.0),
                    balans=59.16)

        assert uit.ruine, "een rekening van EUR 59 overleeft een val van 1000 punten"
        assert uit.eind == 0.0
        # EN DOOR DE BROKER, niet doordat de balans toevallig op nul uitkwam.
        # Zonder deze regel slaagt de test ook als de stop-out helemaal niet
        # bestaat, en dan bewaakt hij niets.
        assert uit.trades[-1].reden == "uitgegooid", (
            f"de rekening ging dood met reden '{uit.trades[-1].reden}' in plaats "
            f"van door de marge-stop-out")

    def test_een_grote_rekening_overleeft_diezelfde_val(self):
        koersen = [4000.0] * 200 + list(np.linspace(4000.0, 3000.0, 300))
        m1 = _bars(koersen)
        uit = draai(m1, regels=Regels(uitvoering=GRATIS, lengte=20,
                                      atr_lengte=20, sigma=0.5),
                    balans=500_000.0)
        assert not uit.ruine


class TestDeNulhypothese:
    def test_koop_en_hou_rekent_de_stijging(self):
        m1 = _bars([4000.0 + i * 0.1 for i in range(1000)])
        # 999 bars x 0,1 punt = 99,9 punten omhoog, op 0,01 lot = $99,90.
        eind = koop_en_hou(m1, balans=1000.0, eurusd=1.10)
        assert eind == pytest.approx(1000.0 + 99.9 * 0.01 * CONTRACT / 1.10, rel=0.01)

    def test_in_een_dalende_markt_verliest_koop_en_hou(self):
        m1 = _bars([4000.0 - i * 0.1 for i in range(1000)])
        assert koop_en_hou(m1, balans=1000.0) < 1000.0


class TestDeVloerVanEenCentLot:
    def test_te_grote_minimumtrades_worden_geteld_en_niet_verzwegen(self):
        """Onder 0,01 lot bestaat niet. Op EUR 59 breekt dat je eigen 2%-grens,
        en dat hoort in de uitslag te staan in plaats van weggemoffeld."""

        uit = draai(_terugkeermarkt(), regels=Regels(uitvoering=GRATIS),
                    balans=59.16)
        assert uit.te_klein > 0, (
            "op EUR 59 past 0,01 lot volgens deze meting altijd binnen 2% risico")

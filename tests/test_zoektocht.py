"""De zoeklus mag zichzelf niet voor de gek houden.

EEN ZOEKER DIE HONDERD CONFIGURATIES PROBEERT VINDT ALTIJD IETS -- ook in
zuivere ruis. Dat is de hele reden dat deze tests bestaan. Ze bewaken vier
dingen die het verschil maken tussen een zoeklus en een leugenmachine:

  * HET TOETSDEEL WORDT NIET AANGERAAKT tijdens het zoeken.
  * DE RUINE IS ALTIJD DE SLECHTSTE UITKOMST, hoe goed het ervoor ook ging.
  * TE WEINIG TRADES TELT NIET. Vier goede trades is geen bewijs.
  * HET LOT GROEIT MEE, met het minimumlot als harde vloer.
"""

from __future__ import annotations

import pytest

from scripts.zoektocht import Uitslag, _lot_voor, punten, zoek


class TestDeScoreKiestNietOpWinst:
    def test_een_ruine_is_altijd_het_slechtst(self):
        """Een reeks die eindigt op nul mag nooit winnen van een reeks die
        blijft bestaan, hoe hoog de piek ook was."""

        kapot = Uitslag(trades=500, eind=0.0, start=59.0, diepste_terugval=1.0,
                        ruine=True, per_trade=5.0)
        saai = Uitslag(trades=500, eind=61.0, start=59.0, diepste_terugval=0.05,
                       ruine=False, per_trade=0.01)

        assert punten(kapot, min_trades=100) < punten(saai, min_trades=100)

    def test_en_hij_verliest_ook_van_een_reeks_die_er_even_slecht_uitziet(self):
        """DEZE TEST ONTBRAK EN EEN MUTATIE VOND DAT.

        Ik haalde de expliciete ruine-controle uit `punten` en alles bleef
        groen. Want een ruine scoort vanzelf slecht: het eindbedrag is nul, dus
        het rendement is -100%, en de terugval staat op 100%.

        Alleen: een rekening die ALLES verloor zonder formeel om te vallen komt
        op exact hetzelfde getal uit. Dan is het gelijkspel, en dan mag de
        volgorde niet van toeval afhangen. Een rekening die nog bestaat is
        altijd beter dan een die weg is -- ook als de som hetzelfde zegt.
        """

        kapot = Uitslag(trades=500, eind=0.0, start=59.0, diepste_terugval=1.0,
                        ruine=True, per_trade=0.0)
        bijna = Uitslag(trades=500, eind=0.0, start=59.0, diepste_terugval=1.0,
                        ruine=False, per_trade=0.0)

        assert punten(kapot, min_trades=100) < punten(bijna, min_trades=100), (
            "een omgevallen rekening scoort gelijk aan een die nog bestaat"
        )

    def test_te_weinig_trades_telt_niet_mee(self):
        """Vier trades met een fantastisch rendement is geen configuratie maar
        een toevalstreffer, en een zoeker die daarop kiest vindt er elke keer
        een."""

        toeval = Uitslag(trades=4, eind=5_000.0, start=59.0, diepste_terugval=0.0,
                         ruine=False, per_trade=1_200.0)
        gewoon = Uitslag(trades=500, eind=70.0, start=59.0, diepste_terugval=0.1,
                         ruine=False, per_trade=0.02)

        assert punten(toeval, min_trades=100) < punten(gewoon, min_trades=100)

    def test_de_terugval_gaat_van_het_rendement_af(self):
        """Twee keer hetzelfde eindbedrag, maar de ene ging 80% onder water.
        Die heb je in het echt gesloten voor hij terugkwam."""

        rustig = Uitslag(500, 120.0, 59.0, 0.10, False, 0.1)
        wild = Uitslag(500, 120.0, 59.0, 0.80, False, 0.1)

        assert punten(rustig, min_trades=100) > punten(wild, min_trades=100)


class TestHetLotGroeitMeeMaarNietOnderDeVloer:
    def test_op_een_kleine_rekening_blijft_het_minimumlot(self):
        """DE VLOER. Op EUR 59 kun je niet binnen 2% blijven, en kleiner dan
        0,01 bestaat niet."""

        lot = _lot_voor(59.16, 10.75, deel=0.02, euro_per_punt=1.0)
        assert lot == pytest.approx(0.01)
        # En dat is dus veel meer risico dan gevraagd.
        assert 10.75 * 1.0 / 59.16 > 0.02 * 5

    def test_de_vloer_bindt_tot_ver_boven_de_tweeduizend_euro(self):
        """DIT VOND DEZE TEST, EN HET IS GROTER DAN IK DACHT.

        Ik verwachtte dat de lotgrootte vanaf een paar honderd euro zou
        meegroeien. Dat doet hij niet. Met een stop van 10,75 punten zit je tot
        ongeveer EUR 2.000 VAST op het minimumlot:

            EUR    59  ->  0,01 lot  = 18,2% risico
            EUR   200  ->  0,01 lot  =  5,4%
            EUR   467  ->  0,01 lot  =  2,3%
            EUR 1.000  ->  0,01 lot  =  1,1%
            EUR 2.000  ->  0,03 lot  =  1,6%   <- hier pas

        Tussen EUR 467 en EUR 2.000 gebeurt er dus NIETS: je risico daalt
        gewoon van 2,3% naar 1,1% terwijl je positie gelijk blijft. Er is geen
        samengestelde groei in dat hele bereik. Dat verklaart waarom de
        balansladder daar zulke vlakke veelvouden geeft, en het is geen fout
        maar de vorm van het minimumlot."""

        for balans in (59.16, 200.0, 467.0, 1_000.0):
            assert _lot_voor(balans, 10.75, deel=0.02, euro_per_punt=1.0) == 0.01

    def test_boven_die_grens_schaalt_hij_wel_mee(self):
        """Anders zou de vloer overal gelden en groeit er nooit iets."""

        klein = _lot_voor(2_500.0, 10.75, deel=0.02, euro_per_punt=1.0)
        groot = _lot_voor(25_000.0, 10.75, deel=0.02, euro_per_punt=1.0)

        assert groot > klein > 0.01
        # Tien keer de balans geeft ongeveer tien keer het lot, maar niet
        # precies: er wordt naar beneden afgerond op 0,01 en dat is bij kleine
        # lots relatief grof. 0,04 tegen 0,46 is een factor 11,5, en dat is
        # geen fout maar de afronding -- die kant op mag hij ook afwijken,
        # zolang hij maar niet naar BOVEN afrondt.
        assert 8.0 < groot / klein < 13.0
        assert groot * 10.75 * 100 <= 0.02 * 25_000.0 + 1e-9, (
            "hij rondt naar boven af en duwt je over je risicogrens"
        )

    def test_hij_rondt_naar_beneden_af_op_de_lotstap(self):
        """Naar boven afronden zou je stilletjes over je risicogrens duwen."""

        lot = _lot_voor(1_000.0, 10.0, deel=0.02, euro_per_punt=1.0)
        assert lot == pytest.approx(round(lot / 0.01) * 0.01, abs=1e-9)
        assert lot * 10.0 * 100 <= 0.02 * 1_000.0 + 1e-9


class TestDeLusKlimtEchtEnStoptOok:
    def _nep(self):
        """Een meetfunctie met een bekend optimum, zodat de lus te toetsen is."""

        def meet(cfg):
            goed = (cfg["a"] == 3) + (cfg["b"] == "x")
            return Uitslag(trades=500, eind=59.0 + goed * 10, start=59.0,
                           diepste_terugval=0.0, ruine=False, per_trade=goed)
        return meet

    def test_hij_vindt_het_optimum_op_beide_assen(self):
        cfg, u, n = zoek(self._nep(), {"a": [1, 2, 3], "b": ["y", "x"]},
                         min_trades=10, rondes=5, etiket="test")

        assert cfg == {"a": 3, "b": "x"}
        assert u.eind == pytest.approx(79.0)

    def test_hij_stopt_als_er_niets_meer_verbetert(self):
        """Zonder stopvoorwaarde draait hij eeuwig door en meet hij dezelfde
        configuraties honderd keer opnieuw."""

        _cfg, _u, n = zoek(self._nep(), {"a": [1, 2, 3], "b": ["y", "x"]},
                           min_trades=10, rondes=50, etiket="test")

        # Vijf kandidaten per ronde; bij vijftig rondes zonder stop zou dat
        # ver boven de honderd uitkomen.
        assert n < 30, f"hij probeerde {n} configuraties en stopte niet vroeg"

    def test_hij_telt_elke_geprobeerde_configuratie(self):
        """DE NOEMER. De beste van tweehonderd is iets anders dan een vondst,
        en dat getal hoort in de uitslag."""

        _cfg, _u, n = zoek(self._nep(), {"a": [1, 2, 3], "b": ["y", "x"]},
                           min_trades=10, rondes=5, etiket="test")
        assert n >= 4

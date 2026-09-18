"""Sectie 20: de mechaniek van de ladder, zonder MT5.

WAT HIER BEWAAKT WORDT. Niet of de ladder goed is -- dat is precies wat de
meting moet uitwijzen. Deze tests bewaken de vier keuzes die een grid er
stilletjes beter uit laten zien dan hij is:

  * DE VOLGORDE BINNEN EEN BAR. Eerst bijvullen, dan pas op winst kijken.
    Andersom sluit je manden die in werkelijkheid nog dieper gingen.
  * DE DIEPSTE STAND OP DE LOW. Op de close meten verzwijgt precies het moment
    waarop de rekening het krapst stond.
  * DE RUINE IN DE LUS. Een mand die de balans halverwege opmaakt, sluit niet
    meer -- de broker liquideert. Achteraf sommeren maakt van een ruine een
    gewone verliezer.
  * KOSTEN PER BEEN. Een grid opent tien keer zoveel posities als een gewone
    regel; kosten per mand rekenen laat negen van de tien spreads verdwijnen.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pandas as pd
import pytest

from scripts.section_twenty_pullback_ladder import (
    COMMISSIE_PER_LOT_PER_KANT,
    CONTRACT,
    KLOKKEN,
    Instelling,
    rapport,
    rooster,
    simuleer,
    stapel_omhoog,
    stapel_richting,
    kosten_per_been,
)


def _frame(prijzen: list[tuple[float, float, float, float]], start="2026-01-05 09:00"):
    index = pd.date_range(start, periods=len(prijzen), freq="1min", tz=UTC)
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c} for o, h, l, c in prijzen],
        index=index,
    )


def _stijgende_stapel(index) -> dict[str, pd.DataFrame]:
    """Alle klokken omhoog, zodat de instapvoorwaarde niet in de weg zit.

    De reeks begint een minuut VOOR de M1-bars, want `stapel_omhoog` heeft een
    voorgaande gesloten bar nodig om "hoger dan de vorige" te kunnen zeggen --
    en zonder die extra bar opent de eerste mand nooit.
    """

    reeks = index.union(pd.DatetimeIndex([index[0] - pd.Timedelta(minutes=1)]))
    frame = pd.DataFrame(
        {"open": range(len(reeks)), "close": [i + 0.5 for i in range(len(reeks))]},
        index=reeks,
    )
    return {naam: frame for naam in ("M1", "M5", "M15")}


class TestDeVolgordeBinnenEenBar:
    def test_eerst_bijvullen_en_pas_daarna_op_winst_kijken(self):
        """DE KERN. Een bar die zowel een nieuw been raakt als het doel, moet
        het BEEN vullen. Andersom ziet elke ladder er ondieper uit dan hij was,
        en de diepte is het enige getal dat er bij een grid toe doet."""

        # Bar 1 opent de mand op 100. Bar 2 zakt naar 99 (twee benen erbij bij
        # stap 0,5) en stijgt in dezelfde bar naar 101.
        m1 = _frame([(100.0, 100.0, 100.0, 100.0), (100.0, 101.0, 99.0, 100.5)])
        stapels = _stijgende_stapel(m1.index)

        manden, _ = simuleer(
            m1, stapels,
            instelling=Instelling(stap=0.5, spread=0.0, lot=0.01),
            balans=10_000.0,
        )

        assert manden, "er is geen mand geopend"
        assert manden[0].aantal_benen == 3, (
            "de benen op 99,5 en 99,0 zijn niet gevuld; dan wordt de ladder te "
            "ondiep gerapporteerd"
        )


class TestDeNogOpenMandTeltMee:
    """DE FOUT DIE EEN GRID ONVERSLAANBAAR MAAKT OP PAPIER.

    Een mand sluit alleen wanneer hij in winst komt. Dus alles wat NIET
    terugkwam blijft openstaan -- en tel je alleen de GESLOTEN manden, dan is
    elke mand in je uitslag per definitie een winnaar.

    Dat is overlevingsselectie in zijn zuiverste vorm, en bij dit mechanisme is
    de nog-open mand juist degene waar het antwoord in zit. De eerste versie
    van `simuleer` liet hem vallen; deze test vond dat.
    """

    def test_een_mand_die_nooit_terugkwam_staat_in_de_uitslag(self):
        m1 = _frame([
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 90.0, 90.0),     # zakt weg en komt niet terug
        ])
        stapels = _stijgende_stapel(m1.index)

        manden, kapot = simuleer(
            m1, stapels,
            instelling=Instelling(stap=100.0, spread=0.0, lot=0.01),
            balans=10_000.0,
        )

        assert not kapot
        assert len(manden) == 1, (
            "de openstaande mand is weggelaten; dan bestaat je uitslag alleen "
            "uit manden die toevallig terugkwamen"
        )
        assert manden[0].gesloten is None, "hij hoort als OPEN gemarkeerd te zijn"
        assert manden[0].resultaat_euro < 0, (
            "hij wordt niet op de laatste koers gewaardeerd"
        )

    def test_de_diepste_stand_komt_van_de_low(self):
        """Op de close meten verzwijgt het moment waarop de rekening het krapst
        stond, en dat is nou juist het moment dat de ruine bepaalt."""

        m1 = _frame([
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 90.0, 99.9),   # diep de diepte in, sluit vlakbij
        ])
        stapels = _stijgende_stapel(m1.index)

        manden, _ = simuleer(
            m1, stapels,
            instelling=Instelling(stap=100.0, spread=0.0, lot=0.01),
            balans=10_000.0,
        )

        # Een been op 100, low 90 -> tien punten onder water, min de kosten.
        verwacht = (90.0 - 100.0) * 0.01 * CONTRACT - kosten_per_been(0.0, 0.01)
        assert manden[0].diepste_euro == pytest.approx(verwacht), (
            "de diepste stand is op de close gemeten in plaats van op de low"
        )
        # En de mand zelf staat bijna vlak: dat is het hele verschil.
        assert manden[0].resultaat_euro > verwacht


class TestDeRuineIsGeenGewoneVerliezer:
    def test_de_rekening_die_opgaat_stopt_de_simulatie(self):
        """Een mand die de balans opmaakt, sluit niet meer. Doorrekenen alsof
        hij netjes terugkwam, is precies hoe een grid op papier overleeft."""

        m1 = _frame([
            (4000.0, 4000.0, 4000.0, 4000.0),
            (4000.0, 4000.0, 3000.0, 3000.0),   # 1000 punten omlaag
            (3000.0, 4100.0, 3000.0, 4100.0),   # en het "komt altijd terug"
        ])
        stapels = _stijgende_stapel(m1.index)

        manden, kapot = simuleer(
            m1, stapels,
            instelling=Instelling(stap=1000.0, spread=0.0, lot=0.01),
            balans=100.0,
        )

        assert kapot, "de rekening overleeft een verlies groter dan de balans"
        assert manden[-1].resultaat_euro == pytest.approx(-100.0), (
            "een ruine kost precies de balans en niet meer of minder"
        )
        assert len(manden) == 1, (
            "er wordt doorgehandeld na de ruine; de derde bar mag niet meer "
            "meetellen"
        )

    def test_zonder_ruine_loopt_hij_gewoon_door(self):
        """Als deze test niet bestond, zou `simuleer` altijd 'kapot' kunnen
        teruggeven en nog steeds groen zijn."""

        m1 = _frame([
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 99.0, 99.0),
            (99.0, 102.0, 99.0, 102.0),
        ])
        stapels = _stijgende_stapel(m1.index)

        manden, kapot = simuleer(
            m1, stapels,
            instelling=Instelling(stap=0.5, spread=0.0, lot=0.01),
            balans=10_000.0,
        )

        assert not kapot
        assert manden[0].resultaat_euro > 0


class TestKostenPerBeen:
    def test_elke_poot_betaalt_zijn_eigen_spread_twee_keer(self):
        """EEN GRID OPENT TIEN KEER ZOVEEL POSITIES. Kosten per mand rekenen
        laat negen van de tien spreads verdwijnen, en juist die stapel kosten
        is wat het mechanisme duur maakt."""

        m1 = _frame([
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 99.0, 99.0),     # drie benen: 100, 99.5, 99
            (99.0, 101.0, 99.0, 101.0),
        ])
        stapels = _stijgende_stapel(m1.index)

        zonder, _ = simuleer(
            m1, stapels,
            instelling=Instelling(stap=0.5, spread=0.0, lot=0.01),
            balans=10_000.0,
        )
        met, _ = simuleer(
            m1, stapels,
            instelling=Instelling(stap=0.5, spread=0.16, lot=0.01),
            balans=10_000.0,
        )

        benen = zonder[0].aantal_benen
        assert benen == 3
        verschil = zonder[0].resultaat_euro - met[0].resultaat_euro
        # Alleen het SPREAD-deel verschilt; de commissie zit in allebei.
        verwacht = (kosten_per_been(0.16, 0.01) - kosten_per_been(0.0, 0.01)) * benen
        assert verschil == pytest.approx(verwacht), (
            "de kosten schalen niet met het aantal benen"
        )


class TestGoudBetaaltGeenCommissie:
    """De kostenformule mag niet opnieuw het FOREX-getal pakken.

    `commission_per_lot_per_side: 2.75` in `config/eightcap.yaml` geldt alleen
    voor forex. Daaronder staat `commission_by_asset_class` met `metal: 0.0`,
    gemeten uit de rekening zelf. Ik pakte de eerste en rekende daarmee kosten
    door die goud nooit betaalt. Deze test leest allebei uit de config en houdt
    de constante vast aan de juiste.
    """

    @staticmethod
    def _uit_config() -> dict:
        import yaml

        pad = Path(__file__).resolve().parents[1] / "config" / "eightcap.yaml"
        with open(pad, encoding="utf-8") as f:
            return yaml.safe_load(f)["risk"]

    def test_de_constante_volgt_de_metaalregel_uit_de_config(self):
        risk = self._uit_config()
        metaal = risk["commission_by_asset_class"]["metal"]

        assert metaal == 0.0, "de config zegt niet langer nul voor metaal"
        assert COMMISSIE_PER_LOT_PER_KANT == metaal, (
            "de meting rekent een andere commissie dan de config voor goud zegt"
        )
        assert COMMISSIE_PER_LOT_PER_KANT != risk["commission_per_lot_per_side"], (
            "dit is het FOREX-getal; goud valt niet op die regel terug"
        )

    def test_de_kosten_van_een_been_zijn_puur_spread(self):
        # 0,16 punt spread, heen en terug, 0,01 lot van 100 ounce = EUR 0,32.
        assert kosten_per_been(0.16, 0.01) == pytest.approx(0.32)
        assert kosten_per_been(0.0, 0.01) == pytest.approx(0.0), (
            "zonder spread hoort een goudbeen niets te kosten"
        )


class TestDeMandstopDoetIets:
    def test_met_mandstop_wordt_er_afgekapt_en_zonder_niet(self):
        m1 = _frame([
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 80.0, 80.0),
            (80.0, 101.0, 80.0, 101.0),
        ])
        stapels = _stijgende_stapel(m1.index)

        open_variant, _ = simuleer(
            m1, stapels,
            instelling=Instelling(stap=100.0, spread=0.0, lot=0.01, mandstop_deel=None),
            balans=1_000.0,
        )
        begrensd, _ = simuleer(
            m1, stapels,
            instelling=Instelling(stap=100.0, spread=0.0, lot=0.01, mandstop_deel=0.02),
            balans=1_000.0,
        )

        assert not open_variant[0].afgekapt
        assert open_variant[0].resultaat_euro > 0, "onbegrensd wacht het uit en wint"
        assert begrensd[0].afgekapt, "de mandstop heeft niets gedaan"
        assert begrensd[0].resultaat_euro < 0, "afkappen kost geld, dat is het punt"

    def test_max_benen_begrenst_de_ladder(self):
        m1 = _frame([
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 90.0, 90.0),    # zou 21 benen geven bij stap 0,5
        ])
        stapels = _stijgende_stapel(m1.index)

        for grens in (5, 10):
            manden, _ = simuleer(
                m1, stapels,
                instelling=Instelling(stap=0.5, spread=0.0, lot=0.01, max_benen=grens),
                balans=100_000.0,
            )
            assert manden[0].aantal_benen == grens


class TestDeInstapKijktNietVooruit:
    def test_de_stapel_gebruikt_de_laatste_GESLOTEN_bar(self):
        """Een klok die met de lopende bar meebeweegt, weet op het moment van
        instappen dingen die je live niet hebt. Bij een M1-ladder is dat extra
        verleidelijk omdat die bar zo vaak ververst."""

        index = pd.date_range("2026-01-05 09:00", periods=4, freq="1min", tz=UTC)
        # De laatste bar is bullish, de bar ervoor niet.
        frame = pd.DataFrame(
            {"open": [10.0, 10.0, 10.0, 1.0], "close": [9.0, 9.0, 9.0, 99.0]},
            index=index,
        )
        stapels = {naam: frame for naam in ("M1", "M5", "M15")}

        # Op het moment van de derde bar is de laatste GESLOTEN bar de tweede,
        # en die is bearish.
        assert not stapel_omhoog(stapels, index[2])
        # Pas ná de vierde bar is de stapel omhoog.
        assert stapel_omhoog(stapels, index[3] + pd.Timedelta(minutes=1))


class TestHetRapportZetDeRuineBoven:
    def test_ruine_en_diepte_staan_voor_trefkans(self):
        """DE VOLGORDE IS DE BOODSCHAP. Een grid heeft bij vrijwel elke
        steekproef een trefkans boven de 90%. Dat getal bovenaan zetten
        verbergt de uitkomst in plaats van hem te tonen."""

        m1 = _frame([
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 99.0, 101.0),
        ])
        stapels = _stijgende_stapel(m1.index)
        manden, kapot = simuleer(
            m1, stapels,
            instelling=Instelling(stap=0.5, spread=0.0, lot=0.01),
            balans=1_000.0,
        )

        uit = rapport(manden, kapot, balans=1_000.0)
        sleutels = list(uit)
        assert sleutels.index("ruine") < sleutels.index("trefkans")
        assert sleutels.index("diepste_euro") < sleutels.index("netto_euro")
        assert sleutels.index("meeste_benen") < sleutels.index("trefkans")
        # En de ergste mand staat er voluit bij, want een gemiddelde verbergt
        # precies de mand die de rekening had gekost.
        assert "ergste_mand" in uit


class TestHetAantalConfiguratiesKlopt:
    def test_het_rooster_is_de_tweeenzeventig_uit_de_hypothese(self):
        """DE NOEMER. De beste van 72 is iets anders dan een ontdekking, en dat
        getal hoort bij de uitslag te staan."""

        configs = rooster()
        assert len(configs) == 72
        assert len({c.naam for c in configs}) == 72, "er zitten dubbele in"


class TestDeLadderWerktBeideKantenOp:
    """DE GROOTSTE TEKORTKOMING VAN DE EERSTE VERSIE, en de eigenaar wees hem aan.

    `stapel_omhoog` gaf alleen True als ALLES omhoog stond. De sectie kon dus
    niet shorten -- hij kocht of hij deed niets.

    Dat maakte de hele meting waardeloos zonder dat er iets aan de rekensom
    mankeerde. Goud ging in het gemeten venster ruim 70% omhoog. Een regel die
    alleen koopt komt in zo'n markt bij elke terugval vanzelf goed: de mand
    loopt weg, je koopt bij, en de trend haalt je terug. Dat is de stijging en
    niet de strategie -- en het verklaart de EUR 132.847.

    De regel is TRENDVOLGEND: alle klokken omhoog -> kopen en bijkopen op de
    dip; alle klokken omlaag -> verkopen en bijverkopen op de rally.
    """

    def _klokken(self, index, richting):
        """Alle zeven klokken dezelfde kant op."""

        reeks = index.union(pd.DatetimeIndex([index[0] - pd.Timedelta(hours=2)]))
        n = len(reeks)
        if richting > 0:
            data = {"open": range(n), "close": [i + 0.5 for i in range(n)]}
        else:
            data = {"open": range(n, 0, -1), "close": [i - 0.5 for i in range(n, 0, -1)]}
        frame = pd.DataFrame(data, index=reeks)
        return {naam: frame for naam, _ in KLOKKEN}

    def test_bij_een_dalende_stapel_wordt_er_VERKOCHT(self):
        """De kern. Zonder dit is de sectie long-only en meet ze de trend."""

        m1 = _frame([
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 100.0, 101.0),      # rally: been erbij op 100,5 en 101
            (101.0, 101.0, 98.0, 98.0),        # en terug omlaag -> winst
        ])
        stapels = self._klokken(m1.index, -1)

        manden, _ = simuleer(m1, stapels,
                             instelling=Instelling(stap=0.5, spread=0.0, lot=0.01),
                             balans=10_000.0)

        assert manden, "er is geen enkele mand geopend bij een dalende stapel"
        assert manden[0].kant == -1, "hij opende een KOOPmand in een downtrend"
        assert manden[0].aantal_benen >= 3, (
            "de benen zijn niet bijgevuld op de rally; bij een verkoopmand "
            "liggen ze HOGER en vult de high ze"
        )
        assert manden[0].resultaat_euro > 0, "de verkoopmand verdient niets als de prijs zakt"

    def test_bij_een_stijgende_stapel_wordt_er_nog_steeds_GEKOCHT(self):
        """Anders zou de reparatie de ene fout door de andere vervangen."""

        m1 = _frame([
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 99.0, 99.0),
            (99.0, 102.0, 99.0, 102.0),
        ])
        stapels = self._klokken(m1.index, +1)

        manden, _ = simuleer(m1, stapels,
                             instelling=Instelling(stap=0.5, spread=0.0, lot=0.01),
                             balans=10_000.0)

        assert manden and manden[0].kant == 1
        assert manden[0].resultaat_euro > 0

    def test_een_verdeelde_stapel_opent_niets(self):
        """Zeven klokken die het niet eens zijn is geen trend. Zonder deze eis
        handelt hij overal en meet je ruis."""

        index = pd.date_range("2026-01-05 09:00", periods=4, freq="1min", tz=UTC)
        omhoog = pd.DataFrame({"open": [1, 2, 3, 4], "close": [1.5, 2.5, 3.5, 4.5]},
                              index=index)
        omlaag = pd.DataFrame({"open": [4, 3, 2, 1], "close": [3.5, 2.5, 1.5, 0.5]},
                              index=index)
        gemengd = {naam: (omhoog if i % 2 else omlaag)
                   for i, (naam, _) in enumerate(KLOKKEN)}

        assert stapel_richting(gemengd, index[-1]) == 0

    def test_de_zeven_klokken_staan_er_alle_zeven_in(self):
        """Gevraagd is M1, M2, M3, M5, M15, M30 en M60. Er stonden er vijf."""

        namen = [naam for naam, _ in KLOKKEN]
        assert namen == ["M1", "M2", "M3", "M5", "M15", "M30", "M60"]

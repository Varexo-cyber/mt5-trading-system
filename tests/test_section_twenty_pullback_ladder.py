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

import pandas as pd
import pytest

from scripts.section_twenty_pullback_ladder import (
    CONTRACT,
    Instelling,
    rapport,
    rooster,
    simuleer,
    stapel_omhoog,
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

        verwacht = (90.0 - 100.0) * 0.01 * CONTRACT
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
        verwacht = 0.16 * 2 * 0.01 * CONTRACT * benen
        assert verschil == pytest.approx(verwacht), (
            "de kosten schalen niet met het aantal benen"
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

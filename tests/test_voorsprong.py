"""De voorsprongmeter, en de drie manieren waarop hij zou kunnen liegen.

DIT BESTAND IS HET BELANGRIJKSTE VAN HET NIEUWE BEGIN. Als de meter zelf fout
is, is elk idee dat hij goedkeurt een week werk in de verkeerde richting -- en
precies dat is deze week gebeurd.

  * VOORUITKIJKEN. Elk signaal moet op tijdstip T hetzelfde zeggen, ongeacht wat
    er na T gebeurt. De klokkenstapel las de slotkoers van de bar die nog liep.
  * HIJ MOET EEN ECHTE VOORSPRONG WEL ZIEN. Een meter die altijd "niets
    gevonden" zegt, is ook altijd veilig en volstrekt nutteloos.
  * HIJ MOET OP RUIS NIETS VINDEN. Vijf ideeen keer vijf horizonnen is
    vijfentwintig toetsen; zonder rem levert dat vanzelf "vondsten" op.
"""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from scripts.voorsprong import (
    HORIZONNEN,
    SIGNALEN,
    TOETSDEEL,
    meet,
    rapport,
    signaal_momentum,
    splits,
)


def _bars(slot: list[float], start="2026-01-05 00:00") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(slot), freq="1min", tz=UTC)
    reeks = pd.Series(slot, index=index, dtype=float)
    return pd.DataFrame({
        "open": reeks.shift(1).fillna(reeks.iloc[0]),
        "high": reeks + 0.1,
        "low": reeks - 0.1,
        "close": reeks,
    })


def _dwaling(n: int, zaad: int = 1, stap: float = 0.3) -> pd.DataFrame:
    """Zuivere ruis: een willekeurige wandeling zonder enig patroon."""

    rng = np.random.default_rng(zaad)
    return _bars(list(4000.0 + np.cumsum(rng.normal(0, stap, n))))


class TestGeenEnkelSignaalKijktVooruit:
    """DE ALGEMENE EIS, en hij vangt vooruitkijken ongeacht hoe het is
    opgeschreven: het oordeel op T mag niet veranderen als je alles vanaf T
    omgooit."""

    #: Waar geknipt wordt. VIJFENTWINTIG PUNTEN EN NIET EEN, en dat is geen
    #: grondigheid maar noodzaak. Bij één knippunt vergelijk je één oordeel, en
    #: de kans dat een vooruitkijkend signaal daar toevallig hetzelfde teken
    #: geeft is ruwweg de helft. Mijn eerste versie deed dat en liet de mutatie
    #: "haal de shift weg" gewoon passeren -- dezelfde te zwakke bewaking die
    #: `TestDeInstapKijktNietVooruit` in sectie 20 had.
    KNIPPEN = tuple(range(800, 2800, 80))

    @pytest.mark.parametrize("naam", list(SIGNALEN))
    def test_de_toekomst_verandert_het_oordeel_niet(self, naam: str):
        maak = SIGNALEN[naam]
        basis = _dwaling(3000, zaad=4)
        goed = maak(basis).to_numpy()

        betrapt = []
        for knip in self.KNIPPEN:
            anders = basis.copy()
            # Alles VANAF de knip stort in: het verleden blijft identiek, de
            # toekomst is onherkenbaar. Een signaal dat op bar T alleen bars tot
            # en met T-1 leest, kan op bar `knip` niets van deze val merken.
            val = float(basis["close"].iloc[knip - 1]) * 0.5
            for kolom, extra in (("open", 0.0), ("close", 0.0),
                                 ("high", 0.1), ("low", -0.1)):
                anders.iloc[knip:, anders.columns.get_loc(kolom)] = val + extra
            if maak(anders).to_numpy()[knip] != goed[knip]:
                betrapt.append(knip)

        assert not betrapt, (
            f"{naam}: op {len(betrapt)} van de {len(self.KNIPPEN)} knippunten "
            f"verandert het oordeel OP dat moment als alleen de TOEKOMST "
            f"verandert. Dit signaal leest de bar die nog loopt -- precies "
            f"waarmee de klokkenstapel EUR 398.572 'verdiende'. Knippunten: "
            f"{betrapt[:5]}"
        )


class TestDeMeterZietEenEchteVoorsprong:
    def test_een_ingebouwd_patroon_wordt_gevonden(self):
        """Een reeks waarin momentum ECHT werkt: elke beweging zet door.

        Zonder deze test zou een meter die overal p = 1 teruggeeft ook slagen,
        en die zou elk idee afkeuren -- ook een goed idee.
        """
        rng = np.random.default_rng(5)
        prijs, koersen = 4000.0, []
        richting = 1.0
        for i in range(4000):
            if i % 200 == 0:
                richting = -richting
            prijs += richting * 0.5 + rng.normal(0, 0.2)
            koersen.append(prijs)
        m1 = _bars(koersen)

        uit = meet(m1, signaal_momentum(m1), naam="momentum", horizon=30,
                   rondes=300)
        assert uit.aantal > 100
        assert uit.verschil > 0, "een echt doorzettend patroon levert geen winst op"
        assert uit.p <= 0.05, (
            f"een ingebouwde trend wordt niet opgemerkt (p = {uit.p:.3f})"
        )

    def test_het_omgekeerde_patroon_geeft_een_negatief_verschil(self):
        """Een markt die naar zijn gemiddelde terugveert hoort momentum te STRAFFEN.

        NIET met een zigzag die elke bar wisselt -- dat was mijn eerste poging en
        die is bij een lookback van vijf juist TRENDVOLGEND: vijf is oneven, dus
        van de afwisseling blijft er netto een halve beweging over en momentum
        wijst precies goed. De meter had gelijk en de testmarkt was fout.

        Dit is een echte terugkeer: elke afwijking van 4000 wordt voor tien
        procent teruggetrokken, op elke afstand. Daar hoort momentum te verliezen.
        """
        rng = np.random.default_rng(6)
        koersen, prijs = [], 4000.0
        for _ in range(6000):
            prijs = 4000.0 + (prijs - 4000.0) * 0.90 + rng.normal(0, 1.0)
            koersen.append(prijs)
        m1 = _bars(koersen)

        uit = meet(m1, signaal_momentum(m1, lengte=5), naam="momentum",
                   horizon=5, rondes=300)
        assert uit.verschil < 0, (
            "in een markt die terugveert levert momentum winst op")


class TestDeAchtergrondIsDeNulhypothese:
    """"+0,8 punt na dit signaal" zegt niets als de markt na ELK moment
    +0,8 punt doet.

    Dit is de fout die goud twee jaar lang verborg: het steeg 70%, dus elke
    regel die long gaat "verdient" geld. Sectie 20 mat de stijging en niet
    zichzelf. De achtergrond moet die stijging eraf halen.
    """

    @staticmethod
    def _stijgende_markt(n: int = 6000):
        rng = np.random.default_rng(21)
        return _bars(list(4000.0 + np.cumsum(rng.normal(0.05, 0.3, n))))

    def test_willekeurig_long_gaan_in_een_bullmarkt_voegt_niets_toe(self):
        m1 = self._stijgende_markt()
        rng = np.random.default_rng(3)
        # Een signaal dat niets weet: op de helft van de bars long, willekeurig.
        blind = pd.Series(rng.choice([0, 1], size=len(m1)), index=m1.index)

        uit = meet(m1, blind, naam="blind long", horizon=60, rondes=300)

        assert uit.met_signaal > 0.5, (
            "in een stijgende markt hoort blind long gaan wel degelijk punten "
            "op te leveren -- anders test deze markt niets")
        assert abs(uit.verschil) < 0.5 * uit.met_signaal, (
            f"de stijging wordt niet van het resultaat afgetrokken: "
            f"met signaal {uit.met_signaal:+.3f}, achtergrond "
            f"{uit.achtergrond:+.3f}, verschil {uit.verschil:+.3f}")
        assert uit.p > 0.05, "blind long gaan wordt als voorsprong gerapporteerd"


class TestOpRuisWordtNietsGevonden:
    def test_een_willekeurige_wandeling_levert_geen_vondst(self):
        m1 = _dwaling(6000, zaad=11)
        uit = meet(m1, signaal_momentum(m1), naam="momentum", horizon=30,
                   rondes=300)
        assert uit.p > 0.05, (
            "op zuivere ruis wordt momentum als voorsprong gerapporteerd"
        )

    def test_het_rapport_noemt_hoeveel_toetsen_er_zijn_gedaan(self):
        """De straf voor veel proberen hoort in de uitslag te staan.

        Vijf ideeen keer vijf horizonnen is vijfentwintig toetsen. Bij p = 0,05
        verwacht je er ruim EEN die er goed uitziet zonder dat er iets is. Staat
        dat er niet bij, dan leest de eerste de beste uitschieter als vondst.
        """
        regels = rapport(_dwaling(6000, zaad=12), toets=False, rondes=50)
        tekst = "\n".join(regels)
        verwacht = len(SIGNALEN) * len(HORIZONNEN)
        assert f"{verwacht} toetsen gedaan" in tekst
        assert "zonder dat er iets is" in tekst


class TestHetToetsdeelBlijftDicht:
    def test_oefenen_en_toetsen_delen_geen_enkele_bar(self):
        m1 = _dwaling(1000)
        oefen, toets = splits(m1, toets=False), splits(m1, toets=True)

        assert len(oefen) + len(toets) == len(m1)
        assert oefen.index.intersection(toets.index).empty, (
            "oefendeel en toetsdeel overlappen; dan is het toetsdeel waardeloos"
        )
        assert oefen.index[-1] < toets.index[0], "het toetsdeel is niet het NIEUWSTE deel"
        assert len(toets) == pytest.approx(len(m1) * TOETSDEEL, abs=1)

    def test_het_rapport_zegt_welk_deel_het_gemeten_heeft(self):
        m1 = _dwaling(2000)
        assert any("OEFENDEEL" in r for r in rapport(m1, toets=False, rondes=20))
        assert any("TOETSDEEL" in r for r in rapport(m1, toets=True, rondes=20))

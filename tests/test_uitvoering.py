"""De broker, en vooral: wanneer hij je eruit gooit.

WAT HIER BEWAAKT WORDT. Niet of de getallen kloppen -- die komen uit de
terminal. Wel of de MECHANIEK klopt, want dit is de laag die de vorige meting
te vriendelijk maakte:

  * STOP-OUT GAAT VOOR BALANS NUL. Een rekening met EUR 30 eigen vermogen en
    EUR 80 marge is weg, ook al staat er nog geld op.
  * DE FORMULE EN DE ZOEKER MOETEN HETZELFDE ZEGGEN. `balans_voor_beweging`
    rekent het antwoord uit; `overleefde_beweging` zoekt het stap voor stap.
    Lopen die uit elkaar, dan is er een fout in een van de twee en weet je niet
    welke.
  * MARGE GROEIT MET DE LADDER. Elk been houdt opnieuw marge in. Dit is de
    reden dat een grid op een kleine rekening eerder sterft dan het verlies
    alleen suggereert.
"""

from __future__ import annotations

import math
import sqlite3

import pytest

from scripts.uitvoering import (
    CONTRACT,
    RAW_GOUD,
    ZONDER_STOPOUT,
    Uitvoering,
    balans_voor_beweging,
    benen_bij_beweging,
    overleefde_beweging,
    slippage_uit_db,
    toestand_na_beweging,
)


class TestDeMargeRekensom:
    def test_marge_per_been_op_de_echte_maten(self):
        # 0,01 lot goud van 4000 bij 1:500: 100 x 0,01 x 4000 / 500 = EUR 8.
        assert RAW_GOUD.marge_voor(0.01, 4000.0) == pytest.approx(8.0)

    def test_marge_groeit_recht_evenredig_met_de_benen(self):
        een = RAW_GOUD.marge_voor(0.01, 4000.0, benen=1)
        tien = RAW_GOUD.marge_voor(0.01, 4000.0, benen=10)
        assert tien == pytest.approx(een * 10)

    def test_zonder_posities_is_de_margin_level_oneindig(self):
        assert RAW_GOUD.margin_level(59.0, 0.0) == math.inf
        assert not RAW_GOUD.vliegt_eruit(59.0, 0.0)


class TestDeStopOutGaatVoorBalansNul:
    def test_je_vliegt_eruit_terwijl_er_nog_geld_op_staat(self):
        # Eigen vermogen 30, marge 80 -> level 37,5% en dat is onder de 50%.
        assert RAW_GOUD.vliegt_eruit(30.0, 80.0)
        assert RAW_GOUD.margin_level(30.0, 80.0) == pytest.approx(0.375)

    def test_de_oude_regel_laat_diezelfde_rekening_gewoon_doorlopen(self):
        # Dit is exact wat de vorige meting deed: alleen balans <= 0.
        assert not ZONDER_STOPOUT.vliegt_eruit(30.0, 80.0)

    def test_precies_op_de_grens_blijf_je_staan(self):
        # 50% van 80 is 40. Op 40 nog net niet, eronder wel.
        assert not RAW_GOUD.vliegt_eruit(40.0, 80.0)
        assert RAW_GOUD.vliegt_eruit(39.99, 80.0)

    def test_onder_de_margin_call_mag_er_niets_meer_bij(self):
        # Level 100% is de grens: daarop nog wel, eronder niet.
        assert RAW_GOUD.mag_bijopenen(80.0, 80.0)
        assert not RAW_GOUD.mag_bijopenen(79.0, 80.0)


class TestWatEenRekeningUithoudt:
    def test_benen_tellen_klopt_met_de_stap(self):
        # Nul punten beweging is al één been: de instap zelf.
        assert benen_bij_beweging(0.0, 0.5) == 1
        assert benen_bij_beweging(5.0, 0.5) == 11
        assert benen_bij_beweging(1.0, 0.5) == 3

    def test_het_zwevende_verlies_is_de_driehoek_en_niet_het_aantal_benen(self):
        # Vijf benen op 0, 0.5, 1.0, 1.5, 2.0 punten onder water:
        # 0 + 0,5 + 1 + 1,5 + 2 = 5 punten x 0,01 lot x 100 = EUR 5.
        n, zwevend, _, _ = toestand_na_beweging(
            1000.0, 2.0, stap=0.5, lot=0.01, prijs=4000.0,
            uitvoering=Uitvoering(spread=0.0, commissie_per_lot_per_kant=0.0))
        assert n == 5
        assert zwevend == pytest.approx(-5.0)

    def test_de_formule_en_de_zoeker_geven_hetzelfde_antwoord(self):
        """DE BELANGRIJKSTE TEST HIER.

        `balans_voor_beweging` keert de stop-out-regel om; `overleefde_beweging`
        loopt hem been voor been af. Twee wegen naar hetzelfde getal, en als ze
        uiteenlopen is een van beide fout.
        """
        for punten in (2.0, 5.0, 20.0, 50.0):
            nodig = balans_voor_beweging(
                punten, stap=0.5, lot=0.01, prijs=4000.0)
            # Met precies genoeg overleef je die beweging...
            gehaald = overleefde_beweging(
                nodig + 1e-6, stap=0.5, lot=0.01, prijs=4000.0)
            assert gehaald >= punten, (
                f"met EUR {nodig:.2f} haalt hij {gehaald} van de {punten} punten")
            # ...en met een euro minder niet meer.
            krapper = overleefde_beweging(
                nodig - 1.0, stap=0.5, lot=0.01, prijs=4000.0)
            assert krapper < punten, (
                f"EUR {nodig:.2f} was blijkbaar ruimer dan nodig voor {punten}")

    def test_op_negenenvijftig_euro_is_een_gewone_gouddag_dodelijk(self):
        """Geen backtest nodig: dit is rekenwerk.

        Een normale dagrange op goud is 40 tot 80 punten. Als EUR 59 met een
        halve-punt-ladder daar niet doorheen komt, dan is dat het antwoord op
        de vraag, en geen enkele gunstige datareeks verandert het.
        """
        gehaald = overleefde_beweging(59.16, stap=0.5, lot=0.01, prijs=4000.0)
        assert gehaald < 40.0, (
            "als dit ooit boven de 40 uitkomt is de marge-rekensom veranderd")

    def test_de_stopout_maakt_het_meetbaar_erger_dan_alleen_balans_nul(self):
        met = overleefde_beweging(59.16, stap=0.5, lot=0.01, prijs=4000.0)
        zonder = overleefde_beweging(59.16, stap=0.5, lot=0.01, prijs=4000.0,
                                     uitvoering=ZONDER_STOPOUT)
        assert met < zonder, (
            "de stop-out hoort de rekening eerder te doden dan balans nul")

    def test_meer_balans_houdt_meer_beweging_uit(self):
        vorig = -1.0
        for balans in (59.16, 500.0, 2_000.0, 10_000.0):
            nu = overleefde_beweging(balans, stap=0.5, lot=0.01, prijs=4000.0)
            assert nu > vorig
            vorig = nu


class TestDeKosten:
    def test_een_been_kost_de_spread_een_keer_heen_en_terug(self):
        # 0,14 punt x 0,01 lot x 100 ounce = EUR 0,14 per been.
        assert RAW_GOUD.kosten_per_been(0.01) == pytest.approx(0.14)

    def test_commissie_en_swap_zijn_nul_op_goud(self):
        assert RAW_GOUD.commissie_per_lot_per_kant == 0.0
        assert RAW_GOUD.swap_kosten(0.01, nachten=730, benen=50) == 0.0

    def test_slippage_staat_los_van_de_spread(self):
        # Een marktexit van tien benen slipt tien keer.
        assert RAW_GOUD.slippage_kosten(0.01, benen=10) == pytest.approx(
            RAW_GOUD.slippage * 0.01 * CONTRACT * 10)


class TestSlippageUitDeEigenFills:
    @staticmethod
    def _db(tmp_path, waarden):
        pad = tmp_path / "journal.db"
        db = sqlite3.connect(pad)
        db.execute("CREATE TABLE order_attempts (symbol TEXT, ok INTEGER, "
                   "slippage_pips REAL)")
        db.executemany("INSERT INTO order_attempts VALUES (?, 1, ?)",
                       [("XAUUSD", w) for w in waarden])
        db.commit()
        db.close()
        return pad

    def test_te_weinig_fills_geeft_geen_getal(self, tmp_path):
        assert slippage_uit_db(self._db(tmp_path, [1.0, 2.0, 3.0])) is None

    def test_de_mediaan_komt_in_koerspunten_terug(self, tmp_path):
        # Mediaan van 1,2,3,4,5 pips = 3 pips; punt = 0,01 -> 0,03 koerspunten.
        pad = self._db(tmp_path, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert slippage_uit_db(pad) == pytest.approx(0.03)

    def test_slippage_wordt_op_grootte_gemeten_en_niet_op_teken(self, tmp_path):
        pad = self._db(tmp_path, [-3.0, -3.0, -3.0, -3.0, -3.0])
        assert slippage_uit_db(pad) == pytest.approx(0.03)

    def test_een_ontbrekende_database_is_geen_crash(self, tmp_path):
        assert slippage_uit_db(tmp_path / "bestaat-niet.db") is None

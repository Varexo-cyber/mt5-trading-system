"""Sectie 22: de consistentietoets zelf moet kloppen.

Een gereedschap dat een trackrecord beoordeelt, moet zelf niet de fout maken
waar het op controleert. Deze tests bewaken vier dingen:

  * DE REKENSOM KLOPT OP EEN BEKEND GEVAL. Aurix is met de hand nagerekend en
    komt op twee decimalen uit; wijkt de module daarvan af, dan rekent hij fout.
  * DE ONAFHANKELIJKHEIDSTOETS KAN BEIDE KANTEN OP. Een toets die altijd
    "weerlegd" zegt keurt niets, en een die dat nooit zegt ook niet.
  * DE VORM VAN DE WIN/VERLIES-VERHOUDING IS GEEN OORDEEL. Dit is de fout die
    ik zelf maakte: een verlies van 5x de winst noemde ik "de handtekening van
    een grid". Dat is het niet -- een wijde stop met een klein doel geeft
    exact dezelfde vorm. Alleen trefkans MIN quitte-punt telt.
  * DE KOP EN HET RENDEMENT WORDEN UIT ELKAAR GEHOUDEN.
"""

from __future__ import annotations

import pytest

from scripts.section_twentytwo_claim_audit import (
    AURIX,
    ELIO,
    Trackrecord,
    audit,
    is_de_kop_een_rendement,
    is_de_trefkans_versprongen,
    is_er_voorsprong,
    rekent_het_op,
    zijn_de_trades_onafhankelijk,
)


class TestDeRekensomKloptOpEenBekendGeval:
    def test_aurix_komt_uit_op_zijn_eigen_gepubliceerde_cijfers(self):
        """Met de hand nagerekend: PF 1,65 en verwachting +$7,03 tegen de
        opgegeven +$7,04."""

        uit = rekent_het_op(AURIX)

        assert uit["profit_factor"] == pytest.approx(1.65, abs=0.01)
        assert uit["verwachting"] == pytest.approx(7.04, abs=0.02)
        assert not uit["pf_wijkt_af"]
        assert not uit["verwachting_wijkt_af"]

    def test_elio_ook(self):
        uit = rekent_het_op(ELIO)

        assert uit["profit_factor"] == pytest.approx(3.99, abs=0.01)
        assert uit["verwachting"] == pytest.approx(1069.51, abs=1.0)
        assert uit["totale_winst"] == pytest.approx(2_978_582, rel=0.001)

    def test_een_verzonnen_cijfer_wordt_gezien(self):
        """Als `pf_wijkt_af` nooit True kan worden, controleert hij niets."""

        vals = Trackrecord(naam="x", trades=100, winnaars=50,
                           gem_winst=1.0, gem_verlies=1.0, opgegeven_pf=9.9)
        assert rekent_het_op(vals)["pf_wijkt_af"]


class TestDeVormIsGeenOordeel:
    def test_een_wijde_stop_met_klein_doel_heet_geen_grid(self):
        """DE FOUT DIE IK ZELF MAAKTE. Een gemiddeld verlies van vijf keer de
        gemiddelde winst ziet eruit als een grid en is dat niet per se. Deze
        module mag daar dus geen oordeel aan hangen -- alleen aan de
        voorsprong."""

        uit = is_er_voorsprong(ELIO)

        assert uit["verhouding"] == pytest.approx(4.78, abs=0.05)
        assert uit["breakeven_trefkans"] == pytest.approx(0.827, abs=0.005)
        # 95% raak tegen 82,7% quitte: er IS voorsprong, ondanks de vorm.
        assert uit["voorsprong"] > 0

    def test_dezelfde_vorm_zonder_voorsprong_wordt_wel_afgekeurd(self):
        """Anders zou de toets de vorm goedkeuren in plaats van de voorsprong."""

        zwak = Trackrecord(naam="x", trades=1000, winnaars=800,
                           gem_winst=1.0, gem_verlies=5.0)
        uit = is_er_voorsprong(zwak)

        assert uit["breakeven_trefkans"] == pytest.approx(5 / 6, abs=0.001)
        assert uit["voorsprong"] < 0, "80% raak bij 1:5 verliest geld"


class TestDeOnafhankelijkheidstoetsKanBeideKantenOp:
    def test_vierhonderd_trades_zonder_verliezer_weerlegt_de_bewering(self):
        """DE TOETS DIE BESLIST. Bij 95% trefkans is 416 op 416 ongeveer
        1 op 1,8 miljard."""

        uit = zijn_de_trades_onafhankelijk(ELIO)

        assert uit["toetsbaar"]
        assert uit["kans"] < 1e-8
        assert uit["weerlegt_onafhankelijkheid"]

    def test_een_korte_foutloze_reeks_weerlegt_niets(self):
        """Tien op tien bij 95% is 1 op 1,7 -- daar is niets bijzonders aan.
        Een toets die daar al aan slaat, roept overal fraude."""

        kort = Trackrecord(naam="x", trades=2785, winnaars=2646,
                           gem_winst=1.0, gem_verlies=1.0, reeks=(10, 10))
        uit = zijn_de_trades_onafhankelijk(kort)

        assert uit["toetsbaar"]
        assert not uit["weerlegt_onafhankelijkheid"]

    def test_een_reeks_met_verliezers_is_niet_tegenstrijdig(self):
        met_verlies = Trackrecord(naam="x", trades=1000, winnaars=950,
                                  gem_winst=1.0, gem_verlies=1.0, reeks=(400, 380))
        uit = zijn_de_trades_onafhankelijk(met_verlies)

        assert not uit["toetsbaar"]
        assert uit["verliezers_in_reeks"] == 20

    def test_zonder_reeks_wordt_er_niets_beweerd(self):
        """Aurix gaf geen foutloze reeks; dan hoort hier geen oordeel te komen."""

        assert zijn_de_trades_onafhankelijk(AURIX) is None


class TestDeKopIsGeenRendement:
    def test_de_abs_gain_wordt_nagerekend_en_niet_overgenomen(self):
        uit = is_de_kop_een_rendement(AURIX)

        assert uit["rendement_op_inleg"] == pytest.approx(0.4945, abs=0.001)
        assert uit["abs_gain_klopt"]
        # En de kop is tien keer zo groot als wat er verdiend is.
        assert uit["factor"] > 9

    def test_bij_elio_ook(self):
        uit = is_de_kop_een_rendement(ELIO)

        assert uit["rendement_op_inleg"] == pytest.approx(14.18, abs=0.05)
        assert uit["abs_gain_klopt"]


class TestDeTrefkansdrift:
    def test_de_sprong_van_79_naar_99_wordt_gezien(self):
        uit = is_de_trefkans_versprongen(ELIO)

        assert uit["eerder_trefkans"] == pytest.approx(0.793, abs=0.01)
        assert uit["nu_trefkans"] == pytest.approx(0.99, abs=0.01)
        assert uit["sprong"] > 0.15


class TestHetRapportZegtHetHardop:
    def test_de_weerlegging_staat_er_met_zoveel_woorden(self):
        tekst = audit(ELIO)

        assert "WEERLEGD" in tekst
        assert "416 trades op rij" in tekst

    def test_aurix_wordt_niet_weerlegd(self):
        """Een gereedschap dat alles afkeurt, keurt niets."""

        tekst = audit(AURIX)

        assert "WEERLEGD" not in tekst

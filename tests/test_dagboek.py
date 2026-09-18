"""Het dagboek moet de trade laten zien die je NIET wil zien.

Gevraagd werd: "hey hier zou ik in hebben gestaan, hey jammer man, verloren."
Dat is een ander soort uitvoer dan een totaal, en er zijn drie manieren om het
onbruikbaar te maken:

  * DE ERGSTE TRADES VERSTOPPEN. Een dagboek dat met de mooiste begint is
    reclame. Waar je van leert is de reeks die misging.
  * HET VERLOOP WEGLATEN. Een trade die +1 EUR opleverde nadat hij 40 EUR
    onder water stond, is iets heel anders dan een die rechtstreeks naar +1
    liep -- en in het totaal zien die er identiek uit.
  * DE BALANS WEGLATEN. Dezelfde trade is op EUR 59 dodelijk en op EUR 5.000
    een schrammetje.
"""

from __future__ import annotations

from datetime import UTC

import pandas as pd
import pytest

from scripts.dagboek import Regel, toon


def _regel(euro, zwevend=0.0, balans=100.0, uur=9):
    return Regel(moment=pd.Timestamp(f"2026-01-05 {uur:02d}:00", tz=UTC),
                 sectie="S20", reden="KOOP 4000.00   M1+M5 omhoog",
                 verloop="3 benen   diepst -5.00 EUR   12 min",
                 euro=euro, zwevend=zwevend, balans_na=balans, lot=0.01)


class TestDeErgsteStaanBovenaan:
    def test_de_grootste_verliezer_staat_in_het_eerste_blok(self):
        """Niet ergens op regel 40.000 tussen de winnaars."""

        regels = [_regel(1.0) for _ in range(50)] + [_regel(-99.0)]
        tekst = "\n".join(toon(regels, naam="test", balans=100.0))

        kop = tekst.index("DE TIEN ERGSTE")
        besten = tekst.index("DE TIEN BESTE")
        assert kop < besten, "de beste trades staan boven de ergste"
        assert "-99.000" in tekst[kop:besten], (
            "de grootste verliezer staat niet in het ergste-blok"
        )

    def test_een_winnaar_heet_geen_jammer_en_omgekeerd(self):
        assert "WINST" in str(_regel(5.0))
        assert "JAMMER" in str(_regel(-5.0))
        assert "vlak" in str(_regel(0.0))


class TestHetVerloopStaatErbij:
    def test_de_reden_en_het_verloop_staan_in_elke_regel(self):
        """Zonder de reden kun je achteraf niet zien waarom een reeks misging."""

        tekst = str(_regel(1.0))
        assert "M1+M5 omhoog" in tekst, "de reden ontbreekt"
        assert "diepst" in tekst, "het verloop ontbreekt"
        assert "balans" in tekst, "de balans ontbreekt"

    def test_de_diepste_zwevende_stand_wordt_apart_genoemd(self):
        """Het getal dat bepaalt of je die trade in het echt had uitgezeten."""

        regels = [_regel(1.0, zwevend=-2.0), _regel(1.0, zwevend=-88.0, uur=14)]
        tekst = "\n".join(toon(regels, naam="test", balans=100.0))

        assert "-88.00" in tekst
        assert "14:00" in tekst, "hij noemt niet WANNEER het het diepst stond"


class TestKleineBedragenVerdwijnenNiet:
    def test_bij_het_minimumlot_zijn_centen_nog_zichtbaar(self):
        """Met twee decimalen staat er bij 0,01 lot overal 0,00 en lijkt elke
        trade hetzelfde. Dan is het dagboek onleesbaar precies waar het
        interessant wordt."""

        tekst = str(_regel(0.004))
        assert "0.004" in tekst, "centen worden weggerond"

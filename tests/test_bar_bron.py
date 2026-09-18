"""Waar de bars vandaan komen, en de vier fouten die ik daarin zelf maakte.

WAAROM DIT BESTAAT. Sectie 20 en 21 waren geschreven, getest en gecommit --
en zouden bij de eerste echte run op de Windows-machine meteen zijn omgevallen.
De rekenkern was getest op verzonnen bars; de AANSLUITING op MT5 door niets.

De vier fouten, alle vier stil tot het moment dat je hem draait:

  * `MT5Connector(credentials, settings)` in plaats van
    `MT5Connector(settings.mt5, credentials)` -- argumenten omgedraaid.
  * `fetch_mt5_history(..., args.days)` terwijl die functie een START en een
    EIND als datetime wil. Een int waar een datum hoort.
  * het symbool ging rechtstreeks door in plaats van via
    `settings.instruments.broker_symbol()`. Ik had de overlay net toegevoegd
    en hem vervolgens niet gebruikt, dus "XAUUSD" naar een broker die het
    "XAUUSD.i" noemt.
  * drie tijdframes apart ophalen, elk met eigen gaten en een eigen laatste
    bar -- dan kijkt de stapel op het ene tijdframe naar een kaars die op het
    andere nog niet bestaat.

Een testsuite die de rekenkern bewaakt en de aansluiting niet, geeft groen
licht aan iets dat nog nooit gedraaid heeft.
"""

from __future__ import annotations

import ast
from datetime import UTC
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


def _met_mt5() -> list[Path]:
    return [p for p in SCRIPTS if "fetch_mt5_history(" in p.read_text()]


def _met_eigen_symboolnaam() -> list[Path]:
    """Scripts die een KANONIEKE symboolnaam in hun eigen code hebben staan.

    NIET ELK SCRIPT HEEFT broker_symbol NODIG. `fetch_history.py` haalt zijn
    universum bij de connector op en krijgt de namen dus al met suffix binnen;
    daar zou de eis onzin zijn. De fout die hier bewaakt wordt, treedt op zodra
    een script ZELF "XAUUSD" opschrijft en dat rechtstreeks doorgeeft.
    """

    uit = []
    for pad in _met_mt5():
        tekst = pad.read_text()
        if 'default="XAUUSD"' in tekst or 'default="XAUJPY"' in tekst:
            uit.append(pad)
    return uit


class TestDeAansluitingKlopt:
    def test_er_zijn_scripts_om_te_controleren(self):
        assert _met_mt5(), "deze test is blind geworden"

    def test_er_zijn_scripts_met_een_eigen_symboolnaam(self):
        assert _met_eigen_symboolnaam(), "deze controle is blind geworden"

    @pytest.mark.parametrize("pad", _met_eigen_symboolnaam(), ids=lambda p: p.name)
    def test_het_symbool_gaat_door_de_overlay(self, pad):
        """DE OVERLAY LADEN EN NIET GEBRUIKEN is geen halve fix maar geen fix.
        De suffix zit in `broker_symbol`, en zonder dat levert je vraag stil
        niets op."""

        tekst = pad.read_text()
        assert "broker_symbol(" in tekst, (
            f"{pad.name} vraagt een symbool op zonder de brokersuffix"
        )

    @pytest.mark.parametrize("pad", _met_mt5(), ids=lambda p: p.name)
    def test_fetch_krijgt_een_start_en_een_eind_en_geen_aantal_dagen(self, pad):
        """`fetch_mt5_history` wil datetimes. Een int glijdt er zonder
        typefout doorheen en valt pas om in de terminal."""

        boom = ast.parse(pad.read_text())
        aanroepen = [
            n for n in ast.walk(boom)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "fetch_mt5_history"
        ]
        assert aanroepen, f"{pad.name} noemt fetch_mt5_history maar roept hem niet aan"
        for aanroep in aanroepen:
            assert len(aanroep.args) >= 5, (
                f"{pad.name}: fetch_mt5_history krijgt {len(aanroep.args)} argumenten; "
                "market, symbol, timeframe, START en EIND zijn er vijf"
            )
            # Het vierde argument mag geen kaal getal of `args.days` zijn.
            vierde = aanroep.args[3]
            assert not isinstance(vierde, ast.Constant), (
                f"{pad.name}: een constante waar een startdatum hoort"
            )
            bron = ast.unparse(vierde)
            assert "days" not in bron or "timedelta" in bron, (
                f"{pad.name}: '{bron}' ziet eruit als een aantal dagen, niet als "
                "een startmoment"
            )

    @pytest.mark.parametrize("pad", _met_mt5(), ids=lambda p: p.name)
    def test_de_connector_krijgt_de_config_als_eerste(self, pad):
        """`MT5Connector(config, credentials)`. Omgedraaid geeft het pas een
        fout als de terminal antwoordt."""

        boom = ast.parse(pad.read_text())
        for aanroep in ast.walk(boom):
            if not (isinstance(aanroep, ast.Call)
                    and isinstance(aanroep.func, ast.Name)
                    and aanroep.func.id == "MT5Connector"):
                continue
            assert aanroep.args, f"{pad.name}: MT5Connector zonder config"
            eerste = ast.unparse(aanroep.args[0])
            assert "settings" in eerste or "config" in eerste, (
                f"{pad.name}: MT5Connector krijgt '{eerste}' als eerste argument; "
                "dat hoort de MT5-config te zijn"
            )


class TestDeCsvWegWerktEcht:
    """DE BRUG ZELF. Zonder deze test is "je kunt hem ook met --csv draaien"
    een bewering."""

    def _schrijf(self, tmp_path: Path) -> Path:
        import numpy as np

        # ECHTE OHLC: open is de VORIGE close, niet dezelfde prijs.
        #
        # De eerste versie zette open == close op elke bar. Dan is `close >
        # open` nooit waar, staat geen enkele M1-bar ooit bullish en vindt
        # sectie 20 geen enkele mand -- op verzonnen bars die er verder prima
        # uitzien. Een fixture die de instapvoorwaarde onmogelijk maakt, test
        # de code niet maar zichzelf.
        rng = np.random.default_rng(4)
        index = pd.date_range("2026-03-02 00:00", periods=3000, freq="1min", tz=UTC)
        prijs = 4000 + np.cumsum(rng.normal(0, 0.4, len(index)))
        opens = np.concatenate([[prijs[0]], prijs[:-1]])
        frame = pd.DataFrame(
            {
                "open": opens,
                "high": np.maximum(opens, prijs) + 0.2,
                "low": np.minimum(opens, prijs) - 0.2,
                "close": prijs,
            },
            index=index,
        )
        pad = tmp_path / "bars.csv"
        frame.to_csv(pad, index_label="time")
        return pad

    def test_een_uitgevoerde_csv_wordt_teruggelezen_met_tijdzone(self, tmp_path):
        from scripts.section_twenty_pullback_ladder import _lees_csv

        frame = _lees_csv(str(self._schrijf(tmp_path)))

        assert frame.index.tz is not None, "de tijdzone is kwijt"
        assert list(frame.columns[:4]) == ["open", "high", "low", "close"]
        assert frame.index.is_monotonic_increasing

    def test_een_csv_zonder_de_juiste_kolommen_wordt_geweigerd(self, tmp_path):
        """Stil doorgaan met een half bestand meet iets anders dan je denkt."""

        from scripts.section_twenty_pullback_ladder import _lees_csv

        pad = tmp_path / "kaal.csv"
        pd.DataFrame(
            {"close": [1.0, 2.0]},
            index=pd.date_range("2026-03-02", periods=2, freq="1min", tz=UTC),
        ).to_csv(pad, index_label="time")

        with pytest.raises(SystemExit, match="mist kolommen"):
            _lees_csv(str(pad))

    def test_de_hogere_klokken_komen_uit_dezelfde_m1(self, tmp_path):
        """EEN BRON IS BETER DAN DRIE. Losse reeksen hebben eigen gaten en een
        eigen laatste bar, en dan kijkt de stapel op M15 naar een kaars die op
        M1 nog niet bestaat."""

        from scripts.section_twenty_pullback_ladder import _hersample, _lees_csv

        m1 = _lees_csv(str(self._schrijf(tmp_path)))
        m15 = _hersample(m1, "15min")

        assert len(m15) == pytest.approx(len(m1) / 15, rel=0.05)
        assert m15.index[0] >= m1.index[0]
        assert m15.index[-1] <= m1.index[-1]
        # En de OHLC klopt echt, niet alleen de lengte.
        eerste = m1.iloc[:15]
        assert m15.iloc[0]["high"] == pytest.approx(eerste["high"].max())
        assert m15.iloc[0]["low"] == pytest.approx(eerste["low"].min())
        assert m15.iloc[0]["open"] == pytest.approx(eerste.iloc[0]["open"])
        assert m15.iloc[0]["close"] == pytest.approx(eerste.iloc[-1]["close"])

    def test_beide_secties_draaien_van_begin_tot_eind_op_die_csv(self, tmp_path):
        """De echte proef: geen MT5, wel een uitslag."""

        from scripts.section_twenty_pullback_ladder import (
            Instelling as L, _hersample, _lees_csv, simuleer,
        )
        from scripts.section_twentyone_straddle import Instelling as S, draai

        m1 = _lees_csv(str(self._schrijf(tmp_path)))
        stapels = {n: _hersample(m1, r)
                   for n, r in (("M1", "1min"), ("M5", "5min"), ("M15", "15min"))}

        manden, _ = simuleer(m1, stapels, instelling=L(), balans=400.0)
        uit = draai(m1, instelling=S(), om_de=60)

        assert manden, "sectie 20 vindt geen enkele mand op 3000 bars"
        assert uit["straddle"], "sectie 21 vindt geen enkele cyclus"
        assert len(uit["straddle"]) == len(uit["een_richting"])

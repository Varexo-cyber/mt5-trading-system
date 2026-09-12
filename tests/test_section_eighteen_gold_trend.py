"""Sectie achttien: meedoen met de trend, en elke stilte noemt zichzelf."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from analysis import SectionEighteenGoldTrendD1
from config.loader import load_settings
from config.schema import SectionEighteenGoldTrendConfig
from core.types import MarketContext, Series, Timeframe


def _context(prices, symbol: str = "XAUUSD", noise: float = 18.0) -> MarketContext:
    index = pd.date_range("2024-01-01", periods=len(prices), freq="1D", tz="UTC")
    close = np.asarray(prices, dtype=float)
    rng = np.random.default_rng(7)
    high = close + np.abs(rng.normal(0.0, noise, len(close)))
    low = close - np.abs(rng.normal(0.0, noise, len(close)))
    frame = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 100.0, "spread": 1.0},
        index=index,
    )
    series = Series(symbol=symbol, timeframe=Timeframe.D1, df=frame, fetched_at=datetime.now(UTC))
    return MarketContext(
        symbol=symbol, now=index[-1].to_pydatetime(), series={Timeframe.D1: series}
    )


def _rising(days: int = 200, drift: float = 1.6, seed: int = 4) -> np.ndarray:
    """Goud met drift en dagelijkse ruis.

    SEED 4 EN NIET 11, en dat is geen cosmetiek: seed 11 eindigt toevallig ONDER
    zijn eigen 50-daags gemiddelde, dus de sectie gaf daar terecht geen signaal
    en elke stoptest kreeg `None` terug. Een fixture die niet doet wat de test
    nodig heeft geeft een rood dat niets over de code zegt.
    """

    rng = np.random.default_rng(seed)
    return 2000.0 + np.cumsum(rng.normal(drift, 22.0, days))


def _section(**overrides) -> SectionEighteenGoldTrendD1:
    return SectionEighteenGoldTrendD1(SectionEighteenGoldTrendConfig(enabled=True, **overrides))


class TestEachSilenceNamesItself:
    """De meest herhaalde fout in dit project, en de duurste.

    Sectie zes had drie verschillende storingen die dezelfde zin printten, en
    dat kostte drie dagen zoeken naar iets dat er niet was. Elke reden om niets
    te doen moet van de andere te onderscheiden zijn in de tekst zelf.
    """

    def test_the_wrong_market_says_which_market_it_wants(self) -> None:
        signal = _section().analyze(_context(_rising(), symbol="NDX100"))

        assert signal.score == 0.0
        assert "XAUUSD" in signal.reasoning

    def test_too_few_bars_says_how_many_it_has_and_needs(self) -> None:
        signal = _section(trend_bars=50, atr_period=14).analyze(_context(_rising(days=20)))

        assert "20" in signal.reasoning
        assert "65" in signal.reasoning

    def test_price_under_the_average_says_it_does_not_go_short(self) -> None:
        signal = _section().analyze(_context(np.linspace(2600.0, 2000.0, 150)))

        assert signal.score == 0.0
        assert "short" in signal.reasoning

    def test_a_stretched_price_says_how_far_and_what_the_limit_is(self) -> None:
        # Een rechte lijn heeft nauwelijks ATR, dus staat het slot ver boven het
        # gemiddelde gemeten in ATR -- precies waar deze poort voor is.
        signal = _section().analyze(_context(np.linspace(2000.0, 2600.0, 150), noise=0.0))

        assert signal.score == 0.0
        assert "ATR boven het gemiddelde" in signal.reasoning

    def test_the_reasons_are_all_different(self) -> None:
        section = _section()
        reasons = {
            section.analyze(_context(_rising(), symbol="NDX100")).reasoning,
            section.analyze(_context(_rising(days=20))).reasoning,
            section.analyze(_context(np.linspace(2600.0, 2000.0, 150))).reasoning,
            section.analyze(_context(np.linspace(2000.0, 2600.0, 150), noise=0.0)).reasoning,
        }

        assert len(reasons) == 4, f"twee storingen delen een zin: {reasons}"


class TestWhatItIsBuiltToDo:
    def test_it_participates_instead_of_scalping(self) -> None:
        """DIT IS DE HELE REDEN DAT DEZE SECTIE BESTAAT.

        Goud deed +1250,93 R in het holdoutjaar en sectie zes ving +4,21 R. Een
        deelnamevoertuig hoort in een aanhoudende trend het GROOTSTE DEEL van de
        dagen long te staan, niet een handvol. De grens hier is bewust ruim: wat
        vastligt is "meedoen", niet een precies percentage.
        """

        prices = _rising(days=200)
        section = _section()
        long_days = sum(
            1 for day in range(70, len(prices)) if section.analyze(_context(prices[:day])).score > 0
        )

        assert long_days / (len(prices) - 70) > 0.5, "dit doet niet mee, dit scalpt"

    def test_it_never_goes_short(self) -> None:
        """Long only is ontwerp, geen voorkeur. Short is een tweede hypothese."""

        section = _section()
        for prices in (_rising(), np.linspace(2600.0, 2000.0, 200), _rising(drift=-1.6, seed=4)):
            assert section.analyze(_context(prices)).score >= 0.0

    def test_the_stop_is_wide_and_below_the_entry(self) -> None:
        """Een trend van weken overleeft geen stop van een halve ATR.

        Die wordt uitgeschud op ruis, en dan betaal je de spread opnieuw voor
        dezelfde these -- exact het kostenprobleem waar de rest van dit systeem
        aan bezweek.
        """

        signal = _section(stop_atr=2.0).analyze(_context(_rising()))

        assert signal.score > 0.0
        assert signal.invalidation_price is not None
        atr = signal.details["daily_atr"]
        entry = _rising()[-1]
        assert signal.invalidation_price < entry
        assert entry - signal.invalidation_price == pytest.approx(2.0 * atr, rel=1e-6)

    def test_a_wider_stop_setting_actually_widens_the_stop(self) -> None:
        """Anders is de parameter een sierknop, en die zijn er hier al genoeg."""

        prices = _rising()
        smal = _section(stop_atr=2.0).analyze(_context(prices))
        breed = _section(stop_atr=3.0).analyze(_context(prices))

        assert breed.invalidation_price < smal.invalidation_price


class TestItIsWiredEverywhereItNeedsToBe:
    """SECTIE ELF GING LIVE ZONDER HERKOMST en dat is dezelfde fout in het klein.

    Een sectie die in de allowlist, de gewichten en de guards staat maar wiens
    TRADES niet toewijsbaar zijn, is een sectie waarvan je achteraf niet kunt
    zeggen wat hij gedaan heeft.
    """

    def test_it_has_its_own_trade_origin(self) -> None:
        from core.trade_origin import origin_for_setup_family

        origin = origin_for_setup_family("section_eighteen_gold_trend_d1")

        assert origin is not None
        assert origin.comment == "JARVIS-S18-AU-D1"

    def test_the_runner_builds_it(self) -> None:
        from runner.service import build_analysis_modules

        settings = load_settings(overlay="config/eightcap.yaml", env_overrides=False)
        names = {getattr(m, "name", "") for m in build_analysis_modules(settings)}

        assert "section_eighteen_gold_trend_d1" in names

    def test_the_measurement_bench_knows_it(self) -> None:
        """Een sectie die niet in het boek staat wordt ook niet gemeten.

        Zo bleven drie M1-detectors maandenlang ongetoetst: ze draaiden wel,
        maar `dryrun-live.cmd` kende hun naam niet.
        """

        source = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "scripts"
            / "dry_run_sections.py"
        ).read_text(encoding="utf-8")

        assert source.count("section_eighteen_gold_trend_d1") >= 2

    def test_it_is_shadow_and_carries_no_weight(self) -> None:
        """Gewicht dragen en toestemming hebben is EEN toestand met twee kanten.

        Gewicht zonder toestemming is misleidend; toestemming zonder gewicht is
        erger, want de engine test `if weight > 0` en zo'n sectie mag handelen
        zonder ergens mee te tellen.
        """

        settings = load_settings(overlay="config/eightcap.yaml", env_overrides=False)
        confluence = settings.analysis.confluence
        live = "section_eighteen_gold_trend_d1" in confluence.live_enabled_modules

        assert not live, "deze sectie heeft geen holdout en mag geen geld uitgeven"
        assert confluence.weights["section_eighteen_gold_trend_d1"] == 0.0

    def test_the_hypothesis_was_written_before_the_code(self) -> None:
        """Het eerste document in dit project dat in die volgorde geschreven is.

        `TEMPLATE.md` stelt al twee jaar de juiste lat en over 70 documenten
        staat er geen enkele afgevinkte regel. Deze sectie bestaat alleen als
        de voorspellingen en de faalvoorwaarden er VOORAF staan.
        """

        doc = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "docs"
            / "hypotheses"
            / "section_eighteen_gold_trend_d1.md"
        ).read_text(encoding="utf-8")

        for heading in ("Voorspellingen, vooraf vastgelegd", "Waar dit MOET falen"):
            assert heading in doc, heading
        # De faalvoorwaarde die "het is gewoon de goudtrend" afvangt.
        assert "2,12" in doc
        # En het aantal geprobeerde combinaties, want zonder dat getal is er
        # geen eerlijke lat te berekenen.
        assert "maximaal 6" in doc


def test_the_launcher_defaults_to_development_and_makes_the_holdout_deliberate() -> None:
    """De holdout mag precies EEN keer bekeken worden.

    Standaard op de holdout draaien maakt van "een keer kijken" iets dat je per
    ongeluk doet, en twee keer kijken en de gunstigste houden is exact de fout
    die sectie vijf drie keer van teken deed wisselen.
    """

    from pathlib import Path

    launcher = (Path(__file__).resolve().parents[1] / "sectie18.cmd").read_text(encoding="utf-8")

    assert "set VENSTER=--days 360" in launcher, "standaard is de ontwikkelperiode"
    assert "2024-09-01" in launcher and "2025-08-31" in launcher
    # De waarschuwing zelf moet er staan, niet alleen de optie.
    flat = " ".join(launcher.split())
    assert "precies EEN keer" in flat
    assert "section_eighteen_gold_trend_d1" in launcher

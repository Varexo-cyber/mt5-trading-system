"""Section three, found by the search that ran after section two shipped.

The first search produced only variants of the retest. `retest_slow` looked
like a second strategy and shared 47.5% of its trades with `impulse_retest` --
the same section wearing a hat. So round three was restricted to mechanisms
that can fire when a retest cannot, and only this one lived:

    FX M30, 31,376 trades with every section-two bar removed
    hit 62.1%   net +0.164R   train +0.168 / holdout +0.161
    every year positive 2012-2022; 114 of 114 months positive

What these tests guard is the set of things that would make the live module a
different trade from the measured one, each worth a specific amount:

    impulse >= 1.5 ATR on the BODY   at 1.0 ATR the edge halves to +0.083R
    tolerance 0.25 into the block    at 0.50 it drops to +0.146R
    the 1:1 target                   at 1.5R it drops to +0.099R
    stop 1.0 ATR                     below 0.8 the confluence floor widens it
    it can trade alone               all 31,376 trades were taken unaccompanied
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.order_block import OrderBlock
from config.schema import OrderBlockConfig
from core.types import MarketContext, Series, Tick, Timeframe


def _frame(opens, closes, half=0.1) -> pd.DataFrame:
    opens = np.asarray(opens, dtype=float)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + half,
            "low": np.minimum(opens, closes) - half,
            "close": closes,
            "tick_volume": 100,
            "spread": 2,
        },
        index=pd.date_range("2026-08-28", periods=len(closes), freq="30min", tz="UTC"),
    )


def _context(frame: pd.DataFrame, symbol: str = "TEST") -> MarketContext:
    last = float(frame["close"].iloc[-1])
    when = frame.index[-1].to_pydatetime()
    return MarketContext(
        symbol,
        when,
        {Timeframe.M30: Series(symbol, Timeframe.M30, frame, when)},
        Tick(symbol, when, bid=last - 0.001, ask=last + 0.001),
    )


def _impulse_and_return(impulse_atr: float, back_to_atr: float, *, down: bool = False):
    """A quiet base, one red candle, one big impulse, then a return.

    `back_to_atr` is where price ends up relative to the block's edge, in ATR,
    so the fixture states distances in the unit the module works in rather
    than in prices that stop meaning anything if the noise changes.
    """
    sign = -1.0 if down else 1.0
    rng = np.random.default_rng(4)
    n = 120
    base_c = 100.0 + rng.normal(0, 0.08, n).cumsum() * 0.1
    base_o = base_c - rng.normal(0, 0.05, n)
    # The block: one candle the OTHER way, immediately before the impulse.
    block_open = float(base_c[-1])
    block_close = block_open - sign * 0.30
    # A rough ATR for this fixture: bar ranges are around 0.2-0.4.
    unit = 0.30
    imp_open = block_close
    imp_close = imp_open + sign * impulse_atr * unit
    edge = max(block_open, block_close) if not down else min(block_open, block_close)
    target = edge + sign * back_to_atr * unit
    tail_o = np.linspace(float(imp_close), float(target), 8)
    opens = np.concatenate([base_o, [block_open, imp_open], tail_o])
    closes = np.concatenate([base_c, [block_close, imp_close], tail_o])
    return _frame(opens, closes)


class TestTheImpulseIsTheFilter:
    def test_a_block_behind_a_real_impulse_is_a_signal(self) -> None:
        engine = OrderBlock(OrderBlockConfig())

        signal = engine.analyze(_context(_impulse_and_return(3.0, 0.0)))

        assert signal.score > 0, signal.reasoning
        assert "absorbed by" in signal.reasoning

    def test_a_block_behind_a_gentle_move_is_not(self) -> None:
        """THE MEASURED THRESHOLD. At 1.0 ATR the edge halves. A candle that
        small ran over nobody, so there is no one left in the zone."""
        engine = OrderBlock(OrderBlockConfig())

        signal = engine.analyze(_context(_impulse_and_return(0.8, 0.0)))

        assert signal.score == 0
        assert "no unspent block" in signal.reasoning

    def test_the_threshold_reads_the_body_not_the_range(self) -> None:
        """A 1.5 ATR range with a doji body is indecision, not absorption.
        Measuring the range would let every volatile bar qualify."""
        import inspect

        from analysis import order_block

        source = " ".join(inspect.getsource(order_block).split())

        assert "impulse = (c[i] - o[i]) / unit" in source

    def test_the_shipped_threshold_is_the_measured_one(self) -> None:
        assert OrderBlockConfig().minimum_impulse_atr == pytest.approx(1.5)

    def test_a_bigger_impulse_scores_higher(self) -> None:
        engine = OrderBlock(OrderBlockConfig())

        small = engine.analyze(_context(_impulse_and_return(1.6, 0.0)))
        large = engine.analyze(_context(_impulse_and_return(4.0, 0.0)))

        assert abs(large.score) > abs(small.score)


class TestItWaitsForTheZone:
    def test_price_still_above_the_zone_is_refused(self) -> None:
        engine = OrderBlock(OrderBlockConfig())

        signal = engine.analyze(_context(_impulse_and_return(3.0, 2.0)))

        assert signal.score == 0
        assert "waiting for the zone" in signal.reasoning

    def test_a_short_mirrors_it(self) -> None:
        engine = OrderBlock(OrderBlockConfig())

        signal = engine.analyze(_context(_impulse_and_return(3.0, 0.0, down=True)))

        assert signal.score < 0, signal.reasoning

    def test_a_zone_price_has_blown_through_is_not_offered(self) -> None:
        """Otherwise the zone trade becomes a knife catch: the block failed,
        price is past where the stop would have been, and the module is still
        pointing at it."""
        engine = OrderBlock(OrderBlockConfig())

        signal = engine.analyze(_context(_impulse_and_return(3.0, -4.0)))

        assert signal.score <= 0, signal.reasoning

    def test_the_tolerance_is_the_measured_one(self) -> None:
        """0.25 into the body, not 0.50: +0.172R against +0.146R. The zone is
        defended at its edge; reaching deeper is reaching past the orders."""
        assert OrderBlockConfig().zone_tolerance_atr == pytest.approx(0.25)


class TestWhatIsSentIsWhatWasMeasured:
    def test_the_stop_clears_the_confluence_floor_untouched(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

        assert settings.analysis.order_block.stop_atr >= settings.analysis.confluence.min_stop_atr

    def test_the_family_trades_one_to_one(self) -> None:
        """Measured here separately, not inherited from section two: 0.75R
        gives +0.147R, 1.0R gives +0.172R, 1.5R gives +0.099R."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        confluence = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence

        assert confluence.target_r_multiple_by_family.get("order_block") == pytest.approx(1.0)

    def test_gold_gets_a_stop_it_can_afford(self) -> None:
        """0.10 ATR of spread over a 1.0 ATR stop is 20% of R and the live
        gate refuses it. At 1.5 ATR it is 6.7%."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        gold = settings.analysis.order_block.stop_atr_by_symbol["XAUUSD"]

        assert gold > settings.analysis.order_block.stop_atr
        assert 0.10 / gold <= settings.analysis.confluence.max_spread_share_of_stop

    def test_the_per_symbol_stop_reaches_the_published_invalidation(self) -> None:
        """A per-symbol setting nothing reads is the defect this account keeps
        producing. Same bars, different symbol name, different stop."""
        engine = OrderBlock(OrderBlockConfig(stop_atr_by_symbol={"XAUUSD": 1.5}))
        frame = _impulse_and_return(3.0, 0.0)

        major = engine.analyze(_context(frame, "EURUSD"))
        gold = engine.analyze(_context(frame, "XAUUSD"))

        assert major.invalidation_price is not None
        assert gold.invalidation_price is not None
        assert gold.invalidation_price < major.invalidation_price


class TestItCanTradeAlone:
    def test_the_weakest_qualifying_setup_clears_both_gates(self) -> None:
        """All 31,376 measured trades were taken on this signal by itself. If
        the engine only lets it trade when something else agrees, what runs is
        not what was measured -- and it would look like the section simply not
        firing."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        confluence = settings.analysis.confluence
        config = settings.analysis.order_block

        assert config.base_confidence >= confluence.lone_module_minimum_confidence
        assert config.base_score * config.base_confidence >= confluence.score_threshold

    def test_a_real_signal_clears_them(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        confluence = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence
        engine = OrderBlock(OrderBlockConfig())

        signal = engine.analyze(_context(_impulse_and_return(1.6, 0.0)))

        assert signal.score != 0, signal.reasoning
        assert signal.confidence >= confluence.lone_module_minimum_confidence
        assert abs(signal.score) * signal.confidence >= confluence.score_threshold


class TestTheLiveWiring:
    def test_the_runner_builds_it(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from runner.service import build_analysis_modules

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

        assert "order_block" in {m.name for m in build_analysis_modules(settings)}

    def test_it_is_shadow_weighted_and_broken_out(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

        assert "order_block" not in settings.analysis.confluence.live_enabled_modules
        assert settings.analysis.confluence.weights.get("order_block", 0.0) > 0.0
        assert "order_block" in settings.risk.section_breakers

    def test_it_runs_on_a_different_clock_from_section_two(self) -> None:
        """M30 against section two's M15. Two sections crowding the same bars
        on the same symbol is one position's worth of edge taking two of four
        slots."""
        assert OrderBlockConfig().timeframe == "M30"
        from config.schema import ImpulseRetestConfig

        assert OrderBlockConfig().timeframe != ImpulseRetestConfig().timeframe

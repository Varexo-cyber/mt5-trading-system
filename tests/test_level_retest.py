"""The first module on this account with a number behind it.

Ten years of M15, eight instruments, 1.84 million bars. Parameters chosen on
2012-2017; the 2018-2021 holdout read once, for the sweep's best cell:

    best cell, train    49,700 trades   hit 44.7%   +53.6 sigma   E +0.340R
    best cell, holdout  30,507 trades   hit 44.9%   +42.9 sigma   E +0.347R
    SHIPPED cell        55,582 trades   hit 37.8%   +22.3 sigma   E +0.134R

The shipped cell carries a wider stop so `min_stop_atr` cannot silently widen
it, and has no holdout of its own -- what is out-of-sample here is the family.

Against the same breaks entered AT the break instead:

    donchian_break  190,505 signals  E -0.067R  -29 sigma

These tests do not re-measure that. They guard the things that would make the
shipped module a different strategy from the measured one -- and every one of
them is a way the edge silently goes to zero while the module keeps emitting
signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.level_retest import LevelRetest
from config.schema import LevelRetestConfig
from core.types import MarketContext, Series, Tick, Timeframe

BARS = 200


def _frame(closes: np.ndarray, spread: float = 0.5) -> pd.DataFrame:
    index = pd.date_range("2026-08-28", periods=len(closes), freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + spread,
            "low": closes - spread,
            "close": closes,
            "tick_volume": 100,
            "spread": 2,
        },
        index=index,
    )


def _context(frame: pd.DataFrame, price: float | None = None) -> MarketContext:
    last = float(frame["close"].iloc[-1]) if price is None else price
    when = frame.index[-1].to_pydatetime()
    return MarketContext(
        "TEST",
        when,
        {Timeframe.M15: Series("TEST", Timeframe.M15, frame, when)},
        Tick("TEST", when, bid=last - 0.01, ask=last + 0.01),
    )


HALF_BAR = 0.25


def _broke_and_returned(atr_above_level: float, *, down: bool = False) -> pd.DataFrame:
    """A range, a break out of it, then a return to a stated distance.

    The distance is given in ATR and converted here, because that is the unit
    the module works in and a hard-coded price would silently stop meaning what
    the test says the moment the fixture's volatility changes.

    The base oscillates rather than ramping. A ramp cannot break its own
    channel: with bars half a point tall, the twenty-bar high sits half a point
    above the last close, so any climb slower than that never clears it -- the
    first version of this fixture produced no breaks at all and read as a bug
    in the module.
    """
    sign = -1.0 if down else 1.0
    base = 100.0 + sign * 0.5 * np.sin(np.linspace(0, 12 * np.pi, 130))
    # One decisive bar out of the range, then a drift back toward the edge.
    breakout = np.array([100.0 + sign * 2.0])
    frame = _frame(np.concatenate([base, breakout]), spread=HALF_BAR)

    # The level is the range edge the break cleared, and the ATR is whatever
    # the frame actually has once that break bar is in it.
    level = (
        float(frame["high"].iloc[-21:-1].max())
        if not down
        else float(frame["low"].iloc[-21:-1].min())
    )
    previous = frame["close"].shift(1)
    spans = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    unit = float(spans.rolling(14).mean().iloc[-1])

    target = level + sign * atr_above_level * unit
    back = np.linspace(float(breakout[0]), target, 12)
    return _frame(np.concatenate([base, breakout, back]), spread=HALF_BAR)


class TestItWantsTheLevelAndNotAPullback:
    def test_a_return_to_the_level_is_a_signal(self) -> None:
        engine = LevelRetest(LevelRetestConfig())

        signal = engine.analyze(_context(_broke_and_returned(0.05)))

        assert signal.score > 0, signal.reasoning
        assert "returned to within" in signal.reasoning

    def test_a_shallow_dip_from_the_high_is_not(self) -> None:
        """THE MEASURED DISTINCTION. 103 is a pullback off the 104 high and it
        is what the account traded for two months. Against the broken level it
        is about three ATR up, and the measurement says that entry is worth
        -0.067R while the one below is worth +0.35R."""
        engine = LevelRetest(LevelRetestConfig())

        signal = engine.analyze(_context(_broke_and_returned(1.20)))

        assert signal.score == 0
        assert "above the level it broke" in signal.reasoning

    def test_the_tolerance_is_the_shipped_one(self) -> None:
        """0.15 and not 0.35. The sweep is monotone and changes sign:

            0.15 ATR  +11.3 points over chance
            0.35 ATR   +5.0
            0.60 ATR   -3.4  <- negative

        Loosening this does not admit more of the same trade. It admits a
        different, losing one.
        """
        assert LevelRetestConfig().tolerance_atr == pytest.approx(0.15)

    def test_a_short_measures_the_level_the_other_way(self) -> None:
        engine = LevelRetest(LevelRetestConfig())

        signal = engine.analyze(_context(_broke_and_returned(0.05, down=True)))

        assert signal.score < 0, signal.reasoning


class TestTheStopIsPartOfTheStrategy:
    def test_it_publishes_an_invalidation_beyond_the_level(self) -> None:
        """The stop distance is not decoration -- E moves from +0.134R at 0.75
        ATR to +0.340R at 0.35 ATR. A module that emits a direction and lets
        something else pick the stop is not the thing that was measured."""
        engine = LevelRetest(LevelRetestConfig())

        signal = engine.analyze(_context(_broke_and_returned(0.05)))

        assert signal.invalidation_price is not None
        assert signal.key_levels
        level = signal.key_levels[0]
        assert signal.invalidation_price < level

    def test_the_stop_clears_the_confluence_floor(self) -> None:
        """`min_stop_atr` silently WIDENS a stop below it. A module measured at
        0.5 ATR and floored to 0.8 would trade a strategy nobody tested, so the
        shipped distance is chosen to survive the floor untouched."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        config = settings.analysis.level_retest
        total = config.tolerance_atr + config.stop_beyond_atr

        assert total >= settings.analysis.confluence.min_stop_atr


class TestABrokenLevelCanStopBeingOne:
    def test_a_level_price_has_closed_back_through_is_not_offered(self) -> None:
        """Otherwise the "retest" is a knife catch: the break failed, price is
        below where the stop would have been, and the module keeps pointing at
        a level that has already been given up."""
        engine = LevelRetest(LevelRetestConfig())

        # All the way back through the level and well beyond the stop. Price
        # this far down has broken the other side of the range, so a SHORT
        # here is correct and expected; what must not happen is the long being
        # re-offered at a level the market has walked away from.
        signal = engine.analyze(_context(_broke_and_returned(-4.0)))

        assert signal.score <= 0, signal.reasoning

    def test_a_break_older_than_the_lookback_is_forgotten(self) -> None:
        engine = LevelRetest(LevelRetestConfig(lookback_bars=4))

        signal = engine.analyze(_context(_broke_and_returned(0.05)))

        assert signal.score == 0
        assert "no unspent" in signal.reasoning

    def test_a_market_that_never_broke_anything_says_so(self) -> None:
        engine = LevelRetest(LevelRetestConfig())
        rng = np.random.default_rng(3)
        quiet = np.full(BARS, 100.0) + rng.normal(0, 0.3, BARS)

        signal = engine.analyze(_context(_frame(quiet)))

        assert signal.score == 0


class TestItIsWiredIn:
    def test_the_runner_builds_it(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from runner.service import build_analysis_modules

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

        assert "level_retest" in {m.name for m in build_analysis_modules(settings)}

    def test_it_carries_a_weight_so_the_backtest_can_grade_it(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        confluence = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence

        assert confluence.weights.get("level_retest", 0.0) > 0.0

    def test_it_reads_the_timeframe_it_was_measured_on(self) -> None:
        """M15. The edge has never been shown to survive on M1, and the three
        M1 modules on this account are precisely the ones that were never
        measured at all."""
        assert LevelRetestConfig().timeframe == "M15"

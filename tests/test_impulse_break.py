"""One violent bar with follow-through, without buying the top of every spike.

The hole this fills was found in live data rather than reasoned into being. A
GBPCAD move the reviewer described as "a large, fresh down-impulse — M15 last
candle body -1.34 ATR, M5 3-bar drift -2.3 ATR" fired no directional module at
all: `drift_continuation` needs 65% of eight bars to agree and an impulse gives
about 25%, `fast_ema_cross` needs a cross a vertical move has already left
behind. Only `trend_momentum` spoke, off a slow EMA, so the trade went out as a
24-hour swing and was refused for chasing.

Most of what follows is about the four floors, because a module that fires on
"one big bar" without them is a machine for buying the end of every spike.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from analysis.impulse_break import ImpulseBreak
from config.schema import ImpulseBreakConfig
from core.types import MarketContext, Series, Tick, Timeframe

NOW = datetime(2026, 1, 6, 10, tzinfo=UTC)


def context(bars: list[tuple[float, float, float, float]]) -> MarketContext:
    """`bars` are (open, high, low, close)."""
    index = pd.date_range("2026-01-01", periods=len(bars), freq="15min", tz=UTC)
    frame = pd.DataFrame(
        {
            "open": [b[0] for b in bars],
            "high": [b[1] for b in bars],
            "low": [b[2] for b in bars],
            "close": [b[3] for b in bars],
            "tick_volume": 100,
            "spread": 10,
            "real_volume": 0,
        },
        index=index,
    )
    return MarketContext(
        symbol="GBPCAD",
        now=NOW,
        series={Timeframe.M15: Series("GBPCAD", Timeframe.M15, frame, NOW)},
        tick=Tick("GBPCAD", NOW, 1.7000, 1.7002),
    )


def quiet(n: int = 40, price: float = 1.7000, size: float = 0.0004) -> list:
    """Flat bars with a consistent range, so ATR is a known quantity."""
    return [(price, price + size, price - size, price) for _ in range(n)]


def analyse(bars: list, **overrides):  # type: ignore[no-untyped-def]
    return ImpulseBreak(ImpulseBreakConfig(**overrides)).analyze(context(bars))


class TestItSeesWhatNothingElseDid:
    def test_a_large_decisive_down_bar_is_a_short(self) -> None:
        """The GBPCAD shape: a body well over an ATR, closing on its low."""
        bars = quiet()
        bars.append((1.7000, 1.7001, 1.6970, 1.6972))

        signal = analyse(bars)

        assert signal.score < 0
        assert "down impulse" in signal.reasoning

    def test_a_large_decisive_up_bar_is_a_long(self) -> None:
        bars = quiet()
        bars.append((1.7000, 1.7030, 1.6999, 1.7028))

        assert analyse(bars).score > 0

    def test_the_stop_is_the_origin_of_the_impulse(self) -> None:
        """Price back through where the repricing started means it did not
        hold, which is precisely the claim being made."""
        bars = quiet()
        bars.append((1.7000, 1.7030, 1.6999, 1.7028))

        assert analyse(bars).invalidation_price == 1.7000

    def test_it_still_fires_two_bars_after_the_impulse(self) -> None:
        """A window, not a single bar. The scanner cannot be looking at every
        symbol on the exact bar an impulse prints."""
        bars = quiet()
        bars.append((1.7000, 1.7030, 1.6999, 1.7028))
        bars.append((1.7028, 1.7032, 1.7026, 1.7030))

        signal = analyse(bars)

        assert signal.score > 0
        assert signal.details["bars_since_impulse"] == 1


class TestTheFourFloors:
    def test_a_small_body_says_nothing(self) -> None:
        bars = quiet()
        bars.append((1.7000, 1.7006, 1.6996, 1.7004))

        signal = analyse(bars)

        assert signal.score == 0
        assert "below the" in signal.reasoning

    def test_a_wide_bar_with_a_small_body_is_indecision(self) -> None:
        """The body, not the range. A bar that travelled far and closed where
        it opened is a fight, not a repricing."""
        bars = quiet()
        bars.append((1.7000, 1.7040, 1.6960, 1.7002))

        assert analyse(bars).score == 0

    def test_a_big_body_that_closed_mid_range_is_a_rejection(self) -> None:
        """A large body with a large opposing wick. Joining that means joining
        the reversal of the move you meant to join."""
        bars = quiet()
        bars.append((1.7000, 1.7060, 1.6998, 1.7028))

        signal = analyse(bars)

        assert signal.score == 0
        assert "rejection" in signal.reasoning

    def test_a_stale_impulse_is_not_an_entry(self) -> None:
        """The mechanism is about the minutes after the repricing. An hour
        later the liquidity has come back."""
        bars = quiet()
        bars.append((1.7000, 1.7030, 1.6999, 1.7028))
        bars += [(1.7028, 1.7030, 1.7026, 1.7028) for _ in range(6)]

        assert analyse(bars).score == 0

    def test_an_impulse_the_market_gave_back_is_refused(self) -> None:
        """Half of it retraced means the break was rejected and the spike was
        the whole story."""
        bars = quiet()
        bars.append((1.7000, 1.7030, 1.6999, 1.7028))
        bars.append((1.7028, 1.7029, 1.7008, 1.7010))

        signal = analyse(bars)

        assert signal.score == 0
        assert "given back" in signal.reasoning

    def test_each_floor_is_load_bearing(self) -> None:
        """Relaxing the one under test lets the same bars through, which proves
        that floor is doing the work rather than something else rejecting it."""
        # A bar closing at 55% of its own range: refused by the 66% default,
        # accepted at 50%. The numbers are measured against the module rather
        # than guessed — the config floor is `ge=0.5`, so a fixture needing a
        # lower relaxation could not be expressed and the test would pass by
        # raising instead of by proving anything.
        half_hearted = quiet()
        half_hearted.append((1.7000, 1.70525, 1.6998, 1.7028))

        assert analyse(half_hearted).score == 0
        assert analyse(half_hearted, minimum_close_location=0.5).score > 0


class TestItNeverGuesses:
    def test_too_little_history_says_nothing(self) -> None:
        assert analyse(quiet(n=6)).score == 0

    def test_a_missing_chart_says_nothing(self) -> None:
        empty = MarketContext(symbol="GBPCAD", now=NOW, series={}, tick=None)

        assert ImpulseBreak().analyze(empty).score == 0

    def test_disabling_it_silences_it(self) -> None:
        bars = quiet()
        bars.append((1.7000, 1.7030, 1.6999, 1.7028))

        assert analyse(bars, enabled=False).score == 0


class TestTheEngineTreatsItAsWhatItIs:
    def test_it_is_registered_as_an_intraday_module(self) -> None:
        """A three-hour thesis handed H1 planning authority and a 24-hour
        target is not the trade this module found — the exact mistake that
        produced the GBPCAD swing short in the first place."""
        from config.schema import ConfluenceConfig

        assert "impulse_break" in ConfluenceConfig().intraday_modules

    def test_it_is_not_a_trend_continuation_module(self) -> None:
        """It measures a move rather than inferring a trend, so a range on H1
        does not contradict it. The wrong classification cost 6,726 refusals in
        a day when `drift_continuation` carried it."""
        from config.schema import ConfluenceConfig

        assert "impulse_break" not in ConfluenceConfig().trend_continuation_modules

    def test_only_its_strongest_reading_can_carry_a_trade_alone(self) -> None:
        """REVERSED ON 25 AUGUST, deliberately.

        This asserted that even the module's WEAKEST reading cleared the bar,
        which at 26 it did -- and the HK50 short that cost EUR 3.13 was exactly
        that: fired alone at 0.45, the floor exactly, scoring 27.0 against 26.0.

        A lone module's confluence score is `raw x confidence`, so a bar of 45
        asks this module for 0.75 confidence against a ceiling of 0.80. The
        weakest reading is refused and only a genuinely convinced one stands
        alone, which is what the owner asked for in as many words."""
        from pathlib import Path

        from config.loader import load_settings

        live = load_settings(
            overlay=Path(__file__).resolve().parent.parent / "config" / "eightcap.yaml",
            env_overrides=False,
        ).analysis
        module = live.impulse_break

        bar = live.confluence.score_threshold
        # The HK50 reading -- lone, at the floor -- is now refused.
        assert module.score * module.base_confidence < bar
        # And a convinced one still gets through, or the module is decoration.
        assert module.score * module.maximum_confidence > bar

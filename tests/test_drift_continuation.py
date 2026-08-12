"""Join a move that is already happening, without joining chop.

GBPUSD fell steadily for most of 12 August. The engine spent that day trying to
buy it — 233 M15 and 111 M5 refusals reading "price is moving against the long"
— and never once proposed a short. Not because a gate stopped one, but because
no module was looking: `trend_momentum` runs 20/50 EMAs on H4 and H1 and goes
quiet long before a market turns, and `liquidity_sweep` needs a wick through a
20-bar extreme on the last candle and is a reversal pattern anyway.

The risk in filling that hole is obvious and it is what most of this file
guards. A module that fires on "price moved" fires constantly in a range, and
buying and selling a sideways market alternately is how an account is emptied
by the spread rather than by being wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from analysis.drift_continuation import DriftContinuation
from config.schema import DriftContinuationConfig
from core.types import MarketContext, Series, Tick, Timeframe

BARS = 60


def series_from(closes: list[float]) -> MarketContext:
    index = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz=UTC)
    close = pd.Series(closes, index=index)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "tick_volume": 100,
            "spread": 10,
            "real_volume": 0,
        },
        index=index,
    )
    now = datetime(2026, 1, 6, 10, tzinfo=UTC)
    return MarketContext(
        symbol="GBPUSD",
        now=now,
        series={Timeframe.M15: Series("GBPUSD", Timeframe.M15, frame, now)},
        tick=Tick("GBPUSD", now, 1.3500, 1.3501),
    )


def steady(step: float) -> MarketContext:
    """A clean one-way grind — the GBPUSD shape."""
    return series_from([1.3500 + i * step for i in range(BARS)])


def chop(amplitude: float, net: float = 0.0) -> MarketContext:
    """Up, down, up, down, ending wherever `net` says.

    The amplitude has to stay small relative to the net drift or the zigzag
    inflates its own ATR and the setup is thrown out by the distance floor
    instead of the consistency one — which passes the test for the wrong
    reason and leaves the condition under test unexercised.
    """
    closes = []
    for i in range(BARS):
        zig = amplitude if i % 2 else -amplitude
        closes.append(1.3500 + zig + i * net)
    return series_from(closes)


def analyse(ctx: MarketContext, **overrides):  # type: ignore[no-untyped-def]
    return DriftContinuation(DriftContinuationConfig(**overrides)).analyze(ctx)


class TestItJoinsAMoveThatIsActuallyHappening:
    def test_a_sustained_decline_produces_a_short(self) -> None:
        """The trade that was never proposed on 12 August."""
        signal = analyse(steady(-0.0006))

        assert signal.score < 0
        assert "down" in signal.reasoning

    def test_a_sustained_rise_produces_a_long(self) -> None:
        assert analyse(steady(0.0006)).score > 0

    def test_a_bigger_move_is_held_more_confidently(self) -> None:
        """Measured with the saturation point pushed out, because a perfectly
        clean synthetic trend clears the real one several times over and both
        sides would pin at the ceiling — which is correct behaviour and proves
        nothing about the scaling."""
        gentle = analyse(steady(-0.0002), confident_drift_atr=10.0)
        strong = analyse(steady(-0.0012), confident_drift_atr=10.0)

        assert strong.confidence > gentle.confidence

    def test_the_stop_sits_at_the_far_end_of_the_move(self) -> None:
        """Price back through where the drift started means the thing this is
        built on is gone. A signal with no invalidation cannot be sized."""
        signal = analyse(steady(-0.0006))
        last_close = 1.3500 + (BARS - 1) * -0.0006

        assert signal.invalidation_price is not None
        assert signal.invalidation_price > last_close


class TestItStaysOutOfChop:
    """The whole risk of this module, and the reason it is not simply the
    refused long turned upside down."""

    def test_a_market_going_nowhere_says_nothing(self) -> None:
        assert analyse(chop(0.0020)).score == 0

    def test_the_same_net_drift_without_consistency_is_refused(self) -> None:
        """A market that ends lower having gone up, down, up and down has the
        same net movement as one that ground steadily lower. Only the second is
        going somewhere, and net drift alone cannot tell them apart."""
        zigzag = analyse(chop(0.0002, net=-0.0002))

        assert "2.67 ATR" in zigzag.reasoning, "it cleared the distance floor"
        assert zigzag.score == 0
        assert "chop with a net" in zigzag.reasoning

    def test_a_move_too_small_to_matter_says_nothing(self) -> None:
        """A tenth of an ATR is not a move, it is breathing."""
        assert analyse(steady(-0.00002)).score == 0

    def test_the_consistency_floor_is_what_rejects_the_zigzag(self) -> None:
        """Drop the floor to nothing and the same zigzag passes — which proves
        it is this condition doing the work and not the drift size."""
        zigzag = chop(0.0002, net=-0.0002)

        assert analyse(zigzag).score == 0
        assert analyse(zigzag, minimum_consistency=0.0).score != 0


class TestItNeverGuesses:
    def test_too_little_history_says_nothing(self) -> None:
        assert analyse(series_from([1.35, 1.3501, 1.3502])).score == 0

    def test_a_missing_timeframe_says_nothing(self) -> None:
        now = datetime(2026, 1, 6, 10, tzinfo=UTC)
        empty = MarketContext(symbol="GBPUSD", now=now, series={}, tick=None)

        assert analyse(empty).score == 0

    def test_disabling_it_silences_it(self) -> None:
        assert analyse(steady(-0.0006), enabled=False).score == 0


class TestTheRangeGateCoversIt:
    def test_it_is_registered_as_a_trend_continuation_module(self) -> None:
        """Without this it would fire freely in a measured range, which is
        exactly the failure the consistency floor only partly covers. The
        confluence engine refuses continuation modules when the regime
        classifier says range, and it looks the name up in this tuple."""
        from config.schema import ConfluenceConfig

        assert "drift_continuation" in ConfluenceConfig().trend_continuation_modules

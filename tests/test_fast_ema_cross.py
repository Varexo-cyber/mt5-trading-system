"""A 9/20 cross on five minutes, without trading every brush of the averages.

Every other directional module reads a slow chart: 20/50 EMAs on H4 and H1, a
break of structure, or at the fastest a specific wick on M15. Nothing looked at
the timeframe a day trade is actually entered on.

The danger is not subtle. A 9/20 cross on M5 is the classic quick entry and the
classic way to be sawn to pieces — two averages brushing all day in a quiet
market, each brush a trade paying the spread. Almost everything below is about
the three floors that separate a cross from a touch, because without them this
module is a machine for donating the spread.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from analysis.fast_ema_cross import FastEmaCross
from config.schema import FastEmaCrossConfig
from core.types import MarketContext, Series, Tick, Timeframe


def context_from(closes: list[float]) -> MarketContext:
    index = pd.date_range("2026-01-01", periods=len(closes), freq="5min", tz=UTC)
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
        symbol="EURUSD",
        now=now,
        series={Timeframe.M5: Series("EURUSD", Timeframe.M5, frame, now)},
        tick=Tick("EURUSD", now, 1.1000, 1.1001),
    )


def turned(down_bars: int, up_bars: int = 90, step: float = 0.0004) -> MarketContext:
    """A market that fell, then turned up `up_bars` bars ago.

    Note that `up_bars` is not the age of the cross. A 9/20 pair takes about
    twelve bars to actually cross after the turn, so `up_bars=13` is a cross
    printed on the last bar and `up_bars=30` is one eighteen bars old. Those
    numbers were measured against the module rather than guessed, because a
    fixture that never crosses at all passes a "no signal" assertion for
    entirely the wrong reason.
    """
    closes = [1.1000 - i * step for i in range(down_bars)]
    bottom = closes[-1]
    closes += [bottom + (i + 1) * step for i in range(up_bars)]
    return context_from(closes)


def analyse(ctx: MarketContext, **overrides):  # type: ignore[no-untyped-def]
    return FastEmaCross(FastEmaCrossConfig(**overrides)).analyze(ctx)


class TestItCatchesTheTurn:
    def test_a_fresh_upward_cross_is_a_long(self) -> None:
        signal = analyse(turned(down_bars=90, up_bars=13))

        assert signal.score > 0
        assert "crossed above" in signal.reasoning

    def test_a_fresh_downward_cross_is_a_short(self) -> None:
        closes = [1.1000 + i * 0.0004 for i in range(90)]
        top = closes[-1]
        closes += [top - (i + 1) * 0.0004 for i in range(13)]

        assert analyse(context_from(closes)).score < 0

    def test_it_carries_an_invalidation_level(self) -> None:
        """A signal without one cannot be sized, so it would be refused later
        for a missing stop and the module would look silent rather than wrong."""
        signal = analyse(turned(down_bars=90, up_bars=13))

        assert signal.invalidation_price is not None


class TestTheThreeFloorsThatKeepItOutOfNoise:
    def test_a_stale_cross_is_a_state_and_not_an_entry(self) -> None:
        """Forty minutes after the cross this is just "the EMAs are up", which
        is what `trend_momentum` already reports. Taking it as an entry signal
        means entering the same move over and over."""
        signal = analyse(turned(down_bars=90, up_bars=30))

        assert signal.score == 0
        assert "a state, not an entry" in signal.reasoning

    def test_averages_brushing_are_not_averages_crossing(self) -> None:
        """A market going nowhere crosses its own EMAs constantly. Reading each
        touch as a signal is reading noise at high frequency."""
        closes = [1.1000 + (0.00002 if i % 2 else -0.00002) for i in range(120)]

        signal = analyse(context_from(closes))

        assert signal.score == 0
        assert "brushing" in signal.reasoning

    def test_a_cross_price_has_already_failed_is_refused(self) -> None:
        """The averages crossed up and price closed back under the slow one —
        a cross that failed in the time it took to print.

        Tested with the separation floor relaxed, and that is worth stating
        plainly: on any move violent enough to put price back through the slow
        average within three bars, the same bar inflates ATR and the separation
        floor rejects it first. This third check is therefore a guard on a
        state the other two nearly always catch, kept because "nearly always"
        is not "always" and a real reversal bar is not a straight line. Without
        relaxing the other floor the branch never runs and the test would pass
        while proving nothing.
        """
        closes = [1.1000 - i * 0.0004 for i in range(90)]
        bottom = closes[-1]
        closes += [bottom + (i + 1) * 0.0004 for i in range(12)]
        closes += [closes[-1] - 0.0016]

        signal = analyse(context_from(closes), minimum_separation_atr=0.05)

        assert signal.score == 0
        assert "closed back through" in signal.reasoning

    def test_each_floor_is_load_bearing(self) -> None:
        """Relax the freshness window and the stale cross passes — which proves
        that floor is doing the work rather than something else rejecting it."""
        stale = turned(down_bars=90, up_bars=30)

        assert analyse(stale).score == 0
        assert analyse(stale, max_bars_since_cross=50).score > 0


class TestItNeverGuesses:
    def test_too_little_history_says_nothing(self) -> None:
        assert analyse(context_from([1.10 + i * 0.0001 for i in range(20)])).score == 0

    def test_a_missing_fast_chart_says_nothing(self) -> None:
        now = datetime(2026, 1, 6, 10, tzinfo=UTC)
        empty = MarketContext(symbol="EURUSD", now=now, series={}, tick=None)

        assert analyse(empty).score == 0

    def test_disabling_it_silences_it(self) -> None:
        assert analyse(turned(down_bars=90, up_bars=13), enabled=False).score == 0


class TestTheEngineTreatsItAsWhatItIs:
    def test_it_is_deliberately_not_a_trend_continuation_module(self) -> None:
        """The range veto is for premises, and this module has a measurement.

        `trend_momentum` INFERS a trend from EMA alignment on H4 and H1, and a
        measured range genuinely contradicts that inference. This module
        measures a separation in ATR on M5 over the last three bars. An H1
        range says nothing about whether those three bars separated, and a fast
        cross inside a wider range is the ordinary way an intraday move starts.

        What keeps it out of chop is its own separation floor, tested above.
        """
        from config.schema import ConfluenceConfig

        assert ConfluenceConfig().trend_continuation_modules == ("trend_momentum",)

    def test_a_clean_cross_can_clear_the_threshold_on_its_own(self) -> None:
        """The arithmetic that kept every quick entry off the book.

        A lone module's confluence score is `raw score x confidence` — the
        weight cancels out, because the denominator runs over the same firing
        modules. At 50 points across a 0.45-0.80 confidence band this module
        produces 22.5 to 40, so a threshold of 35 silently demanded 0.50 ATR of
        separation while the module's own floor is 0.15. Everything in between
        fired and was discarded as "confluence score below threshold" — 5,992
        times in a single day, and not one quick entry ever reached the book.

        The threshold has to sit below what a clean cross actually scores, or
        this module is decoration.
        """
        from pathlib import Path

        from config.loader import load_settings

        # The live overlay, not the schema default: this module only runs on
        # the Eightcap account and the threshold that governs it lives there.
        live = load_settings(
            overlay=Path(__file__).resolve().parent.parent / "config" / "eightcap.yaml",
            env_overrides=False,
        ).analysis
        module = live.fast_ema_cross
        # A cross separated by 0.40 ATR: well clear of the module's own 0.15
        # floor, and nothing anyone would call remarkable.
        clean = module.base_confidence + 0.40 * module.separation_confidence_scale

        assert module.score * clean > live.confluence.score_threshold

    def test_it_is_registered_as_an_intraday_module(self) -> None:
        """Without this a five-minute signal is handed a swing plan: H1
        planning authority and a target twenty-four hours out. That is not the
        trade the module found."""
        from config.schema import ConfluenceConfig

        assert "fast_ema_cross" in ConfluenceConfig().intraday_modules

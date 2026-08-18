"""Four families the system could not see, and the floors that keep them honest.

The eight existing directional modules all read the same price series a
different way — a trend, a cross, a break, a drift, a resumed pullback, a
micro-break. They fire together and, measured over 180 days, they lose
together: `trend_momentum` significantly so at -0.382R a trade over 62 of them,
t = -3.26. A ninth reader of the same kind would have been a ninth correlated
loser.

These four look at something none of the eight do: how compressed the range is
against its own history, how far price is from its own mean, what hour of the
day it is, and what weekday it is.

What is tested here is mostly the REFUSALS. A signal that fires when it should
is easy; a signal that fires when it should not is what costs money, and each
of these families has one specific way of doing that:

    squeeze         the first bar of every session, if compression is not
                    measured against the instrument's own history
    mean reversion  catching a knife — fading a move that is still going
    session         breaking a range that is dead, or one that already moved
    seasonality     five weekdays tested is five chances at a two-sigma fluke
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from analysis import MeanReversion, Seasonality, SessionBreakout, VolatilitySqueeze
from config.schema import (
    MeanReversionConfig,
    SeasonalityConfig,
    SessionBreakoutConfig,
    VolatilitySqueezeConfig,
)
from core.types import MarketContext, Series, Timeframe

NOW = datetime(2026, 8, 18, 9, 0, tzinfo=UTC)


def series_from(
    closes: list[float],
    *,
    timeframe: Timeframe,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    start: datetime | None = None,
) -> MarketContext:
    index = pd.date_range(
        start or (NOW - timeframe.duration * len(closes)),
        periods=len(closes),
        freq=timeframe.duration,
        tz=UTC,
    )
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": highs if highs is not None else [c + 0.0002 for c in closes],
            "low": lows if lows is not None else [c - 0.0002 for c in closes],
            "close": closes,
            "spread": 2,
        },
        index=index,
    )
    return MarketContext(
        symbol="TEST",
        now=NOW,
        series={timeframe: Series("TEST", timeframe, frame, NOW)},
    )


# ------------------------------------------------------------------ squeeze ---


def squeeze_bars(expansion: float, *, quiet: float = 0.0002) -> MarketContext:
    """250 quiet bars, then one bar that spans `expansion` times the coil."""
    rng = np.random.default_rng(5)
    closes = list(1.1000 + np.cumsum(rng.normal(0, quiet / 8, 250)))
    highs = [c + quiet / 2 for c in closes]
    lows = [c - quiet / 2 for c in closes]
    coil = max(highs[-12:]) - min(lows[-12:])
    closes.append(closes[-1] + coil * expansion * 0.9)
    highs.append(closes[-1] + coil * 0.05)
    lows.append(closes[-2] - coil * 0.05)
    return series_from(closes, timeframe=Timeframe.M15, highs=highs, lows=lows)


class TestVolatilitySqueeze:
    def test_a_real_release_scores_a_direction(self) -> None:
        signal = VolatilitySqueeze().analyze(squeeze_bars(3.0))

        assert signal.score > 0
        assert signal.confidence > 0
        assert signal.invalidation_price is not None

    def test_a_quiet_bar_after_a_coil_is_not_a_release(self) -> None:
        """Without this, the module reports a squeeze on every bar of every
        quiet market — the coil is real and nothing has happened yet."""
        signal = VolatilitySqueeze().analyze(squeeze_bars(0.3))

        assert signal.score == 0
        assert "still coiled" in signal.reasoning

    def test_an_expansion_without_a_coil_says_nothing(self) -> None:
        """The floor that stops this being "the first bar of London". A market
        that was never compressed cannot have released, however big the bar."""
        rng = np.random.default_rng(9)
        # Quiet for most of the window, then WIDENING into the end, so the
        # recent range sits high in its own history rather than low.
        closes = list(1.1000 + np.cumsum(rng.normal(0, 0.0001, 240)))
        closes += list(closes[-1] + np.cumsum(rng.normal(0, 0.0020, 10)))
        closes.append(closes[-1] + 0.004)
        context = series_from(closes, timeframe=Timeframe.M15)

        signal = VolatilitySqueeze().analyze(context)

        assert signal.score == 0
        assert "not coiled" in signal.reasoning

    def test_the_stop_sits_beyond_the_far_side_of_the_coil(self) -> None:
        """The near edge is inside the noise the compression is made of, and
        the breakout bar has already traded through it."""
        signal = VolatilitySqueeze().analyze(squeeze_bars(3.0))
        bottom = signal.details["range_bottom"]

        assert signal.invalidation_price is not None
        assert signal.invalidation_price < bottom

    def test_disabled_means_silent(self) -> None:
        module = VolatilitySqueeze(VolatilitySqueezeConfig(enabled=False))

        assert module.analyze(squeeze_bars(3.0)).score == 0

    def test_too_little_history_is_not_a_refusal_of_the_setup(self) -> None:
        context = series_from([1.1] * 40, timeframe=Timeframe.M15)
        signal = VolatilitySqueeze().analyze(context)

        assert signal.score == 0
        assert "needs" in signal.reasoning


# ----------------------------------------------------------- mean reversion ---


def stretched(retrace: float) -> MarketContext:
    """A long calm base, a hard push up, then `retrace` of the last bar given back."""
    closes = [1.1000] * 80
    for step in range(6):
        closes.append(1.1000 + 0.0012 * (step + 1))
    highs = [c + 0.00005 for c in closes]
    lows = [c - 0.00005 for c in closes]
    top = closes[-1] + 0.0004
    highs[-1] = top
    lows[-1] = closes[-1] - 0.0004
    bar_range = top - lows[-1]
    # The closing bar must not print a new high, or it becomes the extreme
    # itself and the module would be measuring the pullback against a bar that
    # has not happened yet.
    closes.append(top - bar_range * retrace)
    highs.append(min(closes[-1] + 0.00005, top - 0.00001))
    lows.append(closes[-1] - 0.00005)
    return series_from(closes, timeframe=Timeframe.M15, highs=highs, lows=lows)


class TestMeanReversion:
    def test_a_stalled_extreme_is_faded(self) -> None:
        signal = MeanReversion().analyze(stretched(0.6))

        assert signal.score < 0, signal.reasoning
        assert signal.invalidation_price is not None

    def test_a_move_still_extending_is_left_alone(self) -> None:
        """The knife-catch. Fading a move because it has gone far is the most
        expensive mistake in this family, and the stall test is the only thing
        standing between this module and that mistake."""
        signal = MeanReversion().analyze(stretched(0.0))

        assert signal.score == 0
        assert "still going" in signal.reasoning

    def test_an_ordinary_market_is_not_an_extreme(self) -> None:
        rng = np.random.default_rng(3)
        closes = list(1.1000 + np.cumsum(rng.normal(0, 0.0002, 120)))

        signal = MeanReversion().analyze(series_from(closes, timeframe=Timeframe.M15))

        assert signal.score == 0
        assert "SD from its" in signal.reasoning

    def test_the_stop_sits_beyond_the_extreme_that_was_faded(self) -> None:
        signal = MeanReversion().analyze(stretched(0.6))
        extreme = signal.details["extreme"]

        assert signal.invalidation_price is not None
        assert signal.invalidation_price > extreme

    def test_the_short_side_mirrors(self) -> None:
        """A sign error here is invisible in production and inverts every trade
        the module ever produces."""
        context = stretched(0.6)
        frame = context.series[Timeframe.M15].df
        mirrored = 2.2 - frame[["open", "high", "low", "close"]]
        flipped = frame.copy()
        flipped["open"] = mirrored["open"]
        flipped["close"] = mirrored["close"]
        flipped["high"] = mirrored["low"]
        flipped["low"] = mirrored["high"]

        signal = MeanReversion().analyze(
            MarketContext(
                symbol="TEST",
                now=NOW,
                series={Timeframe.M15: Series("TEST", Timeframe.M15, flipped, NOW)},
            )
        )

        assert signal.score > 0, signal.reasoning

    def test_disabled_means_silent(self) -> None:
        module = MeanReversion(MeanReversionConfig(enabled=False))

        assert module.analyze(stretched(0.6)).score == 0


# ---------------------------------------------------------- session breakout ---


def overnight(break_size: float, *, range_width: float = 0.0030) -> MarketContext:
    """Bars from 00:00, a range until 07:00, then a move after it."""
    start = datetime(2026, 8, 18, 0, 0, tzinfo=UTC)
    closes: list[float] = []
    base = 1.1000
    # 00:00-07:00 on M15 is 28 bars, oscillating inside the range.
    for i in range(28):
        closes.append(base + (range_width / 2 if i % 2 else -range_width / 2))
    # Then the session that resolves it: 8 bars drifting to the break.
    for i in range(8):
        closes.append(base + range_width / 2 + break_size * (i + 1) / 8)
    highs = [c + 0.00005 for c in closes]
    lows = [c - 0.00005 for c in closes]
    return MarketContext(
        symbol="TEST",
        now=start + timedelta(hours=9),
        series={
            Timeframe.M15: Series(
                "TEST",
                Timeframe.M15,
                pd.DataFrame(
                    {
                        "open": closes,
                        "high": highs,
                        "low": lows,
                        "close": closes,
                        "spread": 2,
                    },
                    index=pd.date_range(start, periods=len(closes), freq="15min", tz=UTC),
                ),
                start + timedelta(hours=9),
            )
        },
    )


class TestSessionBreakout:
    def test_a_break_of_the_overnight_range_scores(self) -> None:
        signal = SessionBreakout().analyze(overnight(0.0020))

        assert signal.score > 0, signal.reasoning
        assert signal.invalidation_price is not None

    def test_price_still_inside_the_range_says_nothing(self) -> None:
        signal = SessionBreakout().analyze(overnight(0.0))

        assert signal.score == 0
        assert "still inside" in signal.reasoning

    def test_a_dead_overnight_range_is_refused(self) -> None:
        """Breaking a five-pip range is not a signal, it is a quote moving."""
        signal = SessionBreakout().analyze(overnight(0.0020, range_width=0.00002))

        assert signal.score == 0
        assert "under the" in signal.reasoning

    def test_the_stop_is_the_far_side_of_the_range(self) -> None:
        signal = SessionBreakout().analyze(overnight(0.0020))

        assert signal.invalidation_price == signal.details["range_bottom"]

    def test_disabled_means_silent(self) -> None:
        module = SessionBreakout(SessionBreakoutConfig(enabled=False))

        assert module.analyze(overnight(0.0020)).score == 0


# ------------------------------------------------------------- seasonality ---


def daily(bias_on_weekday: int | None, *, size: float = 0.004) -> MarketContext:
    """700 daily bars; optionally one weekday given a consistent drift."""
    rng = np.random.default_rng(11)
    index = pd.date_range("2023-01-02", periods=700, freq="D", tz=UTC)
    price = 100.0
    closes = []
    for stamp in index:
        step = rng.normal(0, 0.002)
        if bias_on_weekday is not None and stamp.weekday() == bias_on_weekday:
            step += size
        price *= 1 + step
        closes.append(price)
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "spread": 2,
        },
        index=index,
    )
    moment = datetime(2026, 8, 18, 9, 0, tzinfo=UTC)  # a Tuesday
    return MarketContext(
        symbol="TEST",
        now=moment,
        series={Timeframe.D1: Series("TEST", Timeframe.D1, frame, moment)},
    )


class TestSeasonality:
    def test_a_measurable_weekday_bias_is_reported(self) -> None:
        signal = Seasonality().analyze(daily(bias_on_weekday=1))

        assert signal.score > 0, signal.reasoning
        assert signal.details["t_statistic"] >= 2.0

    def test_a_market_with_no_weekday_bias_says_nothing(self) -> None:
        """Which is what this will do on most instruments on most days, and
        saying so is the module working rather than the module failing."""
        signal = Seasonality().analyze(daily(bias_on_weekday=None))

        assert signal.score == 0
        assert "noise, not its calendar" in signal.reasoning

    def test_it_cannot_carry_a_trade_by_itself(self) -> None:
        """Its confidence ceiling sits under the 0.65 a lone module needs, so
        the strongest weekday effect on record still cannot open a position on
        its own. A weekday lean is background, never a trade."""
        signal = Seasonality().analyze(daily(bias_on_weekday=1, size=0.02))

        assert signal.confidence <= SeasonalityConfig().maximum_confidence
        assert SeasonalityConfig().maximum_confidence < 0.65

    def test_a_thin_sample_is_refused_rather_than_believed(self) -> None:
        """700 daily bars hold about a hundred of each weekday, so a floor of
        140 is one this fixture cannot clear — which is the point. A weekday
        claim on forty observations is a story."""
        module = Seasonality(SeasonalityConfig(minimum_samples=140))

        signal = module.analyze(daily(bias_on_weekday=1))

        assert signal.score == 0
        assert "under the 140" in signal.reasoning

    def test_disabled_means_silent(self) -> None:
        module = Seasonality(SeasonalityConfig(enabled=False))

        assert module.analyze(daily(bias_on_weekday=1)).score == 0


# ------------------------------------------------------------ the allowlist ---


class TestNoneOfThemCanTradeYet:
    """Measurable offline, inert live. A positive weight is what makes a module
    count in a backtest; `live_enabled_modules` is a separate allowlist, and
    the confluence engine zeroes the weight of anything not on it in live mode.

    This is the discipline that produced the only real finding on this account,
    and it is worth a test rather than a promise.
    """

    def _confluence(self):  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence

    def test_they_are_weighted_so_the_backtest_can_see_them(self) -> None:
        weights = self._confluence().weights
        for module in ("volatility_squeeze", "mean_reversion", "session_breakout", "seasonality"):
            assert weights.get(module, 0.0) > 0, module

    def test_but_none_is_on_the_live_allowlist(self) -> None:
        allowed = set(self._confluence().live_enabled_modules)
        for module in ("volatility_squeeze", "mean_reversion", "session_breakout", "seasonality"):
            assert module not in allowed, module

    def test_seasonality_cannot_reach_the_lone_module_floor(self) -> None:
        config = self._confluence()

        assert SeasonalityConfig().maximum_confidence < config.lone_module_minimum_confidence

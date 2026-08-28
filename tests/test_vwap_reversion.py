"""Section two, replacing `drift_burst` on 89 setups at 19% hit and -64.07R.

VWAP is the volume-weighted average price since the session opened: not an
indicator over the chart but a summary of it. A desk working a large order is
measured against it, so it buys below and sells above, and that pressure is the
only defensible half of the usual "price snaps back to VWAP" telling.

WHY THIS ACCOUNT NEEDS A MEAN-REVERSION READER, from the scorecard rather than
from theory: `range` -2.04R and `transition` -1.76R against `trend_up` +0.20R
and `trend_down` +0.18R. Every loss is in a sideways market. And the only
mean-reversion reader already on the book fired nine times in a day against
`trend_momentum`'s 58,612 -- nine detectors read the same shape and the one
that does not barely speaks.

WHAT THESE TESTS GUARD is the half that keeps it from standing in front of a
train. Distance alone cannot tell a mispricing from a repricing: two sigma
below VWAP on a quiet drift and two sigma below on a release are the same
number with opposite futures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.vwap_reversion import VwapReversion, session_vwap
from config.schema import VwapReversionConfig
from core.types import MarketContext, Series, Tick, Timeframe

BARS = 300


def _frame(closes: np.ndarray, volume: float | np.ndarray = 100.0) -> pd.DataFrame:
    index = pd.date_range("2026-08-28", periods=len(closes), freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.05,
            "low": closes - 0.05,
            "close": closes,
            "tick_volume": volume,
            "spread": 2,
        },
        index=index,
    )


def _context(frame: pd.DataFrame) -> MarketContext:
    price = float(frame["close"].iloc[-1])
    return MarketContext(
        "TEST",
        frame.index[-1].to_pydatetime(),
        {Timeframe.M5: Series("TEST", Timeframe.M5, frame, frame.index[-1].to_pydatetime())},
        Tick("TEST", frame.index[-1].to_pydatetime(), bid=price - 0.01, ask=price + 0.01),
    )


def _stretched(sigmas: float, *, still_moving: bool) -> pd.DataFrame:
    """A session that sat flat and then moved away from its own average.

    `still_moving` decides whether the last bars keep extending the stretch or
    hold it, which is the one thing separating a stretch to fade from a trend
    in progress.
    """
    flat = np.full(BARS - 20, 100.0) + np.random.default_rng(3).normal(0, 0.05, BARS - 20)
    if still_moving:
        tail = np.linspace(100.0, 100.0 + sigmas, 20)
    else:
        # Runs out to the stretch and then stops dead for the last bars.
        tail = np.concatenate([np.linspace(100.0, 100.0 + sigmas, 14), np.full(6, 100.0 + sigmas)])
    return _frame(np.concatenate([flat, tail]))


class TestTheReadingItself:
    def test_vwap_uses_the_typical_price_and_not_the_close(self) -> None:
        """A bar is a range of trades, not one trade at its end. Using the
        close biases the average toward wherever each bar happened to finish --
        which is exactly the quantity a deviation is then measured from."""
        # A little noise, because a perfectly flat frame has zero dispersion
        # and `session_vwap` correctly refuses it -- that refusal is asserted
        # separately below, and leaning on it here would have made this test
        # pass for the wrong reason.
        closes = np.full(50, 100.0) + np.random.default_rng(11).normal(0, 0.02, 50)
        frame = _frame(closes)
        # Push every high up so typical price sits above the close.
        frame["high"] = closes + 3.0

        reading = session_vwap(frame, stall_bars=3)

        assert reading is not None
        # (high + low + close) / 3 with high 3.0 above and low 0.05 below puts
        # the typical price about 0.98 over the close. A close-based VWAP would
        # land on 100.
        assert reading.vwap == pytest.approx(100.98, abs=0.05)

    def test_a_frame_with_no_dispersion_at_all_is_refused(self) -> None:
        """Every bar identical means every deviation is zero, and dividing by
        that dispersion would be a division by zero dressed as a signal."""
        assert session_vwap(_frame(np.full(50, 100.0)), stall_bars=3) is None

    def test_volume_actually_weights_it(self) -> None:
        """Otherwise this is a simple moving average wearing another name."""
        closes = np.concatenate([np.full(40, 100.0), np.full(10, 110.0)])
        weight = np.concatenate([np.full(40, 1.0), np.full(10, 100.0)])

        heavy = session_vwap(_frame(closes, weight), stall_bars=3)
        even = session_vwap(_frame(closes, 1.0), stall_bars=3)

        assert heavy is not None and even is not None
        # The ten bars at 110 carry almost all the volume, so VWAP sits near
        # them rather than near the forty quiet ones.
        assert heavy.vwap > even.vwap + 5.0

    def test_no_volume_is_refused_rather_than_silently_averaged(self) -> None:
        """Falling back to equal weights computes a different statistic and
        calls it VWAP. A reader downstream cannot tell the difference."""
        assert session_vwap(_frame(np.full(50, 100.0), 0.0), stall_bars=3) is None

    def test_a_session_too_young_to_have_a_shape_says_so(self) -> None:
        """VWAP over four bars is the price, and a deviation from it is zero by
        construction."""
        assert session_vwap(_frame(np.full(5, 100.0)), stall_bars=3) is None


class TestItFadesAStretchThatHasStopped:
    def test_a_stalled_stretch_above_vwap_is_a_short(self) -> None:
        engine = VwapReversion(VwapReversionConfig())

        signal = engine.analyze(_context(_stretched(3.0, still_moving=False)))

        assert signal.score < 0, signal.reasoning
        assert "sigma above the session VWAP" in signal.reasoning

    def test_a_stalled_stretch_below_vwap_is_a_long(self) -> None:
        engine = VwapReversion(VwapReversionConfig())
        frame = _stretched(3.0, still_moving=False)
        frame = _frame(200.0 - frame["close"].to_numpy())

        signal = engine.analyze(_context(frame))

        assert signal.score > 0, signal.reasoning
        assert "below the session VWAP" in signal.reasoning

    def test_a_stretch_still_extending_is_refused(self) -> None:
        """THE WHOLE POINT. A price two sigma below VWAP during a release is
        not mispriced, it is repriced, and the distance alone cannot tell the
        two apart. Without this the module fades every trend it meets."""
        engine = VwapReversion(VwapReversionConfig())

        signal = engine.analyze(_context(_stretched(3.0, still_moving=True)))

        assert signal.score == 0
        assert "still extending" in signal.reasoning

    def test_price_near_vwap_is_not_a_setup(self) -> None:
        engine = VwapReversion(VwapReversionConfig())
        quiet = np.full(BARS, 100.0) + np.random.default_rng(5).normal(0, 0.05, BARS)

        signal = engine.analyze(_context(_frame(quiet)))

        assert signal.score == 0
        assert "under the" in signal.reasoning


class TestTheScoreIsBounded:
    def test_a_bigger_stretch_scores_higher(self) -> None:
        engine = VwapReversion(VwapReversionConfig())

        small = engine.analyze(_context(_stretched(2.5, still_moving=False)))
        large = engine.analyze(_context(_stretched(6.0, still_moving=False)))

        assert abs(large.score) >= abs(small.score)

    def test_an_absurd_print_cannot_outrank_the_population_it_was_built_for(self) -> None:
        """A five-sigma move is usually a data fault or a halt. Saturation is
        what stops the worst input producing the strongest signal."""
        config = VwapReversionConfig()
        engine = VwapReversion(config)

        signal = engine.analyze(_context(_stretched(40.0, still_moving=False)))

        assert abs(signal.score) <= config.maximum_score
        assert signal.confidence <= config.maximum_confidence

    def test_a_bare_minimum_stretch_scores_under_the_live_threshold(self) -> None:
        """It should need a second reader to agree. Only a real outlier speaks
        for itself, and `score_threshold` on this account is 35."""
        assert VwapReversionConfig().base_score < 35.0


class TestItIsWiredInButNotLoose:
    def test_the_runner_builds_it(self) -> None:
        """A module nothing constructs is a file, not a detector."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from runner.service import build_analysis_modules

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

        assert "vwap_reversion" in {m.name for m in build_analysis_modules(settings)}

    def test_it_carries_a_weight_so_the_backtest_can_grade_it(self) -> None:
        """WITHOUT A WEIGHT IT IS UNMEASURABLE. It would not vote in the
        backtest either, appear in no table, and be a detector that cannot by
        construction ever be judged -- which is exactly what happened to the
        three M1 modules for months."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        confluence = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence

        assert confluence.weights.get("vwap_reversion", 0.0) > 0.0

    def test_and_it_cannot_touch_real_money_yet(self) -> None:
        """The weight is not permission. `drift_burst` traded nothing for its
        whole life and was graded on observation; this inherits that, and a
        tenth detector that loses is worth less than nothing."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from config.schema import TradingMode

        confluence = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence

        assert "vwap_reversion" not in confluence.live_enabled_modules
        assert confluence.effective_weights(TradingMode.MICRO_LIVE)["vwap_reversion"] == 0.0

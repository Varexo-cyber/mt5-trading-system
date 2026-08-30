"""Section two, and the first strategy here that earned its place.

Ninety-four detectors over sixteen instruments and six timeframes, 2010-2022.
Survived a Bonferroni bar across the whole grid, a holdout that had to reach
two sigma on its own, a sigma clustered by day so simultaneous signals across
correlated pairs stop counting as independent, and a coin-flip control that
turned out NOT to read zero and had to be subtracted from everything:

    M15   18,828 trades   hit 67.9%   net +0.279R  (train +0.272 / test +0.285)
    H1     5,235 trades   hit 67.9%   net +0.281R  (train +0.276 / test +0.284)

Positive in all eleven years, +0.181R to +0.394R.

These tests do not re-measure that; `scripts/measure_edges.py` does. They guard
the five things that would leave the module measuring one trade and sending
another -- and each of them is worth a specific, quantified amount:

    the impulse filter      without it the same retest nets roughly zero
    the 1:1 target          at 3:1 the same setups net +0.016R
    the 0.15 ATR tolerance  by 0.60 ATR out the edge changes sign
    the 1.00 ATR stop       below 0.8 the confluence floor silently widens it
    gold stays out          works there (+0.324R gross), unaffordable (-0.099R)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.impulse_retest import ImpulseRetest
from config.schema import ImpulseRetestConfig
from core.types import MarketContext, Series, Tick, Timeframe

HALF_BAR = 0.25


def _frame(closes: np.ndarray) -> pd.DataFrame:
    index = pd.date_range("2026-08-28", periods=len(closes), freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + HALF_BAR,
            "low": closes - HALF_BAR,
            "close": closes,
            "tick_volume": 100,
            "spread": 2,
        },
        index=index,
    )


def _context(frame: pd.DataFrame) -> MarketContext:
    last = float(frame["close"].iloc[-1])
    when = frame.index[-1].to_pydatetime()
    return MarketContext(
        "TEST",
        when,
        {Timeframe.M15: Series("TEST", Timeframe.M15, frame, when)},
        Tick("TEST", when, bid=last - 0.01, ask=last + 0.01),
    )


def _atr_of(frame: pd.DataFrame) -> float:
    previous = frame["close"].shift(1)
    spans = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(spans.rolling(14).mean().iloc[-1])


def _broke_and_returned(impulse_atr: float, back_to_atr: float, *, down: bool = False):
    """A range, a break of a stated size, then a return to a stated distance.

    Both distances are given in ATR and converted here, because that is the
    unit the module works in -- a hard-coded price stops meaning what the test
    says the moment the fixture's volatility changes.

    The base oscillates rather than ramps: with bars a quarter-point tall, a
    twenty-bar high sits above the last close by that much, so a slow climb can
    never clear its own channel and the fixture would produce no breaks at all.
    """
    sign = -1.0 if down else 1.0
    base = 100.0 + sign * 0.5 * np.sin(np.linspace(0, 12 * np.pi, 130))
    probe = _frame(np.concatenate([base, [100.0 + sign * 2.0]]))
    level = (
        float(probe["high"].iloc[-21:-1].max())
        if not down
        else float(probe["low"].iloc[-21:-1].min())
    )
    unit = _atr_of(probe)
    breakout = np.array([level + sign * impulse_atr * unit])
    tail = np.linspace(float(breakout[0]), level + sign * back_to_atr * unit, 10)
    return _frame(np.concatenate([base, breakout, tail]))


class TestTheImpulseFilterIsTheStrategy:
    def test_a_decisive_break_retested_is_a_signal(self) -> None:
        engine = ImpulseRetest(ImpulseRetestConfig())

        signal = engine.analyze(_context(_broke_and_returned(1.4, 0.05)))

        assert signal.score > 0, signal.reasoning
        assert "retested to within" in signal.reasoning

    def test_a_weak_break_retested_is_not(self) -> None:
        """THE MEASURED DISTINCTION, and the whole reason this module exists
        rather than the plain retest. The same return to the same level, after
        a break that only poked through, nets roughly zero."""
        engine = ImpulseRetest(ImpulseRetestConfig())

        signal = engine.analyze(_context(_broke_and_returned(0.4, 0.05)))

        assert signal.score == 0
        assert "no unspent" in signal.reasoning

    def test_the_shipped_threshold_is_a_full_atr(self) -> None:
        assert ImpulseRetestConfig().minimum_impulse_atr == pytest.approx(1.0)

    def test_a_bigger_impulse_scores_higher(self) -> None:
        """Measured monotone, so the score has to say so rather than treating
        every qualifying break as identical."""
        engine = ImpulseRetest(ImpulseRetestConfig())

        small = engine.analyze(_context(_broke_and_returned(1.05, 0.05)))
        large = engine.analyze(_context(_broke_and_returned(2.6, 0.05)))

        assert abs(large.score) > abs(small.score)


class TestItWantsTheLevelAndNotAPullback:
    def test_a_shallow_dip_from_the_high_is_refused(self) -> None:
        engine = ImpulseRetest(ImpulseRetestConfig())

        signal = engine.analyze(_context(_broke_and_returned(1.4, 1.1)))

        assert signal.score == 0
        assert "above the level it broke" in signal.reasoning

    def test_the_tolerance_is_the_measured_one(self) -> None:
        """Monotone and sign-changing: 0.15 ATR is the edge, 0.60 ATR is a
        losing trade. Slack here is not tolerance, it is the edge given away."""
        assert ImpulseRetestConfig().tolerance_atr == pytest.approx(0.15)

    def test_a_short_measures_the_level_the_other_way(self) -> None:
        engine = ImpulseRetest(ImpulseRetestConfig())

        signal = engine.analyze(_context(_broke_and_returned(1.4, 0.05, down=True)))

        assert signal.score < 0, signal.reasoning

    def test_a_level_price_has_abandoned_is_not_offered(self) -> None:
        engine = ImpulseRetest(ImpulseRetestConfig())

        signal = engine.analyze(_context(_broke_and_returned(1.4, -3.0)))

        assert signal.score <= 0, signal.reasoning


class TestWhatIsSentIsWhatWasMeasured:
    def test_the_stop_clears_the_confluence_floor_untouched(self) -> None:
        """`min_stop_atr` WIDENS a stop below it, silently. 0.50 ATR measured
        better (+0.242R at a 2R target) and would have been floored to 0.8 --
        a strategy nobody tested. 1.00 ATR passes through."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        config = settings.analysis.impulse_retest

        assert (
            config.tolerance_atr + config.stop_beyond_atr
            >= settings.analysis.confluence.min_stop_atr
        )

    def test_the_family_trades_one_to_one(self) -> None:
        """The account plans 3.0. This family measured +0.279R at 1:1 and
        +0.016R at 3:1, so shipping it under the account default would trade a
        measured strategy at nearly nothing."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        confluence = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence

        assert confluence.target_r_multiple_by_family.get("impulse_retest") == pytest.approx(1.0)

    def test_the_override_actually_reaches_the_target(self) -> None:
        """A config nothing reads is the defect this account keeps producing.
        `_reachable_target` defaults the family to an empty string, so a
        missing wire would leave every test above green and every trade out at
        3R."""
        import inspect

        from analysis import confluence

        source = " ".join(inspect.getsource(confluence).split())

        assert "target_r_multiple_by_family.items()" in source
        assert "setup_family=setup_family," in source

    def test_it_publishes_the_stop_it_was_measured_with(self) -> None:
        engine = ImpulseRetest(ImpulseRetestConfig())

        signal = engine.analyze(_context(_broke_and_returned(1.4, 0.05)))

        assert signal.invalidation_price is not None
        assert signal.key_levels
        assert signal.invalidation_price < signal.key_levels[0]


class TestTheLiveWiring:
    def test_the_runner_builds_it(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from runner.service import build_analysis_modules

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

        assert "impulse_retest" in {m.name for m in build_analysis_modules(settings)}

    def test_it_is_live_and_weighted(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        confluence = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence

        assert "impulse_retest" in confluence.live_enabled_modules
        assert confluence.weights.get("impulse_retest", 0.0) > 0.0

    def test_it_has_a_breaker_matched_to_its_expected_hit_rate(self) -> None:
        """68% expected, so 45% losers over forty trades is about four sigma
        the wrong way -- a measurement that does not hold on this feed rather
        than a bad run."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        breakers = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).risk.section_breakers

        assert "impulse_retest" in breakers
        assert breakers["impulse_retest"].maximum_loss_share <= 0.5

    def test_it_reads_the_timeframe_it_was_measured_on(self) -> None:
        assert ImpulseRetestConfig().timeframe == "M15"

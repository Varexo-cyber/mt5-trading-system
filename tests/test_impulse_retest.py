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
    gold's wider stop       at the family stop its spread is 20% of R and the
                            live gate refuses it; at 1.50 ATR it is 6.7%
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


class TestGoldGetsAStopItCanAfford:
    """The owner asked for gold on 30 August and was right that it belongs --
    but not for the reason given.

    "Minimal risk" does not make an unaffordable trade affordable. Cost in R is
    spread/R, a RATIO: halve the lot and the win, the loss and the spread all
    halve together, leaving the R-multiple untouched. Lot size decides what a
    trade pays in euros, never whether it pays.

    What does decide it is the denominator. Gold carries ~0.10 ATR of spread
    against 0.04 on a major, so at a 1.00 ATR stop it spends 20% of R on
    execution and `max_spread_share_of_stop: 0.08` refuses it. At 1.50 ATR the
    same spread is 6.7% and the gate passes it untouched.

        stop 1.50 ATR, 1R target, 1,942 trades
        gross +0.214R   cost 0.133R   net +0.083R   (train +0.065/test +0.100)
    """

    def test_gold_carries_a_wider_stop_than_the_majors(self) -> None:
        config = ImpulseRetestConfig()

        gold = config.stop_beyond_atr_by_symbol.get("XAUUSD")

        assert gold is not None
        assert gold > config.stop_beyond_atr

    def test_the_wider_stop_clears_the_live_spread_gate(self) -> None:
        """THE WHOLE REASON FOR THE OVERRIDE. Gold at the family's own stop
        would be refused by `max_spread_share_of_stop`, so it must be given a
        stop it fits inside rather than the gate being loosened around it."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        config = settings.analysis.impulse_retest
        gold_r = config.tolerance_atr + config.stop_beyond_atr_by_symbol["XAUUSD"]
        # Gold's spread measured as a share of ATR in the study.
        assert 0.10 / gold_r <= settings.analysis.confluence.max_spread_share_of_stop

    def test_a_major_is_untouched_by_the_override(self) -> None:
        config = ImpulseRetestConfig()

        assert "EURUSD" not in config.stop_beyond_atr_by_symbol

    def test_the_override_actually_reaches_the_stop(self) -> None:
        """A per-symbol setting nothing reads is the defect this account keeps
        producing. Two contexts, same bars, different symbol names: the
        published invalidation must differ."""
        engine = ImpulseRetest(ImpulseRetestConfig(stop_beyond_atr_by_symbol={"XAUUSD": 1.35}))
        frame = _broke_and_returned(1.4, 0.05)

        major = _context(frame)
        gold = MarketContext("XAUUSD", major.now, major.series, major.tick)
        wide = engine.analyze(gold)
        narrow = engine.analyze(major)

        assert wide.invalidation_price is not None
        assert narrow.invalidation_price is not None
        assert wide.invalidation_price < narrow.invalidation_price


class TestItCanActuallyTradeOnItsOwn:
    """THE MEASURED STRATEGY IS A STANDALONE ONE.

    18,828 trades at 67.9% were taken on this signal alone -- no second module
    agreeing, no confluence. If the live engine will only let it trade when
    something else happens to concur, what runs is not what was measured.

    `lone_module_minimum_confidence` is 0.65 on this account. The module first
    shipped with a base confidence of 0.58, and because the fill sits at
    `level + tolerance` the `closeness` term is near zero on almost every real
    signal -- so a typical setup landed around 0.60 and was silently refused.
    It would have looked like the strategy just not firing.
    """

    def test_the_weakest_qualifying_setup_still_clears_the_lone_gate(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        config = settings.analysis.impulse_retest

        assert config.base_confidence >= settings.analysis.confluence.lone_module_minimum_confidence

    def test_the_weakest_qualifying_setup_still_clears_the_score_bar(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        confluence = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence
        config = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.impulse_retest

        # A lone module's score is |raw| x confidence.
        assert config.base_score * config.base_confidence >= confluence.score_threshold

    def test_a_real_signal_at_the_tolerance_edge_clears_it(self) -> None:
        """Not the config in the abstract -- an actual signal off actual bars,
        filled where the limit really rests."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        confluence = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence
        engine = ImpulseRetest(ImpulseRetestConfig())

        signal = engine.analyze(_context(_broke_and_returned(1.05, 0.10)))

        assert signal.score != 0, signal.reasoning
        assert signal.confidence >= confluence.lone_module_minimum_confidence
        assert abs(signal.score) * signal.confidence >= confluence.score_threshold

    def test_the_confidence_is_anchored_to_the_measured_hit_rate(self) -> None:
        """0.68, because that is what it hits. Not a preference."""
        assert ImpulseRetestConfig().base_confidence == pytest.approx(0.68, abs=0.01)


class TestTheLiveWiring:
    def test_the_runner_builds_it(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from runner.service import build_analysis_modules

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

        assert "impulse_retest" in {m.name for m in build_analysis_modules(settings)}

    def test_it_is_shadowed_but_still_weighted(self) -> None:
        """OFF SINCE 30 AUGUST, on thirty days of this broker's own data:
        48 trades, 39.6% win, -10.00R, EUR -68.05, beside order_block's 204
        trades at 54.4% and +85.14. Run alone order_block was +39.5% over the
        window; the pair was +7.9%. Section two was eating three quarters of
        what section three earned.

        NOT proven bad -- 48 trades at 39.6% is -1.44 sigma and a neutral
        strategy does that regularly -- but it has never had a sample worth
        reading (7, 9, 10, 48 trades), and "not proven bad" does not earn real
        money.

        AND IT CAME BACK ON, 31 August, on exactly that measurement: 358
        trades over 180 days, 81.3% win, +0.058 R a trade against
        order_block's +0.026, 89 of 116 days green. Keeping the weight while
        it was off is what made that measurement possible -- a module with no
        weight cannot be judged, which is how three M1 detectors went unjudged
        for months."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        confluence = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence

        assert "impulse_retest" not in confluence.live_enabled_modules
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

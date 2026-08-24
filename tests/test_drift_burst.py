"""Section two: is this move real, or is it someone in a hurry?

The first reader on this account that does not look at the shape of the price
series. It runs a hypothesis test on it and then fades the answer, which is what
makes it the only candidate that can ever be a genuine second family for the
nine detectors that all follow.

The tests that matter here are the two that decide whether the thing works at
all: what it does on pure noise, and whether it can see a burst that is short
and violent rather than long and gentle. The first version could not, and no
amount of reading the code would have said so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.drift_burst import DriftBurst, burst_statistic
from config.schema import DriftBurstConfig

DRIFT_WINDOW = 10
VOL_WINDOW = 120
HALF_LIFE = 5.0
LAGS = 2
BAR_SIGMA = 0.0004


def path(returns: np.ndarray, start: float = 100.0) -> pd.Series:
    return pd.Series(start * np.exp(np.cumsum(np.r_[0.0, returns])))


def statistic(returns: np.ndarray):  # type: ignore[no-untyped-def]
    return burst_statistic(
        path(returns),
        drift_window=DRIFT_WINDOW,
        volatility_window=VOL_WINDOW,
        half_life=HALF_LIFE,
        noise_lags=LAGS,
    )


def quiet_then_burst(rng, bars: int, per_bar: float) -> np.ndarray:  # type: ignore[no-untyped-def]
    calm = rng.normal(0.0, BAR_SIGMA, VOL_WINDOW - bars)
    run = rng.normal(per_bar, BAR_SIGMA, bars)
    return np.r_[calm, run]


class TestTheStatisticOnNoise:
    """The threshold is 4.0 because it was MEASURED against this, not because
    the paper says 4. Over 4,000 random walks at these settings the statistic
    reaches a 99th percentile near 2.6 and a maximum near 3.6."""

    def test_pure_noise_stays_well_under_the_threshold(self) -> None:
        rng = np.random.default_rng(1234)
        readings = [statistic(rng.normal(0.0, BAR_SIGMA, VOL_WINDOW)) for _ in range(600)]
        values = np.array([abs(r.t_stat) for r in readings if r is not None])

        assert len(values) == 600
        assert np.percentile(values, 99) < 3.5
        assert values.max() < DriftBurstConfig().t_threshold + 1.0

    def test_a_flat_series_produces_no_reading_rather_than_a_zero(self) -> None:
        """ "No reading" and "a reading of zero" are different statements and
        only one of them is evidence."""
        assert statistic(np.zeros(VOL_WINDOW)) is None

    def test_too_little_history_produces_no_reading(self) -> None:
        assert statistic(np.full(VOL_WINDOW // 2, 0.0001)) is None


class TestItCanSeeAShortViolentBurst:
    """THE BUG THIS CLASS EXISTS FOR.

    Written with a single window the test was blind to exactly the shape it was
    built to find: measured over 3,000 synthetic paths, a 20-bar burst fired 68%
    of the time while a 5-bar burst covering 127 basis points — harder and
    faster — fired 0.0%. Never once. The burst was inflating the very
    denominator meant to judge it.
    """

    def test_a_five_bar_burst_fires(self) -> None:
        rng = np.random.default_rng(99)
        fired = sum(
            1 for _ in range(200) if abs(statistic(quiet_then_burst(rng, 5, 0.0015)).t_stat) >= 4.0
        )

        assert fired / 200 > 0.85

    def test_a_three_bar_burst_fires_when_it_is_hard_enough(self) -> None:
        rng = np.random.default_rng(100)
        fired = sum(
            1 for _ in range(200) if abs(statistic(quiet_then_burst(rng, 3, 0.0025)).t_stat) >= 4.0
        )

        assert fired / 200 > 0.85

    def test_the_volatility_window_must_be_longer_than_the_drift_window(self) -> None:
        """The whole correction, as a refusal rather than a comment."""
        reading = burst_statistic(
            path(np.full(VOL_WINDOW, 0.0005)),
            drift_window=30,
            volatility_window=30,
            half_life=HALF_LIFE,
            noise_lags=LAGS,
        )

        assert reading is None

    def test_a_wild_market_that_merely_drifts_is_not_a_burst(self) -> None:
        """The reading has to separate "moved a long way" from "moved further
        than its own volatility explains". A market whose every bar is violent
        goes a long way and says nothing."""
        rng = np.random.default_rng(7)
        wild = rng.normal(0.0012, 0.006, VOL_WINDOW)
        reading = statistic(wild)

        assert reading is not None
        assert abs(reading.move_bp) > 25.0  # it travelled
        assert abs(reading.t_stat) < 4.0  # and it means nothing


class TestTheSignal:
    @staticmethod
    def context(returns: np.ndarray):  # type: ignore[no-untyped-def]
        from datetime import UTC

        from core.types import MarketContext, Series, Timeframe

        closes = path(returns)
        index = pd.date_range("2026-08-24", periods=len(closes), freq="1min", tz=UTC)
        frame = pd.DataFrame(
            {
                "open": closes.to_numpy(),
                "high": closes.to_numpy() * 1.00002,
                "low": closes.to_numpy() * 0.99998,
                "close": closes.to_numpy(),
                "tick_volume": 100,
                "spread": 2,
                "real_volume": 0,
            },
            index=index,
        )
        now = index[-1].to_pydatetime()
        return MarketContext(
            symbol="EURUSD",
            now=now,
            series={Timeframe.M1: Series("EURUSD", Timeframe.M1, frame, now)},
            tick=None,
        )

    def test_it_fades_an_upward_burst(self) -> None:
        """The entire thesis in one assertion. Two thirds of these revert, so a
        burst UP is a reason to be short — the opposite of every other detector
        on this account, which is what makes it a second opinion at all."""
        rng = np.random.default_rng(3)
        signal = DriftBurst().analyze(self.context(quiet_then_burst(rng, 6, 0.002)))

        assert signal.details["t_stat"] > 0  # the burst ran up
        assert signal.score < 0  # and the module says short

    def test_it_fades_a_downward_burst(self) -> None:
        rng = np.random.default_rng(4)
        signal = DriftBurst().analyze(self.context(-quiet_then_burst(rng, 6, 0.002)))

        assert signal.details["t_stat"] < 0
        assert signal.score > 0

    def test_a_quiet_market_gets_no_opinion(self) -> None:
        rng = np.random.default_rng(5)
        signal = DriftBurst().analyze(self.context(rng.normal(0.0, BAR_SIGMA, VOL_WINDOW)))

        assert signal.score == 0.0
        assert signal.confidence == 0.0

    def test_a_large_t_on_a_tiny_move_is_refused(self) -> None:
        """The statistic is a RATIO, so a dead-quiet instrument that twitches
        produces a large t on a two-basis-point move. The research describes
        25-200bp events; under that floor the reversion has nothing in it to
        collect and the spread takes what is left."""
        rng = np.random.default_rng(6)
        tiny = np.r_[
            rng.normal(0.0, 0.000002, VOL_WINDOW - 6),
            np.full(6, 0.00002),
        ]
        signal = DriftBurst().analyze(self.context(tiny))

        assert abs(signal.details["t_stat"]) >= 4.0
        assert abs(signal.details["move_bp"]) < 25.0
        assert signal.score == 0.0
        assert "bp" in signal.reasoning

    def test_the_reading_shows_its_working(self) -> None:
        """A fade this contrarian has to be auditable months later, or the next
        person reading the journal cannot tell a burst from a rounding error."""
        rng = np.random.default_rng(8)
        signal = DriftBurst().analyze(self.context(quiet_then_burst(rng, 6, 0.002)))

        for key in ("t_stat", "drift_per_bar", "volatility_per_bar", "effective_bars", "move_bp"):
            assert key in signal.details, key

    def test_it_says_nothing_without_enough_bars(self) -> None:
        rng = np.random.default_rng(9)
        signal = DriftBurst().analyze(self.context(rng.normal(0.0, BAR_SIGMA, 40)))

        assert signal.score == 0.0
        assert "closed M1 bars" in signal.reasoning

    def test_disabling_it_silences_it(self) -> None:
        rng = np.random.default_rng(10)
        module = DriftBurst(DriftBurstConfig(enabled=False))
        signal = module.analyze(self.context(quiet_then_burst(rng, 6, 0.002)))

        assert signal.score == 0.0


class TestItCannotReachAnOrder:
    """Section two runs as paper, and that is a property of the wiring rather
    than a promise in a comment."""

    def _confluence(self):  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence

    def test_it_is_weighted_so_the_backtest_can_see_it(self) -> None:
        assert self._confluence().weights["drift_burst"] > 0

    def test_live_zeroes_it_before_the_engine_scores_anything(self) -> None:
        from core.types import TradingMode

        confluence = self._confluence()

        assert "drift_burst" not in confluence.live_enabled_modules
        assert confluence.effective_weights(TradingMode.MICRO_LIVE)["drift_burst"] == 0.0

    def test_it_is_its_own_evidence_family(self) -> None:
        """Filing it under an existing family would let it corroborate a reader
        it has nothing in common with — the exact failure families exist to
        prevent. It reads immediacy; nothing else here does."""
        from analysis.evidence_families import family_for

        family = family_for("drift_burst")

        assert family == "immediacy"
        for other in ("trend_momentum", "impulse_break", "market_structure", "liquidity_sweep"):
            assert family_for(other) != family


class TestTheConfigCannotBeSetToSomethingIncoherent:
    def test_the_volatility_window_may_not_be_the_drift_window(self) -> None:
        with pytest.raises(ValueError, match="must exceed drift_window"):
            DriftBurstConfig(drift_window=60, volatility_window=60)

    def test_saturation_must_leave_the_score_room_to_grow(self) -> None:
        with pytest.raises(ValueError, match="must exceed t_threshold"):
            DriftBurstConfig(t_threshold=6.0, t_saturation=5.0)

    def test_confidence_may_not_narrow_to_nothing(self) -> None:
        with pytest.raises(ValueError, match="below base_confidence"):
            DriftBurstConfig(base_confidence=0.8, maximum_confidence=0.5)


class TestTheObserverRecordsWithoutTrading:
    """What section two would have done, written down and resolved later by the
    same machinery that has graded blocked setups for months.

    It hangs off `_record_skip` deliberately. `drift_burst` FADES, so it fires
    most often on exactly the symbols the engine declined — hooking it to the
    traded path would have measured it only on the setups it had nothing to do
    with.
    """

    def _runner(self, recorded: list):  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from runner.service import JarvisRunner

        runner = object.__new__(JarvisRunner)
        runner.settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        runner._cycle_contexts = {}
        runner.recorder = SimpleNamespace(  # type: ignore[assignment]
            has_unresolved_shadow_trade=lambda *_: False,
            record_shadow_trade=lambda **kw: recorded.append(kw) or 1,
        )
        return runner

    def _context(self, rng, bars: int = 6, per_bar: float = 0.002):  # type: ignore[no-untyped-def]
        from core.types import Tick

        context = TestTheSignal.context(quiet_then_burst(rng, bars, per_bar))
        frame = context.series[next(iter(context.series))].df
        last = float(frame["close"].iloc[-1])
        return context.__class__(
            symbol=context.symbol,
            now=context.now,
            series=context.series,
            tick=Tick(context.symbol, context.now, last * 0.99995, last * 1.00005),
        )

    def test_a_burst_is_written_down_as_a_shadow_trade(self) -> None:
        recorded: list = []
        runner = self._runner(recorded)
        context = self._context(np.random.default_rng(3))
        runner._cycle_contexts["EURUSD"] = context
        signals = [DriftBurst().analyze(context)]

        runner._observe_section_two(42, "EURUSD", signals)

        assert len(recorded) == 1
        assert str(recorded[0]["blocked_by"]) == "SECTION_2_OBSERVED"
        assert recorded[0]["symbol"] == "EURUSD"

    def test_the_plan_fades_the_burst_and_is_the_right_way_round(self) -> None:
        """A burst up is recorded as a SHORT whose stop sits above entry and
        whose target sits below it. Getting this backwards would record the
        opposite of the thesis and quietly measure the wrong strategy."""
        recorded: list = []
        runner = self._runner(recorded)
        context = self._context(np.random.default_rng(3))
        runner._cycle_contexts["EURUSD"] = context

        runner._observe_section_two(42, "EURUSD", [DriftBurst().analyze(context)])

        plan = recorded[0]
        assert plan["direction"].name == "SHORT"
        assert plan["sl"] > plan["entry_price"] > plan["tp"]

    def test_a_quiet_market_records_nothing(self) -> None:
        recorded: list = []
        runner = self._runner(recorded)
        rng = np.random.default_rng(5)
        context = TestTheSignal.context(rng.normal(0.0, BAR_SIGMA, VOL_WINDOW))
        runner._cycle_contexts["EURUSD"] = context

        runner._observe_section_two(42, "EURUSD", [DriftBurst().analyze(context)])

        assert recorded == []

    def test_it_never_records_twice_for_one_open_observation(self) -> None:
        recorded: list = []
        runner = self._runner(recorded)
        runner.recorder.has_unresolved_shadow_trade = lambda *_: True
        context = self._context(np.random.default_rng(3))
        runner._cycle_contexts["EURUSD"] = context

        runner._observe_section_two(42, "EURUSD", [DriftBurst().analyze(context)])

        assert recorded == []

    def test_a_symbol_with_no_context_this_cycle_is_skipped(self) -> None:
        """`_record_skip` runs on paths where the scan never built a context —
        a data failure, for one — and reaching into an empty map there would
        turn a missing measurement into a crash in the live loop."""
        recorded: list = []
        runner = self._runner(recorded)

        runner._observe_section_two(42, "EURUSD", [])

        assert recorded == []

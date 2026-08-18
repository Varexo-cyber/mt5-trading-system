"""Judge the plan against the stop it will actually carry.

The runner widens the stop for costs a few lines before it asks whether the
target is reachable — on a €160 account, on nearly every candidate, because a
two-pip stop spends most of its risk on commission and slippage. Widening does
two things at once:

    it LOWERS reward-to-risk        the target did not move, the stop did
    it makes the trade HARDER TO STOP OUT   which is the entire point of it

The reach gate only ever saw the first. It measured an unconditional reach rate
— how often price ever covered the distance, stop or no stop — which the wider
stop cannot change, and compared it against a break-even requirement the wider
stop had just raised. So widening could only ever count against a trade, and
the mechanism that exists to make small-account trades viable was quietly
disqualifying them.

The measurement that fixes it was already in the codebase. The confluence
engine will not propose a target without positive first-touch expectancy; the
runner simply never re-asked with the real stop in hand. Both now call the same
function, which is the other half of the fix: two copies of a statistic that
must agree is how the two gates came to disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.target_reach import first_touch_runs, measure, measure_first_touch


def bars(highs: list[float], lows: list[float], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes})


def walk(bars_count: int = 500, noise: float = 0.0004, seed: int = 11) -> pd.DataFrame:
    """A random walk that whips both ways — where the two statistics differ most."""
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0.0, noise, bars_count))
    return pd.DataFrame(
        {"open": close, "high": close + 0.0003, "low": close - 0.0003, "close": close}
    )


class TestTheStopEndsTheWindow:
    def test_a_dip_through_the_stop_discards_the_rally_that_followed(self) -> None:
        """The window that made the old number wrong. Price falls a full R on
        bar one and rallies three R afterwards; the trade was closed on bar one
        and the rally belongs to somebody else."""
        frame = bars(
            highs=[100.0, 100.0, 104.0, 104.0],
            lows=[100.0, 98.0, 100.0, 100.0],
            closes=[100.0, 99.0, 103.0, 103.0],
        )

        runs = first_touch_runs(frame, risk=2.0, bars_ahead=3, long=True)

        assert runs is not None
        assert runs[0] == pytest.approx(0.0)

    def test_the_same_window_counts_as_a_win_once_the_stop_is_wide_enough(self) -> None:
        """Identical price history, wider stop, opposite verdict — this is the
        credit the old gate never gave."""
        frame = bars(
            highs=[100.0, 100.0, 104.0, 104.0],
            lows=[100.0, 98.0, 100.0, 100.0],
            closes=[100.0, 99.0, 103.0, 103.0],
        )

        runs = first_touch_runs(frame, risk=3.0, bars_ahead=3, long=True)

        assert runs is not None
        assert runs[0] == pytest.approx(4.0)

    def test_it_says_nothing_rather_than_guessing(self) -> None:
        """No history, no stop, no columns: the question was not answerable,
        and a fabricated number is worse than none."""
        assert first_touch_runs(walk(), risk=0.0, bars_ahead=24, long=True) is None
        assert first_touch_runs(walk(bars_count=10), risk=0.01, bars_ahead=24, long=True) is None
        assert (
            first_touch_runs(pd.DataFrame({"close": [1.0]}), risk=1.0, bars_ahead=1, long=True)
            is None
        )
        assert (
            measure_first_touch(walk(), distance=0.0, risk=0.01, bars_ahead=24, long=True) is None
        )


class TestWideningTheStopIsCreditedAndNotOnlyCharged:
    """Same market, same target, two stops. The gate must move the way the
    trade actually moved."""

    def _verdicts(self, risk: float):  # type: ignore[no-untyped-def]
        frame = walk()
        return (
            measure(frame, distance=0.0020, bars_ahead=24, long=True, reward_risk=0.0020 / risk),
            measure_first_touch(frame, distance=0.0020, risk=risk, bars_ahead=24, long=True),
        )

    def test_the_unconditional_rate_cannot_see_the_stop_at_all(self) -> None:
        tight, _ = self._verdicts(0.0005)
        wide, _ = self._verdicts(0.0015)

        assert tight.forward_pct == pytest.approx(wide.forward_pct)
        # ...while the bar it is measured against moves a long way. That gap,
        # and nothing about the market, is what refused the widened trades.
        assert wide.required_pct > tight.required_pct

    def test_the_first_touch_rate_rises_with_the_stop_that_bought_it(self) -> None:
        _, tight = self._verdicts(0.0005)
        _, wide = self._verdicts(0.0015)

        assert tight is not None and wide is not None
        assert wide.forward_pct > tight.forward_pct

    def test_a_tight_stop_is_judged_more_harshly_than_before_not_less(self) -> None:
        """This is not a loosening. On a stop the market goes through, the
        honest measurement is the stricter one."""
        unconditional, first_touch = self._verdicts(0.0005)

        assert first_touch is not None
        assert first_touch.forward_pct < unconditional.forward_pct

    def test_the_requirement_is_the_hit_rate_that_returns_zero(self) -> None:
        verdict = measure_first_touch(
            walk(), distance=0.0020, risk=0.0010, bars_ahead=24, long=True, cost_r=0.1
        )

        assert verdict is not None
        assert verdict.reward_risk == pytest.approx(2.0)
        assert verdict.required_pct == pytest.approx(100.0 * 1.1 / 3.0, abs=0.01)

    def test_expectancy_and_the_stated_requirement_agree(self) -> None:
        """`clears_break_even` is written as a rate comparison so an operator
        margin can bite; with no margin it must still be the same statement as
        `expected_r > 0`, or the refusal text describes a different test than
        the one that ran."""
        for risk in (0.0004, 0.0008, 0.0012, 0.0020):
            verdict = measure_first_touch(
                walk(), distance=0.0020, risk=risk, bars_ahead=24, long=True, cost_r=0.05
            )
            assert verdict is not None
            assert verdict.clears_break_even is (verdict.expected_r > 0.0)

    def test_the_refusal_names_the_stop_it_was_measured_against(self) -> None:
        """The operator reading the journal has to be able to check it, and the
        old text could not distinguish a plan from its widened twin."""
        text = measure_first_touch(
            walk(), distance=0.0020, risk=0.0010, bars_ahead=24, long=True, cost_r=0.1
        ).describe()

        assert "before the 2.00RR stop" in text
        assert "break even" in text
        assert "R" in text


class TestTheRunnerAsksItOfTheStopItWillSend:
    """The live sequence, in order: engine approves a plan, `_widen_stop_for_costs`
    pushes the stop out so the commission is not the trade, and only then is the
    target questioned. The question has to be about the widened plan."""

    @staticmethod
    def _runner():  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from runner.service import JarvisRunner

        runner = object.__new__(JarvisRunner)
        runner.settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        return runner

    @staticmethod
    def _context(frame: pd.DataFrame, entry: float):  # type: ignore[no-untyped-def]
        from datetime import UTC, datetime

        from core.types import MarketContext, Series, Tick, Timeframe

        now = datetime(2026, 8, 18, 10, 0, tzinfo=UTC)
        return MarketContext(
            symbol="EURUSD.i",
            now=now,
            series={
                Timeframe.H1: Series(
                    symbol="EURUSD.i", timeframe=Timeframe.H1, df=frame, fetched_at=now
                )
            },
            tick=Tick(symbol="EURUSD.i", time=now, bid=entry - 0.00005, ask=entry + 0.00005),
        )

    @staticmethod
    def _idea(stop_distance: float, entry: float, target: float):  # type: ignore[no-untyped-def]
        from analysis.confluence import TradeIdea
        from core.types import Direction

        return TradeIdea(
            symbol="EURUSD.i",
            approved=True,
            direction=Direction.LONG,
            score=44.0,
            confidence=0.8,
            entry=entry,
            stop_loss=entry - stop_distance,
            take_profit=target,
            reason="test",
            signals=(),
            horizon="intraday",
            planning_timeframe="H1",
            expected_horizon_minutes=24 * 60,
        )

    def test_the_widened_plan_is_measured_against_its_widened_stop(self) -> None:
        frame = walk()
        entry = float(frame["close"].iloc[-1])
        target = entry + 0.0020

        runner = self._runner()
        context = self._context(frame, entry)
        tight = runner._target_survival(context, self._idea(0.0005, entry, target))
        wide = runner._target_survival(context, self._idea(0.0015, entry, target))

        assert tight is not None and wide is not None
        assert wide.forward_pct > tight.forward_pct
        assert wide.reward_risk < tight.reward_risk

    def test_the_switch_returns_the_runner_to_the_old_unconditional_gate(self) -> None:
        """Off means off — no half-measure that leaves a second opinion running."""
        frame = walk()
        entry = float(frame["close"].iloc[-1])
        runner = self._runner()
        runner.settings = runner.settings.model_copy(
            update={
                "analysis": runner.settings.analysis.model_copy(
                    update={
                        "confluence": runner.settings.analysis.confluence.model_copy(
                            update={"first_touch_target_test": False}
                        )
                    }
                )
            }
        )

        verdict = runner._target_survival(
            self._context(frame, entry), self._idea(0.0010, entry, entry + 0.0020)
        )

        assert verdict is None

    def test_it_refuses_nothing_on_ignorance(self) -> None:
        """An instrument the measurement cannot reach has not failed the test,
        it has not taken it — and must be judged on everything else."""
        from core.types import MarketContext

        frame = walk()
        entry = float(frame["close"].iloc[-1])
        runner = self._runner()
        empty = MarketContext(
            symbol="EURUSD.i",
            now=self._context(frame, entry).now,
            series={},
            tick=None,
        )

        assert runner._target_survival(empty, self._idea(0.0010, entry, entry + 0.0020)) is None


class TestTheEnginePlansOnTheStopTheRunnerWillSend:
    """The same fix one step earlier, and the larger half of it.

    `_widen_stop_for_costs` fires on nearly every candidate on this account, so
    the analysis was choosing a target, measuring how often it is reached and
    computing an expectancy — all against a stop it was about to be told was
    too narrow to trade. 1,710 refusals in two live hours read "no target
    between 1.00R and 3.00R pays on this market" and were measured against a
    stop the broker would never have received.

    Nothing about the order changes: the runner widens to this same distance a
    few steps later either way, so the executed stop, the risk and the lot size
    are identical.
    """

    @staticmethod
    def _runner(*, planning: bool = True):  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from core.instrument import InstrumentSpec
        from runner.service import JarvisRunner
        from tests.fakes.fake_mt5 import eurusd_spec

        runner = object.__new__(JarvisRunner)
        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        # The live overlay has this off on its own measured evidence; the
        # mechanism is still tested, because turning it back on has to mean
        # measuring it and not rediscovering it.
        runner.settings = settings.model_copy(
            update={
                "analysis": settings.analysis.model_copy(
                    update={
                        "confluence": settings.analysis.confluence.model_copy(
                            update={"plan_on_cost_floor": planning}
                        )
                    }
                )
            }
        )
        spec = InstrumentSpec.from_mt5(eurusd_spec())
        runner.broker = type("_B", (), {"spec": staticmethod(lambda _s: spec)})()
        return runner, spec

    def test_the_live_overlay_leaves_the_analysis_planning_as_it_was(self) -> None:
        """One live hour: the refusal it was built to reduce did not move (855/h
        to 881/h) while setups fell 52/h to 37/h and runway failures went from
        6% to 16% of the setups that formed. Off until it can be measured
        against a clean baseline."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

        assert settings.analysis.confluence.plan_on_cost_floor is False

    def test_switched_off_the_engine_never_hears_about_the_floor(self) -> None:
        from datetime import UTC, datetime

        from core.types import MarketContext

        runner, _ = self._runner(planning=False)
        context = MarketContext(
            symbol="EURUSD", now=datetime(2026, 8, 18, tzinfo=UTC), series={}, tick=None
        )

        runner._attach_cost_floor("EURUSD", context)

        assert "min_stop_for_costs" not in context.meta

    def test_the_floor_is_the_distance_the_runner_would_have_widened_to(self) -> None:
        from analysis.confluence import TradeIdea
        from core.types import Direction

        runner, spec = self._runner()
        floor = runner._cost_implied_min_stop(spec)
        assert floor > 0

        # A stop far under the floor, put through the widening the runner does.
        idea = TradeIdea(
            symbol="EURUSD",
            approved=True,
            direction=Direction.LONG,
            score=44.0,
            confidence=0.8,
            entry=1.1000,
            stop_loss=1.1000 - floor / 10.0,
            take_profit=1.1050,
            reason="test",
            signals=(),
        )

        widened = runner._widen_stop_for_costs(idea, spec)

        # To the tick, because the stop is normalised onto the broker's price
        # grid on the way out.
        distance = widened.entry - widened.stop_loss
        assert distance == pytest.approx(floor, abs=spec.tick_size)

        # And the point of the exercise: the gate that refused the trade now
        # passes it. `_COST_MARGIN` exists so one tick of rounding cannot land
        # the stop a hair inside the boundary and lose to the very rule the
        # widening is there to satisfy.
        from risk.position_sizer import PositionSizer

        sizer = PositionSizer(runner.settings)
        commission = runner.settings.risk.commission_per_lot(spec.asset_class.value)
        assert (
            sizer._cost_share(spec, distance, commission)
            <= runner.settings.risk.max_cost_share_of_risk
        )

    def test_a_stop_already_wide_enough_is_left_alone(self) -> None:
        """The floor is a floor. It must never pull a structural stop inwards,
        and it must not move one that already clears the costs."""
        from analysis.confluence import TradeIdea
        from core.types import Direction

        runner, spec = self._runner()
        idea = TradeIdea(
            symbol="EURUSD",
            approved=True,
            direction=Direction.LONG,
            score=44.0,
            confidence=0.8,
            entry=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            reason="test",
            signals=(),
        )

        assert runner._widen_stop_for_costs(idea, spec).stop_loss == idea.stop_loss

    def test_the_engine_is_handed_the_floor_before_it_evaluates(self) -> None:
        from datetime import UTC, datetime

        from core.types import MarketContext

        runner, spec = self._runner()
        context = MarketContext(
            symbol="EURUSD", now=datetime(2026, 8, 18, tzinfo=UTC), series={}, tick=None
        )

        runner._attach_cost_floor("EURUSD", context)

        assert context.meta["min_stop_for_costs"] == pytest.approx(
            runner._cost_implied_min_stop(spec)
        )

    def test_a_symbol_whose_spec_cannot_be_read_plans_exactly_as_before(self) -> None:
        """Not a refusal on ignorance, and not a floor of nothing either — the
        key must be absent so the engine takes its old path."""
        from datetime import UTC, datetime

        from core.errors import TradingSystemError
        from core.types import MarketContext

        runner, _ = self._runner()

        def boom(_symbol: str) -> None:
            raise TradingSystemError("no spec")

        runner.broker = type("_B", (), {"spec": staticmethod(boom)})()
        context = MarketContext(
            symbol="EURUSD", now=datetime(2026, 8, 18, tzinfo=UTC), series={}, tick=None
        )

        runner._attach_cost_floor("EURUSD", context)

        assert "min_stop_for_costs" not in context.meta

    def test_the_engine_reads_it_defensively(self) -> None:
        """A backtest, a unit test and any caller that never set it must get
        zero rather than an exception."""
        from datetime import UTC, datetime

        from analysis.confluence import ConfluenceEngine
        from core.types import MarketContext

        bare = MarketContext(
            symbol="EURUSD", now=datetime(2026, 8, 18, tzinfo=UTC), series={}, tick=None
        )
        rubbish = MarketContext(
            symbol="EURUSD",
            now=datetime(2026, 8, 18, tzinfo=UTC),
            series={},
            tick=None,
            meta={"min_stop_for_costs": "wide-ish"},
        )

        assert ConfluenceEngine._cost_floor(bare) == 0.0
        assert ConfluenceEngine._cost_floor(rubbish) == 0.0


class TestAWiderStopIsGivenTheTimeItNeeds:
    """The regression that shipped, and it made things worse rather than better.

    Flooring the stop at what the costs demand makes 1R a longer distance — on
    a quiet instrument a two-pip stop becomes seven — while the window it had
    to be covered in stayed at the profile's nominal bars. On a synthetic walk
    at 0.0002 per bar, the same market that cleared at +0.21R expected with the
    chart's own stop was refused outright at the cost floor: nothing between
    1.00R and 3.00R was reached even once in twenty-four bars.

    Price does not travel in a straight line, so covering a distance takes its
    square in time. The runway check has used that law all along.
    """

    @staticmethod
    def _engine():  # type: ignore[no-untyped-def]
        from analysis.confluence import ConfluenceEngine
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        return ConfluenceEngine([], settings.analysis.confluence)

    def test_a_stop_the_chart_chose_is_measured_over_the_bars_it_always_was(self) -> None:
        engine = self._engine()
        base = engine.config.target_horizon_bars

        assert engine._horizon_bars(0.0010, 0.0010, None) == base
        assert engine._horizon_bars(0.0008, 0.0010, None) == base

    def test_the_window_grows_with_the_square_of_the_widening(self) -> None:
        engine = self._engine()
        base = engine.config.target_horizon_bars

        assert engine._horizon_bars(0.0020, 0.0010, None) == base * 4

    def test_the_square_law_is_capped_before_it_runs_away(self) -> None:
        """A ten-fold widening asks for a hundred times the window, which stops
        describing a trade anyone would sit in."""
        engine = self._engine()
        base = engine.config.target_horizon_bars

        assert engine._horizon_bars(0.0100, 0.0010, None) == int(
            base * engine.config.max_cost_horizon_stretch
        )

    def test_a_missing_natural_stop_changes_nothing(self) -> None:
        engine = self._engine()

        assert engine._horizon_bars(0.0020, 0.0, None) == engine.config.target_horizon_bars

    def test_the_quiet_market_that_regressed_is_payable_again(self) -> None:
        """The measurement itself, on the walk that caught it. At the cost floor
        over the nominal window nothing pays; over the window the square law
        asks for, it does."""
        # Tight wicks on purpose: a quiet instrument, which is where the cost
        # floor is the largest multiple of the stop the chart wanted.
        rng = np.random.default_rng(3)
        close = 1.10 + np.cumsum(rng.normal(0.0, 0.0002, 500))
        frame = pd.DataFrame(
            {"open": close, "high": close + 0.00005, "low": close - 0.00005, "close": close}
        )
        floored, natural = 0.00068, 0.00020

        def best_edge(bars: int) -> float:
            runs = first_touch_runs(frame, risk=floored, bars_ahead=bars, long=True)
            assert runs is not None
            cost_r = 0.00006 / floored
            edges = [
                float((runs >= r * floored).mean()) * (r - cost_r)
                - (1.0 - float((runs >= r * floored).mean())) * (1.0 + cost_r)
                for r in np.arange(1.0, 3.001, 0.05)
            ]
            return max(edges)

        engine = self._engine()
        stretched = engine._horizon_bars(floored, natural, None)

        assert stretched > engine.config.target_horizon_bars
        assert best_edge(engine.config.target_horizon_bars) <= 0.0
        assert best_edge(stretched) > 0.0


class TestTheEngineAndTheRunnerMeasureTheSameThing:
    def test_the_confluence_engine_delegates_to_the_shared_walk(self) -> None:
        """Two implementations of one statistic is the bug underneath the bug."""
        from analysis.confluence import ConfluenceEngine
        from core.types import Direction

        frame = walk()
        closes = frame["close"].to_numpy()

        mine = ConfluenceEngine._first_touch_reach(frame, closes, Direction.SHORT, 0.0010, 24)
        theirs = first_touch_runs(frame, risk=0.0010, bars_ahead=24, long=False)

        assert mine is not None and theirs is not None
        assert np.array_equal(mine, theirs)

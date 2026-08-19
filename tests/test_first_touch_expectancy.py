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

from analysis.target_reach import (
    first_touch_outcomes,
    first_touch_runs,
    measure,
    measure_first_touch,
)


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
    def _runner(*, planning: bool = True):  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from runner.service import JarvisRunner

        runner = object.__new__(JarvisRunner)
        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        # This gate only speaks when the ENGINE also planned on the widened
        # stop, so the mechanism is exercised with that on. The live overlay has
        # it off, and `test_it_is_silent_when_the_engine_planned_a_different_
        # trade` holds that end.
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
        return runner

    def test_it_speaks_even_when_the_engine_planned_on_the_narrower_stop(self) -> None:
        """It used to fall silent here, and that silence had a good reason.

        `_widen_stop_for_costs` runs just above this. With the engine NOT
        planning on the widened stop, the analysis chose its target against the
        narrow one and this re-measured the wide one — refusing a target the
        analysis had proved payable, on geometry the analysis never saw. 69 of
        140 live setups in six hours.

        The unfairness was in the arithmetic, not in asking the question.
        Widening lowers reward-to-risk AND makes the position much harder to
        stop out; the two-outcome form could only see the first, because it
        counted every unresolved window as a stop-out. Stop-outs are counted
        directly now, so a wider stop earns its credit and the question is fair
        to ask again.

        It matters because the alternative is worse: with this silent the
        break-even branch is decided by the UNCONDITIONAL reach, which does not
        know the stop exists at all and therefore cannot credit anything.
        """
        frame = walk()
        entry = float(frame["close"].iloc[-1])
        runner = self._runner(planning=False)

        verdict = runner._target_survival(
            self._context(frame, entry), self._idea(0.0010, entry, entry + 0.0020)
        )

        assert verdict is not None
        assert verdict.resolved_windows > 0
        # And it is measured against the stop the ORDER will carry.
        assert verdict.reward_risk == pytest.approx(2.0, abs=0.01)

    def test_a_wider_stop_is_credited_and_not_only_charged(self) -> None:
        """The property the unconditional reading structurally cannot have.

        Hold the target still and widen the stop. Reward-to-risk falls, so the
        break-even requirement rises — that is the charge, and it is real. The
        credit is that the position becomes far harder to stop out, which the
        unconditional reach cannot see: it returns the same number whatever the
        stop is, so widening can only ever count against a trade.
        """
        frame = walk()
        entry = float(frame["close"].iloc[-1])
        runner = self._runner(planning=False)
        context = self._context(frame, entry)
        target = entry + 0.0020

        narrow = runner._target_survival(context, self._idea(0.0010, entry, target))
        wide = runner._target_survival(context, self._idea(0.0030, entry, target))

        assert narrow is not None and wide is not None
        assert wide.reward_risk < narrow.reward_risk  # charged
        assert wide.required_pct > narrow.required_pct  # charged
        assert wide.forward_pct > narrow.forward_pct  # and credited

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

        mine = ConfluenceEngine._first_touch_outcomes(frame, closes, Direction.SHORT, 0.0010, 24)
        theirs = first_touch_runs(frame, risk=0.0010, bars_ahead=24, long=False)

        assert mine is not None and theirs is not None
        assert np.array_equal(mine.run, theirs)
        # The excursion is the half of the record `first_touch_runs` exposes;
        # the engine needs the other half too, because a window that expired
        # unresolved is not a stop-out and must not be priced as one.
        assert mine.stopped.shape == mine.run.shape
        assert mine.settle_r.shape == mine.run.shape


class TestAWindowThatExpiredIsNotAStopOut:
    """The arithmetic error that produced a day with no trades at all.

    The search priced two outcomes where there are three:

        edge = hit * (RR - cost) - (1 - hit) * (1 + cost)

    Everything short of the target was charged a full stop, including every
    window in which price drifted sideways and the horizon expired without the
    stop ever being touched. The expired share grows with the target and with a
    short horizon, so the penalty was largest exactly where this account lives.
    """

    @staticmethod
    def _market(drift: float, seed: int = 7, bars: int = 4000, sigma: float = 0.0004):
        rng = np.random.default_rng(seed)
        close = 1.10 + np.cumsum(rng.normal(drift, sigma, bars))
        wick = np.abs(rng.normal(0.0, sigma * 0.6, bars))
        return pd.DataFrame({"close": close, "high": close + wick, "low": close - wick})

    def _outcomes(self, drift: float):
        frame = self._market(drift)
        risk = float(np.abs(np.diff(frame["close"].to_numpy())).mean()) * 8
        return first_touch_outcomes(frame, risk=risk, bars_ahead=24, long=True), risk

    def test_the_three_outcomes_add_up_to_every_window(self) -> None:
        outcomes, risk = self._outcomes(0.0)
        won = outcomes.run >= 1.0 * risk
        lost = outcomes.stopped & ~won
        expired = ~outcomes.stopped & ~won

        assert int(won.sum() + lost.sum() + expired.sum()) == outcomes.windows
        # The point of the whole change: expiring is the COMMON case at 1R.
        assert expired.mean() > 0.4

    def test_a_market_with_a_real_edge_is_no_longer_refused(self) -> None:
        outcomes, risk = self._outcomes(0.00008)
        odds = outcomes.expectancy_r(distance=1.0 * risk, risk=risk, cost_r=0.15)
        two_outcome = odds.reach * (1.0 - 0.15) - (1.0 - odds.reach) * 1.15

        assert two_outcome < 0.0  # what the old form said: refuse
        assert odds.expected_r > 0.0  # what the windows actually did
        assert odds.target_is_the_exit(reward_risk=1.0, cost_r=0.15)

    def test_a_coin_flip_stays_refused_at_every_distance(self) -> None:
        """The check that matters. Pricing expired windows honestly must not
        conjure an edge out of a market that has none."""
        outcomes, risk = self._outcomes(0.0)

        for reward_risk in (0.6, 1.0, 1.5, 2.0, 3.0):
            odds = outcomes.expectancy_r(distance=reward_risk * risk, risk=risk, cost_r=0.15)
            assert odds.expected_r < 0.0, reward_risk

    def test_trading_into_the_trend_the_wrong_way_stays_refused(self) -> None:
        outcomes, risk = self._outcomes(-0.00008)

        for reward_risk in (0.6, 1.0, 2.0):
            odds = outcomes.expectancy_r(distance=reward_risk * risk, risk=risk, cost_r=0.15)
            assert odds.expected_r < 0.0, reward_risk

    def test_a_target_the_market_never_touches_is_not_an_exit(self) -> None:
        """Drift alone can make expectancy positive with a reach of zero.

        Every window ends a little in front, nothing is ever stopped, and the
        distance is never covered. The trade would sit there: this system
        leaves at the target or the stop, and neither arrives. The break-even
        rate is still enforced, on the windows that actually resolved.
        """
        close = 1.10 + np.arange(600, dtype=float) * 0.00002
        frame = pd.DataFrame({"close": close, "high": close, "low": close})
        outcomes = first_touch_outcomes(frame, risk=0.005, bars_ahead=24, long=True)

        assert outcomes is not None
        odds = outcomes.expectancy_r(distance=0.005, risk=0.005, cost_r=0.0)

        assert odds.reach == 0.0
        assert odds.resolved_windows == 0
        # Two independent refusals, and either alone is enough.
        #
        # Capping the credit on an expired window at zero means the drift buys
        # nothing: price is up at every horizon and the expectancy is still not
        # positive, because none of it has been banked and the position is
        # still open. And with nothing resolved, the target has not been shown
        # to be the exit either.
        assert odds.expected_r <= 0.0
        assert not odds.target_is_the_exit(reward_risk=1.0, cost_r=0.0)

    def test_the_break_even_rate_is_applied_to_the_resolved_windows(self) -> None:
        """`(1 + cost) / (1 + RR)` was derived for a two-outcome trade.

        Applying it to a sample that is mostly unresolved windows is what made
        every target look unpayable; applying it to the resolved ones is what
        it was always for.
        """
        outcomes, risk = self._outcomes(0.00008)
        odds = outcomes.expectancy_r(distance=1.5 * risk, risk=risk, cost_r=0.15)

        assert odds.reach < 0.46  # against the whole sample: fails break-even
        assert odds.resolved_reach > 0.46  # against the resolved ones: clears it
        assert odds.target_is_the_exit(reward_risk=1.5, cost_r=0.15)

    def test_paper_profit_on_an_open_position_is_not_counted_as_earnings(self) -> None:
        """The hole the first version of this fix opened, closed.

        Crediting an expired window with wherever price stood made a distance
        nothing reaches look like the best trade on the board: at 3R on a
        trending series the target is touched in 0.5% of windows and the whole
        +0.97R came from unrealised profit on positions still open. What is
        showing in your favour you may yet give back; what is showing against
        you, you already hold.
        """
        outcomes, risk = self._outcomes(0.00012)
        distant = outcomes.expectancy_r(distance=3.0 * risk, risk=risk, cost_r=0.15)
        near = outcomes.expectancy_r(distance=1.0 * risk, risk=risk, cost_r=0.15)

        assert distant.reach < 0.02  # never actually touched
        assert distant.expected_r < 0.0  # and therefore not a trade
        assert near.expected_r > distant.expected_r
        # The curve has an interior maximum now instead of climbing forever
        # into distances nothing reaches.
        assert (
            near.expected_r
            > outcomes.expectancy_r(distance=0.6 * risk, risk=risk, cost_r=0.15).expected_r
        )

    def test_a_thin_resolved_sample_cannot_decide_on_its_own(self) -> None:
        """A rate off 36 windows is a sample, not a measurement."""
        outcomes, risk = self._outcomes(0.00012)
        odds = outcomes.expectancy_r(distance=3.0 * risk, risk=risk, cost_r=0.15)

        assert odds.resolved_windows < 120
        assert not odds.target_is_the_exit(reward_risk=3.0, cost_r=0.15, minimum_resolved=120)
        # And the backstop is genuinely a backstop: by the time the sample is
        # this thin the arithmetic has already refused the distance anyway.
        assert odds.expected_r < 0.0


class TestTheTwoFloorsMayNotSitOnTopOfEachOther:
    """NDX100 SHORT, 19 August: approved, reviewed, then RR_BELOW_MINIMUM.

    `reward:risk is 1:0.52, below the required 1:0.60`, on a 3,995 pip stop
    against a 2,077 pip target. Nothing was wrong with the setup. The target
    search lands on its own floor on nearly every market, both floors were
    0.60, and the quote moved between planning and sizing.
    """

    @staticmethod
    def _settings(**risk_overrides):  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH,
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml",
            env_overrides=False,
        )

    def test_the_live_overlay_leaves_the_plan_room_to_move(self) -> None:
        settings = self._settings()
        confluence = settings.analysis.confluence

        assert confluence.minimum_r_multiple > settings.risk.min_risk_reward
        assert confluence.minimum_r_multiple >= settings.risk.min_risk_reward * (
            1.0 + confluence.target_planning_margin
        )

    def test_a_config_that_puts_them_level_is_refused_at_load(self) -> None:
        """Equal is the knife edge, and it shipped. A validator that only
        refused `min_risk_reward > floor` allowed exactly the arrangement that
        cost the trade."""
        from config.schema import Settings

        settings = self._settings()
        raw = settings.model_dump()
        # Level, which is exactly how the live overlay was configured.
        raw["analysis"]["confluence"]["minimum_r_multiple"] = settings.risk.min_risk_reward

        with pytest.raises(ValueError, match="RR_BELOW_MINIMUM"):
            Settings.model_validate(raw)

    def test_zero_margin_makes_level_floors_legal_again(self) -> None:
        """The escape hatch has to actually work, or the validator is a wall."""
        from config.schema import Settings

        settings = self._settings()
        raw = settings.model_dump()
        raw["analysis"]["confluence"]["minimum_r_multiple"] = settings.risk.min_risk_reward
        raw["analysis"]["confluence"]["target_planning_margin"] = 0.0

        assert Settings.model_validate(raw).analysis.confluence.minimum_r_multiple == (
            settings.risk.min_risk_reward
        )

    def test_the_margin_can_be_set_to_zero_to_accept_the_edge(self) -> None:
        """Expressible, so the old behaviour is a decision and not an accident."""
        from config.schema import ConfluenceConfig

        assert ConfluenceConfig(target_planning_margin=0.0).target_planning_margin == 0.0


class TestTheWindowIsSizedToWhatItHasToWatch:
    """1,683 refusals an hour reading "no target pays" about plans that were
    never measured.

    Covering a distance d takes (d / speed)^2 bars, so a target at 0.75R of an
    eight-ATR stop needs 36 and every profile handed out 24. Past six ATR of
    stop, every window expires with neither the target nor the stop touched —
    and an expired window is charged a round trip, so the expectancy converges
    on minus the cost however good the market is. A structural swing stop
    reaching the last swing low is routinely eight to sixteen ATR, so this was
    not an edge case; it was most of them.
    """

    @staticmethod
    def _market(drift: float, seed: int = 9, bars: int = 4000):
        rng = np.random.default_rng(seed)
        close = 1.10 + np.cumsum(rng.normal(drift, 0.0004, bars))
        wick = np.abs(rng.normal(0.0, 0.00024, bars))
        frame = pd.DataFrame({"close": close, "high": close + wick, "low": close - wick})
        return frame, float(np.abs(np.diff(close)).mean())

    @staticmethod
    def _fitted(frame, risk: float, per_bar: float, base: int = 24) -> int:
        needed = int((risk * 0.75 / per_bar) ** 2)
        return max(base, min(needed, min(len(frame), 400) // 2))

    def _verdict(self, frame, risk: float, bars: int):
        outcomes = first_touch_outcomes(frame, risk=risk, bars_ahead=bars, long=True)
        assert outcomes is not None
        odds = outcomes.expectancy_r(distance=0.75 * risk, risk=risk, cost_r=0.10)
        passes = odds.expected_r > 0 and odds.target_is_the_exit(
            reward_risk=0.75, cost_r=0.10, minimum_resolved=30
        )
        return odds, passes

    def test_a_wide_stop_becomes_measurable_instead_of_refused(self) -> None:
        frame, per_bar = self._market(0.00008)
        risk = per_bar * 32

        flat, flat_passes = self._verdict(frame, risk, 24)
        fitted, fitted_passes = self._verdict(frame, risk, self._fitted(frame, risk, per_bar))

        assert flat.resolved_windows < 30  # nothing resolved: unmeasurable
        assert not flat_passes
        assert fitted.resolved_windows > 1_000
        assert fitted_passes and fitted.expected_r > 0.5

    def test_a_stop_the_flat_window_already_covered_is_left_alone(self) -> None:
        """The change must reach only the plans that needed it. A four-ATR stop
        resolves inside 24 bars, so its window does not move and neither does
        its answer."""
        frame, per_bar = self._market(0.00008)
        risk = per_bar * 4

        assert self._fitted(frame, risk, per_bar) == 24

    def test_a_coin_flip_is_still_refused_over_the_longer_window(self) -> None:
        """The check that matters. A longer ruler must not turn a market with
        no edge into a trade — it only lets the market answer."""
        frame, per_bar = self._market(0.0)

        for multiple in (16, 32):
            risk = per_bar * multiple
            odds, passes = self._verdict(frame, risk, self._fitted(frame, risk, per_bar))
            assert not passes, multiple
            assert odds.expected_r < 0.0, multiple

    def test_trading_into_the_trend_the_wrong_way_is_still_refused(self) -> None:
        frame, per_bar = self._market(-0.00008)

        for multiple in (16, 32):
            risk = per_bar * multiple
            odds, passes = self._verdict(frame, risk, self._fitted(frame, risk, per_bar))
            assert not passes, multiple
            assert odds.expected_r < -0.5, multiple

    def test_the_window_never_outruns_the_history_that_measures_it(self) -> None:
        """A stretch past the data does not widen the window, it removes the
        gate: `_reachable_target` gives up and returns the planned distance
        UNMEASURED. A test caught exactly that, approving a trade it was
        written to refuse with a 34,560-minute horizon.
        """
        from analysis.confluence import ConfluenceEngine
        from config.schema import ConfluenceConfig, HorizonProfileConfig

        frame, per_bar = self._market(0.00008, bars=100)
        engine = ConfluenceEngine([], ConfluenceConfig())
        profile = HorizonProfileConfig(planning_timeframe="H1", target_horizon_bars=24)

        # A stop so wide the square law would ask for thousands of bars.
        horizon = engine._horizon_bars(per_bar * 200, per_bar * 200, profile, frame=frame)

        assert horizon <= len(frame) // 3
        assert horizon <= min(len(frame), 400) // 2

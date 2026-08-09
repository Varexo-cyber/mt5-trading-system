"""The exit study has to be measuring the same market the backtester does.

Two claims carry everything downstream, and both are cheap to break by
accident. The first is that `rolling_drift` is `drift_score` — if the fast
version drifts from the live reader by even a little, the policy is trained on
a signal the guard does not have. The second is that a walked position ends
where `PessimisticBacktester` says it ends; a study that filled more kindly
than the backtester would produce a policy tuned to a market nobody trades in.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from analysis.position_health import drift_score
from backtesting.engine import BacktestAssumptions, BacktestOrder, PessimisticBacktester
from backtesting.exit_study import (
    R_EDGES,
    ExitSample,
    ReplayedPosition,
    give_back_curve,
    hold_table,
    render_give_back,
    render_hold_table,
    rolling_drift,
    study,
)
from core.types import Direction

START = datetime(2026, 3, 2, 8, 0, tzinfo=UTC)


def frame(closes: list[float], *, spread: float = 0.0005) -> pd.DataFrame:
    """Bars that open where the last one closed, so nothing gaps by accident.

    A fixture that gaps produces stop fills at the open rather than at the
    stop, which silently changes what every assertion below is measuring.
    """
    rows = []
    previous = closes[0]
    for close in closes:
        rows.append(
            {
                "open": previous,
                "high": max(previous, close) + spread,
                "low": min(previous, close) - spread,
                "close": close,
                "tick_volume": 100,
            }
        )
        previous = close
    index = pd.date_range(START, periods=len(rows), freq="5min", tz=UTC)
    return pd.DataFrame(rows, index=index)


def wandering(count: int, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    walk = 1.1000 + np.cumsum(rng.normal(0.0, 0.0004, count))
    return frame([float(value) for value in walk])


class TestTheFastDriftIsTheLiveDrift:
    """`rolling_drift` exists only because the naive version is a polyfit per
    bar per order. Speed is worthless if the number changed on the way."""

    def test_it_matches_drift_score_bar_for_bar(self) -> None:
        bars = wandering(400)
        fast = rolling_drift(bars, bars=12)

        checked = 0
        for position in range(20, len(bars)):
            slow = drift_score(bars.iloc[: position + 1], 1, bars=12)
            if slow is None:
                continue
            assert fast[position] == pytest.approx(slow, abs=1e-9), f"bar {position}"
            checked += 1

        assert checked > 300, "the comparison has to actually run"

    def test_a_short_reads_the_negative_of_a_long(self) -> None:
        bars = wandering(200)
        fast = rolling_drift(bars, bars=12)

        for position in (50, 120, 199):
            against = drift_score(bars.iloc[: position + 1], -1, bars=12)
            assert fast[position] == pytest.approx(-against, abs=1e-9)

    def test_a_frame_too_short_to_read_says_so(self) -> None:
        assert np.all(np.isnan(rolling_drift(wandering(10), bars=12)))

    def test_the_opening_bars_are_undefined_rather_than_guessed(self) -> None:
        """No ATR window and no slope window means no reading. Zero would be a
        reading — it would say 'going nowhere', which is a claim."""
        fast = rolling_drift(wandering(200), bars=12)
        assert np.all(np.isnan(fast[:13]))
        assert np.isfinite(fast[20])


class TestWalkingAPositionMatchesTheBacktester:
    ASSUMPTIONS = BacktestAssumptions(exit_slippage_bps=0.0, round_trip_commission_bps=0.0)

    def order(self, bars: pd.DataFrame, direction: Direction, **kwargs) -> BacktestOrder:  # type: ignore[no-untyped-def]
        entry = float(bars.iloc[0]["close"])
        sign = int(direction)
        defaults = {
            "entry": entry,
            "stop_loss": entry - 0.0050 * sign,
            "take_profit": entry + 0.0100 * sign,
        }
        return BacktestOrder(
            symbol="EURUSD",
            decided_at=bars.index[0].to_pydatetime(),
            direction=direction,
            modules=("range_fade",),
            **{**defaults, **kwargs},
        )

    @pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
    def test_the_outcome_and_the_r_agree_with_the_engine(self, direction: Direction) -> None:
        bars = wandering(500, seed=3)
        orders = [self.order(bars, direction)]

        walked = study(orders, bars, assumptions=self.ASSUMPTIONS)
        engine = PessimisticBacktester(self.ASSUMPTIONS).run(bars, orders)

        assert len(walked) == 1 and len(engine.trades) == 1
        assert walked[0].outcome == engine.trades[0].outcome
        assert walked[0].final_r == pytest.approx(engine.trades[0].net_r, abs=1e-9)
        assert walked[0].bars_held == engine.trades[0].holding_bars

    def test_it_agrees_across_a_whole_book_of_orders(self) -> None:
        """One matching trade is a coincidence. The engine and the walk have
        separate implementations of the same fill rules and they have to stay
        that way, so this checks a spread of entries and both directions."""
        bars = wandering(600, seed=11)
        orders = [
            self.order(
                bars.iloc[start:],
                Direction.LONG if start % 2 else Direction.SHORT,
            )
            for start in range(0, 400, 17)
        ]

        walked = study(orders, bars, assumptions=self.ASSUMPTIONS)
        engine = PessimisticBacktester(self.ASSUMPTIONS).run(bars, orders)

        assert len(walked) == len(engine.trades) > 15
        for mine, theirs in zip(walked, engine.trades, strict=True):
            assert mine.outcome == theirs.outcome
            assert mine.final_r == pytest.approx(theirs.net_r, abs=1e-9)

    def test_costs_cancel_out_of_the_edge(self) -> None:
        """The whole table rests on this. `take now` and `hold out` both pay the
        same round trip, so what is left when you subtract them is price."""
        bars = wandering(500, seed=5)
        orders = [self.order(bars, Direction.LONG)]

        free = study(orders, bars, assumptions=self.ASSUMPTIONS)[0]
        charged = study(
            orders,
            bars,
            assumptions=BacktestAssumptions(exit_slippage_bps=0.0, round_trip_commission_bps=4.0),
        )[0]

        assert charged.final_r < free.final_r, "the fixture has to actually charge something"
        for cheap, dear in zip(free.samples, charged.samples, strict=True):
            assert dear.edge_of_holding == pytest.approx(cheap.edge_of_holding, abs=1e-9)

    def test_the_closing_bar_is_not_offered_as_a_decision(self) -> None:
        """By the time that bar's close is known the stop or target has already
        filled. Recording it would be offering a choice nobody ever had."""
        bars = wandering(500, seed=3)
        walked = study([self.order(bars, Direction.LONG)], bars, assumptions=self.ASSUMPTIONS)[0]

        assert walked.outcome in {"SL", "TP", "SL_FIRST_AMBIGUOUS"}
        assert len(walked.samples) == walked.bars_held - 1

    def test_the_peak_ratchets_and_never_falls(self) -> None:
        bars = wandering(500, seed=9)
        walked = study([self.order(bars, Direction.LONG)], bars, assumptions=self.ASSUMPTIONS)[0]

        peaks = [sample.peak_r for sample in walked.samples]
        assert peaks == sorted(peaks)
        assert walked.peak_r >= max(peaks)

    def test_bars_since_peak_counts_from_the_high(self) -> None:
        """A long that runs for three bars and then goes sideways.

        Entry is the first close and the fill is the bar after it, so the walk
        starts at 1.1010 and makes new highs for three bars. From the fourth
        the market only revisits the high it already made — an equal high is
        not a new one, so the counter starts climbing there. That is the state
        the banking rule cares about: the move has stopped, whatever the last
        tick did.
        """
        closes = [1.1000, 1.1010, 1.1020, 1.1030] + [1.1025] * 6
        bars = frame(closes, spread=0.00001)
        order = self.order(bars, Direction.LONG, stop_loss=1.0900, take_profit=1.2000)

        walked = study([order], bars, assumptions=self.ASSUMPTIONS)[0]
        since = [sample.bars_since_peak for sample in walked.samples]

        assert since == [0, 0, 0, 1, 2, 3, 4, 5, 6]

    def test_an_order_with_no_future_bars_is_dropped_not_crashed(self) -> None:
        bars = wandering(200)
        late = BacktestOrder(
            symbol="EURUSD",
            decided_at=bars.index[-1].to_pydatetime() + timedelta(days=5),
            direction=Direction.LONG,
            entry=1.10,
            stop_loss=1.09,
            take_profit=1.12,
        )
        assert study([late], bars) == []

    def test_a_zero_risk_order_is_dropped(self) -> None:
        bars = wandering(200)
        broken = BacktestOrder(
            symbol="EURUSD",
            decided_at=bars.index[0].to_pydatetime(),
            direction=Direction.LONG,
            entry=1.10,
            stop_loss=1.10,
            take_profit=1.12,
        )
        assert study([broken], bars) == []


def sample(take_now: float, hold_out: float, drift: float | None) -> ExitSample:
    return ExitSample(
        bars_held=5,
        take_now_r=take_now,
        hold_out_r=hold_out,
        peak_r=max(take_now, 0.0),
        bars_since_peak=0,
        drift=drift,
    )


def position(samples: list[ExitSample], *, peak: float, final: float) -> ReplayedPosition:
    return ReplayedPosition(
        symbol="EURUSD",
        playbook="range_fade",
        outcome="SL",
        final_r=final,
        peak_r=peak,
        bars_held=len(samples) + 1,
        samples=tuple(samples),
    )


class TestTheHoldTable:
    def test_a_stalled_move_that_gives_it_back_reads_take_it(self) -> None:
        rows = hold_table(
            [position([sample(0.40, -0.90, 0.0)] * 40, peak=0.40, final=-0.90)],
            min_samples=30,
        )

        assert len(rows) == 1
        assert rows[0].pace == "stalled"
        assert rows[0].verdict == "TAKE IT"
        assert rows[0].edge < 0

    def test_a_running_move_that_pays_reads_hold(self) -> None:
        rows = hold_table(
            [position([sample(0.40, 1.50, 1.2)] * 40, peak=1.5, final=1.5)],
            min_samples=30,
        )

        assert rows[0].pace == "running" and rows[0].verdict == "hold"

    def test_losing_moments_are_not_in_the_table(self) -> None:
        """'Should I take this' is not a question about a losing position. The
        answer there is the stop, and the stop is already at the broker."""
        rows = hold_table(
            [position([sample(-0.40, -1.0, 0.0)] * 60, peak=0.0, final=-1.0)],
            min_samples=1,
        )

        assert rows == []

    def test_thin_buckets_are_dropped_rather_than_reported(self) -> None:
        thin = [position([sample(0.40, 0.10, 0.0)] * 5, peak=0.4, final=0.1)]

        assert hold_table(thin, min_samples=30) == []
        assert len(hold_table(thin, min_samples=1)) == 1

    def test_a_reading_that_could_not_be_taken_is_its_own_pace(self) -> None:
        """Twelve bars have not passed yet. That is not 'going nowhere' — it is
        no information, and the two must not share a row."""
        rows = hold_table(
            [position([sample(0.40, 0.10, None)] * 40, peak=0.4, final=0.1)], min_samples=30
        )

        assert rows[0].pace == "unknown"

    def test_the_buckets_cover_every_profit_without_overlapping(self) -> None:
        moments = [sample(value, 0.0, 0.0) for value in (0.05, 0.2, 0.4, 0.6, 0.9, 1.2, 3.0)]
        rows = hold_table([position(moments, peak=3.0, final=0.0)], min_samples=1)

        assert len(rows) == len(moments), "each profit landed in its own bucket"
        assert rows[-1].r_ceiling == float("inf"), "the top bucket is open-ended"
        assert rows[0].r_floor == R_EDGES[0]

    def test_it_renders_without_rows(self) -> None:
        assert "Widen the window" in render_hold_table([])

    def test_it_renders_with_rows(self) -> None:
        rendered = render_hold_table(
            hold_table([position([sample(0.40, -0.9, 0.0)] * 40, peak=0.4, final=-0.9)])
        )

        assert "HOLD OR TAKE" in rendered and "TAKE IT" in rendered


class TestTheGiveBackCurve:
    def test_it_counts_what_each_trade_kept(self) -> None:
        rows = give_back_curve(
            [position([], peak=0.60, final=0.15) for _ in range(20)], min_positions=10
        )

        assert len(rows) == 1
        assert rows[0].mean_peak_r == pytest.approx(0.60)
        assert rows[0].kept == pytest.approx(0.25)

    def test_a_trade_that_never_went_into_profit_has_nothing_to_keep(self) -> None:
        assert give_back_curve([position([], peak=0.0, final=-1.0)] * 20, min_positions=1) == []

    def test_it_renders(self) -> None:
        rows = give_back_curve([position([], peak=0.60, final=0.15)] * 20, min_positions=10)

        assert "BEST MOMENT" in render_give_back(rows)
        assert "No position ever went into profit" in render_give_back([])


class TestItSurvivesTheRealReplay:
    """The unit tests above drive the study with hand-built orders. This drives
    it with the orders the theories actually produce, because the two have gone
    out of step before: `BacktestOrder.modules` carries the playbook name and a
    study that lost it would report every row under 'unknown'."""

    def test_the_playbooks_orders_walk_and_bucket(self) -> None:
        from analysis.playbooks import FadeConfig, PlaybookEngine, RangeFade
        from backtesting.playbook_replay import PlaybookReplay
        from config.loader import load_settings
        from core.types import Timeframe

        base = wandering(2400, seed=21)
        frames = {Timeframe.M5: base}
        for timeframe, rule in ((Timeframe.M15, "15min"), (Timeframe.H1, "1h")):
            frames[timeframe] = (
                base.resample(rule)
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "tick_volume": "sum",
                    }
                )
                .dropna()
            )

        confluence = load_settings(env_overrides=False).analysis.confluence
        engine = PlaybookEngine([RangeFade(FadeConfig())], confluence)
        replay = PlaybookReplay(engine, history_bars=200)
        orders = replay.orders(
            "EURUSD",
            frames,
            point=0.00001,
            start=base.index[600].to_pydatetime(),
            end=base.index[-1].to_pydatetime(),
        )
        if not orders:
            pytest.skip("the random walk produced no fade setups; nothing to walk")

        walked = study(orders, base)

        assert walked, "orders that exist must produce positions"
        assert {p.playbook for p in walked} == {"range_fade"}
        assert all(p.bars_held >= 1 for p in walked)
        # Rendering is where an empty or malformed table shows up, and it is the
        # only part the operator ever sees.
        render_hold_table(hold_table(walked, min_samples=1))
        render_give_back(give_back_curve(walked, min_positions=1))


def moment(take_now: float, drift: float | None, hold_out: float = 0.0) -> ExitSample:
    return ExitSample(
        bars_held=1,
        take_now_r=take_now,
        hold_out_r=hold_out,
        peak_r=max(take_now, 0.0),
        bars_since_peak=0,
        drift=drift,
    )


def walked(path: list[ExitSample], final: float) -> ReplayedPosition:
    return ReplayedPosition(
        symbol="EURUSD",
        playbook="range_fade",
        outcome="SL",
        final_r=final,
        peak_r=max((s.peak_r for s in path), default=0.0),
        bars_held=len(path) + 1,
        samples=tuple(path),
    )


class TestReplayingAWholeRule:
    """The hold-or-take table cannot answer "where should the threshold be".
    Every row in it is conditional on having reached that level, so a rule that
    banks at 0.15R does not collect the 1.50R rows — it deletes them. Averaging
    per moment and reading a threshold off the result is how a backtest talks
    itself into taking profit far too early.
    """

    def test_a_rule_that_never_fires_returns_what_the_trade_did(self) -> None:
        from backtesting.exit_study import HOLD_EVERYTHING, apply_policy

        outcome = apply_policy([walked([moment(0.9, 0.0)], final=-1.0)], HOLD_EVERYTHING)

        assert outcome.total_r == pytest.approx(-1.0)
        assert outcome.banked == 0

    def test_it_takes_the_first_moment_the_rule_fires(self) -> None:
        """Not the best one. A rule runs forward in time and cannot see which
        of its future triggers would have been the good one."""
        from backtesting.exit_study import ExitPolicy, apply_policy

        path = [moment(0.2, 0.0), moment(0.8, 0.0), moment(1.4, 0.0)]

        outcome = apply_policy([walked(path, final=-1.0)], ExitPolicy(take_at_r=0.15))

        assert outcome.total_r == pytest.approx(0.2)
        assert outcome.banked == 1

    def test_a_rule_that_bans_a_pace_waits_for_a_state_it_accepts(self) -> None:
        from backtesting.exit_study import ExitPolicy, apply_policy

        path = [moment(0.5, -1.0), moment(0.6, 1.0)]  # retracing, then running
        policy = ExitPolicy(take_at_r=0.15, act_when_against=False)

        outcome = apply_policy([walked(path, final=-1.0)], policy)

        assert outcome.total_r == pytest.approx(0.6), "it skipped the retrace and took the run"

    def test_taking_early_forfeits_what_the_trade_went_on_to_do(self) -> None:
        """The whole reason the per-moment table cannot answer this. Banking at
        0.2R on a trade that ended at +3R costs 2.8R, and no row in the
        hold-or-take table shows that, because the trade never reaches the rows
        that would."""
        from backtesting.exit_study import ExitPolicy, apply_policy

        winner = walked([moment(0.2, 0.0), moment(3.0, 0.0)], final=3.0)

        early = apply_policy([winner], ExitPolicy(take_at_r=0.15))
        patient = apply_policy([winner], ExitPolicy(take_at_r=2.5))

        assert early.total_r == pytest.approx(0.2)
        assert patient.total_r == pytest.approx(3.0)

    def test_a_position_with_no_recorded_moments_still_counts(self) -> None:
        """A trade stopped out on its first bar has no decision points, and
        dropping it would quietly flatter every policy by removing the fastest
        losers from the denominator."""
        from backtesting.exit_study import ExitPolicy, apply_policy

        outcome = apply_policy([walked([], final=-1.2)], ExitPolicy(take_at_r=0.15))

        assert outcome.trades == 1
        assert outcome.total_r == pytest.approx(-1.2)
        assert outcome.banked == 0

    def test_an_unreadable_pace_is_acted_on_by_default(self) -> None:
        """Before twelve bars there is no drift at all. Early in the trade with
        a small profit, "no information" is not a reason to leave money out."""
        from backtesting.exit_study import ExitPolicy, apply_policy

        outcome = apply_policy([walked([moment(0.4, None)], final=-1.0)], ExitPolicy(0.15))

        assert outcome.banked == 1

    def test_the_sweep_includes_holding_as_its_first_row(self) -> None:
        """Every policy needs something honest to be measured against, and it
        has to be the same trades over the same bars."""
        from backtesting.exit_study import sweep_policies

        outcomes = sweep_policies([walked([moment(0.5, 0.0)], final=-1.0)])

        assert outcomes[0].banked == 0
        assert len(outcomes) > 5

    def test_the_report_says_so_when_nothing_beats_holding(self) -> None:
        """The answer this project has to be able to print. A sweep that always
        finds a winner is a sweep that is fitting noise."""
        from backtesting.exit_study import render_policies, sweep_policies

        # Every trade runs to a big win, so taking anything early is worse.
        winners = [walked([moment(0.2, 0.0), moment(5.0, 0.0)], final=5.0)] * 20

        rendered = render_policies(sweep_policies(winners))

        assert "No threshold beat holding" in rendered

    def test_the_report_names_the_best_rule_when_there_is_one(self) -> None:
        from backtesting.exit_study import render_policies, sweep_policies

        # Every trade touches +0.5R and then goes to a full stop.
        givers = [walked([moment(0.5, 0.0)], final=-1.0)] * 20

        rendered = render_policies(sweep_policies(givers))

        assert "Best on this window" in rendered
        assert "0.50R/RS" in rendered or "0.40R/RS" in rendered

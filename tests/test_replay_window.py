"""The replay window is where look-ahead bias would enter, so it is pinned twice.

The slice was `frame[frame.index + duration <= decided_at].tail(history_bars)`:
a boolean mask over the whole frame, built per decision per timeframe. Over 90
days the M5 frame is around 26,000 rows, so a five-symbol run spent hundreds of
millions of row comparisons finding a cut point in a sorted index, and the tool
built to make measurement cheap took long enough that nobody would run it twice.

It is a binary search now. Making it faster is worthless if it also makes it
different, and "different" here means a bar the analysis could not have seen —
which does not fail loudly, it produces a backtest that looks better than
reality. So this compares the two implementations directly, on an index with
the awkward shapes: weekend gaps, a decision landing exactly on a close, and a
decision before enough history exists.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pandas as pd
import pytest

from backtesting.replay import REPLAY_TIMEFRAMES, HistoricalContextReplay
from core.types import Timeframe

START = datetime(2026, 3, 2, tzinfo=UTC)


def frame(timeframe: Timeframe, count: int) -> pd.DataFrame:
    """Bars on a real calendar: weekdays only, so weekends leave gaps."""
    stamps: list[pd.Timestamp] = []
    cursor = pd.Timestamp(START)
    while len(stamps) < count:
        if cursor.weekday() < 5:
            stamps.append(cursor)
        cursor += timeframe.duration
    index = pd.DatetimeIndex(stamps)
    values = [100.0 + i * 0.01 for i in range(count)]
    return pd.DataFrame(
        {
            "open": values,
            "high": [v + 0.05 for v in values],
            "low": [v - 0.05 for v in values],
            "close": values,
            "spread": 2,
        },
        index=index,
    )


def ending_at(timeframe: Timeframe, count: int) -> pd.DataFrame:
    """`count` bars of one timeframe, all ending at a shared moment.

    `frame` above starts every timeframe at the same instant, which is fine for
    comparing two window implementations and useless for driving the replay: by
    the time the hourly series reaches its 300th bar the daily one holds twelve,
    and the replay correctly refuses every decision for want of history.
    """
    end = pd.Timestamp(START) + pd.Timedelta(days=400)
    index = pd.date_range(end=end, periods=count, freq=timeframe.duration)
    values = [100.0 + i * 0.01 for i in range(count)]
    return pd.DataFrame(
        {
            "open": values,
            "high": [v + 0.05 for v in values],
            "low": [v - 0.05 for v in values],
            "close": values,
        },
        index=index,
    )


def old_window(data: pd.DataFrame, timeframe: Timeframe, decided_at, history_bars: int):
    """The implementation this replaced, kept here as the reference."""
    return data[data.index + timeframe.duration <= decided_at].tail(history_bars)


def new_window(data: pd.DataFrame, timeframe: Timeframe, decided_at, history_bars: int):
    close_times = data.index + timeframe.duration
    cut = int(close_times.searchsorted(pd.Timestamp(decided_at), side="right"))
    return data.iloc[max(0, cut - history_bars) : cut]


class TestTheFastWindowIsTheSameWindow:
    @pytest.mark.parametrize("timeframe", list(REPLAY_TIMEFRAMES))
    def test_every_decision_selects_identical_bars(self, timeframe: Timeframe) -> None:
        data = frame(timeframe, 600)
        for offset in range(0, 600, 37):
            decided_at = (data.index[min(offset, 599)] + timeframe.duration).to_pydatetime()
            assert new_window(data, timeframe, decided_at, 300).equals(
                old_window(data, timeframe, decided_at, 300)
            )

    def test_a_decision_exactly_on_a_close_includes_that_bar(self) -> None:
        """`side="right"` and `<=` have to agree on the boundary. Off by one
        here is either a bar thrown away or a bar the analysis could not have
        seen, and the second one silently improves every backtest."""
        data = frame(Timeframe.H1, 400)
        decided_at = (data.index[200] + Timeframe.H1.duration).to_pydatetime()

        window = new_window(data, Timeframe.H1, decided_at, 300)

        assert window.index[-1] == data.index[200]
        assert window.equals(old_window(data, Timeframe.H1, decided_at, 300))

    def test_a_decision_one_tick_before_a_close_excludes_it(self) -> None:
        data = frame(Timeframe.H1, 400)
        decided_at = (
            data.index[200] + Timeframe.H1.duration - timedelta(seconds=1)
        ).to_pydatetime()

        window = new_window(data, Timeframe.H1, decided_at, 300)

        assert window.index[-1] == data.index[199]
        assert window.equals(old_window(data, Timeframe.H1, decided_at, 300))

    def test_before_any_bar_has_closed_both_return_nothing(self) -> None:
        data = frame(Timeframe.H1, 400)
        decided_at = (data.index[0] - timedelta(days=1)).to_pydatetime()

        assert len(new_window(data, Timeframe.H1, decided_at, 300)) == 0
        assert len(old_window(data, Timeframe.H1, decided_at, 300)) == 0


class TestTheReplayStillRefusesThinHistory:
    def test_a_decision_without_120_bars_is_skipped(self) -> None:
        """The floor that keeps an indicator from being computed on nothing."""

        class _Engine:
            config = None

            def evaluate(self, *_args, **_kwargs):  # pragma: no cover - never reached
                raise AssertionError("evaluated a decision with too little history")

        frames = {tf: frame(tf, 130) for tf in REPLAY_TIMEFRAMES}
        replay = HistoricalContextReplay(_Engine())  # type: ignore[arg-type]

        produced = list(
            replay.ideas(
                "TEST",
                frames,
                point=0.00001,
                start=frames[Timeframe.H1].index[0].to_pydatetime(),
                end=frames[Timeframe.H1].index[5].to_pydatetime(),
            )
        )

        assert produced == []


class TestTheDecisionClockIsTheStrategyClock:
    """Loading M1 is not measuring M1 when decisions are still sampled hourly."""

    @staticmethod
    def _engine():  # type: ignore[no-untyped-def]
        class _Engine:
            config = type("_C", (), {"weights": {}, "minimum_confidence": 0.0})()

            def evaluate(self, ctx, _mode):  # type: ignore[no-untyped-def]
                from analysis.confluence import TradeIdea

                return TradeIdea(ctx.symbol, False, None, 0.0, 0.0, 0.0, 0.0, 0.0, "clock", ())

        return _Engine()

    def test_m1_clock_evaluates_every_closed_minute(self) -> None:
        frames = {tf: ending_at(tf, 400) for tf in (*REPLAY_TIMEFRAMES, Timeframe.M1)}
        minute = frames[Timeframe.M1]
        start = minute.index[300].to_pydatetime()
        end = (minute.index[-1] + Timeframe.M1.duration).to_pydatetime()

        replay = HistoricalContextReplay(
            self._engine(),  # type: ignore[arg-type]
            decision_timeframe=Timeframe.M1,
        )
        produced = list(replay.ideas("TEST", frames, point=0.01, start=start, end=end))

        assert len(produced) == 100
        assert all(
            right[0] - left[0] == Timeframe.M1.duration
            for left, right in pairwise(produced)
        )

    def test_m1_clock_uses_the_m1_quote_not_a_stale_m5_close(self) -> None:
        seen_mid: list[float] = []

        class _QuoteEngine:
            config = type("_C", (), {"weights": {}, "minimum_confidence": 0.0})()

            def evaluate(self, ctx, _mode):  # type: ignore[no-untyped-def]
                from analysis.confluence import TradeIdea

                seen_mid.append((ctx.tick.bid + ctx.tick.ask) / 2)
                return TradeIdea(ctx.symbol, False, None, 0.0, 0.0, 0.0, 0.0, 0.0, "x", ())

        frames = {tf: ending_at(tf, 400) for tf in (*REPLAY_TIMEFRAMES, Timeframe.M1)}
        minute = frames[Timeframe.M1].copy()
        minute.loc[:, "close"] = range(len(minute))
        minute.loc[:, "spread"] = 2.0
        frames[Timeframe.M1] = minute
        start = minute.index[300].to_pydatetime()
        end = (minute.index[301] + Timeframe.M1.duration).to_pydatetime()

        replay = HistoricalContextReplay(
            _QuoteEngine(),  # type: ignore[arg-type]
            decision_timeframe=Timeframe.M1,
        )
        produced = list(replay.ideas("TEST", frames, point=0.01, start=start, end=end))

        assert len(produced) == 2
        assert produced[0][1] == pytest.approx(0.02)
        # A decision at the open time of bar 300 may only use bar 299, which
        # has actually closed. The next decision advances to bar 300.
        assert seen_mid == [299.0, 300.0]

    def test_missing_requested_clock_is_an_explicit_error(self) -> None:
        frames = {tf: ending_at(tf, 400) for tf in REPLAY_TIMEFRAMES}
        replay = HistoricalContextReplay(
            self._engine(),  # type: ignore[arg-type]
            decision_timeframe=Timeframe.M1,
        )

        with pytest.raises(ValueError, match="decision timeframe M1 is missing"):
            list(
                replay.ideas(
                    "TEST",
                    frames,
                    point=0.01,
                    start=frames[Timeframe.H1].index[300].to_pydatetime(),
                    end=(frames[Timeframe.H1].index[-1] + Timeframe.H1.duration).to_pydatetime(),
                )
            )

    def test_context_enricher_runs_before_the_engine(self) -> None:
        seen: list[bool] = []

        class _Engine:
            config = type("_C", (), {"weights": {}, "minimum_confidence": 0.0})()

            def evaluate(self, ctx, _mode):  # type: ignore[no-untyped-def]
                from analysis.confluence import TradeIdea

                seen.append(bool(ctx.meta.get("peers_attached")))
                return TradeIdea(ctx.symbol, False, None, 0.0, 0.0, 0.0, 0.0, 0.0, "x", ())

        frames = {tf: ending_at(tf, 400) for tf in REPLAY_TIMEFRAMES}
        replay = HistoricalContextReplay(
            _Engine(),  # type: ignore[arg-type]
            context_enricher=lambda ctx: ctx.meta.update({"peers_attached": True}),
        )
        hourly = frames[Timeframe.H1].index

        list(
            replay.ideas(
                "TEST",
                frames,
                point=0.01,
                start=hourly[300].to_pydatetime(),
                end=(hourly[-1] + Timeframe.H1.duration).to_pydatetime(),
            )
        )

        assert seen and all(seen)


class TestTheRoundTripCrossesTheSpread:
    """Every backtest this project ever ran was filled at the mid on both sides.

    Both replays computed the broker's own recorded spread in order to build a
    tick, handed that tick to the engine, and then dropped it. The backtester
    charged commission and exit slippage and nothing for the crossing — so a
    long was bought at the mid and sold at the mid, which is not a trade anyone
    can place.

    At this account's stop widths it is not a rounding error. On a 7-pip stop
    on EURUSD a 0.8-pip spread is 0.11R, against the 0.16R the backtester was
    charging in total: the largest single cost was the one missing.
    """

    @staticmethod
    def _order(spread: float):  # type: ignore[no-untyped-def]
        from backtesting.engine import BacktestOrder
        from core.types import Direction

        return BacktestOrder(
            symbol="EURUSD",
            decided_at=START,
            direction=Direction.LONG,
            entry=1.1000,
            stop_loss=1.0993,  # a 7-pip stop
            take_profit=1.1021,
            spread=spread,
        )

    @staticmethod
    def _winning_bars():  # type: ignore[no-untyped-def]
        closes = [1.1000 + i * 0.0004 for i in range(12)]
        index = pd.date_range(START, periods=len(closes), freq="5min", tz=UTC)
        return pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.0002 for c in closes],
                "low": [c - 0.0001 for c in closes],
                "close": closes,
            },
            index=index,
        )

    def test_a_wider_spread_costs_more(self) -> None:
        from backtesting.engine import PessimisticBacktester

        engine = PessimisticBacktester()
        bars = self._winning_bars()

        free = engine.run(bars, [self._order(0.0)]).trades[0]
        real = engine.run(bars, [self._order(0.00008)]).trades[0]

        assert real.costs_r > free.costs_r
        assert real.net_r < free.net_r

    def test_the_spread_is_charged_once_and_in_units_of_the_stop(self) -> None:
        """A long is filled at the ask and closed at the bid: one crossing.
        `risk.PositionSizer._cost_share` counts it the same way, and two
        definitions of one cost would eventually disagree."""
        from backtesting.engine import PessimisticBacktester

        engine = PessimisticBacktester()
        bars = self._winning_bars()
        spread = 0.00008

        free = engine.run(bars, [self._order(0.0)]).trades[0]
        real = engine.run(bars, [self._order(spread)]).trades[0]

        assert real.costs_r - free.costs_r == pytest.approx(spread / 0.0007, rel=1e-6)

    def test_a_hand_built_order_still_defaults_to_no_spread(self) -> None:
        """Zero is the honest default for an order nobody attached a quote to —
        a lower bound that says so, rather than an invented average."""
        from backtesting.engine import BacktestOrder
        from core.types import Direction

        assert (
            BacktestOrder(
                symbol="X",
                decided_at=START,
                direction=Direction.LONG,
                entry=1.0,
                stop_loss=0.9,
                take_profit=1.2,
            ).spread
            == 0.0
        )

    def test_the_replay_hands_the_recorded_spread_to_the_order(self) -> None:
        """The half that was missing: the number existed and never travelled."""

        class _Engine:
            config = type("_C", (), {"weights": {"m": 1.0}, "minimum_confidence": 0.0})()

            def evaluate(self, ctx, _mode):  # type: ignore[no-untyped-def]
                from analysis.confluence import TradeIdea
                from core.types import Direction, Signal

                price = float(ctx.series[Timeframe.M5].df["close"].iloc[-1])
                return TradeIdea(
                    symbol=ctx.symbol,
                    approved=True,
                    direction=Direction.LONG,
                    score=50.0,
                    confidence=0.8,
                    entry=price,
                    stop_loss=price - 0.001,
                    take_profit=price + 0.002,
                    reason="test",
                    signals=(Signal("m", 50.0, 0.8),),
                )

        # Every timeframe has to END together, or the daily frame has only a
        # dozen closed bars by the time the hourly one reaches its decisions and
        # the replay skips every one of them for want of history.
        frames = {tf: ending_at(tf, 400) for tf in REPLAY_TIMEFRAMES}
        for data in frames.values():
            data["spread"] = 8  # points
        replay = HistoricalContextReplay(_Engine())  # type: ignore[arg-type]

        hourly = frames[Timeframe.H1].index
        produced = replay.orders(
            "TEST",
            frames,
            point=0.00001,
            start=hourly[300].to_pydatetime(),
            end=(hourly[-1] + Timeframe.H1.duration).to_pydatetime(),
        )

        assert produced
        assert all(order.spread == pytest.approx(8 * 0.00001) for order in produced)

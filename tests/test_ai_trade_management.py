"""Event-driven AI management reacts to evidence, not polling noise."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from analysis.position_health import PositionHealth
from core.types import Direction, Position, Tick
from runner.service import JarvisRunner, _SupervisionSnapshot

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)


class _Journal:
    def open_trade_by_ticket(self, _ticket: int) -> dict[str, float]:
        return {"sl": 90.0, "mfe_r": 0.2}


class _Broker:
    bid = 101.0

    def tick(self, _symbol: str) -> Tick:
        return Tick("TEST", NOW, self.bid, self.bid + 0.1)


def _runner() -> JarvisRunner:
    runner = object.__new__(JarvisRunner)
    runner.journal = _Journal()  # type: ignore[assignment]
    runner.broker = _Broker()  # type: ignore[assignment]
    # `equity` alongside `last_health`: the supervision trigger reads it to
    # express profit as a share of the account, which is the rung the R
    # ladder cannot see on a wide-stopped trade. Zero means "no equity
    # known", which switches that trigger off and leaves the R ladder as the
    # only one — the behaviour these tests were written against.
    runner.manager = SimpleNamespace(last_health={}, equity=0.0)  # type: ignore[assignment]
    runner.settings = SimpleNamespace(  # type: ignore[assignment]
        trade_management=SimpleNamespace(
            supervision_interval_minutes=15.0,
            supervision_event_driven=True,
            supervision_min_interval_minutes=2.0,
            supervision_profit_step_r=0.25,
            supervision_giveback_trigger_fraction=0.25,
            giveback_arm_r=0.5,
        )
    )
    runner._supervised_at = {}
    runner._supervision_due_at = {}
    runner._supervision_snapshots = {}
    return runner


def _position() -> Position:
    return Position(
        ticket=1,
        symbol="TEST",
        direction=Direction.LONG,
        volume=0.01,
        price_open=100.0,
        sl=95.0,
        tp=120.0,
        profit=1.0,
        swap=0.0,
        opened_at=NOW - timedelta(hours=1),
    )


def test_a_new_position_is_reviewed_immediately() -> None:
    triggered = _runner()._supervision_trigger(_position(), NOW)

    assert triggered is not None
    assert triggered[0] == "position_opened"


def test_worsening_health_brings_the_review_forward() -> None:
    runner = _runner()
    runner._supervised_at[1] = NOW - timedelta(minutes=3)
    runner._supervision_due_at[1] = NOW + timedelta(minutes=12)
    runner._supervision_snapshots[1] = _SupervisionSnapshot(0.1, 0.2, 0.5, "healthy", 0.0)
    runner.manager.last_health[1] = PositionHealth(
        "deteriorating", 0.6, "tighten", (), "structure weakened"
    )

    triggered = runner._supervision_trigger(_position(), NOW)

    assert triggered is not None
    assert triggered[0] == "health_worsened:healthy->deteriorating"


def test_the_cost_cooldown_blocks_repeated_reconsideration() -> None:
    runner = _runner()
    runner._supervised_at[1] = NOW - timedelta(seconds=30)
    runner._supervision_due_at[1] = NOW + timedelta(minutes=14)
    runner._supervision_snapshots[1] = _SupervisionSnapshot(0.1, 0.2, 0.5, "healthy", 0.0)
    runner.manager.last_health[1] = PositionHealth(
        "broken", 1.0, "exit", (), "structure invalidated"
    )

    assert runner._supervision_trigger(_position(), NOW) is None


class TestMoneyMovesEvenWhenRDoesNot:
    """The live CADCHF long, and why nothing asked the reviewer about it.

    Entry 0.58542, stop 0.58422, price 0.58595: EUR 2.82 on a EUR 130 account,
    over two percent of everything, and only 0.44R because the stop was twelve
    pips wide. Every supervision trigger was written in R, so the 0.25R ladder
    had last spoken at EUR 1.60 and the next rung was EUR 3.20. The money moved
    and nobody was asked.
    """

    @staticmethod
    def _snapshot(r_now: float, pct: float):  # type: ignore[no-untyped-def]
        from runner.service import _SupervisionSnapshot

        return _SupervisionSnapshot(
            r_now=r_now,
            peak_r=r_now,
            giveback_fraction=0.0,
            health_verdict="healthy",
            health_severity=0.0,
            profit_pct_of_equity=pct,
        )

    def test_the_cash_ladder_fires_where_the_r_ladder_is_silent(self) -> None:
        """0.25R to 0.44R crosses no R rung; 1.2% to 2.2% crosses two cash ones."""
        from config.schema import TradeManagementConfig

        config = TradeManagementConfig()
        before = self._snapshot(0.25, 1.2)
        after = self._snapshot(0.44, 2.2)

        r_step = config.supervision_profit_step_r
        assert int(after.r_now / r_step) == int(before.r_now / r_step), "the R ladder is silent"

        cash_step = config.supervision_profit_step_equity_pct
        assert int(after.profit_pct_of_equity / cash_step) > int(
            before.profit_pct_of_equity / cash_step
        )

    def test_it_can_be_switched_off(self) -> None:
        from config.schema import TradeManagementConfig

        assert TradeManagementConfig(supervision_profit_step_equity_pct=0.0)


class TestTheReviewerIsToldWhetherTheProfitIsSafe:
    """It had every number needed to work this out and was never told it."""

    @staticmethod
    def _payload(stop: float):  # type: ignore[no-untyped-def]
        from datetime import UTC, datetime

        import pandas as pd

        from advisory.providers import build_supervision_payload
        from core.types import Direction, MarketContext, Position, Series, Tick, Timeframe

        now = datetime(2026, 8, 14, 19, 40, tzinfo=UTC)
        index = pd.date_range("2026-08-14", periods=60, freq="15min", tz=UTC)
        close = pd.Series([0.5850 + i * 0.00002 for i in range(60)], index=index)
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
        context = MarketContext(
            symbol="CADCHF.i",
            now=now,
            series={Timeframe.M15: Series("CADCHF.i", Timeframe.M15, frame, now)},
            tick=Tick("CADCHF.i", now, 0.58595, 0.58600),
        )
        position = Position(
            ticket=134663779,
            symbol="CADCHF.i",
            direction=Direction.LONG,
            volume=0.05,
            price_open=0.58542,
            sl=stop,
            tp=0.58691,
            profit=2.82,
            swap=0.0,
            opened_at=now,
        )
        return build_supervision_payload(position, context, {"account_equity": 130.0})

    def test_a_stop_below_entry_is_reported_as_unprotected(self) -> None:
        """The live shape: 2.2% of the account showing, none of it safe."""
        payload = self._payload(stop=0.58422)

        assert payload["profit_is_protected"] is False
        assert payload["unrealised_pct_of_account"] == pytest.approx(2.17, abs=0.05)
        assert payload["stop_distance_from_entry_in_r"] == pytest.approx(-1.0, abs=0.01)

    def test_a_stop_above_entry_is_reported_as_protected(self) -> None:
        payload = self._payload(stop=0.58580)

        assert payload["profit_is_protected"] is True
        assert payload["stop_distance_from_entry_in_r"] > 0

    def test_no_stop_at_all_is_never_called_protected(self) -> None:
        """Fails closed. A missing stop is the least protected state there is."""
        payload = self._payload(stop=0.0)

        assert payload["profit_is_protected"] is False

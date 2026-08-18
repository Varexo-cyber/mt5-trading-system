"""The detector scorecard has to be right before anyone acts on it.

This measurement exists to decide which modules get switched off, so an error
in its arithmetic would not produce a wrong number on a screen — it would
produce a deleted detector. The MT5 half cannot be exercised here; the part
that turns replayed orders into per-detector evidence can, and it is the part
that carries the conclusion.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from backtesting.engine import BacktestOrder, PessimisticBacktester
from core.types import Direction
from scripts.backtest_modules import evidence_for

START = datetime(2026, 5, 1, tzinfo=UTC)


def frame(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range(START, periods=len(closes), freq="5min", tz=UTC)
    values = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": values,
            "high": values + 0.0002,
            "low": values - 0.0002,
            "close": values,
            "spread": 2,
        },
        index=index,
    )


def order(modules: tuple[str, ...], *, minute: int, symbol: str = "EURUSD.i") -> BacktestOrder:
    """A long that wins if price climbs 20 pips before losing 10."""
    return BacktestOrder(
        symbol=symbol,
        decided_at=START + timedelta(minutes=minute),
        direction=Direction.LONG,
        entry=1.1000,
        stop_loss=1.0990,
        take_profit=1.1020,
        modules=modules,
        score=40.0,
        confidence=0.8,
    )


class TestOneTradeCountsForEveryDetectorBehindIt:
    def test_a_shared_trade_appears_under_each_module(self) -> None:
        """`WHEN PRESENT` is deliberately generous — a good detector rides along
        with bad ones — and the table has to actually be built that way."""
        rising = frame([1.1000 + i * 0.0002 for i in range(60)])
        orders = [order(("trend_momentum", "liquidity_sweep"), minute=0)]

        evidence = evidence_for(
            {"trend_momentum": orders, "liquidity_sweep": orders},
            {"EURUSD.i": rising},
            PessimisticBacktester(),
        )

        assert {item.module for item in evidence} == {"trend_momentum", "liquidity_sweep"}
        assert all(item.trades == 1 for item in evidence)
        assert all(item.total_r > 0 for item in evidence)

    def test_a_losing_detector_reads_negative(self) -> None:
        falling = frame([1.1000 - i * 0.0002 for i in range(60)])
        orders = [order(("fast_ema_cross",), minute=0)]

        (evidence,) = evidence_for(
            {"fast_ema_cross": orders}, {"EURUSD.i": falling}, PessimisticBacktester()
        )

        assert evidence.total_r < 0
        assert evidence.win_rate == 0.0
        assert evidence.expectancy_r == evidence.total_r / evidence.trades


class TestOneSymbolCannotSuppressAnother:
    def test_slots_are_counted_per_instrument(self) -> None:
        """Non-overlapping means one slot per INSTRUMENT. Pooling the orders
        across symbols first would let a EURUSD trade suppress a XAUUSD entry
        the live account would happily hold beside it — which understates every
        detector that fires broadly.
        """
        rising = frame([1.1000 + i * 0.0002 for i in range(60)])
        orders = [
            order(("trend_momentum",), minute=0, symbol="EURUSD.i"),
            order(("trend_momentum",), minute=0, symbol="GBPUSD.i"),
        ]

        (evidence,) = evidence_for(
            {"trend_momentum": orders},
            {"EURUSD.i": rising, "GBPUSD.i": rising},
            PessimisticBacktester(),
        )

        assert evidence.trades == 2

    def test_a_symbol_without_bars_is_dropped_rather_than_guessed(self) -> None:
        rising = frame([1.1000 + i * 0.0002 for i in range(60)])
        orders = [
            order(("trend_momentum",), minute=0, symbol="EURUSD.i"),
            order(("trend_momentum",), minute=0, symbol="NOBARS"),
        ]

        (evidence,) = evidence_for(
            {"trend_momentum": orders}, {"EURUSD.i": rising}, PessimisticBacktester()
        )

        assert evidence.trades == 1
        assert evidence.proposals == 2


class TestTheWorstDetectorIsListedFirst:
    def test_rows_are_ordered_by_expectancy(self) -> None:
        """The table is read to find what to switch off, so the thing to switch
        off has to be at the top."""
        rising = frame([1.1000 + i * 0.0002 for i in range(60)])
        falling = frame([1.1000 - i * 0.0002 for i in range(60)])

        evidence = evidence_for(
            {
                "winner": [order(("winner",), minute=0)],
                "loser": [order(("loser",), minute=0, symbol="GBPUSD.i")],
            },
            {"EURUSD.i": rising, "GBPUSD.i": falling},
            PessimisticBacktester(),
        )

        assert [item.module for item in evidence] == ["loser", "winner"]

    def test_a_detector_that_never_closed_a_trade_is_left_out(self) -> None:
        """Zero trades is not a score of zero. A row of dashes invites somebody
        to read it as neutral evidence."""
        assert (
            evidence_for(
                {"never_fired": []}, {"EURUSD.i": frame([1.1] * 60)}, PessimisticBacktester()
            )
            == []
        )

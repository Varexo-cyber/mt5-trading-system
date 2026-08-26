"""Halting new risk on a missing closing deal is right for seconds and a
deadlock after that.

WHAT HAPPENED. The owner opened a manual XAUUSD position, removed its stop and
sized it at 1 lot by accident, and lost about 100 EUR. Afterwards the runner
printed this on every single cycle:

    NEW RISK HALTED: SYSTEM_HALTED - broker/journal reconciliation
    cycle: 434/823 scanned, 239 eligible, 0 analysed, 0 opened

`reconcile` finds a journal trade the broker no longer holds, asks for the
closing deal to settle the accounting, cannot get one, and emits
BROKER_CLOSED_PENDING_HISTORY. `run_once` halts new risk on that. Correct --
deal history lags the position list, and trading on a book you cannot
reconcile is exactly what should be refused.

But nothing ever resolved it. The row stayed open, the next cycle re-detected
it, and the next cycle halted again. A restart did not help either: the halt
lives in memory but is rebuilt within one cycle. The account stops trading for
ever over a number that is never going to arrive.

And what is unknown there is the ACCOUNTING, not the risk. The broker has no
position, so there is nothing left on the account to be wrong about. Refusing
to trade over a missing profit-and-loss figure protects nothing at all.

So the halt is now bounded. Inside the grace window it still halts; past it the
trade is closed in the journal with its P&L left NULL rather than invented, and
the account trades again.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from execution.manager import PENDING_HISTORY_GRACE_SECONDS, PositionManager


class _Journal:
    def __init__(self) -> None:
        self.rows = [{"id": 7, "ticket": 4242}]
        self.settled: list[tuple[int, str]] = []

    def open_trades(self):  # type: ignore[no-untyped-def]
        return list(self.rows)

    def open_trade_by_ticket(self, ticket):  # type: ignore[no-untyped-def]
        return None

    def settle_unrecoverable(self, trade_id, reason):  # type: ignore[no-untyped-def]
        self.settled.append((trade_id, reason))
        self.rows = [row for row in self.rows if row["id"] != trade_id]


def _manager(journal: _Journal) -> PositionManager:
    manager = PositionManager.__new__(PositionManager)
    manager.journal = journal  # type: ignore[assignment]
    manager._pending_history_since = {}
    manager.broker = SimpleNamespace(closed_position=lambda ticket: None)  # type: ignore[assignment]
    return manager


def test_it_still_halts_while_the_deal_could_plausibly_arrive() -> None:
    """The behaviour worth keeping. A deal that is seconds late is normal, and
    trading on a book you cannot reconcile is not."""
    journal = _Journal()
    manager = _manager(journal)

    events = manager.reconcile([])

    assert [event.action for event in events] == ["BROKER_CLOSED_PENDING_HISTORY"]
    assert journal.settled == []


def test_past_the_grace_window_it_settles_instead_of_halting_for_ever() -> None:
    """The deadlock, removed. Same trade, same missing deal, enough time
    elapsed that the deal is not coming."""
    journal = _Journal()
    manager = _manager(journal)
    manager.reconcile([])  # starts the clock on this ticket
    manager._pending_history_since[4242] = time.monotonic() - PENDING_HISTORY_GRACE_SECONDS - 1

    events = manager.reconcile([])

    assert [event.action for event in events] == ["BROKER_CLOSED_UNRECOVERABLE"]
    assert journal.settled and journal.settled[0][0] == 7


def test_the_settled_action_is_not_one_that_halts_the_account() -> None:
    """The whole point. `run_once` halts on a fixed set of reconciliation
    actions, and the new one must not be in it -- otherwise this changes the
    message and nothing else."""
    import inspect

    from runner.service import JarvisRunner

    source = inspect.getsource(JarvisRunner.run_once)
    start = source.index("reconciliation_failures")
    halting = source[start : source.index("}", start)]

    assert "BROKER_CLOSED_PENDING_HISTORY" in halting
    assert "BROKER_CLOSED_UNRECOVERABLE" not in halting


def test_a_deal_that_does_arrive_clears_the_timer() -> None:
    """A ticket that recovers must not carry its old grace clock, or a second
    unrelated gap on the same ticket would be written off immediately."""
    journal = _Journal()
    manager = _manager(journal)
    manager.reconcile([])
    assert 4242 in manager._pending_history_since

    manager.broker = SimpleNamespace(  # type: ignore[assignment]
        closed_position=lambda ticket: SimpleNamespace(
            reason="SL", deal_tickets=(1,), exit_price=1.0, pnl_money=-1.0, closed_at=None
        )
    )
    manager.reconcile([])

    assert 4242 not in manager._pending_history_since


@pytest.mark.parametrize("waited", [0.0, PENDING_HISTORY_GRACE_SECONDS / 2])
def test_the_grace_window_is_long_enough_to_be_generous(waited: float) -> None:
    """Fifteen minutes against a deal feed that normally lags by seconds. The
    window exists to recover every honest case, not to hurry the write-off."""
    journal = _Journal()
    manager = _manager(journal)
    manager.reconcile([])
    manager._pending_history_since[4242] = time.monotonic() - waited

    events = manager.reconcile([])

    assert [event.action for event in events] == ["BROKER_CLOSED_PENDING_HISTORY"]

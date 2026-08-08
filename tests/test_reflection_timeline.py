"""A reflection has to see the journey, not the departure and arrival boards.

For its whole life the post-trade reflection was handed entry, exit and P&L and
asked what went wrong. A trade that reached +0.9R, had its stop pulled to break
even, drifted for forty minutes and closed flat was, in that payload, identical
to one that never moved. Every lesson it produced was written without knowing
what the system had actually done — which is most of what there is to learn.

These tests hold the fix in place: the guard's actions, in order, with the R
they fired at and the reason each gave, plus what the trade was worth at its
best and how much of that it kept.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from journal.database import Journal

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


class FrozenClock:
    def now(self) -> datetime:
        return NOW


@pytest.fixture
def journal(tmp_path):  # type: ignore[no-untyped-def]
    made = Journal(tmp_path / "trading.db", clock=FrozenClock())
    made.open()
    yield made
    made.close()


def a_trade(journal: Journal, ticket: int = 555) -> int:
    """One open row, enough for actions to hang off.

    The ticket is a parameter because the column is UNIQUE — as it should be,
    since a broker never reuses one while a position is live.
    """
    cursor = journal.conn.execute(
        """
        INSERT INTO trades (
            symbol, direction, volume, entry_price, sl, tp, risk_money, risk_pct,
            sl_distance_pips, planned_rr, opened_at, equity_before, entry_state, ticket
        ) VALUES ('EURUSD.i','LONG',0.01,1.1000,1.0990,1.1015,1.0,2.0,10.0,1.5,?,151.0,'FILLED',?)
        """,
        (NOW.isoformat(), ticket),
    )
    journal.conn.commit()
    return int(cursor.lastrowid or 0)


class TestReadingTheGuardsActionsBack:
    def test_an_untouched_trade_has_an_empty_timeline(self, journal: Journal) -> None:
        assert journal.management_actions_for(a_trade(journal)) == []

    def test_actions_come_back_oldest_first(self, journal: Journal) -> None:
        """Order is the whole information content. Out of sequence, 'the stop
        moved to break even and then it drifted' and 'it drifted and then the
        stop moved' read the same, and only one of them is a mistake."""
        trade_id = a_trade(journal)
        for minute, action in ((5, "BREAK_EVEN"), (12, "PROFIT_LOCK"), (41, "PEAK_STALL")):
            journal.conn.execute(
                "INSERT INTO management_actions (trade_id, ts, action, r_at_action, note) "
                "VALUES (?, ?, ?, ?, ?)",
                (trade_id, NOW.replace(minute=minute).isoformat(), action, 0.3, "because"),
            )
        journal.conn.commit()

        timeline = journal.management_actions_for(trade_id)

        assert [item["action"] for item in timeline] == [
            "BREAK_EVEN",
            "PROFIT_LOCK",
            "PEAK_STALL",
        ]

    def test_each_entry_carries_the_r_and_the_reason(self, journal: Journal) -> None:
        """Without the R the timeline says what happened and not whether it
        should have; without the reason it says nothing about why."""
        trade_id = a_trade(journal)
        journal.conn.execute(
            "INSERT INTO management_actions "
            "(trade_id, ts, action, old_sl, new_sl, r_at_action, note) "
            "VALUES (?, ?, 'BREAK_EVEN', 1.0990, 1.1000, 0.31, 'move is running')",
            (trade_id, NOW.isoformat()),
        )
        journal.conn.commit()

        entry = journal.management_actions_for(trade_id)[0]

        assert entry["r_at_the_time"] == pytest.approx(0.31)
        assert entry["why"] == "move is running"
        assert entry["stop_moved_from"] == pytest.approx(1.0990)
        assert entry["stop_moved_to"] == pytest.approx(1.1000)

    def test_two_actions_in_the_same_second_keep_their_order(self, journal: Journal) -> None:
        """The guard runs once a second and can fire twice inside one tick.
        Timestamps alone would leave those two in whatever order SQLite felt
        like, so the insert id breaks the tie."""
        trade_id = a_trade(journal)
        for action in ("BREAK_EVEN", "PARTIAL_CLOSE"):
            journal.conn.execute(
                "INSERT INTO management_actions (trade_id, ts, action) VALUES (?, ?, ?)",
                (trade_id, NOW.isoformat(), action),
            )
        journal.conn.commit()

        timeline = journal.management_actions_for(trade_id)

        assert [item["action"] for item in timeline] == ["BREAK_EVEN", "PARTIAL_CLOSE"]

    def test_another_trades_actions_do_not_leak_in(self, journal: Journal) -> None:
        mine, theirs = a_trade(journal, 555), a_trade(journal, 556)
        journal.conn.execute(
            "INSERT INTO management_actions (trade_id, ts, action) VALUES (?, ?, 'BREAK_EVEN')",
            (theirs, NOW.isoformat()),
        )
        journal.conn.commit()

        assert journal.management_actions_for(mine) == []
        assert len(journal.management_actions_for(theirs)) == 1


class TestTheReflectionPromptAsksForIt:
    """A payload carrying the timeline is worth nothing if the instructions
    still ask the old question."""

    def instructions(self) -> str:
        from advisory.providers import _REFLECTION_INSTRUCTIONS

        return _REFLECTION_INSTRUCTIONS

    def test_it_names_the_timeline_field(self) -> None:
        assert "what_the_system_did_and_when" in self.instructions()

    def test_it_asks_for_judgement_on_what_was_knowable_then(self) -> None:
        """Judging a decision by what the next hour revealed teaches the system
        to widen stops, which is the one direction it must never learn."""
        text = self.instructions().lower()

        assert "information available then" in text
        assert "widen stops" in text

    def test_it_still_forbids_the_practices_the_operator_banned(self) -> None:
        text = self.instructions().lower()

        for banned in ("martingale", "averaging down", "grid trading", "raising risk"):
            assert banned in text, banned

    def test_it_says_zero_lessons_is_an_acceptable_answer(self) -> None:
        """Otherwise every ordinary trade produces an invented lesson, and the
        real ones drown."""
        assert "zero is an acceptable" in self.instructions().lower()

    def test_it_asks_for_repeats_rather_than_novelty(self) -> None:
        """The evidence count is the only thing separating a pattern from an
        anecdote, and it only accumulates if the model is willing to repeat
        itself."""
        assert "say the same thing again" in self.instructions().lower()

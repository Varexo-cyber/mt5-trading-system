"""The closed-trade view the deck was missing.

Everything the operator judges the system by — what today cost, which stop was
hit, whether this week is up — is read off this module. A quiet arithmetic bug
here does not crash anything; it just reports a losing week as a flat one, so
the numbers are asserted rather than eyeballed.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dashboard.ledger import (
    closed_trades,
    day_start,
    health_caption,
    live_health,
    management_baseline_report,
    operation_label,
    recent_management,
    starting_equity,
    summarise,
    timeline_candidates,
    trade_timeline,
    week_start,
)
from journal.database import Journal, iso

# A Tuesday afternoon, comfortably inside a trading day and week.
NOW = datetime(2026, 8, 4, 14, 30, tzinfo=UTC)


@pytest.fixture
def database(tmp_path) -> Path:  # type: ignore[no-untyped-def]
    """A real journal at the current schema, not a hand-rolled table.

    Building the schema by hand would let the ledger's SELECT drift away from
    the columns the system actually writes and still pass.
    """
    from core.clock import SimulatedClock

    path = tmp_path / "trading.db"
    Journal(path, SimulatedClock(NOW)).open().close()
    return path


def write_trade(
    path: Path,
    *,
    ticket: int,
    symbol: str = "EURUSD",
    pnl: float = 0.0,
    closed_at: datetime = NOW,
    exit_reason: str = "SL_HIT",
    entry_state: str = "OPEN",
    pnl_r: float | None = None,
    risk_money: float = 2.0,
) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO trades (ticket, symbol, direction, volume, entry_price, sl, tp, "
            "risk_money, risk_pct, sl_distance_pips, opened_at, closed_at, exit_price, "
            "exit_reason, pnl_money, pnl_r, equity_before, entry_state) "
            "VALUES (?, ?, 'LONG', 0.01, 1.1, 1.09, 1.12, ?, 2.0, 10.0, ?, ?, 1.11, ?, ?, ?, "
            "100.0, ?)",
            (
                ticket,
                symbol,
                risk_money,
                # `iso`, not `.isoformat()`: the journal always writes
                # microseconds, and a fixture that does not would let a
                # timestamp-format mismatch pass unnoticed.
                iso(closed_at - timedelta(hours=1)),
                iso(closed_at),
                exit_reason,
                pnl,
                pnl_r,
                entry_state,
            ),
        )


# ------------------------------------------------------------- boundaries ---


def test_the_day_starts_at_the_fx_rollover() -> None:
    """Not midnight. A day measured differently than the risk limits measure it
    would disagree with them every evening between 21:00 and 00:00."""
    assert day_start(NOW) == datetime(2026, 8, 3, 21, 0, tzinfo=UTC)


def test_an_evening_after_the_rollover_belongs_to_the_next_day() -> None:
    evening = datetime(2026, 8, 4, 22, 15, tzinfo=UTC)
    assert day_start(evening) == datetime(2026, 8, 4, 21, 0, tzinfo=UTC)


def test_the_week_opens_on_sunday_evening() -> None:
    """Sunday 2 August 21:00 opens the week containing Tuesday the 4th."""
    assert week_start(NOW) == datetime(2026, 8, 2, 21, 0, tzinfo=UTC)


def test_sunday_evening_is_already_the_new_week() -> None:
    sunday_night = datetime(2026, 8, 9, 22, 0, tzinfo=UTC)
    assert week_start(sunday_night) == datetime(2026, 8, 9, 21, 0, tzinfo=UTC)


# ----------------------------------------------------------------- reading ---


def test_a_missing_database_reads_as_no_trades(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Before the first run there is no file. The deck must render anyway."""
    assert closed_trades(tmp_path / "nothing.db", day_start(NOW)) == []
    assert starting_equity(tmp_path / "nothing.db", "DAY", day_start(NOW)) is None


def test_open_positions_are_not_history(database: Path) -> None:
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO trades (ticket, symbol, direction, volume, entry_price, sl, tp, "
            "risk_money, risk_pct, sl_distance_pips, opened_at, equity_before) "
            "VALUES (1, 'EURUSD', 'LONG', 0.01, 1.1, 1.09, 1.12, 2.0, 2.0, 10.0, ?, 100.0)",
            (iso(NOW),),
        )
    assert closed_trades(database, day_start(NOW)) == []


def test_trades_before_the_boundary_are_yesterday(database: Path) -> None:
    write_trade(database, ticket=1, closed_at=NOW - timedelta(days=2))
    write_trade(database, ticket=2, closed_at=NOW)
    tickets = [trade.ticket for trade in closed_trades(database, day_start(NOW))]
    assert tickets == [2]


def test_abandoned_entries_are_not_losses(database: Path) -> None:
    """An entry the broker refused never held risk.

    Counting it would report a loss that never happened and drag the win rate
    down for a trade that was never placed.
    """
    write_trade(database, ticket=1, pnl=0.0, entry_state="ABANDONED", exit_reason="rejected")
    write_trade(database, ticket=2, pnl=1.5, exit_reason="TP_HIT")
    trades = closed_trades(database, day_start(NOW))
    assert [trade.ticket for trade in trades] == [2]


def test_newest_first(database: Path) -> None:
    write_trade(database, ticket=1, closed_at=NOW - timedelta(hours=3))
    write_trade(database, ticket=2, closed_at=NOW - timedelta(hours=1))
    assert [t.ticket for t in closed_trades(database, day_start(NOW))] == [2, 1]


# ------------------------------------------------------------------- R unit ---


def test_r_is_derived_when_the_column_is_empty(database: Path) -> None:
    """R stays comparable as the account grows; a blank one makes a losing week
    look like a quiet one."""
    write_trade(database, ticket=1, pnl=-2.0, pnl_r=None, risk_money=2.0)
    (trade,) = closed_trades(database, day_start(NOW))
    assert trade.pnl_r == pytest.approx(-1.0)


def test_a_recorded_r_is_trusted_over_the_derivation(database: Path) -> None:
    """Partial closes and scaled exits make money/risk the wrong sum; when the
    system wrote an R, that is the real one."""
    write_trade(database, ticket=1, pnl=3.0, pnl_r=0.8, risk_money=2.0)
    (trade,) = closed_trades(database, day_start(NOW))
    assert trade.pnl_r == pytest.approx(0.8)


def test_no_risk_recorded_leaves_r_blank_rather_than_infinite(database: Path) -> None:
    write_trade(database, ticket=1, pnl=1.0, pnl_r=None, risk_money=0.0)
    (trade,) = closed_trades(database, day_start(NOW))
    assert trade.pnl_r is None


# ------------------------------------------------------------------ outcome ---


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("SL_HIT", "stop loss geraakt"),
        ("TP_HIT", "target geraakt"),
        ("AI_EXIT", "Claude sloot hem"),
        ("GIVEBACK_EXIT", "winst veiliggesteld"),
        ("EVENING_FLAT", "plat vóór de avondspread"),
        ("HEALTH_EXIT", "systeem zag het draaien"),
        ("HEALTH_SECURE", "systeem zag het draaien"),
        ("TIME_EXIT", "te lang niets gedaan"),
        ("NEWS_FLATTEN", "nieuwsblokkade"),
        ("PARTIAL_CLOSE", "deels gesloten"),
        ("ORPHAN_CLOSE", "noodsluiting"),
    ],
)
def test_exit_reasons_read_as_plain_language(database: Path, reason: str, expected: str) -> None:
    write_trade(database, ticket=1, exit_reason=reason)
    (trade,) = closed_trades(database, day_start(NOW))
    assert trade.outcome == expected


def test_an_unlabelled_exit_shows_its_own_token(database: Path) -> None:
    """Flattening the unknown into "gesloten" would hide any exit path nobody
    has named yet — exactly the one worth noticing."""
    write_trade(database, ticket=1, exit_reason="BROKER_MARGIN_CALL")
    (trade,) = closed_trades(database, day_start(NOW))
    assert trade.outcome == "BROKER_MARGIN_CALL"


# ------------------------------------------------------------------ summary ---


def test_the_summary_adds_up(database: Path) -> None:
    write_trade(database, ticket=1, pnl=3.11, pnl_r=1.5, symbol="EURJPY", exit_reason="TP_HIT")
    write_trade(database, ticket=2, pnl=-1.77, pnl_r=-1.0, symbol="AUDJPY")
    write_trade(database, ticket=3, pnl=1.15, pnl_r=0.5, symbol="US2000", exit_reason="AI_EXIT")

    summary = summarise(database, "Vandaag", day_start(NOW), "DAY", day_start(NOW))

    assert len(summary.trades) == 3
    assert summary.realised == pytest.approx(2.49)
    assert summary.wins == 2
    assert summary.losses == 1
    assert summary.win_rate == pytest.approx(2 / 3)
    assert summary.total_r == pytest.approx(1.0)
    assert summary.best is not None and summary.best.symbol == "EURJPY"
    assert summary.worst is not None and summary.worst.symbol == "AUDJPY"


def test_a_breakeven_close_counts_as_a_loss_not_a_win(database: Path) -> None:
    """Zero is not profit. Calling it a win would inflate the win rate with
    trades that paid nothing — which is precisely what break-even stops
    produce, so the distortion would grow as management improves.
    """
    write_trade(database, ticket=1, pnl=0.0, exit_reason="SL_HIT")
    summary = summarise(database, "Vandaag", day_start(NOW), "DAY", day_start(NOW))
    assert (summary.wins, summary.losses) == (0, 1)


def test_an_empty_period_does_not_divide_by_zero(database: Path) -> None:
    summary = summarise(database, "Vandaag", day_start(NOW), "DAY", day_start(NOW))
    assert summary.win_rate == 0.0
    assert summary.realised == 0.0
    assert summary.best is None and summary.worst is None


def test_the_starting_equity_comes_from_the_risk_anchor(database: Path, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The same mark the daily loss limit measures against, so "we started at
    X" on the deck cannot disagree with the number the breaker uses."""
    from core.clock import SimulatedClock

    journal = Journal(database, SimulatedClock(NOW)).open()
    journal.set_equity_mark("DAY", day_start(NOW), 95.06)
    journal.close()

    summary = summarise(database, "Vandaag", day_start(NOW), "DAY", day_start(NOW))
    assert summary.starting_equity == pytest.approx(95.06)


def test_the_anchor_lookup_uses_the_journals_own_timestamp_format(database: Path) -> None:
    """The first version formatted the key with `.isoformat()`.

    The journal writes microseconds, so the exact-match lookup found nothing —
    no error, no empty result to notice, just a deck that permanently claimed
    it did not know what the day opened at. Pinning the encoding here is the
    only thing that catches it, because both spellings are valid ISO-8601.
    """
    from core.clock import SimulatedClock

    boundary = day_start(NOW)
    journal = Journal(database, SimulatedClock(NOW)).open()
    journal.set_equity_mark("DAY", boundary, 88.0)
    journal.close()

    with sqlite3.connect(database) as conn:
        (stored,) = conn.execute("SELECT period_key FROM equity_marks").fetchone()
    assert stored != boundary.isoformat(), "if these ever match, this test proves nothing"
    assert starting_equity(database, "DAY", boundary) == pytest.approx(88.0)


# ------------------------------------------------------- management log ---


def write_action(
    path: Path, *, trade_id: int, action: str, note: str = "", r: float | None = None
) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO management_actions (trade_id, ts, action, note, r_at_action) "
            "VALUES (?, ?, ?, ?, ?)",
            (trade_id, iso(NOW), action, note, r),
        )


def test_the_management_log_reads_newest_first(database: Path) -> None:
    write_trade(database, ticket=1)
    write_action(database, trade_id=1, action="BREAK_EVEN")
    write_action(database, trade_id=1, action="GIVEBACK_EXIT", r=0.7)

    rows = recent_management(database)

    assert [row["Wat"] for row in rows] == ["winst veiliggesteld", "stop naar break-even"]
    assert rows[0]["R"] == pytest.approx(0.7)
    assert rows[0]["Markt"] == "EURUSD"


def test_the_trade_timeline_contains_plan_actions_and_exit(database: Path) -> None:
    write_trade(database, ticket=77, pnl=1.0, pnl_r=0.5, exit_reason="GIVEBACK_EXIT")
    write_action(database, trade_id=1, action="BREAK_EVEN", note="protected", r=0.6)

    candidates = timeline_candidates(database)
    timeline = trade_timeline(database, 77)

    assert [row["ticket"] for row in candidates] == [77]
    assert [row["Gebeurtenis"] for row in timeline["events"]] == [
        "positie geopend",
        "stop naar break-even",
        "positie definitief gesloten",
    ]


def test_management_baseline_report_adds_the_measured_lift(database: Path) -> None:
    write_trade(database, ticket=88, pnl=1.0, pnl_r=0.5, exit_reason="HEALTH_SECURE")
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO management_baselines (trade_id, observed_at, resolved_at, outcome, "
            "baseline_pnl_r, actual_pnl_r, lift_r) VALUES (1,?,?,?,?,?,?)",
            (iso(NOW), iso(NOW), "SL", -1.0, 0.5, 1.5),
        )

    report = management_baseline_report(database)

    assert report["summary"]["n"] == 1
    assert report["summary"]["lift_r"] == pytest.approx(1.5)
    assert report["rows"][0]["ticket"] == 88


def test_an_unmapped_action_shows_its_own_token(database: Path) -> None:
    """A newly added action must be visible the first time it fires, not hidden
    behind a generic label nobody would question."""
    write_trade(database, ticket=1)
    write_action(database, trade_id=1, action="SOME_NEW_RULE")
    assert recent_management(database)[0]["Wat"] == "SOME_NEW_RULE"


def test_the_management_log_respects_its_limit(database: Path) -> None:
    write_trade(database, ticket=1)
    for _ in range(10):
        write_action(database, trade_id=1, action="ATR_TRAIL")
    assert len(recent_management(database, limit=4)) == 4


def test_a_missing_database_has_no_management_log(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert recent_management(tmp_path / "nothing.db") == []


class TestOperationLabel:
    """The mode tile has three states, and the third one is why it exists.

    The heartbeat is written when a cycle *completes*. On a cold cache the
    first cycle pulls every timeframe for the whole catalogue, so for several
    minutes the process is up, the pid file is on disk, and there is no
    heartbeat. Reading that as OFF tells a live operator that nothing is
    running — and the obvious response, starting it again, puts two instances
    on one account fighting over the same positions.
    """

    def test_not_running_is_off(self) -> None:
        assert operation_label(False, {}) == "OFF"

    def test_a_stale_heartbeat_from_a_dead_process_is_still_off(self) -> None:
        assert operation_label(False, {"operation": "experimental_live"}) == "OFF"

    def test_running_without_a_heartbeat_yet_is_starting(self) -> None:
        assert operation_label(True, {}) == "STARTING"

    def test_running_reports_the_operation(self) -> None:
        assert operation_label(True, {"operation": "experimental_live"}) == "EXPERIMENTAL_LIVE"
        assert operation_label(True, {"operation": "monitor"}) == "MONITOR"

    def test_a_heartbeat_missing_its_operation_does_not_claim_to_be_starting(self) -> None:
        """A malformed heartbeat is a different problem from a cold start."""
        assert operation_label(True, {"ts": "2026-08-06T08:15:00Z"}) == "OFF"


class TestHealthCaption:
    """Three different problems used to share one misleading sentence.

    "geen live oordeel (draait Jarvis?)" was shown for every cause at once.
    Seen on a live deck beside a mode tile reading EXPERIMENTAL_LIVE, that
    question answers itself, and the operator is left concluding the panel is
    broken rather than learning which of three real faults they have.
    """

    def test_jarvis_stopped_says_so_plainly(self) -> None:
        caption = health_caption(None, jarvis_running=False)
        assert "Jarvis draait niet" in caption

    def test_jarvis_running_but_no_entry_points_at_the_guard(self) -> None:
        """The case from the live deck. Never ask a question with a visible answer."""
        caption = health_caption(None, jarvis_running=True)
        assert "draait Jarvis?" not in caption
        assert "guard_tick_failed" in caption

    def test_an_unmanaged_position_says_what_is_holding_it(self) -> None:
        """The most serious state, and it used to render as a shrug.

        No health read means no give-back, no profit lock, no peak stall and no
        time exit — the broker stop is the only thing on the trade.
        """
        caption = health_caption(
            {
                "verdict": "unmanaged",
                "action": "hold",
                "reason": "no open trade on record for this ticket",
                "signals": [],
                "age_seconds": 1.0,
            }
        )
        assert "NIET BEHEERD" in caption
        assert "no open trade on record" in caption

    def test_a_normal_reading_is_unchanged(self) -> None:
        caption = health_caption(
            {
                "verdict": "healthy",
                "action": "hold",
                "reason": "",
                "signals": [],
                "age_seconds": 2.0,
            }
        )
        assert "gezond" in caption
        assert "oud, niet live" not in caption

    def test_a_stale_reading_is_labelled_as_stale(self) -> None:
        """A frozen file otherwise renders a twenty-minute-old verdict with a
        green tick, which is the most confidently wrong state available."""
        caption = health_caption(
            {
                "verdict": "healthy",
                "action": "hold",
                "reason": "",
                "signals": [],
                "age_seconds": 1200.0,
            }
        )
        assert "gezond" in caption
        assert "20 min oud, niet live" in caption

    def test_an_unknown_age_is_not_guessed_at(self) -> None:
        """A file without `recorded_at` gives no age. Silence beats a made-up one."""
        caption = health_caption(
            {"verdict": "healthy", "action": "hold", "signals": [], "age_seconds": None}
        )
        assert "niet live" not in caption


class TestLiveHealthAge:
    def test_each_entry_carries_the_files_age(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        written = datetime(2026, 8, 6, 9, 40, tzinfo=UTC)
        path = tmp_path / "position_health.json"
        path.write_text(
            json.dumps(
                {
                    "recorded_at": written.isoformat(),
                    "positions": [{"ticket": 134372061, "verdict": "healthy"}],
                }
            ),
            encoding="utf-8",
        )

        entries = live_health(path, now=written + timedelta(minutes=4))
        assert entries[134372061]["age_seconds"] == pytest.approx(240.0)

    def test_a_missing_file_is_an_empty_map_not_a_crash(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        assert live_health(tmp_path / "nope.json") == {}

    def test_a_corrupt_file_is_an_empty_map(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "position_health.json"
        path.write_text("{ not json", encoding="utf-8")
        assert live_health(path) == {}

    def test_an_unparseable_timestamp_leaves_the_age_unknown(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Rather than defaulting to zero, which would read as freshly written."""
        path = tmp_path / "position_health.json"
        path.write_text(
            json.dumps({"recorded_at": "whenever", "positions": [{"ticket": 1}]}),
            encoding="utf-8",
        )
        assert live_health(path)[1]["age_seconds"] is None

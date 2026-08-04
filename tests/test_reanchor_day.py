"""Re-anchoring must be impossible on a day the system itself lost money.

The daily limit measures from where the day started, and the anchor is
write-once precisely so a restart cannot hand back a fresh budget after a bad
morning. The one case that justifies moving it is a loss the system had no part
in — trades placed by hand in the terminal, with the anchor written before them.

That distinction is the whole safety property, so it is what these tests cover.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from config.loader import load_settings
from core.clock import SimulatedClock
from core.types import Direction
from journal.database import Journal, iso
from journal.recorder import Recorder
from risk.position_sizer import PositionSizer
from tests.fakes.fake_mt5 import eurusd_spec

# Wednesday 10:00 UTC — inside the day that began Tuesday 21:00.
NOW = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    with Journal(tmp_path / "j.db", SimulatedClock(NOW)) as j:
        yield j


def test_the_anchor_is_write_once(journal: Journal) -> None:
    """The property re-anchoring must not casually undo.

    Without it, restarting after losing 2% would erase the loss and hand back a
    full daily budget — a limit that resets on relaunch is not a limit.
    """
    day = journal.day_start()
    journal.set_equity_mark("DAY", day, 100.0)
    journal.set_equity_mark("DAY", day, 90.0)

    assert journal.equity_mark("DAY", day) == 100.0


def test_a_day_with_no_system_trades_is_the_re_anchorable_case(journal: Journal) -> None:
    journal.set_equity_mark("DAY", journal.day_start(), 100.0)
    assert journal.trades_since(journal.day_start()) == 0


def test_a_day_with_system_trades_is_not(journal: Journal) -> None:
    """The script refuses here, and this is the condition it refuses on."""
    from core.instrument import InstrumentSpec

    settings = load_settings(env_overrides=False)
    day = journal.day_start()
    journal.set_equity_mark("DAY", day, 100.0)

    sizing = PositionSizer(settings).size(
        spec=InstrumentSpec.from_mt5(eurusd_spec()),
        equity=10_000.0,
        direction=Direction.LONG,
        entry=1.08500,
        sl=1.08300,
        tp=1.09100,
    )
    Recorder(journal, SimulatedClock(NOW), settings).record_trade_open(
        cycle_pk=None,
        sizing=sizing,
        ticket=1,
        entry_price=1.085,
        equity_before=100.0,
        opened_at=NOW - timedelta(hours=2),
    )
    journal.conn.commit()

    assert journal.trades_since(day) == 1, "a trade inside the day must be counted"


def test_moving_the_anchor_leaves_the_equity_peak_alone(journal: Journal) -> None:
    """Re-anchoring is not a reset. The drawdown breaker runs off the peak and
    a bad morning has already spent part of it — that must not come back."""
    day = journal.day_start()
    journal.set_equity_mark("DAY", day, 100.0)
    journal.record_equity_peak(100.0)

    journal.conn.execute(
        "UPDATE equity_marks SET equity = ? WHERE period = 'DAY' AND period_key = ?",
        (90.55, iso(day)),
    )
    journal.conn.commit()

    assert journal.equity_mark("DAY", day) == 90.55
    assert journal.equity_peak() == 100.0, "the peak, and so the hard floor, is untouched"


def test_the_weekly_anchor_is_untouched_by_a_daily_re_anchor(journal: Journal) -> None:
    day, week = journal.day_start(), journal.week_start()
    journal.set_equity_mark("DAY", day, 100.0)
    journal.set_equity_mark("WEEK", week, 100.0)

    journal.conn.execute(
        "UPDATE equity_marks SET equity = ? WHERE period = 'DAY' AND period_key = ?",
        (90.55, iso(day)),
    )
    journal.conn.commit()

    assert journal.equity_mark("WEEK", week) == 100.0

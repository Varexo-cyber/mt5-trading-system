"""The horizon has to survive the trip from the engine to the journal.

THE DEFECT THIS GUARDS IS NOT A WRONG ANSWER, IT IS AN ANSWER THAT NEVER
ARRIVES, and that is the shape of nearly everything found on this account: a
check that exists, is correct, is tested, and sits on a path the code does not
take. `_time_exit_deadline` can be perfect and still read NULL forever if the
value is never written, and every one of its own tests would still pass.

So this asserts the plumbing rather than the arithmetic: the column exists, the
recorder writes what it is handed, and the manager reads back the same number
through the interface it actually uses — a journal row.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from config.loader import load_settings
from config.schema import Settings
from core.clock import SimulatedClock
from core.instrument import InstrumentSpec
from core.types import Direction
from execution.manager import PositionManager
from journal.database import Journal
from journal.recorder import Recorder
from risk.position_sizer import PositionSizer, SizingResult
from tests.fakes.fake_mt5 import eurusd_spec

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(NOW)


@pytest.fixture
def settings() -> Settings:
    return load_settings(env_overrides=False)


@pytest.fixture
def journal(tmp_path: Path, clock: SimulatedClock) -> Journal:
    with Journal(tmp_path / "j.db", clock) as j:
        yield j


@pytest.fixture
def recorder(journal: Journal, clock: SimulatedClock, settings: Settings) -> Recorder:
    return Recorder(journal, clock, settings)


def _sizing(settings: Settings) -> SizingResult:
    return PositionSizer(settings).size(
        spec=InstrumentSpec.from_mt5(eurusd_spec()),
        equity=10_000.0,
        direction=Direction.LONG,
        entry=1.08500,
        sl=1.08300,
        tp=1.09100,
    )


class TestTheColumnExists:
    def test_the_trades_table_can_hold_a_plan_length(self, journal: Journal) -> None:
        columns = {
            row["name"] for row in journal.query("PRAGMA table_info(trades)")  # type: ignore[index]
        }

        assert {"horizon", "expected_horizon_minutes"} <= columns

    def test_an_existing_journal_gains_them_by_migration(self, journal: Journal) -> None:
        """The account has a live journal with real history in it. This has to
        arrive as a migration on that file, not only as a fresh CREATE TABLE
        that a new install would get."""
        from journal.database import _MIGRATIONS

        statements = " ".join(_MIGRATIONS[8])

        assert "ALTER TABLE trades ADD COLUMN horizon" in statements
        assert "ALTER TABLE trades ADD COLUMN expected_horizon_minutes" in statements


class TestTheRecorderWritesIt:
    def test_a_planned_trade_carries_its_horizon(
        self, recorder: Recorder, journal: Journal, settings: Settings
    ) -> None:
        trade_id = recorder.record_entry_intent(
            cycle_pk=None,
            sizing=_sizing(settings),
            equity_before=10_000.0,
            horizon="intraday",
            expected_horizon_minutes=180,
        )

        row = journal.query("SELECT * FROM trades WHERE id = ?", (trade_id,))[0]

        assert row["horizon"] == "intraday"
        assert row["expected_horizon_minutes"] == 180

    def test_an_unplanned_position_writes_nothing_rather_than_a_guess(
        self, recorder: Recorder, journal: Journal, settings: Settings
    ) -> None:
        """A position adopted from the terminal has no plan behind it. NULL is
        the truthful value and the manager falls back to the old constant on
        it; a default of 1,440 would have been a lie that happened to look
        like the old behaviour."""
        trade_id = recorder.record_trade_open(
            cycle_pk=None,
            sizing=_sizing(settings),
            ticket=1,
            entry_price=1.08500,
            equity_before=10_000.0,
        )

        row = journal.query("SELECT * FROM trades WHERE id = ?", (trade_id,))[0]

        assert row["horizon"] is None
        assert row["expected_horizon_minutes"] is None


class TestTheManagerReadsItBack:
    """Through a real `sqlite3.Row`, because that is the type the guard loop
    holds and it raises `IndexError` for an unknown column where a dict raises
    `KeyError`. A test that only ever passes dicts proves the wrong thing."""

    def test_the_deadline_comes_from_the_row_the_recorder_wrote(
        self, recorder: Recorder, journal: Journal, settings: Settings
    ) -> None:
        trade_id = recorder.record_entry_intent(
            cycle_pk=None,
            sizing=_sizing(settings),
            equity_before=10_000.0,
            horizon="intraday",
            expected_horizon_minutes=180,
        )
        row = journal.query("SELECT * FROM trades WHERE id = ?", (trade_id,))[0]

        deadline = PositionManager._time_exit_deadline(settings.trade_management, row, 1.0)

        # Three hours of plan and a 1.5 multiple, against the 24 it got before.
        assert deadline == pytest.approx(4.5)

    def test_an_unplanned_row_still_gets_the_old_constant(
        self, recorder: Recorder, journal: Journal, settings: Settings
    ) -> None:
        trade_id = recorder.record_trade_open(
            cycle_pk=None,
            sizing=_sizing(settings),
            ticket=2,
            entry_price=1.08500,
            equity_before=10_000.0,
        )
        row = journal.query("SELECT * FROM trades WHERE id = ?", (trade_id,))[0]

        deadline = PositionManager._time_exit_deadline(settings.trade_management, row, 1.0)

        assert deadline == pytest.approx(settings.trade_management.time_exit_hours)


class TestTheEngineStillProducesTheNumber:
    """The source end of the same pipe. If the engine stops writing minutes,
    everything above keeps passing while every trade silently falls back to the
    constant — which is precisely how this defect existed unnoticed."""

    @pytest.mark.parametrize(
        ("horizon", "minutes"),
        [("swing", 24 * 60), ("intraday", 12 * 15), ("quick", 6 * 5)],
    )
    def test_each_profile_still_means_the_hours_this_work_assumed(
        self, horizon: str, minutes: int
    ) -> None:
        confluence = load_settings(
            overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence
        profile = confluence.horizon_profiles[horizon]
        from core.types import Timeframe

        planned = (
            Timeframe.parse(profile.planning_timeframe).duration.total_seconds()
            / 60
            * profile.target_horizon_bars
        )

        assert planned == pytest.approx(minutes)

"""Two writes that destroyed good data at the end of a trade.

Both were found in one live postmortem of EURGBP.i, which printed

    entry              0.00000
    best it reached    —

on a trade whose own management log said, three lines further down, "peak
0.92R, now 0.83R; stop secures 0.46R at the broker".

Both bugs have the same shape: a late write overwrites a value that was
already correct. Neither affected a live decision — the manager reads the
broker's `price_open` and keeps the peak in memory — so the trade was handled
properly and only its record was wrong. That is the kind of corruption that
survives for months while quietly poisoning every postmortem, the weekly
report, and everything the account learns from its own history.
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
from journal.database import Journal
from journal.recorder import Recorder
from risk.position_sizer import PositionSizer, SizingResult
from tests.fakes.fake_mt5 import eurusd_spec

NOW = datetime(2026, 8, 6, 9, 58, tzinfo=UTC)


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(NOW)


@pytest.fixture
def settings() -> Settings:
    return load_settings(env_overrides=False)


@pytest.fixture
def journal(tmp_path: Path, clock: SimulatedClock) -> Journal:
    with Journal(tmp_path / "j.db", clock) as opened:
        yield opened


@pytest.fixture
def recorder(journal: Journal, clock: SimulatedClock, settings: Settings) -> Recorder:
    return Recorder(journal, clock, settings)


@pytest.fixture
def sizing(settings: Settings) -> SizingResult:
    return PositionSizer(settings).size(
        spec=InstrumentSpec.from_mt5(eurusd_spec()),
        equity=10_000.0,
        direction=Direction.LONG,
        entry=1.08500,
        sl=1.08300,
        tp=1.09100,
    )


def pending(recorder: Recorder, sizing: SizingResult) -> int:
    """A recorded intent, exactly as the sizer files one before sending."""
    return recorder.record_trade_open(
        cycle_pk=None,
        sizing=sizing,
        ticket=None,
        entry_price=sizing.entry,
        equity_before=10_000.0,
        entry_state="PENDING",
    )


class TestEntryPriceSurvivesPromotion:
    def test_a_real_fill_price_wins(
        self, journal: Journal, recorder: Recorder, sizing: SizingResult
    ) -> None:
        trade_id = pending(recorder, sizing)
        journal.promote_pending_entry(trade_id, ticket=134372678, entry_price=1.08512)

        row = journal.query("SELECT * FROM trades WHERE id = ?", (trade_id,))[0]
        assert row["entry_price"] == pytest.approx(1.08512)
        assert row["ticket"] == 134372678
        assert row["entry_state"] == "OPEN"

    def test_real_fill_can_replace_every_entry_derived_metric(
        self, journal: Journal, recorder: Recorder, sizing: SizingResult
    ) -> None:
        trade_id = pending(recorder, sizing)
        journal.promote_pending_entry(
            trade_id,
            ticket=134859872,
            entry_price=1.08512,
            filled_risk_money=7.25,
            filled_risk_pct=0.725,
            filled_sl_distance_pips=12.4,
            filled_planned_rr=2.51,
        )

        row = journal.query("SELECT * FROM trades WHERE id = ?", (trade_id,))[0]
        assert row["entry_price"] == pytest.approx(1.08512)
        assert row["risk_money"] == pytest.approx(7.25)
        assert row["risk_pct"] == pytest.approx(0.725)
        assert row["sl_distance_pips"] == pytest.approx(12.4)
        assert row["planned_rr"] == pytest.approx(2.51)

    def test_partial_fill_metrics_are_rejected(
        self, journal: Journal, recorder: Recorder, sizing: SizingResult
    ) -> None:
        trade_id = pending(recorder, sizing)

        with pytest.raises(ValueError, match="must be supplied together"):
            journal.promote_pending_entry(
                trade_id,
                ticket=134859872,
                entry_price=1.08512,
                filled_planned_rr=2.51,
            )

    def test_a_missing_fill_price_keeps_the_sizing_price(
        self, journal: Journal, recorder: Recorder, sizing: SizingResult
    ) -> None:
        """The live case. 0.00000 is not a price, and it replaced a good one."""
        trade_id = pending(recorder, sizing)
        journal.promote_pending_entry(trade_id, ticket=134372678, entry_price=0.0)

        row = journal.query("SELECT * FROM trades WHERE id = ?", (trade_id,))[0]
        assert row["entry_price"] == pytest.approx(sizing.entry)
        # Still promoted. The position is real, and an intent left PENDING is
        # closed as an orphan on the next reconcile.
        assert row["ticket"] == 134372678
        assert row["entry_state"] == "OPEN"

    def test_the_adoption_path_is_guarded_by_the_same_code(
        self, journal: Journal, recorder: Recorder, sizing: SizingResult
    ) -> None:
        """Both routes into OPEN had the overwrite written out separately.

        Fixing one copy of a duplicated bug and leaving the other has already
        happened once in this codebase, so they now share a method.
        """
        pending(recorder, sizing)
        trade_id = journal.claim_pending_entry(
            symbol=sizing.symbol,
            direction="LONG",
            volume=sizing.volume,
            ticket=987654,
            entry_price=0.0,
            volume_tolerance=0.001,
        )
        assert trade_id is not None
        row = journal.query("SELECT * FROM trades WHERE id = ?", (trade_id,))[0]
        assert row["entry_price"] == pytest.approx(sizing.entry)
        assert row["ticket"] == 987654


class TestExcursionsSurviveClosing:
    def test_the_ratchet_is_not_wiped_by_the_closing_write(
        self, journal: Journal, recorder: Recorder, sizing: SizingResult
    ) -> None:
        """The live case. Neither caller passes these, so every trade lost them.

        The guard ratchets the peak once a second for the whole life of the
        trade; the closing write then overwrote both columns with its own
        unset defaults.
        """
        trade_id = pending(recorder, sizing)
        journal.promote_pending_entry(trade_id, ticket=1, entry_price=1.08512)
        journal.update_excursions(trade_id, mae_r=-0.20, mfe_r=0.92)

        recorder.record_trade_close(
            trade_id,
            exit_price=1.08300,
            pnl_money=-100.0,
            exit_reason="SL",
            equity_after=9_900.0,
        )

        row = journal.query("SELECT * FROM trades WHERE id = ?", (trade_id,))[0]
        assert row["mfe_r"] == pytest.approx(0.92)
        assert row["mae_r"] == pytest.approx(-0.20)

    def test_an_explicit_value_still_wins(
        self, journal: Journal, recorder: Recorder, sizing: SizingResult
    ) -> None:
        """The broker-recovery path reconstructs closures the guard never saw,
        and its numbers must be able to replace an incomplete ratchet."""
        trade_id = pending(recorder, sizing)
        journal.promote_pending_entry(trade_id, ticket=1, entry_price=1.08512)
        journal.update_excursions(trade_id, mae_r=-0.20, mfe_r=0.92)

        recorder.record_trade_close(
            trade_id,
            exit_price=1.08300,
            pnl_money=-100.0,
            exit_reason="SL",
            equity_after=9_900.0,
            mae_r=-1.00,
            mfe_r=1.40,
        )

        row = journal.query("SELECT * FROM trades WHERE id = ?", (trade_id,))[0]
        assert row["mfe_r"] == pytest.approx(1.40)
        assert row["mae_r"] == pytest.approx(-1.00)

    def test_a_trade_with_no_ratchet_stays_unknown_rather_than_zero(
        self, journal: Journal, recorder: Recorder, sizing: SizingResult
    ) -> None:
        """ "Unknown" and "it never moved" are different facts.

        A postmortem that cannot tell them apart is worse than one that admits
        it does not know.
        """
        trade_id = pending(recorder, sizing)
        journal.promote_pending_entry(trade_id, ticket=1, entry_price=1.08512)

        recorder.record_trade_close(
            trade_id,
            exit_price=1.08300,
            pnl_money=-100.0,
            exit_reason="SL",
            equity_after=9_900.0,
        )

        row = journal.query("SELECT * FROM trades WHERE id = ?", (trade_id,))[0]
        assert row["mfe_r"] is None
        assert row["mae_r"] is None

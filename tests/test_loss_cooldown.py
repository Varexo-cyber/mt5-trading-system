"""Taking the same instrument again a minute after it took money off us.

The live sequence, 6 August:

    18:51:29  GBPNZD short opened
    18:53:43  GBPNZD short closed, -0.71R
    18:54:58  GBPNZD short opened again
    18:57:36  GBPNZD short closed, -0.48R

Nothing objected, because every position filter asks what is open *now* and
by 18:54:58 nothing was. The playbook then read a window of M15 bars that was
almost exactly the window it had read three minutes earlier, and reached the
same conclusion — not because it re-evaluated the setup but because it
re-encountered it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from config.schema import LossCooldownConfig
from core.instrument import InstrumentSpec
from core.types import Direction
from filters.base import FilterContext
from filters.loss_cooldown import LossCooldownFilter
from risk.reasons import Reason
from tests.fakes.fake_mt5 import eurusd_spec

CLOSED = datetime(2026, 8, 6, 18, 53, 43, tzinfo=UTC)
SPEC = InstrumentSpec.from_mt5(eurusd_spec())


def context(now: datetime, symbol: str = "GBPNZD.i") -> FilterContext:
    return FilterContext(symbol=symbol, spec=SPEC, now=now, direction=Direction.SHORT)


def filter_for(closed_at: datetime | None, **overrides) -> LossCooldownFilter:  # type: ignore[no-untyped-def]
    return LossCooldownFilter(LossCooldownConfig(**overrides), lambda _symbol: closed_at)


def test_the_seventy_five_second_re_entry_is_refused() -> None:
    """The exact case, to the second."""
    verdict = filter_for(CLOSED).check(context(datetime(2026, 8, 6, 18, 54, 58, tzinfo=UTC)))

    assert not verdict.passed
    assert verdict.reason is Reason.LOSS_COOLDOWN
    assert verdict.data["minutes_since_loss"] == 1.2


def test_the_instrument_comes_back_once_the_window_has_passed() -> None:
    verdict = filter_for(CLOSED).check(context(CLOSED + timedelta(minutes=21)))

    assert verdict.passed


def test_the_boundary_belongs_to_the_trade() -> None:
    """Exactly the cooldown is over, not still running. An off-by-one here
    would be invisible and would cost a setup a day."""
    assert filter_for(CLOSED).check(context(CLOSED + timedelta(minutes=20))).passed
    assert not filter_for(CLOSED).check(context(CLOSED + timedelta(minutes=19, seconds=59))).passed


def test_an_instrument_with_no_losing_history_is_free() -> None:
    assert filter_for(None).check(context(CLOSED)).passed


def test_the_other_direction_is_refused_too() -> None:
    """Deliberate. Taking the other side after being wrong can be a real
    reversal read, but flipping within minutes is the same churn with a minus
    sign, and the cost of being wrong here is one setup out of the twelve this
    system ranks every cycle."""
    ctx = FilterContext(
        symbol="GBPNZD.i", spec=SPEC, now=CLOSED + timedelta(minutes=2), direction=Direction.LONG
    )

    assert not filter_for(CLOSED).check(ctx).passed


def test_a_journal_that_cannot_answer_blocks_rather_than_passes() -> None:
    """Same rule as the calendar: no data is not permission."""

    def broken(_symbol: str) -> datetime:
        raise RuntimeError("database is locked")

    verdict = LossCooldownFilter(LossCooldownConfig(), broken).check(context(CLOSED))

    assert not verdict.passed
    assert verdict.reason is Reason.LOSS_COOLDOWN


def test_a_naive_timestamp_does_not_take_the_filter_down() -> None:
    """A journal row written before the timestamps carried a zone would raise
    on the subtraction, and a filter that raises blocks every symbol it has
    history for — including the ones it should be clearing."""
    naive = CLOSED.replace(tzinfo=None)

    verdict = filter_for(naive).check(context(CLOSED + timedelta(minutes=30)))

    assert verdict.passed


def test_it_can_be_switched_off() -> None:
    assert filter_for(CLOSED, enabled=False).check(context(CLOSED)).passed
    assert filter_for(CLOSED, minutes=0).check(context(CLOSED)).passed

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


WON = datetime(2026, 8, 25, 6, 58, 1, tzinfo=UTC)


def filter_after(
    last_loss: datetime | None, last_close: datetime | None, **overrides
):  # type: ignore[no-untyped-def]
    return LossCooldownFilter(
        LossCooldownConfig(**overrides),
        lambda _symbol: last_loss,
        lambda _symbol: last_close,
    )


class TestTheSameChurnAfterAWinner:
    """25 August, live:

        06:58:01  USDJPY short closed, +EUR 1.40
        06:58:37  USDJPY short opened again
        07:08:01  USDJPY short closed, -EUR 1.81, EUR 1.32 of it commission

    Thirty-six seconds, and nothing objected, because the previous close was a
    winner. The mechanism this filter exists to stop is a statement about the
    CHART, not about the last outcome: a winner consumes exactly the same few
    minutes of the window, and the window read next is exactly as stale.
    """

    def test_the_thirty_six_second_re_entry_is_refused(self) -> None:
        verdict = filter_after(None, WON).check(
            context(datetime(2026, 8, 25, 6, 58, 37, tzinfo=UTC), "USDJPY.i")
        )

        assert not verdict.passed
        assert verdict.reason is Reason.LOSS_COOLDOWN
        assert "closed a winner" in verdict.detail

    def test_the_window_after_a_win_is_shorter_than_after_a_loss(self) -> None:
        """After a loss the market has demonstrably disagreed; after a win it
        has not, and a continuation on a fresh leg is a real thing to allow."""
        six_minutes = WON + timedelta(minutes=6)

        assert filter_after(None, WON).check(context(six_minutes, "USDJPY.i")).passed
        assert not filter_after(WON, WON).check(context(six_minutes, "USDJPY.i")).passed

    def test_a_loss_is_still_reported_as_a_loss(self) -> None:
        """Both readings see the same row when the last close lost. Describing
        it as an ordinary close would put the wrong reason in the journal and
        the wrong window in the report."""
        verdict = filter_after(CLOSED, CLOSED).check(context(CLOSED + timedelta(minutes=1)))

        assert not verdict.passed
        assert "closed a loser" in verdict.detail
        assert verdict.data["cooldown_minutes"] == 20.0

    def test_an_older_loss_does_not_hide_a_recent_win(self) -> None:
        """A loss last week and a win a minute ago: the win is what makes the
        chart stale, and the long window has expired."""
        stale_loss = WON - timedelta(days=3)

        verdict = filter_after(stale_loss, WON).check(
            context(WON + timedelta(seconds=36), "USDJPY.i")
        )

        assert not verdict.passed
        assert "closed a winner" in verdict.detail

    def test_zero_restores_the_losses_only_behaviour(self) -> None:
        verdict = filter_after(None, WON, minutes_after_a_win=0.0).check(
            context(WON + timedelta(seconds=36), "USDJPY.i")
        )

        assert verdict.passed

    def test_a_caller_that_passes_no_reader_is_unchanged(self) -> None:
        """Older wiring must keep the filter it asked for rather than silently
        getting a stricter one."""
        verdict = LossCooldownFilter(LossCooldownConfig(), lambda _symbol: None).check(
            context(WON + timedelta(seconds=36), "USDJPY.i")
        )

        assert verdict.passed

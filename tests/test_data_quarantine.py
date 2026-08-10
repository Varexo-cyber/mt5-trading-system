"""Stop re-deriving an answer the broker cannot change.

The live deck showed the same eight symbols refused every cycle for reasons
that are properties of the feed, not of the minute:

    SPCX W1: 8 closed bars available, 50 required
    HSBC H4: 80 bars missing inside trading weeks (5.3%, limit 5.0%)

Each of those cost a full multi-timeframe fetch, on a one-vCPU VPS, against an
800-symbol catalogue, every cycle.

THE SAFETY PROPERTY THESE TESTS EXIST FOR, and it is the only one that matters:
a hold can never make a trade possible. A held symbol was already refused and
is still refused. The candidate set shrinks here and never grows, so no spread,
session, liveliness or risk rule can be reached differently because of it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from core.data_quarantine import DataQuarantine

NOW = datetime(2026, 8, 10, 17, 6, 32, tzinfo=UTC)
HSBC = "HSBC H4: 80 bars missing inside trading weeks (5.3% of the window, limit 5.0%)."


@pytest.fixture
def quarantine() -> DataQuarantine:
    return DataQuarantine(initial_minutes=30.0, backoff_multiple=4.0, max_minutes=240.0)


class TestHolding:
    def test_a_fresh_symbol_is_analysed(self, quarantine: DataQuarantine) -> None:
        assert quarantine.hold_for("HSBC", NOW) is None

    def test_one_failure_holds_it_for_the_initial_window(self, quarantine: DataQuarantine) -> None:
        quarantine.record_failure("HSBC", HSBC, NOW)

        assert quarantine.hold_for("HSBC", NOW + timedelta(minutes=29)) is not None
        assert quarantine.hold_for("HSBC", NOW + timedelta(minutes=31)) is None

    def test_the_hold_is_per_symbol(self, quarantine: DataQuarantine) -> None:
        quarantine.record_failure("HSBC", HSBC, NOW)
        assert quarantine.hold_for("EURUSD", NOW) is None

    def test_expiry_lets_it_be_tried_again(self, quarantine: DataQuarantine) -> None:
        """A broker that backfills history has to be noticed."""
        quarantine.record_failure("SPCX", "8 closed bars available, 50 required", NOW)
        assert quarantine.hold_for("SPCX", NOW + timedelta(hours=2)) is None


class TestBackoff:
    def test_repeat_failures_lengthen_the_hold(self, quarantine: DataQuarantine) -> None:
        """Eight weekly bars where fifty are needed is not an hourly question."""
        first = quarantine.record_failure("SPCX", "too short", NOW)
        second = quarantine.record_failure("SPCX", "too short", NOW)
        third = quarantine.record_failure("SPCX", "too short", NOW)

        assert first.until == NOW + timedelta(minutes=30)
        assert second.until == NOW + timedelta(minutes=120)
        assert third.until == NOW + timedelta(minutes=240)

    def test_the_hold_is_capped(self, quarantine: DataQuarantine) -> None:
        """Nothing is held out forever without another look."""
        for _ in range(20):
            hold = quarantine.record_failure("SPCX", "too short", NOW)
        assert hold.until == NOW + timedelta(minutes=240)

    def test_an_expired_hold_keeps_its_count(self, quarantine: DataQuarantine) -> None:
        """A symbol about to fail its fifth time should not restart at an hour."""
        for _ in range(3):
            quarantine.record_failure("SPCX", "too short", NOW)
        later = NOW + timedelta(days=2)
        assert quarantine.hold_for("SPCX", later) is None

        again = quarantine.record_failure("SPCX", "too short", later)
        assert again.failures == 4
        assert again.until == later + timedelta(minutes=240)


class TestRelease:
    def test_a_clean_analysis_releases_the_symbol(self, quarantine: DataQuarantine) -> None:
        quarantine.record_failure("HSBC", HSBC, NOW)
        quarantine.clear("HSBC")
        assert quarantine.hold_for("HSBC", NOW) is None

    def test_release_forgets_the_count_too(self, quarantine: DataQuarantine) -> None:
        """A backfilled feed is genuinely fixed; it does not owe its old strikes."""
        for _ in range(4):
            quarantine.record_failure("HSBC", HSBC, NOW)
        quarantine.clear("HSBC")

        fresh = quarantine.record_failure("HSBC", HSBC, NOW)
        assert fresh.failures == 1
        assert fresh.until == NOW + timedelta(minutes=30)

    def test_clearing_an_unheld_symbol_is_harmless(self, quarantine: DataQuarantine) -> None:
        quarantine.clear("EURUSD")


class TestReporting:
    def test_held_lists_soonest_to_return_first(self, quarantine: DataQuarantine) -> None:
        quarantine.record_failure("SPCX", "too short", NOW)
        quarantine.record_failure("SPCX", "too short", NOW)
        quarantine.record_failure("HSBC", HSBC, NOW)

        assert [hold.symbol for hold in quarantine.held(NOW)] == ["HSBC", "SPCX"]

    def test_held_omits_the_expired(self, quarantine: DataQuarantine) -> None:
        quarantine.record_failure("HSBC", HSBC, NOW)
        assert quarantine.held(NOW + timedelta(hours=2)) == ()

    def test_the_summary_says_what_and_when(self, quarantine: DataQuarantine) -> None:
        hold = quarantine.record_failure("HSBC", HSBC, NOW)
        summary = hold.summary(NOW + timedelta(minutes=10))

        assert "HSBC" in summary
        assert "20 min" in summary
        assert "bars missing" in summary

    def test_a_multiline_reason_is_trimmed(self, quarantine: DataQuarantine) -> None:
        """This ends up in a journal row and on the deck, not in a stack trace."""
        hold = quarantine.record_failure("HSBC", "first line\nsecond line\nthird", NOW)
        assert hold.reason == "first line"


class TestDisabled:
    def test_disabled_never_holds_anything(self) -> None:
        off = DataQuarantine(enabled=False)
        off.record_failure("HSBC", HSBC, NOW)
        assert off.hold_for("HSBC", NOW) is None

    def test_it_is_on_by_default_in_the_shipped_config(self) -> None:
        settings = load_settings(DEFAULT_CONFIG_PATH, env_overrides=False)
        config = settings.scanner.data_quarantine

        assert config.enabled is True
        assert config.max_minutes >= config.initial_minutes


class TestReopening:
    """The hold is a bet that nothing changed. A new bar settles that bet.

    The operator's question was the right one: if the exchange opens again
    tomorrow morning, why should the symbol sit out a clock set last night?
    It should not, and it does not — the venue's own first new bar releases it,
    whatever the deadline says.
    """

    MONDAY_CLOSE = datetime(2026, 8, 10, 16, 30, tzinfo=UTC)
    TUESDAY_OPEN = datetime(2026, 8, 11, 7, 0, tzinfo=UTC)

    def test_a_new_bar_releases_the_hold_before_the_deadline(
        self, quarantine: DataQuarantine
    ) -> None:
        quarantine.record_failure("HSBC", HSBC, NOW, latest_bar=self.MONDAY_CLOSE)

        one_minute_later = NOW + timedelta(minutes=1)
        assert quarantine.hold_for("HSBC", one_minute_later, self.MONDAY_CLOSE) is not None
        assert quarantine.hold_for("HSBC", one_minute_later, self.TUESDAY_OPEN) is None

    def test_the_same_bar_keeps_the_hold(self, quarantine: DataQuarantine) -> None:
        """Nothing has printed since it failed, so nothing has changed."""
        quarantine.record_failure("HSBC", HSBC, NOW, latest_bar=self.MONDAY_CLOSE)
        assert quarantine.hold_for("HSBC", NOW, self.MONDAY_CLOSE) is not None

    def test_an_older_bar_does_not_release_it(self, quarantine: DataQuarantine) -> None:
        """A scan reporting a staler bar is a worse view, not a better market."""
        quarantine.record_failure("HSBC", HSBC, NOW, latest_bar=self.MONDAY_CLOSE)
        stale = self.MONDAY_CLOSE - timedelta(hours=4)
        assert quarantine.hold_for("HSBC", NOW, stale) is not None

    def test_an_unknown_bar_falls_back_to_the_clock(self, quarantine: DataQuarantine) -> None:
        """No bar to compare is the case the deadline exists for."""
        quarantine.record_failure("HSBC", HSBC, NOW, latest_bar=self.MONDAY_CLOSE)

        assert quarantine.hold_for("HSBC", NOW, None) is not None
        assert quarantine.hold_for("HSBC", NOW + timedelta(minutes=31), None) is None

    def test_a_hold_taken_without_a_bar_still_expires(self, quarantine: DataQuarantine) -> None:
        quarantine.record_failure("SPCX", "too short", NOW)

        assert quarantine.hold_for("SPCX", NOW, self.TUESDAY_OPEN) is not None
        assert quarantine.hold_for("SPCX", NOW + timedelta(minutes=31)) is None

    def test_the_backstop_never_exceeds_four_hours(self, quarantine: DataQuarantine) -> None:
        """An overnight market is re-examined several times before it opens.

        Dropping from a fetch every cycle to one an hour already removes 98.3%
        of the waste; a whole day removes 99.93%. Those last seven-hundredths
        of a percentage point do not pay for a day of blindness.
        """
        for _ in range(10):
            hold = quarantine.record_failure("SPCX", "too short", NOW)
        assert hold.minutes_left(NOW) <= 240.0

    def test_the_summary_mentions_the_new_bar_route(self, quarantine: DataQuarantine) -> None:
        hold = quarantine.record_failure("HSBC", HSBC, NOW, latest_bar=self.MONDAY_CLOSE)
        assert "next new bar" in hold.summary(NOW)

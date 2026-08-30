"""The exit that closed 59 trades and won none of them.

`health_broken_at` was tried at three settings and wrote its own kill condition
into the config on the way past: "blijft `lift` negatief, dan is er geen
instelling die dit repareert."

    broken_at 0.65   13 trades   0 won   -EUR 20,33   lift -0.40R
    broken_at 0.75   12 trades   0 won   -EUR 36,82   lift -0.06R
    broken_at 0.85   34 trades   0 won   -EUR 45,89   lift -0.42R

Zero winners is not damning by itself -- the rule fires on trades whose thesis
broke, so losers are what it is supposed to be catching. What damns it is the
only column that asks the counterfactual:

    stepping in was   -0.26R
    doing nothing was +0.14R

Doing nothing was profitable. The condition has been met, so the close is off
while the reading stays on: stops still tighten, profits still get secured, and
every suppressed close is journalled with the R it would have taken, so the
counterfactual keeps accruing instead of going dark.
"""

from __future__ import annotations

from analysis.position_health import PositionHealth
from config.loader import DEFAULT_CONFIG_PATH, load_settings
from execution.manager import PositionManager
from tests.test_position_guard import ENTRY, STOP, BrokerStub, JournalStub, at, position


def _manager(broker: BrokerStub, journal: JournalStub, *, closes: bool):  # type: ignore[no-untyped-def]
    settings = load_settings(env_overrides=False)
    settings = settings.model_copy(
        update={
            "trade_management": settings.trade_management.model_copy(
                update={"health_exit_closes": closes}
            )
        }
    )
    return PositionManager(broker, journal, settings)  # type: ignore[arg-type]


def _act(broker: BrokerStub, manager, action: str = "exit"):  # type: ignore[no-untyped-def]
    reading = PositionHealth("broken", 1.0, action, (), "momentum turned")
    return manager._act_on_health(position(), reading, -0.5, ENTRY - STOP, broker.tick("EURUSD"))


class TestTheCloseIsOff:
    def test_a_reading_that_would_have_closed_does_not(self) -> None:
        """The same conditions that close with the switch on -- a far stop and
        a thin spread, so affordability is not what is stopping it."""
        broker, journal = BrokerStub(spread=0.2), JournalStub()
        at(broker, -0.25)

        event = _act(broker, _manager(broker, journal, closes=False))

        assert broker.closed == [], "nothing may be closed on a health reading"
        assert event is not None and event.action == "HEALTH_EXIT_SUPPRESSED"

    def test_the_switch_really_is_what_stops_it(self) -> None:
        """Without this the test above passes for any reason at all -- an
        unaffordable spread, a stub that never closes, a reading the engine
        ignored. The identical call with the switch on must close."""
        broker, journal = BrokerStub(spread=0.2), JournalStub()
        at(broker, -0.25)

        event = _act(broker, _manager(broker, journal, closes=True))

        assert event is not None and event.action == "HEALTH_EXIT"
        assert broker.closed == [(555, None)]

    def test_the_suppressed_event_carries_no_close(self) -> None:
        """`_record_management` writes a close row when exit_price AND
        pnl_money are both set. A suppressed exit that filled them in would
        book a closed trade that is still open at the broker."""
        broker, journal = BrokerStub(spread=0.2), JournalStub()
        at(broker, -0.25)

        event = _act(broker, _manager(broker, journal, closes=False))

        assert event.exit_price is None
        assert event.pnl_money is None
        assert event.r_at_action == -0.5, "the R it would have closed at is the measurement"


class TestTheReadingItselfStaysOn:
    def test_securing_a_profit_is_untouched(self) -> None:
        """Only the loss-cutting close was measured as harmful. Banking a
        winner as it turns is a different action with a different record, and
        switching it off here would be collateral damage nobody measured."""
        broker, journal = BrokerStub(spread=0.2), JournalStub()
        at(broker, -0.25)

        event = _act(broker, _manager(broker, journal, closes=False), action="secure")

        assert event is not None and event.action == "HEALTH_SECURE"
        assert broker.closed == [(555, None)]

    def test_a_hold_is_still_a_hold(self) -> None:
        broker, journal = BrokerStub(spread=0.2), JournalStub()
        at(broker, -0.25)

        assert _act(broker, _manager(broker, journal, closes=False), action="hold") is None


class TestTheLiveAccountHasItOff:
    def test_the_shipped_overlay_switches_it_off(self) -> None:
        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

        assert settings.trade_management.health_exit_closes is False

    def test_the_base_config_leaves_it_on(self) -> None:
        """The measurement is this account's, on this broker, at this size.
        Turning it off for everyone would be generalising fifty-nine trades."""
        assert load_settings(env_overrides=False).trade_management.health_exit_closes is True

"""The fast layer: watching open positions between cycles.

Management used to run once per full cycle. A cycle scans the catalogue and can
take most of a minute, so a trade could run to 1.6R and hand every bit of it
back between two consecutive looks — the rules were correct and simply were not
being asked often enough.

Three things make the one-second cadence work, and each is asserted here: the
ATR is cached so the pass stays cheap, the peak R is persisted so a restart
cannot forget the trade was ahead, and the guard never calls the adviser.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta

import pytest

from advisory.providers import Supervision
from analysis.position_health import PositionHealth
from config.loader import load_settings
from core.instrument import AssetClass
from core.types import Direction, Position, Tick, Timeframe
from execution.manager import ATR_CACHE_SECONDS, ManagementEvent, PositionManager

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
ENTRY = 100.0
STOP = 98.0  # risk of 2.0 price units, so 1R = 2.0


# ----------------------------------------------------------------- doubles ---


@dataclass
class OrderResult:
    ok: bool = True
    filled_price: float | None = 100.0
    filled_volume: float | None = 0.01
    retcode_name: str = "DONE"


@dataclass
class BrokerStub:
    """Just enough broker to drive `manage`, and it counts its own calls.

    The call counts are the point of half these tests: the guard's whole
    licence to run every second is that a pass is cheap.
    """

    price: float = ENTRY
    atr: float = 1.0
    spread: float = 0.0
    #: Per-bar drift in the returned series. Zero is featureless, which reads as
    #: `healthy`; a negative value makes the health engine see a market falling
    #: away under a long, which is what turns the give-back from "hold" into
    #: "bank it".
    drift: float = 0.0
    asset_class: AssetClass = AssetClass.FOREX
    rate_calls: dict[int, int] = field(default_factory=dict)
    closed: list[tuple[int, float | None]] = field(default_factory=list)
    modified: list[float] = field(default_factory=list)

    @property
    def atr_calls(self) -> int:
        """Only the H1 fetches. The health readers pull M1 and M5 on their own
        much shorter TTL, so a single total would conflate two caches."""
        return self.rate_calls.get(Timeframe.H1.mt5_value, 0)

    def tick(self, symbol: str) -> Tick:
        # The whole spread sits on the ask, so `price` for the long these tests
        # use is `self.price` exactly. Splitting it symmetrically would shift R
        # by half a spread and quietly retune every threshold asserted below.
        return Tick(symbol=symbol, bid=self.price, ask=self.price + self.spread, time=NOW)

    def spec(self, _symbol: str):  # type: ignore[no-untyped-def]
        klass = self.asset_class

        class Spec:
            volume_min = 0.01
            asset_class = klass

            @staticmethod
            def normalize_price(price: float) -> float:
                return round(price, 5)

            @staticmethod
            def money_per_lot(distance: float) -> float:
                """A round 1.0 per price unit, so a 2.0 stop is 2.00 a lot and
                the commission-in-R arithmetic below is readable by hand."""
                return abs(distance)

            @staticmethod
            def round_volume_down(volume: float) -> float:
                return round(volume, 2)

            @staticmethod
            def pips_to_price(pips: float) -> float:
                """One price unit per pip, matching `money_per_lot` above so
                the cost arithmetic stays readable by hand."""
                return pips

        return Spec()

    def copy_rates(self, _symbol, timeframe, count):  # type: ignore[no-untyped-def]
        self.rate_calls[timeframe] = self.rate_calls.get(timeframe, 0) + 1
        # A flat series whose true range is exactly `atr` on every bar, and
        # which is deliberately featureless: no swings, no drift, so the health
        # readers stay silent and these tests keep measuring what they name.
        base = int(datetime(2026, 8, 4, 0, 0, tzinfo=UTC).timestamp())
        return [
            {
                "time": base + index * 60,
                "high": 100.0 + self.atr + index * self.drift,
                "low": 100.0 + index * self.drift,
                "close": 100.0 + index * self.drift,
                "open": 100.0 + index * self.drift,
                "tick_volume": 1,
                "spread": 0,
                "real_volume": 0,
            }
            for index in range(count)
        ]

    def close_position(self, position: Position, volume: float | None = None) -> OrderResult:
        self.closed.append((position.ticket, volume))
        return OrderResult(filled_price=self.price, filled_volume=volume or position.volume)

    def closed_position(self, _ticket: int):  # type: ignore[no-untyped-def]
        """Deal history has not caught up yet, which is the ordinary case.

        The manager already handles it — the event is recorded as *_SENT and
        reconciliation fills in the settled price afterwards. Returning None
        keeps the stub honest about that rather than inventing a record the
        broker has not produced.
        """
        return

    def modify_stops(self, _position, sl, tp) -> OrderResult:  # type: ignore[no-untyped-def]
        del tp
        self.modified.append(sl)
        return OrderResult(filled_price=sl)


@dataclass
class JournalStub:
    """Holds the excursion ratchet the way the real journal does."""

    peak_r: float = 0.0
    trough_r: float = 0.0
    partial_taken: bool = False
    #: What 1R is worth in account currency. NOT NULL in the real table, and
    #: the denominator of every R the account reports. Generous enough here
    #: that the banking rule's money cap does not bind by accident — the tests
    #: that care about the cap set it themselves.
    risk_money: float = 10.0

    def open_trade_by_ticket(self, ticket: int):  # type: ignore[no-untyped-def]
        return {
            "id": 1,
            "ticket": ticket,
            "sl": STOP,
            "volume": 0.02,
            "mfe_r": self.peak_r,
            "risk_money": self.risk_money,
        }

    def management_action_exists(self, _ticket, _actions) -> bool:  # type: ignore[no-untyped-def]
        return self.partial_taken

    def update_excursions(self, _trade_id, *, mae_r: float, mfe_r: float) -> None:
        self.peak_r = max(self.peak_r, mfe_r)
        self.trough_r = min(self.trough_r, mae_r)

    def open_trades(self):  # type: ignore[no-untyped-def]
        return []


def position(volume: float = 0.02, opened_at: datetime = NOW) -> Position:
    return Position(
        ticket=555,
        symbol="EURUSD",
        direction=Direction.LONG,
        volume=volume,
        price_open=ENTRY,
        sl=STOP,
        tp=ENTRY + 10.0,
        profit=1.0,
        swap=0.0,
        opened_at=opened_at,
    )


def manager_for(broker: BrokerStub, journal: JournalStub, **overrides) -> PositionManager:  # type: ignore[no-untyped-def]
    settings = load_settings(env_overrides=False)
    if overrides:
        management = settings.trade_management.model_copy(update=overrides)
        settings = settings.model_copy(update={"trade_management": management})
    return PositionManager(broker, journal, settings)  # type: ignore[arg-type]


def test_a_rejected_emergency_close_remains_explicitly_open() -> None:
    broker, journal = BrokerStub(), JournalStub()
    broker.close_position = lambda _p, volume=None: OrderResult(  # type: ignore[assignment]
        ok=False, filled_price=None, retcode_name="MARKET_CLOSED"
    )
    manager = manager_for(broker, journal)

    events = manager.reconcile([replace(position(), sl=0.0)])

    assert [event.action for event in events] == ["EMERGENCY_CLOSE_REJECTED"]
    assert events[0].exit_price is None


def at(broker: BrokerStub, r: float) -> None:
    """Move the market to a given R for the fixture's position."""
    broker.price = ENTRY + r * (ENTRY - STOP)


# ------------------------------------------------------------- excursions ---


def test_the_peak_is_recorded_as_the_trade_runs() -> None:
    """`mfe_r` and `mae_r` existed as columns and nothing ever wrote them, so
    every postmortem reported both as unknown."""
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal)

    at(broker, 1.2)
    manager.manage([position()], NOW)
    at(broker, 0.9)
    manager.manage([position()], NOW)

    assert journal.peak_r == pytest.approx(1.2)


def test_the_trough_is_recorded_too() -> None:
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal)

    at(broker, -0.4)
    manager.manage([position()], NOW)

    assert journal.trough_r == pytest.approx(-0.4)


def test_the_peak_only_ratchets_upward() -> None:
    """A late retrace must not erase the fact that the trade was once ahead —
    that memory is the entire input to the give-back rule."""
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=0.0)

    at(broker, 2.0)
    manager.manage([position()], NOW)
    at(broker, 0.1)
    manager.manage([position()], NOW)

    assert journal.peak_r == pytest.approx(2.0)


# ---------------------------------------------------------------- giveback ---


#: Old enough for the health engine to have an opinion. Below `MIN_AGE_MINUTES`
#: every reading is "healthy" by definition, which would make the give-back
#: tests below pass for the wrong reason.
def running(minutes: int = 60) -> Position:
    return position(opened_at=NOW - timedelta(minutes=minutes))


def test_a_fading_trade_banks_what_is_left() -> None:
    """The AUDJPY case. It peaked around 0.57R — a euro against 1.77 of risk —
    and ended at -0.28R, because both this rule and break-even armed at 1.0R
    and neither ever engaged."""
    broker, journal = BrokerStub(drift=-0.4), JournalStub()
    manager = manager_for(broker, journal)

    at(broker, 0.6)
    manager.manage([running()], NOW)
    at(broker, 0.25)  # 58% of the gain handed back, and the drift is against us
    events = manager.manage([running()], NOW)

    assert [event.action for event in events] == ["GIVEBACK_EXIT"]
    assert broker.closed == [(555, None)], "the whole position, not a partial"


def test_a_gain_below_the_arming_level_is_not_protected_yet() -> None:
    """Some floor is unavoidable, or every trade is closed on its first tick of
    profit."""
    broker, journal = BrokerStub(drift=-0.4), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=0.5)

    at(broker, 0.3)
    manager.manage([running()], NOW)
    at(broker, 0.05)
    events = manager.manage([running()], NOW)

    assert not any(event.action == "GIVEBACK_EXIT" for event in events)


def test_a_healthy_pullback_is_held_rather_than_banked() -> None:
    """The "it goes further" case, and the reason this is a judgement rather
    than a tripwire. Closing a live trade on a wobble pays the spread to
    abandon a move that is still working."""
    broker, journal = BrokerStub(), JournalStub()  # featureless: reads healthy
    manager = manager_for(broker, journal)

    at(broker, 1.0)
    manager.manage([running()], NOW)
    at(broker, 0.45)  # 55% back, but nothing is wrong with the move
    events = manager.manage([running()], NOW)

    assert not any(event.action == "GIVEBACK_EXIT" for event in events)
    assert broker.closed == []


def test_conviction_does_not_survive_handing_back_nearly_everything() -> None:
    """The part that is not a judgement. However intact the read, a gain that
    has almost entirely gone is not a position worth holding — otherwise a
    permanently healthy reading rides every winner back to entry."""
    broker, journal = BrokerStub(), JournalStub()  # healthy throughout
    manager = manager_for(broker, journal, giveback_hard_fraction=0.8)

    at(broker, 1.0)
    manager.manage([running()], NOW)
    at(broker, 0.1)  # 90% gone
    events = manager.manage([running()], NOW)

    assert [event.action for event in events] == ["GIVEBACK_EXIT"]
    assert "too much to hold" in events[0].detail


def test_the_reason_records_which_way_it_was_decided() -> None:
    """The journal line is the only record of *why*, and "banked because the
    move stopped working" and "banked because too much was gone" are different
    lessons to read back."""
    broker, journal = BrokerStub(drift=-0.4), JournalStub()
    manager = manager_for(broker, journal)

    at(broker, 0.8)
    manager.manage([running()], NOW)
    at(broker, 0.3)
    (event,) = [e for e in manager.manage([running()], NOW) if e.action == "GIVEBACK_EXIT"]

    assert "gave back" in event.detail
    assert "read:" in event.detail


def test_a_new_high_is_never_a_giveback() -> None:
    """At a fresh peak there is nothing handed back, and a rule that divided by
    a peak it had just set would read 0% as 100%."""
    broker, journal = BrokerStub(drift=-0.4), JournalStub()
    manager = manager_for(broker, journal)

    at(broker, 2.5)
    events = manager.manage([running()], NOW)

    assert not any(event.action == "GIVEBACK_EXIT" for event in events)


def test_the_peak_survives_a_restart() -> None:
    """A new manager reads the peak from the journal, so a crash mid-trade
    cannot hand the position a clean slate and let it give back everything."""
    broker, journal = BrokerStub(drift=-0.4), JournalStub()
    at(broker, 1.0)
    manager_for(broker, journal).manage([running()], NOW)

    reborn = manager_for(BrokerStub(drift=-0.4, price=ENTRY + 0.2 * (ENTRY - STOP)), journal)
    events = reborn.manage([running()], NOW)

    assert journal.peak_r == pytest.approx(1.0)
    assert [event.action for event in events] == ["GIVEBACK_EXIT"]


def test_zero_disables_the_rule() -> None:
    broker, journal = BrokerStub(drift=-0.4), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=0.0)

    at(broker, 1.0)
    manager.manage([running()], NOW)
    at(broker, 0.0)
    events = manager.manage([running()], NOW)

    assert not any(event.action == "GIVEBACK_EXIT" for event in events)


def test_a_refused_close_is_not_reported_as_an_exit() -> None:
    """Recording a close the broker rejected would leave the journal believing
    the position is flat while the money is still at risk."""
    broker, journal = BrokerStub(drift=-0.4), JournalStub()
    broker.close_position = lambda _p, volume=None: OrderResult(ok=False, filled_price=None)  # type: ignore[assignment]
    manager = manager_for(broker, journal)

    at(broker, 1.0)
    manager.manage([running()], NOW)
    at(broker, 0.2)
    events = manager.manage([running()], NOW)

    assert not any(event.action == "GIVEBACK_EXIT" for event in events)


# --------------------------------------------------------------- the cost ---


def test_the_atr_is_not_refetched_on_every_pass() -> None:
    """Sixty round-trips an hour to watch an average of fourteen hourly bars
    move in the fourth decimal is what makes a one-second cadence unaffordable."""
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=0.0)

    at(broker, 0.5)
    for _ in range(20):
        manager.manage([position()], NOW)

    assert broker.atr_calls == 1


def test_the_cache_expires() -> None:
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=0.0)

    at(broker, 0.5)
    manager.manage([position()], NOW)
    # Reach into the deadline rather than sleeping a minute in a unit test.
    manager._atr_cache = {
        key: (deadline - ATR_CACHE_SECONDS - 1.0, value)
        for key, (deadline, value) in manager._atr_cache.items()
    }
    manager.manage([position()], NOW)

    assert broker.atr_calls == 2


def test_the_cache_is_keyed_per_symbol() -> None:
    """One cache entry for everything would price a gold position off an
    EURUSD ATR — a stop dozens of times too tight or too wide."""
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=0.0)

    at(broker, 0.5)
    manager.manage([position()], NOW)
    manager.manage([replace(position(), symbol="XAUUSD")], NOW)

    assert broker.atr_calls == 2


def test_the_cache_uses_a_monotonic_deadline() -> None:
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=0.0)

    at(broker, 0.5)
    manager.manage([position()], NOW)
    ((deadline, _),) = manager._atr_cache.values()

    assert time.monotonic() < deadline <= time.monotonic() + ATR_CACHE_SECONDS


# ------------------------------------------------------------- ordering ---


def test_the_giveback_pre_empts_the_partial_and_the_trail() -> None:
    """Half-closing a position we have already decided to leave would pay two
    spreads to exit and carry the rest through the reversal."""
    broker, journal = BrokerStub(drift=-0.4), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=1.0, giveback_fraction=0.4)

    at(broker, 2.0)
    manager.manage([running()], NOW)  # break-even moves the stop here
    before = len(broker.modified)
    at(broker, 1.1)  # above partial_close_at_r, but 45% given back
    events = manager.manage([running()], NOW)

    assert [event.action for event in events] == ["GIVEBACK_EXIT"]
    assert broker.closed == [(555, None)], "closed whole, not halved"
    assert len(broker.modified) == before, "no stop moves on the way out"


def test_a_stalled_trade_still_times_out() -> None:
    """The give-back rule must not shadow the time exit: a trade that never
    ran has no peak, so the earlier `continue` cannot swallow it."""
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, time_exit_hours=4.0)

    at(broker, 0.05)
    events = manager.manage([position(opened_at=NOW - timedelta(hours=9))], NOW)

    assert [event.action for event in events] == ["TIME_EXIT"]


def test_a_rejected_time_exit_is_never_reported_as_closed() -> None:
    broker, journal = BrokerStub(), JournalStub()
    broker.close_position = lambda _p, volume=None: OrderResult(  # type: ignore[assignment]
        ok=False, filled_price=None, retcode_name="MARKET_CLOSED"
    )
    manager = manager_for(broker, journal, time_exit_hours=4.0)

    at(broker, 0.05)
    events = manager.manage([position(opened_at=NOW - timedelta(hours=9))], NOW)

    assert [event.action for event in events] == ["TIME_EXIT_REJECTED"]
    assert events[0].exit_price is None
    assert "still open" in events[0].detail


def test_management_events_are_still_typed() -> None:
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal)
    at(broker, 1.6)
    manager.manage([position()], NOW)
    at(broker, 0.1)
    (event,) = manager.manage([position()], NOW)
    assert isinstance(event, ManagementEvent)
    assert event.r_at_action == pytest.approx(0.1)


#: The health read a stalled position is handed in these tests.
#:
#: `watch` and not `healthy` on purpose: the stall rule now asks the readers
#: before it closes anything, and a healthy read buys the trade more time. These
#: tests are about the stall MECHANISM — the clock, the new-high reset, the
#: per-position timing — so they hand it the case where the clock is allowed to
#: stand. `TestPeakStallAsksTheReadFirst` covers the other branch.
STALLED = PositionHealth("watch", 0.30, "hold", (), "the move has gone quiet")


# ------------------------------------------------------- evening wind-down ---

EVENING = datetime(2026, 8, 4, 20, 30, tzinfo=UTC)  # 22:30 in Amsterdam
AFTERNOON = datetime(2026, 8, 4, 14, 0, tzinfo=UTC)


def test_positions_are_flattened_before_the_evening_spread() -> None:
    """The rollover block stopped us opening into the worst half hour and said
    nothing about what was already on, so a position entered in a 1-pip market
    was carried into a 6-pip one and charged the difference on the way out.
    """
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal)

    at(broker, 0.3)
    events = manager.manage([position()], EVENING)

    assert [event.action for event in events] == ["EVENING_FLAT"]
    assert broker.closed == [(555, None)]


def test_the_afternoon_is_left_alone() -> None:
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal)

    at(broker, 0.3)
    events = manager.manage([position()], AFTERNOON)

    assert not any(event.action == "EVENING_FLAT" for event in events)
    assert broker.closed == []


def test_a_winner_is_flattened_too() -> None:
    """ "Let this one run, it is almost at target" is the reasoning that produces
    the loss: the widening spread moves the market away from the target and
    toward the stop at the same time."""
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal)

    at(broker, 1.4)
    events = manager.manage([position()], EVENING)

    assert [event.action for event in events] == ["EVENING_FLAT"]


def test_the_flatten_pre_empts_every_other_rule() -> None:
    """Everything else assumes we intend to still be in the trade."""
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=1.0, giveback_fraction=0.5)

    at(broker, 2.0)
    manager.manage([position()], AFTERNOON)  # records the peak
    before = len(broker.modified)
    at(broker, 0.5)  # would be a give-back exit in the afternoon
    events = manager.manage([position()], EVENING)

    assert [event.action for event in events] == ["EVENING_FLAT"]
    assert len(broker.modified) == before, "no stop moves on the way out"


def test_switching_it_off_leaves_positions_open() -> None:
    broker, journal = BrokerStub(), JournalStub()
    settings = load_settings(env_overrides=False)
    session = settings.filters.session.model_copy(update={"evening_flat_from": None})
    filters = settings.filters.model_copy(update={"session": session})
    manager = PositionManager(broker, journal, settings.model_copy(update={"filters": filters}))  # type: ignore[arg-type]

    at(broker, 0.3)
    events = manager.manage([position()], EVENING)

    assert not any(event.action == "EVENING_FLAT" for event in events)


def test_a_refused_close_is_not_recorded_as_flat() -> None:
    """Recording a close the broker rejected would leave the journal believing
    the position is gone while the money is still at risk."""
    broker, journal = BrokerStub(), JournalStub()
    broker.close_position = lambda _p, volume=None: OrderResult(ok=False, filled_price=None)  # type: ignore[assignment]
    manager = manager_for(broker, journal)

    at(broker, 0.3)
    events = manager.manage([position()], EVENING)

    assert not any(event.action == "EVENING_FLAT" for event in events)


def test_a_continuous_market_is_not_flattened() -> None:
    """Crypto has no FX rollover and no reason to be closed at 22:15 Amsterdam
    time. Flattening it would book a spread cost for a calendar it does not
    follow."""
    broker = BrokerStub(asset_class=AssetClass.CRYPTO)
    manager = manager_for(broker, JournalStub())

    at(broker, 0.3)
    events = manager.manage([replace(position(), symbol="BTCUSD")], EVENING)

    assert not any(event.action == "EVENING_FLAT" for event in events)
    assert broker.closed == []


def test_a_stock_is_not_flattened_by_the_fx_evening_rule() -> None:
    broker = BrokerStub(asset_class=AssetClass.STOCK)
    manager = manager_for(broker, JournalStub())

    at(broker, 0.3)
    events = manager.manage([replace(position(), symbol="AAPL")], EVENING)

    assert not any(event.action == "EVENING_FLAT" for event in events)
    # A stock may still be closed by its own cash-session runway rule. This
    # assertion is specifically about not applying the FX rollover rule to it.


def test_health_uses_confirmed_closed_bars_only() -> None:
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal)

    frame = manager._bars("EURUSD", Timeframe.M1, 40)

    assert frame is not None
    assert len(frame) == 40
    # The broker returned 41 rows. Bar zero/current is represented by the final
    # row in this fake; the manager must remove it before health analysis.
    base = datetime(2026, 8, 4, 0, 0, tzinfo=UTC)
    assert frame.index[-1].to_pydatetime() == base + timedelta(minutes=39)


def test_low_confidence_supervision_is_a_hard_hold() -> None:
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal)

    event = manager.apply_supervision(
        position(),
        Supervision("close", "weak hunch", confidence=0.20, provider="test"),
    )

    assert event is not None
    assert event.action == "AI_SUPERVISION_UNDER_THRESHOLD"
    assert broker.closed == []


def test_the_audjpy_case_end_to_end() -> None:
    """The trade that prompted all of this, as a regression.

    Peaked at 0.57R — EUR 1.00 against EUR 1.77 of risk — then reversed to
    -0.28R. Under the old thresholds nothing engaged: the give-back armed at
    1.0R and so did break-even, both above anything that trade would reach, and
    a euro of profit turned into fifty cents of loss with the machine watching.
    """
    broker, journal = BrokerStub(drift=-0.4), JournalStub()
    manager = manager_for(broker, journal)

    at(broker, 0.57)  # the peak, exactly as it happened
    manager.manage([running()], NOW)
    assert journal.peak_r == pytest.approx(0.57, abs=0.01)

    at(broker, 0.20)  # 65% of the gain gone, market moving against us
    events = manager.manage([running()], NOW)

    assert [event.action for event in events] == ["GIVEBACK_EXIT"]
    assert events[0].r_at_action is not None
    assert events[0].r_at_action > 0, "banked in profit, not at a loss"


def test_the_old_thresholds_would_have_let_it_run_to_a_loss() -> None:
    """The counterfactual, so the regression above cannot pass for the wrong
    reason. At the settings this account was running, nothing fires."""
    broker, journal = BrokerStub(drift=-0.4), JournalStub()
    manager = manager_for(broker, journal, giveback_arm_r=1.0, break_even_at_r=1.0)

    at(broker, 0.57)
    manager.manage([running()], NOW)
    at(broker, 0.20)
    events = manager.manage([running()], NOW)

    assert not any(event.action == "GIVEBACK_EXIT" for event in events)
    assert broker.closed == []


class TestAgedButProfitable:
    """The gap between the time exit and the give-back.

    `time_exit_min_abs_r` is 0.3 and the give-back arms at 0.5R, so a position
    on +0.4R after a day and a half belonged to neither rule. It stayed open
    indefinitely, paying swap for one of two slots on a small account while
    doing nothing with it.

    Judged on the *peak*, not the current price. A trade that ran to 2R and
    came back to 0.4 has demonstrated something and belongs to the give-back;
    one whose best moment in a whole day was 0.4R has demonstrated the
    opposite.
    """

    @staticmethod
    def verdict(
        age_hours: float,
        r_now: float,
        peak_r: float,
        *,
        deadline: float | None = 24.0,
        stale_peak: float = 1.0,
    ) -> str | None:
        from config.schema import TradeManagementConfig
        from execution.manager import PositionManager

        config = TradeManagementConfig(time_exit_stale_peak_r=stale_peak)
        return PositionManager._time_exit_verdict(config, age_hours, deadline, r_now, peak_r)

    def test_a_young_trade_is_never_closed_on_the_clock(self) -> None:
        assert self.verdict(2.0, 0.05, 0.05) is None
        assert self.verdict(23.9, 0.40, 0.40) is None

    def test_the_original_rule_still_closes_a_flat_trade(self) -> None:
        assert self.verdict(25.0, 0.05, 0.10) == "went nowhere"

    def test_the_gap_case_is_now_banked(self) -> None:
        """+0.4R after 25 hours, best-ever 0.4R. Take it and free the slot."""
        reason = self.verdict(25.0, 0.40, 0.40)
        assert reason is not None
        assert "never got going" in reason

    def test_a_trade_that_ran_and_came_back_is_left_to_the_giveback(self) -> None:
        """Same current R, a completely different trade.

        Peaking at 2R is evidence the thesis worked. That position is the
        give-back rule's to judge, and closing it here would pre-empt a rule
        that reads whether the move is still working.
        """
        assert self.verdict(25.0, 0.40, 2.00) is None

    def test_a_loser_is_never_realised_by_this_rule(self) -> None:
        """It banks a modest profit; it does not cut a trade the old rule held.

        -0.8R past the deadline stays open exactly as before — that position
        belongs to its stop, or to the health reader, not to a clock.
        """
        assert self.verdict(25.0, -0.80, 0.10) is None

    def test_a_strong_winner_past_the_deadline_keeps_running(self) -> None:
        assert self.verdict(50.0, 1.80, 2.10) is None

    def test_the_drawdown_posture_shortens_the_deadline(self) -> None:
        """Patience below 1.0 brings the same judgement forward."""
        assert self.verdict(13.0, 0.40, 0.40, deadline=24.0) is None
        assert self.verdict(13.0, 0.40, 0.40, deadline=12.0) is not None

    def test_the_threshold_is_configurable(self) -> None:
        assert self.verdict(25.0, 0.40, 1.50, stale_peak=1.0) is None
        assert self.verdict(25.0, 0.40, 1.50, stale_peak=2.0) is not None

    def test_no_deadline_configured_means_no_time_exit_at_all(self) -> None:
        assert self.verdict(500.0, 0.40, 0.40, deadline=None) is None


class TestProfitLock:
    """Break-even protects the entry; this protects the move.

    Between `break_even_at_r` (0.6) and `partial_close_at_r` (1.5) nothing
    touched the stop, so a trade could run to 1.4R over several hours and hand
    every cent of it back to a stop still sitting at entry — right for hours,
    paid nothing.

    It overlaps the give-back exit on purpose. The give-back lives inside our
    own loop; a stop lives at the broker and survives a VPS reboot, a dropped
    terminal, and this process dying at three in the morning. This account is
    meant to be left alone overnight.
    """

    @staticmethod
    def lock(
        peak_r: float,
        r_now: float,
        current_sl: float,
        *,
        entry: float = 1.0800,
        equity: float = 0.0,
        risk_money: float = 0.0,
    ):
        """Return (event, requested_sl) from one profit-lock evaluation."""
        from types import SimpleNamespace

        from config.schema import TradeManagementConfig
        from core.types import Direction, Position
        from execution.manager import PositionManager

        sent: dict[str, float] = {}

        def modify_stops(position, sl, tp):  # type: ignore[no-untyped-def]
            sent["sl"] = sl
            return SimpleNamespace(ok=True)

        manager = PositionManager.__new__(PositionManager)
        manager.settings = SimpleNamespace(trade_management=TradeManagementConfig())
        # Set explicitly because these fixtures bypass __init__. The real
        # object always has it; the account-relative profit protection reads
        # it, and zero means "no equity known", which switches that route off
        # and leaves the R floors as the only gate — the behaviour these
        # tests were written against.
        manager.equity = equity
        manager.broker = SimpleNamespace(
            modify_stops=modify_stops,
            spec=lambda symbol: SimpleNamespace(normalize_price=lambda price: round(price, 5)),
        )
        position = Position(
            ticket=1,
            symbol="EURUSD",
            direction=Direction.LONG,
            volume=0.01,
            price_open=entry,
            sl=current_sl,
            tp=entry + 0.0040,
            profit=1.0,
            swap=0.0,
            opened_at=datetime(2026, 8, 6, 10, 0, tzinfo=UTC),
            magic=1,
        )
        event = manager._profit_lock(position, r_now, peak_r, risk=0.0010, risk_money=risk_money)
        return event, sent.get("sl")

    def test_below_the_arming_peak_nothing_moves(self) -> None:
        event, sl = self.lock(peak_r=0.6, r_now=0.6, current_sl=1.0800)
        assert event is None and sl is None

    def test_at_the_arming_peak_it_secures_half(self) -> None:
        """Peak 0.7R with a 10-pip risk puts the stop 3.5 pips above entry."""
        event, sl = self.lock(peak_r=0.7, r_now=0.7, current_sl=1.0800)
        assert event is not None
        assert event.action == "PROFIT_LOCK"
        assert sl == pytest.approx(1.08035, abs=1e-6)

    def test_the_nzdcad_trade_would_have_kept_most_of_its_gain(self) -> None:
        """The live case this threshold was moved for.

        NZDCAD long peaked at 0.92R — EUR 1.60 on an EUR 87 account — and
        stopped out at 0.13R for 22 cents. The lock armed at 1.0R and so never
        fired, the give-back allowed an 80% drain while the health read stayed
        healthy, and the break-even stop sat below both and took the trade.

        Armed at 0.7R the lock secures 0.46R, which is EUR 0.80 — still not the
        EUR 1.60 the peak was worth, but nearly four times what the crudest
        rule in the file actually delivered.
        """
        event, sl = self.lock(peak_r=0.92, r_now=0.92, current_sl=1.08010)
        assert event is not None
        secured_r = (sl - 1.0800) / 0.0010
        assert secured_r == pytest.approx(0.46, abs=0.01)

    def test_it_ratchets_off_the_peak_not_the_current_price(self) -> None:
        """A trade that reached 2R keeps its 1R stop after pulling back.

        This is the whole reason it reads the peak: measured from the current
        price, a pullback would walk the stop backwards, which is the one thing
        a stop must never do.
        """
        event, sl = self.lock(peak_r=2.0, r_now=1.2, current_sl=1.0805)
        assert event is not None
        assert sl == pytest.approx(1.08100, abs=1e-6)

    def test_a_stop_already_better_is_left_alone(self) -> None:
        """The ATR trail may already have moved past it. No retreat."""
        event, sl = self.lock(peak_r=2.0, r_now=1.9, current_sl=1.0815)
        assert event is None and sl is None

    def test_it_never_strangles_the_winner(self) -> None:
        """Half the peak, not all of it — the market must be allowed to breathe.

        A stop tucked under the high is taken out by ordinary noise, and this
        one always leaves at least half the excursion as room.
        """
        _, sl = self.lock(peak_r=3.0, r_now=3.0, current_sl=1.0800)
        assert sl is not None
        high_water = 1.0800 + 3.0 * 0.0010
        assert sl < high_water
        assert (high_water - sl) == pytest.approx(0.0015, abs=1e-6)


class TestStopsOnlyMoveWhenItMatters:
    """Every rule that touches a stop ends in `continue`.

    That makes a pointless stop move expensive in a way that is invisible when
    you read the rules one at a time: it does not merely cost a broker
    round-trip, it costs the position its turn at every rule below. Break-even
    recomputes from a live ATR, so before `_STOP_IMPROVEMENT_R` a fractionally
    rising ATR nudged the target up and re-fired the rule on every pass — and
    the profit lock, two rules further down, never ran at all.

    Found by replaying the real rules over bar history rather than by reading
    them, which is exactly the class of bug a replay is for.
    """

    @staticmethod
    def at_break_even(broker: BrokerStub) -> Position:
        """The position as it stands after break-even has moved the stop."""
        return replace(position(), sl=broker.modified[-1])

    def test_break_even_fires_once_and_then_lets_the_next_rule_through(self) -> None:
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)

        at(broker, 0.9)
        first = manager.manage([position()], NOW)
        second = manager.manage([self.at_break_even(broker)], NOW)

        assert [event.action for event in first] == ["BREAK_EVEN"]
        # Not BREAK_EVEN a second time: half of the 0.9R peak, secured.
        assert [event.action for event in second] == ["PROFIT_LOCK"]

    def test_a_stop_move_worth_nothing_is_not_made(self) -> None:
        """A hundredth of an R is noise in the ATR, not a decision."""
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)
        risk = ENTRY - STOP

        assert not manager._worth_moving(position(), STOP + risk * 0.005, risk)
        assert manager._worth_moving(position(), STOP + risk * 0.5, risk)

    def test_a_stop_is_never_walked_backwards(self) -> None:
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)
        risk = ENTRY - STOP

        assert not manager._worth_moving(replace(position(), sl=ENTRY), ENTRY - risk * 0.5, risk)

    def test_a_zero_risk_position_moves_nothing(self) -> None:
        """Guarding the division. A journal that records the stop at entry has
        no R to measure a stop move against, and the rule that divides by it
        would take the whole guard down."""
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)

        assert not manager._worth_moving(position(), ENTRY + 1.0, 0.0)


class TestPeakStall:
    """Leaving near the high, instead of confirming the retrace afterwards.

    Every other exit in the manager measures how much has been *given back*,
    so every one of them can only act after the money has gone. This measures
    what a person watches: the trade stopped making new highs.

    NZDCAD is the case. Peak 0.92R — EUR 1.60 on an EUR 87 account — closed at
    0.13R for 22 cents. The operator's own read, forty minutes before the
    system's, was "it has been sitting at the same high for a while, take it".
    """

    @staticmethod
    def manager():  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        from config.schema import TradeManagementConfig
        from execution.manager import PositionManager

        closed: list[float] = []

        def close_position(position, volume=None):  # type: ignore[no-untyped-def]
            closed.append(position.profit)
            return SimpleNamespace(ok=True, filled_price=1.0900, filled_volume=position.volume)

        instance = PositionManager.__new__(PositionManager)
        instance.settings = SimpleNamespace(trade_management=TradeManagementConfig())
        # See the note in TestProfitLock.lock: these fixtures bypass __init__.
        instance.equity = 0.0
        instance.broker = SimpleNamespace(close_position=close_position)
        instance._peak_seen = {}
        instance.closed = closed
        return instance

    @staticmethod
    def position(ticket: int = 1):  # type: ignore[no-untyped-def]
        from core.types import Direction, Position

        return Position(
            ticket=ticket,
            symbol="NZDCAD",
            direction=Direction.LONG,
            volume=0.01,
            price_open=0.8400,
            sl=0.8390,
            tp=0.8440,
            profit=1.60,
            swap=0.0,
            opened_at=datetime(2026, 8, 6, 8, 24, tzinfo=UTC),
            magic=1,
        )

    def at(self, minutes: float) -> datetime:
        return datetime(2026, 8, 6, 9, 0, tzinfo=UTC) + timedelta(minutes=minutes)

    def test_the_nzdcad_trade_is_banked_near_its_high(self) -> None:
        """0.92R, standing still for seven minutes. Take it."""
        manager = self.manager()
        position = self.position()

        assert manager._peak_stall_exit(position, 0.92, 0.92, self.at(0), STALLED) is None
        event = manager._peak_stall_exit(position, 0.90, 0.92, self.at(7), STALLED)

        assert event is not None
        assert event.action == "PEAK_STALL"
        assert manager.closed == [1.60]

    def test_a_new_high_resets_the_clock(self) -> None:
        """The whole mechanism: while it keeps working it can never stall.

        A trade printing a new high every few minutes runs for as long as it
        keeps doing that, however many hours it takes.
        """
        manager = self.manager()
        position = self.position()
        peak = 0.70
        for minute in range(0, 60, 5):
            assert manager._peak_stall_exit(position, peak, peak, self.at(minute), STALLED) is None
            peak += 0.10
        assert manager.closed == []

    def test_a_flickering_last_decimal_is_not_a_new_high(self) -> None:
        """`peak_r` comes off a live tick and wobbles constantly.

        Without an epsilon the clock resets on every pass and the rule never
        fires once — which is worse than not having it, because it would look
        installed.
        """
        manager = self.manager()
        position = self.position()
        manager._peak_stall_exit(position, 0.90, 0.9000, self.at(0), STALLED)
        # Well inside the wait, so only the epsilon decides whether the clock
        # survives these passes.
        for minute in (1, 2, 3):
            assert (
                manager._peak_stall_exit(
                    position, 0.90, 0.9000 + minute * 0.0001, self.at(minute), STALLED
                )
                is None
            )
        event = manager._peak_stall_exit(position, 0.90, 0.9007, self.at(7), STALLED)
        assert event is not None

    def test_too_soon_is_left_alone(self) -> None:
        manager = self.manager()
        position = self.position()
        manager._peak_stall_exit(position, 0.92, 0.92, self.at(0), STALLED)
        assert manager._peak_stall_exit(position, 0.92, 0.92, self.at(5), STALLED) is None

    def test_a_small_gain_is_not_worth_protecting(self) -> None:
        """Below the arming R the noise band is wide enough that no new high
        says nothing at all."""
        manager = self.manager()
        position = self.position()
        manager._peak_stall_exit(position, 0.40, 0.40, self.at(0), STALLED)
        assert manager._peak_stall_exit(position, 0.40, 0.40, self.at(30), STALLED) is None

    def test_a_trade_that_already_gave_it_back_belongs_to_the_giveback(self) -> None:
        """This rule leaves at the top; it does not confirm a retrace.

        Once price is far below the peak the money is gone and the give-back
        rule owns the decision — which weighs the health read, as it should.
        """
        manager = self.manager()
        position = self.position()
        manager._peak_stall_exit(position, 2.00, 2.00, self.at(0), STALLED)
        assert manager._peak_stall_exit(position, 0.80, 2.00, self.at(30), STALLED) is None

    def test_switching_it_off_forgets_the_position_entirely(self) -> None:
        manager = self.manager()
        manager.settings.trade_management = manager.settings.trade_management.model_copy(
            update={"peak_stall_minutes": 0.0}
        )
        position = self.position()
        assert manager._peak_stall_exit(position, 0.92, 0.92, self.at(0), STALLED) is None
        assert manager._peak_stall_exit(position, 0.92, 0.92, self.at(30), STALLED) is None
        assert manager._peak_seen == {}

    def test_each_position_is_timed_separately(self) -> None:
        manager = self.manager()
        first, second = self.position(1), self.position(2)
        manager._peak_stall_exit(first, 0.92, 0.92, self.at(0), STALLED)
        manager._peak_stall_exit(second, 0.92, 0.92, self.at(5), STALLED)

        assert manager._peak_stall_exit(first, 0.92, 0.92, self.at(7), STALLED) is not None
        assert manager._peak_stall_exit(second, 0.92, 0.92, self.at(7), STALLED) is None

    def test_a_restart_resets_the_clock_toward_holding(self) -> None:
        """Losing the timer must never be able to close a trade sooner.

        A fresh manager has no memory of the peak, so the wait starts again —
        the safe direction for a rule whose action is to exit.
        """
        manager = self.manager()
        position = self.position()
        manager._peak_stall_exit(position, 0.92, 0.92, self.at(0), STALLED)

        restarted = self.manager()
        assert restarted._peak_stall_exit(position, 0.92, 0.92, self.at(7), STALLED) is None
        assert restarted._peak_stall_exit(position, 0.92, 0.92, self.at(14), STALLED) is not None


class TestUnmanagedIsVisible:
    """A position the loop skips gets nothing from the manager, and used to
    say nothing about it either.

    Both paths in were a bare `continue`. A trade taking one of them has no
    health read, no give-back, no profit lock, no peak stall and no time exit
    — the broker stop is the only thing holding it. That is a legitimate state
    after a manual trade or a half-recovered restart, and a serious one to be
    in unknowingly.
    """

    @staticmethod
    def manager_with(journal_row):  # type: ignore[no-untyped-def]
        broker = BrokerStub()

        class Journal(JournalStub):
            def open_trade_by_ticket(self, ticket):  # type: ignore[no-untyped-def]
                return journal_row

        return manager_for(broker, Journal()), broker

    def test_a_position_absent_from_the_journal_is_reported(self) -> None:
        manager, _ = self.manager_with(None)
        manager.manage([position()], NOW)

        health = manager.last_health[555]
        assert health.verdict == "unmanaged"
        assert "no open trade on record" in health.reason

    def test_a_zero_width_stop_is_reported_with_the_price(self) -> None:
        """1R is zero, and every rule in the file divides by it."""
        manager, _ = self.manager_with(
            {"id": 1, "ticket": 555, "sl": ENTRY, "volume": 0.02, "mfe_r": 0.0}
        )
        manager.manage([position()], NOW)

        health = manager.last_health[555]
        assert health.verdict == "unmanaged"
        assert "the same as entry" in health.reason

    def test_it_reaches_the_published_summary(self) -> None:
        """The deck reads `summary()`, so the reason has to survive the trip."""
        manager, _ = self.manager_with(None)
        manager.manage([position()], NOW)

        summary = manager.last_health[555].summary()
        assert summary["verdict"] == "unmanaged"
        assert summary["reason"]
        assert summary["action"] == "hold"

    def test_a_managed_position_is_never_marked_unmanaged(self) -> None:
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)
        at(broker, 0.3)
        manager.manage([position(opened_at=NOW - timedelta(minutes=30))], NOW)

        assert manager.last_health[555].verdict != "unmanaged"


class TestTheClosingHour:
    """The hour before the wind-down, which nothing treated as different.

    `EVENING_FLAT` closes everything at the deadline whatever it is worth, and
    that is the backstop. Before it, every profit rule behaved as though 21:00
    were any other hour: the peak stall waits six minutes for a peak that will
    not come, and the give-back waits for a drain that the closing spread
    supplies for free.

    A live ASX200 long: opened 20:41, carried past the cash close, shut at
    22:27 for -0.76.

    There is no R threshold in this rule and there must not be one. It asks
    whether the rest of the target is reachable in the session that is left,
    and whether what is on the table beats the spread and commission it costs
    to collect. A number chosen in advance can answer neither.
    """

    #: Inside the closing hour for forex (wind-down 20:15 UTC), outside the
    #: flatten window itself.
    CLOSING = datetime(2026, 8, 4, 19, 40, tzinfo=UTC)

    @staticmethod
    def far_target(broker: BrokerStub) -> Position:
        """A target far enough away that this market cannot reach it today.

        The stub's ATR is 1.0 on the speed timeframe, and displacement grows
        with the square root of the bars, so 10 units away is 100 bars — well
        past the 35 minutes of session left at `CLOSING`.
        """
        del broker
        return replace(position(), tp=ENTRY + 10.0)

    def test_a_profit_is_taken_when_the_rest_is_out_of_reach(self) -> None:
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)

        at(broker, 0.5)
        events = manager.manage([self.far_target(broker)], self.CLOSING)

        assert [event.action for event in events] == ["SESSION_DECAY"]
        assert broker.closed == [(555, None)]

    def test_a_target_still_within_reach_is_held(self) -> None:
        """The point of measuring instead of thresholding: same clock, same R,
        and the answer flips because the target is close enough to happen."""
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)

        at(broker, 0.5)
        near = replace(position(), tp=broker.price + 0.5)
        events = manager.manage([near], self.CLOSING)

        assert not any(event.action == "SESSION_DECAY" for event in events)

    def test_a_loser_is_left_to_the_rules_that_understand_it(self) -> None:
        """Closing a losing trade because it is late is the time exit's job,
        and it weighs things this rule cannot see."""
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)

        at(broker, -0.4)
        events = manager.manage([self.far_target(broker)], self.CLOSING)

        assert not any(event.action == "SESSION_DECAY" for event in events)

    def test_a_profit_smaller_than_the_cost_of_taking_it_is_not_taken(self) -> None:
        """Banking below the exit cost hands the broker the profit and calls it
        discipline. The floor is measured, not chosen."""
        broker, journal = BrokerStub(), JournalStub()
        broker.spread = (ENTRY - STOP) * 0.40  # 40% of the stop, just to leave
        manager = manager_for(broker, journal)

        at(broker, 0.2)
        events = manager.manage([self.far_target(broker)], self.CLOSING)

        assert not any(event.action == "SESSION_DECAY" for event in events)

    def test_the_middle_of_the_day_is_left_alone(self) -> None:
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)

        at(broker, 0.5)
        events = manager.manage([self.far_target(broker)], AFTERNOON)

        assert not any(event.action == "SESSION_DECAY" for event in events)
        assert broker.closed == []

    def test_it_outranks_the_peak_stall_and_the_give_back(self) -> None:
        """Neither of those knows the session is ending, so both would sit on
        the profit while the reason to hold it drains away."""
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal, giveback_arm_r=0.4, giveback_fraction=0.4)

        at(broker, 1.5)
        manager.manage([self.far_target(broker)], AFTERNOON)  # records the peak
        at(broker, 0.8)
        events = manager.manage([self.far_target(broker)], self.CLOSING)

        assert [event.action for event in events] == ["SESSION_DECAY"]

    def test_switching_it_off_restores_the_old_behaviour(self) -> None:
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal, session_decay_enabled=False)

        at(broker, 0.5)
        events = manager.manage([self.far_target(broker)], self.CLOSING)

        assert not any(event.action == "SESSION_DECAY" for event in events)

    def test_a_refused_close_is_not_recorded_as_an_exit(self) -> None:
        broker, journal = BrokerStub(), JournalStub()
        broker.close_position = lambda _p, volume=None: OrderResult(  # type: ignore[assignment]
            ok=False, filled_price=None
        )
        manager = manager_for(broker, journal)

        at(broker, 0.5)
        events = manager.manage([self.far_target(broker)], self.CLOSING)

        assert not any(event.action == "SESSION_DECAY" for event in events)


def test_the_closing_hour_estimate_cannot_take_the_guard_down() -> None:
    """`atr` raises on a frame too short to measure, and this runs every second
    on every open position. I shipped it without the guard and the replay suite
    caught it: no estimate must mean "do not act", never an exception out of
    the loop that is watching the money.
    """
    broker, journal = BrokerStub(), JournalStub()
    full = broker.copy_rates

    def thin(symbol, timeframe, count):  # type: ignore[no-untyped-def]
        """Too few M5 bars to measure a pace with, everything else normal."""
        rows = full(symbol, timeframe, count)
        return rows[:3] if timeframe == Timeframe.M5.mt5_value else rows

    broker.copy_rates = thin  # type: ignore[assignment]
    manager = manager_for(broker, journal)

    at(broker, 0.5)
    events = manager.manage([replace(position(), tp=ENTRY + 10.0)], TestTheClosingHour.CLOSING)

    assert not any(event.action == "SESSION_DECAY" for event in events)


class TestBankingAProfitWorthTaking:
    """Take a sum worth taking, unless the move is clearly still running.

    Every other rule here holds by default and acts on evidence of trouble.
    This one is the other way round on purpose. The operator put it plainly:
    on a hundred-euro account, sixty or eighty cents is a fine amount to bank,
    and a profit you can see beats a bigger one you are hoping for.

    The live case: USDCHF long entered 0.81009, peaked 7.1 pips up — about 76
    cents — never reached a target 14.5 pips away, and closed at +2.7 pips for
    29 cents. It kept 38% of its best moment while a rule that simply took the
    76 cents sat there unwritten.
    """

    EQUITY = 123.43

    @staticmethod
    def running(broker: BrokerStub, *, with_us: bool) -> None:
        """Point the M1 series with the long or against it."""
        broker.drift = 0.25 if with_us else -0.25

    def manager(self, broker: BrokerStub, journal: JournalStub, **overrides):  # type: ignore[no-untyped-def]
        made = manager_for(broker, journal, **overrides)
        made.equity = self.EQUITY
        return made

    def test_a_worthwhile_profit_on_a_stalling_move_is_taken(self) -> None:
        broker, journal = BrokerStub(), JournalStub()
        manager = self.manager(broker, journal)

        at(broker, 0.5)
        # 0.74 on 123.43 is 0.60% of the account — exactly the threshold.
        events = manager.manage([replace(position(), profit=0.80)], NOW)

        assert [event.action for event in events] == ["PROFIT_BANKED"]
        assert broker.closed == [(555, None)]

    def test_a_move_running_hard_is_banked_because_that_is_exhaustion(self) -> None:
        """This test asserted the opposite, and the opposite was wrong.

        `backtest.cmd --exits --days 90` walked every position the theories
        would have opened and measured what an extra minute of patience was
        worth at each in-profit moment. A running move is negative at all
        seven profit levels, on 2,300 to 6,300 observations each: -0.092R at
        0.00-0.15R, -0.192R at 0.30-0.50R, still -0.086R at 1.50R and above.

        A hard run is exhaustion, not confirmation. Holding through one is the
        single most expensive thing this rule used to do.
        """
        broker, journal = BrokerStub(), JournalStub()
        self.running(broker, with_us=True)
        manager = self.manager(broker, journal)

        at(broker, 0.5)
        events = manager.manage([replace(position(), profit=0.80)], NOW)

        assert [event.action for event in events] == ["PROFIT_BANKED"]

    def test_a_move_retracing_hard_is_the_one_thing_that_earns_a_hold(self) -> None:
        """The other side of the same measurement. Against the position is the
        only pace where waiting paid: +0.036R at 0.00-0.15R rising to +0.242R
        above 1.50R. Price coming back is where it comes back from."""
        broker, journal = BrokerStub(), JournalStub()
        self.running(broker, with_us=False)
        manager = self.manager(broker, journal)

        at(broker, 0.5)
        events = manager.manage([replace(position(), profit=0.80)], NOW)

        assert not any(event.action == "PROFIT_BANKED" for event in events)

    def test_a_profit_too_small_to_bother_with_is_left_alone(self) -> None:
        broker, journal = BrokerStub(), JournalStub()
        manager = self.manager(broker, journal)

        at(broker, 0.3)
        events = manager.manage([replace(position(), profit=0.20)], NOW)

        assert not any(event.action == "PROFIT_BANKED" for event in events)

    def test_the_threshold_scales_with_the_account(self) -> None:
        """ "Eighty cents on a hundred euro" and "ten euro on a thousand" are one
        rule said twice, so it is written once and nobody edits it after a
        deposit."""
        broker, journal = BrokerStub(), JournalStub()
        manager = self.manager(broker, journal)
        manager.equity = 1000.0

        at(broker, 0.5)
        small = manager.manage([replace(position(), profit=0.80)], NOW)
        assert not any(event.action == "PROFIT_BANKED" for event in small)

        events = manager.manage([replace(position(), profit=8.00)], NOW)
        assert [event.action for event in events] == ["PROFIT_BANKED"]

    def test_a_lot_rounded_down_does_not_disarm_the_rule(self) -> None:
        """The live AUDCAD long, and the reason this rule had never once fired.

        The sizer rounds the lot *down* to the broker's 0.01 step, so the risk
        actually carried is whatever fits under the intended 2%. On that trade
        it was EUR 1.00 on a EUR 151 account — 0.66%, not 2%. Six tenths of a
        percent of equity is then EUR 0.91, which is 0.91R: the rule was not
        banking anything, it was waiting for nearly the whole target. The trade
        peaked in profit and closed at -1.61R, and PROFIT_BANKED appears
        nowhere in the last twenty trades.

        Capping the threshold against the risk the trade actually carries
        restores the 0.3R the configuration always claimed to be doing.
        """
        broker, journal = BrokerStub(), JournalStub(risk_money=1.00)
        manager = self.manager(broker, journal)
        manager.equity = 151.0

        assert manager.equity * 0.6 / 100.0 > 0.50, "the equity share alone would refuse this"

        at(broker, 0.5)
        events = manager.manage([replace(position(), profit=0.50)], NOW)

        assert [event.action for event in events] == ["PROFIT_BANKED"]

    def test_the_r_cap_never_banks_more_eagerly_than_the_account_allows(self) -> None:
        """Both readings have to agree, and the lower one wins. On a well-sized
        position 0.3R is the larger of the two and the account's own sense of a
        sum worth having is what binds — otherwise a deposit-scaled rule would
        quietly start taking three cents off a big trade on a small account."""
        broker, journal = BrokerStub(), JournalStub(risk_money=100.0)
        manager = self.manager(broker, journal)
        manager.equity = 50.0  # 0.6% is EUR 0.30; 0.3R would be EUR 30.00

        at(broker, 0.5)
        events = manager.manage([replace(position(), profit=0.40)], NOW)

        assert [event.action for event in events] == ["PROFIT_BANKED"]

    def test_a_row_without_a_recorded_risk_leaves_the_cap_off(self) -> None:
        """A recovered row missing the column must cost the rule its cap, not
        take down the loop watching every open position."""
        broker, journal = BrokerStub(), JournalStub()
        journal.open_trade_by_ticket = lambda ticket: {  # type: ignore[assignment]
            "id": 1,
            "ticket": ticket,
            "sl": STOP,
            "volume": 0.02,
            "mfe_r": 0.0,
        }
        manager = self.manager(broker, journal)

        at(broker, 0.5)
        events = manager.manage([replace(position(), profit=0.80)], NOW)

        assert [event.action for event in events] == ["PROFIT_BANKED"]

    def test_a_losing_position_is_never_banked(self) -> None:
        broker, journal = BrokerStub(), JournalStub()
        manager = self.manager(broker, journal)

        at(broker, -0.5)
        events = manager.manage([replace(position(), profit=-1.50)], NOW)

        assert not any(event.action == "PROFIT_BANKED" for event in events)

    def test_without_an_equity_reading_the_rule_is_off(self) -> None:
        """A share of an equity nobody set is a share of nothing, and the safe
        reading of that is to do nothing rather than to bank everything."""
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)  # equity left at 0

        at(broker, 0.5)
        events = manager.manage([replace(position(), profit=5.00)], NOW)

        assert not any(event.action == "PROFIT_BANKED" for event in events)

    def test_it_can_be_switched_off(self) -> None:
        broker, journal = BrokerStub(), JournalStub()
        manager = self.manager(broker, journal, bank_enabled=False)

        at(broker, 0.5)
        events = manager.manage([replace(position(), profit=5.00)], NOW)

        assert not any(event.action == "PROFIT_BANKED" for event in events)

    def test_it_outranks_the_stop_moving_rules(self) -> None:
        """Banking real money beats adjusting a stop. Break-even would
        otherwise fire first and the position would still be open."""
        broker, journal = BrokerStub(), JournalStub()
        manager = self.manager(broker, journal)

        at(broker, 0.9)
        events = manager.manage([replace(position(), profit=1.20)], NOW)

        assert [event.action for event in events] == ["PROFIT_BANKED"]
        assert broker.modified == [], "no stop moves on the way out"

    def test_a_refused_close_is_not_recorded_as_banked(self) -> None:
        broker, journal = BrokerStub(), JournalStub()
        broker.close_position = lambda _p, volume=None: OrderResult(  # type: ignore[assignment]
            ok=False, filled_price=None
        )
        manager = self.manager(broker, journal)

        at(broker, 0.5)
        events = manager.manage([replace(position(), profit=0.80)], NOW)

        assert not any(event.action == "PROFIT_BANKED" for event in events)


class TestTheFridayGap:
    """Seventy-five minutes on a Friday where the system refused to open a
    position and left the ones it had.

    `block_friday_after` stops entries at 19:00 UTC because of the weekend, and
    `minutes_of_runway` correctly reported zero from then. But the flatten only
    knew about the generic evening window at 20:15, so between the two the
    system was saying "there is no time left to open anything" while a losing
    position sat in exactly the thin Friday book the gate exists to avoid.
    """

    FRIDAY = datetime(2026, 8, 7, 19, 30, tzinfo=UTC)  # inside the old gap
    THURSDAY = datetime(2026, 8, 6, 19, 30, tzinfo=UTC)  # the same clock time
    FRIDAY_EARLY = datetime(2026, 8, 7, 18, 30, tzinfo=UTC)  # before the cut-off

    def test_a_position_is_flattened_at_the_friday_cut_off(self) -> None:
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)

        at(broker, -0.3)
        events = manager.manage([position()], self.FRIDAY)

        assert [event.action for event in events] == ["EVENING_FLAT"]
        assert "Friday cut-off" in events[0].detail

    def test_the_same_hour_on_a_thursday_is_left_alone(self) -> None:
        """The fix must reach the Friday deadline and nothing else."""
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)

        at(broker, -0.3)
        events = manager.manage([position()], self.THURSDAY)

        assert not any(event.action == "EVENING_FLAT" for event in events)

    def test_before_the_cut_off_friday_trades_normally(self) -> None:
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)

        at(broker, -0.3)
        events = manager.manage([position()], self.FRIDAY_EARLY)

        assert not any(event.action == "EVENING_FLAT" for event in events)

    def test_a_winner_goes_flat_too(self) -> None:
        """Same reasoning as the evening rule. "Let this one run over the
        weekend" is how a gap turns a good trade into a bad one."""
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)

        at(broker, 1.4)
        events = manager.manage([position()], self.FRIDAY)

        assert [event.action for event in events] == ["EVENING_FLAT"]

    def test_a_continuous_market_is_still_exempt(self) -> None:
        """Crypto has no FX rollover and no weekend."""
        broker = BrokerStub(asset_class=AssetClass.CRYPTO)
        manager = manager_for(broker, JournalStub())

        at(broker, 0.3)
        events = manager.manage([replace(position(), symbol="BTCUSD")], self.FRIDAY)

        assert not any(event.action == "EVENING_FLAT" for event in events)

    def test_the_flatten_and_the_runway_agree(self) -> None:
        """The bug was two definitions of one deadline. Zero runway and "stay
        in the market" must never both be true again."""
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)

        for moment in (self.FRIDAY, self.FRIDAY_EARLY, self.THURSDAY):
            runway = manager._runway_minutes(moment, "forex")
            flat = manager._should_be_flat(moment, "forex")
            assert not (runway == 0 and not flat), f"{moment}: no runway yet still holding"


class TestPayingToLeaveWhenTheStopIsAlreadyThere:
    """A health reading is a reason to want out. It is not an argument that
    leaving is affordable, and on this account the two come apart often.

    There is already a guaranteed exit sitting at the broker, costing nothing
    to keep. So closing at market is never "get out" — it is "get out *here*
    instead of *there*", and all it can win is the gap between the two. The
    live AUDCAD long closed on a health reading for -1.61R against a -1.00R
    plan with its stop 10.9 pips away: most of that last stretch was bought,
    not suffered.

    Driven through `_act_on_health` directly. Coaxing the stub's bar series
    into producing a genuine `exit` reading would be testing the health engine,
    which has its own suite; what needs pinning here is that a reading asking
    to close does not get to.
    """

    @staticmethod
    def reading(action: str) -> PositionHealth:
        return PositionHealth("broken", 1.0, action, (), "momentum turned")

    def manager(self, broker: BrokerStub, journal: JournalStub, **risk):  # type: ignore[no-untyped-def]
        settings = load_settings(env_overrides=False)
        if risk:
            settings = settings.model_copy(update={"risk": settings.risk.model_copy(update=risk)})
        return PositionManager(broker, journal, settings)  # type: ignore[arg-type]

    def act(self, broker: BrokerStub, manager, action: str = "exit"):  # type: ignore[no-untyped-def]
        return manager._act_on_health(
            position(), self.reading(action), -0.5, ENTRY - STOP, broker.tick("EURUSD")
        )

    def test_a_far_stop_and_a_thin_spread_still_closes(self) -> None:
        """0.75R of room to the stop against 0.05R to cross it. The reading
        wins, and it should."""
        broker, journal = BrokerStub(spread=0.2), JournalStub()
        at(broker, -0.25)

        event = self.act(broker, self.manager(broker, journal))

        assert event is not None and event.action == "HEALTH_EXIT"
        assert broker.closed == [(555, None)]

    def test_a_near_stop_and_a_wide_spread_is_left_to_the_stop(self) -> None:
        """0.05R of room against 0.25R to collect it. Closing here buys 5% of
        the risk for 25% of it, and the stop was going to do it for free."""
        broker, journal = BrokerStub(spread=0.5), JournalStub()
        at(broker, -0.95)

        assert self.act(broker, self.manager(broker, journal)) is None
        assert broker.closed == [], "nothing may be sold to save less than the sale costs"

    def test_it_applies_to_securing_a_profit_too(self) -> None:
        """A trade at +0.05R over a stop that has been pulled to break even is
        the same arithmetic wearing a happier face."""
        broker, journal = BrokerStub(spread=0.5), JournalStub()
        at(broker, -0.95)

        assert self.act(broker, self.manager(broker, journal), "secure") is None
        assert broker.closed == []

    def test_the_slippage_the_stop_would_suffer_counts_for_leaving(self) -> None:
        """Running to the stop is not free either. An AUDNZD stop at 1.19722
        filled at 1.19705, and comparing a certain market fill against a stop
        fill that never happens would be dishonest in the broker's favour."""
        broker, journal = BrokerStub(spread=0.5), JournalStub()
        at(broker, -0.95)

        blind = self.manager(broker, journal)
        measured = self.manager(broker, journal, stop_slippage_pips={"forex": 1.0})

        assert self.act(broker, blind) is None
        assert self.act(broker, measured) is not None, "0.55R saved now beats 0.25R to leave"

    def test_an_ai_close_faces_the_same_arithmetic(self) -> None:
        """The gate the health reading passes, applied to the adviser too.

        Thirty days measured the asymmetry. HEALTH_EXIT, gated, averaged
        -0.45R over nine trades. Letting the stop do it, BROKER_SL, averaged
        -0.57R over six. AI_CLOSE, ungated, averaged -0.60R over eight -- worse
        than leaving the position alone. The gated rule beat the stop and the
        ungated one lost to it, which is the shape of a missing gate rather
        than a bad adviser. An opinion about direction does not change what
        crossing the spread costs.
        """
        broker, journal = BrokerStub(spread=0.5), JournalStub()
        at(broker, -0.95)  # 0.05R of room against 0.25R to collect it

        event = self.manager(broker, journal).apply_supervision(
            position(),
            Supervision("close", "momentum has gone", confidence=0.90, provider="test"),
        )

        assert event is not None and event.action == "AI_EXIT_NOT_WORTH_PAYING"
        assert broker.closed == [], "the free stop two pips away was the cheaper exit"

    def test_an_ai_close_with_real_distance_to_save_still_goes_through(self) -> None:
        """A cost comparison, not a veto on the adviser."""
        broker, journal = BrokerStub(spread=0.2), JournalStub()
        at(broker, -0.25)  # 0.75R of room against 0.05R to cross it

        event = self.manager(broker, journal).apply_supervision(
            position(),
            Supervision("close", "target will not be reached", confidence=0.90, provider="test"),
        )

        assert event is not None
        assert event.action in {"AI_CLOSE", "AI_CLOSE_SENT"}
        assert broker.closed, "there was real distance to save and it was taken"

    def test_a_tighten_is_never_blocked_by_this(self) -> None:
        """Moving a stop costs nothing to place and risks less afterwards.
        There is no crossing to buy, so there is nothing to weigh."""
        broker, journal = BrokerStub(spread=0.5), JournalStub()
        at(broker, -0.95)

        event = self.act(broker, self.manager(broker, journal), "tighten")

        assert event is None or event.action == "HEALTH_TIGHTEN"
        assert broker.closed == []

    def test_a_hold_is_still_a_hold(self) -> None:
        broker, journal = BrokerStub(spread=0.2), JournalStub()
        at(broker, -0.25)

        assert self.act(broker, self.manager(broker, journal), "hold") is None
        assert broker.closed == []

    def test_without_a_stop_the_reading_stands(self) -> None:
        """Nothing to compare against. A position with no stop is the one case
        where a market close genuinely is the only exit there is."""
        broker, journal = BrokerStub(spread=0.5), JournalStub()
        at(broker, -0.95)
        manager = self.manager(broker, journal)

        event = manager._act_on_health(
            replace(position(), sl=0.0),
            self.reading("exit"),
            -0.95,
            ENTRY - STOP,
            broker.tick("EURUSD"),
        )

        assert event is not None and broker.closed == [(555, None)]

    def test_without_a_price_the_reading_stands(self) -> None:
        """No read on where we are is not an argument for staying in."""
        broker, journal = BrokerStub(spread=0.5), JournalStub()
        at(broker, -0.95)
        manager = self.manager(broker, journal)

        event = manager._act_on_health(position(), self.reading("exit"), -0.95, ENTRY - STOP, None)

        assert event is not None and broker.closed == [(555, None)]

    def test_a_zero_risk_position_does_not_divide_by_it(self) -> None:
        broker, journal = BrokerStub(spread=0.5), JournalStub()
        manager = self.manager(broker, journal)

        assert manager._worth_paying_to_leave(position(), 0.0, broker.tick("EURUSD")) is True


class TestTheAccountLearningWhenToTakeIt:
    """The one thing the database is allowed to move, and the bound that makes
    it safe.

    Everywhere else the rule is absolute: nothing read from Postgres may change
    a threshold, because a learning system that can rewrite its own risk
    controls is how an account dies. This is the exception, and it is bounded
    by direction rather than by trust — `_worth_taking` takes the MINIMUM, so
    the learned value can only ever bank sooner. Earlier is less exposure. The
    worst case is money left on the table.
    """

    #: Deliberately large, so the equity share never binds and each test
    #: measures the R terms it is about. `test_the_equity_share_still_caps_it`
    #: drops it back to the real account to check the other direction.
    EQUITY = 1000.0

    class Learned:
        """A brain that has made up its mind."""

        def __init__(self, take_at_r: float | None) -> None:
            self.take_at_r = take_at_r
            self.calls = 0

        def learned_bank_threshold(self, **_: object) -> float | None:
            self.calls += 1
            return self.take_at_r

    def manager(self, brain=None, **overrides):  # type: ignore[no-untyped-def]
        settings = load_settings(env_overrides=False)
        if overrides:
            management = settings.trade_management.model_copy(update=overrides)
            settings = settings.model_copy(update={"trade_management": management})
        made = PositionManager(BrokerStub(), JournalStub(), settings, brain)  # type: ignore[arg-type]
        made.equity = self.EQUITY
        return made

    def test_a_lower_learned_level_is_used(self) -> None:
        """0.18R against a configured 0.3R: the account's own history says take
        it sooner, and sooner is always allowed."""
        made = self.manager(self.Learned(0.18), bank_at_r=0.3)

        assert made._worth_taking(risk_money=10.0) == pytest.approx(1.8)

    def test_a_higher_learned_level_is_ignored(self) -> None:
        """The bound. If the database ever answered 0.9R — from a bug, a bad
        migration, or a stretch of luck — honouring it would mean holding
        positions longer than the operator configured, on the say-so of a
        remote table nobody reviewed."""
        made = self.manager(self.Learned(0.9), bank_at_r=0.3)

        assert made._worth_taking(risk_money=10.0) == pytest.approx(3.0)

    def test_no_brain_leaves_the_configured_threshold_alone(self) -> None:
        assert self.manager(None, bank_at_r=0.3)._worth_taking(10.0) == pytest.approx(3.0)

    def test_a_brain_with_too_little_evidence_changes_nothing(self) -> None:
        """It returns None until forty trades have closed, and None must read
        as 'no opinion' rather than as zero."""
        assert self.manager(self.Learned(None), bank_at_r=0.3)._worth_taking(10.0) == pytest.approx(
            3.0
        )

    def test_the_equity_share_still_caps_it(self) -> None:
        """Three sentences now say the same thing and the trade gets the
        smallest. A learned level must not be able to talk past the operator's
        own sense of a sum worth having."""
        made = self.manager(self.Learned(0.9), bank_at_r=0.9)
        made.equity = 151.0

        # 0.6% of 151 is 0.906, well under 0.9R of a EUR 10 risk.
        assert made._worth_taking(risk_money=10.0) == pytest.approx(0.906, abs=1e-3)

    def test_it_is_asked_once_and_then_cached(self) -> None:
        """The guard runs every second on every position and this is a GROUP BY
        over the whole trade history."""
        brain = self.Learned(0.2)
        made = self.manager(brain)
        for _ in range(50):
            made._worth_taking(risk_money=10.0)

        assert brain.calls == 1

    def test_a_brain_that_raises_does_not_reach_the_guard(self) -> None:
        """`Brain` already swallows its own failures, but the manager must not
        depend on that: a null object, a stub or a future implementation could
        all throw, and this runs inside the loop watching real money."""

        class Broken:
            def learned_bank_threshold(self, **_: object) -> float | None:
                raise RuntimeError("boom")

        made = self.manager(Broken(), bank_at_r=0.3)

        assert made._worth_taking(risk_money=10.0) == pytest.approx(3.0)


def test_the_learned_bank_threshold_can_be_switched_off_on_evidence() -> None:
    """The mechanism is sound; whether banking sooner is *better* is measured.

    `management_baselines` replayed ten banked trades against their own
    untouched stop and target. PEAK_STALL took +0.43R where holding paid
    +1.92R; PROFIT_BANKED took +0.33R against +1.20R; nine of the ten did worse
    than doing nothing. The learned threshold can only lower the bar, so it
    makes that fire sooner and more often -- the wrong direction until the
    baseline turns.

    The flag exists rather than a deleted call so switching back is one line.
    """
    broker, journal = BrokerStub(), JournalStub()

    class OpinionatedBrain:
        def learned_bank_threshold(self, **_: object) -> float:
            return 0.05  # far below the configured 0.30

    settings = load_settings(env_overrides=False)
    management = settings.trade_management.model_copy(update={"use_learned_bank_threshold": False})
    off = PositionManager(
        broker,  # type: ignore[arg-type]
        journal,  # type: ignore[arg-type]
        settings.model_copy(update={"trade_management": management}),
        OpinionatedBrain(),
    )
    on = PositionManager(
        broker,  # type: ignore[arg-type]
        journal,  # type: ignore[arg-type]
        settings,
        OpinionatedBrain(),
    )
    off.equity = on.equity = 10_000.0

    assert off._worth_taking(100.0) > on._worth_taking(
        100.0
    ), "with the learned floor off, the bar to bank must stay higher"


def test_it_is_on_by_default_and_off_for_this_account() -> None:
    """Default on: lowering the bar can only reduce exposure. Off here because
    this account's own baseline says the reduction costs more than it saves."""
    from config.loader import DEFAULT_CONFIG_PATH

    assert load_settings(env_overrides=False).trade_management.use_learned_bank_threshold is True
    live = load_settings(overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml", env_overrides=False)
    assert live.trade_management.use_learned_bank_threshold is False


class TestAWideStopMustNotHideRealMoney:
    """The live CADCHF trade, and the reason this route exists.

    Long at 0.58542, stop 0.58422, price 0.58595. EUR 2.82 of profit on a
    EUR 130 account — over two percent of everything — and only 0.44R, because
    the stop was twelve pips wide. Break-even (0.6R), peak-stall (0.6R) and the
    profit lock (0.7R) were all out of reach, so the whole amount sat behind a
    stop still BELOW the entry price. The operator saw that in one glance.

    Every rule in the manager is written in R, and R is the width of the stop.
    A wide structural stop therefore makes real money look like a small number
    to every rule whose job is to protect money.
    """

    def test_the_cadchf_shape_now_arms_the_lock(self) -> None:
        """0.44R with EUR 6 of risk is EUR 2.64 on EUR 130 — over the 1% bar."""
        event, sl = TestProfitLock.lock(
            peak_r=0.44,
            r_now=0.44,
            current_sl=1.0790,
            equity=130.0,
            risk_money=6.0,
        )

        assert event is not None
        assert event.action == "PROFIT_LOCK"
        assert sl is not None and sl > 1.0800, "the stop must end up above the entry"

    def test_it_says_which_number_armed_it(self) -> None:
        """So the journal shows why a 0.44R trade got a lock that reads 0.7R."""
        event, _ = TestProfitLock.lock(
            peak_r=0.44, r_now=0.44, current_sl=1.0790, equity=130.0, risk_money=6.0
        )

        assert event is not None
        assert "% of the account" in event.detail

    def test_it_still_secures_the_configured_fraction_and_no_more(self) -> None:
        """It is the same lock, reached by a different door. Half of a 0.44R
        peak is 0.22R, which on a 10-pip risk is 2.2 pips above entry — not a
        stop jammed under the high, which is the expensive habit."""
        event, sl = TestProfitLock.lock(
            peak_r=0.44, r_now=0.44, current_sl=1.0790, equity=130.0, risk_money=6.0
        )

        assert event is not None
        assert sl == pytest.approx(1.08022, abs=1e-6)

    def test_small_money_still_waits_for_the_r_floor(self) -> None:
        """The route is account-relative, not a way round the floor. The same
        0.44R holding EUR 0.60 on EUR 130 is under 1% and changes nothing."""
        event, sl = TestProfitLock.lock(
            peak_r=0.44,
            r_now=0.44,
            current_sl=1.0790,
            equity=130.0,
            risk_money=1.35,
        )

        assert event is None and sl is None

    def test_an_unknown_equity_cannot_arm_it(self) -> None:
        """Fails closed. Without equity the share of the account is unknowable,
        and a rule that moves stops must not act on a number it does not have."""
        event, _ = TestProfitLock.lock(
            peak_r=0.44, r_now=0.44, current_sl=1.0790, equity=0.0, risk_money=6.0
        )

        assert event is None

    def test_it_is_not_the_old_blind_cash_bank(self) -> None:
        """The rule this resembles and must not become. `bank_enabled` closed a
        winner the moment a number went green and was measured losing to the
        plan nine times in ten. This one moves a stop and closes nothing — the
        position is still open with its take-profit intact.

        Note this route is NOT a replacement for `_bank_worthwhile_profit`,
        which is separately enabled on this account and closes on the money.
        The two answer different questions: that one asks whether to take the
        cash now, this one asks whether the cash may stay exposed behind a stop
        below entry while the trade runs on. A lock that closed anything would
        be the banking rule wearing a different name."""
        event, _ = TestProfitLock.lock(
            peak_r=0.44, r_now=0.44, current_sl=1.0790, equity=130.0, risk_money=6.0
        )

        assert event is not None
        assert event.action == "PROFIT_LOCK"
        assert event.exit_price is None, "a lock must never close the position"
        assert event.pnl_money is None

    def test_switching_the_route_off_restores_the_old_behaviour(self) -> None:
        """Zero disables it, and then the CADCHF shape waits for 0.7R again."""
        from config.schema import TradeManagementConfig

        assert TradeManagementConfig().capital_protection_at_equity_pct > 0
        off = TradeManagementConfig(capital_protection_at_equity_pct=0.0)

        assert off.capital_protection_at_equity_pct == 0.0


class TestASharedIsNotCarriedThroughItsOwnClose:
    """ENR lost 3.61R on a stop that is defined as costing 1.00R.

    Siemens Energy, exit BROKER_SL, and 2.61R went straight through it. There
    is one way that happens: the share stopped trading while its exchange was
    shut and reopened somewhere else. The stop was there; there was no market
    to fill it in.

    A single share trades about eight hours a day and stands still for sixteen.
    Every rule in this system is written in R, and R assumes the stop binds.
    Sixteen hours a day it does not, which makes an overnight share position a
    trade without a working stoploss — the one thing this account forbids
    outright.

    The wind-down machinery already closed forex before the rollover. Shares
    were simply not on its list.
    """

    def test_the_overlay_flattens_shares_as_well_as_forex(self) -> None:
        from config.loader import load_settings

        session = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml"
        ).filters.session

        assert "stock" in session.evening_flat_asset_classes

    def test_shares_wind_down_before_the_earliest_cash_close(self) -> None:
        """15:30 UTC is Xetra in summer — the earliest close this account meets."""
        from config.loader import load_settings

        session = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml"
        ).filters.session

        assert session.evening_flat_by_class["stock"] == "15:30"
        assert session.evening_flat_by_class["stock"] < session.evening_flat_from

    def test_a_share_wind_down_may_never_be_pushed_past_the_forex_one(self) -> None:
        """The guard that stops this being widened back out by an edit."""
        from config.schema import SessionFilterConfig

        with pytest.raises(ValueError, match="may only be earlier"):
            SessionFilterConfig(
                evening_flat_from="20:15",
                evening_flat_by_class={"stock": "22:00"},
            )


class TestALosingTradeIsNotLeftToRunToItsStop:
    """Three of six trades on 15 August went entry-to-full-stop untouched.

    CHFJPY peaked at 0.12R and closed at -1.21R. UK100 peaked at 0.00R and
    closed at -0.97R. ENR peaked at 0.47R and closed at -3.61R. Every exit was
    BROKER_SL: not one rule on the account was able to speak.

    That was not caution, it was arithmetic. The give-back arms at 0.5R, the
    peak-stall at 0.6R, the profit ladders read `max(x, 0.0)`, and the free
    supervisor holds until it has five comparable states it does not yet have.
    The health reader was the only layer left, and its `tighten` rung required
    `r_now >= 0.2` — profit — so on the trades it was actually describing it
    fell through to `hold`.

    The reading existed, was correct, was logged, and could act on every trade
    except a losing one.
    """

    @staticmethod
    def _verdict(r_now: float):  # type: ignore[no-untyped-def]
        import numpy as np
        import pandas as pd

        from analysis.position_health import assess_position

        # Two INDEPENDENT families, which is what `deteriorating` needs and
        # what the corroboration rule is there to enforce: the drift readers
        # (a long walking steadily down) and the liquidity reader (a spread
        # that has widened to a large share of the trade's own risk). Two
        # drift readers agreeing would be one observation counted twice.
        falling = pd.DataFrame(
            {
                "open": np.linspace(1.1000, 1.0950, 60),
                "high": np.linspace(1.1002, 1.0952, 60),
                "low": np.linspace(1.0998, 1.0948, 60),
                "close": np.linspace(1.1000, 1.0950, 60),
            }
        )
        return assess_position(
            sign=1,
            r_now=r_now,
            age_minutes=180.0,
            fast=falling,
            structure=falling,
            spread=0.0006,
            risk=0.0010,
            fast_bar_minutes=15.0,
        )

    def test_a_deteriorating_loser_no_longer_falls_through_to_hold(self) -> None:
        health = self._verdict(r_now=-0.45)

        assert health.verdict in ("deteriorating", "broken")
        assert health.action != "hold", "the losing trade got no answer at all"

    def test_a_trade_sitting_on_its_entry_is_still_left_alone(self) -> None:
        """The floor keeps meaning something: inside 0.2R a stop move is noise."""
        health = self._verdict(r_now=-0.05)

        assert health.action == "hold"

    def test_the_tightened_stop_lands_between_price_and_the_original(self) -> None:
        """Under water the stop cannot be 'half of what it is worth' — that
        level sits on the far side of the live price. It comes half way in from
        the price to the original stop instead, so the worst case shrinks."""
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)
        losing = replace(position(), profit=-1.0)

        at(broker, -0.3)
        moved = manager._act_on_health(
            losing,
            PositionHealth("deteriorating", 0.6, "tighten", (), "structure gone"),
            r_now=-0.3,
            risk=ENTRY - STOP,
            tick=broker.tick("EURUSD"),
        )

        assert moved is not None and moved.action == "HEALTH_TIGHTEN"
        assert broker.modified, "no stop was actually sent to the broker"
        sent = broker.modified[-1]
        assert STOP < sent < broker.price, "the new stop is not between price and the old one"

    def test_it_never_widens_a_stop(self) -> None:
        """The one thing this may never do, whatever the reading says."""
        broker, journal = BrokerStub(), JournalStub()
        manager = manager_for(broker, journal)
        already_tight = replace(position(), sl=ENTRY - 0.1 * (ENTRY - STOP), profit=-1.0)

        at(broker, -0.3)
        moved = manager._act_on_health(
            already_tight,
            PositionHealth("deteriorating", 0.6, "tighten", (), "structure gone"),
            r_now=-0.3,
            risk=ENTRY - STOP,
            tick=broker.tick("EURUSD"),
        )

        assert moved is None


class TestPeakStallAsksTheReadFirst:
    """The last exit in this file that closed a live position on a timer alone.

    `_giveback_exit` has consulted the health read for weeks — "is this still
    working?" — and this one, which fires while the money is still on the table
    and is therefore the more expensive of the two to get wrong, never asked.
    The account's own replay agrees: PEAK_STALL banked +0.54R where leaving the
    position alone returned +1.17R, a lift of -0.64R.

    A healthy read DOUBLES the wait rather than cancelling it. Cancelling would
    hand the decision to the readers outright, and a market pausing quietly at
    its high is exactly the shape they call healthy — nothing is going wrong, it
    simply stopped. The rule would then never fire again, which is not
    "analysis first", it is analysis only.
    """

    HEALTHY = PositionHealth("healthy", 0.0, "hold", (), "nothing wrong")

    @staticmethod
    def at(minute: int) -> datetime:
        return NOW + timedelta(minutes=minute)

    def _manager(self):  # type: ignore[no-untyped-def]
        return manager_for(BrokerStub(), JournalStub(), peak_stall_minutes=6.0)

    def test_a_healthy_read_buys_the_trade_more_time(self) -> None:
        manager = self._manager()
        manager._peak_stall_exit(position(), 0.92, 0.92, self.at(0), self.HEALTHY)

        assert manager._peak_stall_exit(position(), 0.92, 0.92, self.at(7), self.HEALTHY) is None

    def test_but_the_clock_still_has_the_last_word(self) -> None:
        """Twice the wait and it goes, read or no read. A move that has stopped
        does not start again because a reader has not noticed yet."""
        manager = self._manager()
        manager._peak_stall_exit(position(), 0.92, 0.92, self.at(0), self.HEALTHY)

        event = manager._peak_stall_exit(position(), 0.92, 0.92, self.at(13), self.HEALTHY)

        assert event is not None
        assert event.action == "PEAK_STALL"

    def test_a_worried_read_gets_no_extension(self) -> None:
        manager = self._manager()
        manager._peak_stall_exit(position(), 0.92, 0.92, self.at(0), STALLED)

        assert manager._peak_stall_exit(position(), 0.92, 0.92, self.at(7), STALLED) is not None

    def test_the_new_high_reset_still_outranks_everything(self) -> None:
        """A trade that is still making highs has not stalled on any reading."""
        manager = self._manager()
        for minute in range(0, 20, 3):
            verdict = manager._peak_stall_exit(
                position(), 0.92 + minute * 0.01, 0.92 + minute * 0.01, self.at(minute), STALLED
            )
            assert verdict is None

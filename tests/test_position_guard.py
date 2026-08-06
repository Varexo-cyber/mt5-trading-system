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
        return Tick(symbol=symbol, bid=self.price, ask=self.price, time=NOW)

    def spec(self, _symbol: str):  # type: ignore[no-untyped-def]
        klass = self.asset_class

        class Spec:
            volume_min = 0.01
            asset_class = klass

            @staticmethod
            def normalize_price(price: float) -> float:
                return round(price, 5)

            @staticmethod
            def round_volume_down(volume: float) -> float:
                return round(volume, 2)

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

    def open_trade_by_ticket(self, ticket: int):  # type: ignore[no-untyped-def]
        return {"id": 1, "ticket": ticket, "sl": STOP, "volume": 0.02, "mfe_r": self.peak_r}

    def management_action_exists(self, _ticket, _actions) -> bool:  # type: ignore[no-untyped-def]
        return self.partial_taken

    def update_excursions(self, _trade_id, *, mae_r: float, mfe_r: float) -> None:
        self.peak_r = max(self.peak_r, mfe_r)
        self.trough_r = min(self.trough_r, mae_r)


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


def test_management_events_are_still_typed() -> None:
    broker, journal = BrokerStub(), JournalStub()
    manager = manager_for(broker, journal)
    at(broker, 1.6)
    manager.manage([position()], NOW)
    at(broker, 0.1)
    (event,) = manager.manage([position()], NOW)
    assert isinstance(event, ManagementEvent)
    assert event.r_at_action == pytest.approx(0.1)


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

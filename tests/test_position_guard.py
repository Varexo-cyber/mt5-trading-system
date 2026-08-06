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
    def lock(peak_r: float, r_now: float, current_sl: float, *, entry: float = 1.0800):
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
        event = manager._profit_lock(position, r_now, peak_r, risk=0.0010)
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

        assert manager._peak_stall_exit(position, 0.92, 0.92, self.at(0)) is None
        event = manager._peak_stall_exit(position, 0.90, 0.92, self.at(7))

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
            assert manager._peak_stall_exit(position, peak, peak, self.at(minute)) is None
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
        manager._peak_stall_exit(position, 0.90, 0.9000, self.at(0))
        # Well inside the wait, so only the epsilon decides whether the clock
        # survives these passes.
        for minute in (1, 2, 3):
            assert (
                manager._peak_stall_exit(position, 0.90, 0.9000 + minute * 0.0001, self.at(minute))
                is None
            )
        event = manager._peak_stall_exit(position, 0.90, 0.9007, self.at(7))
        assert event is not None

    def test_too_soon_is_left_alone(self) -> None:
        manager = self.manager()
        position = self.position()
        manager._peak_stall_exit(position, 0.92, 0.92, self.at(0))
        assert manager._peak_stall_exit(position, 0.92, 0.92, self.at(5)) is None

    def test_a_small_gain_is_not_worth_protecting(self) -> None:
        """Below the arming R the noise band is wide enough that no new high
        says nothing at all."""
        manager = self.manager()
        position = self.position()
        manager._peak_stall_exit(position, 0.40, 0.40, self.at(0))
        assert manager._peak_stall_exit(position, 0.40, 0.40, self.at(30)) is None

    def test_a_trade_that_already_gave_it_back_belongs_to_the_giveback(self) -> None:
        """This rule leaves at the top; it does not confirm a retrace.

        Once price is far below the peak the money is gone and the give-back
        rule owns the decision — which weighs the health read, as it should.
        """
        manager = self.manager()
        position = self.position()
        manager._peak_stall_exit(position, 2.00, 2.00, self.at(0))
        assert manager._peak_stall_exit(position, 0.80, 2.00, self.at(30)) is None

    def test_switching_it_off_forgets_the_position_entirely(self) -> None:
        manager = self.manager()
        manager.settings.trade_management = manager.settings.trade_management.model_copy(
            update={"peak_stall_minutes": 0.0}
        )
        position = self.position()
        assert manager._peak_stall_exit(position, 0.92, 0.92, self.at(0)) is None
        assert manager._peak_stall_exit(position, 0.92, 0.92, self.at(30)) is None
        assert manager._peak_seen == {}

    def test_each_position_is_timed_separately(self) -> None:
        manager = self.manager()
        first, second = self.position(1), self.position(2)
        manager._peak_stall_exit(first, 0.92, 0.92, self.at(0))
        manager._peak_stall_exit(second, 0.92, 0.92, self.at(5))

        assert manager._peak_stall_exit(first, 0.92, 0.92, self.at(7)) is not None
        assert manager._peak_stall_exit(second, 0.92, 0.92, self.at(7)) is None

    def test_a_restart_resets_the_clock_toward_holding(self) -> None:
        """Losing the timer must never be able to close a trade sooner.

        A fresh manager has no memory of the peak, so the wait starts again —
        the safe direction for a rule whose action is to exit.
        """
        manager = self.manager()
        position = self.position()
        manager._peak_stall_exit(position, 0.92, 0.92, self.at(0))

        restarted = self.manager()
        assert restarted._peak_stall_exit(position, 0.92, 0.92, self.at(7)) is None
        assert restarted._peak_stall_exit(position, 0.92, 0.92, self.at(14)) is not None


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

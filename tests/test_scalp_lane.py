"""Section six's own route to an order.

WHY THE LANE EXISTS, as arithmetic. The confluence score is a weighted mean of
(raw score x confidence) over the agreeing modules, and `candle_momentum` tops
out at 45 x 0.75 = 33.75 against a bar of 45. It could not open a trade alone.
Worse, a mean is dragged down by its weakest term, so joining a strong reader
made matters worse rather than better: market_structure alone scores 70, and
56.4 with this agreeing. A scalp voting in a swing engine lowers every score it
touches.

No threshold fixes that. A scalp's evidence is small and short-lived because
that is what a scalp is.

WHAT THESE TESTS ARE ACTUALLY FOR. The lane is a second path to a live order,
and a second path is where a guard gets forgotten. Every test below asks the
same question about a different guard: does the thing that already stops the
main path also stop this one? The list matters more than the happy case.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.types import Direction, Signal
from risk.reasons import Reason


class _Recorder:
    def __init__(self) -> None:
        self.cycles: list[dict] = []
        self.skips: list[tuple] = []
        self.shadows: list[dict] = []
        self.intents = 0

    def record_cycle(self, **kwargs):  # type: ignore[no-untyped-def]
        self.cycles.append(kwargs)
        return len(self.cycles)

    def record_sizing(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        return None

    def record_entry_intent(self, **kwargs) -> int:  # type: ignore[no-untyped-def]
        self.intents += 1
        return self.intents

    def record_order_attempt(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        return None

    def record_shadow_trade(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.shadows.append(kwargs)

    def open_shadow_count(self, reason) -> int:  # type: ignore[no-untyped-def]
        return 0


def runner(**overrides):  # type: ignore[no-untyped-def]
    """A JarvisRunner with only the parts the lane touches."""
    from config.loader import DEFAULT_CONFIG_PATH, load_settings
    from core.types import TradingMode
    from runner.service import JarvisRunner, OperationMode

    settings = load_settings(
        DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
    )
    # The shipped overlay loads in backtest; the lane refuses outright unless
    # the mode is live, which is itself one of the guards under test.
    settings = settings.model_copy(
        update={
            "system": settings.system.model_copy(
                update={"mode": overrides.get("mode", TradingMode.MICRO_LIVE)}
            )
        }
    )
    service = object.__new__(JarvisRunner)
    service.settings = settings
    service.operation = OperationMode.EXPERIMENTAL_LIVE
    service.recorder = _Recorder()
    service.sent: list = []  # type: ignore[attr-defined]

    def order_send(request, spec):  # type: ignore[no-untyped-def]
        service.sent.append(request)  # type: ignore[attr-defined]
        return SimpleNamespace(
            ok=True, filled_price=request.reference_price, retcode_name="DONE", comment=""
        )

    # The real spec, not a stand-in: the sizer reads a dozen fields off it and
    # a mock that satisfies today's code would quietly stop covering tomorrow's.
    from core.instrument import InstrumentSpec
    from tests.fakes.fake_mt5 import xauusd_spec

    spec = InstrumentSpec.from_mt5(xauusd_spec())
    service.broker = SimpleNamespace(  # type: ignore[assignment]
        account=lambda: SimpleNamespace(equity=176.0),
        spec=lambda symbol: spec,
        order_send=order_send,
    )
    service._managed_positions = lambda: overrides.get("positions", ())  # type: ignore[assignment]
    service._entry_still_allowed = lambda: overrides.get("entry_allowed", True)  # type: ignore[assignment]
    service._tripped_section = lambda idea: overrides.get("tripped")  # type: ignore[assignment]
    service._journal_cycle_context = lambda *a, **k: {}  # type: ignore[assignment]
    service._promote_confirmed_entry = lambda **k: None  # type: ignore[assignment]
    service._record_skip = lambda *a, **k: service.recorder.skips.append(a)  # type: ignore[assignment]
    service._scalp_plan = lambda *a, **k: overrides.get(  # type: ignore[assignment]
        "plan", (Direction.LONG, 2400.0, 2399.0, 2403.0)
    )
    service._cycle_contexts = {"XAUUSD": SimpleNamespace(tick=SimpleNamespace(spread=0.25))}
    service.risk = SimpleNamespace(  # type: ignore[assignment]
        build_state=lambda account, positions: SimpleNamespace(),
        check_margin=lambda *a: SimpleNamespace(
            approved=overrides.get("margin_ok", True), detail="no margin"
        ),
        assert_not_forbidden=lambda sizing, state: None,
        # The book-wide ceiling. `room` is what the cap still allows, so a
        # value under the scalp's own risk means the book is full.
        room_for_more_risk=lambda state, wanted, spec=None: overrides.get("room", wanted),
        open_risk_pct=lambda state, spec=None: overrides.get("used", 0.0),
    )
    service.journal = SimpleNamespace(abandon_pending_entry=lambda *a: None)  # type: ignore[assignment]
    return service


SIGNAL = Signal("candle_momentum", 45.0, 0.7, reasoning="a decisive minute")


def run(service, signals=(SIGNAL,), reason=Reason.NO_SIGNAL):  # type: ignore[no-untyped-def]
    return service._run_scalp_lane("cycle-1", "XAUUSD", list(signals), reason, {})


class TestItOpensATrade:
    def test_a_clean_scalp_reaches_the_broker(self) -> None:
        service = runner()

        assert run(service) is True
        assert len(service.sent) == 1

    def test_it_is_sized_at_the_fixed_lot(self) -> None:
        """Not risk-sized. At 5% of a EUR 176 account a scalp would carry EUR 9
        behind a stop measured in seconds, which is not what was asked for."""
        service = runner()
        run(service)

        assert service.sent[0].volume == 0.01

    def test_it_carries_the_stop_and_target_from_the_candle(self) -> None:
        service = runner()
        run(service)

        assert service.sent[0].sl == 2399.0
        assert service.sent[0].tp == 2403.0

    def test_it_is_marked_so_its_own_book_can_be_counted(self) -> None:
        """The concurrency cap reads the broker's position list by this
        comment rather than keeping a ledger, so a restart or a manual close
        cannot put the two out of step."""
        from runner.service import _SCALP_COMMENT

        service = runner()
        run(service)

        assert service.sent[0].comment == _SCALP_COMMENT


class TestEveryGuardThatStopsTheMainPathStopsThisOne:
    """A second route to a live order is where a guard gets forgotten."""

    def test_the_kill_switch_and_capital_floor_still_apply(self) -> None:
        service = runner(entry_allowed=False)

        assert run(service) is False
        assert service.sent == []

    def test_its_own_breaker_still_applies(self) -> None:
        service = runner(tripped="candle_momentum stopped itself: 6 losses in a row")

        assert run(service) is False
        assert service.sent == []

    def test_a_failed_margin_check_still_refuses(self) -> None:
        service = runner(margin_ok=False)

        assert run(service) is False
        assert service.sent == []

    def test_a_full_book_still_refuses(self) -> None:
        """The book-wide risk ceiling, which this lane was not asking about.

        An omission rather than a decision: the lane's docstring lists what it
        deliberately keeps no copy of -- news, kill switch, capital floor,
        margin, its own breaker -- and `max_total_open_risk_pct` is not on that
        list. Every other route to an order goes through `room_for_more_risk`.
        This one sized a fixed lot, checked the margin and sent it, so two
        scalps landed ON TOP of whatever the swing book carried instead of
        inside the same ceiling.
        """
        service = runner(room=0.0, used=24.0)

        assert run(service) is False
        assert service.sent == []
        assert service.recorder.skips, "a refusal has to be recorded, not silent"
        assert service.recorder.skips[0][3] is Reason.RISK_EXCEEDS_CAP

    def test_room_for_exactly_this_scalp_is_still_room(self) -> None:
        """Refused only when the book is genuinely too full. A lane that
        refuses whenever anything else is open would never trade again."""
        service = runner()  # room defaults to exactly what the scalp wants

        assert run(service) is True
        assert len(service.sent) == 1

    def test_a_news_window_still_refuses(self) -> None:
        """Enforced by `_scalp_plan`, which reads the same `_NEWS_BLOCKS` the
        rest of the account uses -- not by a second calendar living here."""
        service = runner(plan=None)

        assert run(service) is False
        assert service.sent == []

    def test_a_non_live_mode_never_sends(self) -> None:
        """Backtest and paper must not reach the broker through this path any
        more than through the main one."""
        from core.types import TradingMode

        service = runner(mode=TradingMode.BACKTEST)

        assert run(service) is False
        assert service.sent == []

    def test_monitor_mode_never_sends(self) -> None:
        from runner.service import OperationMode

        service = runner()
        service.operation = OperationMode.MONITOR

        assert run(service) is False
        assert service.sent == []

    def test_it_does_nothing_when_the_lane_is_switched_off(self) -> None:
        from dataclasses import replace as _replace  # noqa: F401

        service = runner()
        service.settings = service.settings.model_copy(
            update={
                "analysis": service.settings.analysis.model_copy(
                    update={
                        "candle_momentum": service.settings.analysis.candle_momentum.model_copy(
                            update={"own_lane_enabled": False}
                        )
                    }
                )
            }
        )

        assert run(service) is False
        assert service.sent == []


class TestItNeverGradesTheSameDecisionTwice:
    def test_a_scalp_it_took_is_not_also_written_down_as_paper(self) -> None:
        """Recorded on paper AND opened live would grade one decision against
        two books, and they would disagree the moment a fill differed from the
        plan. The caller returns early on a taken scalp; this pins the half of
        that contract the lane owns."""
        service = runner()
        run(service)

        assert service.recorder.shadows == []

    def test_the_journal_row_says_a_trade_was_taken(self) -> None:
        service = runner()
        run(service)

        cycle = service.recorder.cycles[-1]
        assert cycle["reason"] is Reason.OK
        assert cycle["traded"] is True
        assert cycle["weights"] == {"candle_momentum": 1.0}


class TestTheFixedLotIsACeilingAndNotAnOverride:
    def test_a_stop_too_wide_for_the_account_is_still_refused(self) -> None:
        """The fixed lot must never force through a trade the ordinary sizer
        would have refused -- that would be the lane quietly buying its way
        past the account's own limits."""
        service = runner()
        # A stop 2000 points wide on a EUR 176 account: no lot survives it.
        service._scalp_plan = lambda *a, **k: (Direction.LONG, 2400.0, 400.0, 4400.0)

        assert run(service) is False
        assert service.sent == []
        assert service.recorder.skips


@pytest.mark.parametrize("attribute", ["_run_scalp_lane", "_scalp_plan", "_open_scalp_count"])
def test_the_lane_is_wired_into_the_runner(attribute: str) -> None:
    from runner.service import JarvisRunner

    assert hasattr(JarvisRunner, attribute)


class TestTheLaneCanNeverTakeSectionOneDown:
    """Bought with a live incident. A malformed log call inside the lane raised
    AFTER the order had gone through, the exception travelled up into
    `_analyse_candidate`, and the account printed "candidate analysis failed;
    continuing with the rest of the batch" -- the line a genuinely broken
    detector produces -- on every scalp it opened.

    Section six is an experiment at 0.01 lot. Section one is the account.
    """

    def test_an_exception_in_the_lane_is_contained(self) -> None:
        service = runner()

        def explode(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("anything at all")

        service._run_scalp_lane = explode  # type: ignore[assignment]

        assert (
            service._scalp_lane_took_it("cycle-1", "XAUUSD", [SIGNAL], Reason.NO_SIGNAL, {})
            is False
        )

    def test_a_contained_failure_is_still_reported(self, caplog) -> None:  # type: ignore[no-untyped-def]
        """Dropped, never silent. An unexpected exception from a path that
        sends live orders is worth an ERROR line every single time."""
        import logging

        service = runner()

        def explode(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("anything at all")

        service._run_scalp_lane = explode  # type: ignore[assignment]

        with caplog.at_level(logging.ERROR):
            service._scalp_lane_took_it("cycle-1", "XAUUSD", [SIGNAL], Reason.NO_SIGNAL, {})

        assert "section six lane failed" in caplog.text

    def test_a_working_lane_still_reports_that_it_took_the_trade(self) -> None:
        service = runner()

        assert service._scalp_lane_took_it("cycle-1", "XAUUSD", [SIGNAL], Reason.NO_SIGNAL, {})
        assert len(service.sent) == 1


class TestAScalpBanksWhatItHasWhenItsMinuteIsOver:
    """THE FAR BACKSTOP, and it ships disabled -- `_scalp_verdict` replaced it.

    A clock was the first answer to the give-back problem and the wrong one: it
    closes a trade that is still running and holds one that has already turned.
    It knows the time and nothing about the trade. Kept as a mechanism for a
    position nothing else has an opinion about, and tested here with the clock
    switched on explicitly.

    26 August, live: +EUR 1.20 against a EUR 3.00 target, then back to
    -EUR 0.90 against a EUR 1.01 stop. A round trip of more than 2R.

    The thesis had an expiry and the exit did not. Section six enters on one
    decisive M1 candle, and its own hypothesis says the flow lasts only while
    the participant is still working their order -- after a few minutes there
    is nothing left to ride. `_giveback_exit` would have closed it at +0.60R
    and never saw the peak, because management samples on the cycle and a gold
    candle travels further between two samples than the whole trade is worth.

    A clock does not need to see the peak. It only needs to know the reason is
    gone.
    """

    def manager(self, **overrides):  # type: ignore[no-untyped-def]
        from datetime import UTC, datetime

        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from core.instrument import InstrumentSpec
        from core.types import Direction, Position
        from execution.manager import PositionManager
        from tests.fakes.fake_mt5 import xauusd_spec

        closed: list = []
        service = PositionManager.__new__(PositionManager)
        # A scalp is now identified by the order comment OR by our own books,
        # so the double needs both. The journal here says "not section six",
        # which makes the comment the only positive marker in these tests --
        # exactly the situation the fallback exists to survive.
        service._scalp_tickets = {}
        service.journal = SimpleNamespace(  # type: ignore[attr-defined]
            trade_opened_by_section_six=lambda _ticket: False
        )
        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        # The clock ships DISABLED -- `_scalp_verdict` replaced it and it stays
        # only as a far backstop. Enabled here because this class tests the
        # mechanism, not the shipped default; that is asserted separately.
        service.settings = settings.model_copy(
            update={
                "analysis": settings.analysis.model_copy(
                    update={
                        "candle_momentum": settings.analysis.candle_momentum.model_copy(
                            update={"maximum_age_minutes": 4.0}
                        )
                    }
                )
            }
        )
        service.broker = SimpleNamespace(  # type: ignore[attr-defined]
            spec=lambda symbol: InstrumentSpec.from_mt5(xauusd_spec()),
            close_position=lambda position: (
                closed.append(position),
                SimpleNamespace(ok=True, filled_price=2401.0),
            )[1],
        )
        opened = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)
        position = Position(
            ticket=1,
            symbol="XAUUSD",
            direction=Direction.LONG,
            volume=0.01,
            price_open=2400.0,
            sl=2398.8,
            tp=2403.6,
            profit=overrides.get("profit", 1.20),
            swap=0.0,
            opened_at=opened,
            magic=1,
            comment=overrides.get("comment", "jarvis-scalp"),
        )
        now = opened.replace(minute=overrides.get("minutes", 6))
        tick = SimpleNamespace(spread=overrides.get("spread", 0.25))
        return service, position, tick, now, closed

    def test_a_winning_scalp_past_its_minute_is_banked(self) -> None:
        service, position, tick, now, closed = self.manager()

        event = service._stale_scalp_exit(position, 1.19, tick, now)

        assert event is not None
        assert event.action == "SCALP_EXPIRED"
        assert closed

    def test_a_losing_scalp_keeps_its_stop(self) -> None:
        """Closing a loser early pays a spread to book a loss the stop books
        for free. The clock is for winners."""
        service, position, tick, now, closed = self.manager(profit=-0.90)

        assert service._stale_scalp_exit(position, -0.89, tick, now) is None
        assert closed == []

    def test_a_young_scalp_is_left_alone(self) -> None:
        service, position, tick, now, closed = self.manager(minutes=2)

        assert service._stale_scalp_exit(position, 1.19, tick, now) is None
        assert closed == []

    def test_a_gain_that_has_not_cleared_the_round_trip_is_left_alone(self) -> None:
        """A few cents of profit is not a gain, it is the spread not yet paid.
        Banking it hands the difference to the broker."""
        service, position, tick, now, closed = self.manager(profit=0.02)

        assert service._stale_scalp_exit(position, 0.02, tick, now) is None
        assert closed == []

    def test_it_never_touches_a_swing_trade(self) -> None:
        """Section one's positions are not scalps and have no expiring thesis.
        The clock is keyed on the lane's own order comment."""
        service, position, tick, now, closed = self.manager(comment="jarvis-exp-live")

        assert service._stale_scalp_exit(position, 1.19, tick, now) is None
        assert closed == []


class TestTheScalpIsJudgedEverySecondAndNotOnAClock:
    """The owner's rule, in his words: "hey dit gaat nog verder stijgen, ghalas
    hold. En hey dit zakt al een beetje, ok laat me claimen. Hey dit staat
    verlies, nee man dit gaat verder zakken, eruit."

    A clock was the first answer and the wrong one -- it closes a trade that is
    still running and holds one that has already turned. These ask the two
    questions that matter instead, at the cadence the guard already runs.
    """

    def watcher(self, **overrides):  # type: ignore[no-untyped-def]
        from datetime import UTC, datetime

        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from core.instrument import InstrumentSpec
        from core.types import Direction, Position
        from execution.manager import PositionManager
        from tests.fakes.fake_mt5 import xauusd_spec

        closed: list = []
        service = PositionManager.__new__(PositionManager)
        # A scalp is now identified by the order comment OR by our own books,
        # so the double needs both. The journal here says "not section six",
        # which makes the comment the only positive marker in these tests --
        # exactly the situation the fallback exists to survive.
        service._scalp_tickets = {}
        service.journal = SimpleNamespace(  # type: ignore[attr-defined]
            trade_opened_by_section_six=lambda _ticket: False
        )
        service.settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        service.broker = SimpleNamespace(  # type: ignore[attr-defined]
            spec=lambda symbol: InstrumentSpec.from_mt5(xauusd_spec()),
            copy_rates=lambda symbol, tf, n: [
                {"close": overrides.get("last_close", 2400.0)} for _ in range(n)
            ],
            close_position=lambda position: (
                closed.append(position),
                SimpleNamespace(ok=True, filled_price=overrides.get("price", 2401.0)),
            )[1],
        )
        position = Position(
            ticket=1,
            symbol="XAUUSD",
            direction=Direction.LONG,
            volume=0.01,
            price_open=2400.0,
            sl=2398.0,
            tp=2402.8,
            profit=1.0,
            swap=0.0,
            opened_at=datetime(2026, 8, 26, 9, 0, tzinfo=UTC),
            magic=1,
            comment=overrides.get("comment", "jarvis-scalp"),
        )
        price = overrides.get("price", 2401.0)
        tick = SimpleNamespace(bid=price, ask=price + 0.25, spread=0.25)
        return service, position, tick, closed

    def test_it_holds_while_the_move_is_still_going_its_way(self) -> None:
        """At its own high-water mark there is nothing given back, so it runs."""
        service, position, tick, closed = self.watcher(price=2401.40)
        # peak 5.6 spreads, price is AT the peak
        peak_r = (1.40) / 2.0

        assert service._scalp_verdict(position, 0.70, peak_r, tick, 2.0) is None
        assert closed == []

    def test_it_claims_once_the_gain_sags_past_its_leash(self) -> None:
        """Peaked 5.6 spreads, now 2.8. That is 2.8 given back against a leash
        of max(1.0, 5.6 x 0.40) = 2.24, so the move is done."""
        service, position, tick, closed = self.watcher(price=2400.70)

        event = service._scalp_verdict(position, 0.35, 1.40 / 2.0, tick, 2.0)

        assert event is not None
        assert event.action == "SCALP_CLAIMED"
        assert closed

    def test_a_sag_inside_the_leash_is_held_so_the_move_can_continue(self) -> None:
        """THE CHANGE, and the reason for it. Peaked 5.6 spreads, now 3.6 --
        two spreads given back, which a fixed one-spread leash would have
        closed. An ordinary retracement is not the end of a move, and closing
        on it is how a trade on its way to twelve spreads is capped at four.
        """
        service, position, tick, closed = self.watcher(price=2400.90)

        assert service._scalp_verdict(position, 0.45, 1.40 / 2.0, tick, 2.0) is None
        assert closed == []

    def test_the_leash_is_never_tighter_than_a_spread(self) -> None:
        """At a small peak the share is smaller than the noise floor, and the
        floor wins. Below a spread of retreat is the bid and the ask taking
        turns."""
        service, position, tick, _closed = self.watcher(price=2400.20)
        peak_spreads = 2.0
        peak_r = peak_spreads * 0.25 / 2.0

        # gained 0.8, so 1.2 given back: past the 1.0 floor, but 2.0 x 0.40 =
        # 0.8 would not have been.
        event = service._scalp_verdict(position, 0.10, peak_r, tick, 2.0)

        assert event is not None and event.action == "SCALP_CLAIMED"

    def test_a_peak_that_was_never_worth_two_spreads_is_left_alone(self) -> None:
        """What the minimum means now: the PEAK has to have been worth
        something, not the live price at the instant we happen to look.

        This test used to say the opposite. It fed a trade that had peaked 5.6
        spreads up and sagged to 0.8, and asserted it should be held -- the
        defect, written down and pinned green. The floor was tested against the
        live gain, so a trade fell out of claim range by giving back exactly
        the profit the rule exists to protect.
        """
        service, position, tick, closed = self.watcher(price=2400.20)
        never_ran = 0.30 / 2.0  # peaked 1.2 spreads, under the 2.0 floor

        assert service._scalp_verdict(position, 0.10, never_ran, tick, 2.0) is None
        assert closed == []

    def test_a_scalp_that_peaked_and_sagged_all_the_way_back_is_claimed(self) -> None:
        """The case above, corrected. Peaked 5.6 spreads, now 0.8: there is
        still money on the table and the move is plainly over."""
        service, position, tick, closed = self.watcher(price=2400.20)

        event = service._scalp_verdict(position, 0.10, 1.40 / 2.0, tick, 2.0)

        assert event is not None
        assert event.action == "SCALP_CLAIMED"
        assert closed

    def test_the_live_xauusd_trade_that_gave_back_one_euro_three(self) -> None:
        """25 August, and the reason any of this changed.

        BUY XAUUSD 0.01 at 4656.56, stop 4653.15, spread 0.29. It ran to about
        3.6 spreads -- 1.05 dollars, 0.90 euro on the phone -- and then gold
        fell a dollar a minute back through the entry. The old rule needed the
        LIVE gain to sit between 2.0 and 2.1 spreads at the moment it was
        asked: a window three cents wide, crossed between two guard ticks.
        Below 2.0 the claim was disqualified for good, so the trade held all
        the way to its stop and closed at -2.33.
        """
        service, position, tick, _closed = self.watcher(price=2400.20)
        risk, spread = 2.0, 0.25
        peak_r = 3.6 * spread / risk  # the 3.6-spread high-water mark

        event = service._scalp_verdict(position, 0.10, peak_r, tick, risk)

        assert event is not None, "this is the trade that rode to its stop"
        assert event.action == "SCALP_CLAIMED"

    def test_a_scalp_still_at_its_peak_is_never_claimed(self) -> None:
        """Arming on the peak must not turn into "close as soon as it is
        armed". Nothing given back, nothing to claim."""
        service, position, tick, closed = self.watcher(price=2401.40)

        assert service._scalp_verdict(position, 0.70, 1.40 / 2.0, tick, 2.0) is None
        assert closed == []

    def test_it_cuts_a_loser_the_market_is_still_moving_away_from(self) -> None:
        service, position, tick, closed = self.watcher(price=2399.00, last_close=2400.00)

        event = service._scalp_verdict(position, -0.50, 0.0, tick, 2.0)

        assert event is not None
        assert event.action == "SCALP_CUT"
        assert closed

    def test_a_loser_that_has_stopped_falling_keeps_its_stop(self) -> None:
        """Closing it pays a spread to book a loss the stop books for free."""
        service, position, tick, closed = self.watcher(price=2399.50, last_close=2399.45)

        assert service._scalp_verdict(position, -0.25, 0.0, tick, 2.0) is None
        assert closed == []

    def test_it_never_touches_a_swing_trade(self) -> None:
        service, position, tick, closed = self.watcher(price=2400.90, comment="jarvis-exp-live")

        assert service._scalp_verdict(position, 0.45, 1.40 / 2.0, tick, 2.0) is None
        assert closed == []

    def test_the_clock_is_off_by_default_now(self) -> None:
        """A judgement replaced it. The clock stays only as a far backstop and
        ships disabled."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        config = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.candle_momentum

        assert config.maximum_age_minutes == 0.0
        assert config.scalp_claim_spreads > 0


class TestALaneIsNotIdentifiedByAFieldTheBrokerOwns:
    """The order comment was the only marker, and it is not our field.

    MT5 truncates it at 31 characters and brokers rewrite it -- on a stop-out
    it commonly comes back as "[sl 4653.15]". If it does not survive the round
    trip, every scalp rule declines every scalp it is handed and the position
    drops through to the swing rules: break-even at 0.6R, partial close at
    1.5R. A trade meant to be in and out inside a minute gets held like a swing
    trade, nothing raises, and nothing logs.

    So the journal answers too. It is ours, it is written before the order is
    sent, and the broker cannot touch it.
    """

    def watcher(self, *, comment: str, in_journal: bool, price: float = 2400.70):  # type: ignore[no-untyped-def]
        from datetime import UTC, datetime

        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from core.instrument import InstrumentSpec
        from core.types import Position
        from execution.manager import PositionManager
        from tests.fakes.fake_mt5 import xauusd_spec

        closed: list = []
        service = PositionManager.__new__(PositionManager)
        service.settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        service._scalp_tickets = {}
        service.journal = SimpleNamespace(  # type: ignore[attr-defined]
            trade_opened_by_section_six=lambda _ticket: in_journal
        )
        service.broker = SimpleNamespace(  # type: ignore[attr-defined]
            spec=lambda symbol: InstrumentSpec.from_mt5(xauusd_spec()),
            copy_rates=lambda symbol, tf, n: [{"close": 2400.0} for _ in range(n)],
            close_position=lambda position: (
                closed.append(position),
                SimpleNamespace(ok=True, filled_price=2400.90),
            )[1],
        )
        position = Position(
            ticket=1,
            symbol="XAUUSD",
            direction=Direction.LONG,
            volume=0.01,
            price_open=2400.0,
            sl=2398.0,
            tp=2402.8,
            profit=1.0,
            swap=0.0,
            opened_at=datetime(2026, 8, 26, 9, 0, tzinfo=UTC),
            magic=1,
            comment=comment,
        )
        tick = SimpleNamespace(bid=price, ask=price + 0.25, spread=0.25)
        return service, position, tick, closed

    def test_a_scalp_whose_comment_the_broker_replaced_is_still_a_scalp(self) -> None:
        """The failure this was written for. Same trade, same sag, and the only
        difference is a field we never owned."""
        service, position, tick, closed = self.watcher(comment="[sl 4653.15]", in_journal=True)

        event = service._scalp_verdict(position, 0.45, 1.40 / 2.0, tick, 2.0)

        assert event is not None, "a rewritten comment silently demoted it to a swing trade"
        assert event.action == "SCALP_CLAIMED"
        assert closed

    def test_a_swing_trade_is_still_never_touched(self) -> None:
        """Both markers have to be absent, and both are."""
        service, position, tick, closed = self.watcher(comment="jarvis-exp-live", in_journal=False)

        assert service._scalp_verdict(position, 0.45, 1.40 / 2.0, tick, 2.0) is None
        assert closed == []

    def test_the_comment_alone_is_still_enough(self) -> None:
        """The cheap marker keeps working when the broker behaves, so the
        journal is never consulted for the ordinary case."""
        service, position, tick, _c = self.watcher(comment="jarvis-scalp", in_journal=False)

        event = service._scalp_verdict(position, 0.35, 1.40 / 2.0, tick, 2.0)

        assert event is not None and event.action == "SCALP_CLAIMED"

    def test_an_unreadable_journal_does_not_freeze_the_answer(self) -> None:
        """A lookup that raises must not be cached. One bad second would
        otherwise mean the position is never treated as a scalp again."""

        def boom(_ticket):  # type: ignore[no-untyped-def]
            raise RuntimeError("journal is locked")

        service, position, _tick, _closed = self.watcher(comment="", in_journal=False)
        service.journal = SimpleNamespace(trade_opened_by_section_six=boom)

        assert service._is_scalp(position) is False
        assert position.ticket not in service._scalp_tickets

        service.journal = SimpleNamespace(trade_opened_by_section_six=lambda _t: True)
        assert service._is_scalp(position) is True

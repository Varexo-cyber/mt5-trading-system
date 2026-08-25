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

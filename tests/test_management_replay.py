"""The management rules, driven over real-shaped price history.

Every other test of the management layer hands `PositionManager` a price and
asks what it does. That proves a rule in isolation and proves nothing about the
sequence — whether the stop the break-even rule set is the stop that actually
fills two bars later, whether a peak recorded at 11:04 is still remembered at
11:12, whether a rule fires at all before the broker's own stop takes the trade
away from it.

This drives the real manager, bar by bar, and the tests below are mostly about
the harness being honest rather than about the rules being right. A replay that
can see the future, or that reads the same stale bars for a thousand passes,
will happily report that everything works.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from backtesting.management_replay import (
    ReplayTrade,
    _ReplayBroker,
    frame_from_bars,
    replay_management,
)
from config.loader import load_settings
from core.instrument import InstrumentSpec
from core.types import Direction, Timeframe
from tests.fakes.fake_mt5 import eurusd_spec

OPENED = datetime(2026, 8, 4, 11, 0, tzinfo=UTC)
ENTRY = 1.10000
STOP = 1.09800  # 1R = 0.00200, twenty pips
TARGET = 1.10400  # 2R

#: Enough range for the health readers to have something to measure, far too
#: little to reach any level these tests place.
WICK = 0.00002


@pytest.fixture
def spec() -> InstrumentSpec:
    return InstrumentSpec.from_mt5(eurusd_spec())


@pytest.fixture
def settings():  # type: ignore[no-untyped-def]
    return load_settings(env_overrides=False)


def bars(closes, *, start=OPENED, highs=None, lows=None, opens=None):  # type: ignore[no-untyped-def]
    """An M1 series from a list of closes.

    Each bar opens where the last one closed unless told otherwise, so the
    series is continuous and nothing gaps by accident — a gap changes where a
    stop fills, and a test that gapped without meaning to would be measuring
    something other than what it says.
    """
    rows = []
    for index, close in enumerate(closes):
        opened = opens[index] if opens else (closes[index - 1] if index else close)
        rows.append(
            {
                "time": int((start + timedelta(minutes=index)).timestamp()),
                "open": opened,
                "high": highs[index] if highs else max(opened, close) + WICK,
                "low": lows[index] if lows else min(opened, close) - WICK,
                "close": close,
                "tick_volume": 1,
            }
        )
    return frame_from_bars(rows)


def at_r(r: float) -> float:
    return ENTRY + r * (ENTRY - STOP)


def trade(**overrides) -> ReplayTrade:  # type: ignore[no-untyped-def]
    base = {
        "symbol": "EURUSD",
        "direction": Direction.LONG,
        "entry": ENTRY,
        "stop": STOP,
        "target": TARGET,
        "volume": 0.10,
        "opened_at": OPENED,
    }
    base.update(overrides)
    return ReplayTrade(**base)  # type: ignore[arg-type]


# ------------------------------------------------------- the harness first ---


def test_the_broker_never_serves_a_bar_past_the_cursor(spec) -> None:  # type: ignore[no-untyped-def]
    """The one bug a replay must be structurally unable to have."""
    frame = bars([1.10000, 1.10010, 1.10020, 1.10030, 1.10040])
    broker = _ReplayBroker(spec_=spec, frame=frame, cursor=1)

    rows = broker.copy_rates("EURUSD", Timeframe.M1.mt5_value, 10)

    assert len(rows) == 2
    assert rows[-1]["close"] == pytest.approx(1.10010)


def test_a_coarse_bar_is_only_as_formed_as_the_clock_allows(spec) -> None:  # type: ignore[no-untyped-def]
    """Resampling the whole frame and then filtering by label would hand back a
    complete 11:00 to 11:05 bar at 11:02. That is lookahead in a convincing
    disguise, and it is the mistake this asserts against."""
    frame = bars([1.10000, 1.10010, 1.10020, 1.19000, 1.19000])
    broker = _ReplayBroker(spec_=spec, frame=frame, cursor=2)

    rows = broker.copy_rates("EURUSD", Timeframe.M5.mt5_value, 5)

    assert len(rows) == 1
    # The spike two minutes into the future is not in it.
    assert rows[0]["high"] == pytest.approx(1.10020 + WICK)


def test_the_manager_re_reads_the_market_on_every_bar(spec, settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The manager's bar cache expires on `time.monotonic`, which barely moves
    while a replay burns through a day in a second. Uncleared, every pass after
    the first would read the first bar's view of the world forever, and every
    rule that reads shape would be judging the market as it was at the open."""
    cursors: list[int] = []
    original = _ReplayBroker.copy_rates

    def counting(self, symbol, timeframe, count):  # type: ignore[no-untyped-def]
        cursors.append(self.cursor)
        return original(self, symbol, timeframe, count)

    monkeypatch.setattr(_ReplayBroker, "copy_rates", counting)
    frame = bars([at_r(index / 100.0) for index in range(30)])
    replay_management(trade(), frame, settings, spec, max_bars=5)

    # Not one read shared by five bars: the market is looked at again on each.
    assert len(set(cursors)) == 5


# ----------------------------------------------- the broker fills first ---


def test_the_stop_takes_the_trade_before_any_rule_sees_it(spec, settings) -> None:  # type: ignore[no-untyped-def]
    frame = bars([at_r(0.2), at_r(-0.5), at_r(-1.2)])

    outcome = replay_management(trade(), frame, settings, spec)

    assert outcome.exit_reason == "STOP"
    assert outcome.exit_r == pytest.approx(-1.0)


def test_a_gap_through_the_stop_fills_at_the_open(spec, settings) -> None:  # type: ignore[no-untyped-def]
    """Filling at the stop price when the market opened below it is how a
    backtest reports an exit that was never available."""
    gapped = at_r(-1.6)
    frame = bars([at_r(0.1), gapped], opens=[at_r(0.1), gapped], highs=[at_r(0.1), gapped])

    outcome = replay_management(trade(), frame, settings, spec)

    assert outcome.exit_reason == "STOP"
    assert outcome.exit_r == pytest.approx(-1.6)


def test_a_bar_that_touches_both_levels_is_read_as_the_stop(spec, settings) -> None:  # type: ignore[no-untyped-def]
    """Which came first is unknowable from a bar. The pessimistic reading is
    the only one that cannot flatter the result."""
    frame = bars([ENTRY], highs=[TARGET + 0.0001], lows=[STOP - 0.0001])

    outcome = replay_management(trade(), frame, settings, spec)

    assert outcome.exit_reason == "STOP"


def test_the_target_ends_it_when_only_the_target_is_touched(spec, settings) -> None:  # type: ignore[no-untyped-def]
    frame = bars([at_r(0.3), at_r(1.4)], highs=[at_r(0.3), TARGET + 0.0001])

    outcome = replay_management(trade(), frame, settings, spec)

    assert outcome.exit_reason == "TARGET"
    assert outcome.exit_r == pytest.approx(2.0)


# ------------------------------------------------- the loop feeds back ---


def test_a_stop_the_manager_moved_is_the_stop_that_fills(spec, settings) -> None:  # type: ignore[no-untyped-def]
    """The whole reason for driving the real manager rather than a copy.

    Break-even fires above 0.6R and the profit lock walks the stop up behind
    the peak. If the replay dropped either modification the trade would run all
    the way back to -1R; because it does not, the collapse on the last bar
    fills at the locked stop and the trade is a winner.
    """
    frame = bars([at_r(0.3), at_r(0.9), at_r(0.9), ENTRY - 0.0003])

    outcome = replay_management(trade(), frame, settings, spec)

    actions = [step.action for step in outcome.steps]
    assert "BREAK_EVEN" in actions
    assert "PROFIT_LOCK" in actions
    assert outcome.exit_reason == "STOP"
    # Half of the 0.9R peak, secured at the broker before price collapsed.
    assert outcome.exit_r == pytest.approx(0.45, abs=0.02)


def test_the_peak_is_remembered_across_bars(spec, settings) -> None:  # type: ignore[no-untyped-def]
    frame = bars([at_r(0.4), at_r(1.1), at_r(0.5), at_r(0.4)])

    outcome = replay_management(trade(), frame, settings, spec)

    assert outcome.peak_r == pytest.approx(1.1, abs=0.01)
    assert outcome.trough_r <= 0.0


def test_a_trade_that_stalls_near_its_peak_is_banked_by_a_rule(spec, settings) -> None:  # type: ignore[no-untyped-def]
    """The complaint that started all of this: a trade goes well into profit,
    sits there, and hands it all back. The peak-stall rule exists to stop that
    and had never once run against real price history."""
    closes = [at_r(0.3), at_r(0.7), at_r(0.9)] + [at_r(0.88)] * 12
    frame = bars(closes)

    outcome = replay_management(trade(), frame, settings, spec)

    assert outcome.exit_reason != "STOP"
    assert outcome.exit_r is not None and outcome.exit_r > 0.5
    assert [step.action for step in outcome.steps]


def test_nothing_acting_at_all_is_reported_as_the_finding(spec, settings) -> None:  # type: ignore[no-untyped-def]
    frame = bars([at_r(-0.2), at_r(-0.4), at_r(-1.1)])

    outcome = replay_management(trade(), frame, settings, spec)

    assert outcome.steps == ()
    assert "No rule ever acted" in outcome.render()


def test_the_replay_can_end_with_the_trade_still_open(spec, settings) -> None:  # type: ignore[no-untyped-def]
    frame = bars([at_r(0.1), at_r(0.2)])

    outcome = replay_management(trade(), frame, settings, spec)

    assert outcome.exit_r is None
    assert outcome.exit_reason == "never closed"
    assert "still open" in outcome.render()


def test_max_bars_stops_the_walk(spec, settings) -> None:  # type: ignore[no-untyped-def]
    frame = bars([at_r(0.1)] * 50)

    outcome = replay_management(trade(), frame, settings, spec, max_bars=7)

    assert outcome.bars == 7


# ------------------------------------------------------------- the costs ---


def charging(settings, per_side: float):  # type: ignore[no-untyped-def]
    risk = settings.risk.model_copy(update={"commission_per_lot_per_side": per_side})
    return settings.model_copy(update={"risk": risk})


def test_commission_comes_out_of_the_replayed_result(spec, settings) -> None:  # type: ignore[no-untyped-def]
    """The journal's `pnl_r` is money the broker moved, so it is already net.
    A gross replay compared against it would credit these rules with a cost
    they never paid, on exactly the narrow stops where this account bleeds."""
    frame = bars([at_r(0.3), at_r(1.4)], highs=[at_r(0.3), TARGET + 0.0001])

    free = replay_management(trade(), frame, charging(settings, 0.0), spec)
    charged = replay_management(trade(), frame, charging(settings, 2.75), spec)

    # 1R is 20 pips on EURUSD: EUR 200 a lot, against EUR 5.50 round trip.
    assert charged.commission_r == pytest.approx(5.50 / 200.0)
    assert free.exit_r == pytest.approx(2.0)
    assert charged.exit_r == pytest.approx(2.0 - 5.50 / 200.0)
    assert charged.gross_r == pytest.approx(2.0)


def test_a_narrow_stop_is_where_the_commission_hurts(spec, settings) -> None:  # type: ignore[no-untyped-def]
    """Five pips instead of twenty, and the same EUR 5.50 becomes four times
    the share of the risk. This is why it cannot be left out."""
    narrow = trade(stop=ENTRY - 0.00050, target=ENTRY + 0.00100)
    frame = bars([ENTRY + 0.0002, ENTRY + 0.0011], highs=[ENTRY + 0.0002, ENTRY + 0.0011])

    outcome = replay_management(narrow, frame, charging(settings, 2.75), spec)

    assert outcome.commission_r == pytest.approx(0.11, abs=0.005)
    assert "commission" in outcome.render()


def test_an_unfinished_trade_is_charged_nothing(spec, settings) -> None:  # type: ignore[no-untyped-def]
    """No exit, no result to deduct from. Reporting a cost against a trade
    that never closed would be inventing a loss."""
    frame = bars([at_r(0.1), at_r(0.2)])

    outcome = replay_management(trade(), frame, charging(settings, 2.75), spec)

    assert outcome.exit_r is None
    assert outcome.gross_r is None


# ------------------------------------------------------ what it reports ---


def test_the_improvement_is_measured_against_what_actually_happened(spec, settings) -> None:  # type: ignore[no-untyped-def]
    frame = bars([at_r(0.2), at_r(-1.2)])

    outcome = replay_management(
        trade(actual_pnl_r=-1.32, actual_exit_reason="BROKER_SL"), frame, settings, spec
    )

    assert outcome.improvement_r == pytest.approx(0.32)
    rendered = outcome.render()
    assert "actually got" in rendered
    assert "better" in rendered


def test_no_comparison_is_offered_when_there_is_nothing_to_compare_to(spec, settings) -> None:  # type: ignore[no-untyped-def]
    frame = bars([at_r(0.2), at_r(-1.2)])

    outcome = replay_management(trade(), frame, settings, spec)

    assert outcome.improvement_r is None
    assert "actually got" not in outcome.render()


def test_a_short_is_replayed_the_right_way_up(spec, settings) -> None:  # type: ignore[no-untyped-def]
    """A sign error here would report every short as its own mirror image, and
    the numbers would still look plausible."""
    short = trade(
        direction=Direction.SHORT,
        entry=ENTRY,
        stop=ENTRY + 0.00200,
        target=ENTRY - 0.00400,
    )
    frame = bars([ENTRY - 0.0005, ENTRY - 0.0045], lows=[ENTRY - 0.0005, ENTRY - 0.0045])

    outcome = replay_management(short, frame, settings, spec)

    assert outcome.exit_reason == "TARGET"
    assert outcome.exit_r == pytest.approx(2.0)

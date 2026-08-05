"""Naming the one gate that stopped a cycle.

Four separate times in one session the counts were already on screen and the
question "why is it not trading" still had to be asked out loud: monitor mode,
a circuit breaker disabled to zero that then blocked at 0.0%, the posture
throttle cutting sixteen setups to one, and a losing-streak halving that put
every trade under the broker's minimum lot.

A list of five numbers is not an answer. The largest one, named, is.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from runner.service import CycleSummary, JarvisRunner

NOW = datetime(2026, 8, 5, 13, 0, tzinfo=UTC)


def summary(opened: int = 0, analysed: int = 66) -> CycleSummary:
    return CycleSummary(
        started_at=NOW,
        finished_at=NOW,
        inspected=88,
        rejected=22,
        deep_analysed=analysed,
        candidates=10,
        trades_opened=opened,
        next_cursor=0,
        universe_size=88,
    )


def report(caplog, opened: int, reasons: dict[str, int]):  # type: ignore[no-untyped-def]
    runner = JarvisRunner.__new__(JarvisRunner)
    with caplog.at_level(logging.WARNING, logger="runner.service"):
        runner._report_dominant_blocker(summary(opened=opened), reasons)
    return [r for r in caplog.records if getattr(r, "event", "") == "no_trades_dominant_reason"]


def test_the_biggest_blocker_is_named(caplog) -> None:  # type: ignore[no-untyped-def]
    records = report(
        caplog,
        0,
        {"NO_SIGNAL": 100, "RR_BELOW_MINIMUM": 8, "TRADE_SKIPPED_UNDERCAPITALIZED": 3},
    )
    assert len(records) == 1
    assert records[0].reason == "NO_SIGNAL"
    assert records[0].count == 100


def test_a_cycle_that_traded_says_nothing(caplog) -> None:  # type: ignore[no-untyped-def]
    """The line exists to explain silence. A cycle that opened a trade has
    nothing to explain, and warning anyway would train people to ignore it."""
    assert report(caplog, 1, {"NO_SIGNAL": 100}) == []


def test_no_rejections_at_all_says_nothing(caplog) -> None:  # type: ignore[no-untyped-def]
    """Nothing reached deep analysis, so no gate is to blame — the prescan
    breakdown on the same log line is where that answer lives."""
    assert report(caplog, 0, {}) == []


def test_the_undercapitalized_case_is_reported(caplog) -> None:  # type: ignore[no-untyped-def]
    """The exact shape of tonight's blocker: a handful of candidates, all of
    them priced out by a halved risk budget."""
    records = report(caplog, 0, {"TRADE_SKIPPED_UNDERCAPITALIZED": 9, "NO_SIGNAL": 4})
    assert records[0].reason == "TRADE_SKIPPED_UNDERCAPITALIZED"


def test_the_message_carries_the_scale(caplog) -> None:  # type: ignore[no-untyped-def]
    """ "8 rejected" means nothing without knowing whether 8 or 800 were looked
    at."""
    records = report(caplog, 0, {"AI_VETO": 8})
    assert "of 66 analysed" in records[0].getMessage()

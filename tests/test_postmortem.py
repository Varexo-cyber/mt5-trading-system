from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from learning.postmortem import PostmortemAnalyzer


class JournalRows:
    @staticmethod
    def query(sql: str, _params):  # type: ignore[no-untyped-def]
        if "FROM shadow_trades" in sql:
            return [
                {"blocked_by": "AI_VETO", "pnl_r": 2.0},
                {"blocked_by": "AI_VETO", "pnl_r": -1.0},
                {"blocked_by": "NEWS_BLACKOUT", "pnl_r": -1.0},
            ]
        return []


def test_postmortem_measures_rejected_plans_by_gate() -> None:
    end = datetime(2026, 8, 9, tzinfo=UTC)
    result = PostmortemAnalyzer(JournalRows(), {}).analyze(end - timedelta(days=7), end)  # type: ignore[arg-type]

    ai = next(item for item in result.counterfactuals if item.blocked_by == "AI_VETO")
    assert ai.observations == 2
    assert ai.win_rate == pytest.approx(0.5)
    assert ai.expectancy_r == pytest.approx(0.5)

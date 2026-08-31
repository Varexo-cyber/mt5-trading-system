from __future__ import annotations

import pandas as pd

from scripts.audit_section_four_candidates import _resolve_one


def _frame(high: list[float], low: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"high": high, "low": low})


def test_close_entry_starts_resolving_on_the_next_bar() -> None:
    frame = _frame([102.0, 101.4], [98.0, 100.2])
    assert _resolve_one(
        frame,
        index=0,
        direction=1,
        entry=100.0,
        unit=1.0,
        ratio=0.5,
        same_bar=False,
    ) == (1, 1)


def test_same_bar_limit_ambiguity_is_a_loss() -> None:
    frame = _frame([101.0], [99.0])
    assert _resolve_one(
        frame,
        index=0,
        direction=1,
        entry=100.0,
        unit=1.0,
        ratio=0.5,
        same_bar=True,
    ) == (0, 0)

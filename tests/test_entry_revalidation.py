"""A paid AI approval is valid only for the state and price it reviewed."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

from advisory.providers import Advice
from analysis.confluence import TradeIdea
from config.loader import load_settings
from core.instrument import AssetClass
from core.types import AccountSnapshot, Direction, MarketContext, Position, Series, Tick, Timeframe
from risk.reasons import Reason
from runner.service import AnalysedCandidate, JarvisRunner

NOW = datetime(2026, 8, 10, 14, 0, tzinfo=UTC)


def account(login: int = 5049535) -> AccountSnapshot:
    return AccountSnapshot(
        login=login,
        server="Eightcap-Live",
        currency="EUR",
        balance=153.0,
        equity=153.0,
        margin=0.0,
        margin_free=153.0,
        margin_level=0.0,
        leverage=500,
        is_demo=False,
        taken_at=NOW,
    )


def context(*, bid: float, ask: float) -> MarketContext:
    closes = 100.0 + np.sin(np.arange(80) / 5.0) * 0.10
    opens = np.r_[closes[0], closes[:-1]]
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 0.25,
            "low": np.minimum(opens, closes) - 0.25,
            "close": closes,
            "tick_volume": 100,
            "spread": 2,
            "real_volume": 0,
        },
        index=pd.date_range("2026-08-10 07:20", periods=80, freq="5min", tz=UTC),
    )
    return MarketContext(
        "TEST",
        NOW,
        {Timeframe.M5: Series("TEST", Timeframe.M5, frame, NOW)},
        Tick("TEST", NOW, bid, ask),
    )


def position(ticket: int) -> Position:
    return Position(ticket, "OTHER", Direction.LONG, 0.01, 1.0, 0.9, 1.2, 0.0, 0.0, NOW)


def candidate(reviewed: MarketContext) -> AnalysedCandidate:
    idea = TradeIdea(
        "TEST",
        True,
        Direction.LONG,
        70.0,
        0.75,
        100.0,
        98.0,
        104.0,
        "test",
        (),
    )
    return AnalysedCandidate("TEST", "cycle-1", idea, reviewed)


def runner(
    *,
    fresh_context: MarketContext,
    fresh_account: AccountSnapshot,
    fresh_positions: list[Position],
) -> JarvisRunner:
    service = JarvisRunner.__new__(JarvisRunner)
    service.settings = load_settings(env_overrides=False)  # type: ignore[assignment]
    service.data = SimpleNamespace(  # type: ignore[assignment]
        get_context=lambda _symbol, force_refresh=False: fresh_context
    )
    service.broker = SimpleNamespace(  # type: ignore[assignment]
        account=lambda: fresh_account,
        positions=lambda: fresh_positions,
        spec=lambda _symbol: SimpleNamespace(asset_class=AssetClass.STOCK),
    )
    service.risk = SimpleNamespace(  # type: ignore[assignment]
        build_state=lambda _account, positions: SimpleNamespace(
            positions_in=lambda symbol: tuple(p for p in positions if p.symbol == symbol)
        )
    )
    return service


def approve() -> Advice:
    return Advice(True, 0.8, "sound", provider="fake", said_yes=True)


def test_account_switch_during_review_invalidates_the_approval() -> None:
    reviewed = context(bid=99.98, ask=100.0)
    service = runner(
        fresh_context=reviewed,
        fresh_account=account(login=999),
        fresh_positions=[],
    )

    verdict = service._revalidate_approved_entry(
        candidate=candidate(reviewed),
        reviewed_idea=candidate(reviewed).idea,
        reviewed_account=account(),
        reviewed_positions=[],
        was_addon=False,
        advice=approve(),
        latency_seconds=5.0,
    )

    assert not verdict.passed
    assert verdict.reason is Reason.ENTRY_STATE_CHANGED_DURING_REVIEW


def test_manual_position_change_during_review_invalidates_the_approval() -> None:
    reviewed = context(bid=99.98, ask=100.0)
    service = runner(
        fresh_context=reviewed,
        fresh_account=account(),
        fresh_positions=[position(2)],
    )

    verdict = service._revalidate_approved_entry(
        candidate=candidate(reviewed),
        reviewed_idea=candidate(reviewed).idea,
        reviewed_account=account(),
        reviewed_positions=[position(1)],
        was_addon=False,
        advice=approve(),
        latency_seconds=5.0,
    )

    assert not verdict.passed
    assert verdict.reason is Reason.ENTRY_STATE_CHANGED_DURING_REVIEW


def test_large_price_move_during_review_is_not_sent_with_old_sizing() -> None:
    reviewed = context(bid=99.98, ask=100.0)
    service = runner(
        fresh_context=context(bid=100.98, ask=101.0),
        fresh_account=account(),
        fresh_positions=[],
    )

    plan = candidate(reviewed)
    verdict = service._revalidate_approved_entry(
        candidate=plan,
        reviewed_idea=plan.idea,
        reviewed_account=account(),
        reviewed_positions=[],
        was_addon=False,
        advice=approve(),
        latency_seconds=20.0,
    )

    assert not verdict.passed
    assert verdict.reason is Reason.ENTRY_MOVED_DURING_REVIEW
    assert verdict.extra["review_drift"]["drift_atr"] > 0.25

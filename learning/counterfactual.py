"""Resolve blocked trade plans as passive counterfactual observations.

Nothing here can place, close or modify an order. It merely asks what would
have happened if a rejected setup had followed its original SL/TP plan. That
turns future filter tuning into an evidence question instead of an intuition.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from core.broker import Broker
from core.types import Direction, Timeframe
from infra.logging import get_logger
from journal.recorder import Recorder

log = get_logger(__name__)


def resolve_counterfactuals(
    recorder: Recorder,
    broker: Broker,
    now: datetime,
    *,
    max_age: timedelta = timedelta(hours=72),
    limit: int = 50,
) -> int:
    """Resolve a small queue of hypothetical trades from completed M15 bars."""
    resolved = 0
    for row in recorder.unresolved_shadow_trades(limit):
        opened = datetime.fromisoformat(str(row["opened_at"]))
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=UTC)
        try:
            raw = broker.copy_rates_range(str(row["symbol"]), Timeframe.M15.mt5_value, opened, now)
            frame = future_bars(raw, opened)
        except Exception:
            log.exception(
                "counterfactual price path unavailable",
                extra={"event": "counterfactual_data_error", "symbol": row["symbol"]},
            )
            continue
        if frame.empty:
            continue
        outcome, pnl_r = classify_path(
            frame,
            Direction[str(row["direction"])],
            float(row["entry_price"]),
            float(row["sl"]),
            float(row["tp"]),
            timed_out=now - opened >= max_age,
        )
        if outcome is None:
            continue
        recorder.resolve_shadow_trade(int(row["id"]), outcome=outcome, pnl_r=pnl_r)
        resolved += 1
    return resolved


def resolve_management_baselines(
    recorder: Recorder,
    broker: Broker,
    now: datetime,
    *,
    max_age: timedelta = timedelta(hours=72),
    limit: int = 50,
) -> int:
    """Compare closed managed trades with their untouched original SL/TP.

    This is deliberately passive. A health exit or Claude close is allowed to
    finish normally; afterwards this resolver keeps reading M15 history until
    the original stop, original target or the fixed horizon is reached.
    """
    resolved = 0
    for row in recorder.journal.unresolved_management_baselines(limit):
        entry = float(row["entry_price"] or 0.0)
        sl = float(row["sl"] or 0.0)
        tp = float(row["tp"] or 0.0)
        risk = abs(entry - sl)
        if min(entry, sl, tp, risk) <= 0:
            continue
        opened = datetime.fromisoformat(str(row["opened_at"]))
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=UTC)
        try:
            raw = broker.copy_rates_range(str(row["symbol"]), Timeframe.M15.mt5_value, opened, now)
            frame = future_bars(raw, opened)
        except Exception:
            log.exception(
                "management baseline price path unavailable",
                extra={"event": "management_baseline_data_error", "symbol": row["symbol"]},
            )
            continue
        if frame.empty:
            continue
        outcome, baseline_r = classify_path(
            frame,
            Direction[str(row["direction"])],
            entry,
            sl,
            tp,
            timed_out=now - opened >= max_age,
        )
        if outcome is None:
            continue
        recorded_r = row["pnl_r"]
        actual_r = (
            float(recorded_r)
            if recorded_r is not None
            else float(row["pnl_money"] or 0.0) / float(row["risk_money"] or 1.0)
        )
        recorder.record_management_baseline(
            trade_id=int(row["id"]),
            outcome=outcome,
            baseline_pnl_r=baseline_r,
            actual_pnl_r=actual_r,
            observed_at=now,
        )
        resolved += 1
    return resolved


def classify_path(
    frame: pd.DataFrame,
    direction: Direction,
    entry: float,
    sl: float,
    tp: float,
    *,
    timed_out: bool,
) -> tuple[str | None, float]:
    """Classify first-touch outcome with the same pessimism as the backtester."""
    for bar in frame.itertuples(index=False):
        low, high = float(bar.low), float(bar.high)
        sl_hit = low <= sl if direction is Direction.LONG else high >= sl
        tp_hit = high >= tp if direction is Direction.LONG else low <= tp
        if sl_hit and tp_hit:
            return "SL_FIRST_AMBIGUOUS", -1.0
        if sl_hit:
            return "SL", -1.0
        if tp_hit:
            risk = abs(entry - sl)
            return "TP", abs(tp - entry) / risk if risk > 0 else 0.0
    if not timed_out:
        return None, 0.0
    last = float(frame.iloc[-1]["close"])
    risk = abs(entry - sl)
    signed = (last - entry) * int(direction)
    return "TIMEOUT", signed / risk if risk > 0 else 0.0


def future_bars(raw: object, opened: datetime) -> pd.DataFrame:
    """Return only bars whose opening timestamp is strictly after the decision.

    MT5 includes the bar containing ``opened`` in a range request. Its high and
    low contain price action from before the hypothetical decision and may also
    include price action that was still forming at that instant. Treating it as
    executable evidence is a quiet look-ahead leak. Missing timestamps cannot
    prove ordering, so they fail closed as an empty frame.
    """
    frame = pd.DataFrame(raw)
    if frame.empty or "time" not in frame.columns:
        return frame.iloc[0:0].copy()
    moment = opened if opened.tzinfo is not None else opened.replace(tzinfo=UTC)
    timestamps = pd.to_datetime(frame["time"], unit="s", utc=True, errors="coerce")
    filtered = frame.loc[timestamps > moment].copy()
    return filtered.reset_index(drop=True)

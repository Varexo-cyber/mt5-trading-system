"""Resolve blocked trade plans as passive counterfactual observations.

Nothing here can place, close or modify an order. It merely asks what would
have happened if a rejected setup had followed its original SL/TP plan. That
turns future filter tuning into an evidence question instead of an intuition.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

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
    on_resolved: Callable[[Mapping[str, Any]], None] | None = None,
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
        if on_resolved is not None:
            payload = dict(row)
            payload.update({"resolved_at": now, "outcome": outcome, "pnl_r": pnl_r})
            try:
                on_resolved(payload)
            except Exception:
                log.exception(
                    "counterfactual outcome persistence failed",
                    extra={"event": "counterfactual_persistence_error", "symbol": row["symbol"]},
                )
        resolved += 1
    return resolved


def resolve_management_baselines(
    recorder: Recorder,
    broker: Broker,
    now: datetime,
    *,
    max_age: timedelta = timedelta(hours=72),
    limit: int = 50,
    on_resolved: Callable[[Mapping[str, Any]], None] | None = None,
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
        # A trade cannot have opened after now, and asking the terminal for a
        # backwards range gets a bare "Call failed" with a traceback -- every
        # fifteen minutes, forever, because the row never resolves and comes
        # back on the next pass.
        #
        # It happens when a position is read before the broker-to-UTC offset
        # has been learned: `positions()` normalises with an offset that starts
        # at zero, so a server clock three hours ahead writes a timestamp three
        # hours in the future. That is a bug at the writing end and this cannot
        # repair it -- inventing an opening time would put fiction into the
        # evidence -- but it can refuse to retry a question with no answer, and
        # say what is actually wrong instead of printing MT5's shrug.
        if opened >= now:
            log.warning(
                "management baseline skipped: the trade is recorded as opening in the future",
                extra={
                    "event": "management_baseline_future_open",
                    "symbol": row["symbol"],
                    "opened_at": opened.isoformat(),
                    "now": now.isoformat(),
                    "ahead_by_minutes": round((opened - now).total_seconds() / 60.0, 1),
                },
            )
            continue
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
        # HOW FAR IT WENT AFTER WE LET GO.
        #
        # Everything above reduces the replay to one binary answer: would the
        # untouched plan have hit its stop or its target. That grades the exit
        # and teaches nothing about the decision, because "we closed at +0.1R
        # and it ran to +2R" and "we closed at +0.1R and it collapsed" produce
        # the same verdict whenever the original stop was eventually hit.
        #
        # The excursion after the exit is the number that separates them, and
        # it is the one the owner actually asked for: keep watching the market
        # after the trade is over and see what it could have been.
        best_after, worst_after = _post_exit_excursion(frame, row, entry, risk)
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
        if on_resolved is not None:
            # Handed out rather than written here, for the same reason the
            # shadow resolver does it: this module reads bars and does
            # arithmetic, and knows nothing about where the answer should
            # durably live. The caller owns that.
            on_resolved(
                {
                    "trade_id": int(row["id"]),
                    "symbol": str(row["symbol"] or ""),
                    "direction": str(row["direction"] or ""),
                    # The rule that actually closed it. Grouping by this is the
                    # whole value: "our exits cost us" names nothing to stop
                    # doing, while AI_CLOSE and PEAK_STALL are different
                    # decisions with different records.
                    "exit_action": str(row["exit_reason"] or ""),
                    "baseline_pnl_r": baseline_r,
                    "actual_pnl_r": actual_r,
                    "after_exit_best_r": best_after,
                    "after_exit_worst_r": worst_after,
                }
            )
        resolved += 1
    return resolved


def _post_exit_excursion(
    frame: pd.DataFrame, row: Any, entry: float, risk: float
) -> tuple[float | None, float | None]:
    """Best and worst R the trade would have reached AFTER it was closed.

    Measured from the exit onward rather than from the open, because the
    question is about the decision to get out, not about the trade. A close at
    +0.1R that was followed by a run to +2R and one that was followed by a
    collapse are the same row in every other column here.

    Returns (None, None) when the exit time cannot be read or no bar follows
    it. An unmeasured excursion is not a zero one, and recording it as zero
    would quietly report every unresolvable trade as a perfect exit.
    """
    raw_closed = row["closed_at"] if "closed_at" in row.keys() else None  # noqa: SIM118
    if not raw_closed or risk <= 0:
        return None, None
    try:
        closed = datetime.fromisoformat(str(raw_closed))
    except (TypeError, ValueError):
        return None, None
    if closed.tzinfo is None:
        closed = closed.replace(tzinfo=UTC)
    # The time lives in a COLUMN, not in the index.
    #
    # `future_bars` ends with `.reset_index(drop=True)`, so every frame reaching
    # this function carries a plain integer RangeIndex and its timestamps stay
    # in `time` as epoch seconds. Comparing that index against a datetime threw
    # TypeError out of pandas, up through the resolver, and killed the runner at
    # launch — a passive learning pass taking down live trading.
    #
    # Read the timestamps the same way `future_bars` does, from the same column,
    # so the two can never disagree about what a bar's time is.
    if "time" not in frame.columns:
        return None, None
    timestamps = pd.to_datetime(frame["time"], unit="s", utc=True, errors="coerce")
    after = frame.loc[timestamps > closed]
    if after.empty:
        return None, None
    sign = int(Direction[str(row["direction"])])
    highs = (after["high"] - entry) * sign / risk
    lows = (after["low"] - entry) * sign / risk
    best = float(highs.max()) if sign > 0 else float((-(after["low"] - entry) / risk).max())
    worst = float(lows.min()) if sign > 0 else float((-(after["high"] - entry) / risk).min())
    return round(best, 3), round(worst, 3)


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

"""Everything an analyst would need, assembled from what the account did.

This is the half that decides whether the thinking is worth anything. A model
asked "is my trading system good" produces advice; the same model handed the
last twenty trades with their peaks, every gate that refused a setup and what
the rules were set to at the time produces an argument you can check.

So the work here is deterministic and testable, and the model gets none of it
until it is complete. Nothing in this module calls an API.

Three things are gathered, because the interesting questions all live between
them rather than inside any one:

1. **What happened** — closed trades, what each was worth at its best against
   what it returned, and which rule closed it.
2. **What was refused, and why** — the reason histogram. A system that opened
   two trades from three thousand decisions is described far better by the
   2,998 refusals than by the two.
3. **What the rules were** — the settings actually in force. Without them the
   first two are anecdotes; with them they are evidence about a configuration.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

#: Exits the system chose, as opposed to price reaching a level left at the
#: broker. The ratio between them is the single most diagnostic number here: a
#: book of nothing but broker stop-outs means no management rule ever acted.
OUR_EXITS = frozenset(
    {
        "PEAK_STALL",
        "GIVEBACK_EXIT",
        "PROFIT_LOCK",
        "TIME_EXIT",
        "HEALTH_EXIT",
        "HEALTH_SECURE",
        "SPREAD_SQUEEZE_EXIT",
        "EVENING_FLAT",
        "PARTIAL_CLOSE",
    }
)


@dataclass(frozen=True, slots=True)
class TradeFact:
    """One closed trade, reduced to what an argument can be built on."""

    symbol: str
    direction: str
    stop_pips: float | None
    peak_r: float | None
    pnl_r: float | None
    pnl_money: float | None
    exit_reason: str
    held_minutes: float | None

    @property
    def kept(self) -> float | None:
        """Share of the best moment that survived to the exit.

        The number neither `pnl_r` nor `peak_r` gives alone, and the one that
        separates a strategy that is wrong from a strategy that is right and
        hands it back. Those need completely different fixes.
        """
        if self.peak_r is None or self.pnl_r is None or self.peak_r <= 0:
            return None
        return self.pnl_r / self.peak_r

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "stop_pips": self.stop_pips,
            "peak_r": self.peak_r,
            "returned_r": self.pnl_r,
            "returned_money": self.pnl_money,
            "kept_share_of_peak": None if self.kept is None else round(self.kept, 3),
            "exit": self.exit_reason,
            "held_minutes": self.held_minutes,
            "exit_chosen_by_system": self.exit_reason in OUR_EXITS,
        }


@dataclass(frozen=True, slots=True)
class Evidence:
    """The complete case file handed to the analyst."""

    window_hours: float
    generated_at: datetime
    trades: list[TradeFact] = field(default_factory=list)
    refusals: dict[str, int] = field(default_factory=dict)
    open_positions: list[dict[str, Any]] = field(default_factory=list)
    rules: dict[str, Any] = field(default_factory=dict)
    account: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Arithmetic the model should not have to redo, and might get wrong.

        Anything a spreadsheet can settle is settled here. What is left for the
        model is the part a spreadsheet cannot do: deciding which of these
        numbers is the one that matters.
        """
        closed = [trade for trade in self.trades if trade.pnl_r is not None]
        ours = [trade for trade in closed if trade.exit_reason in OUR_EXITS]
        kept = [trade.kept for trade in closed if trade.kept is not None]
        wins = [trade for trade in closed if (trade.pnl_r or 0) > 0]
        overshot = [trade for trade in closed if (trade.pnl_r or 0) < -1.0]
        return {
            "closed_trades": len(closed),
            "net_r": round(sum(trade.pnl_r or 0.0 for trade in closed), 2),
            "net_money": round(sum(trade.pnl_money or 0.0 for trade in closed), 2),
            "wins": len(wins),
            "losses": len(closed) - len(wins),
            "exits_chosen_by_the_system": len(ours),
            "exits_left_to_the_broker": len(closed) - len(ours),
            # Below -1.00R a stop-out cost more than the risk model said it
            # would, which is a cost or execution problem rather than a
            # strategy one and is worth separating.
            "stop_outs_worse_than_minus_one_r": len(overshot),
            "median_kept_share_of_peak": (round(sorted(kept)[len(kept) // 2], 3) if kept else None),
            "total_decisions": sum(self.refusals.values()),
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            "window_hours": self.window_hours,
            "generated_at": self.generated_at.isoformat(),
            "account": self.account,
            "summary": self.summary(),
            "closed_trades": [trade.as_dict() for trade in self.trades],
            "why_everything_else_was_refused": self.refusals,
            "open_positions": self.open_positions,
            "rules_in_force": self.rules,
        }


def _minutes(opened: str | None, closed: str | None) -> float | None:
    if not opened or not closed:
        return None
    try:
        start = datetime.fromisoformat(opened)
        end = datetime.fromisoformat(closed)
    except ValueError:
        return None
    return round((end - start).total_seconds() / 60.0, 1)


def read_trades(db: sqlite3.Connection, since: datetime, limit: int) -> list[TradeFact]:
    rows = db.execute(
        "SELECT symbol, direction, sl_distance_pips, mfe_r, pnl_r, pnl_money, exit_reason, "
        "opened_at, closed_at FROM trades WHERE closed_at IS NOT NULL AND closed_at >= ? "
        "ORDER BY closed_at DESC LIMIT ?",
        (since.isoformat(), limit),
    ).fetchall()
    return [
        TradeFact(
            symbol=str(row["symbol"]),
            direction=str(row["direction"]),
            stop_pips=row["sl_distance_pips"],
            peak_r=row["mfe_r"],
            pnl_r=row["pnl_r"],
            pnl_money=row["pnl_money"],
            exit_reason=str(row["exit_reason"] or "unknown"),
            held_minutes=_minutes(row["opened_at"], row["closed_at"]),
        )
        for row in rows
    ]


def read_refusals(db: sqlite3.Connection, since: datetime) -> dict[str, int]:
    rows = db.execute(
        "SELECT reason, COUNT(*) AS n FROM analysis_cycles WHERE ts >= ? "
        "GROUP BY reason ORDER BY n DESC",
        (since.isoformat(),),
    ).fetchall()
    return {str(row["reason"]): int(row["n"]) for row in rows}


def read_open_positions(path: Path) -> list[dict[str, Any]]:
    """The fast layer's live read, including how old it is.

    The age matters more than the verdict. A health reading nine minutes old on
    a loop that runs every second is the finding, and an analyst that cannot
    see the timestamp would report the stale verdict as current.
    """
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = payload.get("positions")
    if not isinstance(entries, list):
        return []
    written = payload.get("recorded_at")
    age = None
    if isinstance(written, str):
        try:
            age = round((datetime.now(UTC) - datetime.fromisoformat(written)).total_seconds(), 1)
        except ValueError:
            age = None
    return [{**entry, "reading_age_seconds": age} for entry in entries if isinstance(entry, dict)]


def read_rules(settings: Any) -> dict[str, Any]:
    """The settings actually in force, not the file on disk.

    Deliberately a hand-picked list rather than the whole tree. The full dump
    is thousands of tokens of defaults nobody has ever changed, and burying the
    six numbers that decide behaviour inside it is how a reader — human or
    otherwise — ends up commenting on the wrong one.
    """
    risk, confluence = settings.risk, settings.analysis.confluence
    management, filters = settings.trade_management, settings.filters
    return {
        "risk_per_trade_pct": settings.effective_risk_pct(),
        "max_positions": settings.effective_max_positions(),
        "commission_per_lot_per_side": risk.commission_per_lot_per_side,
        "max_cost_share_of_risk": risk.max_cost_share_of_risk,
        "stop_slippage_pips": dict(risk.stop_slippage_pips),
        "min_risk_reward": risk.min_risk_reward,
        "score_threshold": confluence.score_threshold,
        "min_stop_atr": confluence.min_stop_atr,
        "require_entry_confirmation": confluence.require_entry_confirmation,
        "confirmation_max_adverse_atr": confluence.confirmation_max_adverse_atr,
        "break_even_at_r": management.break_even_at_r,
        "profit_lock_from_r": management.profit_lock_from_r,
        "profit_lock_fraction": management.profit_lock_fraction,
        "giveback_arm_r": management.giveback_arm_r,
        "peak_stall_minutes": management.peak_stall_minutes,
        "peak_stall_arm_r": management.peak_stall_arm_r,
        "time_exit_hours": management.time_exit_hours,
        "guard_interval_seconds": settings.system.guard_interval_seconds,
        "min_guard_seconds": settings.system.min_guard_seconds,
        "loop_interval_seconds": settings.system.loop_interval_seconds,
        "max_positions_per_currency": filters.currency_exposure.max_positions_per_currency,
        "evening_flat_from": filters.session.evening_flat_from,
        "tradable_sessions": list(filters.session.tradable_sessions),
    }


def gather(
    journal_path: Path,
    settings: Any,
    *,
    window_hours: float = 24.0,
    max_trades: int = 30,
    health_path: Path | None = None,
    account: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> Evidence:
    """Assemble the case file. Never raises on a missing or partial journal.

    A half-complete picture is still worth reading, and an analyst that cannot
    run because one table is empty is an analyst nobody keeps.
    """
    moment = now or datetime.now(UTC)
    since = moment - timedelta(hours=window_hours)
    trades: list[TradeFact] = []
    refusals: dict[str, int] = {}

    if journal_path.exists():
        db = sqlite3.connect(f"file:{journal_path}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            trades = read_trades(db, since, max_trades)
            refusals = read_refusals(db, since)
        except sqlite3.Error:
            pass
        finally:
            db.close()

    return Evidence(
        window_hours=window_hours,
        generated_at=moment,
        trades=trades,
        refusals=refusals,
        open_positions=read_open_positions(health_path) if health_path else [],
        rules=read_rules(settings),
        account=account or {},
    )


def dominant_refusal(refusals: dict[str, int]) -> tuple[str, int] | None:
    """The gate that stopped the most, ignoring the ones that mean success."""
    counted = Counter({k: v for k, v in refusals.items() if k != "OK"})
    return counted.most_common(1)[0] if counted else None

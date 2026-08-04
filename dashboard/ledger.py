"""Closed trades, and what today and this week actually did.

The deck could show what was open and what was being considered, and nothing at
all about what had already happened. "Which stop was hit, what did today cost,
where did we start" had no answer anywhere in the interface — the numbers only
existed in the broker's own history and in the journal, neither of which an
operator reads mid-session.

Everything here is derived from the journal rather than recomputed, so the deck
reports the trades the system actually recorded taking. A position closed by
hand in the terminal has no journal row and will not appear; that is honest
rather than complete, and the distinction matters when judging whether the
system is performing.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# The journal's own timestamp encoding. Formatting these here by hand meant
# `starting_equity` looked up a key the journal never wrote — the microseconds
# `iso` always emits are part of the stored string, so an exact-match lookup
# silently found nothing and the deck reported no opening balance, forever.
from journal.database import iso


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """One finished trade as the journal recorded it."""

    ticket: int | None
    symbol: str
    direction: str
    volume: float
    entry_price: float
    exit_price: float | None
    stop_loss: float
    take_profit: float
    pnl_money: float
    pnl_r: float | None
    exit_reason: str
    opened_at: str
    closed_at: str
    risk_money: float

    @property
    def won(self) -> bool:
        return self.pnl_money > 0

    @property
    def outcome(self) -> str:
        """Plain language for what ended it, from the recorded exit reason.

        The raw reason is a machine token that also drives `GROUP BY`, so it is
        not softened at the source; this is the reading of it, and unknown
        tokens fall through unchanged rather than being flattened into
        "closed", which would hide any exit path nobody has labelled yet.
        """
        reason = self.exit_reason.upper()
        if "SL" in reason or "STOP" in reason:
            return "stop loss geraakt"
        if "TP" in reason or "TAKE_PROFIT" in reason:
            return "target geraakt"
        if reason.startswith("AI_"):
            return "Claude sloot hem"
        if "TIME_EXIT" in reason:
            return "te lang niets gedaan"
        if "NEWS" in reason:
            return "nieuwsblokkade"
        if "PARTIAL" in reason:
            return "deels gesloten"
        if "ORPHAN" in reason or "EMERGENCY" in reason:
            return "noodsluiting"
        return self.exit_reason or "gesloten"


@dataclass(frozen=True, slots=True)
class PeriodSummary:
    """What a stretch of trading came to."""

    label: str
    started_at: datetime
    starting_equity: float | None
    trades: tuple[ClosedTrade, ...]

    @property
    def realised(self) -> float:
        return sum(trade.pnl_money for trade in self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for trade in self.trades if trade.won)

    @property
    def losses(self) -> int:
        return sum(1 for trade in self.trades if not trade.won)

    @property
    def win_rate(self) -> float:
        return self.wins / len(self.trades) if self.trades else 0.0

    @property
    def total_r(self) -> float:
        return sum(trade.pnl_r or 0.0 for trade in self.trades)

    @property
    def best(self) -> ClosedTrade | None:
        return max(self.trades, key=lambda t: t.pnl_money, default=None)

    @property
    def worst(self) -> ClosedTrade | None:
        return min(self.trades, key=lambda t: t.pnl_money, default=None)


def _row_to_trade(row: sqlite3.Row) -> ClosedTrade:
    risk = float(row["risk_money"] or 0.0)
    pnl = float(row["pnl_money"] or 0.0)
    recorded_r = row["pnl_r"]
    return ClosedTrade(
        ticket=int(row["ticket"]) if row["ticket"] is not None else None,
        symbol=str(row["symbol"]),
        direction=str(row["direction"]),
        volume=float(row["volume"]),
        entry_price=float(row["entry_price"]),
        exit_price=float(row["exit_price"]) if row["exit_price"] is not None else None,
        stop_loss=float(row["sl"]),
        take_profit=float(row["tp"]),
        pnl_money=pnl,
        # Derived when the column is empty, because R is the unit that stays
        # comparable as the account grows and a blank one makes a losing week
        # look like a quiet one.
        pnl_r=(float(recorded_r) if recorded_r is not None else (pnl / risk if risk > 0 else None)),
        exit_reason=str(row["exit_reason"] or ""),
        opened_at=str(row["opened_at"]),
        closed_at=str(row["closed_at"]),
        risk_money=risk,
    )


def closed_trades(path: Path, since: datetime) -> list[ClosedTrade]:
    """Every trade the journal closed since `since`, newest first."""
    if not path.exists():
        return []
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ticket, symbol, direction, volume, entry_price, exit_price, sl, tp, "
                "pnl_money, pnl_r, exit_reason, opened_at, closed_at, risk_money "
                "FROM trades WHERE closed_at IS NOT NULL AND closed_at >= ? "
                # ABANDONED rows are entries the broker refused. They never held
                # risk and counting them would report losses that never happened.
                "AND COALESCE(entry_state, 'OPEN') != 'ABANDONED' " "ORDER BY closed_at DESC",
                (iso(since),),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [_row_to_trade(row) for row in rows]


def starting_equity(path: Path, period: str, period_key: datetime) -> float | None:
    """The equity anchor a period began at, if one was written."""
    if not path.exists():
        return None
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "SELECT equity FROM equity_marks WHERE period = ? AND period_key = ?",
                (period, iso(period_key)),
            ).fetchone()
    except sqlite3.Error:
        return None
    return None if row is None else float(row[0])


def day_start(now: datetime, boundary: str = "21:00") -> datetime:
    """Start of the trading day containing `now`.

    The FX rollover, not midnight — matching `Journal.day_start`, because a
    summary measured against a different boundary than the risk limits would
    disagree with them for several hours every evening.
    """
    hour, minute = (int(part) for part in boundary.split(":"))
    moment = now.astimezone(UTC)
    candidate = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate > moment:
        candidate -= timedelta(days=1)
    return candidate


def week_start(now: datetime, boundary: str = "21:00") -> datetime:
    """Start of the trading week containing `now` (Sunday rollover)."""
    start = day_start(now, boundary)
    # Sunday evening opens the FX week; weekday() has Sunday at 6.
    return start - timedelta(days=(start.weekday() + 1) % 7)


def summarise(
    path: Path,
    label: str,
    since: datetime,
    period: str,
    period_key: datetime,
) -> PeriodSummary:
    return PeriodSummary(
        label=label,
        started_at=since,
        starting_equity=starting_equity(path, period, period_key),
        trades=tuple(closed_trades(path, since)),
    )


def as_rows(trades: Sequence[ClosedTrade]) -> list[dict[str, Any]]:
    """Table rows for the deck, in the operator's language."""
    return [
        {
            "Gesloten (UTC)": trade.closed_at,
            "Markt": trade.symbol,
            "Richting": trade.direction,
            "Lots": trade.volume,
            "Entry": trade.entry_price,
            "Exit": trade.exit_price,
            "Resultaat": trade.pnl_money,
            "R": trade.pnl_r,
            "Hoe het eindigde": trade.outcome,
            "Ticket": trade.ticket,
        }
        for trade in trades
    ]

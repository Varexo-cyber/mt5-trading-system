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

import json
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
        if "GIVEBACK" in reason:
            return "winst veiliggesteld"
        if "EVENING_FLAT" in reason:
            return "plat vóór de avondspread"
        if reason.startswith("HEALTH_"):
            return "systeem zag het draaien"
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


#: Machine actions in the operator's language. Anything not listed here shows
#: its own token rather than a shrug, so a newly added action is visible the
#: first time it fires instead of hiding behind a generic label.
ACTION_LABELS = {
    "BREAK_EVEN": "stop naar break-even",
    "ATR_TRAIL": "stop meegetrokken",
    "PARTIAL_CLOSE": "deels afgebouwd",
    "PARTIAL_CLOSE_RECOVERED": "deelafbouw hersteld",
    "GIVEBACK_EXIT": "winst veiliggesteld",
    "EVENING_FLAT": "plat vóór de avondspread",
    "HEALTH_TIGHTEN": "stop aangetrokken, trade verzwakt",
    "HEALTH_SECURE": "winst gepakt, trade draaide",
    "HEALTH_EXIT": "vroeg eruit, thesis gebroken",
    "TIME_EXIT": "gesloten, ging nergens heen",
    "ADOPTED": "positie geadopteerd na crash",
    "ORPHAN_CLOSE": "onbekende positie gesloten",
    "EMERGENCY_CLOSE": "noodsluiting, geen stop",
    "EMERGENCY_CLOSE_REJECTED": "noodsluiting geweigerd — positie nog open",
    "ORPHAN_CLOSE_REJECTED": "sluiting onbekende positie geweigerd",
    "TIME_EXIT_REJECTED": "tijdsluiting geweigerd — positie nog open",
    "AI_SUPERVISION_UNDER_THRESHOLD": "Claude-actie geweigerd: te onzeker",
}


#: How the fast layer's verdict reads on the deck, and how urgent it looks.
HEALTH_LABELS = {
    "healthy": ("gezond", "✅"),
    "watch": ("in de gaten houden", "👀"),
    "deteriorating": ("verslechtert", "⚠️"),
    "broken": ("thesis gebroken", "🚨"),
    "unknown": ("nog niet gelezen", "…"),
    # Not a reading — the absence of one. The trade is held by its broker stop
    # and nothing in the fast layer is watching it.
    "unmanaged": ("NIET BEHEERD — alleen de broker-stop houdt deze", "🚨"),
}


#: Beyond this the published read is old enough that showing it as "live" would
#: be a lie. The guard writes every second, so anything past half a minute means
#: the guard is not ticking — not that the market is quiet.
HEALTH_STALE_SECONDS = 30.0


def live_health(path: Path, now: datetime | None = None) -> dict[int, dict[str, Any]]:
    """The per-second read, keyed by ticket, as the runner last published it.

    Each entry carries `age_seconds`, measured from the file's own
    `recorded_at`. Without it a frozen file is indistinguishable from a live
    one, and the panel would keep showing a verdict from twenty minutes ago
    with a green tick beside it — the most confidently wrong state available.

    Returns an empty map when the file is missing or unreadable. That is a real
    answer too, and `health_caption` says which one it is.
    """
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("positions", [])
    except (OSError, ValueError):
        return {}
    if not isinstance(entries, list):
        return {}

    age = None
    written = payload.get("recorded_at")
    if isinstance(written, str):
        try:
            moment = datetime.fromisoformat(written)
        except ValueError:
            moment = None
        if moment is not None:
            age = max(0.0, ((now or datetime.now(UTC)) - moment).total_seconds())

    return {
        int(entry["ticket"]): {**entry, "age_seconds": age}
        for entry in entries
        if isinstance(entry, dict) and "ticket" in entry
    }


def health_caption(entry: dict[str, Any] | None, *, jarvis_running: bool = True) -> str:
    """One line an operator can read at a glance, signals included.

    A missing entry used to render as "geen live oordeel (draait Jarvis?)" for
    every possible cause at once. Seen on a live deck beside a mode tile reading
    EXPERIMENTAL_LIVE, that question answers itself and the operator is left
    concluding the panel is broken.

    The causes are different problems with different fixes, so they get
    different sentences: Jarvis stopped, the guard has stopped ticking, or the
    position is open but nothing is managing it.
    """
    if not entry:
        if not jarvis_running:
            return "…  geen live oordeel — Jarvis draait niet"
        return (
            "⚠️  Jarvis draait, maar deze positie staat niet in de laatste meting. "
            "Meestal betekent dat de guard-lus niet rondkomt — check de log op "
            "`guard_tick_failed`."
        )

    age = entry.get("age_seconds")
    stale = isinstance(age, int | float) and age > HEALTH_STALE_SECONDS

    label, icon = HEALTH_LABELS.get(str(entry.get("verdict")), (str(entry.get("verdict")), "•"))
    parts = [f"{icon}  **{label}**"]
    action = str(entry.get("action", "hold"))
    if action != "hold":
        parts.append(
            {"tighten": "stop aantrekken", "secure": "winst pakken", "exit": "eruit"}[action]
        )
    if entry.get("verdict") == "unmanaged" and entry.get("reason"):
        parts.append(str(entry["reason"]))
    for signal in entry.get("signals", []) or []:
        parts.append(str(signal.get("detail", "")))
    if stale:
        parts.append(f"⏳ deze meting is {float(age) / 60:.0f} min oud, niet live")
    return "  ·  ".join(part for part in parts if part)


def operation_label(running: bool, heartbeat: dict[str, Any]) -> str:
    """What the mode tile shows: OFF, STARTING, or the running operation.

    Three states, because the two obvious ones are not enough. The heartbeat is
    written when a cycle *completes*, and the first cycle on a cold cache pulls
    every timeframe for the whole catalogue — minutes in which the process is
    up, the pid file is on disk, and no heartbeat exists yet.

    Collapsing that into OFF is the worst available answer on a live account:
    it reads as "nothing is running" at precisely the moment an operator is
    checking whether the system they just started came up correctly, and the
    natural response to it — start it again — puts two instances on one account
    fighting over the same positions.
    """
    if not running:
        return "OFF"
    if not heartbeat:
        return "STARTING"
    return str(heartbeat.get("operation", "OFF")).upper()


def recent_management(path: Path, limit: int = 25) -> list[dict[str, Any]]:
    """What the mechanical layer has actually been doing, newest first.

    The guard runs about once a second and its whole justification is that it
    reacts between cycles. Without this the operator has no way to tell it apart
    from a system that is doing nothing at all.
    """
    if not path.exists():
        return []
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT a.ts, a.action, a.note, a.r_at_action, t.symbol, t.ticket "
                "FROM management_actions a JOIN trades t ON t.id = a.trade_id "
                "ORDER BY a.id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [
        {
            "Wanneer (UTC)": row["ts"],
            "Markt": row["symbol"],
            "Ticket": row["ticket"],
            "Wat": ACTION_LABELS.get(str(row["action"]), str(row["action"])),
            "R": row["r_at_action"],
            "Toelichting": row["note"],
        }
        for row in rows
    ]


def timeline_candidates(path: Path, limit: int = 50) -> list[dict[str, Any]]:
    """Recent real trades available for the operator's full timeline."""
    if not path.exists():
        return []
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ticket, symbol, direction, opened_at, closed_at FROM trades "
                "WHERE ticket IS NOT NULL AND COALESCE(entry_state, 'OPEN') != 'ABANDONED' "
                "ORDER BY opened_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [dict(row) for row in rows]


def trade_timeline(path: Path, ticket: int) -> dict[str, Any]:
    """Entry, every management action and final exit for one broker ticket."""
    if not path.exists():
        return {}
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            trade = conn.execute("SELECT * FROM trades WHERE ticket = ?", (ticket,)).fetchone()
            if trade is None:
                return {}
            actions = conn.execute(
                "SELECT ts, action, note, old_sl, new_sl, old_tp, new_tp, "
                "volume_closed, r_at_action FROM management_actions "
                "WHERE trade_id = ? ORDER BY ts, id",
                (int(trade["id"]),),
            ).fetchall()
    except sqlite3.Error:
        return {}
    events: list[dict[str, Any]] = [
        {
            "Tijd (UTC)": trade["opened_at"],
            "Bron": "systeem",
            "Gebeurtenis": "positie geopend",
            "R": None,
            "Uitleg": (
                f"{trade['direction']} {float(trade['volume']):g} lots @ "
                f"{float(trade['entry_price']):g}; oorspronkelijke SL {float(trade['sl']):g}, "
                f"TP {float(trade['tp']):g}, gepland {float(trade['planned_rr'] or 0):.2f}R"
            ),
        }
    ]
    events.extend(
        {
            "Tijd (UTC)": row["ts"],
            "Bron": "Claude" if str(row["action"]).startswith("AI_") else "mechanisch",
            "Gebeurtenis": ACTION_LABELS.get(str(row["action"]), str(row["action"])),
            "R": row["r_at_action"],
            "Uitleg": row["note"],
        }
        for row in actions
    )
    if trade["closed_at"] is not None:
        events.append(
            {
                "Tijd (UTC)": trade["closed_at"],
                "Bron": "broker/journaal",
                "Gebeurtenis": "positie definitief gesloten",
                "R": trade["pnl_r"],
                "Uitleg": (
                    f"{trade['exit_reason'] or 'gesloten'} @ {trade['exit_price']}; "
                    f"resultaat {float(trade['pnl_money'] or 0):+.2f}"
                ),
            }
        )
    return {"trade": dict(trade), "events": events}


def management_baseline_report(path: Path, limit: int = 30) -> dict[str, Any]:
    """Measured result of Jarvis management versus untouched original SL/TP."""
    if not path.exists():
        return {}
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            summary = conn.execute(
                "SELECT COUNT(*) n, SUM(actual_pnl_r) actual_r, "
                "SUM(baseline_pnl_r) baseline_r, SUM(lift_r) lift_r "
                "FROM management_baselines"
            ).fetchone()
            rows = conn.execute(
                "SELECT t.ticket, t.symbol, t.exit_reason, b.outcome, b.actual_pnl_r, "
                "b.baseline_pnl_r, b.lift_r, b.resolved_at FROM management_baselines b "
                "JOIN trades t ON t.id=b.trade_id ORDER BY b.resolved_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    except sqlite3.Error:
        return {}
    return {"summary": dict(summary) if summary else {}, "rows": [dict(row) for row in rows]}


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

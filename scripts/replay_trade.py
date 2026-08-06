"""Re-run today's management rules over a trade that already happened.

The journal records what the rules did. It cannot record what a *different*
set of rules would have done, and after every change to the management layer
that is the only question worth asking: would this have helped the trades we
actually took, or does it only look better on paper?

This answers it without spending anything. It pulls the M1 bars the trade lived
through from MT5, drives the real `PositionManager` over them minute by minute,
and prints what would have happened next to what did. No API calls, no orders,
nothing written anywhere — the journal is opened read-only.

    python scripts/replay_trade.py                # the last closed trade
    python scripts/replay_trade.py GBPAUD         # the last GBPAUD trade
    python scripts/replay_trade.py --ticket 123   # one exact trade
    python scripts/replay_trade.py --all 20       # the last twenty, as a table

Read the table, not any single row. One trade improving by 1.9R proves nothing;
twenty trades whose total is worse than what the account already did is a
finding, and the cheapest one available.
"""

from __future__ import annotations

import argparse
import contextlib
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtesting.management_replay import (
    BASE,
    ReplayOutcome,
    ReplayTrade,
    frame_from_bars,
    replay_management,
)
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from core.types import Direction

#: How long after the entry to keep feeding bars. Past the 24h time exit, so a
#: trade the rules would have held to the deadline still reaches it.
WINDOW_HOURS = 30.0

#: Closes a person made, recovered from MT5 deal history: CLIENT is the desktop
#: terminal, MOBILE the phone, WEB the browser.
#:
#: Shown, but kept out of the headline. That number claims the *rules* would
#: have done better or worse, and a trade closed by hand measures the hand.
#: Including one is how a tool built to check our reasoning ends up flattering
#: it — the first run of this script counted a phone close as +1.58R of credit
#: to rules that never got to act.
BY_HAND = frozenset({"BROKER_CLIENT", "BROKER_MOBILE", "BROKER_WEB"})

#: Exits the replay is structurally unable to reproduce, so it always looks
#: like it held on longer and did better or worse for a reason that is about
#: the harness rather than the rules. Bar history carries no spread series.
UNREPRODUCIBLE = frozenset({"SPREAD_SQUEEZE", "SPREAD_SQUEEZE_EXIT"})


def connect_journal(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def closed_trades(
    db: sqlite3.Connection, symbol: str, ticket: int, limit: int
) -> list[sqlite3.Row]:
    """Trades that finished, newest first. Symbol matches loosely: the broker
    calls it `GBPAUD.i` and nobody types the suffix."""
    if ticket:
        return db.execute("SELECT * FROM trades WHERE ticket = ?", (ticket,)).fetchall()
    where, params = "closed_at IS NOT NULL", []
    if symbol:
        where += " AND UPPER(symbol) LIKE ?"
        params.append(f"%{symbol.upper()}%")
    return db.execute(
        f"SELECT * FROM trades WHERE {where} ORDER BY closed_at DESC LIMIT ?",
        [*params, limit],
    ).fetchall()


def to_replay_trade(row: sqlite3.Row) -> ReplayTrade | None:
    """The journal row as something replayable, or None with a reason printed.

    Two rows in the live journal have `entry_price` at zero — the clobber bug
    that `_mark_open` now guards against. They cannot be replayed at all: with
    no entry there is no R, and every threshold in the system is written in R.
    """
    entry, stop = float(row["entry_price"] or 0.0), float(row["sl"] or 0.0)
    if entry <= 0 or stop <= 0 or entry == stop:
        print(f"  {row['symbol']}: no usable entry/stop on record, cannot replay")
        return None
    opened = datetime.fromisoformat(row["opened_at"])
    return ReplayTrade(
        symbol=str(row["symbol"]),
        direction=Direction.LONG if str(row["direction"]).upper() == "LONG" else Direction.SHORT,
        entry=entry,
        stop=stop,
        # A trade opened without a take-profit gets one far enough away that it
        # can never fill, so the replay measures the rules rather than a level
        # invented here.
        target=float(row["tp"] or 0.0) or entry + (entry - stop) * 100,
        volume=float(row["volume"] or 0.01),
        opened_at=opened if opened.tzinfo else opened.replace(tzinfo=UTC),
        actual_pnl_r=row["pnl_r"],
        actual_exit_reason=str(row["exit_reason"] or ""),
    )


def history(connector: MT5Connector, symbol: str, opened_at: datetime) -> object:
    """The M1 bars the trade lived through, one minute before it opened."""
    return connector.copy_rates_range(
        symbol,
        BASE.mt5_value,
        opened_at - timedelta(minutes=1),
        opened_at + timedelta(hours=WINDOW_HOURS),
    )


def summarise(outcomes: list[tuple[ReplayTrade, ReplayOutcome]], skipped: int) -> None:
    print()
    header = f"  {'symbol':<12}{'actual':>8}{'replay':>8}{'diff':>8}   "
    # Both exit reasons, side by side. Without the real one the interesting
    # rows are unreadable: a trade that scratched at -0.01R and replays to
    # TARGET means something completely different if the broker took it on a
    # break-even stop than if the evening wind-down closed it early.
    print(header + f"{'closed by (real)':<22}closed by (replay)")
    print("  " + "-" * 88)
    actual_total = replay_total = 0.0
    counted = by_hand = caveated = 0
    for trade, outcome in outcomes:
        real = trade.actual_exit_reason or "unknown"
        got = "—" if trade.actual_pnl_r is None else f"{trade.actual_pnl_r:+.2f}R"
        would = "open" if outcome.exit_r is None else f"{outcome.exit_r:+.2f}R"
        diff, note = "—", ""
        if real in BY_HAND:
            by_hand += 1
            note = "  (by hand, not counted)"
        elif outcome.improvement_r is not None:
            diff = f"{outcome.improvement_r:+.2f}R"
            actual_total += trade.actual_pnl_r or 0.0
            replay_total += outcome.exit_r or 0.0
            counted += 1
            if real in UNREPRODUCIBLE:
                caveated += 1
                note = "  *"
        print(
            f"  {trade.symbol:<12}{got:>8}{would:>8}{diff:>8}   "
            f"{real[:21]:<22}{outcome.exit_reason}{note}"
        )

    print("  " + "-" * 88)
    if not counted:
        print("  Nothing left to compare once hand-closed trades are set aside.\n")
        return
    change = replay_total - actual_total
    print(f"  {counted} trades: {actual_total:+.2f}R actually, {replay_total:+.2f}R replayed")
    if change > 0:
        print(f"  The rules as they stand now would have been {change:+.2f}R better.")
    elif change < 0:
        print(f"  The rules as they stand now would have been {change:+.2f}R WORSE.")
        print("  That is the useful answer. Do not deploy a change this says makes it worse.")
    else:
        print("  No difference. The changes did not touch these trades.")

    print()
    if by_hand:
        print(f"  {by_hand} trade(s) were closed by hand from a terminal or phone and are")
        print("  shown but not counted. That total is about the rules; a manual close")
        print("  measures the hand, and crediting the rules for it would be flattery.")
    if caveated:
        print("  * The real exit used the spread, which bar history does not carry, so")
        print("    the replay could never fire that rule. Those rows say more about the")
        print("    harness than about the rules.")
    if skipped:
        print(f"  {skipped} further trade(s) could not be replayed at all, so they are not in")
        print("  the total above. A sample that excludes what it could not read is not a")
        print("  random sample, and this one excludes the oldest trades.")
    if counted < 20:
        print(f"  {counted} trades is not a sample. It is a hint about the mechanics.")
    print("  Costs: commission is charged, spread is not, and bars are not ticks — a")
    print("  stop and a target in the same minute resolve stop-first. This measures")
    print("  whether the rules do what they should, not whether they make money.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", nargs="?", default="", help="e.g. GBPAUD (suffix optional)")
    parser.add_argument("--ticket", type=int, default=0, help="exact broker ticket")
    parser.add_argument("--all", type=int, default=0, help="replay the last N closed trades")
    parser.add_argument("--db", default="journal/trading.db", help="journal path")
    args = parser.parse_args(argv)

    path = ROOT / args.db
    if not path.exists():
        print(f"No journal at {path}.")
        return 1

    db = connect_journal(path)
    try:
        rows = closed_trades(db, args.symbol, args.ticket, args.all or 1)
    finally:
        db.close()
    if not rows:
        print("No closed trade matched. Try scripts/postmortem.py --list 20.")
        return 1

    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=False),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    try:
        connector.connect()
    except Exception as exc:  # noqa: BLE001 - the caller only needs the reason
        print(f"Could not connect to MT5: {type(exc).__name__}: {exc}")
        print("This needs the terminal running — it reads bar history, nothing else.")
        return 1

    outcomes: list[tuple[ReplayTrade, ReplayOutcome]] = []
    skipped = 0
    try:
        for row in rows:
            trade = to_replay_trade(row)
            if trade is None:
                skipped += 1
                continue
            spec = connector.spec(trade.symbol)
            frame = frame_from_bars(history(connector, trade.symbol, trade.opened_at))
            if frame.empty:
                print(f"  {trade.symbol}: no bars available for that window")
                skipped += 1
                continue
            outcome = replay_management(trade, frame, settings, spec, max_bars=len(frame))
            outcomes.append((trade, outcome))
            if not args.all:
                print(outcome.render())
    finally:
        with contextlib.suppress(Exception):
            connector.shutdown()

    if args.all:
        summarise(outcomes, skipped)
    return 0 if outcomes else 1


if __name__ == "__main__":
    raise SystemExit(main())

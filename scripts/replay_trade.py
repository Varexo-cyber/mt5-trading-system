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


def summarise(outcomes: list[tuple[ReplayTrade, ReplayOutcome]]) -> None:
    print()
    print(f"  {'symbol':<12}{'actual':>9}{'replay':>9}{'diff':>9}   what closed it")
    print("  " + "-" * 74)
    actual_total = replay_total = 0.0
    counted = 0
    for trade, outcome in outcomes:
        got = "—" if trade.actual_pnl_r is None else f"{trade.actual_pnl_r:+.2f}R"
        would = "open" if outcome.exit_r is None else f"{outcome.exit_r:+.2f}R"
        diff = "—"
        if outcome.improvement_r is not None:
            diff = f"{outcome.improvement_r:+.2f}R"
            actual_total += trade.actual_pnl_r or 0.0
            replay_total += outcome.exit_r or 0.0
            counted += 1
        print(f"  {trade.symbol:<12}{got:>9}{would:>9}{diff:>9}   {outcome.exit_reason}")

    if not counted:
        print("\n  Nothing comparable. No trade had both a recorded result and a replay.\n")
        return
    change = replay_total - actual_total
    print("  " + "-" * 74)
    print(f"  {counted} trades: {actual_total:+.2f}R actually, {replay_total:+.2f}R replayed")
    if change > 0:
        print(f"  The rules as they stand now would have been {change:+.2f}R better.")
    elif change < 0:
        print(f"  The rules as they stand now would have been {change:+.2f}R WORSE.")
        print("  That is the useful answer. Do not deploy a change this says makes it worse.")
    else:
        print("  No difference. The changes did not touch these trades.")
    print("\n  Bars are not ticks: a stop and a target in the same minute resolve")
    print("  stop-first here, and there is no spread series, so this is the")
    print("  pessimistic reading of the mechanics — not a profitability estimate.\n")


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
    try:
        for row in rows:
            trade = to_replay_trade(row)
            if trade is None:
                continue
            spec = connector.spec(trade.symbol)
            frame = frame_from_bars(history(connector, trade.symbol, trade.opened_at))
            if frame.empty:
                print(f"  {trade.symbol}: no bars available for that window")
                continue
            outcome = replay_management(trade, frame, settings, spec, max_bars=len(frame))
            outcomes.append((trade, outcome))
            if not args.all:
                print(outcome.render())
    finally:
        with contextlib.suppress(Exception):
            connector.shutdown()

    if args.all:
        summarise(outcomes)
    return 0 if outcomes else 1


if __name__ == "__main__":
    raise SystemExit(main())

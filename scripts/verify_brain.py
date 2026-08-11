"""Create the Postgres schema and prove the brain can read and write.

Nothing in `brain/` has been run against a real database. The environment it
was written in blocks port 5432 and blocks HTTPS to the Neon host, so the DSN
is untested from here and the SQL has never been executed by Postgres.

Run this on the VPS once, before trusting the layer:

    .venv-live\\Scripts\\python.exe scripts\\verify_brain.py
    .venv-live\\Scripts\\python.exe scripts\\verify_brain.py --stats

It applies `brain/schema.sql` (safe to repeat), writes a marked test row into
every table, reads it back through the same queries the runner uses, and then
deletes what it wrote. Nothing else in the database is touched.

The connection string comes from NEON_DATABASE_URL in config/.env, which is
gitignored. It is never printed here — only the host, so a typo is visible
without the password reaching a terminal, a screenshot or a log.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain import DSN_ENV, Brain, build_brain
from brain.store import MIN_TRADES_TO_LEARN
from config.loader import load_credentials, load_settings


def redacted(dsn: str) -> str:
    """Host and database only. The password never reaches a terminal."""
    parts = urlsplit(dsn)
    return f"{parts.hostname or '?'}{parts.path or ''}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", action="store_true", help="also show what is stored")
    parser.add_argument("--keep", action="store_true", help="leave the test rows behind")
    args = parser.parse_args(argv)

    # Populates the process environment from config/.env, the same way the
    # runner does. required=False so a machine with no MT5 credentials can
    # still check the database.
    load_credentials(required=False)

    brain = build_brain(account="verify")
    if not isinstance(brain, Brain):
        print()
        print(f"  No {DSN_ENV} in config/.env, or psycopg is not installed.")
        print()
        print("  Add this line to config/.env (which is gitignored):")
        print(f"      {DSN_ENV}=postgresql://USER:PASSWORD@HOST/neondb?sslmode=require")
        print()
        print('  Then:  .venv-live\\Scripts\\pip.exe install "psycopg[binary]"')
        print()
        return 1

    print()
    print(f"  BRAIN — {redacted(brain.dsn)}")
    print("  " + "-" * 70)

    if not brain.migrate():
        print(f"  schema     FAILED   {brain.status.last_error}")
        print()
        print("  Nothing was written. Check the DSN, and that the role may CREATE TABLE.")
        print()
        return 1
    print("  schema     ok       every table, index and view is in place")

    now = datetime.now(UTC)
    decision_id = brain.record_decision(
        decided_at=now,
        symbol="__VERIFY__",
        reason="NO_SIGNAL",
        mode="verify",
        detail="written by verify_brain.py",
        equity=100.0,
        filters={"spread_pips": 1.2, "headline_count": 0},
    )
    print(f"  decision   {'ok' if decision_id else 'FAILED':<9}id={decision_id}")

    trade_id = brain.record_trade_opened(
        ticket=999_999_999,
        decision_id=decision_id,
        symbol="__VERIFY__",
        direction="LONG",
        volume=0.01,
        opened_at=now - timedelta(minutes=30),
        entry=1.1000,
        stop_loss=1.0990,
        take_profit=1.1015,
        risk_money=1.0,
    )
    print(f"  trade      {'ok' if trade_id else 'FAILED':<9}id={trade_id}")

    if trade_id:
        brain.record_trade_event(
            trade_id=trade_id,
            happened_at=now - timedelta(minutes=20),
            action="BREAK_EVEN",
            reason="written by verify_brain.py",
            r_at_action=0.31,
        )
        print("  event      ok")

    brain.record_trade_closed(
        ticket=999_999_999,
        closed_at=now,
        exit_price=1.1008,
        exit_reason="PROFIT_BANKED",
        pnl_money=0.72,
        pnl_r=0.72,
        mfe_r=0.9,
        mae_r=-0.2,
    )
    brain.record_lessons(
        ["verify_brain.py wrote this lesson and is about to delete it"],
        learned_at=now,
        symbol="__VERIFY__",
        direction="LONG",
        pnl_r=0.72,
        trade_id=trade_id,
    )
    brain.record_supervision(
        trade_id=trade_id,
        asked_at=now - timedelta(minutes=10),
        symbol="__VERIFY__",
        action="hold",
        confidence=0.62,
        reasoning="written by verify_brain.py",
        r_at_the_time=0.55,
        applied=False,
        latency_ms=1234,
        model="verify",
    )
    brain.record_counterfactual(
        symbol="__VERIFY__",
        direction="SHORT",
        blocked_by="AI_VETO",
        opened_at=now - timedelta(minutes=45),
        entry=1.1000,
        stop_loss=1.1010,
        take_profit=1.0980,
        resolved_at=now,
        outcome="TP",
        pnl_r=2.0,
    )
    print("  close      ok")
    print("  lesson     ok")
    print("  supervision ok")
    print("  counterfactual ok")

    # Read back through the exact queries the runner uses, not through
    # hand-written SELECTs. A schema that stores fine and cannot be read by the
    # briefing is the failure worth catching here.
    scoreboard = brain.scoreboard(symbol="__VERIFY__")
    lessons = brain.lessons(symbol="__VERIFY__")
    gates = brain.gate_scoreboard(symbol="__VERIFY__")
    readback_ok = bool(scoreboard and lessons and gates)
    print(
        f"  read back  {'ok' if readback_ok else 'FAILED':<9}"
        f"{len(scoreboard)} scoreline(s), {len(lessons)} lesson(s), "
        f"{len(gates)} gate scoreline(s)"
    )
    for line in scoreboard:
        print(f"               {line.summary()}")
    for lesson in lessons:
        print(f"               {lesson.summary()}")
    if not any(line.blocked_by == "AI_VETO" for line in gates):
        print("  counterfactual read back FAILED")
        readback_ok = False

    if not args.keep:
        brain._run("DELETE FROM supervisions WHERE symbol = '__VERIFY__'")
        brain._run("DELETE FROM lessons WHERE symbol = '__VERIFY__'")
        brain._run("DELETE FROM counterfactuals WHERE symbol = '__VERIFY__'")
        brain._run("DELETE FROM trades WHERE symbol = '__VERIFY__'")
        brain._run("DELETE FROM decisions WHERE symbol = '__VERIFY__'")
        print("  cleanup    ok       test rows removed")

    if args.stats:
        print()
        print("  WHAT IS STORED")
        print("  " + "-" * 70)
        for table in (
            "decisions",
            "counterfactuals",
            "trades",
            "trade_events",
            "supervisions",
            "lessons",
            "headlines",
        ):
            row = brain._run(f"SELECT COUNT(*) FROM {table}", fetch="one")
            print(f"  {table:<16}{row[0] if row else '?':>10}")

        # Row counts prove the memory is filling. They say nothing about
        # whether anything reads it, and a database nothing reads is
        # decoration. These two are the only places a stored trade changes what
        # the system does, so they are the only honest answer to "is it
        # learning" -- printed as the numbers they actually produce.
        print()
        print("  WHAT IT HAS LEARNED, AND WHAT THAT CHANGES")
        print("  " + "-" * 70)

        settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
        configured = settings.trade_management.bank_at_r
        learned = brain.learned_bank_threshold()
        if learned is None:
            need = MIN_TRADES_TO_LEARN
            print(f"  banking     not yet    needs {need} closed trades in a band; using")
            print(f"                         the configured {configured:.2f}R")
        else:
            effective = min(configured, learned)
            print(f"  banking     {learned:.2f}R      its own history says take profit here")
            print(f"                         configured {configured:.2f}R, so it now banks at ")
            print(f"                         {effective:.2f}R -- earlier, never later")

        learning = settings.learning
        estimates = brain.edge_calibrations(
            minimum_trades=learning.selection_min_trades,
            shrinkage_trades=learning.selection_shrinkage_trades,
            points_per_r=learning.selection_points_per_r,
            modifier_cap=learning.selection_modifier_cap,
        )
        if not estimates:
            print(f"  ranking     not yet    no segment has {learning.selection_min_trades} trades")
        else:
            print("  ranking     ok         ordering only; cannot approve a refused setup")
            for item in sorted(estimates, key=lambda e: e.modifier)[:8]:
                where = " ".join(part for part in item.key if part != "*") or "everything"
                print(
                    f"               {where:<26}{item.trades:>4} trades  "
                    f"{item.mean_r:+.2f}R  ->  {item.modifier:+.2f} punten"
                )

    print("  " + "-" * 70)
    print(f"  {brain.status.summary()}")
    print()
    if brain.status.failures or not readback_ok:
        print("  Some writes failed. The trading system would carry on without")
        print("  them — the brain is memory, not a risk control — but the memory")
        print("  would be incomplete. Fix this before relying on it.")
        print()
        return 1
    print("  Ready. The runner will start writing on its next cycle.")
    print()
    brain.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

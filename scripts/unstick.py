"""Show, and optionally clear, whatever is holding new risk halted.

WHY THIS EXISTS. `reconcile` halts new risk when the journal says a trade is
live and the broker neither holds it nor can produce its closing deal. That is
correct for a few seconds and used to be permanent: the row stayed open, every
cycle re-detected it, every cycle halted again, and a restart rebuilt the halt
within one cycle. The runner now writes such a trade off after fifteen minutes
by itself, so this is for the operator who does not want to wait, and for
seeing exactly WHICH row is doing it before touching anything.

    python scripts/unstick.py            # report only, changes nothing
    python scripts/unstick.py --settle   # close the unrecoverable rows

`--settle` writes `closed_at` and a reason and leaves `pnl_money` NULL rather
than inventing a figure. Every report that sums realised money skips it, which
is the honest treatment: the trade happened and what it returned is not known.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from core.clock import LiveClock
from journal.database import Journal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settle",
        action="store_true",
        help="close the rows listed. Without it this only reports.",
    )
    parser.add_argument("--db", default=None, help="journal path")
    args = parser.parse_args(argv)

    settings = load_settings(DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml")
    path = Path(args.db) if args.db else Path(settings.journal.database_path)
    if not path.exists():
        print(f"  No journal at {path}.")
        return 1

    journal = Journal(path, LiveClock()).open()
    try:
        rows = journal.open_trades()
        if not rows:
            print("\n  The journal holds no open trades. Nothing here is halting anything.\n")
            return 0

        print(f"\n  {len(rows)} trade(s) the journal believes are still live at the broker:\n")
        print(f"  {'id':>6}{'ticket':>12}  {'symbol':<12}{'opened':<22}{'volume':>9}")
        print("  " + "-" * 64)
        for row in rows:
            print(
                f"  {int(row['id']):>6}{int(row['ticket'] or 0):>12}  "
                f"{row['symbol']!s:<12}{str(row['opened_at'])[:19]:<22}"
                f"{float(row['volume']):>9.2f}"
            )

        print(
            "\n  If MT5 shows no position for one of these tickets, that row is what\n"
            "  halts new risk. The runner writes it off by itself after fifteen\n"
            "  minutes; --settle does it now.\n"
        )
        if not args.settle:
            print("  Nothing changed. Re-run with --settle to close them.\n")
            return 0

        for row in rows:
            journal.settle_unrecoverable(
                int(row["id"]),
                "settled by the operator: the broker no longer holds this position and no "
                "closing deal was recoverable; profit and loss unknown",
            )
            print(f"  settled trade {int(row['id'])} (ticket {int(row['ticket'] or 0)})")
        print(
            "\n  Done. Restart Jarvis: the halt lives in memory and the rows that\n"
            "  rebuilt it are now closed.\n"
        )
        return 0
    finally:
        journal.close()


if __name__ == "__main__":
    raise SystemExit(main())

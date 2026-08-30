"""The verdict for a run that already happened, straight from its CSV.

    python scripts/verdict_from_csv.py runtime/history-impulse_retest.csv

WHY THIS EXISTS. `_live_config_report` returns early when the section it was
asked about is not on `live_enabled_modules`, so a run measuring a SHADOWED
section prints its trades and none of the judgement: no sigma, no monthly
table, no break-even comparison, no tick boxes. The 180-day impulse_retest run
came back with +20.60 R and EUR +126.73 and no way to tell whether either
number meant anything.

Every decision is already in the CSV. Re-running a twenty-minute measurement to
get arithmetic that could be done on the file it already wrote is not a
reasonable thing to ask of anyone.

This reads that file and applies exactly the same bars, deliberately importing
them from `dry_run_sections` rather than restating them -- a second copy of a
threshold is how the two would eventually disagree.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dry_run_sections import Decision, _is_this_real


def rows_from(path: Path) -> tuple[list[Decision], bool]:
    """Every TRADE row, and whether the file carries a managed column.

    The CSV names its columns `result_r_fixed_stop` and `managed_r_LIVE`
    precisely so this cannot total the wrong one. An older file without the
    managed column is read on the fixed stop and says so.
    """
    trades: list[Decision] = []
    managed = False
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("outcome") != "TRADE":
                continue
            managed = managed or bool(row.get("managed_r_LIVE"))
            trades.append(
                Decision(
                    when=datetime.fromisoformat(row["when"]),
                    symbol=row["symbol"],
                    module=row["module"],
                    outcome="TRADE",
                    result_r=_number(row.get("result_r_fixed_stop")),
                    pnl_money=_number(row.get("pnl_money_fixed_stop")),
                    managed_r=_number(row.get("managed_r_LIVE")),
                    managed_money=_number(row.get("managed_money_LIVE")),
                    pass_key=(row["module"], ""),
                )
            )
    return trades, managed


def _number(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="a CSV written by dry_run_sections")
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="judge the fixed stop instead of the break-even exit the account runs",
    )
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"no such file: {path}")

    trades, has_managed = rows_from(path)
    if not trades:
        raise SystemExit(f"{path} has no TRADE rows")
    managed = has_managed and not args.fixed

    print(f"\n{'=' * 78}")
    print(f"VERDICT FROM {path.name}")
    print(f"{'=' * 78}")
    print(f"  {len(trades)} trades")
    if not has_managed:
        print("  this file has no managed column, so the FIXED stop is judged")

    sections = sorted({d.module for d in trades})
    keys = [(name, "") for name in sections]
    for name in sections:
        rows = [d for d in trades if d.module == name]
        closed = [d for d in rows if (d.managed_r if managed else d.result_r) is not None]
        if not closed:
            continue
        total = sum((d.managed_r if managed else d.result_r) or 0.0 for d in closed)
        money = sum((d.managed_money if managed else d.pnl_money) or 0.0 for d in closed)
        won = sum(1 for d in closed if ((d.managed_r if managed else d.result_r) or 0) > 0)
        print(
            f"  {name:<18} {len(closed):>5} trades  {won / len(closed):>5.1%} win"
            f"  {total:+8.2f} R  EUR {money:+9.2f}"
        )

    _is_this_real(trades, keys, managed)


if __name__ == "__main__":
    main()

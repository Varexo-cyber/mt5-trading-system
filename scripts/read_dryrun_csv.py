"""Re-read a finished dry run's CSV instead of running it again.

    python scripts/read_dryrun_csv.py runtime/sectie10.csv

WHY THIS EXISTS. `dry_run_sections` writes every decision to CSV and then
throws the objects away, so a question the report did not happen to answer
costs another full run -- and a 180-day M1 replay over five metals is two
hours of walking bars that have not changed.

The 4 September run is the case in point. It took two hours, and the one
table that answers "do the four new crosses carry their own weight, or is
XAUUSD carrying them" had been added to the reporter that morning, after
that run's launcher was already on disk. The bars were the same bars. The
answer was sitting in the CSV.

So: same tables, no terminal, no fetch, no MT5. Anything computed from
`when`, `symbol`, `module` and the two R columns can be asked here.

WHICH R COLUMN. `result_r_fixed_stop` is the broker stop and target the
research measured; `managed_r_LIVE` is the same trade under the break-even
rule the account actually runs. A fixed-exit family has them equal by
construction. `--exit` picks; the default is the live one, because that is
the book the account holds.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Row:
    __slots__ = ("fixed", "live", "module", "outcome", "symbol", "when")

    def __init__(self, raw: dict[str, str]) -> None:
        self.when = datetime.fromisoformat(raw["when"])
        self.symbol = raw["symbol"]
        self.module = raw["module"]
        self.outcome = raw["outcome"]
        self.fixed = _number(raw.get("result_r_fixed_stop"))
        self.live = _number(raw.get("managed_r_LIVE"))


def _number(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load(path: Path, exit_column: str) -> list[Row]:
    """Taken trades with a resolved result, in time order.

    A row whose outcome is not TRADE is a refusal, and a TRADE with an empty
    R never reached a barrier before the window ended. Neither is a result,
    and averaging them in as zero is how an unfinished trade becomes a
    scratch.
    """
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = [Row(raw) for raw in csv.DictReader(handle)]
    picked = [
        row for row in rows if row.outcome == "TRADE" and getattr(row, exit_column) is not None
    ]
    picked.sort(key=lambda row: row.when)
    return picked


def _line(label: str, values: list[float], early: list[float], late: list[float]) -> None:
    if not values:
        return
    won = sum(1 for v in values if v > 0)
    print(
        f"    {label:<14}{len(values):>7}{sum(values):>+10.2f}"
        f"{sum(values) / len(values):>+11.3f}{sum(early):>+9.2f}"
        f"{sum(late):>+9.2f}{won / len(values):>7.1%}"
    )


def _table(title: str, groups: dict[str, list[Row]], split: datetime, column: str) -> None:
    print(f"\n  {title}")
    print(
        f"    {'':<14}{'trades':>7}{'total R':>10}{'per trade':>11}"
        f"{'early':>9}{'late':>9}{'hit':>7}"
    )
    ranked = sorted(
        groups.items(),
        key=lambda kv: -sum(getattr(r, column) or 0.0 for r in kv[1]),
    )
    for name, got in ranked:
        values = [getattr(r, column) for r in got]
        _line(
            str(name),
            [v for v in values if v is not None],
            [getattr(r, column) for r in got if r.when < split],
            [getattr(r, column) for r in got if r.when >= split],
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--exit",
        choices=("live", "fixed"),
        default="live",
        help="managed_r_LIVE (default, what the account runs) or result_r_fixed_stop",
    )
    parser.add_argument(
        "--module",
        default="",
        help="only this section, e.g. section_ten_gold_m1",
    )
    args = parser.parse_args(argv)

    column = "live" if args.exit == "live" else "fixed"
    if not args.csv.exists():
        raise SystemExit(f"{args.csv} does not exist. Run sectie10.cmd or dryrun-live.cmd first.")
    rows = load(args.csv, column)
    if args.module:
        rows = [row for row in rows if row.module == args.module]
    if not rows:
        raise SystemExit(
            f"{args.csv} holds no resolved trades"
            + (f" for {args.module}" if args.module else "")
            + ". Every row is a refusal, or the run took none."
        )

    order = [row.when for row in rows]
    split = order[int(len(order) * 0.6)]
    total = sum(getattr(r, column) or 0.0 for r in rows)
    print("=" * 74)
    print(f"  {args.csv}  --  {len(rows)} resolved trades, exit = {args.exit}")
    print(f"  {order[0]:%Y-%m-%d} to {order[-1]:%Y-%m-%d}, early/late split at {split:%Y-%m-%d}")
    print(f"  total {total:+.2f} R over {len({r.symbol for r in rows})} markets")
    print("=" * 74)

    by_module: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_module[row.module].append(row)
    _table("PER SECTION", by_module, split, column)

    for module, got in sorted(by_module.items()):
        markets: dict[str, list[Row]] = defaultdict(list)
        for row in got:
            markets[row.symbol].append(row)
        if len(markets) < 2:
            continue
        _table(f"{module} — PER MARKET", markets, split, column)
        losing = [
            name
            for name, rows_here in markets.items()
            if sum(getattr(r, column) or 0.0 for r in rows_here) < 0
        ]
        if losing:
            names = ", ".join(sorted(losing))
            print(f"    {len(losing)} of {len(markets)} markets negative: {names}")
        print(
            "    A section is only as widened as its worst market. One symbol\n"
            "    carrying four is not a wider section, it is the old one plus noise."
        )

        hours: dict[str, list[Row]] = defaultdict(list)
        for row in got:
            hours[f"{row.when.hour:02d}:00"].append(row)
        _table(f"{module} — PER UTC HOUR", dict(sorted(hours.items())), split, column)
        both = sorted(
            name
            for name, rows_here in hours.items()
            if sum(getattr(r, column) or 0.0 for r in rows_here if r.when < split) < 0
            and sum(getattr(r, column) or 0.0 for r in rows_here if r.when >= split) < 0
        )
        if both:
            print(f"    negative in BOTH halves: {', '.join(both)}")
            print(
                "    Read that as a candidate and not a decision. Cutting the worst\n"
                "    hours out of the sample that named them finds a bad block in any\n"
                "    sequence -- both halves agreeing is better evidence than a total,\n"
                "    and it is still the same 180 days."
            )

    print(
        "\n  Every number here is the same run, re-read. Nothing was re-measured,\n"
        "  so it cannot disagree with the report that produced this file."
    )


if __name__ == "__main__":
    main()

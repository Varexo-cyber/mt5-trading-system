"""Explain which live section contributed to each red replay month.

This deliberately reads an existing ``hoeveel*.csv``.  It does not fetch bars,
rerun signals, or change the live configuration.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Trade:
    month: str
    module: str
    result_r: float
    pnl_eur: float


def _number(row: dict[str, str], preferred: str, fallback: str) -> float:
    value = row.get(preferred, "").strip() or row.get(fallback, "").strip()
    return float(value) if value else 0.0


def load_trades(path: Path) -> list[Trade]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"when", "module", "outcome", "result_r_fixed_stop", "pnl_money_fixed_stop"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: ontbrekende kolommen: {', '.join(sorted(missing))}")
        trades = []
        for row in reader:
            if row["outcome"].strip().upper() != "TRADE":
                continue
            month = datetime.fromisoformat(row["when"].strip()).strftime("%Y-%m")
            trades.append(
                Trade(
                    month=month,
                    module=row["module"].strip() or "ONBEKEND",
                    result_r=_number(row, "managed_r_LIVE", "result_r_fixed_stop"),
                    pnl_eur=_number(row, "managed_money_LIVE", "pnl_money_fixed_stop"),
                )
            )
    return trades


def aggregate(
    trades: list[Trade],
) -> tuple[list[str], list[str], dict[tuple[str, str], list[float]]]:
    months = sorted({trade.month for trade in trades})
    modules = sorted({trade.module for trade in trades})
    totals: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for trade in trades:
        values = totals[(trade.month, trade.module)]
        values[0] += 1
        values[1] += trade.result_r
        values[2] += trade.pnl_eur
    return months, modules, totals


def month_totals(
    months: list[str], modules: list[str], totals: dict[tuple[str, str], list[float]]
) -> dict[str, tuple[float, float]]:
    return {
        month: (
            sum(totals[(month, module)][1] for module in modules),
            sum(totals[(month, module)][2] for module in modules),
        )
        for month in months
    }


def render(path: Path, trades: list[Trade]) -> str:
    months, modules, totals = aggregate(trades)
    per_month = month_totals(months, modules, totals)
    red = [month for month in months if per_month[month][0] < 0]
    lines = [
        "=" * 78,
        f"MAANDCHECK — {path}",
        f"{len(trades)} uitgevoerde trades | {len(red)}/{len(months)} rode maanden op R",
        "LIVE beheerde uitkomst waar aanwezig; anders vaste SL/TP.",
        "",
        "MAANDTOTAAL",
        "maand     status   totaal R    EUR",
    ]
    for month in months:
        result_r, eur = per_month[month]
        status = "ROOD" if result_r < 0 else "groen"
        lines.append(f"{month}   {status:5}  {result_r:+9.2f}  {eur:+10.2f}")

    lines.extend(["", "RODE MAANDEN — laagste bijdrage staat eerst"])
    if not red:
        lines.append("geen rode maanden")
    for month in red:
        result_r, eur = per_month[month]
        lines.append(f"\n{month}  totaal {result_r:+.2f} R / EUR {eur:+.2f}")
        ranked = sorted(modules, key=lambda module: totals[(month, module)][1])
        for module in ranked:
            count, result_r, eur = totals[(month, module)]
            if count:
                lines.append(
                    f"  {module:32} {int(count):4} trades  "
                    f"{result_r:+8.2f} R  EUR {eur:+9.2f}"
                )

    baseline_r = sum(value[0] for value in per_month.values())
    lines.extend(
        [
            "",
            "STATISCHE WEG-LAAT-CHECK",
            "Zelfde geaccepteerde trades blijven staan: diagnose, geen exacte replay.",
            "sectie                           totaal R   verschil   rode maanden",
            f"NIETS WEGLATEN                  {baseline_r:+9.2f}             "
            f"{len(red):2}/{len(months)}",
        ]
    )
    candidates = []
    for module in modules:
        remaining = {
            month: per_month[month][0] - totals[(month, module)][1] for month in months
        }
        result_r = sum(remaining.values())
        red_count = sum(value < 0 for value in remaining.values())
        candidates.append((red_count, -result_r, module, result_r))
    for red_count, _negative_r, module, result_r in sorted(candidates):
        lines.append(
            f"zonder {module:26} {result_r:+9.2f}  {result_r - baseline_r:+9.2f}  "
            f"{red_count:2}/{len(months)}"
        )

    lines.extend(
        [
            "",
            "LET OP: weglaten kan live andere slots en volgende setups vrijmaken.",
            "Vind hiermee de verdachte sectie; bevestig die daarna met een echte replay.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="bestaande hoeveel-CSV('s)")
    args = parser.parse_args()
    failed = False
    for path in args.csv:
        if not path.exists():
            print(f"OVERGESLAGEN: {path} bestaat niet.")
            failed = True
            continue
        try:
            trades = load_trades(path)
            print(render(path, trades))
        except (OSError, ValueError) as exc:
            print(f"FOUT: {exc}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

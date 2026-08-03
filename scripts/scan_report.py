"""Explain what the scanner did to all 847 markets, from its own record.

A cycle line says "185 spread". That is a number, not an answer: it cannot tell
you whether the limits are calibrated or whether they are throwing away half the
catalogue for no reason. The scanner already stores the measured spread, quote
age and asset class of every instrument it looked at, so the answer exists —
this just reads it back.

    python scripts/scan_report.py
    python scripts/scan_report.py --stage spread     # only spread rejections
    python scripts/scan_report.py --limit 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.loader import load_settings
from monitoring.scan_activity import read_scan_activity


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", help="only this rejection stage, e.g. spread or quote")
    parser.add_argument("--limit", type=int, default=25, help="rows in the detail table")
    args = parser.parse_args(argv)

    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    state = read_scan_activity(ROOT / "runtime" / "scan_activity.json")
    symbols: dict[str, dict[str, object]] = dict(state.get("symbols", {}))  # type: ignore[arg-type]
    if not symbols:
        print("No scan activity recorded yet. Let jarvis.py finish one cycle first.")
        return 1

    print(f"\n{len(symbols)} markets, last seen {state.get('updated_at', 'unknown')}\n")

    stages: dict[str, int] = {}
    for row in symbols.values():
        key = "ELIGIBLE" if row.get("status") != "REJECTED" else str(row.get("stage", "?"))
        stages[key] = stages.get(key, 0) + 1
    print("outcome by stage")
    for stage, count in sorted(stages.items(), key=lambda kv: -kv[1]):
        print(f"  {count:5}  {stage}")

    # The spread table is the point of this script. A cap is only meaningful
    # next to the spreads it is judging.
    caps = settings.filters.spread.max_spread_bps
    print("\nspread against the configured cap, by asset class")
    print(
        f"  {'class':10} {'cap':>7} {'passing':>8} {'blocked':>8} {'median':>9} {'p10 blocked':>12}"
    )
    for asset in sorted({str(r.get("asset_class", "?")) for r in symbols.values()}):
        rows = [r for r in symbols.values() if r.get("asset_class") == asset]
        spreads = [float(r["spread_bps"]) for r in rows if r.get("spread_bps") is not None]
        blocked = [
            float(r["spread_bps"])
            for r in rows
            if r.get("stage") == "spread" and r.get("spread_bps") is not None
        ]
        passing = len(rows) - len(blocked)
        cap = caps.get(asset)
        cap_text = f"{cap:.1f}" if cap is not None else "none"
        median = f"{_percentile(spreads, 0.5):.1f}" if spreads else "-"
        # The cheapest blocked market: how far the cap is from letting anything
        # through. A p10 just above the cap means the cap is nearly right; ten
        # times the cap means it is measuring the wrong thing.
        near = f"{_percentile(blocked, 0.10):.1f}" if blocked else "-"
        print(f"  {asset:10} {cap_text:>7} {passing:>8} {len(blocked):>8} {median:>9} {near:>12}")

    wanted = args.stage
    detail = [
        r
        for r in symbols.values()
        if r.get("status") == "REJECTED" and (wanted is None or r.get("stage") == wanted)
    ]
    detail.sort(key=lambda r: float(r.get("spread_bps") or 0.0))
    if detail:
        title = f"closest to passing ({wanted})" if wanted else "closest to passing"
        print(f"\n{title}")
        print(f"  {'symbol':14} {'class':10} {'stage':14} {'spread bps':>11} {'quote age s':>12}")
        for row in detail[: args.limit]:
            spread = row.get("spread_bps")
            age = row.get("quote_age_seconds")
            print(
                f"  {row.get('symbol')!s:14} {row.get('asset_class')!s:10} "
                f"{row.get('stage')!s:14} "
                f"{('-' if spread is None else f'{float(spread):.2f}'):>11} "
                f"{('-' if age is None else f'{float(age):.0f}'):>12}"
            )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

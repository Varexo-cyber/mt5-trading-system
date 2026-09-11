"""Find repeatable properties of winners and losers in an existing replay CSV."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from scripts.month_attribution import load_trades


@dataclass(frozen=True)
class Row:
    when: datetime
    module: str
    direction: str
    result_r: float


def load_rows(path: Path) -> list[Row]:
    # Reuse the encoding-tolerant loader for results, then read the few
    # categorical fields through the same byte-decoding contract.
    import csv
    import io

    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    rows = []
    with io.StringIO(text, newline="") as handle:
        for item in csv.DictReader(handle):
            if item.get("outcome", "").strip().upper() != "TRADE":
                continue
            value = item.get("managed_r_LIVE", "").strip()
            value = value or item.get("result_r_fixed_stop", "").strip()
            if not value:
                continue
            rows.append(
                Row(
                    when=datetime.fromisoformat(item["when"].strip()),
                    module=item["module"].strip(),
                    direction=item.get("direction", "").strip() or "ONBEKEND",
                    result_r=float(value),
                )
            )
    return rows


def _stats(rows: list[Row]) -> tuple[int, float, float, float, float]:
    wins = [row.result_r for row in rows if row.result_r > 0]
    losses = [row.result_r for row in rows if row.result_r <= 0]
    return (
        len(rows),
        sum(row.result_r for row in rows),
        len(wins) / len(rows) * 100 if rows else 0.0,
        sum(wins) / len(wins) if wins else 0.0,
        sum(losses) / len(losses) if losses else 0.0,
    )


def _group(rows: list[Row], key) -> dict[str, list[Row]]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return groups


def _table(title: str, groups: dict[str, list[Row]]) -> list[str]:
    lines = ["", title, "groep                         n    totaal R   win%   gem win  gem verlies"]
    for name, rows in sorted(groups.items()):
        count, total, win_rate, average_win, average_loss = _stats(rows)
        lines.append(
            f"{name:28} {count:4}  {total:+9.2f}  {win_rate:5.1f}  "
            f"{average_win:+7.2f}  {average_loss:+11.2f}"
        )
    return lines


def render(path: Path, rows: list[Row]) -> str:
    if not rows:
        return f"{path}: geen uitgevoerde trades"
    ordered = sorted(rows, key=lambda row: row.when)
    midpoint = ordered[len(ordered) // 2].when
    lines = ["=" * 84, f"WINST/VERLIES-ANALYSE — {path}"]
    lines += _table("PER SECTIE", _group(ordered, lambda row: row.module))
    lines += _table(
        "PER SECTIE + RICHTING",
        _group(ordered, lambda row: f"{row.module} {row.direction}"),
    )
    lines += _table(
        "PER SECTIE + UTC-UUR",
        _group(ordered, lambda row: f"{row.module} {row.when.hour:02d}:00"),
    )
    lines += _table(
        "PER SECTIE + WEEKDAG",
        _group(ordered, lambda row: f"{row.module} {row.when.strftime('%a')}"),
    )

    candidates = []
    hour_groups = _group(ordered, lambda row: f"{row.module}|{row.when.hour:02d}:00")
    for name, group in hour_groups.items():
        early = [row for row in group if row.when < midpoint]
        late = [row for row in group if row.when >= midpoint]
        early_r = sum(row.result_r for row in early)
        late_r = sum(row.result_r for row in late)
        total = early_r + late_r
        if len(group) >= 20 and early and late and early_r < 0 and late_r < 0:
            candidates.append((total, name, len(group), early_r, late_r))
    lines.extend(
        [
            "",
            "HERHAALDE VERLIESUREN — alleen minimaal 20 trades en BEIDE tijdhelften negatief",
            "sectie | uur                     n    totaal R    vroeg R     laat R",
        ]
    )
    if candidates:
        for total, name, count, early_r, late_r in sorted(candidates):
            lines.append(
                f"{name:31} {count:4}  {total:+9.2f}  {early_r:+9.2f}  {late_r:+9.2f}"
            )
    else:
        lines.append("geen robuust herhaald verliesuur gevonden")

    daily = _group(ordered, lambda row: row.when.date().isoformat())
    together = []
    for day, group in daily.items():
        modules = {row.module for row in group}
        total = sum(row.result_r for row in group)
        if len(modules) > 1 and total < 0:
            together.append((total, day, len(group)))
    lines.extend(["", f"RODE DAGEN MET BEIDE SECTIES: {len(together)}"])
    for total, day, count in sorted(together)[:15]:
        lines.append(f"  {day}  {count:3} trades  {total:+.2f} R")
    lines.append("\nDit is kandidaatselectie, geen toestemming om live filters te wijzigen.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    # Fail with the same useful schema/encoding errors as the month reader.
    load_trades(args.csv)
    print(render(args.csv, load_rows(args.csv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

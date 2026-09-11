"""Find repeatable properties of winners and losers in an existing replay CSV."""

from __future__ import annotations

import argparse
import random
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
    mae_r: float = 0.0
    mfe_r: float = 0.0
    spread_pct: float = 0.0
    atr_regime: float = 0.0
    h1_distance: float = 0.0
    h4_distance: float = 0.0
    breakout_atr: float = 0.0
    retest_bars: int = 0
    breakout_body: float = 0.0
    breakout_wick: float = 0.0
    breakout_volume: float = 0.0
    story: str = ""
    context_votes: int = 0
    confirmation: str = ""
    risk_atr: float = 0.0
    available_reward_r: float = 0.0


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
                    mae_r=float(item.get("mae_r") or 0),
                    mfe_r=float(item.get("mfe_r") or 0),
                    spread_pct=float(item.get("spread_stop_pct") or 0),
                    atr_regime=float(item.get("atr_regime_ratio") or 0),
                    h1_distance=float(item.get("h1_distance_atr") or 0),
                    h4_distance=float(item.get("h4_distance_atr") or 0),
                    breakout_atr=float(item.get("breakout_atr") or 0),
                    retest_bars=int(float(item.get("retest_bars") or 0)),
                    breakout_body=float(item.get("breakout_body_atr") or 0),
                    breakout_wick=float(item.get("breakout_wick_share") or 0),
                    breakout_volume=float(item.get("breakout_volume_ratio") or 0),
                    story=item.get("story", "").strip(),
                    context_votes=int(float(item.get("context_votes") or 0)),
                    confirmation=item.get("confirmation", "").strip(),
                    risk_atr=float(item.get("risk_atr") or 0),
                    available_reward_r=float(item.get("available_reward_r") or 0),
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


def _tertiles(rows: list[Row], value, labels: tuple[str, str, str]):
    """Split the trades by their own distribution instead of config floors.

    Section ten already refuses breakouts below its configured minimum. Fixed
    buckets around that same minimum therefore put every accepted trade in one
    row and look diagnostic while measuring nothing.
    """
    finite = sorted(float(value(row)) for row in rows)
    if len(finite) < 3 or finite[0] == finite[-1]:
        return {f"GEEN ONDERSCHEID ({finite[0]:.3f})" if finite else "GEEN DATA": rows}
    low = finite[(len(finite) - 1) // 3]
    high = finite[(2 * (len(finite) - 1)) // 3]
    named = (
        f"{labels[0]} <= {low:.3f}",
        f"{labels[1]} {low:.3f}..{high:.3f}",
        f"{labels[2]} > {high:.3f}",
    )
    if low == high:
        return _group(rows, lambda row: named[0] if value(row) <= low else named[2])
    return _group(
        rows,
        lambda row: named[0] if value(row) <= low else named[1] if value(row) <= high else named[2],
    )


def _diagnostic_table(title: str, rows: list[Row], value, labels) -> list[str]:
    groups = _tertiles(rows, value, labels)
    lines = _table(title, groups)
    if len(groups) == 1:
        lines.append("WAARSCHUWING: dit veld onderscheidt deze trades niet; trek hier geen filterconclusie uit.")
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
    stories = [row for row in ordered if row.story]
    if stories:
        lines += _table(
            "PER MARKTVERHAAL",
            _group(stories, lambda row: f"{row.module} {row.story}"),
        )
        lines += _table(
            "PER CONTEXTSTERKTE",
            _group(stories, lambda row: f"{row.module} {row.context_votes} HTF stemmen"),
        )
        lines += _diagnostic_table(
            "PER STRUCTURELE STOPAFSTAND IN ATR",
            stories,
            lambda row: row.risk_atr,
            ("kleinste 1/3", "middelste 1/3", "grootste 1/3"),
        )
        lines += _diagnostic_table(
            "PER BESCHIKBARE RUIMTE TOT OBJECTIEF IN R",
            stories,
            lambda row: row.available_reward_r,
            ("minste 1/3", "middelste 1/3", "meeste 1/3"),
        )
    lines += _table(
        "PER SECTIE + UTC-UUR",
        _group(ordered, lambda row: f"{row.module} {row.when.hour:02d}:00"),
    )
    lines += _table(
        "PER SECTIE + WEEKDAG",
        _group(ordered, lambda row: f"{row.module} {row.when.strftime('%a')}"),
    )
    lines += _table(
        "PER ATR/VOLATILITEITSREGIME",
        _group(ordered, lambda r: f"{r.module} " + ("compressie <0.8x" if r.atr_regime < .8 else "expansie >1.2x" if r.atr_regime > 1.2 else "normaal")),
    )
    lines += _table(
        "PER SPREAD ALS PERCENTAGE VAN STOP",
        _group(ordered, lambda r: f"{r.module} " + ("<=5%" if r.spread_pct <= 5 else "5-10%" if r.spread_pct <= 10 else ">10%")),
    )
    lines += _table(
        "PER H1/H4 TRENDRELATIE",
        _group(ordered, lambda r: f"{r.module} " + ("beide mee" if (r.h1_distance > 0) == (r.direction == "LONG") and (r.h4_distance > 0) == (r.direction == "LONG") else "gemengd/tegen")),
    )
    s10 = [r for r in ordered if "section_ten" in r.module]
    if s10:
        lines += _diagnostic_table("S10 BREAKOUTGROOTTE (TERTIELEN)", s10, lambda r: r.breakout_atr, ("kleinste 1/3", "middelste 1/3", "grootste 1/3"))
        lines += _table("S10 SNELHEID VAN RETEST", _group(s10, lambda r: "1 bar" if r.retest_bars <= 1 else "2-3 bars" if r.retest_bars <= 3 else "4+ bars"))
        lines += _diagnostic_table("S10 BREAKOUT BODY/DISPLACEMENT (TERTIELEN)", s10, lambda r: r.breakout_body, ("zwakste 1/3", "middelste 1/3", "sterkste 1/3"))
        lines += _diagnostic_table("S10 BREAKOUT WICK (TERTIELEN)", s10, lambda r: r.breakout_wick, ("kleinste wick 1/3", "middelste wick 1/3", "grootste wick 1/3"))
        lines += _diagnostic_table("S10 BREAKOUT VOLUME (TERTIELEN)", s10, lambda r: r.breakout_volume, ("laagste 1/3", "middelste 1/3", "hoogste 1/3"))

    lines.extend(["", "MAE/MFE — excursie voor de werkelijk gebruikte exit", "groep                         n   gem MAE   gem MFE  verliezers die eerst +0.50R zagen"])
    for module, group in sorted(_group(ordered, lambda r: r.module).items()):
        losers = [r for r in group if r.result_r <= 0]
        rescued = sum(r.mfe_r >= .5 for r in losers)
        lines.append(f"{module:28} {len(group):4}  {sum(r.mae_r for r in group)/len(group):8.2f}  {sum(r.mfe_r for r in group)/len(group):8.2f}  {rescued:6}/{len(losers)}")

    prior_groups: dict[str, list[Row]] = defaultdict(list)
    day_total: dict[object, float] = defaultdict(float)
    for row in ordered:
        state = "eerder groen" if day_total[row.when.date()] > 0 else "eerder rood" if day_total[row.when.date()] < 0 else "eerste/flat"
        prior_groups[f"{row.module} {state}"].append(row); day_total[row.when.date()] += row.result_r
    lines += _table("EERDERE WINST/VERLIESTRADES DEZELFDE DAG", prior_groups)

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

    def drawdown(sequence):
        equity = peak = worst = 0.0
        for value in sequence:
            equity += value; peak = max(peak, equity); worst = max(worst, peak - equity)
        return worst

    def streak(sequence):
        current = worst = 0
        for value in sequence:
            current = current + 1 if value <= 0 else 0; worst = max(worst, current)
        return worst

    values = [r.result_r for r in ordered]
    rng = random.Random(7010)
    dds, streaks = [], []
    for _ in range(5000):
        sample = values.copy(); rng.shuffle(sample); dds.append(drawdown(sample)); streaks.append(streak(sample))
    dds.sort(); streaks.sort()
    q = lambda items, p: items[min(len(items) - 1, int(p * len(items)))]
    lines.extend([
        "", "MONTE CARLO — 5.000 herschikkingen van exact dezelfde trades",
        f"werkelijke max drawdown {drawdown(values):.2f}R | mediaan {q(dds,.50):.2f}R | 95% {q(dds,.95):.2f}R | 99% {q(dds,.99):.2f}R",
        f"werkelijke langste verliesreeks {streak(values)} | mediaan {q(streaks,.50)} | 95% {q(streaks,.95)} | 99% {q(streaks,.99)}",
        f"kans op minstens 5/10/15 verliezen: {sum(x>=5 for x in streaks)/len(streaks):.1%} / {sum(x>=10 for x in streaks)/len(streaks):.1%} / {sum(x>=15 for x in streaks)/len(streaks):.1%}",
        "rekeningbuffer bij 2% doelrisico: 95%-drawdown ongeveer " + f"{q(dds,.95)*2:.1f}% (minimum-lot afwijkingen niet meegerekend)",
    ])
    lines.extend(["", "LOSS-STREAK COUNTERFACTUAL — na twee verliezen exact één volgende setup overslaan"])
    for module, group in sorted(_group(ordered, lambda r: r.module).items()):
        kept=[]; consecutive=0; skipped=0
        for row in group:
            if consecutive >= 2:
                skipped += 1; consecutive = 0; continue
            kept.append(row.result_r); consecutive = consecutive + 1 if row.result_r <= 0 else 0
        lines.append(f"{module:28} origineel {sum(r.result_r for r in group):+8.2f}R  scenario {sum(kept):+8.2f}R  {skipped} overgeslagen")

    septembers = _group([r for r in ordered if r.when.month == 9], lambda r: r.when.year)
    lines += _table("SEPTEMBER PER JAAR — nooit automatisch blokkeren", septembers)
    if len(septembers) < 3:
        lines.append("ONVOLDOENDE JAREN: geen septemberfilter voorstellen.")
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

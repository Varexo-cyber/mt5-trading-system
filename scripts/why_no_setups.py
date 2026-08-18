"""Why the engine found nothing — measured over history instead of over one hour.

`why_no_trades.py` reads the live journal, which means every question about an
entry rule can only be answered by putting money behind it and waiting. That is
how three changes were shipped in one day and measured against a single
confounded hour: fewer decisions, a different session, and the direction split
flipped from 52% short to 57% long, all at once. Nothing in that comparison was
the code.

This asks the same question of months of real bars, offline. Same engine, same
overlay, same look-ahead safety as the backtester. No terminal writes, no
orders, no API calls.

    python scripts/why_no_setups.py --days 90
    python scripts/why_no_setups.py --days 90 --symbols EURUSD.i XAUUSD

AND THE PART THAT MATTERS — the A/B:

    python scripts/why_no_setups.py --days 90 --set lone_module_minimum_confidence=0.55

`--set` overrides any field on the confluence config and runs BOTH, the live
value and yours, over the identical bars. What comes back is not an argument
about whether a rule is too strict; it is two setup counts over the same
history. Repeatable, and cheap enough to run before a change rather than after.

WHAT IT MEASURES. Setups formed, per day, and what refused the rest. That is the
top of the funnel and it is where 97.5% of live decisions go — 8,408 of 8,620 in
one measured window. It is NOT the whole funnel: spread, runway, session,
sizing and margin all live in the runner and need a broker, so a setup counted
here can still be refused live. Treat the number as a ceiling that moves, not as
a trade count.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from backtesting.replay import REPLAY_TIMEFRAMES, HistoricalContextReplay
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.data_manager import DataManager
from core.mt5_connector import MT5Connector
from core.types import Timeframe
from scripts.backtest_modules import build_engine

DEFAULT_SYMBOLS = ("EURUSD.i", "GBPUSD.i", "USDJPY.i", "AUDUSD.i", "XAUUSD")

#: Refusal texts carry live numbers — "reads 0.57", "1.00R is reached first 22%
#: of the time" — so grouping on the raw string produces one bucket per
#: decision and no finding at all. Numbers collapse to a placeholder and the
#: SHAPE of the refusal is what gets counted.
_NUMBER = re.compile(r"\d+(?:\.\d+)?%?")


def shape_of(reason: str) -> str:
    return _NUMBER.sub("N", reason)[:110]


def history(
    connector: MT5Connector, symbol: str, start: datetime, end: datetime
) -> dict[Timeframe, pd.DataFrame]:
    frames: dict[Timeframe, pd.DataFrame] = {}
    for timeframe in REPLAY_TIMEFRAMES:
        warmup = start - timeframe.duration * 400
        raw = connector.copy_rates_range(symbol, timeframe.mt5_value, warmup, end)
        frames[timeframe] = DataManager._to_frame(raw)
    return frames


def apply_overrides(settings, pairs: list[str]):  # type: ignore[no-untyped-def]
    """Return settings with confluence fields replaced, types taken from the model.

    Reading the target type off the existing value rather than guessing keeps
    `--set require_direction_advantage=false` from silently becoming the string
    "false", which is truthy and would have made the run a lie.
    """
    confluence = settings.analysis.confluence
    updates: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        key = key.strip()
        if not hasattr(confluence, key):
            raise SystemExit(f"confluence has no field {key!r}")
        current = getattr(confluence, key)
        if isinstance(current, bool):
            updates[key] = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int) and not isinstance(current, bool):
            updates[key] = int(raw)
        elif isinstance(current, float):
            updates[key] = float(raw)
        else:
            updates[key] = raw
    return settings.model_copy(
        update={
            "analysis": settings.analysis.model_copy(
                update={"confluence": confluence.model_copy(update=updates)}
            )
        }
    )


def run(settings, catalogue, stride: int, label: str = ""):  # type: ignore[no-untyped-def]
    """Setups formed and the shape of every refusal, over the supplied bars.

    Prints per symbol as it goes. The loading phase reported progress and this
    one did not, so the run went silent for minutes at the exact point where it
    was doing the work — which reads as a hang, not as patience.
    """
    replay = HistoricalContextReplay(build_engine(settings), decision_stride_bars=stride)
    setups = 0
    decisions = 0
    refusals: Counter[str] = Counter()
    scored: list[float] = []
    for symbol, frames, point, start, end in catalogue:
        started = perf_counter()
        before = setups
        for _decided_at, idea in replay.ideas(symbol, frames, point=point, start=start, end=end):
            decisions += 1
            if idea.approved:
                setups += 1
                continue
            refusals[shape_of(idea.reason)] += 1
            if idea.score > 0:
                scored.append(idea.score)
        print(
            f"    {label}{symbol}: {setups - before} setups  " f"({perf_counter() - started:.0f}s)",
            flush=True,
        )
    return setups, decisions, refusals, scored


def render(label, setups, decisions, refusals, scored, days, threshold) -> str:  # type: ignore[no-untyped-def]
    lines = [
        "",
        "=" * 78,
        f"  {label}",
        "=" * 78,
        "",
        f"  {decisions:,} decisions over {days:.0f} days  ->  {setups:,} setups "
        f"({setups / max(days, 1):.1f} per day)",
        "",
        "  what refused the rest:",
    ]
    for shape, count in refusals.most_common(10):
        lines.append(f"    {count:>7,}x  {shape}")
    if scored:
        near = sum(1 for value in scored if threshold - 5 <= value < threshold)
        lines.append("")
        lines.append(
            f"  {len(scored):,} refusals still reached a positive score, best "
            f"{max(scored):.1f} against the {threshold:.1f} threshold; {near} came "
            f"within 5 points and did not clear it"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=90.0)
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--stride", type=int, default=1, help="decide every Nth H1 bar")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="override a confluence field and run both versions over the same bars",
    )
    args = parser.parse_args(argv)

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

    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    catalogue = []
    try:
        for symbol in args.symbols:
            print(f"  loading {symbol} …", flush=True)
            try:
                spec = connector.spec(symbol)
                catalogue.append(
                    (symbol, history(connector, symbol, start, end), spec.point, start, end)
                )
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not end the run
                print(f"    skipped: {type(exc).__name__}: {exc}")
    finally:
        with contextlib.suppress(Exception):
            connector.disconnect()

    if not catalogue:
        print("\nNo usable history. Nothing to measure.")
        return 1

    threshold = settings.analysis.confluence.score_threshold
    print("\n  measuring as configured …", flush=True)
    live = run(settings, catalogue, args.stride)
    print(render("AS THE ACCOUNT IS CONFIGURED NOW", *live, args.days, threshold))

    if args.overrides:
        changed = apply_overrides(settings, args.overrides)
        print("\n  measuring the variant over the same bars …", flush=True)
        variant = run(changed, catalogue, args.stride, label="variant ")
        print(
            render(
                "WITH " + ", ".join(args.overrides),
                *variant,
                args.days,
                changed.analysis.confluence.score_threshold,
            )
        )
        before, after = live[0], variant[0]
        delta = (after - before) / before if before else float("inf")
        print(
            f"\n  {before:,} setups -> {after:,} setups over the identical bars " f"({delta:+.0%})."
        )
        print(
            "  Setups are a ceiling, not trades: spread, runway, session, sizing "
            "and margin\n  all live in the runner and can still refuse every one of them."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

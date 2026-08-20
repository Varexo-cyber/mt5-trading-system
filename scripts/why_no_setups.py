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
from core.types import Timeframe, TradingMode
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
        elif isinstance(current, dict):
            # `module:value` pairs, comma separated, merged onto whatever the
            # config already holds:
            #
            #     --set lone_module_minimum_confidence_by_module=liquidity_sweep:0.55
            #
            # The module name is checked against the weights table, because the
            # failure this whole tool exists to avoid is measuring a change
            # that was never applied and reporting "no difference". A typo in a
            # detector name would do exactly that, silently.
            merged = dict(current)
            for item in raw.split(","):
                if ":" not in item:
                    raise SystemExit(f"--set {key} expects module:value, got {item!r}")
                module, number = item.split(":", 1)
                module = module.strip()
                if module not in confluence.weights:
                    known = ", ".join(sorted(confluence.weights))
                    raise SystemExit(f"unknown detector {module!r}; known: {known}")
                merged[module] = float(number)
            updates[key] = merged
        elif isinstance(current, tuple):
            # A LIST OF DETECTOR NAMES, comma separated:
            #
            #     --set trend_continuation_modules=trend_momentum,drift_continuation
            #
            # This fell through to the branch below and set the field to a
            # plain STRING — the exact failure this function's docstring says
            # it exists to prevent. Pydantic would either coerce it to a tuple
            # of single characters or reject it, and the run would report a
            # difference that came from neither version of the rule.
            #
            # Names are checked against the weights table for the same reason
            # the dict branch does it: a typo would silently measure nothing
            # and print "no difference".
            names = tuple(item.strip() for item in raw.split(",") if item.strip())
            for name in names:
                if name not in confluence.weights:
                    known = ", ".join(sorted(confluence.weights))
                    raise SystemExit(f"unknown detector {name!r}; known: {known}")
            updates[key] = names
        elif isinstance(current, str):
            updates[key] = raw
        else:
            # Refuse rather than guess. Passing the raw string through is how a
            # field of an unhandled type gets set to something that is not what
            # the operator asked for, and the run then measures neither the
            # current rule nor the proposed one while reporting a number.
            raise SystemExit(
                f"--set cannot yet type a value for {key!r} "
                f"(it holds {type(current).__name__}). Add a branch for it "
                f"rather than letting the run measure something else."
            )
    return settings.model_copy(
        update={
            "analysis": settings.analysis.model_copy(
                update={"confluence": confluence.model_copy(update=updates)}
            )
        }
    )


def run(settings, catalogue, stride: int, label: str = "", mode=TradingMode.BACKTEST):  # type: ignore[no-untyped-def]
    """Setups formed and the shape of every refusal, over the supplied bars.

    Prints per symbol as it goes. The loading phase reported progress and this
    one did not, so the run went silent for minutes at the exact point where it
    was doing the work — which reads as a hang, not as patience.
    """
    replay = HistoricalContextReplay(build_engine(settings), decision_stride_bars=stride, mode=mode)
    setups = 0
    decisions = 0
    refusals: Counter[str] = Counter()
    scored: list[float] = []
    for symbol, frames, point, start, end in catalogue:
        started = perf_counter()
        before = setups
        ideas = replay.ideas(symbol, frames, point=point, start=start, end=end)
        # Three values, not two: the spread rides along since the backtest
        # started charging it. This tool unpacked two and crashed on the
        # first symbol, which is why it could not be run at all.
        for _decided_at, _spread, idea in ideas:
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
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "judge in the account's own trading mode, so `live_enabled_modules` "
            "applies the way it does on the account. Without this the run counts "
            "every weighted detector, including the ones the allowlist excludes."
        ),
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
    # `live_enabled_modules` is consulted only when the mode IS live, and the
    # configured mode is not reliably one: this printed "voting as backtest"
    # while listing the allowlist, so it applied nothing and reported a number
    # anyway. Prefer the account's own live mode when it has one, and say out
    # loud which mode is being measured either way — a run that silently
    # measures the wrong engine is the failure this whole tool exists to avoid.
    if not args.live:
        mode = TradingMode.BACKTEST
    elif settings.mode.is_live:
        mode = settings.mode
    else:
        mode = TradingMode.MICRO_LIVE
    allowed = settings.analysis.confluence.live_enabled_modules
    if args.live:
        muted = sorted(
            module
            for module, weight in settings.analysis.confluence.weights.items()
            if weight > 0 and module not in allowed
        )
        note = "" if settings.mode.is_live else f" (config says {settings.mode.value})"
        print(f"\n  voting as {mode.value}{note}: {len(allowed)} detectors on the allowlist")
        if muted:
            print(f"  weighted but NOT voting live: {', '.join(muted)}")
    print("\n  measuring as configured …", flush=True)
    baseline = run(settings, catalogue, args.stride, mode=mode)
    heading = "AS THE ACCOUNT VOTES LIVE" if args.live else "AS THE ACCOUNT IS CONFIGURED NOW"
    print(render(heading, *baseline, args.days, threshold))

    if args.overrides:
        changed = apply_overrides(settings, args.overrides)
        print("\n  measuring the variant over the same bars …", flush=True)
        variant = run(changed, catalogue, args.stride, label="variant ", mode=mode)
        print(
            render(
                "WITH " + ", ".join(args.overrides),
                *variant,
                args.days,
                changed.analysis.confluence.score_threshold,
            )
        )
        before, after = baseline[0], variant[0]
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

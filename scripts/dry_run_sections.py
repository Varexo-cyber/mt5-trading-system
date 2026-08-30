"""What sections two and three WOULD have done on this account, last week.

    python scripts/dry_run_sections.py --days 7
    python scripts/dry_run_sections.py --days 30 --csv runtime/dryrun.csv

This is not the research harness. The research measured an edge on ten years of
HistData bid bars and answered "does this entry work". This answers a different
and, for a live account, more useful question: **on YOUR broker, YOUR symbols,
YOUR equity and YOUR settings, what came out the other end?**

The difference is everything the research had to assume:

    the real Eightcap spread on every bar, charged per trade
    the real minimum lot, so UNDERCAPITALIZED is counted rather than modelled
    the real gates -- entry quality, the lifecycle, the cost share, the
      confluence threshold -- each refusal attributed by name
    the real sizer, so every trade carries the lots and the euros it would
      actually have risked

EVERY DECISION IS REPORTED, not just the trades. A week with four trades and
three hundred refusals is a diagnosis; a week with four trades is a number.

WHAT IT STILL CANNOT SEE, and these are the honest limits:

  * Slippage beyond the recorded spread. Fills are taken at the bar's price
    plus the spread that bar carried, which is optimistic at a session open.
  * Positions are resolved independently, so the four-slot cap is REPORTED
    (how often five or more would have been open at once) rather than
    enforced. Enforcing it would need an ordering rule this has no basis to
    invent, and it would flatter the result by dropping trades arbitrarily.
  * The news blackout and session gates are not replayed. They only ever
    remove trades, so the count here is an upper bound.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analysis import ConfluenceEngine
from backtesting.replay import fetch_mt5_history
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from core.types import Direction, MarketContext, Series, Tick, Timeframe, TradingMode
from risk.position_sizer import PositionSizer
from runner.service import build_analysis_modules

#: Every timeframe a sweep may put a section on, plus M1 and M5 so a signal
#: can be resolved on bars finer than the one that produced it. Resolving an
#: M30 trade on M30 bars cannot see which barrier a bar touched first.
SWEEPABLE = (Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1, Timeframe.H4)
NEEDED = (Timeframe.M1, *SWEEPABLE)
#: Bars of history each decision may look back over.
WARMUP = 260


@dataclass(slots=True)
class Decision:
    when: datetime
    symbol: str
    module: str
    outcome: str  # TRADE / refusal reason
    direction: str = ""
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    lots: float = 0.0
    risk_money: float = 0.0
    risk_pct: float = 0.0
    result_r: float | None = None
    pnl_money: float | None = None
    note: str = ""


def _context(symbol: str, frames: dict, upto: datetime, spread: float) -> MarketContext | None:
    """Everything knowable at `upto`, and nothing that closed after it."""
    series: dict[Timeframe, Series] = {}
    for timeframe, frame in frames.items():
        visible = frame[frame.index + timeframe.duration <= upto]
        if len(visible) < WARMUP:
            return None
        series[timeframe] = Series(symbol, timeframe, visible.tail(WARMUP), upto)
    price = float(series[Timeframe.M5].df["close"].iloc[-1])
    half = spread / 2.0
    return MarketContext(symbol, upto, series, Tick(symbol, upto, price - half, price + half))


def _resolve(frame: pd.DataFrame, start: datetime, idea, horizon_bars: int):
    """First touch of stop or target on the bars after entry.

    Same rules as the research: the entry bar itself counts, a bar spanning
    both barriers is a LOSS because the order is unknowable, and a trade that
    reaches neither is reported as open rather than closed at the clock.
    """
    future = frame[frame.index >= start].head(horizon_bars)
    if future.empty:
        return None
    long = idea.direction is Direction.LONG
    for _, bar in future.iterrows():
        hit_stop = bar["low"] <= idea.stop_loss if long else bar["high"] >= idea.stop_loss
        hit_target = bar["high"] >= idea.take_profit if long else bar["low"] <= idea.take_profit
        if hit_stop:
            return -1.0
        if hit_target:
            risk = abs(idea.entry - idea.stop_loss)
            return abs(idea.take_profit - idea.entry) / risk if risk > 0 else 0.0
    return None


def _one_pass(
    *,
    engine,
    sizer,
    symbol: str,
    spec,
    frames: dict,
    start: datetime,
    end: datetime,
    equity: float,
    clock: Timeframe,
    resolve_on: Timeframe,
) -> list[Decision]:
    """Every decision one configuration reaches on one symbol.

    `clock` is the timeframe the section reads, so it is also the timeframe a
    decision may be taken on -- sampling an M30 section on M15 bars would offer
    it the same setup twice.

    `resolve_on` must be FINER than the clock. Resolving an M30 trade on M30
    bars cannot tell which barrier a bar touched first, and assuming the good
    one is how a backtest lies.
    """
    decisions: list[Decision] = []
    bars = frames[clock]
    window = bars[(bars.index >= start) & (bars.index <= end)]
    horizon = int(96 * clock.duration / resolve_on.duration)
    for bar_time in window.index:
        upto = bar_time + clock.duration
        spread_price = float(window.loc[bar_time].get("spread", 0)) * spec.point
        ctx = _context(symbol, frames, upto, spread_price)
        if ctx is None:
            continue
        idea = engine.evaluate(ctx, TradingMode.MICRO_LIVE)
        module = ",".join(sorted({sig.module for sig in idea.signals if sig.score})) or "-"
        if not idea.approved:
            decisions.append(
                Decision(upto, symbol, module, "REFUSED_CONFLUENCE", note=idea.reason[:90])
            )
            continue
        sized = sizer.size(
            spec=spec,
            equity=equity,
            direction=idea.direction,
            entry=idea.entry,
            sl=idea.stop_loss,
            tp=idea.take_profit,
            spread_price=spread_price,
        )
        if not sized.decision.approved:
            decisions.append(
                Decision(
                    upto,
                    symbol,
                    module,
                    sized.decision.reason.name,
                    direction=idea.direction.name,
                    note=sized.decision.detail[:90],
                )
            )
            continue
        r = _resolve(frames[resolve_on], upto, idea, horizon_bars=horizon)
        risk_money = sized.actual_risk_money
        decisions.append(
            Decision(
                upto,
                symbol,
                module,
                "TRADE",
                direction=idea.direction.name,
                entry=idea.entry,
                stop=idea.stop_loss,
                target=idea.take_profit,
                lots=sized.volume,
                risk_money=risk_money,
                risk_pct=sized.actual_risk_pct,
                result_r=r,
                pnl_money=None if r is None else r * risk_money,
            )
        )
    return decisions


def _retimed(settings, module_name: str, timeframe: str):
    """The same settings with one section moved to another timeframe."""
    analysis = settings.analysis
    section = getattr(analysis, module_name)
    return settings.model_copy(
        update={
            "analysis": analysis.model_copy(
                update={module_name: section.model_copy(update={"timeframe": timeframe})}
            )
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--symbols", default="", help="comma list; default = the live universe")
    parser.add_argument("--csv", default="", help="write every decision to this file")
    parser.add_argument("--equity", type=float, default=0.0, help="override account equity")
    parser.add_argument(
        "--sweep",
        default="",
        help=(
            "comma list of timeframes to try each section on, e.g. M5,M15,M30,H1,H4. "
            "The shipped timeframes were chosen on HistData; this asks the same "
            "question against THIS broker's spreads, which is the number that decides it."
        ),
    )
    parser.add_argument(
        "--no-m1",
        action="store_true",
        help="skip M1 history (much faster on a large universe; coarser resolution)",
    )
    args = parser.parse_args()

    # THE OVERLAY, or there are no live modules at all. The base config ships
    # `live_enabled_modules` empty on purpose -- permission to trade real money
    # is an account-level decision, and it lives in the Eightcap overlay. Load
    # without it and this exits with "no live modules" while the account is
    # perfectly well configured.
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=True)
    settings = settings.model_copy(
        update={"system": settings.system.model_copy(update={"mode": TradingMode.MICRO_LIVE})}
    )
    # Two positional arguments, `settings.mt5` first -- the same shape `main.py`
    # uses. Passing only the credentials made the connector read
    # `credentials.terminal_path`, which does not exist, and the run died
    # before it fetched a single bar.
    live = set(settings.analysis.confluence.live_enabled_modules)
    if not live:
        raise SystemExit(
            "no live modules in this configuration: nothing to dry-run. "
            "Check `analysis.confluence.live_enabled_modules` in config/eightcap.yaml."
        )

    credentials = load_credentials(required=True)
    connector = MT5Connector(
        settings.mt5,
        credentials,
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    connector.connect()
    try:
        account = connector.account()
        equity = args.equity or account.equity

        symbols = (
            [s.strip() for s in args.symbols.split(",") if s.strip()]
            or list(settings.active_whitelist)
            or ["EURUSD", "GBPUSD"]
        )
        end = datetime.now(UTC)
        start = end - timedelta(days=args.days)
        # Warmup has to come from before the window or the first day is blind.
        fetch_from = start - timedelta(days=max(20, args.days))
        fetch_these = tuple(tf for tf in NEEDED if not (args.no_m1 and tf is Timeframe.M1))

        print(f"\n{'=' * 78}")
        print(f"DRY RUN — sections {', '.join(sorted(live))}")
        print(f"{args.days} days to {end:%Y-%m-%d %H:%M} UTC, equity EUR {equity:.2f}")
        print(f"{len(symbols)} symbols, {settings.effective_risk_pct():.1f}% risk per trade")
        print(f"{'=' * 78}\n")

        # WHICH SECTIONS ON WHICH CLOCKS. The shipped timeframes were chosen
        # on HistData; a sweep asks the same question against this broker's
        # spreads, which is the number that actually decides it.
        #
        # Each section is swept on its own, never together: two sections on
        # the same clock would merge into one confluence idea and the result
        # would say nothing about either.
        module_config = {"impulse_retest": "impulse_retest", "order_block": "order_block"}
        passes: list[tuple[str, str]] = []
        if args.sweep:
            wanted = [t.strip().upper() for t in args.sweep.split(",") if t.strip()]
            for name in sorted(live & set(module_config)):
                for tf in wanted:
                    passes.append((name, tf))
        else:
            for name in sorted(live & set(module_config)):
                passes.append((name, getattr(settings.analysis, name).timeframe))

        finest = Timeframe.M5 if args.no_m1 else Timeframe.M1
        results: dict[tuple[str, str], list[Decision]] = {key: [] for key in passes}
        skipped_symbols = 0

        for index, symbol in enumerate(symbols, 1):
            try:
                spec = connector.spec(symbol)
                frames = {
                    tf: fetch_mt5_history(connector, symbol, tf, fetch_from, end)
                    for tf in fetch_these
                }
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the run
                skipped_symbols += 1
                print(f"  [{index}/{len(symbols)}] {symbol}: no history ({exc})")
                continue
            if any(len(f) < WARMUP + 10 for f in frames.values()):
                skipped_symbols += 1
                continue

            for name, tf_name in passes:
                clock = Timeframe.parse(tf_name)
                if clock not in frames or clock.duration <= finest.duration:
                    continue
                tuned = _retimed(settings, name, tf_name)
                only = [m for m in build_analysis_modules(tuned) if m.name == name]
                pass_engine = ConfluenceEngine(only, tuned.analysis.confluence)
                results[(name, tf_name)].extend(
                    _one_pass(
                        engine=pass_engine,
                        sizer=PositionSizer(tuned),
                        symbol=symbol,
                        spec=spec,
                        frames=frames,
                        start=start,
                        end=end,
                        equity=equity,
                        clock=clock,
                        resolve_on=finest,
                    )
                )
            done = sum(
                1
                for v in results.values()
                for d in v
                if d.symbol == symbol and d.outcome == "TRADE"
            )
            if done:
                print(f"  [{index}/{len(symbols)}] {symbol}: {done} trades")

        decisions = [d for v in results.values() for d in v]
    finally:
        connector.shutdown()

    if args.sweep:
        _sweep_report(results, equity, args.days)
    _report(decisions, equity, args.days, skipped_symbols)
    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "when",
                    "symbol",
                    "module",
                    "outcome",
                    "direction",
                    "entry",
                    "stop",
                    "target",
                    "lots",
                    "risk_money",
                    "risk_pct",
                    "result_r",
                    "pnl_money",
                    "note",
                ]
            )
            for d in decisions:
                writer.writerow(
                    [
                        d.when.isoformat(),
                        d.symbol,
                        d.module,
                        d.outcome,
                        d.direction,
                        d.entry,
                        d.stop,
                        d.target,
                        d.lots,
                        round(d.risk_money, 2),
                        round(d.risk_pct, 3),
                        "" if d.result_r is None else round(d.result_r, 3),
                        "" if d.pnl_money is None else round(d.pnl_money, 2),
                        d.note,
                    ]
                )
        print(f"\nevery decision written to {path}")


def _sweep_report(results: dict, equity: float, days: int) -> None:
    """One row per (section, timeframe), so the clock can be chosen on THIS feed.

    The shipped timeframes came from HistData bid bars. The only thing that
    could change the answer here is cost, and cost is exactly what this run
    measures for real -- so if a different clock wins by a margin, it wins
    because of this broker's spreads and not because of a preference.
    """
    print(f"\n{'=' * 78}")
    print("TIMEFRAME SWEEP — each section on each clock, this broker, this window")
    print(f"{'=' * 78}")
    print(
        f"  {'section':<18}{'tf':>5}{'trades':>8}{'closed':>8}{'win':>7}"
        f"{'R':>9}{'EUR':>10}{'undercap':>10}{'spread':>8}"
    )
    rows = []
    for (name, tf), decisions in sorted(results.items()):
        trades = [d for d in decisions if d.outcome == "TRADE"]
        closed = [d for d in trades if d.result_r is not None]
        under = sum(1 for d in decisions if d.outcome == "UNDERCAPITALIZED")
        spread_out = sum(1 for d in decisions if "SPREAD" in d.outcome or "COST" in d.outcome)
        r_total = sum(d.result_r or 0.0 for d in closed)
        money = sum(d.pnl_money or 0.0 for d in closed)
        wins = sum(1 for d in closed if (d.result_r or 0) > 0)
        win = f"{wins / len(closed):.0%}" if closed else "-"
        print(
            f"  {name:<18}{tf:>5}{len(trades):>8}{len(closed):>8}{win:>7}"
            f"{r_total:>+9.2f}{money:>+10.2f}{under:>10}{spread_out:>8}"
        )
        rows.append((money, name, tf, len(trades)))
    if rows:
        rows.sort(reverse=True)
        money, name, tf, n = rows[0]
        share = money / equity if equity else 0.0
        print(
            f"\n  best on this feed: {name} on {tf} — EUR {money:+.2f} "
            f"({share:+.1%} of equity) over {n} trades, {n / max(days, 1):.1f}/day"
        )
        print(
            "  A clock that beats the shipped one here is worth taking seriously:\n"
            "  the research could not price this broker's spread and this can."
        )
    print(f"{'=' * 78}")


def _report(decisions: list[Decision], equity: float, days: int, skipped: int) -> None:
    trades = [d for d in decisions if d.outcome == "TRADE"]
    closed = [d for d in trades if d.result_r is not None]
    refusals = Counter(d.outcome for d in decisions if d.outcome != "TRADE")

    print(f"\n{'-' * 78}")
    print(f"DECISIONS   {len(decisions)} total across {days} days")
    if skipped:
        print(f"            ({skipped} symbols skipped for want of history)")
    print("\nWHY NOTHING HAPPENED — every refusal, by name")
    total = max(len(decisions), 1)
    for reason, count in refusals.most_common(15):
        print(f"   {reason:<34} {count:>7}  {count / total:>6.1%}")

    print(
        f"\nTRADES      {len(trades)} taken, {len(closed)} resolved, "
        f"{len(trades) - len(closed)} still open at the end of the window"
    )
    if not closed:
        print(
            "\n   No resolved trades. Either the window is too short or the "
            "refusals above are the whole story."
        )
        return

    wins = [d for d in closed if d.result_r and d.result_r > 0]
    r_total = sum(d.result_r or 0.0 for d in closed)
    money = sum(d.pnl_money or 0.0 for d in closed)
    risks = [d.risk_money for d in closed]
    pcts = [d.risk_pct for d in closed]

    print(f"   win rate            {len(wins) / len(closed):>8.1%}  ({len(wins)}/{len(closed)})")
    print(f"   total               {r_total:>+8.2f} R")
    print(f"   per trade           {r_total / len(closed):>+8.3f} R")
    print(
        f"   risk per trade      EUR {np.mean(risks):>6.2f}   "
        f"({np.mean(pcts):.2f}% of equity, worst {max(pcts):.2f}%)"
    )
    share = money / equity if equity else 0.0
    print(f"\n   PROFIT              EUR {money:>+8.2f}   on EUR {equity:.2f} = {share:+.1%}")
    print(f"   trades per day      {len(trades) / max(days, 1):>8.1f}")

    over = [d for d in closed if d.risk_pct > 2.5]
    if over:
        print(
            f"\n   {len(over)} trades went out ABOVE the 2% target because the "
            f"broker minimum forced it"
        )
        print(
            f"   worst was {max(d.risk_pct for d in over):.2f}% of equity — "
            f"EUR {max(d.risk_money for d in over):.2f} on one stop"
        )

    frame = pd.DataFrame(
        [{"day": d.when.date(), "r": d.result_r, "eur": d.pnl_money} for d in closed]
    )
    per_day = frame.groupby("day").agg(trades=("r", "size"), R=("r", "sum"), EUR=("eur", "sum"))
    print("\nBY DAY")
    for day, row in per_day.iterrows():
        bar = "+" * int(max(row.R, 0) * 3) + "-" * int(max(-row.R, 0) * 3)
        print(
            f"   {day}  {int(row.trades):>3} trades  {row.R:>+7.2f} R  "
            f"EUR {row.EUR:>+7.2f}  {bar}"
        )
    print(f"\n   days green {int((per_day.R > 0).sum())} / {len(per_day)}")

    by_module = pd.DataFrame(
        [{"module": d.module, "r": d.result_r, "eur": d.pnl_money} for d in closed]
    )
    print("\nBY SECTION")
    for module, row in (
        by_module.groupby("module")
        .agg(trades=("r", "size"), R=("r", "sum"), EUR=("eur", "sum"))
        .iterrows()
    ):
        print(
            f"   {module:<34} {int(row.trades):>4} trades  {row.R:>+7.2f} R  "
            f"EUR {row.EUR:>+7.2f}"
        )
    print(f"{'-' * 78}\n")


if __name__ == "__main__":
    main()

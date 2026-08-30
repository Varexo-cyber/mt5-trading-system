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
    #: When the trade left. Needed to apply the concurrent-position cap, which
    #: is a rule about how many trades are open AT ONCE and cannot be checked
    #: from entry times alone.
    exit_at: datetime | None = None
    #: Which (section, clock) pass produced it, so the live pair can be picked
    #: back out of a sweep that also measured eight combinations that will
    #: never run together.
    pass_key: tuple[str, str] = ("", "")
    #: The SAME trade under `TradeManagementConfig`'s break-even rule, so the
    #: two exits are compared on identical entries rather than on two runs.
    managed_r: float | None = None
    managed_money: float | None = None


def _context(
    symbol: str, frames: dict, upto: datetime, spread: float, slices: dict | None = None
) -> MarketContext | None:
    """Everything knowable at `upto`, and nothing that closed after it.

    WHY THIS IS WRITTEN WITH `searchsorted` AND NOT A MASK.

    It used to read:

        visible = frame[frame.index + timeframe.duration <= upto]
        ... visible.tail(WARMUP)

    which is correct and quadratic. Every call shifted the ENTIRE index,
    compared the entire thing, copied every matching row, and then threw all
    but the last 260 away. Over a 180-day window that is a 52,000-row M5 frame
    walked in full for each of ~17,000 M15 bars, on each of sixteen markets --
    about 1.4e10 row operations, and it is why a six-month run took hours while
    the MT5 fetch it was blamed on took seconds.

    The bars are sorted, so the cut point is a binary search: `index <= upto -
    duration` is exactly `searchsorted(upto - duration, side="right")`, and the
    warmup is then a plain positional slice. O(log n) instead of O(n), same
    answer -- `test_the_fast_context_matches_the_slow_one` compares them bar
    for bar rather than taking that on trust.

    AND THE SLOW FRAMES BARELY MOVE, so `slices` memoises the DataFrame per
    (timeframe, cut). The Series wrapper is still rebuilt because it carries
    `upto`, which differs every bar; only the slice is shared, and the frames
    are never mutated so sharing one is safe.

    THE SAVING IS 25%, NOT THE SIXTY I FIRST WROTE HERE. On an M30 walk the
    arithmetic is not a guess: M5, M15 and M30 all advance on every bar and
    can never hit, H1 advances every second bar and H4 every eighth, so of
    five slices per bar about 1.25 are reused. `test_it_actually_hits`
    measures it -- 149 misses out of 200 -- and the number in this comment is
    that measurement rather than an estimate that sounded right.
    """
    series: dict[Timeframe, Series] = {}
    for timeframe, frame in frames.items():
        index = frame.index
        # A bar is visible once it has CLOSED, i.e. open + duration <= upto.
        cut = int(index.searchsorted(upto - timeframe.duration, side="right"))
        if cut < WARMUP:
            return None
        window = None if slices is None else slices.get((timeframe, cut))
        if window is None:
            window = frame.iloc[cut - WARMUP : cut]
            if slices is not None:
                if len(slices) > 64:
                    # Only the newest cut of each clock is ever asked for again;
                    # an unbounded dict would hold the whole history twice.
                    slices.clear()
                slices[(timeframe, cut)] = window
        series[timeframe] = Series(symbol, timeframe, window, upto)
    price = float(series[Timeframe.M5].df["close"].iloc[-1])
    half = spread / 2.0
    return MarketContext(symbol, upto, series, Tick(symbol, upto, price - half, price + half))


def _resolve(frame: pd.DataFrame, start: datetime, idea, horizon_bars: int, manage=None):
    """First touch of stop or target on the bars after entry.

    Same rules as the research: the entry bar itself counts, a bar spanning
    both barriers is a LOSS because the order is unknowable, and a trade that
    reaches neither is reported as open rather than closed at the clock.

    Returns `(r, exit_time, managed_r)`. The exit time is what frees the symbol
    for the next trade -- see `_one_pass`. A trade that resolves neither way
    returns `(None, None, None)` and holds the symbol to the end of the window,
    which is what a real open position does.

    `manage` IS THE QUESTION THE RESEARCH NEVER ASKED, and `TradeManagementConfig`
    is switched on for these sections with no per-module exception.

    `break_even_at_r` is 0.25, and a second trigger --
    `capital_protection_at_equity_pct` measured in euros rather than R -- arms
    it as early as 0.125R on the position sizes the minimum-lot override
    produces. The stop then jumps from 0.85 ATR BELOW entry to 0.10 ATR ABOVE
    it.

    That is a different strategy from the measured one. 18,828 trades were
    resolved against a stop that does not move, and these two sections enter AT
    a level, where price oscillates by construction. Every trade that ran a
    little, came back to the level, and then went to target counts as a full
    +1R in the research and would scratch at +0.1R here.

    It cuts both ways -- the same rule rescues a loser that ran first -- so it
    is not an argument, it is an arithmetic question about which population is
    bigger. Both numbers come off the same bar walk, so the comparison is on
    identical trades rather than on two runs.
    """
    future = frame[frame.index >= start].head(horizon_bars)
    if future.empty:
        return None, None, None
    long = idea.direction is Direction.LONG
    risk = abs(idea.entry - idea.stop_loss)
    reward_r = (abs(idea.take_profit - idea.entry) / risk) if risk > 0 else 0.0

    # The managed run walks the same bars with a stop that is allowed to move.
    # `armed` is one-way: a stop that has been pulled up is never pushed back.
    managed_open = manage is not None and risk > 0
    managed_r: float | None = None
    managed_stop = idea.stop_loss
    armed = False
    trigger_r, offset_r = manage or (0.0, 0.0)
    direction_sign = 1.0 if long else -1.0

    fixed_r: float | None = None
    exit_at: datetime | None = None

    for stamp, bar in future.iterrows():
        hit_stop = bar["low"] <= idea.stop_loss if long else bar["high"] >= idea.stop_loss
        hit_target = bar["high"] >= idea.take_profit if long else bar["low"] <= idea.take_profit

        if managed_open:
            # ORDER MATTERS AND IT IS NOT THE FLATTERING ONE. Within a bar the
            # sequence is unknowable, so the managed stop is checked against
            # the price it already sits at BEFORE this bar's excursion is
            # allowed to arm it. Arming on the same bar's high and then
            # surviving that bar's low is the look-ahead this account has
            # already been bitten by twice.
            managed_hit = bar["low"] <= managed_stop if long else bar["high"] >= managed_stop
            if managed_hit and hit_target:
                # Both in one bar: the order is unknowable, so take the stop.
                managed_r = (managed_stop - idea.entry) / risk * direction_sign
            elif managed_hit:
                managed_r = (managed_stop - idea.entry) / risk * direction_sign
            elif hit_target:
                managed_r = reward_r
            if managed_r is not None:
                managed_open = False
            else:
                excursion = (bar["high"] - idea.entry) if long else (idea.entry - bar["low"])
                if not armed and excursion / risk >= trigger_r:
                    armed = True
                    managed_stop = idea.entry + offset_r * risk * direction_sign

        if fixed_r is None:
            if hit_stop:
                fixed_r, exit_at = -1.0, stamp
            elif hit_target:
                fixed_r, exit_at = reward_r, stamp
        if fixed_r is not None and not managed_open:
            break

    if managed_open:
        managed_r = None
    return fixed_r, exit_at, managed_r


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
    pass_key: tuple[str, str] = ("", ""),
    manage: tuple[float, float] | None = None,
) -> list[Decision]:
    """Every decision one configuration reaches on one symbol.

    `clock` is the timeframe the section reads, so it is also the timeframe a
    decision may be taken on -- sampling an M30 section on M15 bars would offer
    it the same setup twice.

    `resolve_on` must be FINER than the clock. Resolving an M30 trade on M30
    bars cannot tell which barrier a bar touched first, and assuming the good
    one is how a backtest lies.

    ONE POSITION PER SYMBOL AT A TIME, which is `Reason.POSITION_ALREADY_OPEN`
    live and was missing here entirely. Without it this loop takes a fresh
    trade on EVERY bar the setup stays valid, and that is not a small
    over-count -- it is biased. A retest that works leaves the level in a bar
    or two and yields one entry; a retest that fails sits on the level and
    grinds, yielding five. Duplicates are drawn from the losers, so the
    measured hit rate falls for a reason that has nothing to do with the
    strategy. The 30 August run read 33% where the research reads 62-68%.
    """
    decisions: list[Decision] = []
    bars = frames[clock]
    window = bars[(bars.index >= start) & (bars.index <= end)]
    horizon = int(96 * clock.duration / resolve_on.duration)
    busy_until: datetime | None = None
    #: Per pass, so it never outlives the frames it points into.
    slices: dict = {}
    for bar_time in window.index:
        upto = bar_time + clock.duration
        if busy_until is not None and upto <= busy_until:
            continue  # a position is open on this symbol; live would refuse
        spread_price = float(window.loc[bar_time].get("spread", 0)) * spec.point
        ctx = _context(symbol, frames, upto, spread_price, slices)
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
        r, exit_at, managed_r = _resolve(
            frames[resolve_on], upto, idea, horizon_bars=horizon, manage=manage
        )
        # Held to the end of the window when it never resolved, exactly as an
        # open position holds the symbol live.
        busy_until = exit_at if exit_at is not None else end + clock.duration
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
                exit_at=exit_at,
                pass_key=pass_key,
                managed_r=managed_r,
                managed_money=None if managed_r is None else managed_r * risk_money,
            )
        )
    return decisions


#: The markets worth measuring first, and the reason is not "the big ones".
#:
#: THE RESEARCH WAS DONE ON ELEVEN FX MAJORS. Both sections were chosen, tuned
#: and holdout-tested on those and on gold; every other market in the broker's
#: catalogue is an extrapolation. A 232-symbol run spends most of its time
#: measuring markets that cannot confirm or refute the finding, and it takes
#: long enough that it does not get run.
#:
#: So this is the confirmation set: the instruments the numbers came from,
#: plus the four index CFDs with enough volume that a spread assumption is
#: defensible on them. Sixteen markets against 232 is roughly a fifteenth of
#: the work.
CORE_UNIVERSE: tuple[str, ...] = (
    # The eleven the research measured.
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
    "EURCHF",
    # Measured separately, and shipped with its own wider stop.
    "XAUUSD",
    # Not measured. Included because they are where the volume is and because
    # the sweep needs SOMETHING outside the training set to disagree with.
    "US30",
    "NDX100",
    "SPX500",
    "GER40",
)


def _core_universe(connector, settings) -> list[str]:
    """`CORE_UNIVERSE` as this broker actually spells it.

    Brokers decorate symbol names -- suffixes for account type, a dot, a
    trailing `m` for micro. Matching the literal string would silently return
    an empty list on a broker that appends anything at all, and an empty list
    reads as "no setups" rather than as "no symbols", which is the failure this
    run keeps producing in other forms.
    """

    def base(name: str) -> str:
        return "".join(ch for ch in name.upper() if ch.isalnum())

    catalogue = [
        item.name for item in connector.symbols() if not settings.instruments.is_ignored(item.name)
    ]

    def rank(name: str, key: str) -> tuple[int, int]:
        """Lower is a better match.

        Shortest-name-wins is NOT good enough and the counterexample is real:
        against `EURUSD`, the decorated `EURUSD.r` is EIGHT characters and the
        unrelated `EURUSDX` is seven, so length alone picks the wrong market
        and does it silently.

        What separates them is what FOLLOWS the key in the undecorated name: a
        broker's decoration starts with a separator (`.`, `-`, `_`, a space),
        another instrument's name continues with a letter.
        """
        if base(name) == key:
            return (0, len(name))
        tail = name.upper().replace(key, "", 1) if key in name.upper() else ""
        separated = bool(tail) and not tail[0].isalnum()
        return (1 if separated else 2, len(name))

    found: list[str] = []
    for wanted in CORE_UNIVERSE:
        key = base(wanted)
        matches = sorted(
            (name for name in catalogue if base(name).startswith(key)),
            key=lambda name: rank(name, key),
        )
        if matches:
            found.append(matches[0])
    return found


def _break_even_rule(settings) -> tuple[float, float] | None:
    """`(trigger_r, offset_r)` for the break-even move, or None if it is off.

    TWO TRIGGERS, AND THE SECOND IS THE ONE THAT BITES ON THIS ACCOUNT.
    `break_even_at_r` is 0.25, but `_is_account_meaningful` arms the same move
    as soon as the open profit clears `capital_protection_at_equity_pct` of
    equity -- one percent -- which on a EUR 215 account is EUR 2.15. Against
    the position sizes the minimum-lot override produces:

        risk 2%  (EUR  4.30)   arms at 0.50 R
        risk 4%  (EUR  8.60)   arms at 0.25 R
        risk 8%  (EUR 17.20)   arms at 0.125 R

    and 56% of last week's trades were forced to 4% or more. So the effective
    trigger is the SMALLER of the two, which is what this returns. Taking
    `break_even_at_r` alone would model a rule the account does not run.

    Equity does not appear here and that is not an omission: the euro trigger
    is `equity * share`, the risk is `equity * risk_pct`, and the equity
    cancels. The crossover is `share / risk_pct` at any account size.

    `break_even_offset_atr` is 0.10 ATR. Both live families are built on a
    one-ATR stop -- 0.85 beyond the level plus up to 0.15 of tolerance for
    `impulse_retest`, 1.00 for `order_block` -- so 0.10 ATR is 0.10R to within
    a rounding error, and the offset is expressed in R here. That
    approximation is worth naming: it is exact for `order_block` and up to 15%
    generous for `impulse_retest`.
    """
    config = settings.trade_management
    offset_atr = float(getattr(config, "break_even_offset_atr", 0.0) or 0.0)
    trigger = float(getattr(config, "break_even_at_r", 0.0) or 0.0)
    if trigger <= 0.0:
        return None
    share = float(getattr(config, "capital_protection_at_equity_pct", 0.0) or 0.0)
    risk_pct = settings.effective_risk_pct()
    if share > 0.0 and risk_pct > 0.0:
        # money >= equity * share/100  =>  r >= share / risk_pct
        trigger = min(trigger, share / risk_pct)
    return trigger, offset_atr


def _under_the_slot_cap(trades: list[Decision], slots: int) -> list[Decision]:
    """The trades that would actually have been opened, cap included.

    `max_concurrent_positions` is 4 on this account and `effective_max_positions`
    cuts it to 2 at this equity. The dry run enforced neither, so it reported a
    portfolio nobody could have held: on a morning when eleven markets break
    together it counted eleven trades where the account can hold two.

    Walked in time order because that is the only order in which the question
    "is a slot free" has an answer. A trade that never resolved holds its slot
    to the end, which is what an open position does.
    """
    if slots <= 0:
        return list(trades)
    open_until: list[datetime] = []
    taken: list[Decision] = []
    for trade in sorted(trades, key=lambda d: d.when):
        open_until = [stamp for stamp in open_until if stamp > trade.when]
        if len(open_until) >= slots:
            continue
        taken.append(trade)
        open_until.append(trade.exit_at or datetime.max.replace(tzinfo=trade.when.tzinfo))
    return taken


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


def build_parser() -> argparse.ArgumentParser:
    """Separate from `main` so a test can feed it the launcher's own command
    line without touching MT5.

    `--limit` was added by a string edit that silently matched nothing, so the
    flag was missing while the code that reads `args.limit` was already there
    and `dryrun.cmd` was already sending it. Two runs died on that. Nothing was
    checking that the launcher and the parser agree, and now something is.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--symbols", default="", help="comma list; default = the live universe")
    parser.add_argument("--csv", default="", help="write every decision to this file")
    parser.add_argument("--equity", type=float, default=0.0, help="override account equity")
    parser.add_argument(
        "--sweep",
        nargs="*",
        default=[],
        metavar="TF",
        help=(
            "timeframes to try each section on, space OR comma separated: "
            "--sweep M5 M15 M30. Both forms are accepted because cmd splits "
            "arguments on commas and the comma form silently loses the rest of "
            "the command line. "
            "The shipped timeframes were chosen on HistData; this asks the same "
            "question against THIS broker's spreads, which is the number that decides it."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only the first N markets (0 = every market). A quick first look.",
    )
    parser.add_argument(
        "--no-m1",
        action="store_true",
        help="skip M1 history (much faster on a large universe; coarser resolution)",
    )
    parser.add_argument(
        "--core",
        action="store_true",
        help=(
            "only the markets the research was done on -- eleven FX majors, gold, "
            "and four index CFDs. Sixteen instead of 232, and they are the ones "
            "that can actually confirm or refute the finding."
        ),
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help=(
            "each section on ITS OWN configured clock and nothing else. Overrides "
            "--sweep. One fifth of the work, which is what makes a month over the "
            "whole catalogue finishable."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

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

        # THE UNIVERSE THE SCANNER ACTUALLY WALKS, not `active_whitelist`.
        # The whitelist is four names; the live scan reads the broker's whole
        # catalogue filtered by asset class and the ignore list, which is why
        # the first run of this reported "4 symbols" against an account that
        # scans a couple of hundred.
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        elif args.core:
            symbols = _core_universe(connector, settings)
            print(f"core universe: {len(symbols)} markets -- {', '.join(symbols)}")
        else:
            # The scanner's OWN classifier, not an approximation of it. A
            # substring match on the folder name would quietly disagree with
            # the live filter, and then this would be measuring a different
            # universe from the one that trades.
            from scanner.universe import UniverseScanner

            wanted = {c.lower() for c in settings.scanner.priority_asset_classes}
            symbols = [
                item.name
                for item in connector.symbols()
                if not settings.instruments.is_ignored(item.name)
                and (not wanted or UniverseScanner._path_class(item.path).value in wanted)
            ]
            if args.limit:
                symbols = symbols[: args.limit]
        end = datetime.now(UTC)
        start = end - timedelta(days=args.days)
        fetch_these = tuple(tf for tf in NEEDED if not (args.no_m1 and tf is Timeframe.M1))

        # WARMUP IS MEASURED IN BARS, SO THE WINDOW IS PER TIMEFRAME.
        #
        # A single 27-day fetch gives M15 about 2,600 bars and H4 about 160 --
        # under the 270 the guard wants, so EVERY symbol was skipped "for want
        # of history" and the first run reported zero decisions on a working
        # account. 1.6x pads the weekends out of the calendar days.
        def _fetch_from(tf: Timeframe) -> datetime:
            bars_needed = (WARMUP + 20) * tf.duration
            return start - max(bars_needed * 1.6, timedelta(days=3))

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
        if args.live_only:
            args.sweep = []
        if args.sweep:
            # Accept "M5 M15" and "M5,M15" and any mix: cmd turns a comma list
            # into separate words before the script ever sees it, so both
            # arrive here and both have to work.
            wanted = [
                piece.strip().upper()
                for chunk in args.sweep
                for piece in str(chunk).split(",")
                if piece.strip()
            ]
            for name in sorted(live & set(module_config)):
                for tf in wanted:
                    passes.append((name, tf))
        else:
            for name in sorted(live & set(module_config)):
                passes.append((name, getattr(settings.analysis, name).timeframe))

        finest = Timeframe.M5 if args.no_m1 else Timeframe.M1
        results: dict[tuple[str, str], list[Decision]] = {key: [] for key in passes}
        skipped_symbols = 0
        manage = _break_even_rule(settings)

        for index, symbol in enumerate(symbols, 1):
            try:
                spec = connector.spec(symbol)
                frames = {
                    tf: fetch_mt5_history(connector, symbol, tf, _fetch_from(tf), end)
                    for tf in fetch_these
                }
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the run
                skipped_symbols += 1
                print(f"  [{index}/{len(symbols)}] {symbol}: no history ({exc})")
                continue
            # Only the clocks a pass will actually use need to be deep enough.
            # Requiring it of every fetched timeframe threw away symbols over a
            # frame nothing was going to read.
            used = {Timeframe.parse(tf) for _, tf in passes} | {finest}
            if any(len(frames.get(tf, [])) < WARMUP + 10 for tf in used if tf in frames):
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
                        pass_key=(name, tf_name),
                        manage=manage,
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
    _live_config_report(results, settings, equity, args.days)
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
                    "managed_r",
                    "managed_money",
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
                        "" if d.managed_r is None else round(d.managed_r, 3),
                        "" if d.managed_money is None else round(d.managed_money, 2),
                        d.note,
                    ]
                )
        print(f"\nevery decision written to {path}")


def _live_exit(decision: Decision, managed: bool) -> float | None:
    """The R this trade ACTUALLY produces under the settings that run.

    THE REPORT JUDGED THE WRONG COLUMN, and it changed the answer.

    `result_r` is the fixed-stop exit -- the one the research measured.
    `managed_r` is the same trade under `TradeManagementConfig`, and
    `break_even_at_r` is switched ON. So `managed_r` is what the account does
    and `result_r` is the counterfactual.

    Every headline, every sigma and every verdict was computed on `result_r`.
    On the 180-day run that read -2.00 R at -0.05 sigma and printed NOT ENOUGH
    TO CONCLUDE, while three lines below it the same report said the
    configuration that actually runs made +66.20 R. The owner was shown the
    number for a setup he does not trade.

    One function, so the two can no longer drift apart.
    """
    return decision.managed_r if managed else decision.result_r


def _live_config_report(results: dict, settings, equity: float, days: int) -> None:
    """WHAT THE ACCOUNT WOULD ACTUALLY HAVE DONE. Read this one.

    THE 30 AUGUST RUN REPORTED -1,120 EUR ON A 215 EUR ACCOUNT and that number
    was not the live configuration. It was the SWEEP TOTAL: both sections on
    five clocks at once, ten combinations that will never run together, with no
    position cap. The two rows that are the shipped configuration were roughly
    flat, and they were on the same screen, unlabelled, between eight rows that
    were not.

    A report that has to be disentangled before it can be read is a report that
    will be misread. So the live pair gets its own block: the section on the
    clock it is configured for, nothing else, under the account's own
    concurrent-position cap.
    """
    live = list(settings.analysis.confluence.live_enabled_modules)
    keys = []
    for name in sorted(live):
        section = getattr(settings.analysis, name, None)
        clock = getattr(section, "timeframe", None)
        if clock is not None and (name, clock) in results:
            keys.append((name, clock))
    if not keys:
        return

    slots = settings.effective_max_positions(equity)
    everything = [d for key in keys for d in results[key] if d.outcome == "TRADE"]
    trades = _under_the_slot_cap(everything, slots)
    managed = _break_even_rule(settings) is not None
    closed = [d for d in trades if _live_exit(d, managed) is not None]

    print("\n" + "=" * 78)
    print("THE LIVE CONFIGURATION -- this is the one that answers the question")
    print("=" * 78)
    print("  " + ", ".join(f"{name} on {clock}" for name, clock in keys))
    print(f"  max {slots} positions at once at EUR {equity:.2f}, one per symbol")
    print(
        "  exit: "
        + ("break-even stop, which is what the account runs" if managed else "fixed stop")
    )
    if len(everything) != len(trades):
        print(f"  {len(everything) - len(trades)} signals dropped: every slot was already busy")

    if not closed:
        print("\n  No resolved trades in this window.")
        return

    wins = [d for d in closed if (_live_exit(d, managed) or 0) > 0]
    total_r = sum(_live_exit(d, managed) or 0.0 for d in closed)
    money = sum((d.managed_money if managed else d.pnl_money) or 0.0 for d in closed)
    print(
        f"\n  {len(trades)} trades taken, {len(closed)} resolved"
        f"   ({len(trades) / max(days, 1):.1f} a day)"
    )
    print(f"  win rate            {len(wins) / len(closed):.1%}  ({len(wins)}/{len(closed)})")
    print(f"  total               {total_r:+.2f} R")
    print(f"  per trade           {total_r / len(closed):+.3f} R")
    print(f"  PROFIT              EUR {money:+.2f}   on EUR {equity:.2f} = {money / equity:+.1%}")
    for name, clock in keys:
        rows = [
            d for d in trades if d.pass_key == (name, clock) and _live_exit(d, managed) is not None
        ]
        if not rows:
            continue
        won = sum(1 for d in rows if (_live_exit(d, managed) or 0) > 0)
        row_r = sum(_live_exit(d, managed) or 0 for d in rows)
        row_money = sum((d.managed_money if managed else d.pnl_money) or 0 for d in rows)
        print(
            f"    {name:<16s} {clock:<4s} {len(rows):>4d} trades  "
            f"{won / len(rows):>5.1%} win  {row_r:+7.2f} R  EUR {row_money:+8.2f}"
        )

    _break_even_verdict(trades, settings)
    _is_this_real(trades, keys, managed)


def _is_this_real(
    trades: list[Decision], keys: list[tuple[str, str]], managed: bool = False
) -> None:
    """Is the sample big enough to conclude anything, and does it clear zero?

    THE QUESTION THE OWNER KEPT HAVING TO ASK ME. Every report so far has
    printed a win rate and a profit and left "does that mean it works" to be
    argued about afterwards. Seven days gave 82 trades and 59.3%, and the only
    honest answer to "is that good" was a paragraph. The answer belongs in the
    output.

    THREE THINGS DECIDE IT, and all three have to hold.

    1. SIZE. A win rate off 10 trades has a 95% interval running from roughly
       26% to 88%. The Wilson interval below says so rather than leaving it to
       be felt.

    2. SIGMA, MEASURED ON DAYS AND NOT ON TRADES. Sixteen markets breaking on
       the same morning are not sixteen independent observations, and treating
       them as such overstates significance by about the square root of the
       number that moved together -- roughly 4x here. This was the single
       largest correction in the original research and leaving it out of the
       live report would reintroduce the error one layer down. So the standard
       error comes from the spread of DAILY totals.

    3. STABILITY. The research's own bar was every year positive and every
       month positive. One good week inside a bad month is not an edge; it is
       where in the month you happened to look.
    """
    closed = [d for d in trades if _live_exit(d, managed) is not None]
    print("\n  IS THIS REAL? — the sample judging itself")
    print(
        "    judging the "
        + ("BREAK-EVEN exit, which is what the account runs" if managed else "fixed stop")
    )
    if len(closed) < 30:
        print(f"    {len(closed)} resolved trades. Not enough to say anything at all.")
        print("    Run a longer window: history.cmd 180")
        return

    wins = sum(1 for d in closed if (_live_exit(d, managed) or 0) > 0)
    n = len(closed)
    rate = wins / n
    # Wilson interval, which stays sane at small n where the normal one does not.
    z = 1.96
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    spread = z / (1 + z * z / n) * float(np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)))

    by_day: dict[object, float] = {}
    for d in closed:
        by_day[d.when.date()] = by_day.get(d.when.date(), 0.0) + (_live_exit(d, managed) or 0.0)
    daily = np.array(list(by_day.values()), dtype=float)
    total_r = float(daily.sum())
    days_traded = len(daily)
    # Standard error of the TOTAL, from the day-to-day spread.
    se = float(daily.std(ddof=1)) * float(np.sqrt(days_traded)) if days_traded > 1 else 0.0
    sigma = total_r / se if se > 0 else 0.0

    print(f"    {n} resolved trades over {days_traded} trading days")
    print(
        f"    win rate            {rate:.1%}"
        f"   95% interval {centre - spread:.1%} to {centre + spread:.1%}"
    )
    print(f"    per trade           {total_r / n:+.3f} R")
    print(f"    total               {total_r:+.2f} R,  {sigma:+.2f} sigma from zero")
    print("       (sigma measured on daily totals: markets that break together")
    print("        are one observation, not sixteen)")

    by_month: dict[str, list[float]] = {}
    for d in closed:
        by_month.setdefault(f"{d.when:%Y-%m}", []).append(_live_exit(d, managed) or 0.0)
    if len(by_month) > 1:
        print("\n    by month")
        for month in sorted(by_month):
            rows = by_month[month]
            won = sum(1 for r in rows if r > 0)
            print(
                f"      {month}   {len(rows):>5d} trades   {won / len(rows):>5.1%} win"
                f"   {sum(rows):+8.2f} R"
            )
    green = sum(1 for rows in by_month.values() if sum(rows) > 0)

    print("\n    VERDICT")
    checks = [
        (n >= 200, f"{n} trades", "at least 200 resolved trades"),
        (sigma >= 2.0, f"{sigma:+.2f} sigma", "at least +2.0 sigma from zero"),
        (
            len(by_month) >= 3,
            f"{len(by_month)} month(s) covered",
            "at least 3 months, so one good stretch cannot carry it",
        ),
        (
            len(by_month) > 1 and green == len(by_month),
            f"{green}/{len(by_month)} months positive",
            "every month positive, which is the bar the research itself met",
        ),
    ]
    for passed, actual, wanted in checks:
        print(f"      [{'x' if passed else ' '}] {wanted:<58} {actual}")
    if all(passed for passed, _a, _w in checks):
        print("\n    -> This clears every bar. On this broker's own data, over this")
        print("       window, these sections made money and it is not noise.")
    else:
        print("\n    -> NOT ENOUGH TO CONCLUDE. The boxes above say what is missing.")
        print("       A positive number here is encouraging and is not evidence.")
    for name, clock in keys:
        rows = [d for d in closed if d.pass_key == (name, clock)]
        if rows and len(rows) < 200:
            print(f"       {name} has only {len(rows)} trades of its own; it is unjudged.")


def _break_even_verdict(trades: list[Decision], settings) -> None:
    """Does moving the stop to break-even help or cost, on THESE trades?

    THE OWNER ASKED FOR THIS RULE AND IT WAS ALREADY ON. It is also the
    largest unmeasured deviation left between the shipped code and the
    research: 18,828 trades were resolved against a stop that DOES NOT MOVE,
    and both live sections enter at a level where price oscillates by
    construction.

    The arithmetic is not one-sided, which is exactly why it needs measuring
    rather than arguing. Break-even rescues a loser that ran a little first and
    scratches a winner that dipped before it ran. At a 68% hit rate it protects
    a third of the book and damages two thirds, so it pays only if losers are
    rescued appreciably more often than winners are scratched.

    Both columns come off ONE bar walk over the same entries, so this is a
    paired comparison and not two runs that might disagree for other reasons.
    """
    rule = _break_even_rule(settings)
    paired = [d for d in trades if d.result_r is not None and d.managed_r is not None]
    if rule is None or not paired:
        return
    trigger, offset = rule

    fixed = sum(d.result_r or 0.0 for d in paired)
    managed = sum(d.managed_r or 0.0 for d in paired)
    fixed_money = sum(d.pnl_money or 0.0 for d in paired)
    managed_money = sum(d.managed_money or 0.0 for d in paired)

    # COUNTED BY WHETHER THE TRADE CHANGED, NOT BY WHETHER IT CHANGED SIGN.
    #
    # These first read `result_r > 0 >= managed_r` and `result_r <= 0 <
    # managed_r`, and the 30 August run printed "0 scratched, 0 rescued" beside
    # a -0.20R difference -- a total that says trades moved next to two counts
    # that say none did. A scratched winner exits at entry PLUS the offset, so
    # it is still positive and the sign test could never see it. The two
    # numbers whose whole job was to explain the total were structurally
    # incapable of doing so.
    scratched = [d for d in paired if (d.managed_r or 0) < (d.result_r or 0) - 1e-9]
    rescued = [d for d in paired if (d.managed_r or 0) > (d.result_r or 0) + 1e-9]

    print("\n  BREAK-EVEN: does protecting the trade pay?")
    print(
        f"    arms at {trigger:.3f}R, stop to entry +{offset:.2f}R"
        f"   ({len(paired)} trades resolved both ways)"
    )
    print(
        f"    stop fixed          {fixed:+8.2f} R   EUR {fixed_money:+8.2f}"
        "   <- what the research measured"
    )
    print(
        f"    stop protected      {managed:+8.2f} R   EUR {managed_money:+8.2f}   <- what runs now"
    )
    print(
        f"    cut short           {len(scratched):>4d}"
        f" ({sum((d.managed_r or 0) - (d.result_r or 0) for d in scratched):+.2f} R)"
        f"     rescued  {len(rescued):>4d}"
        f" ({sum((d.managed_r or 0) - (d.result_r or 0) for d in rescued):+.2f} R)"
    )
    verdict = managed - fixed
    if abs(verdict) < 1e-9:
        print("    -> no difference on this sample")
    elif verdict > 0:
        print(f"    -> protecting the stop GAINED {verdict:+.2f} R here. Keep it on.")
    else:
        print(
            f"    -> protecting the stop COST {verdict:+.2f} R here "
            f"({verdict / max(len(paired), 1):+.3f} R a trade)."
        )
        print("       Turn break_even_at_r off for these families, or accept the cost knowingly.")


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
        # The SWEEP deliberately reads the fixed stop: it compares clocks
        # against each other, and a comparison is cleaner without the exit
        # rule moving underneath it. The LIVE block above reads the managed
        # exit, because that is the one the account trades.
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

    # REFUSED_CONFLUENCE AT 98.4% IS NOT A DIAGNOSIS, IT IS THE ABSENCE OF ONE.
    # The engine writes a sentence saying which gate refused and why, this
    # script already stores it in `note`, and the report threw it away and
    # printed the bucket name instead. The interesting half of the run was in
    # the column nobody totalled.
    #
    # Grouped on the leading words because the tail carries numbers -- "score
    # 38.8 below threshold" is one reason, not four thousand.
    detail = Counter(
        " ".join(d.note.split()[:6])
        for d in decisions
        if d.outcome == "REFUSED_CONFLUENCE" and d.note
    )
    if detail:
        print("\n   ...and what REFUSED_CONFLUENCE actually said:")
        for reason, count in detail.most_common(10):
            print(f"      {reason:<52} {count:>7}  {count / total:>6.1%}")

    _cost_report(decisions)
    _silence_report(decisions)

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


def _cost_report(decisions: list[Decision]) -> None:
    """HOW FAR OVER THE COST LIMIT, and it is the number the account turns on.

    The 30 August sweep put 2,832 FX setups in front of the sizer, across all
    five clocks, and took ONE trade. Every other one died on
    `SL_TOO_TIGHT_FOR_COSTS`. The report said so and stopped there -- which is
    useless, because 13% against a 12% limit and 30% against a 12% limit are
    completely different situations. One is a config decision; the other means
    this broker cannot carry this strategy on FX at any setting.

    The sizer already writes the figure into its refusal: "spread, commission
    and slippage would be 24% of the risk on a 9.6 pip stop". The report was
    grouping refusals on their first six words, so every one of those collapsed
    onto a single line and the percentage -- the only part that decides
    anything -- was discarded.

    THIS IS THE ONE NUMBER THE RESEARCH HAD TO ASSUME. It assumed 0.04 ATR, a
    4% share on a one-ATR stop, and reported +0.279R after charging it. The
    same arithmetic at other costs, using the research's own model:

        cost  4%   ->  +0.278 R      what was assumed
        cost 11%   ->  +0.138 R
        cost 22%   ->  -0.082 R      already negative
    """
    import re

    rows: list[tuple[float, str]] = []
    for d in decisions:
        if d.outcome != "SL_TOO_TIGHT_FOR_COSTS" or not d.note:
            continue
        found = re.search(r"be\s+(\d+(?:\.\d+)?)%\s+of the risk", d.note)
        if found:
            rows.append((float(found.group(1)) / 100.0, d.symbol))
    if not rows:
        return

    shares = np.array([share for share, _s in rows], dtype=float)
    print(f"\nTHE COST WALL — {len(rows)} setups refused because the stop is too tight")
    for q, label in ((0.10, "cheapest 10%"), (0.50, "median"), (0.90, "dearest 10%")):
        value = float(np.quantile(shares, q))
        # The research's own model: net = gross - 2 * cost, which reproduces
        # its published +0.279R at a 4% cost to three decimals.
        print(f"   {label:<14} {value:>6.1%} of the stop   ->  net {0.358 - 2 * value:+.3f} R")
    print("   (net uses the research's own cost model, which reproduces its")
    print("    published +0.279R at the 4% it assumed)")

    print("\n   how many would pass at a higher limit:")
    for limit in (0.12, 0.15, 0.20, 0.25):
        passing = int((shares <= limit).sum())
        net = 0.358 - 2 * float(shares[shares <= limit].mean()) if passing else 0.0
        verdict = "and would still pay" if net > 0.05 else "but would NOT pay"
        print(
            f"      limit {limit:>5.0%}   {passing:>6} of {len(rows)} "
            f"({passing / len(rows):>5.1%})   net {net:+.3f} R {verdict}"
        )
    print("   Raising the limit only helps if the row it admits is still positive.")

    worst = Counter(symbol for share, symbol in rows if share > 0.20)
    if worst:
        names = ", ".join(f"{s} ({c})" for s, c in worst.most_common(6))
        print(f"\n   over 20% of the stop, so unaffordable at any sane limit: {names}")


def _silence_report(decisions: list[Decision]) -> None:
    """WHICH MARKETS SAID NOTHING, and whether the module or a gate silenced them.

    THE 30 AUGUST CORE RUN TOOK 42 TRADES AND NOT ONE OF THEM WAS ON FX. All
    eleven majors -- the eleven the strategy was measured on, where the research
    says roughly 1.6 trades per pair per day -- produced zero in seven days.
    Everything that traded was gold and index CFDs, which is the half of the
    universe nothing was measured on.

    That is the single most important fact in the run and it appeared NOWHERE
    in the report. It was visible only as an absence: symbols 1 to 11 were
    missing from a progress log that prints a line per symbol that traded.
    A finding you can only reach by noticing which lines did not print is a
    finding that gets missed.

    `REFUSED_CONFLUENCE 98.4%` does not help either, because it merges two
    completely different diagnoses:

        the module never fired          -- no setup existed. The market was
                                           quiet, or the detector's own
                                           thresholds are wrong for this feed.
        the module fired and was refused -- a setup existed and a GATE took it.
                                           That is a configuration problem and
                                           it is fixable.

    `_one_pass` already records which modules scored on every refused decision,
    so the two are separable and were simply never separated.
    """
    per_symbol: dict[str, dict[str, int]] = {}
    for d in decisions:
        row = per_symbol.setdefault(d.symbol, {"bars": 0, "fired": 0, "trades": 0})
        row["bars"] += 1
        if d.outcome == "TRADE":
            row["trades"] += 1
            row["fired"] += 1
        elif d.module and d.module != "-":
            row["fired"] += 1
    if not per_symbol:
        return

    silent = [name for name, row in per_symbol.items() if row["trades"] == 0]
    print("\nPER MARKET — did the detector fire, or did a gate refuse it?")
    print(f"   {'symbol':<12} {'bars':>7} {'setups':>8} {'trades':>7}   {'':<28}")
    for name in sorted(per_symbol, key=lambda n: -per_symbol[n]["trades"]):
        row = per_symbol[name]
        if row["trades"]:
            verdict = ""
        elif row["fired"]:
            verdict = f"{row['fired']} setups formed, ALL refused by a gate"
        else:
            verdict = "the detector never fired at all"
        print(
            f"   {name:<12} {row['bars']:>7} {row['fired']:>8} {row['trades']:>7}   {verdict:<28}"
        )

    if silent:
        print(
            f"\n   {len(silent)} of {len(per_symbol)} markets took no trade at all."
            "\n   If those are the markets the strategy was MEASURED on, this run has"
            "\n   not tested the strategy -- it has tested an extrapolation of it."
        )


if __name__ == "__main__":
    main()

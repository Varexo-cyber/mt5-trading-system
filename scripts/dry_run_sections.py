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

# Module level: `_one_clock` prints a heartbeat during the bar walk and it is
# not inside `main`. The local import that used to live in `main` shadowed
# nothing and is gone.
import time as _time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analysis import ConfluenceEngine
from analysis.confluence import TradeIdea
from analysis.section_eleven_legs import LEGS_META_KEY, MIN_LEG_BARS, leg_bars
from analysis.target_reach import measure as measure_target_reach
from backtesting.replay import fetch_mt5_history
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from core.trade_origin import broker_comment
from core.types import Direction, MarketContext, Series, Tick, Timeframe, TradingMode
from filters.base import FilterContext
from filters.liveliness_filter import LivelinessFilter
from risk.position_sizer import PositionSizer
from runner.service import build_analysis_modules

RAW_BTC_SHADOW_SECTIONS = {
    "section_fifteen_btc_m1",
    "section_sixteen_btc_m5",
    "section_seventeen_btc_m15",
}
RAW_BTC_HORIZONS = {Timeframe.M1: 120, Timeframe.M5: 48, Timeframe.M15: 24}
RAW_BTC_MAX_SPREAD_R = {Timeframe.M1: 0.15, Timeframe.M5: 0.12, Timeframe.M15: 0.10}
RAW_BTC_EXECUTION_ALLOWANCE_R = 0.02


#: Bars of a section's OWN clock a trade is followed over. `_one_clock` uses
#: exactly this many, so the reach question the gate asks is the question the
#: trade actually faces: "does the target get hit inside the life this replay
#: gives the trade". A different horizon here would refuse entries on grounds
#: the resolution never applies, or admit ones it silently times out.
REACH_HORIZON = 96


def _historical_jarvis_gate(
    ctx, idea, spec, settings, clock: Timeframe, horizon: int | None = None
) -> tuple[str, str] | None:
    """Reproduce the live gates whose inputs exist in the historical bars.

    FIVE OF THE EIGHT GATES THE ACCOUNT RUNS, and the five is not a shortcut:
    they are exactly the ones whose inputs are IN the bars. Spread against stop
    width, market liveliness, the M1 volume spike, target reach and direction
    advantage all read price and volume history and nothing else. The news
    blackout needs a calendar archive nobody kept, and the AI review needs an
    answer that was never given -- and inventing either would be worse than
    saying they are missing.
    """
    risk = abs(idea.entry - idea.stop_loss)
    if risk <= 0.0:
        return "SPREAD_EATS_THE_STOP", "zero stop distance"

    spread_share = (ctx.tick.spread / risk) if ctx.tick is not None else float("inf")
    confluence = settings.analysis.confluence
    spread_limit = confluence.max_spread_share_of_stop
    for family, family_limit in confluence.max_spread_share_of_stop_by_family.items():
        if family in idea.setup_family:
            spread_limit = family_limit
            break
    if spread_share > spread_limit:
        return (
            "SPREAD_EATS_THE_STOP",
            f"spread {spread_share:.1%} of stop exceeds live {spread_limit:.1%} limit",
        )

    lively = LivelinessFilter(
        settings.filters.liveliness,
        lambda _symbol, timeframe: ctx.series[timeframe],
    ).check(
        FilterContext(
            symbol=ctx.symbol,
            spec=spec,
            now=ctx.now,
            direction=idea.direction,
            tick=ctx.tick,
            open_positions=(),
        )
    )
    if not lively.passed:
        return lively.reason.name, lively.detail

    volume = settings.filters.volume_spike
    minute = ctx.series.get(Timeframe.M1)
    if volume.enabled and minute is not None and "tick_volume" in minute.df.columns:
        values = minute.df["tick_volume"]
        if len(values) >= volume.lookback_bars + 1:
            baseline = float(values.iloc[-(volume.lookback_bars + 1) : -1].median())
            ratio = float(values.iloc[-1]) / baseline if baseline > 0 else 0.0
            if ratio >= volume.extreme_multiple:
                return (
                    "VOLUME_SPIKE",
                    f"last M1 volume {ratio:.1f}x median exceeds {volume.extreme_multiple:.1f}x",
                )

    frame = ctx.series[clock].df
    reward = abs(idea.take_profit - idea.entry)
    bars_ahead = horizon if horizon is not None else RAW_BTC_HORIZONS.get(clock, REACH_HORIZON)
    reach = measure_target_reach(
        frame.tail(400),
        distance=reward,
        bars_ahead=bars_ahead,
        long=idea.direction is Direction.LONG,
        reward_risk=reward / risk,
    )
    config = settings.analysis.confluence
    reach_is_advisory = any(
        family in idea.setup_family for family in config.target_reach_advisory_families
    )
    if reach.measured:
        required = min(100.0, reach.required_pct + config.target_reach_margin_pct)
        hard_floor = required * config.quick_reach_hard_floor_ratio
        direction_margin = max(
            config.direction_advantage_tolerance_pct,
            reach.standard_error_pct,
        )
        if not reach_is_advisory and reach.forward_pct < hard_floor:
            return (
                "TARGET_RARELY_REACHED",
                f"target reach {reach.forward_pct:.1f}% below quick hard floor {hard_floor:.1f}%",
            )
        if (
            not reach_is_advisory
            and config.require_direction_advantage
            and reach.opposite_pct - reach.forward_pct > direction_margin
        ):
            return (
                "TARGET_RARELY_REACHED",
                f"target reach {reach.forward_pct:.1f}% versus {reach.opposite_pct:.1f}% opposite",
            )
    return None


class _RawShadowEngine:
    """Convert one BTC detector firing directly into a replay-only idea."""

    def __init__(self, module, config) -> None:
        self.module = module
        self.config = config

    def evaluate(self, ctx: MarketContext, _mode: TradingMode) -> TradeIdea:
        signal = self.module.analyze(ctx)
        if not signal.score or signal.invalidation_price is None or ctx.tick is None:
            return TradeIdea(
                ctx.symbol,
                False,
                None,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                signal.reasoning or "no section signal",
                (signal,),
                setup_family=self.module.name,
                planning_timeframe=self.config.timeframe,
            )
        direction = Direction.LONG if signal.score > 0 else Direction.SHORT
        entry = float(ctx.tick.ask if direction is Direction.LONG else ctx.tick.bid)
        stop = float(signal.invalidation_price)
        risk = abs(entry - stop)
        if risk <= 0.0:
            return TradeIdea(
                ctx.symbol,
                False,
                None,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                "detector produced no usable stop",
                (signal,),
                setup_family=self.module.name,
                planning_timeframe=self.config.timeframe,
            )
        sign = 1.0 if direction is Direction.LONG else -1.0
        target = entry + sign * float(self.config.target_r) * risk
        return TradeIdea(
            ctx.symbol,
            True,
            direction,
            abs(float(signal.score)),
            float(signal.confidence),
            entry,
            stop,
            target,
            "raw BTC shadow detector firing",
            (signal,),
            setup_family=self.module.name,
            planning_timeframe=self.config.timeframe,
        )


#: Every timeframe a sweep may put a section on.
#:
#: M1 IS IN HERE, AND IT IS THE ONE CASE WHERE THE CLOCK AND THE RESOLUTION
#: FRAME ARE THE SAME. Resolving an M30 trade on M30 bars cannot see which
#: barrier a bar touched first, which is why every other clock is walked out
#: on something finer. Beneath M1 there is nothing, so an M1 trade is resolved
#: on M1 bars and a bar that spans both barriers is unknowable.
#:
#: `_resolve` already books that bar as a LOSS -- not as a win, not as a coin
#: flip. So the bias of an M1 row is PESSIMISTIC: it understates M1. A row
#: that comes back positive anyway means something; a row that comes back
#: negative may be the resolution and not the strategy, and `_sweep_report`
#: says so on the M1 line rather than leaving the reader to know it.
SWEEPABLE = (
    Timeframe.M1,
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.M30,
    Timeframe.H1,
    Timeframe.H4,
)
NEEDED = SWEEPABLE
#: Bars of history each decision may look back over.
WARMUP = 260


#: `--manage-grid`: `(label, trigger in R, where the stop goes in R)`.
#:
#: THE QUESTION IS WHETHER A LOOSER BREAK-EVEN BEATS NO BREAK-EVEN, and it
#: cannot be answered by turning one on and comparing two runs. Break-even
#: frees the symbol earlier, the freed symbol takes the next setup, and the
#: two runs then hold different trades -- so the comparison silently becomes
#: "these entries against those entries" instead of "this exit against that
#: exit". Every level here is resolved on the SAME entry.
#:
#: The stop goes to the entry itself at 0.0 and slightly into profit at 0.1.
#: Both matter: a stop exactly at entry is scratched by the spread on its way
#: past, which is why a locked tick exists at all, and it is also why the
#: locked variants take fewer trades to the target.
#:
#: TRIGGERS ARE IN R AND NOT IN PIPS. "Fifty pips toward the target" is a
#: different rule on every instrument and on every day -- on this account's
#: gold M1 stop it is roughly ten times the risk, which is past the target and
#: would never fire. R is the same distance in every market, which is the
#: whole reason this project measures in it.
@dataclass(frozen=True)
class ExitVariant:
    """One way of managing a trade after entry, resolvable on its own.

    THE POINT IS THAT EVERY VARIANT SEES THE SAME ENTRY. `_resolve` is called
    once per variant on the same bars from the same moment, so what the table
    compares is exit rules and not entry sets. The instant a variant were
    allowed to free its symbol early and take a different next setup, the
    columns would be comparing two strategies and the answer would mean
    nothing.

    Three shapes, and each row is exactly one of them:

        the fixed exit      no trigger, no management -- the entry stop and
                            the target, whichever price reaches first
        a break-even move   `trigger_r` and one of `lock_r` / `lock_atr`,
                            through `_resolve(manage=...)`
        a mechanism         `manage_fields`, through
                            `_resolve(full_management=...)`, with every OTHER
                            mechanism switched off so the row measures the
                            thing it is named after and nothing else

    LOCKS IN R AND LOCKS IN ATR ARE BOTH HERE ON PURPOSE. The live rule is
    `break_even_offset_atr` x the symbol's H1 ATR, which is not a fixed
    fraction of the stop: on section six that offset is about 0.44R while the
    same number on an H1 stop is nearer 0.10R. Measuring only R-locks would
    therefore answer a question the account does not ask, and measuring only
    ATR-locks would hide how much of the effect is just distance.
    """

    label: str
    #: Break-even trigger in R. 0.0 means the stop is never moved.
    trigger_r: float = 0.0
    #: Where the stop goes when it triggers, as a fraction of this trade's own
    #: risk. Mutually exclusive with `lock_atr`.
    lock_r: float = 0.0
    #: Where the stop goes, as a multiple of the symbol's H1 ATR -- the shape
    #: the live rule uses.
    lock_atr: float = 0.0
    #: `TradeManagementConfig` fields to switch back ON for this variant.
    #: Empty means no discretionary management at all.
    manage_fields: tuple[tuple[str, object], ...] = ()
    #: Whether the partial close may fire. Off for trailing rows so the trail
    #: is measured alone -- the two share `partial_close_at_r` as their arm.
    partial: bool = False

    @property
    def kind(self) -> str:
        if self.manage_fields:
            return "mechanism"
        return "break-even" if self.trigger_r > 0.0 else "fixed"

    def stop_offset(self, risk_price: float, atr: float) -> float:
        """The stop's distance above entry, in price, when the trigger fires."""

        if self.lock_atr:
            return self.lock_atr * atr
        return self.lock_r * risk_price


#: EVERY MECHANISM OFF. A mechanism variant switches back on exactly what it
#: is named for, so a row reading "trail 2.0 ATR" is the trail and not the
#: trail plus a profit lock plus a give-back rule that happened to be on.
#:
#: Each of these is a threshold the trade can never reach rather than a
#: missing field, because `_resolve` reads the field either way and a config
#: that omits it would silently fall back to the live default -- which is how
#: a "no break-even" row would quietly have contained one.
EVERY_MECHANISM_OFF: dict[str, object] = {
    "giveback_arm_r": 99.0,
    "peak_stall_arm_r": 99.0,
    "profit_lock_from_r": 99.0,
    "capital_protection_at_equity_pct": 0.0,
    "partial_close_at_r": 99.0,
    "trailing_mode": "none",
    "time_exit_hours": 1.0e6,
    "time_exit_uses_plan_horizon": False,
    "break_even_at_r": 99.0,
    "break_even_offset_atr": 0.0,
}

#: The break-even triggers the owner asked for, in R, plus the two the old
#: six-row grid already carried. R and not pips: "fifty pips toward the
#: target" is a different rule on every instrument and on every day -- on this
#: account's gold M1 stop it is roughly ten times the risk, so it is past the
#: target and would never fire.
#:
#: 1,25 / 1,50 / 2,00 ZIJN ER 7 SEPTEMBER BIJ GEKOMEN, en om een reden die het
#: waard is te onthouden. In de eerste 180-daagse run verloor sectie tien op
#: elke trigger tot en met 0,75 R en WON hij op alle drie de varianten van
#: 1,00 R -- de laatste rij van de tabel. Een uitkomst die precies op de rand
#: van een sweep ligt is meestal de sweep die ophoudt, niet het antwoord: het
#: optimum kan er net buiten liggen, en dan is 1,00 R alleen de dichtstbijzijnde
#: cel en niet de beste. Zonder deze drie is dat verschil niet te zien.
BREAK_EVEN_TRIGGERS: tuple[float, ...] = (
    0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00
)

#: Where the stop goes once the trigger fires. A stop exactly AT entry is
#: scratched by the spread on its way past, which is the whole reason a locked
#: tick exists; it is also why the locked rows carry fewer trades to target.
STOP_PLACEMENTS_CORE: tuple[tuple[str, float, float], ...] = (
    # (suffix, lock in R, lock in ATR)
    ("", 0.0, 0.0),
    (" +0.1R", 0.1, 0.0),
    (" +0.10A", 0.0, 0.10),
)
STOP_PLACEMENTS_WIDE: tuple[tuple[str, float, float], ...] = (
    *STOP_PLACEMENTS_CORE,
    (" +0.25A", 0.0, 0.25),
    (" +0.50A", 0.0, 0.50),
)


def _mechanism_variants(wide: bool) -> tuple[ExitVariant, ...]:
    """Trailing, partial and profit-lock rows, each isolated from the others."""

    rows: list[ExitVariant] = []
    trails = (1.0, 1.5, 2.0, 3.0) if wide else (1.5, 2.0)
    for arm in ((0.5, 1.0, 1.5) if wide else (1.0,)):
        for multiple in trails:
            rows.append(
                ExitVariant(
                    label=f"trail {multiple:.1f}A from {arm:.1f}R",
                    manage_fields=(
                        ("trailing_mode", "atr"),
                        ("trailing_atr_multiple", multiple),
                        ("partial_close_at_r", arm),
                    ),
                    partial=False,
                )
            )
    for at_r in ((0.75, 1.0, 1.5, 2.0) if wide else (1.0, 1.5)):
        for fraction in ((0.25, 0.5) if wide else (0.5,)):
            rows.append(
                ExitVariant(
                    label=f"part {fraction:.0%} @ {at_r:.2f}R",
                    manage_fields=(
                        ("partial_close_at_r", at_r),
                        ("partial_close_fraction", fraction),
                    ),
                    partial=True,
                )
            )
    for from_r in ((0.5, 0.7, 1.0) if wide else (0.7,)):
        rows.append(
            ExitVariant(
                label=f"lock 50% peak>{from_r:.2f}R",
                manage_fields=(
                    ("profit_lock_from_r", from_r),
                    ("profit_lock_fraction", 0.5),
                ),
            )
        )
    return tuple(rows)


def exit_grid(wide: bool = False) -> tuple[ExitVariant, ...]:
    """Every exit rule this replay compares, fixed first.

    `wide` is the owner's "meet ALLES": every stop placement against every
    trigger, and the trailing and partial sweeps at full width. It is a real
    cost -- each variant walks the bars again -- so the narrow grid is the
    default and the launcher says which one it ran.
    """

    placements = STOP_PLACEMENTS_WIDE if wide else STOP_PLACEMENTS_CORE
    rows: list[ExitVariant] = [ExitVariant(label="fixed SL/TP")]
    for trigger in BREAK_EVEN_TRIGGERS:
        for suffix, lock_r, lock_atr in placements:
            if lock_r >= trigger:
                # THE STOP WOULD LAND ON THE PRICE THAT ARMED IT. "Move to
                # +0.1R once the trade is +0.1R" is not a break-even rule; it
                # is an instant exit at the trigger, and it would have sat in
                # the table as a plausible-looking row that always scratches.
                # Only R-locks can be judged here -- an ATR lock's size against
                # this trade's risk is not known until the bar.
                continue
            rows.append(
                ExitVariant(
                    label=f"BE@{trigger:.2f}R{suffix}",
                    trigger_r=trigger,
                    lock_r=lock_r,
                    lock_atr=lock_atr,
                )
            )
    rows.extend(_mechanism_variants(wide))
    return tuple(rows)


#: Kept as the default so `--manage-grid` on its own behaves as before.
MANAGE_GRID: tuple[ExitVariant, ...] = exit_grid(wide=False)


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
    #: WHAT THIS TRADE COSTS PER LOT, and the broker's lot grid. Carried on the
    #: row because the compounding walk has to RE-SIZE each trade at the
    #: balance it would actually have had, and by report time the connector is
    #: shut and the spec is gone.
    #:
    #: Without these three the "compounded" number can only rescale a result
    #: that was sized once at the starting balance -- which silently assumes
    #: the account can always afford the trade. On EUR 252 with a 0.01 minimum
    #: lot that assumption is wrong in both directions: early trades the
    #: account could not have taken, and later ones where the lot step rounds
    #: the stake down.
    risk_per_lot: float = 0.0
    volume_min: float = 0.0
    volume_step: float = 0.0
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
    #: `--manage-grid` only: the same trade again at every break-even trigger
    #: in `MANAGE_GRID`, as `((label, r), ...)`. Measured on the SAME entry, so
    #: the levels are compared against each other and against the fixed exit
    #: without any of them getting a different set of trades to work with.
    grid_r: tuple[tuple[str, float | None], ...] = ()
    #: What the round trip cost this trade, as a fraction of its own stop.
    #:
    #: ALREADY SUBTRACTED from `result_r` and `managed_r`. Kept so the gross
    #: number stays recoverable, because every figure this script produced
    #: before 31 August was gross and the difference has to stay visible.
    cost_r: float = 0.0
    #: Causal short-trend reading at entry. -1 bearish, 0 mixed, +1 bullish.
    trend_m5: int = 0
    trend_m15: int = 0
    fault_grid_r: tuple[tuple[str, float | None], ...] = ()
    mae_r: float = 0.0
    mfe_r: float = 0.0
    spread_stop_pct: float = 0.0
    atr_regime: float = 0.0
    h1_distance_atr: float = 0.0
    h4_distance_atr: float = 0.0
    breakout_atr: float = 0.0
    retest_bars: int = 0
    breakout_body_atr: float = 0.0
    breakout_wick_share: float = 0.0
    breakout_volume_ratio: float = 0.0


def _trade_diagnostics(frame, start, end, idea, ctx, spread_price):
    """Causal entry features plus realised excursions; reporting only."""
    risk = abs(float(idea.entry) - float(idea.stop_loss))
    sign = 1.0 if idea.direction is Direction.LONG else -1.0
    last = end if end is not None else start
    walked = frame.iloc[
        int(frame.index.searchsorted(start, side="left")):
        int(frame.index.searchsorted(last, side="right"))
    ]
    if risk > 0 and not walked.empty:
        mfe = max(0.0, float((walked.high.max() - idea.entry) * sign / risk) if sign > 0 else float((idea.entry - walked.low.min()) / risk))
        mae = max(0.0, float((idea.entry - walked.low.min()) / risk) if sign > 0 else float((walked.high.max() - idea.entry) / risk))
    else:
        mae = mfe = 0.0
    def distance(tf):
        series = ctx.series.get(tf)
        if series is None or len(series.df) < 20:return 0.0
        close = series.df.close.astype(float); ema = close.ewm(span=20, adjust=False).mean(); unit = _atr_of(series.df)[-1]
        return float((idea.entry - ema.iloc[-1]) / unit) if unit and np.isfinite(unit) else 0.0
    own = next((s for s in idea.signals if s.module == "section_ten_gold_m1" and s.score), None)
    details = own.details if own is not None else {}
    clock_frame = ctx.series[Timeframe.parse(details.get("timeframe", "M5"))].df
    atrs = _atr_of(clock_frame); recent = atrs[-100:]; median = float(np.nanmedian(recent)); ratio = float(atrs[-1] / median) if median > 0 else 0.0
    return mae, mfe, (100.0 * spread_price / risk if risk else 0.0), ratio, distance(Timeframe.H1), distance(Timeframe.H4), float(details.get("break_atr", 0.0)), int(details.get("wait_bars", 0)), float(details.get("break_body_atr", 0.0)), float(details.get("break_wick_share", 0.0)), float(details.get("break_volume_ratio", 0.0))


def _short_trend(ctx: MarketContext, timeframe: Timeframe) -> int:
    """Closed-bar EMA20 direction: price side and three-bar slope must agree."""
    series = ctx.series.get(timeframe)
    if series is None or len(series.df) < 23:
        return 0
    close = series.df["close"].astype(float)
    ema = close.ewm(span=20, adjust=False).mean()
    side = float(close.iloc[-1] - ema.iloc[-1])
    slope = float(ema.iloc[-1] - ema.iloc[-4])
    if side > 0.0 and slope > 0.0:
        return 1
    if side < 0.0 and slope < 0.0:
        return -1
    return 0


def _fault_exit_grid(frame, start, idea, baseline_r, baseline_at, cost_r):
    """Closed-M5, two-factor thesis exits; shadow measurement only."""
    labels = (
        "LOSS@-0.15R", "LOSS@-0.25R", "LOSS@-0.35R",
        "GIVEBACK@0.50R", "GIVEBACK@0.75R", "GIVEBACK@1.00R", "SMART",
    )
    if baseline_r is None or baseline_at is None:
        return tuple((label, baseline_r) for label in labels)
    risk = abs(float(idea.entry) - float(idea.stop_loss))
    if risk <= 0 or frame.empty:
        return tuple((label, baseline_r) for label in labels)
    sign = 1.0 if idea.direction is Direction.LONG else -1.0
    first = int(frame.index.searchsorted(start, side="left"))
    last = int(frame.index.searchsorted(baseline_at, side="left"))
    visible = frame.iloc[max(0, first - 160):last]
    m5 = visible.resample("5min", label="right", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    m5 = m5[m5.index < baseline_at]
    if len(m5) < 24:
        return tuple((label, baseline_r) for label in labels)
    close = m5["close"].astype(float)
    ema = close.ewm(span=20, adjust=False).mean()
    prior_low = m5["low"].astype(float).shift(1).rolling(6).min()
    prior_high = m5["high"].astype(float).shift(1).rolling(6).max()
    peak_r, fired = 0.0, {}
    thresholds = {"LOSS@-0.15R": -0.15, "LOSS@-0.25R": -0.25, "LOSS@-0.35R": -0.35}
    first_live = max(23, int(m5.index.searchsorted(start, side="right")))
    for pos in range(first_live, len(m5)):
        price = float(close.iloc[pos])
        r_now = (price - float(idea.entry)) * sign / risk
        favourable = (
            float(m5["high"].iloc[pos]) - float(idea.entry)
            if sign > 0 else float(idea.entry) - float(m5["low"].iloc[pos])
        ) / risk
        peak_r = max(peak_r, favourable)
        slope = float(ema.iloc[pos] - ema.iloc[pos - 3])
        adverse_drift = (price < float(ema.iloc[pos]) and slope < 0) if sign > 0 else (
            price > float(ema.iloc[pos]) and slope > 0
        )
        structure_broken = price < float(prior_low.iloc[pos]) if sign > 0 else (
            price > float(prior_high.iloc[pos])
        )
        if not (adverse_drift and structure_broken):
            continue
        for label, threshold in thresholds.items():
            if label not in fired and r_now <= threshold:
                fired[label] = r_now - cost_r
        givebacks = {
            "GIVEBACK@0.50R": (0.50, 0.00),
            "GIVEBACK@0.75R": (0.75, 0.10),
            "GIVEBACK@1.00R": (1.00, 0.25),
        }
        for label, (needed_peak, floor) in givebacks.items():
            if label not in fired and peak_r >= needed_peak and r_now <= floor:
                fired[label] = r_now - cost_r
        profit_break = peak_r >= 0.50 and r_now <= 0.00
        if "SMART" not in fired and (r_now <= -0.25 or profit_break):
            fired["SMART"] = r_now - cost_r
    return tuple((label, fired.get(label, baseline_r)) for label in labels)


def _context(
    symbol: str,
    frames: dict,
    upto: datetime,
    spread: float,
    slices: dict | None = None,
    legs: dict | None = None,
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
    # The decision price must come from the finest history being replayed.
    # Hard-coding M5 made an M1 pass wake every minute while still seeing a
    # price up to four minutes old.  The detector then missed live M1 zones
    # that the real runner (which uses a tick) took, so the alleged exact
    # replay could not reproduce the account's own trades.
    finest = min(series, key=lambda item: item.duration)
    price = float(series[finest].df["close"].iloc[-1])
    half = spread / 2.0
    ctx = MarketContext(symbol, upto, series, Tick(symbol, upto, price - half, price + half))
    if legs:
        # SECTION ELEVEN'S TWO LEGS, cut to the same instant as everything
        # else here. Live they arrive through `context.meta` because a
        # module may not reach a second instrument; the replay honours the
        # same constraint rather than handing the module a shortcut that
        # does not exist on the account.
        #
        # CLOSED BARS ONLY, on the identical `open + duration <= upto`
        # rule the loop above uses. A leg cut one bar later than the cross
        # is a look-ahead of exactly the size of the lag being traded, and
        # it would flatter the result in the one direction that matters.
        payload: dict = {}
        for canonical, (timeframe, frame) in legs.items():
            cut = int(frame.index.searchsorted(upto - timeframe.duration, side="right"))
            if cut < MIN_LEG_BARS:
                payload = {}
                break
            payload[canonical] = leg_bars(canonical, frame.iloc[cut - WARMUP : cut], upto)
        if payload:
            ctx.meta[LEGS_META_KEY] = payload
    return ctx


def _variant_management(base, variant: ExitVariant):
    """`base` with every mechanism off except the one this variant names.

    VALIDATED, NOT JUST COPIED. `model_copy(update=...)` writes whatever it is
    handed straight past pydantic -- that is how a value outside a field's own
    bounds gets accepted here and then misbehaves inside the bar walk as a
    threshold that is quietly never reached. Re-validating turns that into a
    crash on the first call instead of a column of plausible numbers.

    NOT `lru_cache`d, and that is not an oversight: a pydantic model with a
    dict field is unhashable, so the decorator raises on the first trade. The
    caller memoises per section instead, which is where it can key on the
    section whose settings these are.
    """

    fields = {**base.model_dump(), **EVERY_MECHANISM_OFF, **dict(variant.manage_fields)}
    return type(base).model_validate(fields)


def _horizon_window(index, start, horizon_bars: int) -> tuple[int, int]:
    """`(first, last)` bar a trade opened at `start` is walked over.

    ONE DEFINITION, because the caller needs the same boundary `_resolve` uses
    and computing it twice is how the two drift. `last` is exclusive.
    """
    first = int(index.searchsorted(start, side="left"))
    return first, min(first + horizon_bars, len(index))


def _resolve(
    frame: pd.DataFrame,
    start: datetime,
    idea,
    horizon_bars: int,
    manage=None,
    arrays=None,
    force_close_at: datetime | None = None,
    close_at_horizon: bool = False,
    full_management=None,
    management_atr: float = 0.0,
    management_spread: float = 0.0,
    risk_money: float = 0.0,
    equity: float = 0.0,
    planned_minutes: float | None = None,
    partial_possible: bool = False,
    near_target_tolerance: float = 0.0,
    near_target_max_share: float = 0.0,
):
    """First touch of stop or target on the bars after entry.

    Same rules as the research: the entry bar itself counts, a bar spanning
    both barriers is a LOSS because the order is unknowable, and a trade that
    reaches neither is reported as open rather than closed at the clock.

    Returns `(r, exit_time, managed_r, managed_exit_time)`. A trade that
    resolves neither way returns Nones and holds the symbol to the end of the
    window, which is what a real open position does.

    TWO EXIT TIMES, AND THE SECOND ONE IS NOT DECORATION. Whichever exit the
    account actually takes is the one that frees the symbol for the next
    setup. Only `exit_time` used to be recorded -- the FIXED stop -- so a trade
    that scratched at break-even after four minutes went on occupying its
    symbol until the fixed stop or target resolved, sometimes hours later, and
    every setup in between was silently dropped.

    That is not a rounding error on a fast section. It under-counts trades in
    exact proportion to how often break-even fires and how much earlier it
    fires, which on an M1 strategy is most trades and by a lot.

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
    # SEARCHSORTED, NOT A MASK, and this is the same defect I already fixed
    # once in `_context`. `frame[frame.index >= start]` walks the WHOLE
    # resolution frame -- 52,000 M5 bars over 180 days -- builds a boolean
    # array of that length, copies every matching row, and then keeps the
    # first `horizon_bars` of it. Per trade. `.iterrows()` on the result then
    # costs about 10us a row on top.
    #
    # The bars are sorted, so the start is a binary search and the horizon is
    # a positional slice. The numpy arrays are handed in by the caller because
    # they are the same for every trade on a symbol; extracting them here
    # would rebuild them thousands of times.
    if arrays is None:
        arrays = (
            frame.index,
            frame["high"].to_numpy(),
            frame["low"].to_numpy(),
            frame["close"].to_numpy(),
        )
    if len(arrays) == 3:
        index, highs, lows = arrays
        closes = frame["close"].to_numpy()
    else:
        index, highs, lows, closes = arrays
    first, last = _horizon_window(index, start, horizon_bars)
    if force_close_at is not None:
        last = min(int(index.searchsorted(force_close_at, side="left")) + 1, len(index))
    if first >= last:
        return None, None, None, None
    long = idea.direction is Direction.LONG
    risk = abs(idea.entry - idea.stop_loss)

    # THE DOORSTEP CLOSE, MODELLED WHERE THE TRADE IS RESOLVED.
    #
    # Live, `PositionManager._close_at_the_doorstep` takes a winner that has
    # arrived within a hair of its target rather than waiting for the last
    # tick. If that only existed live, `hoeveel.cmd` would keep reporting the
    # old exit and the owner would be measuring a system he is not running --
    # which is this repository's most repeated defect, and the reason the
    # section-six number had to be thrown away once already.
    #
    # The target moves NEARER by the tolerance, and the R credited moves with
    # it. Crediting the full reward for an exit taken short of the target would
    # pay the replay for money the account never receives.
    target = idea.take_profit
    if near_target_tolerance > 0.0 and risk > 0:
        reward_price = abs(idea.take_profit - idea.entry)
        room = min(near_target_tolerance, near_target_max_share * reward_price)
        if 0.0 < room < reward_price:
            target = idea.take_profit - room if long else idea.take_profit + room
    reward_r = (abs(target - idea.entry) / risk) if risk > 0 else 0.0

    # The managed run walks the same bars with a stop that is allowed to move.
    # `armed` is one-way: a stop that has been pulled up is never pushed back.
    managed_open = (manage is not None or full_management is not None) and risk > 0
    managed_r: float | None = None
    managed_stop = idea.stop_loss
    armed = False
    # OFFSET IN PRICE, not in R. The caller converts the configured ATR
    # multiple using the symbol's H1 ATR at entry, because that is the input
    # `PositionManager._atr_offset` reads live. Passing an R multiple here
    # understated the real distance by 2.4x to 4.4x depending on the section's
    # clock -- see `_break_even_rule`.
    trigger_r, offset_price = manage or (0.0, 0.0)
    direction_sign = 1.0 if long else -1.0
    peak_r = 0.0
    peak_at = start
    remaining = 1.0
    realised_r = 0.0

    fixed_r: float | None = None
    exit_at: datetime | None = None
    managed_at: datetime | None = None

    for position in range(first, last):
        bar_high = float(highs[position])
        bar_low = float(lows[position])
        hit_stop = bar_low <= idea.stop_loss if long else bar_high >= idea.stop_loss
        hit_target = bar_high >= target if long else bar_low <= target

        if managed_open:
            # ORDER MATTERS AND IT IS NOT THE FLATTERING ONE. Within a bar the
            # sequence is unknowable, so the managed stop is checked against
            # the price it already sits at BEFORE this bar's excursion is
            # allowed to arm it. Arming on the same bar's high and then
            # surviving that bar's low is the look-ahead this account has
            # already been bitten by twice.
            managed_hit = bar_low <= managed_stop if long else bar_high >= managed_stop
            if managed_hit and hit_target:
                # Both in one bar: the order is unknowable, so take the stop.
                managed_r = realised_r + remaining * (
                    (managed_stop - idea.entry) / risk * direction_sign
                )
            elif managed_hit:
                managed_r = realised_r + remaining * (
                    (managed_stop - idea.entry) / risk * direction_sign
                )
            elif hit_target:
                managed_r = realised_r + remaining * reward_r
            if managed_r is not None:
                managed_open = False
                managed_at = index[position]
            else:
                excursion = (bar_high - idea.entry) if long else (idea.entry - bar_low)
                excursion_r = excursion / risk
                close_r = (float(closes[position]) - idea.entry) / risk * direction_sign
                if excursion_r > peak_r + 1e-9:
                    peak_r, peak_at = excursion_r, index[position]

                if full_management is not None:
                    age_minutes = (index[position] - start).total_seconds() / 60.0
                    # A soft give-back depends on the live health reader.  In
                    # OHLC replay we conservatively assume HEALTHY, exactly the
                    # branch that holds longest, and apply only the mechanical
                    # hard backstop.
                    if (
                        age_minutes >= full_management.min_discretionary_exit_minutes
                        and full_management.giveback_arm_r > 0
                        and full_management.giveback_hard_fraction > 0
                        and peak_r >= full_management.giveback_arm_r
                        and peak_r > 0
                        and (peak_r - close_r) / peak_r >= full_management.giveback_hard_fraction
                    ):
                        managed_r = realised_r + remaining * close_r
                        managed_open = False
                        managed_at = index[position]
                    # Healthy positions get the same doubled peak-stall wait
                    # as the real manager.  No invented AI verdict is used.
                    stall_wait = (
                        max(
                            full_management.peak_stall_minutes,
                            (planned_minutes or 0.0) * full_management.peak_stall_share_of_horizon,
                        )
                        * 2.0
                    )
                    if (
                        managed_open
                        and stall_wait > 0
                        and age_minutes >= full_management.min_discretionary_exit_minutes
                        and peak_r >= full_management.peak_stall_arm_r
                        and close_r >= peak_r * full_management.peak_stall_near_peak
                        and (index[position] - peak_at).total_seconds() / 60.0 >= stall_wait
                    ):
                        managed_r = realised_r + remaining * close_r
                        managed_open = False
                        managed_at = index[position]

                    deadline_hours = full_management.time_exit_hours
                    if deadline_hours is not None and full_management.time_exit_uses_plan_horizon:
                        plan_hours = max(
                            full_management.time_exit_minimum_hours,
                            (planned_minutes or deadline_hours * 60.0)
                            / 60.0
                            * full_management.time_exit_horizon_multiple,
                        )
                        deadline_hours = min(deadline_hours, plan_hours)
                    timed_out_now = (
                        managed_open
                        and deadline_hours is not None
                        and age_minutes >= deadline_hours * 60
                    )
                    if timed_out_now and (
                        abs(close_r) < full_management.time_exit_min_abs_r
                        or (close_r > 0 and peak_r < full_management.time_exit_stale_peak_r)
                    ):
                        managed_r = realised_r + remaining * close_r
                        managed_open = False
                        managed_at = index[position]

                    # Partial close is only possible when both halves satisfy
                    # the broker's minimum lot.  Most 0.01-lot BTC trades do
                    # not, which is why pretending every trade halves at 1.5R
                    # would overstate Jarvis rather than reproduce it.
                    if (
                        managed_open
                        and partial_possible
                        and remaining == 1.0
                        and excursion_r >= full_management.partial_close_at_r
                    ):
                        fraction = full_management.partial_close_fraction
                        realised_r += fraction * full_management.partial_close_at_r
                        remaining -= fraction

                    # Same broker-resident profit lock: retain a fraction of
                    # the peak, but leave at least the configured spread gap.
                    meaningful_r = (
                        equity
                        * full_management.capital_protection_at_equity_pct
                        / 100.0
                        / risk_money
                        if equity > 0
                        and risk_money > 0
                        and full_management.capital_protection_at_equity_pct > 0
                        else float("inf")
                    )
                    if peak_r >= min(full_management.profit_lock_from_r, meaningful_r):
                        secured_r = peak_r * full_management.profit_lock_fraction
                        if management_spread > 0:
                            secured_r = min(
                                secured_r,
                                peak_r
                                - full_management.profit_lock_minimum_spreads
                                * management_spread
                                / risk,
                            )
                        if secured_r > 0:
                            candidate = idea.entry + secured_r * risk * direction_sign
                            if (long and candidate > managed_stop) or (
                                not long and candidate < managed_stop
                            ):
                                managed_stop = candidate

                    # The real ATR trail starts at the partial-close threshold
                    # even when minimum lot makes the partial itself inert.
                    if (
                        managed_open
                        and full_management.trailing_mode == "atr"
                        and excursion_r >= full_management.partial_close_at_r
                        and management_atr > 0
                    ):
                        candidate = float(closes[position]) - (
                            direction_sign * management_atr * full_management.trailing_atr_multiple
                        )
                        if (long and candidate > managed_stop) or (
                            not long and candidate < managed_stop
                        ):
                            managed_stop = candidate

                    # Break-even follows the profit rules in the live manager.
                    trigger = min(full_management.break_even_at_r, meaningful_r)
                    offset = management_atr * full_management.break_even_offset_atr
                    if managed_open and excursion_r >= max(trigger, offset / risk):
                        candidate = idea.entry + offset * direction_sign
                        if (long and candidate > managed_stop) or (
                            not long and candidate < managed_stop
                        ):
                            managed_stop = candidate

                # A PROTECTIVE STOP CANNOT BE PLACED BEYOND THE PRICE. Live the
                # broker refuses it and `_worth_moving` never gets there; the
                # simulator happily armed a stop above the market and then
                # "filled" it on the same bar, inventing profit that no order
                # could have taken. So the move needs the excursion to cover
                # the offset as well as the trigger.
                reach = max(trigger_r * risk, offset_price)
                if full_management is None and not armed and excursion >= reach:
                    armed = True
                    managed_stop = idea.entry + offset_price * direction_sign

        if fixed_r is None:
            if hit_stop:
                fixed_r, exit_at = -1.0, index[position]
            elif hit_target:
                fixed_r, exit_at = reward_r, index[position]

        # Reproduce the live pre-pause market close for selected comments.
        forced_now = force_close_at is not None and index[position] >= force_close_at
        if forced_now:
            close_r = (float(closes[position]) - idea.entry) / risk * direction_sign
            if fixed_r is None:
                fixed_r, exit_at = close_r, index[position]
            if managed_open:
                managed_r, managed_at = realised_r + remaining * close_r, index[position]
                managed_open = False
        if fixed_r is not None and not managed_open:
            break

    if close_at_horizon and first < last:
        horizon_at = index[last - 1]
        close_r = float(
            np.clip(
                (float(closes[last - 1]) - idea.entry) / risk * direction_sign,
                -1.0,
                reward_r,
            )
        )
        if fixed_r is None:
            fixed_r, exit_at = close_r, horizon_at
        if managed_open:
            managed_r, managed_at = realised_r + remaining * close_r, horizon_at
            managed_open = False
    if managed_open:
        managed_r = None
        managed_at = None
    return fixed_r, exit_at, managed_r, managed_at


def _unresolvable_clocks(clocks: tuple[str, ...], finest: Timeframe) -> str:
    """The complaint to raise before the fetch, or "" if every clock is fine.

    A clock has to be walked out on bars at least as fine as itself. `--no-m1`
    makes M5 the finest frame, so `--no-m1 --sweep M1` asks for something that
    cannot be measured at all.

    IT USED TO BE DROPPED WITH A BARE `continue` -- no row, no zero, no line.
    An absent row in a sweep table reads exactly like "we looked and found
    nothing there", and this file has now shipped that same confusion under
    six different names. Refusing loudly, before twenty minutes of fetching,
    is the whole point.
    """
    too_fine = sorted(
        {tf for tf in clocks if Timeframe.parse(tf).duration < finest.duration},
        key=lambda tf: Timeframe.parse(tf).duration,
    )
    if not too_fine:
        return ""
    return (
        f"{', '.join(too_fine)} cannot be resolved on {finest.value} bars -- a trade has "
        f"to be walked out on bars at least as fine as the clock that produced it. "
        f"Drop --no-m1 so M1 history is fetched, or drop {', '.join(too_fine)} from --sweep."
    )


def _frames_read(
    settings, clock: Timeframe, finest: Timeframe, sections: tuple[str, ...]
) -> tuple[Timeframe, ...]:
    """Only the clocks something actually reads, so `_context` stops slicing
    frames nobody looks at.

    An M15 pass on this account reads M5 (the price `_context` itself takes),
    M15 (the intraday planning timeframe and the clock) and H4 (the higher
    timeframe veto). M30 and H1 were being sliced, wrapped in a Series and
    handed over on every bar for nothing.

    RESOLVED PER SECTION, not by unioning every horizon profile. A first draft
    took the union and got M5, M15, H1, H4, D1 and W1 -- everything, because
    the swing profile plans on H1 and vetoes on W1. It saved nothing. Both live
    families classify as intraday, so only that profile's frames are needed,
    and `_classify_horizon`'s own membership lists decide which profile applies
    rather than an assumption about it.

    The entry-timing frames used to belong here too. They do not any more:
    `entry_timing_exempt_families` covers both live families, so that gate no
    longer runs for them -- a saving that fell out of a correctness fix rather
    than being chased.
    """
    confluence = settings.analysis.confluence
    wanted: set[Timeframe] = {clock, finest, Timeframe.M5}
    if "section_seven_gold_smc" in sections:
        wanted.update({Timeframe.M15, Timeframe.M30, Timeframe.H1, Timeframe.H4})
    for name in sections:
        if any(family in name for family in confluence.strategy_owned_entry_families):
            # A standalone section owns its trigger and stop. Loading an H4
            # frame for it would silently reintroduce the generic trend veto
            # that strategy ownership explicitly disables.
            continue
        if name in confluence.quick_modules:
            horizon = "quick"
        elif name in confluence.intraday_modules:
            horizon = "intraday"
        else:
            horizon = "swing"
        profile = confluence.horizon_profiles[horizon]
        for timeframe in (profile.planning_timeframe, *profile.htf_trend_timeframes):
            try:
                wanted.add(Timeframe.parse(timeframe))
            except (KeyError, ValueError):
                # D1 and W1 are named by the swing profile and never fetched
                # here; a frame that is absent is simply skipped downstream.
                continue
    return tuple(sorted(wanted, key=lambda tf: tf.duration))


def _hopeless_on_cost(sizer, spec, frames: dict, clocks: tuple[Timeframe, ...]) -> float | None:
    """The best cost share this symbol can reach, or None if it can trade.

    ELEVEN OF SIXTEEN CORE MARKETS PRODUCE ZERO TRADES and cost two thirds of
    the run. Every FX setup on this account dies on
    `SL_TOO_TIGHT_FOR_COSTS` -- the 180-day run refused 6,877 of them at a
    median 51% of the stop -- and the sweep walked every bar of all eleven
    anyway to arrive at that same refusal one bar at a time.

    So the cost is checked ONCE per symbol, on the widest clock available,
    using the sizer's own `_cost_share` -- the same function that will refuse
    the trades. If even the widest stop leaves nothing payable, there is
    nothing to learn from walking 12,000 bars to watch it be refused.

    Returns the share when the symbol is hopeless so the report can name it,
    and None when at least one clock is affordable. Deliberately generous: it
    skips only what is over TWICE the account's limit, because a symbol near
    the boundary is exactly the interesting case and must still be measured.
    """
    limit = sizer.settings.risk.max_cost_share_of_risk
    if limit <= 0:
        return None
    best = None
    for clock in clocks:
        frame = frames.get(clock)
        if frame is None or len(frame) < 60:
            continue
        atr = float(np.nanmedian(_atr_of(frame)))
        if not np.isfinite(atr) or atr <= 0:
            continue
        commission = sizer.settings.risk.commission_per_lot(spec.asset_class.value)
        share = sizer._cost_share(spec, atr, commission)
        best = share if best is None else min(best, share)
    if best is None:
        return None
    return best if best > limit * 2.0 else None


def _atr_of(frame: pd.DataFrame, period: int = 14) -> np.ndarray:
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean().to_numpy()


def _one_clock(
    *,
    sections: list,
    symbol: str,
    spec,
    frames: dict,
    start: datetime,
    end: datetime,
    equity: float,
    clock: Timeframe,
    resolve_on: Timeframe,
    needed: tuple[Timeframe, ...] | None = None,
    manage_grid: tuple[ExitVariant, ...] = (),
    fault_exit_grid: bool = False,
    legs: dict | None = None,
) -> dict:
    """Every section that reads one clock, walked ONCE.

    `sections` is `(name, engine, sizer, management, flatten, raw_shadow)` rows.

    WHY ONE WALK INSTEAD OF ONE PER SECTION. A `MarketContext` depends on the
    symbol, the instant and the bars -- not on which module is about to read
    it. Building it per section meant the sweep constructed every context
    TWICE, once for impulse_retest and once for order_block, and context
    building is the single largest cost in the loop. A 180-day sweep was
    ninety minutes on the owner's VPS and half of that was rebuilding
    identical objects.

    ONE POSITION PER SYMBOL AT A TIME, tracked PER SECTION. That is
    `Reason.POSITION_ALREADY_OPEN`, and without it this loop takes a fresh
    trade on every bar a setup stays valid -- which is not merely an
    over-count, it is biased. A retest that works leaves the level in a bar and
    yields one entry; one that fails sits on it and yields five. Duplicates are
    drawn from the losers. Sharing one `busy_until` across sections would be a
    different error, refusing section three a trade because section two is in
    one, so each keeps its own.

    `resolve_on` must be FINER than the clock, or equal to it on M1 where
    nothing finer exists. Resolving an M30 trade on M30 bars cannot tell which
    barrier came first, and assuming the good one is how a backtest lies --
    which is why the equal case is only allowed where there is no alternative,
    and why `_resolve` books an ambiguous bar as a LOSS rather than a win. On
    M1 that makes the row pessimistic, and `_sweep_report` prints that caveat
    beside the number instead of leaving it to be remembered.
    """
    out: dict = {row[0]: [] for row in sections}
    busy: dict = {row[0]: None for row in sections}
    # `(section, variant label) -> TradeManagementConfig`, built on first use.
    variant_configs: dict[tuple[str, str], object] = {}
    #: Per section, trades that reached neither barrier inside their horizon.
    #: Counted and PRINTED, because a timeout used to silently take the
    #: section out of the run and nothing anywhere said so.
    timed_out: dict = {}
    bars = frames[clock]
    window = bars[(bars.index >= start) & (bars.index <= end)]
    horizon = int(96 * clock.duration / resolve_on.duration)
    reading = frames if needed is None else {tf: frames[tf] for tf in needed if tf in frames}
    #: Per clock, so it never outlives the frames it points into.
    slices: dict = {}
    # ONCE PER CLOCK, not once per bar and not once per trade.
    #
    # `window.loc[bar_time].get("spread", 0)` is a label lookup plus a Series
    # construction on every bar, and `_resolve` was masking the whole 52,000-row
    # M5 frame on every trade. Both are the same shape of mistake as the one
    # already fixed in `_context`: work proportional to the history, repeated
    # per step.
    spreads = (
        window["spread"].to_numpy(dtype=float)
        if "spread" in window.columns
        else np.zeros(len(window))
    )
    resolve_frame = frames[resolve_on]
    resolve_arrays = (
        resolve_frame.index,
        resolve_frame["high"].to_numpy(),
        resolve_frame["low"].to_numpy(),
        resolve_frame["close"].to_numpy(),
    )

    # THE H1 ATR, because that is the one live reads.
    #
    # `ExecutionManager._atr_offset` multiplies `break_even_offset_atr` by
    # `_compute_atr`, and that function's own docstring says "Recent H1
    # volatility" -- H1 on every symbol and every section, whatever clock the
    # section trades on. Built once per symbol here and looked up per trade,
    # because rebuilding a 14-bar rolling mean inside the bar loop is the
    # shape of waste this file has already been rewritten twice for.
    #
    # Absent H1 means the offset is unknown, and live refuses to move the stop
    # at all in that case (`_atr_offset` returns None). The lookup below
    # returns 0.0 and the caller reads that the same way.
    hourly = frames.get(Timeframe.H1)
    if hourly is not None and len(hourly) > 15:
        _previous = hourly["close"].shift(1)
        _spans = pd.concat(
            [
                hourly["high"] - hourly["low"],
                (hourly["high"] - _previous).abs(),
                (hourly["low"] - _previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_index = hourly.index
        atr_values = _spans.rolling(14).mean().to_numpy()
    else:
        atr_index, atr_values = None, None

    def _hourly_atr(when: datetime) -> float:
        """The last CLOSED H1 ATR before `when`, or 0.0 for "not known"."""
        if atr_index is None:
            return 0.0
        cut = int(atr_index.searchsorted(when, side="right")) - 1
        if cut < 0:
            return 0.0
        value = float(atr_values[cut])
        return value if np.isfinite(value) and value > 0.0 else 0.0

    # A HEARTBEAT, BECAUSE TWENTY-TWO MINUTES OF SILENCE IS NOT DISTINGUISHABLE
    # FROM A HANG.
    #
    # The per-symbol line only prints when a symbol FINISHES, and one metal on
    # M1 over 180 days is 235,000 bars and about twenty minutes. The owner sat
    # in front of a blank window for half an hour on the six-metal run with no
    # way to tell whether it was working, which is the same failure this file
    # keeps producing in other forms: an absent line reading as nothing
    # happening.
    #
    # ROUGHLY EIGHT TIMES PER CLOCK, WHATEVER THE WINDOW IS.
    #
    # The first version beat every 20,000 bars and only above 40,000, which
    # silently assumed a 180-day run. A 30-day M1 window is about 29,000 bars
    # -- gold trades 23 hours on weekdays only -- so it fell under the floor
    # and printed nothing at all for the entire walk. The threshold reproduced
    # exactly the silence it was added to remove, on the shorter run somebody
    # reaches for precisely because they do not want to wait.
    #
    # A share of the window has no such assumption in it.
    total_bars = len(window.index)
    beat = max(total_bars // 8, 2_000)
    walk_began = _time.perf_counter()
    for step, bar_time in enumerate(window.index):
        if beat and step and step % beat == 0:
            spent = _time.perf_counter() - walk_began
            left = spent / step * (total_bars - step)
            print(
                f"      {symbol} {clock.value}: {step:,}/{total_bars:,} bars"
                f"   ~{left / 60:.0f} min left on this clock",
                flush=True,
            )
        upto = bar_time + clock.duration
        awake = [row for row in sections if busy[row[0]] is None or upto > busy[row[0]]]
        if not awake:
            continue
        spread_price = float(spreads[step]) * spec.point
        ctx = _context(symbol, reading, upto, spread_price, slices, legs=legs)
        if ctx is None:
            continue

        for row in awake:
            name, engine, sizer, section_manage = row[:4]
            flatten_time = row[4] if len(row) > 4 else None
            raw_shadow = bool(row[5]) if len(row) > 5 else False
            jarvis_replay = bool(row[6]) if len(row) > 6 else False
            comparison_exit = row[7] if len(row) > 7 else ""
            full_replay_management = jarvis_replay and not comparison_exit
            idea = engine.evaluate(ctx, TradingMode.MICRO_LIVE)
            module = ",".join(sorted({sig.module for sig in idea.signals if sig.score})) or "-"
            if not idea.approved:
                # WHOSE WORDS GO IN THE NOTE, and this is the difference
                # between a diagnosis and a shrug.
                #
                # `idea.reason` is the ENGINE's verdict, and when the section
                # sent no score at all that verdict is always the same
                # sentence: "no weighted directional evidence". Section eleven
                # produced 204,575 of those in one run and not one of them
                # said whether its legs were missing, stale, or simply quiet.
                #
                # So when the section's OWN signal scored zero, its own
                # reasoning is the more specific fact and it goes in instead.
                # The engine's sentence is kept whenever the section did vote
                # and was outvoted, because then the engine is the one with
                # the answer.
                own = next(
                    (sig for sig in idea.signals if sig.module == name and not sig.score),
                    None,
                )
                why = own.reasoning if own is not None and own.reasoning else idea.reason
                out[name].append(
                    Decision(
                        upto,
                        symbol,
                        module,
                        "NO_SIGNAL" if raw_shadow else "REFUSED_CONFLUENCE",
                        note=why[:90],
                        # THE SECTION THAT WAS REFUSED, not just the detector
                        # that voted. Without it a section which took no trades
                        # has no rows anywhere carrying its name, and BY SECTION
                        # cannot tell "took nothing" from "never ran".
                        pass_key=(name, clock.value),
                    )
                )
                continue
            if raw_shadow:
                # Match the discovery replay: signal on this closed clock bar,
                # fill at the NEXT clock bar's open, with the ATR-sized risk
                # frozen from the signal bar.  Using the signal close here was
                # a different strategy and double-counted the entry spread.
                next_bar = int(bars.index.searchsorted(upto, side="left"))
                if next_bar >= len(bars):
                    continue
                signal_close = float(ctx.series[clock].df["close"].iloc[-1])
                risk_price = abs(signal_close - idea.stop_loss)
                sign = 1.0 if idea.direction is Direction.LONG else -1.0
                entry = float(bars["open"].iloc[next_bar])
                idea = replace(
                    idea,
                    entry=entry,
                    stop_loss=entry - sign * risk_price,
                    take_profit=entry + sign * float(engine.config.target_r) * risk_price,
                )
                entry_spread_price = (
                    float(bars["spread"].iloc[next_bar]) * spec.point
                    if "spread" in bars.columns
                    else 0.0
                )
                minimum_lot_stop_money = spec.money_per_lot(risk_price) * spec.volume_min
                spread_share = entry_spread_price / risk_price if risk_price > 0.0 else float("inf")
                if (
                    minimum_lot_stop_money > equity * 0.02
                    or spread_share > RAW_BTC_MAX_SPREAD_R[clock]
                ):
                    reasons = []
                    if minimum_lot_stop_money > equity * 0.02:
                        reasons.append("minimum-lot stop exceeds 2% research envelope")
                    if spread_share > RAW_BTC_MAX_SPREAD_R[clock]:
                        reasons.append(
                            f"spread {spread_share:.1%} exceeds "
                            f"{RAW_BTC_MAX_SPREAD_R[clock]:.0%} research envelope"
                        )
                    out[name].append(
                        Decision(
                            upto,
                            symbol,
                            module,
                            "OUTSIDE_RESEARCH_ENVELOPE",
                            direction=idea.direction.name,
                            note="; ".join(reasons)[:90],
                            pass_key=(name, clock.value),
                        )
                    )
                    continue
            if jarvis_replay:
                blocked = _historical_jarvis_gate(
                    ctx,
                    idea,
                    spec,
                    sizer.settings,
                    clock,
                    horizon=None if raw_shadow else REACH_HORIZON,
                )
                if blocked is not None:
                    reason, detail = blocked
                    # Keep the untouched trade as a counterfactual. This is
                    # what lets the report answer whether a gate removed
                    # winners or losers instead of merely counting refusals.
                    # THE SECTION'S OWN HORIZON, and `RAW_BTC_HORIZONS[clock]`
                    # was a KeyError waiting for the first non-BTC clock: it
                    # holds M1, M5 and M15 only, and section eight runs H1
                    # while section nine runs M30. The counterfactual is a
                    # replay-wide feature now, so it has to work on every clock
                    # a section can be configured on.
                    clock_horizon = RAW_BTC_HORIZONS.get(clock, REACH_HORIZON)
                    resolved_horizon = int(clock_horizon * clock.duration / resolve_on.duration)
                    # THE SPREAD THIS BAR CARRIED. `entry_spread_price` is set
                    # inside the `raw_shadow` branch above and does not exist
                    # on any other path -- so the first gated setup of a
                    # non-BTC section raised NameError and took the whole run
                    # with it, after however many hours it had already spent.
                    gate_spread = entry_spread_price if raw_shadow else spread_price
                    rejected_manage = section_manage
                    if section_manage is not None:
                        trigger_r, offset_atr = section_manage
                        if offset_atr:
                            offset_price = offset_atr * _hourly_atr(upto)
                            rejected_manage = (
                                (trigger_r, offset_price) if offset_price > 0.0 else None
                            )
                    commission = sizer.settings.risk.commission_per_lot(spec.asset_class.value)
                    rejected_risk_money = (
                        spec.money_per_lot(abs(idea.entry - idea.stop_loss)) + commission
                    ) * float(spec.volume_min)
                    fixed_r, rejected_at, managed_r, _managed_at = _resolve(
                        resolve_frame,
                        upto,
                        idea,
                        horizon_bars=resolved_horizon,
                        manage=rejected_manage,
                        arrays=resolve_arrays,
                        close_at_horizon=True,
                        full_management=sizer.settings.trade_management,
                        management_atr=_hourly_atr(upto),
                        management_spread=gate_spread,
                        risk_money=rejected_risk_money,
                        equity=equity,
                        planned_minutes=clock_horizon * clock.duration.total_seconds() / 60.0,
                        partial_possible=False,
                    )
                    # CHARGED THE SAME WAY THE TAKEN TRADES ARE. A
                    # counterfactual on a different cost basis than the trades
                    # it is being compared against answers a question nobody
                    # asked: the BTC shadow lane pays its research allowance,
                    # everything else pays what the sizer charges.
                    if raw_shadow:
                        rejected_cost = RAW_BTC_EXECUTION_ALLOWANCE_R + (
                            gate_spread / abs(idea.entry - idea.stop_loss)
                        )
                    else:
                        rejected_cost = sizer.cost_share(
                            spec, abs(idea.entry - idea.stop_loss), gate_spread
                        )
                    fixed_r = None if fixed_r is None else fixed_r - rejected_cost
                    managed_r = None if managed_r is None else managed_r - rejected_cost
                    out[name].append(
                        Decision(
                            upto,
                            symbol,
                            module,
                            reason,
                            direction=idea.direction.name,
                            entry=idea.entry,
                            stop=idea.stop_loss,
                            target=idea.take_profit,
                            lots=float(spec.volume_min),
                            risk_money=rejected_risk_money,
                            risk_pct=(rejected_risk_money / equity * 100.0 if equity else 0.0),
                            result_r=fixed_r,
                            pnl_money=(None if fixed_r is None else fixed_r * rejected_risk_money),
                            managed_r=managed_r,
                            managed_money=(
                                None if managed_r is None else managed_r * rejected_risk_money
                            ),
                            cost_r=rejected_cost,
                            exit_at=rejected_at,
                            note=("COUNTERFACTUAL if gate were off: " + detail)[:90],
                            pass_key=(name, clock.value),
                        )
                    )
                    continue
            if raw_shadow:
                volume = float(spec.volume_min)
                commission = sizer.settings.risk.commission_per_lot(spec.asset_class.value)
                actual_risk_money = (
                    spec.money_per_lot(abs(idea.entry - idea.stop_loss)) + commission
                ) * volume
                sized = SimpleNamespace(
                    decision=SimpleNamespace(approved=True),
                    volume=volume,
                    actual_risk_money=actual_risk_money,
                    actual_risk_pct=(actual_risk_money / equity * 100.0 if equity else 0.0),
                )
            else:
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
                out[name].append(
                    Decision(
                        upto,
                        symbol,
                        module,
                        sized.decision.reason.name,
                        direction=idea.direction.name,
                        note=sized.decision.detail[:90],
                        pass_key=(name, clock.value),
                    )
                )
                continue
            # The configured multiple becomes a distance in price, using this
            # symbol's H1 ATR at the moment of entry. When the ATR is unknown
            # the offset is zero and, exactly as live does, the protective move
            # is then skipped rather than made at an invented distance.
            resolved_manage = section_manage
            if section_manage is not None:
                trigger_r, offset_atr = section_manage
                if offset_atr == 0.0:
                    resolved_manage = (trigger_r, 0.0)
                else:
                    # The explicit comparison uses the last CLOSED H1 bar.
                    atr_at = upto - Timeframe.H1.duration if comparison_exit else upto
                    offset_price = offset_atr * _hourly_atr(atr_at)
                    resolved_manage = (trigger_r, offset_price) if offset_price > 0.0 else None

            force_close_at = None
            if flatten_time is not None:
                hour, minute = (int(part) for part in flatten_time.split(":", 1))
                force_close_at = upto.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if force_close_at <= upto:
                    force_close_at += timedelta(days=1)
            raw_horizon = RAW_BTC_HORIZONS.get(clock, 0)
            resolved_horizon = (
                int(raw_horizon * clock.duration / resolve_on.duration)
                if raw_shadow and raw_horizon
                else horizon
            )
            # HOISTED so the grid below judges its partial rows against the
            # same broker minimum this trade actually faced. Recomputing it
            # there, or assuming every trade can halve, would let the partial
            # columns bank a close that this account's 0.01 lot cannot place.
            partial_is_possible = (
                spec.round_volume_down(
                    sized.volume * sizer.settings.trade_management.partial_close_fraction
                )
                >= spec.volume_min
                and spec.round_volume_down(
                    sized.volume
                    - spec.round_volume_down(
                        sized.volume * sizer.settings.trade_management.partial_close_fraction
                    )
                )
                >= spec.volume_min
            )
            # ONE DEFINITION, HANDED TO BOTH THE TRADE AND THE GRID. Computing
            # it twice is how the baseline row and the trade itself end up
            # resolving against two slightly different targets.
            doorstep_spread = entry_spread_price if raw_shadow else spread_price
            near_target = (
                sizer.settings.trade_management.close_near_target_spreads * doorstep_spread
            )
            near_target_share = sizer.settings.trade_management.close_near_target_max_share
            r, exit_at, managed_r, managed_at = _resolve(
                resolve_frame,
                upto,
                idea,
                horizon_bars=resolved_horizon,
                manage=resolved_manage,
                arrays=resolve_arrays,
                force_close_at=force_close_at,
                close_at_horizon=raw_shadow,
                full_management=(
                    sizer.settings.trade_management if full_replay_management else None
                ),
                management_atr=_hourly_atr(upto),
                management_spread=entry_spread_price if raw_shadow else spread_price,
                risk_money=sized.actual_risk_money,
                equity=equity,
                planned_minutes=(
                    raw_horizon * clock.duration.total_seconds() / 60.0 if raw_horizon else None
                ),
                partial_possible=partial_is_possible,
                near_target_tolerance=near_target,
                near_target_max_share=near_target_share,
            )
            # FREED AT THE EXIT THE ACCOUNT ACTUALLY TAKES.
            #
            # This used to be `exit_at`, the FIXED stop, on every run --
            # including the runs judged on the break-even column. A trade that
            # scratched after four minutes kept its symbol occupied until the
            # fixed stop resolved hours later, and every setup in between was
            # dropped without a trace. On an M1 section that is most of the
            # trades.
            #
            # Held to the end of the window when it never resolved, exactly as
            # an open position holds the symbol live.
            # PER SECTION, because the break-even rule is. A section running a
            # fixed stop is freed by the fixed exit; one running break-even is
            # freed by whichever exit it actually took.
            freed = managed_at if (resolved_manage is not None or jarvis_replay) else exit_at
            if comparison_exit:
                freed = managed_at if resolved_manage is not None else exit_at
            if freed is None:
                # A TRADE THAT REACHED NEITHER BARRIER IS A TIMEOUT, NOT AN
                # ETERNAL POSITION, and this line was the difference between a
                # measurement and a fiction.
                #
                # `busy[name] = end + clock.duration` blocked the section for
                # the WHOLE REMAINING RUN. Section eleven took one trade on
                # day two of a ninety-day replay, that trade reached neither
                # its stop nor its target inside its horizon, and the section
                # was dead for the other eighty-eight days -- while the same
                # section over thirty days took sixteen. Two runs of the same
                # config disagreeing by a factor of sixteen, and the cause was
                # here rather than in the market.
                #
                # `_resolve` stops following a trade at the end of its horizon,
                # so that is when the harness stops knowing anything about it
                # and therefore when the section has to be free again.
                _first, _last = _horizon_window(resolve_frame.index, upto, horizon)
                if _last > _first:
                    freed = resolve_frame.index[_last - 1]
                timed_out[name] = timed_out.get(name, 0) + 1
            busy[name] = freed if freed is not None else end + clock.duration
            risk_money = sized.actual_risk_money

            # THE TRADE PAYS ITS OWN SPREAD. It did not until 31 August, and
            # that is the largest silent error this file has carried.
            #
            # `_resolve` walks raw highs and lows against raw entry, stop and
            # target, so a winner returned the full reward and a loser exactly
            # -1.00R. The cost model existed -- `_hopeless_on_cost` used it to
            # skip markets, `_cost_report` printed it, the sizer refused 982
            # setups on it in a single run -- and NONE of it touched the money
            # of a trade that was taken. The survivors were paid as if trading
            # were free. Once again: a number that exists, is correct, is
            # tested, and is not on the path the code takes.
            #
            # It is not a uniform haircut either. `cost_share` is the round
            # trip divided by the STOP DISTANCE, and the stop is one ATR of
            # the clock -- so the same spread is 1% of an H4 stop and 12% of
            # an M1 one. The error therefore grew as the clock shrank, and it
            # was largest in precisely the cell that then looked best:
            # order_block on M1, +34.00R gross over 105 trades.
            #
            # Subtracted ONCE, which is what `cost_share` already is: the
            # sizer's own refusal says a stop-out costs "about 1+cost_share R
            # rather than 1.00R". One definition, one subtraction.
            if raw_shadow:
                cost = RAW_BTC_EXECUTION_ALLOWANCE_R + (
                    entry_spread_price / abs(idea.entry - idea.stop_loss)
                )
            else:
                cost = sizer.cost_share(spec, abs(idea.entry - idea.stop_loss), spread_price)
            r = None if r is None else r - cost
            managed_r = None if managed_r is None else managed_r - cost
            if resolved_manage is None:
                # Fixed-exit families intentionally use the original broker
                # stop and target, so their live result is the fixed result.
                managed_r = r

            # THE SAME ENTRY, WALKED AGAIN AT EVERY TRIGGER IN THE GRID.
            #
            # This deliberately does NOT update `busy`. A break-even level that
            # frees the symbol earlier would take different later setups, and
            # then the columns below would be comparing entry sets rather than
            # exit rules -- which is the question nobody asked. What this table
            # answers is narrower and honest: on the trades this section
            # actually took, which exit would have kept more of them.
            #
            # A level that wins here has earned a full replay with `busy`
            # following it, not a promotion.
            grid_rows: tuple[tuple[str, float | None], ...] = ()
            if manage_grid:
                risk_price = abs(idea.entry - idea.stop_loss)
                grid_atr = _hourly_atr(upto)
                measured: list[tuple[str, float | None]] = []
                for variant in manage_grid:
                    if variant.kind == "fixed":
                        # THE BASELINE IS THE TRADE ITSELF, not a seventh walk
                        # of the same bars. `r` is already the fixed exit with
                        # the cost taken off, so recomputing it here would put
                        # a second, independently-derived number in the column
                        # every other row is judged against.
                        measured.append((variant.label, r))
                        continue
                    management = None
                    manage_pair = None
                    if variant.kind == "mechanism":
                        # BUILT ONCE PER SECTION, not once per trade. The
                        # table is a handful of configs and this runs tens of
                        # thousands of times; rebuilding and re-validating
                        # each one per trade is the difference between a run
                        # that finishes and one that gets abandoned.
                        key = (name, variant.label)
                        management = variant_configs.get(key)
                        if management is None:
                            management = _variant_management(
                                sizer.settings.trade_management, variant
                            )
                            variant_configs[key] = management
                    else:
                        manage_pair = (
                            variant.trigger_r,
                            variant.stop_offset(risk_price, grid_atr),
                        )
                    _f, _e, grid_result, _ga = _resolve(
                        resolve_frame,
                        upto,
                        idea,
                        horizon_bars=resolved_horizon,
                        manage=manage_pair,
                        arrays=resolve_arrays,
                        force_close_at=force_close_at,
                        close_at_horizon=raw_shadow,
                        full_management=management,
                        management_atr=grid_atr,
                        management_spread=entry_spread_price if raw_shadow else spread_price,
                        risk_money=sized.actual_risk_money,
                        equity=equity,
                        planned_minutes=(
                            raw_horizon * clock.duration.total_seconds() / 60.0
                            if raw_horizon
                            else None
                        ),
                        partial_possible=variant.partial and partial_is_possible,
                        # THE GRID GETS IT TOO. The doorstep close is not an
                        # exit RULE being compared here -- it is part of how
                        # every exit resolves -- so leaving it out would judge
                        # each rule against a target the account no longer
                        # uses, including the fixed baseline they are all
                        # measured against.
                        near_target_tolerance=near_target,
                        near_target_max_share=near_target_share,
                    )
                    measured.append(
                        (variant.label, None if grid_result is None else grid_result - cost)
                    )
                grid_rows = tuple(measured)
            diagnostics = _trade_diagnostics(
                resolve_frame, upto,
                managed_at if managed_at is not None else exit_at,
                idea, ctx, spread_price,
            )
            out[name].append(
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
                    # Derived rather than recomputed, so it is exactly the
                    # money the sizer put behind this trade divided by the lots
                    # it chose -- commission included, no second definition.
                    risk_per_lot=(risk_money / sized.volume if sized.volume > 0 else 0.0),
                    volume_min=float(spec.volume_min),
                    volume_step=float(spec.volume_step),
                    result_r=r,
                    pnl_money=None if r is None else r * risk_money,
                    exit_at=(
                        managed_at if comparison_exit == "break-even" else exit_at
                    ),
                    pass_key=(name, clock.value),
                    managed_r=managed_r,
                    managed_money=None if managed_r is None else managed_r * risk_money,
                    cost_r=cost,
                    grid_r=grid_rows,
                    trend_m5=_short_trend(ctx, Timeframe.M5),
                    trend_m15=_short_trend(ctx, Timeframe.M15),
                    fault_grid_r=(
                        _fault_exit_grid(
                            resolve_frame, upto, idea, managed_r,
                            managed_at if managed_at is not None else exit_at, cost,
                        ) if fault_exit_grid else ()
                    ),
                    mae_r=diagnostics[0],
                    mfe_r=diagnostics[1],
                    spread_stop_pct=diagnostics[2],
                    atr_regime=diagnostics[3],
                    h1_distance_atr=diagnostics[4],
                    h4_distance_atr=diagnostics[5],
                    breakout_atr=diagnostics[6],
                    retest_bars=diagnostics[7],
                    breakout_body_atr=diagnostics[8],
                    breakout_wick_share=diagnostics[9],
                    breakout_volume_ratio=diagnostics[10],
                )
            )
    # SAID OUT LOUD, PER CLOCK. A timeout is a trade the harness stopped
    # following, and until 5 September it also took the section out of the run
    # without a word. Even now that the section is freed correctly, a section
    # timing out often is a section whose horizon is too short for its target,
    # and that is worth seeing rather than inferring from a low trade count.
    for name, count in sorted(timed_out.items()):
        taken = sum(1 for row in out[name] if row.outcome == "TRADE")
        share = count / taken if taken else 0.0
        print(
            f"  {name} {clock.value}: {count} of {taken} trades reached neither "
            f"barrier inside the horizon ({share:.0%})"
        )
    return out


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


def _section_ten_universe(spec: str, connector, settings) -> tuple[str, ...]:
    """`--section-ten-symbols` as this broker actually spells it.

    `metals` asks the scanner's own classifier rather than matching on the
    name, because Eightcap's metals are not all called XAU-something and a
    guessed list that matches nothing returns an empty tuple -- which reads
    as "the section took no trades" instead of "the section was given no
    markets". That confusion is this file's oldest defect and it gets an
    explicit refusal here rather than a quiet empty run.
    """
    if spec.strip().lower() != "metals":
        named = tuple(part.strip() for part in spec.split(",") if part.strip())
        if not named:
            raise SystemExit("--section-ten-symbols was given nothing to use")
        return named

    from scanner.universe import UniverseScanner

    found: list[str] = []
    for item in connector.symbols():
        if settings.instruments.is_ignored(item.name):
            continue
        try:
            asset_class = UniverseScanner._path_class(connector.spec(item.name).path).value
        except Exception:  # noqa: BLE001 - one bad symbol is not a reason to stop
            continue
        if asset_class.lower() == "metal":
            found.append(item.name)
    if not found:
        raise SystemExit("--section-ten-symbols metals found no metals in this broker's catalogue")
    return tuple(found)


def _as_this_broker_spells_them(wanted, connector, settings) -> list[str]:
    """Requested market names, resolved to the catalogue's own spelling.

    A name that is already in the catalogue is left alone. A name that is not
    is tried once more through `broker_symbol(canonical_symbol(name))`, and
    kept only if THAT is in the catalogue. Anything still unknown is returned
    exactly as it was typed, so the fetch loop's per-symbol "no history" line
    names the thing the operator actually asked for rather than a rewritten
    guess they never wrote.

    The catalogue is read once. If the connector cannot produce one at all
    (an empty archive, a broker call that fails), nothing is rewritten --
    guessing a suffix against no evidence is how `USDJPY.i.i` happened.
    """

    try:
        catalogue = {item.name for item in connector.symbols()}
    except Exception:  # noqa: BLE001 - no catalogue means no basis to rewrite
        catalogue = set()
    if not catalogue:
        return list(wanted)

    resolved: list[str] = []
    rewritten: list[tuple[str, str]] = []
    for name in wanted:
        if name in catalogue:
            resolved.append(name)
            continue
        spelled = settings.instruments.broker_symbol(
            settings.instruments.canonical_symbol(name)
        )
        if spelled != name and spelled in catalogue:
            resolved.append(spelled)
            rewritten.append((name, spelled))
        else:
            resolved.append(name)
    if rewritten:
        print(
            "  --symbols resolved to this broker's names: "
            + ", ".join(f"{typed} -> {actual}" for typed, actual in rewritten)
        )
    return resolved


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
    """`(trigger_r, offset_atr_multiple)`, or None if break-even is off.

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

    THE SECOND VALUE IS AN ATR MULTIPLE, NOT AN R MULTIPLE, and treating it as
    R was wrong by a factor of four.

    This used to return `break_even_offset_atr` and `_resolve` multiplied it by
    the trade's RISK, on the argument that both live families use a one-ATR
    stop so 0.10 ATR is 0.10R "to within a rounding error". Two things are
    wrong with that.

    The offset live applies is `_atr_offset`, and its ATR is `_compute_atr`,
    whose own docstring reads "Recent H1 volatility". It is H1 on every
    symbol and every section, whatever clock the section trades. So the
    comparison is not stop-ATR against offset-ATR at all; it is an H1 ATR
    against a stop measured on M5, M15 or M30.

        section six    stop 0.80 x M5-ATR    offset 0.10 x H1-ATR
                       H1-ATR is roughly 3.5x an M5-ATR
                       -> the real offset is about 0.44R, modelled as 0.10R

        impulse_retest stop ~1.0 x M15-ATR   -> about 0.24R, modelled as 0.10R

    Understating the offset does not simply flatter or penalise the result: a
    higher stop scratches for MORE when it is hit and is hit MORE OFTEN. Which
    way it moves a given section is not reasonable to argue about, which is
    exactly why it has to be measured at the real distance.

    The caller now converts this multiple into a PRICE using the symbol's H1
    ATR at the moment of entry, the same input live reads.
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


def _under_the_slot_cap(
    trades: list[Decision],
    slots: int,
    *,
    share_between_sections: bool = False,
    refuse_opposite: bool = True,
    legs_per_symbol: int = 1,
) -> tuple[list[Decision], dict[int, str]]:
    """The trades that would actually have been opened, cap included.

    Returns `(taken, refused)`, where `refused` maps a decision's id to WHICH
    rule stopped it. Two rules live here and they are not the same question:
    a symbol this section already holds is the section's own position in the
    way, while a full book is the account's. Reporting both as "no slot free"
    is how raising the slot count from four to ten looked like the fix for a
    number the slot count never controlled.

    Walked in time order because that is the only order in which the question
    "is a slot free" has an answer. A trade that never resolved holds its slot
    to the end, which is what an open position does.
    """
    if slots <= 0:
        return list(trades), {}
    # ONE ENTRY PER OPEN POSITION, NOT PER BUSY SYMBOL.
    #
    # This used to be a dict keyed by symbol, and `len(...) >= slots` therefore
    # counted MARKETS rather than POSITIONS. Two consequences, both wrong and
    # both silent:
    #
    #   Three sections sharing gold were one entry, so a cap of two allowed
    #   three simultaneous positions -- more risk than the account sanctions.
    #
    #   This replay walks four markets, so the count could never exceed four,
    #   and a cap of TEN could not bind at all. Raising the account from four
    #   slots to ten therefore changed nothing in the measurement, and every
    #   refusal blamed on "no slot free" was in fact the same-symbol rule.
    #   That mislabel sent an afternoon after the wrong number.
    #
    # A list of positions is what the account holds, so that is what is kept.
    open_positions: list[tuple[datetime, str, str, str]] = []
    taken: list[Decision] = []
    #: `id(decision) -> why it was refused`. Two different rules, two different
    #: answers, and the reader needs to know which one to argue with.
    refused: dict[int, str] = {}
    for trade in sorted(trades, key=lambda d: d.when):
        open_positions = [row for row in open_positions if row[0] > trade.when]
        # WHO IS HOLDING IT, not just that it is held.
        #
        # `sections_may_share_a_symbol` lets a SECOND section join a symbol
        # another section already has -- separate plans, separate stops --
        # while a second leg of the SAME section stays refused, because that is
        # pyramiding.
        on_this_symbol = [row for row in open_positions if row[1] == trade.symbol]
        # `legs_per_symbol` > 1 LETS A SECTION STACK ON ITS OWN MARKET, and it
        # exists to be measured rather than to be shipped. The owner's question
        # was whether section six is leaving money on the table by refusing a
        # setup while it already holds gold; the honest answer is a number for
        # the extra R AND a number for the extra drawdown, not an argument.
        own = [row for row in on_this_symbol if row[2] == trade.module]
        blocked_by = next(
            (
                row
                for row in on_this_symbol
                if not (
                    (
                        row[2] == trade.module
                        and len(own) < legs_per_symbol
                        # AND THE SAME WAY ROUND, ALWAYS. A second leg is the
                        # same idea again; a leg against the first one is flat
                        # exposure bought with two spreads, and at most one of
                        # the two stops can be hit. Not tied to
                        # `refuse_opposite` -- that setting is about two
                        # SECTIONS disagreeing, which is a real disagreement.
                        # A section reversing itself is a close-and-reverse,
                        # not a stack.
                        and row[3] == trade.direction
                    )
                    or (
                        share_between_sections
                        and row[2] != trade.module
                        and (not refuse_opposite or row[3] == trade.direction)
                    )
                )
            ),
            None,
        )
        if blocked_by is not None:
            refused[id(trade)] = (
                "SYMBOL_ALREADY_HELD"
                if blocked_by[2] == trade.module
                else "SYMBOL_HELD_BY_ANOTHER_SECTION"
            )
            continue
        if len(open_positions) >= slots:
            refused[id(trade)] = "ACCOUNT_POSITION_LIMIT"
            continue
        taken.append(trade)
        until = trade.exit_at or datetime.max.replace(tzinfo=trade.when.tzinfo)
        open_positions.append((until, trade.symbol, trade.module, trade.direction))
    return taken, refused


def _markets_each_section_needs(settings, names: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """`(the markets these sections can trade, the sections that trade anything)`.

    WHY THIS IS NOT A LIST IN A LAUNCHER. The 90-day account replay walked
    eleven markets and SIX of them took no trade at all -- GBPUSD, USDCHF,
    EURJPY, GBPJPY, US30, GER40. Not "took no trade because the strategy
    passed": no section on the book is even ALLOWED to trade them. Six markets
    of bars, three hundred thousand each, judged seven times per bar, to
    produce nothing. That is most of an hour of the owner's evening.

    A hand-written symbol list would fix it once and be wrong the next time a
    section changes market. So the question is asked of the sections
    themselves, and there are THREE places they answer it from:

        `allowed_symbols` on the config   -- sections 7, 8, 9, 10, 15, 16, 17
        `symbol` on the config            -- the XAUJPY sections
        `symbol` on the MODULE class      -- section six, both instances

    All three are read, because reading only the first is exactly the bug this
    file already shipped once: `getattr(config, "allowed_symbols", ())`
    returned an empty tuple for every XAUJPY section, so the widening that
    block exists for never happened for them.

    A SECTION THAT DECLARES NOTHING TRADES EVERYTHING, and it is returned in
    the second list rather than silently ignored. Narrowing the universe while
    such a section is measured would quietly stop measuring it -- the same
    silence, in a new place -- so the caller has to decide out loud.
    """

    from runner.service import build_analysis_modules

    modules = {module.name: module for module in build_analysis_modules(settings)}
    markets: list[str] = []
    unbounded: list[str] = []
    for name in names:
        config = getattr(settings.analysis, name, None)
        declared = tuple(getattr(config, "allowed_symbols", ()) or ())
        if not declared:
            one = getattr(config, "symbol", "") or getattr(modules.get(name), "symbol", "")
            declared = (one,) if one else ()
        if not declared:
            unbounded.append(name)
            continue
        for market in declared:
            if market not in markets:
                markets.append(market)
    return markets, unbounded


def _retimed(settings, module_name: str, timeframe: str):
    """One section, moved to one clock, and ALLOWED TO VOTE.

    THE SHADOW MEASUREMENT WAS STRUCTURALLY IMPOSSIBLE WITHOUT THE SECOND
    HALF, and it took three attempts to reach the actual cause.

    `ConfluenceConfig.effective_weights` forces every module absent from
    `live_enabled_modules` to weight ZERO whenever the mode is live -- which is
    correct, and is exactly what stops a switched-off section spending money.
    This script evaluates in MICRO_LIVE. So `impulse_retest`, switched off on
    30 August, entered every pass at weight zero, failed
    `if weight > 0`, and every single bar came back "no weighted directional
    evidence". Sixteen markets, 180 days, zero trades, and nothing wrong with
    the detector.

    I had already "fixed" this twice at the wrong level: first by keeping the
    module's weight in the config, then by sweeping every known module rather
    than only the live ones. Both were necessary and neither was sufficient,
    because the engine zeroes the weight one layer below both.

    So a pass measuring a section grants that section permission -- on a COPY
    of the settings that exists only for the measurement and is never handed to
    a broker. The live allowlist in `config/eightcap.yaml` is untouched, which
    `test_measuring_a_section_does_not_make_it_live` checks.
    """
    analysis = settings.analysis
    section = getattr(analysis, module_name)
    confluence = analysis.confluence
    allowed = tuple(dict.fromkeys((*confluence.live_enabled_modules, module_name)))
    measurement_weights = dict(confluence.weights)
    if measurement_weights.get(module_name, 0.0) <= 0.0:
        measurement_weights[module_name] = 1.0
    return settings.model_copy(
        update={
            "analysis": analysis.model_copy(
                update={
                    module_name: section.model_copy(update={"timeframe": timeframe}),
                    "confluence": confluence.model_copy(
                        update={
                            "live_enabled_modules": allowed,
                            "weights": measurement_weights,
                        }
                    ),
                }
            )
        }
    )


class _StoredMarket:
    """A broker-shaped object backed by a folder of stored bars.

    Three methods is the whole surface this script uses: `spec` for sizing,
    `account` for the starting equity, and `symbols` for the universe. The
    history itself is read directly from the store rather than through here,
    because `fetch_mt5_history` chunks a range the terminal cannot serve in
    one call and a folder has no such limit.

    `shutdown` exists so the `finally` in main needs no special case. A store
    has nothing to close.
    """

    def __init__(self, store, equity: float) -> None:
        self.store = store
        self._equity = equity or 0.0

    def spec(self, symbol: str):
        return self.store.spec(symbol)

    def symbols(self):
        from types import SimpleNamespace

        specs = [self.store.spec(name) for name in self.store.symbols()]
        return [SimpleNamespace(name=spec.symbol, path=spec.path) for spec in specs]

    def account(self):
        from types import SimpleNamespace

        # EQUITY MUST BE GIVEN when there is no terminal to ask. Defaulting to
        # a number would silently size every trade in the measurement against
        # a balance the account does not have.
        if self._equity <= 0:
            raise SystemExit(
                "--cache has no account to read, so --equity is required:\n"
                "    dryrun.cmd --cache data/history --equity 216"
            )
        return SimpleNamespace(equity=self._equity, currency="EUR")

    def shutdown(self) -> None:
        close = getattr(self.store, "close", None)
        if close is not None:
            close()


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
    parser.add_argument(
        "--start-date",
        default="",
        help="exact first UTC calendar day (YYYY-MM-DD); requires --end-date",
    )
    parser.add_argument(
        "--end-date",
        default="",
        help="exact last UTC calendar day, inclusive (YYYY-MM-DD); requires --start-date",
    )
    parser.add_argument("--symbols", default="", help="comma list; default = the live universe")
    parser.add_argument("--csv", default="", help="write every decision to this file")
    parser.add_argument("--equity", type=float, default=0.0, help="override account equity")
    parser.add_argument(
        "--s5-exit", choices=("fixed", "break-even"), default="",
        help="Replay-only S5 M5 exit comparison; never changes live settings",
    )
    parser.add_argument(
        "--legs-per-symbol",
        type=int,
        default=1,
        help=(
            "how many positions ONE section may hold in one symbol at a time. 1 is "
            "the account; above that measures what stacking would have done, both "
            "in R and in drawdown. Replay-only"
        ),
    )
    parser.add_argument(
        "--fixed-exits",
        action="store_true",
        help="replay with the entry stop and target only: no break-even, no partial, "
        "no trailing, no profit lock. Replay-only; live settings are untouched",
    )
    parser.add_argument(
        "--risk-percent",
        type=float,
        default=0.0,
        help=(
            "shadow-only sizing override for this replay (0 keeps configured risk); "
            "never changes YAML or live Jarvis"
        ),
    )
    parser.add_argument(
        "--manage-grid",
        action="store_true",
        help=(
            "resolve every taken trade again under each exit rule in the grid "
            "and print what each one would have kept, on the same entries as "
            "the fixed exit"
        ),
    )
    parser.add_argument(
        "--strict-risk",
        action="store_true",
        help=(
            "replay-only: refuse the broker minimum lot when its real stop risk "
            "exceeds the configured per-trade target"
        ),
    )
    parser.add_argument(
        "--trend-grid",
        action="store_true",
        help=(
            "compare existing entries with M5, M15 and combined short-trend "
            "alignment; no live setting is changed and no exit grid is run"
        ),
    )
    parser.add_argument(
        "--fault-exit-grid",
        action="store_true",
        help="measure corroborated M5 thesis exits in shadow only; no live change",
    )
    parser.add_argument(
        "--exit-grid",
        choices=("kern", "alles"),
        default="",
        help=(
            "which exit rules to compare; implies --manage-grid. `kern` is the "
            "break-even triggers at three stop placements plus a short trailing "
            "and partial sweep; `alles` is every placement and the full sweep, "
            "which is far slower"
        ),
    )
    parser.add_argument(
        "--btc-research-parity",
        "--raw-btc-shadow",
        dest="btc_research_parity",
        action="store_true",
        help=(
            "S15-S17 shadow replay only: reproduce the frozen research entry, "
            "horizon, cost and eligibility envelope; never affects live Jarvis"
        ),
    )
    parser.add_argument(
        "--btc-jarvis-replay",
        action="store_true",
        help=(
            "S15-S17 second measurement: keep their frozen measured entries, then "
            "apply the reproducible Jarvis portfolio/account rules; news is skipped"
        ),
    )
    parser.add_argument(
        "--section-ten-only",
        action="store_true",
        help=(
            "walk only the markets section ten is allowed to trade. Everything "
            "else in the universe is skipped rather than measured and refused."
        ),
    )
    parser.add_argument(
        "--section-ten-symbols",
        default="",
        help=(
            "comma list replacing section ten's allowed_symbols FOR THIS RUN only, "
            "or the word 'metals' for every metal the scanner finds. The file on "
            "disk is untouched, so this measures a wider section ten without "
            "putting one live."
        ),
    )
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
        "--cache",
        default="",
        help=(
            "read bars from a stored history folder instead of MT5 "
            "(--cache data/history). Fill it once with fetch_history.py; after "
            "that a measurement needs no terminal, no login and no re-download."
        ),
    )
    parser.add_argument(
        "--database",
        default="",
        help=(
            "read the one-file SQLite research archive instead of MT5 "
            "(--database market-history.sqlite3)"
        ),
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
        "--only",
        default="",
        help=(
            "measure just these sections, comma or space separated "
            "(--only impulse_retest). Halves the run when the other one has "
            "already been measured."
        ),
    )
    parser.add_argument(
        "--section-markets",
        action="store_true",
        help=(
            "walk ONLY the markets the measured sections are allowed to trade, "
            "read from the sections themselves. The 90-day book run spent six "
            "of its eleven markets proving that no section may trade them."
        ),
    )
    parser.add_argument(
        "--jarvis-replay",
        action="store_true",
        help=(
            "answer 'if I had started this on day one, where would the account "
            "be now'. Applies the live gates whose inputs exist in historical "
            "bars -- spread against stop width, market liveliness, the M1 "
            "volume spike, target reach and direction advantage -- and then "
            "walks the surviving entries in time order under the SHARED Jarvis "
            "position limit. Slower, and much closer to the account."
        ),
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help=(
            "measure ONLY sections on the live real-money allowlist, each on its "
            "own configured clock. Overrides --sweep."
        ),
    )
    parser.add_argument(
        "--sections-five-to-ten",
        action="store_true",
        help=(
            "measure exactly sections 5 through 10 on their configured clocks; "
            "shadow sections are measured without granting real-money permission"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    # `argv` so a test can drive the entry point. Without it the only way to
    # check that a misconfigured run refuses loudly is to run the real thing
    # against a real terminal, which is why several of those refusals were
    # never tested and two of them did not work.
    args = build_parser().parse_args(argv)

    selected_only = {
        item.strip() for item in args.only.replace(" ", ",").split(",") if item.strip()
    }
    if args.s5_exit and (
        selected_only != {"section_five_ndx100_m5"}
        or not args.jarvis_replay
        or args.sweep
        or args.btc_research_parity
        or args.btc_jarvis_replay
    ):
        raise SystemExit(
            "--s5-exit requires --only section_five_ndx100_m5 --jarvis-replay "
            "without --sweep or BTC modes"
        )
    if args.fixed_exits:
        # BOTH FLAGS WRITE THE SAME FIELD. `--s5-exit` already decides the exit
        # regime for its one section; letting the two run together would mean
        # one silently winning, and the report would name the loser.
        if args.s5_exit:
            raise SystemExit("--fixed-exits and --s5-exit both set the exit regime; choose one")
        # WITHOUT `--jarvis-replay` THERE IS NO MANAGEMENT TO REMOVE. The plain
        # dry run is already entry stop and target only, so the flag would
        # change nothing while the header claimed it had -- a run that reports
        # a comparison it never made.
        if not args.jarvis_replay:
            raise SystemExit("--fixed-exits only means something with --jarvis-replay")
    # BUILT ONCE, HERE, so the report and the walk cannot disagree about which
    # rules were compared. A second call to `exit_grid()` further down would be
    # a second list to keep in step, and a table headed with rules the walk did
    # not measure is worse than no table.
    exit_variants = (
        exit_grid(wide=args.exit_grid == "alles")
        if (args.manage_grid or args.exit_grid)
        else ()
    )
    btc_shadow = args.btc_research_parity or args.btc_jarvis_replay
    if args.jarvis_replay and args.no_m1:
        # THE VOLUME GATE READS M1 AND NOTHING ELSE. Without M1 history it
        # would simply never fire, the run would still print "VOLUME_SPIKE
        # applied", and the claim would be false in the quietest possible way
        # -- a gate that exists, is correct, is tested, and is not on the path
        # the code takes. That is this repository's most-repeated defect, so
        # the combination is refused instead of silently degraded.
        raise SystemExit(
            "--jarvis-replay needs M1 bars: the volume-spike gate reads M1 and only M1, "
            "and --no-m1 would leave it claimed but never applied. Drop --no-m1."
        )
    if args.jarvis_replay and btc_shadow:
        raise SystemExit(
            "--jarvis-replay covers every measured section and the BTC modes are a"
            " separate, narrower replay. Choose one."
        )
    if btc_shadow and (not selected_only or not selected_only.issubset(RAW_BTC_SHADOW_SECTIONS)):
        raise SystemExit(
            "BTC replay modes are restricted to --only section_fifteen_btc_m1,"
            "section_sixteen_btc_m5,section_seventeen_btc_m15"
        )
    if args.btc_research_parity:
        print(
            "BTC RESEARCH PARITY: frozen next-bar entries, horizons, cost allowance and "
            "spread/risk envelope; live Jarvis is unchanged"
        )
    if args.btc_jarvis_replay:
        print(
            "BTC JARVIS REPLAY: frozen S15-S17 entries followed chronologically under "
            "the account position limits; news skipped by operator request"
        )
    if args.jarvis_replay:
        print(
            "JARVIS REPLAY: every measured section under the live spread/stop, "
            "liveliness, volume-spike and target-reach gates, walked in time order "
            "under one shared position book. News is not replayable and is skipped."
        )

    # THE OVERLAY, or there are no live modules at all. The base config ships
    # `live_enabled_modules` empty on purpose -- permission to trade real money
    # is an account-level decision, and it lives in the Eightcap overlay. Load
    # without it and this exits with "no live modules" while the account is
    # perfectly well configured.
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=True)
    if args.s5_exit:
        if settings.analysis.section_five_ndx100_m5.timeframe != "M5":
            raise SystemExit("S5 comparison requires the configured S5 timeframe to be M5")
        if args.s5_exit == "break-even" and _break_even_rule(settings) is None:
            raise SystemExit("No configured break-even rule: cannot label this a BE comparison")
        comment = broker_comment("section_five_ndx100_m5", is_addon=False, experimental_live=True)
        fixed = tuple(
            c
            for c in settings.trade_management.fixed_exit_comments
            if c.casefold() != comment.casefold()
        )
        if args.s5_exit == "fixed":
            fixed += (comment,)
        settings = settings.model_copy(
            update={
                "trade_management": settings.trade_management.model_copy(
                    update={"fixed_exit_comments": fixed}
                )
            }
        )
        print(
            f"S5 RESEARCH COMPARISON: {args.s5_exit}; independent entry/position "
            "sequence. Live configuration unchanged."
        )
        print(
            "Only fixed SL/TP or the configured break-even rule is compared; no "
            "partial/trailing/profit-lock in this comparison."
        )
    if args.fixed_exits:
        # TURNED OFF AT THE SOURCE, not at the one call site that applies it.
        #
        # `_break_even_rule` returns None as soon as `break_even_at_r` is zero,
        # and it is the single thing every part of this script asks. Zeroing it
        # here therefore removes the rule from the trade resolution AND from
        # every header, footnote and column that reports whether break-even was
        # in force. Suppressing it only where it is applied would leave a run
        # that behaves one way and describes itself the other -- which is the
        # single most repeated defect in this repository, and the reason the
        # 180-day section-six number had to be thrown away once already.
        #
        # In memory, on a copy, for this process. `config/eightcap.yaml` is not
        # written and the live account keeps every rule it has.
        settings = settings.model_copy(
            update={
                "trade_management": settings.trade_management.model_copy(
                    update={"break_even_at_r": 0.0}
                )
            }
        )
        print(
            "FIXED-EXIT REPLAY: entry stop and target only -- no break-even, no partial, "
            "no trailing, no profit lock, no peak-stall, no time exit."
        )
        print(
            "  Live configuration is untouched. Every gate, the cost model and the shared "
            "position book are unchanged; only what happens AFTER entry is."
        )
    settings = settings.model_copy(
        update={"system": settings.system.model_copy(update={"mode": TradingMode.MICRO_LIVE})}
    )
    if args.strict_risk:
        settings = settings.model_copy(
            update={
                "risk": settings.risk.model_copy(
                    update={"allow_minimum_lot_above_target": False}
                )
            }
        )
        print(
            "STRICT RISK REPLAY: minimum lot may not exceed the configured risk target; "
            "live configuration is untouched."
        )
    if args.risk_percent:
        if not 0.0 < args.risk_percent <= 20.0:
            raise SystemExit("--risk-percent must be above 0 and at most 20")
        modes = dict(settings.modes)
        modes[TradingMode.MICRO_LIVE.value] = modes[TradingMode.MICRO_LIVE.value].model_copy(
            update={"max_risk_per_trade_pct": args.risk_percent}
        )
        settings = settings.model_copy(
            update={
                "risk": settings.risk.model_copy(
                    update={
                        "risk_per_trade_pct": args.risk_percent,
                        "max_risk_per_trade_pct": args.risk_percent,
                    }
                ),
                "modes": modes,
            }
        )
        print(
            f"SHADOW RISK STRESS: {args.risk_percent:.1f}% per trade; "
            "this does NOT alter live configuration or grant real-money permission"
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

    # BARS FROM DISK OR FROM THE TERMINAL, decided once, here.
    #
    # `_StoredMarket` answers the three things this run asks of a broker --
    # `spec`, `account` and the history fetch -- and answers them from a folder.
    # With it there is no login, no terminal, and no second download of bars
    # that cannot change: a closed M15 candle from June is the same object in
    # September.
    if args.cache and args.database:
        raise SystemExit("choose --cache or --database, not both")
    store = None
    if args.database:
        from backtesting.research_dataset import ResearchDataset

        store = ResearchDataset(Path(args.database), read_only=True)
        if not store.symbols():
            store.close()
            raise SystemExit(f"{args.database} holds no stored markets")
        connector = _StoredMarket(store, args.equity)
        print(f"reading bars from {store.path} — MT5 is not being contacted")
    elif args.cache:
        from backtesting.history_store import HistoryStore

        store = HistoryStore(Path(args.cache))
        if not store.symbols():
            raise SystemExit(
                f"{args.cache} holds no stored markets. Fill it first:\n"
                f"    python scripts/fetch_history.py --days {max(args.days, 180)}"
            )
        connector = _StoredMarket(store, args.equity)
        print(f"reading bars from {store.root} — MT5 is not being contacted")
    else:
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
            # SPELLED THE WAY THIS BROKER SPELLS IT, not the way it was typed.
            #
            # Every other branch here gets its names from `connector.symbols()`
            # or from `_core_universe`, so they already carry Eightcap's
            # suffix. `--symbols` is the one path that goes straight from the
            # command line to the fetch, and `--symbols US30` therefore asks
            # for a market called `US30` on a broker that lists `US30.i`.
            #
            # That does not crash. It prints "no history" once per name and
            # produces a report with no rows -- which reads exactly like a
            # strategy that found nothing, and is the same suffix mismatch that
            # has now silently disabled four separate things in this project.
            #
            # `broker_symbol(canonical_symbol(x))` and not `broker_symbol(x)`:
            # the latter turns an already-suffixed `US30.i` into `US30.i.i`.
            # That is not hypothetical either -- the Control Deck panel shipped
            # with `USDJPY.i.i` in it for exactly this reason.
            symbols = _as_this_broker_spells_them(
                [s.strip() for s in args.symbols.split(",") if s.strip()],
                connector,
                settings,
            )
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

        # SECTION TEN ON MORE THAN ONE METAL, MEASURED WITHOUT PUTTING ONE LIVE.
        #
        # Section ten is the only live component with a positive measured edge
        # per trade, and it sees one market. Widening `allowed_symbols` in the
        # file is a real-money change to an unmeasured setting -- the exact
        # move that put section six live at what later read -71.65 R. This
        # widens it FOR THIS RUN, in memory, so the 180-day replay says what it
        # would have done before anything is decided.
        #
        # The symbols the section may take are also added to what the run
        # fetches. Without that the section would be widened onto markets the
        # loop never visits, and it would come back with the same trade count
        # and look like a widening that changed nothing.
        if args.section_ten_symbols:
            wanted_ten = _section_ten_universe(args.section_ten_symbols, connector, settings)
            settings = settings.model_copy(
                update={
                    "analysis": settings.analysis.model_copy(
                        update={
                            "section_ten_gold_m1": (
                                settings.analysis.section_ten_gold_m1.model_copy(
                                    update={"allowed_symbols": wanted_ten}
                                )
                            )
                        }
                    )
                }
            )
            missing = [name for name in wanted_ten if name not in symbols]
            symbols = symbols + missing
            print(
                f"section ten widened FOR THIS RUN ONLY to {len(wanted_ten)} markets: "
                f"{', '.join(wanted_ten)}"
            )
            if missing:
                print(f"  and {len(missing)} of them added to the fetch: {', '.join(missing)}")

        # WALK ONLY THE MARKETS THE SECTION CAN TRADE.
        #
        # The first section-ten run spent 252 seconds on EURUSD and 505 on
        # GBPUSD before it reached a single metal, with an hour left to go, and
        # section ten cannot take a trade in either of them -- `allowed_symbols`
        # refuses them on the first bar. Sixteen markets to measure six is not
        # thoroughness, it is an hour of walking bars to watch a symbol filter
        # say no twelve thousand times.
        if args.section_ten_only:
            allowed = tuple(settings.analysis.section_ten_gold_m1.allowed_symbols)
            if not allowed:
                raise SystemExit("--section-ten-only: section ten allows no symbols at all")
            # THE SECTION'S LIST *REPLACES* THE UNIVERSE, it does not filter it.
            #
            # This intersected the two, and that quietly threw away the whole
            # point. `--core` is the sixteen markets the research was done on:
            # eleven FX majors, gold, four indices. Five of section ten's six
            # metals are not in it. So the intersection was {XAUUSD} and the
            # run printed "walking 1 markets" and measured exactly the thing
            # that did not need measuring, while claiming to test a widening
            # to six.
            #
            # A section's `allowed_symbols` IS the universe for a run about
            # that section. Anything in it that this broker does not list will
            # fail its own fetch and say so per symbol, which is visible;
            # silently dropping five markets is not.
            dropped = [name for name in symbols if name not in allowed]
            symbols = list(allowed)
            print(
                f"section ten only: walking its own {len(symbols)} markets "
                f"({', '.join(symbols)})"
            )
            if dropped:
                print(f"  skipping {len(dropped)} the section cannot trade")

        stored_window = store.window() if store is not None else None
        if bool(args.start_date) != bool(args.end_date):
            raise SystemExit("--start-date en --end-date moeten samen worden gebruikt")
        if args.start_date:
            try:
                start = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=UTC)
                last_day = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=UTC)
            except ValueError as exc:
                raise SystemExit("datums moeten YYYY-MM-DD zijn") from exc
            if last_day < start:
                raise SystemExit("--end-date moet op of na --start-date liggen")
            # `_one_clock` includes its endpoint. Stop one microsecond before
            # the next midnight so 30 September cannot leak an October bar.
            end = last_day + timedelta(days=1) - timedelta(microseconds=1)
            args.days = (last_day.date() - start.date()).days + 1
        else:
            end = (
                datetime.fromisoformat(stored_window[1])
                if stored_window is not None and stored_window[1]
                else datetime.now(UTC)
            )
            start = end - timedelta(days=args.days)
        # `fetch_these` is decided AFTER `passes`, further down: a live section
        # sitting on M1 needs M1 bars whatever --no-m1 says, and the clock came
        # from the config rather than from this command line.

        # WARMUP IS MEASURED IN BARS, SO THE WINDOW IS PER TIMEFRAME.
        #
        # A single 27-day fetch gives M15 about 2,600 bars and H4 about 160 --
        # under the 270 the guard wants, so EVERY symbol was skipped "for want
        # of history" and the first run reported zero decisions on a working
        # account. 1.6x pads the weekends out of the calendar days.
        def _fetch_from(tf: Timeframe) -> datetime:
            bars_needed = (WARMUP + 20) * tf.duration
            return start - max(bars_needed * 1.6, timedelta(days=3))

        # WHICH SECTIONS ON WHICH CLOCKS. The shipped timeframes were chosen
        # on HistData; a sweep asks the same question against this broker's
        # spreads, which is the number that actually decides it.
        #
        # Each section is swept on its own, never together: two sections on
        # the same clock would merge into one confluence idea and the result
        # would say nothing about either.
        # EVERY MODULE THIS SCRIPT KNOWS, not just the ones allowed to trade.
        #
        # I switched `impulse_retest` off on 30 August and told the owner "the
        # weight stays, so it is still measured". That is true of the engine
        # and FALSE of this script: `passes` was built from
        # `live_enabled_modules`, so switching a section off removed it from
        # the MEASUREMENT as well. The 180-day run he asked for judged one
        # section, and the other -- the one the earlier 180-day data had
        # actually favoured -- produced no rows at all.
        #
        # A module that may not trade is exactly the module that most needs
        # measuring, because that measurement is what decides whether it comes
        # back. So every known module is swept, and the report separates what
        # is live from what is shadowed.
        # EVERY SECTION THIS SCRIPT CAN MEASURE, and `order_block_fast` is on
        # it because it trades real money. A live section missing from here is
        # a live section the "what would the account have done" report cannot
        # see, which is the same silence in a new place.
        module_config = {
            "impulse_retest": "impulse_retest",
            "impulse_retest_m30": "impulse_retest_m30",
            "order_block": "order_block",
            "order_block_fast": "order_block_fast",
            "order_block_m15": "order_block_m15",
            "order_block_h1": "order_block_h1",
            "walkforward_index": "walkforward_index",
            "failed_session_breakout": "failed_session_breakout",
            "section_five_ndx100_m5": "section_five_ndx100_m5",
            "section_six_gold_m5": "section_six_gold_m5",
            "section_six_spx_h1": "section_six_spx_h1",
            "section_seven_gold_smc": "section_seven_gold_smc",
            "section_eight_trend_day_h1": "section_eight_trend_day_h1",
            "section_nine_vwap_m30": "section_nine_vwap_m30",
            "section_ten_gold_m1": "section_ten_gold_m1",
            "human_context_decision": "human_context_decision",
            "section_eleven_xaujpy_legs_m5": "section_eleven_xaujpy_legs_m5",
            "section_twelve_xaujpy_m5": "section_twelve_xaujpy_m5",
            "section_thirteen_xaujpy_m15": "section_thirteen_xaujpy_m15",
            "section_us30_impulse_m1": "section_us30_impulse_m1",
            "section_us30_impulse_m5": "section_us30_impulse_m5",
            "section_us30_orderblock_m1": "section_us30_orderblock_m1",
            "section_us30_orderblock_m5": "section_us30_orderblock_m5",
            "section_fifteen_btc_m1": "section_fifteen_btc_m1",
            "section_sixteen_btc_m5": "section_sixteen_btc_m5",
            "section_seventeen_btc_m15": "section_seventeen_btc_m15",
        }
        measured = set(module_config)
        if args.sections_five_to_ten:
            # THE BOOK, AS IT ACTUALLY STANDS -- not a list written down when it
            # had six entries.
            #
            # This was a hardcoded set including `section_five_ndx100_m5` and
            # `section_nine_vwap_m30`. Both came off the live allowlist on
            # 2 September (-1.09 R over 170 trades and -0.02 R over 6), and a
            # second copy of a list that must agree with `live_enabled_modules`
            # is a copy that disagrees with it the moment one changes. It did,
            # within the hour.
            #
            # Intersecting with `live` means this flag answers "what would the
            # account have done" rather than "what would the account have done
            # in August". A section that comes back on appears here again with
            # no edit.
            #
            # SECTION ELEVEN IS ON IT AS OF 4 SEPTEMBER. The flag is still
            # called `--sections-five-to-ten` because that is what the launcher
            # passes; the set is "the book", not a range of numbers, and a
            # twelfth section belongs here the day it is written.
            book = {
                "failed_session_breakout",
                "section_five_ndx100_m5",
                "section_six_gold_m5",
                "section_eight_trend_day_h1",
                "section_nine_vwap_m30",
                "section_ten_gold_m1",
                "section_eleven_xaujpy_legs_m5",
                "section_twelve_xaujpy_m5",
                "section_thirteen_xaujpy_m15",
                # The three BTC discoveries, live since 6 September. The flag
                # is still called `--sections-five-to-ten` because that is what
                # the launcher passes; the set is "the book", not a range of
                # numbers, and a section belongs here the day it can spend
                # money -- otherwise `dryrun-live.cmd` reports on an account
                # that is not the one running.
                "section_fifteen_btc_m1",
                "section_sixteen_btc_m5",
                "section_seventeen_btc_m15",
                # The four experimental US30 sections. Shadow, so they never
                # reach the live intersection below -- listed so that the day
                # one of them is promoted, `dryrun-live.cmd` already measures
                # it instead of failing on a name it has never seen.
                "section_us30_impulse_m1",
                "section_us30_impulse_m5",
                "section_us30_orderblock_m1",
                "section_us30_orderblock_m5",
            }
            missing = (book & live) - measured
            if missing:
                raise SystemExit(
                    f"sections 5-10 are missing from the dry-run implementation: {sorted(missing)}"
                )
            measured = measured & book & live
            benched = sorted(book - live)
            if benched:
                print(f"  not measured, off the live allowlist: {', '.join(benched)}")
        if args.live_only:
            # `dryrun-live.cmd` means exactly what its name says. Previously
            # this flag only disabled the timeframe sweep, while `measured`
            # still contained every known (including quarantined) module. The
            # report therefore spent most of its time replaying impulse/retest
            # and order-block rows which were not allowed to trade, then mixed
            # them into the all-decisions totals. Keep research/history runs
            # unchanged; only the explicit live-only path gets this filter.
            missing = live - measured
            if missing:
                raise SystemExit(
                    f"live modules are missing from the dry-run implementation: {sorted(missing)}"
                )
            measured = measured & live
        if args.jarvis_replay and not args.only and not args.live_only:
            # THE ACCOUNT REPLAY MEASURES WHAT THE ACCOUNT COULD RUN, and a
            # section switched off in the config could not run. Including the
            # retired research modules would add ten full bar walks that can
            # only ever produce refusals, and would mix modules that cannot
            # trade into a total headed "what would I have made".
            #
            # A PROPERTY, NOT A LIST. `enabled` is asked of each config rather
            # than a second copy of the book being written down here -- a copy
            # that must agree with the config is a copy that disagrees with it
            # the first time one changes, which has already happened twice in
            # this file.
            def _could_have_run(name: str) -> bool:
                config = getattr(settings.analysis, name, None)
                if not getattr(config, "enabled", True):
                    return False
                # A NUMBERED SECTION, or a module the allowlist lets spend
                # money. Everything else in `module_config` is a section-one
                # detector -- impulse/retest, the order-block clocks -- which
                # cannot open a position of its own on this account, so its
                # euros do not belong in a total headed "what would I have
                # made". They are still measurable with `--only`.
                return name in live or name.startswith("section_")

            dropped = sorted(name for name in measured if not _could_have_run(name))
            measured = measured - set(dropped)
            if dropped:
                print(
                    "  not measured, because the account could not have traded them: "
                    f"{', '.join(dropped)}"
                )
            if not measured:
                raise SystemExit(
                    "every known section is disabled, so there is nothing for the account "
                    "replay to measure. That is not EUR 0.00; it is no observations."
                )
        if args.only:
            wanted_sections = {
                piece.strip()
                for chunk in str(args.only).replace(",", " ").split()
                for piece in [chunk]
                if piece.strip()
            }
            unknown = wanted_sections - measured
            if unknown:
                raise SystemExit(
                    f"--only names sections this script does not know: {sorted(unknown)}. "
                    f"Known: {sorted(measured)}"
                )
            measured = wanted_sections
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
            for name in sorted(measured):
                for tf in wanted:
                    passes.append((name, tf))
        else:
            for name in sorted(measured):
                passes.append((name, getattr(settings.analysis, name).timeframe))

        # A SECTION WHOSE MARKETS ARE NOT WALKED IS SILENT, NOT FLAT --
        # the same confusion this file has now shipped seven times, in a new
        # place, and this time it would have hit TWO live sections at once.
        #
        # `--core` is the sixteen markets the research was done on: eleven FX
        # majors, gold, four indices. Section ten's `allowed_symbols` is five
        # metals and section eleven's is four crosses, and only XAUUSD of those
        # is in core. So `dryrun-live.cmd 180` would have replayed section ten
        # on ONE of its five markets and section eleven on NONE of its four,
        # and reported the result as if it were the book.
        #
        # Section eleven would have come back with a zero row, and a zero row
        # reads as "the strategy found nothing" rather than "it was never shown
        # a market it can trade". That is the exact sentence this project keeps
        # writing.
        #
        # So: every measured section's own markets are added to the walk. Not
        # intersected -- added. A market this broker does not list fails its own
        # fetch and says so per symbol, which is visible; dropping it here is
        # not.
        #
        # `--symbols` and `--section-ten-only` DECLARE the universe on purpose,
        # so they are left alone. This fills a gap; it does not overrule a
        # choice.
        # NARROW FIRST, THEN WIDEN. `--section-markets` replaces the universe
        # with exactly what the measured sections declare, so the widening
        # below then has nothing left to add. Both use the same helper, so the
        # narrow set can never disagree with what the widening would have
        # produced.
        if args.section_markets and not args.symbols and not args.section_ten_only:
            wanted, unbounded = _markets_each_section_needs(
                settings, tuple(sorted({name for name, _tf in passes}))
            )
            if unbounded:
                # A SECTION THAT TRADES ANYTHING CANNOT BE NARROWED AROUND.
                # Dropping markets while it is measured would silently stop
                # measuring it, which is the failure this whole file is about,
                # so the universe stays as it was and the run says why.
                print(
                    "  --section-markets ignored: "
                    f"{', '.join(sorted(unbounded))} declare no market and trade the "
                    "whole universe, so narrowing would silently stop measuring them"
                )
            elif not wanted:
                print("  --section-markets ignored: no section declares a market")
            else:
                dropped = [market for market in symbols if market not in wanted]
                symbols = wanted
                print(
                    f"  section markets: {len(symbols)} instead of {len(symbols) + len(dropped)}"
                    f" -- {', '.join(symbols)}"
                )
                if dropped:
                    print(
                        f"    {len(dropped)} markets left out, no section may trade them:"
                        f" {', '.join(dropped)}"
                    )

        section_markets: dict[str, list[str]] = {}
        widen = not args.symbols and not args.section_ten_only
        for name in sorted({name for name, _tf in passes}) if widen else ():
            config = getattr(settings.analysis, name)
            # TWO SPELLINGS OF "THE MARKETS THIS SECTION TRADES", and only one
            # of them was read. Sections five to ten and the BTC three carry
            # `allowed_symbols`; the XAUJPY sections carry a single `symbol`,
            # and `getattr(..., "allowed_symbols", ())` returned an empty tuple
            # for every one of them. So the widening this whole block exists to
            # do did not happen for section eleven, twelve or thirteen, and
            # XAUJPY was walked only if it happened to be in the universe
            # already -- producing exactly the zero row the comment above
            # describes, for the sections it was written to protect.
            allowed = tuple(getattr(config, "allowed_symbols", ()) or ())
            if not allowed:
                one = getattr(config, "symbol", "")
                allowed = (one,) if one else ()
            absent = [market for market in allowed if market not in symbols]
            if absent:
                section_markets[name] = absent
                symbols = symbols + absent
        for name, absent in section_markets.items():
            print(f"  {name} trades {len(absent)} markets the universe missed: {', '.join(absent)}")

        # THE HEADER NAMES WHAT IS MEASURED, not what is allowed to trade.
        #
        # It printed `live_enabled_modules`, so `history-one.cmd
        # impulse_retest 180` announced "DRY RUN — sections order_block" and
        # then spent twenty minutes measuring impulse_retest in silence. The
        # owner reasonably concluded it had hung.
        print(f"\n{'=' * 78}")
        print(f"DRY RUN — measuring {', '.join(sorted({name for name, _tf in passes}))}")
        shadowed = sorted({name for name, _tf in passes} - live)
        if shadowed:
            print(f"  shadowed (measured, NOT permitted real money): {', '.join(shadowed)}")
        print(f"  clocks: {', '.join(sorted({tf for _n, tf in passes}))}")
        # THE PASS COUNT, because "six sections" and "six sections on four
        # clocks" are the same header and a four-fold difference in runtime.
        # A run whose length surprises its owner is one he kills halfway.
        print(f"  {len(passes)} section/clock passes -- every bar is judged {len(passes)} times")
        if not args.sweep:
            print("  (each section on ITS OWN configured clock; --sweep tries them all)")
        print(f"{args.days} days to {end:%Y-%m-%d %H:%M} UTC, equity EUR {equity:.2f}")
        print(f"{len(symbols)} symbols, {settings.effective_risk_pct():.1f}% risk per trade")
        print(f"{'=' * 78}\n")

        # TWO DIFFERENT SITUATIONS, TWO DIFFERENT ANSWERS.
        #
        # `--no-m1 --sweep M1` is a user asking for two contradictory things,
        # and the right response is to say so and stop.
        #
        # `--no-m1 --live-only` when a LIVE section is configured on M1 is not
        # that. The clock came out of `config/eightcap.yaml`, not off this
        # command line, and dying on it would break the launcher that answers
        # "what would the account have done" the moment a section moves to M1.
        # Dropping the row silently would be worse still -- that is the missing
        # row this file has now shipped six times. So the flag loses and the
        # run says out loud that it did.
        finest = Timeframe.M5 if args.no_m1 else Timeframe.M1
        needs_m1 = _unresolvable_clocks(tuple(tf for _n, tf in passes), finest)
        if needs_m1 and args.sweep:
            raise SystemExit(needs_m1)
        if needs_m1:
            print(
                f"  NOTE: --no-m1 ignored. {needs_m1.split(' cannot')[0]} is a selected section's\n"
                f"        own clock and cannot be resolved on M5 bars, so M1 history is\n"
                f"        being fetched after all. This run will be slower than asked."
            )
            args.no_m1 = False
            finest = Timeframe.M1
        if args.sweep:
            required_frames = set(NEEDED)
        else:
            required_frames = {
                timeframe
                for name, tf_name in passes
                for timeframe in _frames_read(settings, Timeframe.parse(tf_name), finest, (name,))
            }
        # The sections themselves run on M5 and therefore do not ordinarily
        # request M15. The trend counterfactual does: without this, every M15
        # reading is silently zero and the table pretends it blocked every
        # trade. Fetch and attach the frame only for this measurement mode.
        if args.trend_grid or args.fault_exit_grid:
            required_frames.update(
                {Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4}
            )
        fetch_these = tuple(
            tf
            for tf in sorted(required_frames, key=lambda item: item.duration)
            if not (args.no_m1 and tf is Timeframe.M1)
        )
        results: dict[tuple[str, str], list[Decision]] = {key: [] for key in passes}
        skipped_symbols = 0
        unresolvable: dict[str, str] = {}
        #: symbol -> [(timeframe, first bar actually stored)] for every market
        #: whose history starts AFTER the window that was asked for.
        short_history: dict[str, list[tuple[str, datetime]]] = {}
        manage = _break_even_rule(settings)

        # WHERE THE TIME ACTUALLY GOES. I estimated this run's length twice
        # and was wrong twice -- ninety minutes, then ten. Guessing at it from
        # a synthetic benchmark cannot see the MT5 fetch, which on this VPS
        # logs "slow MT5 calls" of its own accord. The run now measures itself
        # and says so, so the next conversation about speed starts from a
        # number.
        fetch_seconds = 0.0
        compute_seconds = 0.0
        unaffordable: list[tuple[str, float]] = []

        # SECTION ELEVEN'S TWO LEGS, FETCHED ONCE FOR THE WHOLE RUN.
        #
        # XAUJPY = XAUUSD x USDJPY, and the section is silent without both. The
        # runner attaches them from its own scan; here they are two extra
        # symbol fetches on the section's clock, done before the walk so a
        # failure is one line at the top instead of a section that produces no
        # rows and reads as "found nothing".
        #
        # NAMED BY THE CANONICAL SYMBOL, resolved to the broker's suffixed one
        # for the fetch. The config says USDJPY, Eightcap lists USDJPY.i, and a
        # dict keyed the wrong way is a section that is silent forever with
        # nothing saying why -- that suffix has now caused exactly that failure
        # three times in this project.
        leg_frames: dict = {}
        legs_name = "section_eleven_xaujpy_legs_m5"
        if legs_name in {name for name, _tf in passes}:
            legs_cfg = settings.analysis.section_eleven_xaujpy_legs_m5
            legs_clock = Timeframe.parse(legs_cfg.timeframe)
            for canonical in (legs_cfg.base_leg, legs_cfg.quote_leg):
                broker = settings.instruments.broker_symbol(canonical)
                frame = None
                for candidate in dict.fromkeys((broker, canonical)):
                    try:
                        frame = (
                            store.frame(candidate, legs_clock, _fetch_from(legs_clock), end)
                            if store is not None
                            else fetch_mt5_history(
                                connector, candidate, legs_clock, _fetch_from(legs_clock), end
                            )
                        )
                    except Exception:  # noqa: BLE001 - try the unsuffixed name too
                        continue
                    break
                if frame is None or len(frame) < MIN_LEG_BARS:
                    print(
                        f"  {legs_name}: leg {canonical} unavailable, so this section"
                        " takes no trades in this run (no data is no trade)"
                    )
                    leg_frames = {}
                    break
                leg_frames[canonical] = (legs_clock, frame)
            if leg_frames:
                print(
                    f"  {legs_name}: legs "
                    + " x ".join(
                        f"{name} ({len(frame):,} {clock.value} bars)"
                        for name, (clock, frame) in leg_frames.items()
                    )
                )

        for index, symbol in enumerate(symbols, 1):
            began = _time.perf_counter()
            # THE FETCH IS THE OTHER SILENCE, and on a market this terminal has
            # never been asked for it is the longer one. Five new metals means
            # five M1 downloads of a hundred thousand bars each before a single
            # bar is walked, and nothing was printed until the whole symbol
            # finished. Said before it starts, so the wait has a name.
            print(f"  [{index}/{len(symbols)}] {symbol}: fetching history...", flush=True)
            try:
                spec = connector.spec(symbol)
                frames = {
                    tf: (
                        store.frame(symbol, tf, _fetch_from(tf), end)
                        if store is not None
                        else fetch_mt5_history(connector, symbol, tf, _fetch_from(tf), end)
                    )
                    for tf in fetch_these
                }
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the run
                skipped_symbols += 1
                print(f"  [{index}/{len(symbols)}] {symbol}: no history ({exc})")
                continue
            fetch_seconds += _time.perf_counter() - began
            began = _time.perf_counter()

            # SKIP WHAT CANNOT TRADE, before walking twelve thousand bars to
            # watch every setup be refused one at a time.
            hopeless = (
                None
                if btc_shadow
                else _hopeless_on_cost(
                    PositionSizer(settings),
                    spec,
                    frames,
                    tuple(Timeframe.parse(tf) for _n, tf in passes),
                )
            )
            if hopeless is not None:
                unaffordable.append((symbol, hopeless))
                compute_seconds += _time.perf_counter() - began
                print(
                    f"  [{index}/{len(symbols)}] {symbol}: skipped, a round trip is "
                    f"{hopeless:.0%} of the widest stop",
                    flush=True,
                )
                continue

            # Only the clocks a pass will actually use need to be deep enough.
            # Requiring it of every fetched timeframe threw away symbols over a
            # frame nothing was going to read.
            used = {Timeframe.parse(tf) for _, tf in passes} | {finest}
            if any(len(frames.get(tf, [])) < WARMUP + 10 for tf in used if tf in frames):
                skipped_symbols += 1
                continue

            # A WINDOW THE TERMINAL DOES NOT HAVE IS STILL REPORTED AS THE
            # WINDOW THAT WAS ASKED FOR. `--days 180` fetches from 180 days
            # back and MT5 answers with whatever depth it happens to hold --
            # on M1 that is routinely a fraction of it. The run then prints
            # "180 days" over a replay that covered forty, and every per-day
            # figure below it is divided by the wrong number of days.
            #
            # Same failure as the rest of this file: a missing bar and a quiet
            # market are the same silence. So say it out loud, per symbol, on
            # the clocks the passes actually read.
            for tf in sorted(used, key=lambda t: t.duration):
                frame = frames.get(tf)
                if frame is None or not len(frame):
                    continue
                first = frame.index[0]
                first = first.to_pydatetime() if hasattr(first, "to_pydatetime") else first
                if first.tzinfo is None:
                    first = first.replace(tzinfo=UTC)
                # A day of slack: a fetch that lands on a weekend legitimately
                # starts on the Monday and that is not a truncated history.
                if first > start + timedelta(days=1):
                    short_history.setdefault(symbol, []).append((tf.value, first))

            # GROUPED BY CLOCK so the context is built once instead of once
            # per section. Two sections on one clock read an identical
            # MarketContext, and building it is the largest cost in the loop.
            by_clock: dict[str, list[str]] = {}
            for name, tf_name in passes:
                by_clock.setdefault(tf_name, []).append(name)

            for tf_name, names in by_clock.items():
                clock = Timeframe.parse(tf_name)
                # `<` NOT `<=`. The old test refused a clock equal to the
                # resolution frame, which is exactly the M1 case, and it
                # refused it by `continue` -- no row, no message, no zero. A
                # sweep asked for M1 came back with M1 simply absent, which is
                # the failure mode this file has now produced six times: a
                # silence that reads as "nothing there" when it means "never
                # ran". The refusal that remains is the one that is genuinely
                # impossible -- a clock FINER than what it would be walked out
                # on -- and it says so out loud, once.
                if clock not in frames or clock.duration < finest.duration:
                    if tf_name not in unresolvable:
                        unresolvable[tf_name] = (
                            f"no {tf_name} history"
                            if clock not in frames
                            else f"{tf_name} is finer than the {finest.value} resolution frame"
                        )
                    continue
                group = []
                flattened = {
                    item.casefold() for item in settings.trade_management.pre_close_flatten_comments
                }
                for name in names:
                    tuned = _retimed(settings, name, tf_name)
                    only = [m for m in build_analysis_modules(tuned) if m.name == name]
                    section_manage = manage
                    shadow_trigger = getattr(
                        getattr(tuned.analysis, name), "shadow_break_even_at_r", None
                    )
                    if shadow_trigger is not None and not args.fixed_exits:
                        # `--fixed-exits` MEANS THIS ONE TOO. Zeroing
                        # `break_even_at_r` above removes the account-wide rule,
                        # but a section carrying its own
                        # `shadow_break_even_at_r` would quietly put break-even
                        # back for itself -- three sections do -- and the run
                        # would print "no break-even" over a result that had it.
                        section_manage = (
                            None if shadow_trigger <= 0.0 else (float(shadow_trigger), 0.0)
                        )
                    comment = broker_comment(name, is_addon=False, experimental_live=True)
                    fixed = {
                        item.casefold() for item in settings.trade_management.fixed_exit_comments
                    }
                    flatten_time = None
                    # THE EVENING FLATTEN IS A TIME EXIT, and the header this
                    # flag prints says there is none. Section ten is the one
                    # family that carries it, so without this line a
                    # `--fixed-exits` run of section ten would close positions
                    # on the clock while claiming the stop and target decided
                    # every one of them.
                    if comment.casefold() in flattened and not args.fixed_exits:
                        flatten_time = settings.filters.session.evening_flat_by_class.get(
                            spec.asset_class.value
                        )
                    group.append(
                        (
                            name,
                            (
                                _RawShadowEngine(only[0], getattr(tuned.analysis, name))
                                if btc_shadow
                                else ConfluenceEngine(only, tuned.analysis.confluence)
                            ),
                            PositionSizer(tuned),
                            None if comment.casefold() in fixed else section_manage,
                            flatten_time,
                            btc_shadow,
                            args.btc_jarvis_replay or args.jarvis_replay,
                            # THE SAME SWITCH `--s5-exit fixed` USES, which is
                            # the point: `_one_clock` reads this one field to
                            # decide that the trade runs to its entry stop or
                            # its target and nothing intervenes. A second
                            # mechanism for the same thing is a second thing to
                            # keep in step.
                            args.s5_exit or ("fixed" if args.fixed_exits else ""),
                        )
                    )
                produced = _one_clock(
                    sections=group,
                    symbol=symbol,
                    spec=spec,
                    frames=frames,
                    start=start,
                    end=end,
                    equity=equity,
                    clock=clock,
                    resolve_on=finest,
                    needed=tuple(
                        sorted(
                            set(_frames_read(settings, clock, finest, tuple(names)))
                            | ({Timeframe.M15} if args.trend_grid else set()),
                            key=lambda timeframe: timeframe.duration,
                        )
                    ),
                    manage_grid=exit_variants,
                    fault_exit_grid=args.fault_exit_grid,
                    legs=(leg_frames if legs_name in names else None),
                )
                for name, rows in produced.items():
                    results[(name, tf_name)].extend(rows)
            done = sum(
                1
                for v in results.values()
                for d in v
                if d.symbol == symbol and d.outcome == "TRADE"
            )
            # EVERY SYMBOL, TRADES OR NOT. This printed only when a symbol
            # produced trades, so a run whose first eleven markets are FX --
            # which form setups and lose every one of them to the cost wall --
            # showed nothing at all for eleven markets. Silence and a hang look
            # identical from the outside, and the owner sat on one for ten
            # minutes before asking.
            compute_seconds += _time.perf_counter() - began
            # AN ETA FROM THE RUN'S OWN PACE, not from me. I have estimated the
            # length of this run four times -- ninety minutes, then ten, then
            # six, then forty-five -- and been wrong every time, twice by a
            # factor of four, because the VPS is slower than the machine I
            # benchmark on and the section count keeps changing. Six live
            # sections evaluate every bar six times where two used to evaluate
            # it twice.
            #
            # After two markets there is enough to extrapolate from, and the
            # only honest source for it is the clock on this machine.
            elapsed = fetch_seconds + compute_seconds
            eta = ""
            if index >= 2 and index < len(symbols):
                remaining = elapsed / index * (len(symbols) - index)
                eta = f"   ~{remaining / 60:.0f} min left"
            print(
                f"  [{index}/{len(symbols)}] {symbol}: {done} trades"
                f"   [fetch {fetch_seconds:.0f}s, compute {compute_seconds:.0f}s]{eta}",
                flush=True,
            )

        decisions = [d for v in results.values() for d in v]
    finally:
        connector.shutdown()

    if unresolvable:
        # A clock that was asked for and never walked. This used to be a bare
        # `continue`, so the row simply was not in the table and an absent row
        # reads exactly like a row of zeros.
        print("\nCLOCKS ASKED FOR AND NOT MEASURED")
        for tf_name, why in sorted(unresolvable.items()):
            print(f"    {tf_name:<6} {why}")

    if short_history:
        print(f"\nSHORTER THAN THE {args.days} DAYS ASKED FOR")
        print("  The terminal has less stored history than the window. These markets")
        print("  were replayed from the date shown, not from the start of the window,")
        print("  so their trade counts and per-day figures cover LESS time than the")
        print("  header says. Deepen it in MT5 (Tools > Options > Charts > max bars)")
        print("  and scroll the chart back, or fetch once with ophalen.cmd.")
        for name, rows in sorted(short_history.items()):
            worst = max(rows, key=lambda row: row[1])
            missing = (worst[1] - start).days
            print(
                f"    {name:<12} {worst[0]:<4} starts {worst[1]:%Y-%m-%d}"
                f"   ({missing} of {args.days} days missing)"
            )

    if unaffordable:
        print(
            f"\nSKIPPED ON COST — {len(unaffordable)} markets where even the widest"
            " clock leaves nothing payable"
        )
        for name, share in sorted(unaffordable, key=lambda row: -row[1]):
            print(f"    {name:<12} {share:>6.0%} of the stop")
        print("  These form setups and the sizer refuses every one. Walking their")
        print("  bars measures the same refusal twelve thousand times.")

    total = fetch_seconds + compute_seconds
    if total > 0:
        print(
            f"\nTIME  fetch {fetch_seconds / 60:.1f} min ({fetch_seconds / total:.0%})"
            f"   compute {compute_seconds / 60:.1f} min ({compute_seconds / total:.0%})"
        )
        print("  Whichever of those dominates is the one worth attacking next.")

    if args.sweep:
        _sweep_report(results, equity, args.days, _break_even_rule(settings) is not None)
        _clock_overlap(results, args.days)
    _live_config_report(results, settings, equity, args.days)
    if args.btc_jarvis_replay or args.jarvis_replay:
        # ONE SHARED POSITION BOOK ACROSS EVERY SECTION, applied to the whole
        # run rather than per section. Until now the cap was REPORTED and not
        # enforced -- "how often would five have been open at once" -- and the
        # reason given was that enforcing it needs an ordering rule. There is
        # one, and it is not invented: chronological. The account opens the
        # trade that comes first and refuses the one that arrives with no slot
        # left, and `_under_the_slot_cap` walks exactly that.
        offered = [d for d in decisions if d.outcome == "TRADE"]
        allowed, refused_because = _under_the_slot_cap(
            offered,
            settings.effective_max_positions(equity),
            share_between_sections=settings.risk.sections_may_share_a_symbol,
            refuse_opposite=settings.risk.refuse_opposite_direction_across_sections,
            legs_per_symbol=max(1, args.legs_per_symbol),
        )
        if args.legs_per_symbol > 1:
            print(
                f"\nSTACKING: a section may hold {args.legs_per_symbol} positions in one "
                f"symbol. Replay only; the account holds one."
            )
            print(
                "  Read the DRAWDOWN with the R, not the R on its own. Stacking the same "
                "model on the same market at the same time is not a second bet, it is the "
                "first one at a larger size -- so more R is expected and means nothing by "
                "itself."
            )
        # NAMED SEPARATELY, because they are different problems with different
        # fixes. "The account was full" is answered by the slot count; "this
        # section already holds that symbol" is not, and calling both the first
        # thing is what made raising the cap look like the answer.
        why = {
            "SYMBOL_ALREADY_HELD": "this section already holds a position in this symbol",
            "SYMBOL_HELD_BY_ANOTHER_SECTION": (
                "another section holds this symbol and the two may not share it"
            ),
            "ACCOUNT_POSITION_LIMIT": "every account slot was already occupied",
        }
        allowed_ids = {id(row) for row in allowed}
        for row in offered:
            if id(row) not in allowed_ids:
                reason = refused_because.get(id(row), "ACCOUNT_POSITION_LIMIT")
                row.outcome = reason
                row.note = why[reason]
        if args.btc_jarvis_replay:
            _btc_jarvis_replay_contract(settings, equity, len(offered), len(allowed))
        else:
            _jarvis_replay_contract(settings, equity, len(offered), len(allowed))
        # WHAT THE GATES COST, in both replay modes. A gate is only worth its
        # place if the setups it refused went on to lose, and that question has
        # the same answer shape whichever sections are being replayed.
        _gate_counterfactual_report(decisions)
        if args.jarvis_replay:
            _what_the_account_would_be_worth(allowed, settings, equity, args.days)

    _report(
        decisions,
        equity,
        args.days,
        skipped_symbols,
        _break_even_rule(settings) is not None,
        sections=tuple(sorted({name for name, _tf in passes})),
    )
    if args.trend_grid:
        _trend_grid_report(decisions, _break_even_rule(settings) is not None)
        _s10_quality_filter_report(decisions, _break_even_rule(settings) is not None)
    if args.fault_exit_grid:
        _fault_exit_report(
            decisions, _break_even_rule(settings) is not None, equity
        )
    _gates_this_run_does_not_apply(
        btc_research_parity=args.btc_research_parity,
        btc_jarvis_replay=args.btc_jarvis_replay,
        jarvis_replay=args.jarvis_replay,
    )
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
                    "clock",
                    "direction",
                    "entry",
                    "stop",
                    "target",
                    "lots",
                    "risk_money",
                    "risk_pct",
                    # `result_r` is the fixed stop the research measured;
                    # `managed_r` is the same trade under break-even, which is
                    # what the account runs. Both columns, always, so a
                    # spreadsheet cannot quietly total the wrong one.
                    "result_r_fixed_stop",
                    "pnl_money_fixed_stop",
                    "managed_r_LIVE",
                    "managed_money_LIVE",
                    # Already subtracted from both R columns above. Here so
                    # the gross number stays recoverable and so the size of
                    # the haircut is visible per clock -- it is 1% of an H4
                    # stop and 12% of an M1 one.
                    "cost_r_charged",
                    "mae_r",
                    "mfe_r",
                    "spread_stop_pct",
                    "atr_regime_ratio",
                    "h1_distance_atr",
                    "h4_distance_atr",
                    "breakout_atr",
                    "retest_bars",
                    "breakout_body_atr",
                    "breakout_wick_share",
                    "breakout_volume_ratio",
                    "trend_m5",
                    "trend_m15",
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
                        d.pass_key[1],
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
                        round(d.cost_r, 4),
                        round(d.mae_r, 4),
                        round(d.mfe_r, 4),
                        round(d.spread_stop_pct, 3),
                        round(d.atr_regime, 4),
                        round(d.h1_distance_atr, 4),
                        round(d.h4_distance_atr, 4),
                        round(d.breakout_atr, 4),
                        d.retest_bars,
                        round(d.breakout_body_atr, 4),
                        round(d.breakout_wick_share, 4),
                        round(d.breakout_volume_ratio, 4),
                        d.trend_m5,
                        d.trend_m15,
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


def _under_daily_money_stop(
    trades: list[Decision], limit: float, managed: bool
) -> tuple[list[Decision], int]:
    """Refuse entries after resolved losses reach the fixed daily money stop.

    Exits are settled before each later entry, so a multi-hour position counts
    on the day it actually closes rather than on the day it opened. Open-trade
    drawdown is unavailable to this account-level walk; live is stricter
    because its equity mark includes that floating loss.
    """
    if limit <= 0.0:
        return sorted(trades, key=lambda row: row.when), 0

    import heapq

    accepted: list[Decision] = []
    exits: list[tuple[datetime, int, float]] = []
    daily_pnl: dict[object, float] = {}
    skipped = 0
    serial = 0
    for row in sorted(trades, key=lambda item: item.when):
        while exits and exits[0][0] <= row.when:
            closed_at, _serial, money = heapq.heappop(exits)
            key = closed_at.date()
            daily_pnl[key] = daily_pnl.get(key, 0.0) + money
        if daily_pnl.get(row.when.date(), 0.0) <= -limit:
            skipped += 1
            continue
        accepted.append(row)
        money = row.managed_money if managed else row.pnl_money
        if money is not None:
            serial += 1
            heapq.heappush(exits, (row.exit_at or row.when, serial, float(money)))
    return accepted, skipped


def _trend_grid_report(decisions: list[Decision], managed: bool) -> None:
    """Cost and benefit of aligning S5/S6 entries with closed M5/M15 trend."""
    names = {"section_five_ndx100_m5", "section_six_gold_m5"}
    relevant = [
        row
        for row in decisions
        if row.outcome == "TRADE"
        and row.pass_key[0] in names
        and _live_exit(row, managed) is not None
    ]
    print(f"\n{'=' * 78}")
    print("SHORT-TREND ENTRY FILTER — SAME TRADES, NO LIVE CHANGE")
    print("  Trend = close above/below EMA20 and EMA20 sloping the same way over")
    print("  the last three closed bars. Mixed readings are not called a trend.")
    print("  Positive net saved R means the filter helped; negative means it hurt.")
    for module in sorted(names):
        rows = [row for row in relevant if row.pass_key[0] == module]
        if not rows:
            continue
        baseline = sum(_live_exit(row, managed) or 0.0 for row in rows)
        print(f"\n  {module}   {len(rows)} trades   baseline {baseline:+.2f} R")
        print(
            f"    {'filter':<14}{'kept':>6}{'blocked':>9}{'wins cut':>10}"
            f"{'loss cut':>10}{'net saved':>12}{'result':>10}"
        )

        def direction(row: Decision) -> int:
            return 1 if row.direction == "LONG" else -1

        variants = (
            ("M5 aligned", lambda row: row.trend_m5 == direction(row)),
            ("M15 aligned", lambda row: row.trend_m15 == direction(row)),
            (
                "M5 + M15",
                lambda row: row.trend_m5 == direction(row)
                and row.trend_m15 == direction(row),
            ),
        )
        for label, allowed in variants:
            blocked = [row for row in rows if not allowed(row)]
            blocked_r = [_live_exit(row, managed) or 0.0 for row in blocked]
            wins_cut = sum(value for value in blocked_r if value > 0.0)
            losses_cut = -sum(value for value in blocked_r if value < 0.0)
            saved = losses_cut - wins_cut
            print(
                f"    {label:<14}{len(rows) - len(blocked):>6}{len(blocked):>9}"
                f"{wins_cut:>+10.2f}{losses_cut:>+10.2f}{saved:>+12.2f}"
                f"{baseline + saved:>+10.2f}"
            )
            midpoint = sorted(rows, key=lambda item: item.when)[len(rows) // 2].when
            early = [row for row in rows if row.when < midpoint]
            late = [row for row in rows if row.when >= midpoint]
            def half_saved(part):
                rejected = [_live_exit(row, managed) or 0.0 for row in part if not allowed(row)]
                return -sum(rejected)
            early_saved, late_saved = half_saved(early), half_saved(late)
            verdict = "BEIDE HELPEN" if early_saved > 0 and late_saved > 0 else "NIET ROBUUST"
            print(
                f"      tijdhelften: vroeg {early_saved:+.2f}R | laat {late_saved:+.2f}R"
                f" -> {verdict}"
            )
    print(f"{'=' * 78}")


def _drawdown_r(values: list[float]) -> float:
    equity = peak = worst = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def _s10_quality_filter_report(decisions: list[Decision], managed: bool) -> None:
    """Pre-registered S10 candle-shape candidates on identical entries.

    The limits are deliberately round values next to, rather than exact copies
    of, the exploratory tertile cuts (body 1.719 ATR, wick 0.098). That avoids
    presenting the best accidental decimal from one sample as a trading rule.
    """
    rows = sorted(
        (
            row for row in decisions
            if row.outcome == "TRADE"
            and row.pass_key[0] == "section_ten_gold_m1"
            and _live_exit(row, managed) is not None
        ),
        key=lambda row: row.when,
    )
    if not rows:
        return
    baseline_values = [float(_live_exit(row, managed) or 0.0) for row in rows]
    midpoint = rows[len(rows) // 2].when
    variants = (
        ("compact wick <=0.10", lambda row: row.breakout_wick_share <= 0.10),
        ("sterke body >=1.70", lambda row: row.breakout_body_atr >= 1.70),
        (
            "body+wick samen",
            lambda row: row.breakout_body_atr >= 1.70
            and row.breakout_wick_share <= 0.10,
        ),
    )
    print(f"\n{'=' * 92}")
    print("S10 FALSE-BREAKOUT KANDIDATEN — DEZELFDE ENTRIES, GEEN LIVE WIJZIGING")
    print("  Vaste ronde grenzen uit de eerdere diagnose; vroeg en laat moeten beide helpen.")
    print(
        f"  {'filter':<22}{'keep':>7}{'cut':>7}{'wins cut':>11}{'loss cut':>11}"
        f"{'result':>11}{'DD':>9}{'vroeg':>10}{'laat':>10}  verdict"
    )
    print(
        f"  {'CURRENT':<22}{len(rows):>7}{0:>7}{0.0:>+11.2f}{0.0:>+11.2f}"
        f"{sum(baseline_values):>+11.2f}{_drawdown_r(baseline_values):>9.2f}"
        f"{0.0:>+10.2f}{0.0:>+10.2f}  baseline"
    )
    for label, allowed in variants:
        kept = [row for row in rows if allowed(row)]
        blocked = [row for row in rows if not allowed(row)]
        blocked_values = [float(_live_exit(row, managed) or 0.0) for row in blocked]
        kept_values = [float(_live_exit(row, managed) or 0.0) for row in kept]
        wins_cut = sum(value for value in blocked_values if value > 0.0)
        losses_cut = -sum(value for value in blocked_values if value <= 0.0)

        def saved(part: list[Decision]) -> float:
            return -sum(
                float(_live_exit(row, managed) or 0.0)
                for row in part if not allowed(row)
            )

        early = [row for row in rows if row.when < midpoint]
        late = [row for row in rows if row.when >= midpoint]
        early_saved, late_saved = saved(early), saved(late)
        enough = len(kept) >= 200 and len(kept) >= len(rows) * 0.20
        robust = early_saved > 0.0 and late_saved > 0.0 and enough
        print(
            f"  {label:<22}{len(kept):>7}{len(blocked):>7}{wins_cut:>+11.2f}"
            f"{losses_cut:>+11.2f}{sum(kept_values):>+11.2f}"
            f"{_drawdown_r(kept_values):>9.2f}{early_saved:>+10.2f}"
            f"{late_saved:>+10.2f}  {'KANDIDAAT' if robust else 'AFWIJZEN'}"
        )
    print("  Een kandidaat is nog shadow: pas live na een causale accountreplay.")
    print(f"{'=' * 92}")


def _fault_exit_report(decisions: list[Decision], managed: bool, equity: float) -> None:
    """Show plainly whether a shadow exit saved losses or murdered winners."""
    rows = [
        row for row in decisions
        if row.outcome == "TRADE" and row.fault_grid_r and _live_exit(row, managed) is not None
    ]
    print(f"\n{'=' * 78}")
    print("SMART FAULT EXIT — SHADOW ONLY, NOTHING LIVE CHANGED")
    print("  Fires on CLOSED M5 only when BOTH adverse EMA20 drift and a break")
    print("  beyond the preceding six-bar structure agree. One signal cannot exit.")
    print("  SMART = confirmed failure at -0.25R OR confirmed give-back after +0.50R.")
    modules = sorted({row.pass_key[0] for row in rows})
    groups = [("ACCOUNT TOTAL", rows)] + [
        (module, [row for row in rows if row.pass_key[0] == module])
        for module in modules
    ]
    for module, section in groups:
        baseline_r = sum(_live_exit(row, managed) or 0.0 for row in section)
        baseline_eur = sum((_live_exit(row, managed) or 0.0) * row.risk_money for row in section)
        print(f"\n  {module} — {len(section)} identical trades")
        print(f"    {'exit':<16}{'acted':>7}{'result R':>10}{'net R':>10}{'loss saved':>13}"
              f"{'profit saved':>14}{'profit cut':>12}{'would stand':>14}")
        print(f"    {'CURRENT':<16}{0:>7}{baseline_r:>+10.2f}{0.0:>+10.2f}{0.0:>+13.2f}"
              f"{0.0:>+14.2f}{0.0:>+12.2f}{equity + baseline_eur:>14.2f}")
        labels = [label for label, _value in section[0].fault_grid_r]
        for label in labels:
            candidate = []
            for row in section:
                values = dict(row.fault_grid_r)
                candidate.append(float(values.get(label, _live_exit(row, managed)) or 0.0))
            bases = [float(_live_exit(row, managed) or 0.0) for row in section]
            deltas = [new - old for new, old in zip(candidate, bases, strict=True)]
            loss_saved = sum(
                max(0.0, delta)
                for delta, old in zip(deltas, bases, strict=True) if old <= 0
            )
            profit_saved = sum(
                max(0.0, delta)
                for delta, old in zip(deltas, bases, strict=True) if old > 0
            )
            profit_cut = sum(
                max(0.0, -delta)
                for delta, old in zip(deltas, bases, strict=True) if old > 0
            )
            result_r = sum(candidate)
            result_eur = sum(
                value * row.risk_money
                for value, row in zip(candidate, section, strict=True)
            )
            acted = sum(abs(delta) > 1e-9 for delta in deltas)
            print(f"    {label:<16}{acted:>7}{result_r:>+10.2f}{sum(deltas):>+10.2f}"
                  f"{loss_saved:>+13.2f}{profit_saved:>+14.2f}{profit_cut:>+12.2f}"
                  f"{equity + result_eur:>14.2f}")
            midpoint = sorted(section, key=lambda row: row.when)[len(section) // 2].when
            early_delta = sum(
                delta for delta, row in zip(deltas, section, strict=True)
                if row.when < midpoint
            )
            late_delta = sum(
                delta for delta, row in zip(deltas, section, strict=True)
                if row.when >= midpoint
            )
            if acted < 20:
                verdict = f"TE WEINIG ({acted} acties)"
            elif early_delta > 0 and late_delta > 0:
                verdict = "BEIDE HELPEN"
            else:
                verdict = "NIET ROBUUST"
            print(
                f"      tijdhelften: vroeg {early_delta:+.2f}R | laat {late_delta:+.2f}R"
                f" -> {verdict}"
            )
    print("\n  Positive net R means the exit helped after counting winners it cut.")
    print(f"{'=' * 78}")


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
    trades, refused_here = _under_the_slot_cap(
        everything,
        slots,
        share_between_sections=settings.risk.sections_may_share_a_symbol,
        refuse_opposite=settings.risk.refuse_opposite_direction_across_sections,
    )
    fixed_comments = {item.casefold() for item in settings.trade_management.fixed_exit_comments}
    fixed_names = {
        name
        for name, _clock in keys
        if broker_comment(name, is_addon=False, experimental_live=True).casefold() in fixed_comments
    }
    has_break_even = _break_even_rule(settings) is not None
    all_fixed = len(fixed_names) == len(keys)
    managed = has_break_even and not all_fixed
    trades, refused_daily = _under_daily_money_stop(
        trades, settings.risk.daily_loss_limit_money, managed
    )
    closed = [d for d in trades if _live_exit(d, managed) is not None]

    print("\n" + "=" * 78)
    print("THE LIVE CONFIGURATION -- this is the one that answers the question")
    print("=" * 78)
    print("  " + ", ".join(f"{name} on {clock}" for name, clock in keys))
    print(f"  max {slots} positions at once at EUR {equity:.2f}, one per symbol")
    if settings.risk.daily_loss_limit_money > 0.0:
        print(
            f"  daily loss stop EUR {settings.risk.daily_loss_limit_money:.2f}: "
            f"{refused_daily} later entries refused until the next trading day"
        )
        print("    replay counts resolved P/L; live also counts floating equity loss")
    if all_fixed:
        print("  exit: fixed broker stop/target, which these sections actually run")
    elif fixed_names:
        print("  exit: configured per section (fixed SL/TP or break-even)")
        print("    fixed SL/TP: " + ", ".join(sorted(fixed_names)))
    else:
        print("  exit: break-even stop, which these sections actually run")
    if len(everything) != len(trades):
        # NAMED BY THE RULE THAT REFUSED IT, not by the one that sounds likely.
        #
        # This line read "every slot was already busy" for every refusal the
        # shared book made, and the 180-day run showed 432 of them. All 432
        # were SYMBOL_ALREADY_HELD: a section's own open position in the way,
        # with the account cap never reached once. The owner raised the cap
        # from four slots to ten on the strength of that sentence, re-measured,
        # and nothing moved -- because the slot count was never what refused
        # them. One wrong word cost an afternoon and a live config change.
        # FROM THE WALK'S OWN ANSWER, not from `row.outcome`. This block runs
        # BEFORE the caller stamps the outcomes, so every row here still reads
        # "TRADE" and counting them would have printed an empty table under a
        # heading promising a breakdown. Found by the test, not by the run.
        dropped: dict[str, int] = {}
        for reason in refused_here.values():
            dropped[reason] = dropped.get(reason, 0) + 1
        total_dropped = len(everything) - len(trades)
        print(f"  {total_dropped} signals refused by the shared position book:")
        if dropped:
            names = {
                "SYMBOL_ALREADY_HELD": "the section already held that market",
                "SYMBOL_HELD_BY_ANOTHER_SECTION": "another section held it",
                "ACCOUNT_POSITION_LIMIT": "every account slot was occupied",
            }
            for outcome, count in sorted(dropped.items(), key=lambda row: -row[1]):
                print(f"     {count:>6}  {names.get(outcome, outcome)}")
            if not dropped.get("ACCOUNT_POSITION_LIMIT"):
                print(
                    "     The account cap refused NOTHING here, so raising it cannot "
                    "buy a trade."
                )
        # AND WHICH SECTION PAID FOR IT, because one number cannot say.
        #
        # The 4 September run printed "320 signals dropped" and left it there.
        # Underneath, section six lost 244 of its 600 trades and 30.55 R to a
        # busy slot while section ten lost 76 and 10.93 R -- three quarters of
        # the cost landed on the section that was not being widened.
        #
        # That is the arithmetic that decides whether a widening was worth it.
        # Section ten added 2,390 trades for +28.77 R and the cap took 30.55 R
        # off section six in the same window; a change that pays for itself out
        # of another section's pocket is not obviously a gain, and with one
        # number for the whole book it was invisible.
        kept: dict[str, list[Decision]] = {}
        for row in trades:
            kept.setdefault(row.module, []).append(row)
        seen: dict[str, list[Decision]] = {}
        for row in everything:
            seen.setdefault(row.module, []).append(row)
        rows: list[tuple[float, str, int, float]] = []
        for module, offered in seen.items():
            taken = kept.get(module, [])
            lost = len(offered) - len(taken)
            if lost <= 0:
                continue
            before = sum(_live_exit(d, managed) or 0.0 for d in offered)
            after = sum(_live_exit(d, managed) or 0.0 for d in taken)
            rows.append((before - after, module, lost, before - after))
        if rows:
            print("     what the book cost each section, in trades and in R:")
            for _sort, module, lost, cost in sorted(rows, key=lambda r: -r[0]):
                print(f"       {module:<28}{lost:>6} trades   {cost:>+8.2f} R")
            print("     A section paying for another section's trades is a real cost")
            print("     and it does not show up in that one total above.")
            print("     Where the reason is 'already held', those setups overlap in time")
            print("     with a position the section HAS. Taking them is the same idea at")
            print("     a larger size, not a second one -- read the drawdown with the R.")
            print("     `hoeveel.cmd 180 stapel` measures exactly that.")

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

    if managed:
        _break_even_verdict(trades, settings)
    _manage_grid_report(trades)
    _by_market_report(trades)
    _by_hour_report(trades, settings)
    _hours_for_other_sections(trades, settings)
    _is_this_real(trades, keys, managed)
    _shadow_report(results, settings, slots, managed, days)


def _shadow_report(results: dict, settings, slots: int, managed: bool, days: int) -> None:
    """What a section that is NOT allowed to trade would have done.

    THE POINT OF SWITCHING A SECTION OFF RATHER THAN DELETING IT is that it
    keeps being measured, and that measurement is what decides whether it
    comes back. This script did not honour that: `passes` was built from
    `live_enabled_modules`, so switching `impulse_retest` off on 30 August
    removed it from the 180-day run entirely -- and the earlier 180-day data
    had actually favoured it over the section that stayed.

    Judged on the same exit as the live block, under the same slot cap, so the
    two numbers are comparable rather than merely adjacent.
    """
    live = set(settings.analysis.confluence.live_enabled_modules)
    shadow = sorted(
        {
            (name, clock)
            for (name, clock) in results
            if name not in live
            and getattr(settings.analysis, name, None) is not None
            and getattr(getattr(settings.analysis, name), "timeframe", None) == clock
        }
    )
    if not shadow:
        return

    print("\n" + "-" * 78)
    print("SHADOWED — measured, not permitted to trade")
    print("-" * 78)
    for name, clock in shadow:
        rows, _why = _under_the_slot_cap(
            [d for d in results[(name, clock)] if d.outcome == "TRADE"], slots
        )
        closed = [d for d in rows if _live_exit(d, managed) is not None]
        if not closed:
            print(f"  {name} on {clock}: no resolved trades")
            continue
        won = sum(1 for d in closed if (_live_exit(d, managed) or 0) > 0)
        total_r = sum(_live_exit(d, managed) or 0.0 for d in closed)
        money = sum((d.managed_money if managed else d.pnl_money) or 0.0 for d in closed)
        print(
            f"  {name:<16} {clock:<4} {len(closed):>5} trades  {won / len(closed):>5.1%} win"
            f"  {total_r:+8.2f} R  EUR {money:+9.2f}   ({len(closed) / max(days, 1):.1f} a day)"
        )
    print("  Switched off is not deleted. These are the numbers that decide")
    print("  whether a section comes back, so they have to be printed.")


def _by_hour_report(trades: list[Decision], settings) -> None:
    """Section ten's result per UTC hour, with its own blocked window marked.

    THE ONE THING THAT DECIDES WHETHER 07:00-13:00 DESERVED TO BE SHUT.

    That window was picked by splitting sixteen hours on the same 180 days the
    section was calibrated on and cutting the worst six. On ANY sequence that
    finds a bad block -- it is the definition of the worst six hours of a
    sample, not evidence that those hours are bad. And the six in question are
    the London morning, the busiest and most liquid gold has, which is exactly
    where the most trades and therefore the most spread sit.

    So the hours are open and this prints what they did. Read it the boring
    way: is 07:00-13:00 negative AGAIN, on a wider set of markets, or was it
    six hours that happened to be the worst once? A block that only looks bad
    in the sample it was chosen from has not been confirmed by seeing it again
    in that same sample -- it has to be bad here, with the metals added, to
    mean anything.

    Both halves of the period are printed for the same reason as the
    break-even grid: one number per hour on one sample is how a window like
    this gets chosen in the first place.

    PER MARKET AND NOT PER SECTION, as of 4 September, because reading it per
    section hid the finding it exists to surface. Section ten's blocked window
    is a per-SYMBOL setting: `blocked_hours_by_symbol` REPLACES the global
    07:00-13:00 for any symbol it names, and the four crosses name only 16 and
    17. So the crosses traded the whole London morning while XAUUSD did not,
    the section-level table added the two together, and nothing said which
    market the hours belonged to.

    It cost -168.07 R over 1664 trades in the 180-day run, all of it the
    crosses. The section total was -31.10 R; the hours were the whole of it and
    then some.
    """
    rows = [
        row
        for row in trades
        if row.outcome == "TRADE" and row.result_r is not None and "section_ten" in row.module
    ]
    if not rows:
        return
    config = settings.analysis.section_ten_gold_m1
    blocked = range(config.blocked_start_hour_utc, config.blocked_end_hour_utc)

    order = sorted(row.when for row in rows)
    split = order[int(len(order) * 0.6)]

    print("\nSECTION TEN BY UTC HOUR — did the closed window deserve to be closed?")
    print(
        f"  entry window {config.entry_start_hour_utc:02d}:00-"
        f"{config.entry_end_hour_utc:02d}:00 UTC, "
        + (
            f"blocked {config.blocked_start_hour_utc:02d}:00-"
            f"{config.blocked_end_hour_utc:02d}:00"
            if len(blocked)
            else "no blocked window"
        )
    )
    print(f"    {'hour':<6}{'trades':>7}{'total R':>9}{'per trade':>11}{'early':>8}{'late':>8}   ")

    by_hour: dict[int, list[Decision]] = {}
    for row in rows:
        by_hour.setdefault(int(row.when.hour), []).append(row)

    for hour in sorted(by_hour):
        got = by_hour[hour]
        values = [r.result_r for r in got if r.result_r is not None]
        early = [r.result_r for r in got if r.when < split and r.result_r is not None]
        late = [r.result_r for r in got if r.when >= split and r.result_r is not None]
        mark = "  <- was blocked" if hour in blocked else ""
        print(
            f"    {hour:02d}:00 {len(values):>6}{sum(values):>+9.2f}"
            f"{sum(values) / len(values):>+11.3f}{sum(early):>+8.2f}{sum(late):>+8.2f}{mark}"
        )

    old_block = [r.result_r for r in rows if 7 <= int(r.when.hour) < 13 and r.result_r is not None]
    rest = [r.result_r for r in rows if not 7 <= int(r.when.hour) < 13 and r.result_r is not None]
    if old_block and rest:
        print(
            f"\n  07:00-13:00 together: {len(old_block)} trades, {sum(old_block):+.2f} R "
            f"({sum(old_block) / len(old_block):+.3f} per trade)"
        )
        print(
            f"  every other hour:     {len(rest)} trades, {sum(rest):+.2f} R "
            f"({sum(rest) / len(rest):+.3f} per trade)"
        )
        print("  Negative again, on the wider market set? Then the block goes back.")
        print("  Positive, or level with the rest? Then it was the worst six hours")
        print("  of one sample and shutting them cost trades for nothing.")


def _hours_for_other_sections(trades: list[Decision], settings) -> None:
    """The same hour question for every OTHER section that blocks hours.

    THE TABLE ABOVE IS HARDCODED TO SECTION TEN, and section eleven trades the
    same four crosses on the same clock with its own blocked hours -- so
    `sectie11.cmd` would have printed no hour table at all. An absent table is
    not "the hours are fine"; it is nobody having looked, which is the
    confusion this file keeps shipping under new names.

    It matters here specifically. Section ten's 180-day run found -168.07 R in
    07:00-13:00 UTC on those crosses, against +57.85 R in every other hour, and
    section eleven does not block that window. Whether a fitted model has the
    same hole is an open question and this is what answers it.
    """
    rows = [
        row
        for row in trades
        if row.outcome == "TRADE" and row.result_r is not None and "section_ten" not in row.module
    ]
    by_section: dict[str, list[Decision]] = {}
    for row in rows:
        section = getattr(settings.analysis, row.module, None)
        if section is not None and getattr(section, "blocked_hours_by_symbol", None):
            by_section.setdefault(row.module, []).append(row)
    if not by_section:
        return

    for module in sorted(by_section):
        got = by_section[module]
        config = getattr(settings.analysis, module)
        order = sorted(row.when for row in got)
        split = order[int(len(order) * 0.6)]
        print(f"\n{module.upper()} BY UTC HOUR — which hours pay, and which are open?")
        print(f"    {'hour':<6}{'trades':>7}{'total R':>9}{'per trade':>11}{'early':>8}{'late':>8}")

        by_hour: dict[int, list[Decision]] = {}
        for row in got:
            by_hour.setdefault(int(row.when.hour), []).append(row)
        for hour in sorted(by_hour):
            values = [r.result_r for r in by_hour[hour] if r.result_r is not None]
            early = [r.result_r for r in by_hour[hour] if r.when < split and r.result_r is not None]
            late = [r.result_r for r in by_hour[hour] if r.when >= split and r.result_r is not None]
            # A blocked hour with trades in it means the block is per symbol
            # and this hour's markets are not the ones it names -- exactly how
            # section ten's crosses traded the whole London morning unnoticed.
            shut = {
                market
                for market in getattr(config, "allowed_symbols", ())
                if config.hour_is_blocked(market, hour)
            }
            mark = f"  <- shut for {len(shut)}/{len(config.allowed_symbols)}" if shut else ""
            print(
                f"    {hour:02d}:00 {len(values):>6}{sum(values):>+9.2f}"
                f"{sum(values) / len(values):>+11.3f}{sum(early):>+8.2f}{sum(late):>+8.2f}{mark}"
            )

        scored = [r for r in got if r.result_r is not None]
        london = [r.result_r for r in scored if 7 <= int(r.when.hour) < 13]
        rest = [r.result_r for r in scored if not 7 <= int(r.when.hour) < 13]
        if london and rest:
            print(
                f"\n  07:00-13:00 together: {len(london)} trades, {sum(london):+.2f} R "
                f"({sum(london) / len(london):+.3f} per trade)"
            )
            print(
                f"  every other hour:     {len(rest)} trades, {sum(rest):+.2f} R "
                f"({sum(rest) / len(rest):+.3f} per trade)"
            )
            print("  Section ten lost -0.101 per trade in that window on these same")
            print("  crosses. Negative here too means the same hole, not a new finding.")


def _by_market_report(trades: list[Decision]) -> None:
    """Per section, per market: is a new symbol carrying its weight or diluting?

    THE QUESTION A WIDENING ACTUALLY ASKS, and the first version of this
    report could not answer it. Section ten went from one metal to six and
    came back with 429 trades and +19.66 R against 16 trades on the live
    account. More trades and more R -- but "BY SECTION" adds the six together,
    so five markets carrying a sixth, or one market carrying five, print
    identically.

    That is the difference between a widening that worked and a widening that
    happened to be rescued by the market it started from, and it is the whole
    reason for adding symbols in the first place.

    Sections that trade one market print one row, which costs nothing.
    """
    taken = [row for row in trades if row.outcome == "TRADE" and row.result_r is not None]
    if not taken:
        return
    by_section: dict[str, list[Decision]] = {}
    for row in taken:
        by_section.setdefault(row.module, []).append(row)
    interesting = {
        name: rows for name, rows in by_section.items() if len({r.symbol for r in rows}) > 1
    }
    if not interesting:
        return

    print("\nPER SECTION, PER MARKET — is every symbol paying its own way?")
    print("  A section is only as widened as its worst market. One symbol")
    print("  carrying five is not a wider section, it is the old one plus noise.")

    for module, rows in sorted(interesting.items()):
        order = sorted(row.when for row in rows)
        split = order[int(len(order) * 0.6)]
        print(f"\n  {module}   {len(rows)} trades over {len({r.symbol for r in rows})} markets")
        print(
            f"    {'market':<12}{'trades':>7}{'total R':>9}{'per trade':>11}"
            f"{'early':>8}{'late':>8}{'hit':>7}"
        )
        by_symbol: dict[str, list[Decision]] = {}
        for row in rows:
            by_symbol.setdefault(row.symbol, []).append(row)
        ranked = sorted(by_symbol.items(), key=lambda kv: -sum(r.result_r or 0.0 for r in kv[1]))
        for symbol, got in ranked:
            values = [r.result_r for r in got if r.result_r is not None]
            early = [r.result_r for r in got if r.when < split and r.result_r is not None]
            late = [r.result_r for r in got if r.when >= split and r.result_r is not None]
            won = sum(1 for v in values if v > 0)
            print(
                f"    {symbol:<12}{len(values):>7}{sum(values):>+9.2f}"
                f"{sum(values) / len(values):>+11.3f}{sum(early):>+8.2f}"
                f"{sum(late):>+8.2f}{won / len(values):>7.1%}"
            )
        losing = [s for s, g in by_symbol.items() if sum(r.result_r or 0.0 for r in g) < 0]
        if losing:
            names = ", ".join(sorted(losing))
            print(f"    {len(losing)} of {len(by_symbol)} markets negative: {names}")


def _grid_stats(values: list[float]) -> tuple[float, float, float]:
    """`(total, per trade, hit rate)` for one exit rule's results."""

    if not values:
        return 0.0, 0.0, 0.0
    won = sum(1 for v in values if v > 0)
    return sum(values), sum(values) / len(values), won / len(values)


def _paired_t(differences: list[float]) -> float:
    """t of the mean of `differences` against zero, or 0.0 when undefined.

    PAIRED, AND THAT IS THE WHOLE VALUE OF THIS TABLE. Every rule is resolved
    on the SAME entry, so the difference between two rules is measured trade by
    trade and the enormous variance of the trades themselves cancels out. An
    unpaired comparison of two totals would need several times the sample to
    see the same effect.
    """

    n = len(differences)
    if n < 8:
        # Under eight paired observations a t is arithmetic, not evidence.
        return 0.0
    mean = sum(differences) / n
    var = sum((d - mean) ** 2 for d in differences) / (n - 1)
    if var <= 0.0:
        # Every trade moved by exactly the same amount, or none moved at all.
        return 0.0 if mean == 0.0 else float("inf") * (1.0 if mean > 0 else -1.0)
    return mean / (var / n) ** 0.5


def _grid_line(label: str, rows, early, late, pick, baseline=None, marker: str = "") -> None:
    """One exit rule's row, against the fixed exit on the same trades.

    A free function so the closure cannot capture its caller's loop variables
    -- the kind of binding that works today and silently reports the last
    section's numbers under every section's name the moment the call moves.
    """

    values = [v for v in (pick(r) for r in rows) if v is not None]
    if not values:
        return
    early_v = [v for v in (pick(r) for r in early) if v is not None]
    late_v = [v for v in (pick(r) for r in late) if v is not None]
    total, per_trade, hit = _grid_stats(values)
    t_text = "     -"
    if baseline is not None:
        pairs = [
            (v, b)
            for v, b in ((pick(r), baseline(r)) for r in rows)
            if v is not None and b is not None
        ]
        t = _paired_t([v - b for v, b in pairs])
        t_text = "   inf" if t == float("inf") else f"{t:>6.2f}"
    print(
        f"    {label:<20}{total:>+9.2f}{per_trade:>+11.3f}"
        f"{sum(early_v):>+9.2f}{sum(late_v):>+9.2f}{hit:>7.1%}{t_text}  {marker}"
    )


def _manage_grid_report(trades: list[Decision], variants: tuple[ExitVariant, ...] = ()) -> None:
    """Every exit rule on the same entries, per section, with a verdict.

    WHAT THIS ANSWERS AND WHAT IT DOES NOT.

    It answers: on the trades this section actually took, which way of managing
    the position afterwards would have kept the most of them? Every row is the
    same entries, the same costs, the same flatten time -- only what happens
    after entry differs. That is the narrow question and it is the one worth
    asking first.

    It does NOT answer whether to ship the winner. Any rule that exits earlier
    frees its symbol earlier, an earlier free symbol takes the next setup, and
    a section trading a different set of entries is a different section. A rule
    that wins here has earned a full replay with the position book following
    it, not a promotion.

    AND IT IS A BEST-OF-MANY, WHICH IS THE TRAP. Comparing thirty rules and
    keeping the best one is not the same experiment as testing one rule. On
    pure noise the best of thirty looks good roughly thirty times as easily,
    and this account has around 150 trades in a section -- so the report prints
    the count, splits the period, and names a winner ONLY when it wins in both
    halves and clears a threshold raised for the number of rules compared.
    """

    graded = [
        row for row in trades if row.grid_r and row.result_r is not None and row.outcome == "TRADE"
    ]
    if not graded:
        return
    if not variants:
        # READ FROM WHAT THE WALK ACTUALLY MEASURED, in the order it measured
        # it, rather than from a grid constant this function chose for itself.
        # Those are two lists that have to agree, and when they stop agreeing
        # the table silently prints one rule's numbers under another's name.
        known = {v.label: v for v in (*exit_grid(wide=True), *exit_grid(wide=False))}
        variants = tuple(
            known.get(label, ExitVariant(label=label, trigger_r=1.0))
            for label, _value in graded[0].grid_r
        )

    order = sorted(row.when for row in graded)
    split = order[int(len(order) * 0.6)]
    # BY LABEL, NEVER BY POSITION. The old table read `grid_r[i]` against the
    # grid's i-th entry, so adding, removing or reordering a rule silently
    # relabelled every column -- the reader would have had no way to tell.
    by_label = {v.label: v for v in variants}

    def pick(label):
        def read(row):
            for name, value in row.grid_r:
                if name == label:
                    return value
            return None

        return read

    fixed_label = next((v.label for v in variants if v.kind == "fixed"), "")
    baseline = pick(fixed_label) if fixed_label else (lambda r: r.result_r)

    print("\nEXIT GRID — the same trades, a different rule after entry")
    print(f"  {len(variants)} rules compared on identical entries. Only the exit differs;")
    print("  same setups, same costs, same flatten time, same position book.")
    print("  t is PAIRED against the fixed exit: the trades cancel, so it reads")
    print("  the difference the rule made and not the noise of the section.")

    by_section: dict[str, list[Decision]] = {}
    for row in graded:
        by_section.setdefault(row.module, []).append(row)

    for module, rows in sorted(by_section.items()):
        early = [r for r in rows if r.when < split]
        late = [r for r in rows if r.when >= split]
        print(f"\n  {module}   {len(rows)} trades   ({len(early)} early / {len(late)} late)")
        print(
            f"    {'exit rule':<20}{'total R':>9}{'per trade':>11}"
            f"{'early R':>9}{'late R':>9}{'hit':>7}{'t':>6}"
        )

        verdict = _grid_verdict(rows, early, late, variants, pick, baseline)
        for variant in variants:
            _grid_line(
                variant.label,
                rows,
                early,
                late,
                pick(variant.label),
                None if variant.kind == "fixed" else baseline,
                marker="<-- best" if variant.label == verdict.get("label") else "",
            )
        _print_grid_verdict(verdict)

    print("\n  A stop that moves costs the trades that dip and would have reached")
    print("  the target anyway, and saves the ones that turn. Which effect is")
    print("  larger is a property of the section, not of the idea, so it has to")
    print("  be read per section and never carried across.")
    unread = {label for row in graded for label, _v in row.grid_r} - set(by_label)
    if unread:
        # The walk measured a rule this table has no column for. That is a
        # disagreement between the grid and the report, and printing the table
        # as if it were complete is how a silent one survives.
        print(f"\n  WARNING: {len(unread)} measured rule(s) are missing from this table:")
        print(f"    {', '.join(sorted(unread))}")


def _grid_verdict(rows, early, late, variants, pick, baseline) -> dict:
    """The best rule, or nothing, with the reason it did not qualify.

    THREE HURDLES, and a rule has to clear all of them:

        it beats the fixed exit on the whole period
        it beats it in the EARLY part and in the LATE part separately
        its paired t clears a threshold raised for the number of rules tried

    The third is the one that matters most and it is the one nobody applies.
    Trying thirty rules and keeping the best is thirty chances to be fooled, so
    the bar moves with the count: roughly Bonferroni, which is blunt but errs
    toward refusing a rule rather than shipping one.
    """

    def total(subset, reader):
        return sum(v for v in (reader(r) for r in subset) if v is not None)

    fixed_total = total(rows, baseline)
    fixed_early = total(early, baseline)
    fixed_late = total(late, baseline)

    # Bonferroni on a two-sided 5% test, as a normal-quantile approximation.
    # Blunt on purpose: it refuses more rules than a sharper correction would,
    # and every rule it refuses costs nothing while every rule it wrongly
    # admits gets shipped to a live account.
    tried = max(1, sum(1 for v in variants if v.kind != "fixed"))
    bar = _bonferroni_t(tried)

    ranked = []
    for variant in variants:
        if variant.kind == "fixed":
            continue
        reader = pick(variant.label)
        pairs = [
            (v, b)
            for v, b in ((reader(r), baseline(r)) for r in rows)
            if v is not None and b is not None
        ]
        if not pairs:
            continue
        t = _paired_t([v - b for v, b in pairs])
        ranked.append(
            {
                "label": variant.label,
                "t": t,
                "total": total(rows, reader),
                "early": total(early, reader),
                "late": total(late, reader),
            }
        )
    if not ranked:
        return {"bar": bar, "tried": tried, "why": "no rule produced a comparable result"}

    # RANKED BY t AND NOT BY TOTAL R, and that is a choice worth stating.
    #
    # The largest total is the rule that got lucky on the biggest trades. On
    # 150 trades one gold day can carry a whole column, and shipping that is
    # how a section gets tuned to a week that will not repeat. The paired t
    # asks the other question -- did this rule improve trade after trade --
    # and that is the one that survives.
    #
    # The biggest total is still named below when it differs, because the
    # owner asked which rule pays the most and hiding it would be answering a
    # different question than the one asked.
    ranked.sort(key=lambda row: row["t"], reverse=True)
    best = ranked[0]
    fattest = max(ranked, key=lambda row: row["total"])
    result = {
        "bar": bar,
        "tried": tried,
        "best": best,
        "fixed_total": fixed_total,
        "fattest": fattest if fattest["label"] != best["label"] else None,
    }
    if best["total"] <= fixed_total:
        result["why"] = "no rule beat the fixed exit over the whole period"
        return result
    if best["early"] <= fixed_early or best["late"] <= fixed_late:
        result["why"] = (
            f"{best['label']} leads overall but not in both halves "
            f"(early {best['early']:+.2f} vs {fixed_early:+.2f}, "
            f"late {best['late']:+.2f} vs {fixed_late:+.2f}) -- that is the sample"
        )
        return result
    if best["t"] < bar:
        result["why"] = (
            f"{best['label']} leads by t={best['t']:.2f}, under the t>{bar:.2f} this "
            f"table needs for best-of-{tried}"
        )
        return result
    result["label"] = best["label"]
    return result


def _bonferroni_t(tried: int) -> float:
    """The paired t a best-of-`tried` winner has to clear to mean anything.

    A two-sided 5% test split across `tried` comparisons, as a normal
    quantile -- close enough at these sample sizes and, being an approximation
    that runs slightly HIGH, it errs toward refusing a rule.
    """

    from statistics import NormalDist

    alpha = 0.05 / max(1, tried)
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def _print_grid_verdict(verdict: dict) -> None:
    """Say plainly whether anything won, and say plainly when nothing did."""

    fattest = verdict.get("fattest")
    if fattest is not None:
        print(
            f"    Biggest total: {fattest['label']} at {fattest['total']:+.2f} R "
            f"(t={fattest['t']:.2f}) -- more R, less consistently, so it is not "
            f"the pick."
        )
    if verdict.get("label"):
        best = verdict["best"]
        print(
            f"    VERDICT: {best['label']} beats the fixed exit by "
            f"{best['total'] - verdict['fixed_total']:+.2f} R, in both halves, "
            f"t={best['t']:.2f} over the t>{verdict['bar']:.2f} needed for "
            f"best-of-{verdict['tried']}."
        )
        print("    Worth a full replay with the position book following it. Not live yet.")
        return
    # SAYING NOTHING WON IS A RESULT. An absent verdict line would read as an
    # oversight, and the reader would go back to the numbers and pick the
    # biggest one themselves -- which is the exact thing the hurdles exist to
    # prevent.
    print(f"    VERDICT: keep the exit as configured. {verdict.get('why', 'no rule qualified')}.")


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

    # HOW MUCH OF THE RESULT IS ONE MONTH, which the four boxes do not ask.
    #
    # The 180-day order_block run passed "every month positive" on five of six
    # and read +1.33 sigma, and the shape underneath was:
    #
    #     Mar +4.90  Apr +0.40  May -1.90  Jun +3.30  Jul +0.70  Aug +26.20
    #
    # August alone is 78% of the whole six-month total. Strip it and the other
    # 1,077 trades make +0.007 R each, which is nothing. A result carried by
    # one month is a result that has not repeated, however many trades sit
    # underneath it, and neither the trade count nor the sigma nor the
    # months-positive box can see that.
    concentration = 0.0
    if len(by_month) > 1 and abs(total_r) > 1e-9:
        best = max(by_month.values(), key=lambda rows: sum(rows))
        concentration = sum(best) / total_r
        if concentration > 0.5:
            others = total_r - sum(best)
            others_n = n - len(best)
            print(
                f"\n    CONCENTRATED: one month is {concentration:.0%} of the whole result."
                f"\n    Without it: {others:+.2f} R over {others_n} trades"
                f" = {others / max(others_n, 1):+.4f} R a trade."
            )

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
        (
            concentration <= 0.5,
            f"best month is {concentration:.0%} of the total",
            "no single month carrying more than half the result",
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


def _clock_overlap(results: dict, days: int) -> None:
    """Would two clocks give twice the trades, or the same trade twice?

    THE SWEEP CANNOT ANSWER THIS AND IT IS THE QUESTION BEHIND "more trades".
    Each row is measured in isolation, so M15 at 60 trades beside M30 at 40
    reads as a hundred. If order_block sees the same move on both clocks
    within the hour, running both is one idea at double the stake, not two
    independent chances -- and doubling the stake on one idea is the thing
    this account has a rule against.

    Counted as: a trade on the slower clock that has a trade on the faster
    one, same symbol, same direction, inside one bar of the slower clock.
    Deliberately generous about the window, because the failure mode worth
    catching is "these are the same move", not "these are the same second".
    """
    by_section: dict[str, dict[str, list]] = {}
    for (name, clock), decisions in results.items():
        trades = [d for d in decisions if d.outcome == "TRADE"]
        if trades:
            by_section.setdefault(name, {})[clock] = sorted(trades, key=lambda d: d.when)
    pairs = [(name, clocks) for name, clocks in by_section.items() if len(clocks) > 1]
    if not pairs:
        return

    print(f"\n{'-' * 78}")
    print("WOULD TWO CLOCKS GIVE TWICE THE TRADES?")
    print(f"{'-' * 78}")
    for name, clocks in sorted(pairs):
        order = sorted(clocks, key=lambda c: Timeframe.parse(c).duration)
        for slow_index in range(1, len(order)):
            slow_name = order[slow_index]
            slow = clocks[slow_name]
            window = Timeframe.parse(slow_name).duration
            for fast_name in order[:slow_index]:
                fast = clocks[fast_name]
                seen = 0
                for trade in slow:
                    if any(
                        other.symbol == trade.symbol
                        and other.direction == trade.direction
                        and abs((other.when - trade.when).total_seconds()) <= window.total_seconds()
                        for other in fast
                    ):
                        seen += 1
                share = seen / len(slow) if slow else 0.0
                verdict = (
                    "mostly the SAME move -- running both doubles the stake, not the chances"
                    if share > 0.5
                    else "largely independent -- running both really does add trades"
                )
                print(
                    f"  {name:<16} {slow_name:>4} vs {fast_name:<4} "
                    f"{seen:>4}/{len(slow):<5} ({share:>5.0%})  {verdict}"
                )
    print(f"  ({days} days. A pair over 50% is one idea wearing two hats.)")


#: Gates the LIVE runner applies and this script does not, with what each one
#: refused in the 24 hours of 31 August -- 414 setups formed, 0 trades taken.
#:
#: THIS IS THE GAP BETWEEN 9 TRADES A DAY AND NONE. Every number this file
#: produces comes out of `ConfluenceEngine.evaluate` followed by
#: `PositionSizer.size`, and nothing else. The account runs both of those with
#: eight more gates around them, all of them in `runner/service.py` or
#: `filters/`, none of them reachable from here.
#:
#: Of the 414 setups that formed live in a day, ELEVEN died at something this
#: script models. The other 403 died at gates it has never once applied. So
#: "+36.80 R over 100 days" is not a forecast of the account; it is what the
#: strategy does with those eight gates removed, and the two were being
#: compared as though they were the same measurement.
#:
#: Printed on every run rather than written in a docstring, because the
#: docstring is not what gets screenshotted at two in the morning.
NOT_MODELLED: tuple[tuple[str, int, str], ...] = (
    ("AWAITING_CONFIRMATION", 140, "price ran against the idea over the last 3 M5 bars"),
    ("NEWS_BLACKOUT", 78, "a calendar event was near"),
    ("TARGET_RARELY_REACHED", 62, "the target's historical reach rate was too low"),
    ("SPREAD_EATS_THE_STOP", 62, "spread against stop width, checked before sizing"),
    ("MARKET_TOO_QUIET", 44, "the liveliness filter"),
    ("AWAITING_PULLBACK", 9, "setup lifecycle: alive, not yet entered"),
    ("VOLUME_SPIKE", 7, "volume spike filter"),
    ("ENTRY_OVEREXTENDED", 1, "entry quality"),
)


#: Position management the LIVE account runs and this script does not.
#:
#: THE ENTRY GATES WERE ONLY HALF THE GAP. `_resolve` models exactly one exit
#: rule -- the break-even move -- and `TradeManagementConfig` carries a dozen
#: more that fire on every open position. Reading a dry-run R as the account's
#: result assumes those do nothing, and they are the difference between
#: "+1.00R" and "half off at 1.5R, the rest trailed out at 0.6R".
#:
#: Not necessarily worse. Different, and unmeasured here.
#: WHAT `_resolve` ACTUALLY WALKS when `--jarvis-replay` hands it the live
#: `TradeManagementConfig`. This list used to be inside EXITS_NOT_MODELLED and
#: the contract therefore told the reader that six rules it DOES apply were
#: missing.
#:
#: That mislabel is the usual defect running backwards: instead of claiming a
#: rule it does not apply, the report disowned six it does. It made the account
#: replay read as far cruder than it is, and it is why the owner asked whether
#: the measurement could be made "accurater" -- most of what he was asking for
#: was already there and being denied on screen.
EXITS_MODELLED_UNDER_FULL_MANAGEMENT: tuple[tuple[str, str], ...] = (
    ("partial_close_at_r", "half the position comes off, when the lot allows it"),
    ("trailing_mode atr", "the rest trails the configured ATR behind"),
    ("profit_lock_from_r", "a share of the peak profit is locked in"),
    ("giveback_arm_r", "out if it hands back its configured share of the peak"),
    ("peak_stall_minutes", "out if it stalls near its peak"),
    ("time_exit_hours", "out after the deadline, or the plan horizon multiple"),
)

#: The three that genuinely cannot come out of OHLC. Each needs something bars
#: do not carry: a running health model, the thesis the idea was built on, or a
#: tick-level spread history.
EXITS_NOT_MODELLED: tuple[tuple[str, str], ...] = (
    ("health_tighten_at_r", "the health monitor tightens the stop"),
    ("thesis_invalidation_at_r", "out when the reason for the trade breaks"),
    ("spread_squeeze_share", "out when the spread goes abnormal on the tick"),
)


def _jarvis_replay_contract(settings, equity: float, offered: int, allowed: int) -> None:
    """Exactly what the account replay can and cannot substantiate.

    Printed before the money, not after it, because the number underneath is
    the part that gets screenshotted and the boundary is the part that gets
    forgotten.
    """

    print(f"\n{'=' * 78}")
    print("JARVIS ACCOUNT REPLAY — WHAT IS IN IT")
    print(f"{'=' * 78}")
    print("  APPLIED, on top of everything the ordinary dry run already does")
    print(
        "    live spread against stop width, refused above "
        f"{settings.analysis.confluence.max_spread_share_of_stop:.0%} (SPREAD_EATS_THE_STOP)"
    )
    # A GATE THAT IS SWITCHED OFF IN THE CONFIG IS NOT A GATE THIS RUN APPLIED.
    # Printing the line either way is how a claim outlives the setting behind
    # it, and every reader of this block would then believe a filter ran that
    # returns immediately.
    lively = settings.filters.liveliness.enabled
    print(
        "    market liveliness (MARKET_TOO_QUIET), on causal bar history"
        if lively
        else "    market liveliness: NOT APPLIED, filters.liveliness.enabled is false"
    )
    spike = settings.filters.volume_spike
    print(
        f"    M1 volume spike above {spike.extreme_multiple:.1f}x the "
        f"{spike.lookback_bars}-bar median (VOLUME_SPIKE)"
        if spike.enabled
        else "    M1 volume spike: NOT APPLIED, filters.volume_spike.enabled is false"
    )
    print(
        "    target reach and direction advantage (TARGET_RARELY_REACHED)"
        if settings.analysis.confluence.require_direction_advantage
        else "    target reach applied; direction advantage OFF in config"
    )
    print(
        f"    one shared position book: max {settings.effective_max_positions(equity)} open at "
        f"EUR {equity:.2f}, sections "
        f"{'may' if settings.risk.sections_may_share_a_symbol else 'may not'} share a symbol"
    )
    print(f"      {offered - allowed} of {offered} entries were refused by that book")
    print("    the real Eightcap spread per bar, the real minimum lot, the real sizer")
    print(
        f"    fixed daily loss stop: EUR {settings.risk.daily_loss_limit_money:.2f}; "
        "new entries pause for the rest of that day"
    )
    print(
        "    minimum-lot override: "
        + ("ON" if settings.risk.allow_minimum_lot_above_target else "OFF")
        + f", still capped at {settings.effective_max_risk_pct():.2f}% per trade"
    )
    print("    the configured break-even move and pre-close flatten")
    print("  NOT APPLIED, AND EACH ONE ONLY EVER REMOVES TRADES")
    print("    the news blackout: no calendar archive of the window exists, and")
    print("      guessing one is the opposite of the fail-safe this account runs")
    print("    the AI review: `ai.provider` is local_history and the answers it")
    print("      would have given were never given")
    print("  NOT RECONSTRUCTIBLE FROM BARS AT ALL")
    print("    tick-level fill slippage beyond the recorded spread")
    print("    intrabar ordering of partial / trailing / health / peak-stall exits")
    print("    margin changes, requotes and outages")
    print("  So this is an UPPER BOUND on the entries and an approximation of the")
    print("  exits. It is much closer to the account than the plain dry run, and it")
    print("  is still not a broker statement.")


@dataclass(slots=True)
class _CompoundedRun:
    """What the account actually did, trade by trade, with the stake walking."""

    balance: float
    peak: float
    worst_drawdown: float
    taken: int
    skipped_too_small: int
    skipped_daily_loss: int
    minimum_lot_overrides: int
    first_risk: float
    last_risk: float
    biggest_risk: float
    curve: list[tuple[datetime, float]]
    #: What the balance has to reach before the stake can grow by one lot step
    #: at all, or 0.0 when it already did. On a EUR 252 account with a 0.01
    #: minimum lot this is the number that decides whether compounding is a
    #: real effect or a rounding error, and nothing else in this report can
    #: show it.
    next_step_balance: float


def _compound(trades: list[Decision], settings, equity: float) -> _CompoundedRun:
    """Re-SIZE every trade at the balance it would actually have had.

    NOT A RESCALE, AND THAT IS THE WHOLE POINT OF THIS FUNCTION.

    The old line multiplied each result by `balance_then / balance_start`,
    which quietly assumes the account can always afford the trade it is about
    to take. On EUR 252 with a 0.01 minimum lot that assumption is wrong in
    both directions at once:

      * EARLY, and after a drawdown, the minimum lot risks MORE than the
        percentage allows. Live that trade does not happen -- the sizer refuses
        it as UNDERCAPITALIZED. A rescale takes it anyway, at a stake the
        account could not have posted.
      * LATER, as the balance grows, the stake does not move smoothly. It moves
        in lot steps. A rescale gives fractional lots no broker accepts and
        credits profit on volume that was never on the book.

    So this walks the trades in time order and, at each one, asks the same
    question the sizer asks: how many whole lot steps of THIS instrument does
    `risk_pct` of the CURRENT balance buy? Below the minimum lot the trade is
    dropped and counted, exactly as live drops it.

    `risk_per_lot`, `volume_min` and `volume_step` come off the row itself --
    put there when the trade was booked, because by report time the connector
    is shut and the instrument spec is gone.

    WHAT IS STILL AN APPROXIMATION, said plainly rather than left to be found:
    the ORDER of trades cannot change, and live it would. A trade the account
    could not afford here leaves its slot free, and live that slot would go to
    the next setup instead. This walk keeps the sequence the shared position
    book produced at the starting balance. It is much closer than a rescale
    and it is not a broker statement.
    """

    risk_pct = settings.effective_risk_pct() / 100.0
    balance = equity
    peak = equity
    worst = 0.0
    taken = 0
    skipped = 0
    skipped_daily = 0
    overridden = 0
    first_risk = 0.0
    last_risk = 0.0
    biggest = 0.0
    next_step = 0.0
    curve: list[tuple[datetime, float]] = []

    daily_limit = settings.risk.daily_loss_limit_money
    day = None
    day_start_balance = equity

    for row in sorted(trades, key=lambda item: item.when):
        if balance <= 0.0:
            # A WIPED ACCOUNT STOPS TRADING. Continuing to book results on a
            # negative balance is how a replay produces a recovery that could
            # not have happened.
            break
        result_r = row.managed_r
        if result_r is None:
            continue

        row_day = row.when.date()
        if row_day != day:
            day = row_day
            day_start_balance = balance
        if daily_limit > 0.0 and balance - day_start_balance <= -daily_limit:
            skipped_daily += 1
            continue

        per_lot = row.risk_per_lot
        step = row.volume_step
        minimum = row.volume_min
        if per_lot <= 0.0 or step <= 0.0 or minimum <= 0.0:
            # An older row with no lot economics. Fall back to the rescale
            # rather than dropping the trade, and it stays visible because
            # `taken` and the flat total are printed side by side.
            balance += float(row.managed_money or 0.0) * (balance / equity)
        else:
            wanted = balance * risk_pct
            steps = int(wanted / per_lot / step)
            volume = steps * step
            if volume < minimum - 1e-12:
                minimum_risk = minimum * per_lot
                ceiling_money = balance * settings.effective_max_risk_pct() / 100.0
                fits_daily = daily_limit <= 0.0 or minimum_risk <= daily_limit + 1e-9
                if (
                    settings.risk.allow_minimum_lot_above_target
                    and minimum_risk <= ceiling_money + 1e-9
                    and fits_daily
                ):
                    volume = minimum
                    overridden += 1
                else:
                    skipped += 1
                    continue
            risk_money = volume * per_lot
            balance += float(result_r) * risk_money
            taken += 1
            if first_risk == 0.0:
                first_risk = risk_money
            last_risk = risk_money
            biggest = max(biggest, risk_money)

        peak = max(peak, balance)
        if peak > 0:
            worst = max(worst, (peak - balance) / peak)
        curve.append((row.when, balance))
        # THE BALANCE THAT WOULD BUY ONE MORE LOT STEP. Kept from the last
        # affordable trade, because that is the instrument the account is
        # actually sized against.
        if per_lot > 0.0 and step > 0.0 and risk_pct > 0.0:
            next_step = (minimum + step) * per_lot / risk_pct

    return _CompoundedRun(
        balance=balance,
        peak=peak,
        worst_drawdown=worst,
        taken=taken,
        skipped_too_small=skipped,
        skipped_daily_loss=skipped_daily,
        minimum_lot_overrides=overridden,
        first_risk=first_risk,
        last_risk=last_risk,
        biggest_risk=biggest,
        curve=curve,
        next_step_balance=0.0 if biggest > first_risk else next_step,
    )


def _what_the_account_would_be_worth(
    trades: list[Decision], settings, equity: float, days: int
) -> None:
    """The question in the owner's words: start it N days ago, what is it now.

    TWO NUMBERS, AND THEY ARE NOT THE SAME QUESTION.

    FLAT STAKE sizes every trade off the SAME starting balance. It answers
    "what did the edge pay" without the account's own growth flattering it,
    and every euro in it is one the sizer actually produced.

    COMPOUNDED is what the account does: the stake walks with the balance,
    every trade re-SIZED at the money available at that moment, in whole lot
    steps, with the minimum lot re-checked. A trade the account could not
    afford does not happen -- which is why the two lines can disagree on the
    NUMBER OF TRADES and not only on the euros.
    """

    print(f"\n{'=' * 78}")
    print(f"IF THIS HAD BEEN RUNNING FOR {days} DAYS")
    print(f"{'=' * 78}")
    booked = sorted(
        (row for row in trades if row.managed_money is not None), key=lambda row: row.when
    )
    if not booked:
        # AN EMPTY RUN SAYS SO. A missing block reads as a zero block, and
        # this file has shipped that confusion more times than any other.
        print("  No trade survived to a result. That is not EUR 0.00 of edge; it is")
        print("  zero observations, and the BY SECTION table above says where they went.")
        return

    flat = sum(float(row.managed_money or 0.0) for row in booked)
    wins = sum(1 for row in booked if (row.managed_money or 0.0) > 0)
    total_r = sum(float(row.managed_r or 0.0) for row in booked)
    run = _compound(booked, settings, equity)

    print(f"  started with            EUR {equity:>10.2f}")
    print(f"  risk per trade          {settings.effective_risk_pct():.2f}% of the balance")
    print("\n  FLAT STAKE -- every trade sized off that same starting balance")
    print(f"    {len(booked)} trades, {wins} winners ({wins / len(booked):.0%}), {total_r:+.2f} R")
    print(f"    result              EUR {flat:>+10.2f}   ->  EUR {equity + flat:>10.2f}")

    print("\n  COMPOUNDED -- re-sized at the balance it had, in whole lot steps")
    print(
        f"    result              EUR {run.balance - equity:>+10.2f}   ->  "
        f"EUR {run.balance:>10.2f}   ({(run.balance / equity - 1.0) * 100:+.1f}%)"
    )
    print(f"    highest balance     EUR {run.peak:>10.2f}")
    print(f"    worst drawdown from a peak            {run.worst_drawdown * 100:>5.1f}%")
    if run.first_risk or run.last_risk:
        # THE STAKE GROWING IS THE THING HE ASKED ABOUT. Printed, because a
        # single end balance cannot show whether it grew smoothly or in one
        # jump, and because the first figure is what the account risks TODAY.
        print(
            f"    risk per trade      EUR {run.first_risk:>6.2f} at the start   ->  "
            f"EUR {run.last_risk:>6.2f} at the end   (peak EUR {run.biggest_risk:.2f})"
        )
    if run.next_step_balance > 0.0:
        # THE ANSWER TO "DOES COMPOUNDING EVEN DO ANYTHING HERE", and on this
        # account size it is usually no.
        #
        # 2% of EUR 252 is EUR 5.04 and one minimum lot of gold risks EUR 3.85,
        # so the sizer buys 0.0131 lots and rounds DOWN to 0.01. At EUR 358 it
        # buys 0.0186 and still rounds to 0.01. The stake cannot move until the
        # balance can afford two whole steps -- and until then compounding is
        # arithmetically identical to a flat stake.
        #
        # A rescale hid this completely: it grew the stake smoothly through
        # lot sizes no broker would accept, and produced a curve this account
        # cannot have.
        pct = settings.effective_risk_pct()
        print(f"    THE STAKE NEVER MOVED. It stayed at EUR {run.first_risk:.2f} the whole way.")
        print(
            f"      The minimum lot is indivisible: the balance has to reach about "
            f"EUR {run.next_step_balance:.0f}"
        )
        print(f"      before {pct:.2f}% buys one more lot step. Below that, compounded and flat")
        print("      are the same number and the difference between them is rounding.")
    print(f"    {run.taken} of {len(booked)} trades were affordable")
    if run.minimum_lot_overrides:
        print(
            f"    {run.minimum_lot_overrides} used the broker minimum above the "
            f"{settings.effective_risk_pct():.2f}% target, but stayed below the "
            f"{settings.effective_max_risk_pct():.2f}% hard ceiling and EUR "
            f"{settings.risk.daily_loss_limit_money:.2f} daily stop"
        )
    if run.skipped_daily_loss:
        print(
            f"    {run.skipped_daily_loss} entries were refused after the day had lost "
            f"EUR {settings.risk.daily_loss_limit_money:.2f}; trading resumed next day"
        )
    if run.skipped_too_small:
        # NOT A ROUNDING DETAIL. On this account size it is the difference
        # between a replay and a fantasy: a trade whose minimum lot risks more
        # than the percentage allows is refused live and taken by a rescale.
        print(
            f"    {run.skipped_too_small} were NOT: the minimum lot risked more than "
            f"{settings.effective_risk_pct():.2f}% of the balance at that moment,"
        )
        print("      which is exactly what the live sizer refuses as UNDERCAPITALIZED")

    if run.curve:
        _print_balance_curve(run.curve, equity)

    print(
        "\n  The order of trades is the one the shared position book produced at the"
        "\n  STARTING balance. Live, a trade the account could not afford leaves its"
        "\n  slot free and the next setup takes it, so the real sequence would differ."
    )

    by_section: dict[str, list[Decision]] = {}
    for row in booked:
        by_section.setdefault(row.pass_key[0] or row.module, []).append(row)
    print(f"\n  {'section':<34}{'trades':>7}{'R':>10}{'EUR flat':>11}{'per trade':>11}")
    for name, rows in sorted(
        by_section.items(), key=lambda item: -sum(float(r.managed_money or 0.0) for r in item[1])
    ):
        money = sum(float(r.managed_money or 0.0) for r in rows)
        r_total = sum(float(r.managed_r or 0.0) for r in rows)
        print(
            f"  {name:<34}{len(rows):>7}{r_total:>+10.2f}{money:>+11.2f}"
            f"{r_total / len(rows):>+11.3f}"
        )
    print("  (per section the FLAT euros, because a compounded share of a shared")
    print("   balance cannot be attributed to one section without inventing a rule)")

    live = set(settings.analysis.confluence.live_enabled_modules)
    shadow = sorted(set(by_section) - live)
    if shadow:
        print(
            "\n  Sections NOT on the real-money allowlist, whose euros above are"
            f" hypothetical:\n    {', '.join(shadow)}"
        )


def _print_balance_curve(curve: list[tuple[datetime, float]], equity: float) -> None:
    """The balance month by month, so growth has a shape and not just an end.

    An end balance cannot tell a steady climb from one lucky month followed by
    six flat ones, and this account has already been fooled by exactly that:
    one month carried 74% of a 180-day result.
    """

    by_month: dict[str, float] = {}
    for when, balance in curve:
        by_month[f"{when:%Y-%m}"] = balance
    if len(by_month) < 2:
        return
    print("\n    balance at the end of each month")
    previous = equity
    for month, balance in sorted(by_month.items()):
        change = balance - previous
        bar = "+" * min(int(abs(change) / max(equity, 1.0) * 60), 40)
        print(
            f"      {month}   EUR {balance:>9.2f}   {change:>+9.2f}   "
            f"{'-' if change < 0 else ''}{bar}"
        )
        previous = balance


def _btc_jarvis_replay_contract(settings, equity: float, offered: int, allowed: int) -> None:
    """Print exactly what the second S15-S17 measurement can substantiate.

    A historical OHLC archive can replay deterministic chart and portfolio
    rules.  It cannot recreate tick-order, an unavailable news archive, or a
    review response that was never made.  Naming that boundary in the result
    prevents this command from being mistaken for a broker statement.
    """
    print(f"\n{'=' * 78}")
    print("S15-S17 JARVIS REPLAY CONTRACT")
    print(f"{'=' * 78}")
    print("  APPLIED")
    print("    frozen S15/S16/S17 detector and next-bar entry")
    print("    historical BTCUSD spread plus the configured execution allowance")
    print("    live spread/stop ceiling, target-reach and direction-advantage gates")
    print("    market-liveliness and M1 volume-spike gates on causal bar history")
    print("    strategy stop/target and horizon")
    print("    full OHLC-reproducible management in live order: profit lock,")
    print("      break-even, broker-feasible partial close, ATR trail, hard giveback,")
    print("      healthy-branch peak stall and plan-derived time exit")
    print(
        f"    shared Jarvis position book: max {settings.effective_max_positions(equity)} "
        f"positions at EUR {equity:.2f} ({offered - allowed} of {offered} entries blocked)"
    )
    print("    minimum-lot 2% eligibility envelope used to approve these shadow sections")
    print("  INCLUDED, PASS-THROUGH BY THE LIVE CONFIG")
    print("    entry confirmation: each section has its own closed-bar confirmation")
    print("    pullback lifecycle/entry quality: each section owns its measured entry")
    print("  SKIPPED BY OPERATOR REQUEST")
    print("    historical news blackout")
    print("  NOT RECONSTRUCTIBLE FROM OHLC BARS")
    print("    tick-by-tick fill slippage, AI review answers, margin changes and outages")
    print("    AI health/thesis verdicts, persistent tick spread-squeeze state")
    print("    exact intrabar ordering when one M1 candle crosses multiple actions")
    print("  The total below is therefore the deterministic S15-S17 account replay,")
    print("  not a promise or an exact reconstruction of fills the broker never made.")


#: Which of `NOT_MODELLED` the replay modes actually apply. Named ONCE, and
#: against the same strings the table uses, so a gate that is implemented and
#: a gate that is claimed cannot drift apart -- `test_dry_run_script` asserts
#: every name here exists in the table.
#:
#: Both the contract block and the counterfactual read this. They arrived from
#: two different sessions with two copies of the same four names, which is
#: exactly how a gate ends up applied by one and unreported by the other.
JARVIS_REPLAY_APPLIES: frozenset[str] = frozenset(
    {
        "TARGET_RARELY_REACHED",
        "SPREAD_EATS_THE_STOP",
        "MARKET_TOO_QUIET",
        "VOLUME_SPIKE",
    }
)


def _gate_counterfactual_report(decisions: list[Decision]) -> None:
    """Show whether each historical gate removed profit or loss.

    A GATE IS ONLY WORTH ITS PLACE IF WHAT IT REFUSED WENT ON TO LOSE. Without
    this the replay reports how many setups each gate ate and says nothing
    about whether eating them helped, which is the more expensive half of the
    question.
    """
    rows = [
        row
        for row in decisions
        if row.outcome in JARVIS_REPLAY_APPLIES and row.managed_r is not None
    ]
    if not rows:
        return
    print("\n  GATE COUNTERFACTUALS -- what refused setups did with that gate OFF")
    print("  (positive R means the gate removed profit; negative R means it saved loss)")
    grouped: dict[tuple[str, str], list[Decision]] = {}
    for row in rows:
        grouped.setdefault((row.module, row.outcome), []).append(row)
    for (module, reason), rejected in sorted(grouped.items()):
        total = sum(row.managed_r or 0.0 for row in rejected)
        wins = sum(1 for row in rejected if (row.managed_r or 0.0) > 0.0)
        print(
            f"    {module:<28} {reason:<24} {len(rejected):>5} setups  "
            f"{wins / len(rejected):>5.1%} won  {total:>+9.2f} R"
        )


def _gates_this_run_does_not_apply(
    *,
    btc_research_parity: bool = False,
    btc_jarvis_replay: bool = False,
    jarvis_replay: bool = False,
) -> None:
    """What stands between this number and the account's behaviour."""
    blocked = sum(count for _name, count, _why in NOT_MODELLED)
    print(f"\n{'=' * 78}")
    print("WHAT THIS RUN DID NOT MODEL")
    print(f"{'=' * 78}")
    if btc_jarvis_replay:
        print("  This is the second S15-S17 account replay described in the contract above.")
        print("  Target reach, spread/stop, market quiet and volume spike WERE applied.")
        print("  Mechanical profit-lock, break-even, broker-feasible partial close,")
        print("  ATR trailing, hard giveback, peak-stall and time-exit WERE applied.")
        print("  No unrelated live-day counters are mixed into this replay.\n")
    elif jarvis_replay:
        print("  This is the account replay described in the contract above. Four of the")
        print("  eight gates WERE applied to every section, and so was the shared position")
        print("  book. What is left is what no bar archive contains:\n")
    elif btc_research_parity:
        print("  This S15-S17 run reproduces the frozen research detector, next-bar entry,")
        print("  horizon, spread envelope, 2% minimum-lot envelope and execution allowance.")
        print("  It deliberately does NOT apply these later live-runner gates:\n")
    else:
        print("  This script is ConfluenceEngine.evaluate + PositionSizer.size. The live")
        print("  runner wraps those in eight more gates. None of them are applied here:\n")
    omitted = NOT_MODELLED
    if btc_jarvis_replay:
        omitted = ()
    elif jarvis_replay:
        # ONLY THE ONES STILL MISSING. Leaving the whole table up under a run
        # that applies half of it says the opposite of what happened, and the
        # table is what gets read.
        omitted = tuple(row for row in NOT_MODELLED if row[0] not in JARVIS_REPLAY_APPLIES)
    for name, count, why in omitted:
        print(f"    {name:<24}{count:>5}   {why}")
    if btc_jarvis_replay:
        print("  Remaining limits are the missing tick/AI/intrabar state named above.")
    elif jarvis_replay:
        applied = ", ".join(sorted(JARVIS_REPLAY_APPLIES))
        print(f"\n  APPLIED IN THIS RUN, so absent from the list above: {applied}.")
        print("  The counts beside the remaining names are one real live day (31 August)")
        print("  and are there for scale, not as a prediction of this window.")
    else:
        print(
            f"\n  Those counts are one real day on the live account, 31 August: 414 setups\n"
            f"  formed, {blocked} died at the gates above, 11 at gates this script does have\n"
            f"  (SL_TOO_TIGHT_FOR_COSTS, RISK_EXCEEDS_CAP), and 0 trades were taken.\n"
            f"\n  So read every R and EUR above as WHAT THE STRATEGY DOES WITH THOSE EIGHT\n"
            f"  GATES OFF. It is the right number for choosing a clock or an exit rule,\n"
            f"  and it is not a forecast of the account. `waarom.cmd 24` says which gate\n"
            f"  is actually spending the setups on any given day."
        )

    if btc_jarvis_replay:
        print("\n  EXIT LIMITS STILL NOT RECONSTRUCTIBLE FROM OHLC:")
        print("    AI health/thesis actions, tick-persistent spread squeeze, exact fills")
        print("    and ambiguous within-M1 ordering. The replay uses the conservative")
        print("    HEALTHY branch and stop-first ordering; it does not invent AI answers.")
    else:
        print("\n  AND THE EXITS. Fixed-exit families are judged on their broker SL/TP")
        print("  and their configured pre-close flatten. For every OTHER family this")
        print("  replay walks the live TradeManagementConfig bar by bar and applies:\n")
        for name, what in EXITS_MODELLED_UNDER_FULL_MANAGEMENT:
            print(f"    {name:<30}{what}")
        print("\n  plus the break-even move. What it still cannot apply, because bars")
        print("  do not carry what these read:\n")
        for name, what in EXITS_NOT_MODELLED:
            print(f"    {name:<30}{what}")
        print(
            "\n  Within a bar the ordering of stop, target and every rule above is"
            "\n  unknowable, and this walk takes the stop first. So a managed number"
            "\n  here is close to the account and still not a broker statement."
        )
    print(f"{'=' * 78}")


def _sweep_report(results: dict, equity: float, days: int, managed: bool = False) -> None:
    """One row per (section, timeframe), on the exit the account actually runs.

    The shipped timeframes came from HistData bid bars. The only thing that
    could change the answer here is cost, and cost is exactly what this run
    measures for real -- so if a different clock wins by a margin, it wins
    because of this broker's spreads and not because of a preference.

    THIS TABLE READ THE FIXED STOP AND THAT WAS WRONG. I justified it as
    "comparing clocks is cleaner without the exit rule moving underneath it",
    which sounds reasonable and is not: it compares clocks under an exit the
    account does not take. The gap is not small either --

        order_block M30, 180 days:  fixed -34.00 R  /  break-even +33.60 R

    -- so the whole table read as a row of losses for a configuration that was
    making money. The owner spotted it: "deze klopt niet want dit was nog voor
    die break-even shit".

    Both columns are printed now. LIVE is the one to read; FIXED is kept
    beside it because the difference between them IS the value of the stop
    rule, per clock, which is worth seeing.
    """
    print(f"\n{'=' * 78}")
    print("TIMEFRAME SWEEP — each section on each clock, this broker, this window")
    print(f"{'=' * 78}")
    label = "break-even (LIVE)" if managed else "fixed stop"
    print(f"  ranked on the {label} exit\n")
    print(
        f"  {'section':<18}{'tf':>5}{'trades':>8}{'win':>7}"
        f"{'LIVE R':>9}{'LIVE EUR':>11}{'sigma':>8}{'fixed R':>10}"
    )
    rows = []
    for (name, tf), decisions in sorted(results.items()):
        trades = [d for d in decisions if d.outcome == "TRADE"]
        closed = [d for d in trades if _live_exit(d, managed) is not None]
        live_r = sum(_live_exit(d, managed) or 0.0 for d in closed)
        fixed_r = sum(d.result_r or 0.0 for d in closed)
        money = sum((d.managed_money if managed else d.pnl_money) or 0.0 for d in closed)
        wins = sum(1 for d in closed if (_live_exit(d, managed) or 0) > 0)
        win = f"{wins / len(closed):.0%}" if closed else "-"
        # SIGMA PER ROW, day-clustered. Without it "profitable" and "had a good
        # stretch" print identically, and this table exists to choose a clock.
        by_day: dict[object, float] = {}
        for d in closed:
            by_day[d.when.date()] = by_day.get(d.when.date(), 0.0) + (_live_exit(d, managed) or 0.0)
        daily = np.array(list(by_day.values()), dtype=float) if by_day else np.zeros(0)
        if len(daily) > 1 and daily.std(ddof=1) > 0:
            sigma = live_r / (float(daily.std(ddof=1)) * float(np.sqrt(len(daily))))
        else:
            sigma = 0.0
        thin = "  thin" if len(closed) < 200 else ("  <- clears 2" if sigma >= 2.0 else "")
        print(
            f"  {name:<18}{tf:>5}{len(trades):>8}{win:>7}"
            f"{live_r:>+9.2f}{money:>+11.2f}{sigma:>+8.2f}{fixed_r:>+10.2f}{thin}"
        )
        rows.append((money, name, tf, len(closed)))
    if any(tf == "M1" for _m, _n, tf, _c in rows):
        # THE ONE ROW WHOSE NUMBER IS NOT COMPARABLE TO THE OTHERS. Every
        # other clock is walked out on M1 bars, so a bar holding both barriers
        # is rare. An M1 trade is walked out on the bar it was born on, so
        # "this bar touched the stop and the target" is common -- and
        # `_resolve` books that as a full loss, because within one bar the
        # order is unknowable and guessing in your own favour is how a
        # backtest lies.
        #
        # So M1 is measured with a thumb on the losing side. A positive M1 row
        # is a real finding. A negative one is not evidence against M1; it is
        # the honest floor, and tick data is the only thing that would lift it.
        print(
            "\n  M1 IS RESOLVED ON ITS OWN BARS — nothing finer exists. A bar that\n"
            "  holds both the stop and the target is counted as a LOSS, so this row\n"
            "  is biased AGAINST M1. Positive here means something; negative here\n"
            "  may be the measurement rather than the strategy."
        )
    if rows:
        judgeable = [row for row in rows if row[3] >= 200]
        rows.sort(reverse=True)
        money, name, tf, n = rows[0]
        share = money / equity if equity else 0.0
        print(
            f"\n  best on this feed: {name} on {tf} — EUR {money:+.2f} "
            f"({share:+.1%} of equity) over {n} trades, {n / max(days, 1):.1f}/day"
        )
        if not judgeable:
            print(
                "  BUT NO ROW HAS 200 TRADES. Ranking clocks on samples this small is\n"
                "  reading noise: a seven-day sweep once put four trades in a row and\n"
                "  I drew conclusions from it. Lengthen the window before believing it."
            )
        else:
            print(
                "  A clock that beats the shipped one here is worth taking seriously:\n"
                "  the research could not price this broker's spread and this can."
            )
    print(f"{'=' * 78}")


def _report(
    decisions: list[Decision],
    equity: float,
    days: int,
    skipped: int,
    managed: bool = False,
    sections: tuple[str, ...] = (),
) -> None:
    """Every decision in the run, and BY SECTION at the bottom.

    THIS BLOCK READ THE FIXED-STOP COLUMN TOO, and BY SECTION is the line the
    owner actually asks about -- "is order_block positive and impulse_retest
    not". On the 180-day run it answered that question about a configuration
    the account does not trade. `_live_config_report` was corrected first;
    this is the same defect twenty lines lower and it had to be found
    separately, which is the argument for `_live_exit` existing at all.
    """
    trades = [d for d in decisions if d.outcome == "TRADE"]
    closed = [d for d in trades if _live_exit(d, managed) is not None]
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

    wins = [d for d in closed if (_live_exit(d, managed) or 0) > 0]
    r_total = sum(_live_exit(d, managed) or 0.0 for d in closed)
    money = sum((d.managed_money if managed else d.pnl_money) or 0.0 for d in closed)
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
        [
            {
                "day": d.when.date(),
                "r": _live_exit(d, managed),
                "eur": (d.managed_money if managed else d.pnl_money),
            }
            for d in closed
        ]
    )
    per_day = frame.groupby("day").agg(trades=("r", "size"), R=("r", "sum"), EUR=("eur", "sum"))
    print("\nBY DAY")
    for day, row in per_day.iterrows():
        bar = "+" * int(max(row.R, 0) * 3) + "-" * int(max(-row.R, 0) * 3)
        print(
            f"   {day}  {int(row.trades):>3} trades  {row.R:>+7.2f} R  EUR {row.EUR:>+7.2f}  {bar}"
        )
    print(f"\n   days green {int((per_day.R > 0).sum())} / {len(per_day)}")

    by_module = pd.DataFrame(
        [
            {
                "module": d.module,
                "r": _live_exit(d, managed),
                "eur": (d.managed_money if managed else d.pnl_money),
            }
            for d in closed
        ]
    )
    print("\nBY SECTION")
    totals = (
        by_module.groupby("module").agg(trades=("r", "size"), R=("r", "sum"), EUR=("eur", "sum"))
        if not by_module.empty
        else pd.DataFrame(columns=["trades", "R", "EUR"])
    )
    for module, row in totals.iterrows():
        print(
            f"   {module:<34} {int(row.trades):>4} trades  {row.R:>+7.2f} R  EUR {row.EUR:>+7.2f}"
        )

    # EVERY SECTION THAT WAS MEASURED, INCLUDING THE ONES THAT TOOK NOTHING.
    #
    # "Waar is sectie 10" -- section ten was on the live list, ran, took no
    # trades, and therefore had no rows to group, so it vanished from the only
    # table anyone reads. An absent row and a zero row look identical and mean
    # opposite things: one is a section that found nothing, the other is a
    # section that is not wired in. This project has now shipped that same
    # confusion under seven different names.
    silent = sorted(set(sections) - set(totals.index))
    for name in silent:
        rows = [d for d in decisions if d.pass_key[0] == name]
        if not rows:
            print(f"   {name:<34}    NO DECISIONS AT ALL -- it did not run")
            continue
        why = Counter(d.outcome for d in rows if d.outcome != "TRADE").most_common(1)
        reason = why[0] if why else ("nothing recorded", 0)
        print(
            f"   {name:<34}    0 trades   {len(rows)} decisions, "
            f"mostly {reason[0]} ({reason[1]})"
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

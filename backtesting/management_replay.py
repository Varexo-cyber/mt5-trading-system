"""Run the real management rules over real price history, bar by bar.

The existing backtester replays an entry to its stop or target and stops there.
That measures the *idea*. It says nothing about the layer the operator actually
asks about — give-back, peak stall, profit lock, the health reader — which
until recently was never executed on a live position at all, because a
scheduling bug starved the loop that runs it.

So those rules have been unit-tested against synthetic bars and never once
watched against a market. That is the gap between "well built" and "proven",
and no amount of further rule-writing closes it. This closes it for free: it
costs no API calls and no money, which is the only honest way to get more
confidence out of a €81 account than the account itself can pay for.

This drives `PositionManager` itself — the same object the live guard calls,
not a reimplementation. A reimplementation would prove nothing: it would test a
second copy of the logic, and the second copy is exactly where the divergence
would hide. Everything here is scaffolding around the real thing: a broker that
serves historical bars, a journal that holds the excursion ratchet, and a loop
that steps the clock.

What comes out is a timeline. Which rule fired, at which bar, at what R — and
what the trade would have returned against what it actually did.

WHAT THIS CANNOT TELL YOU. Bars are not ticks: inside one M1 bar the order of
the high and the low is unknown, so a stop and a target reached in the same
minute are resolved stop-first here, which is the pessimistic reading. There is
no spread series in bar history, so the spread-squeeze rule can never fire and
exits fill at the close. And a replay validates *mechanics* — that the rules do
what they are supposed to on real price — not profitability. A profitable
replay over four trades is still four trades.

    from backtesting.management_replay import replay_management
    outcome = replay_management(trade, frame, settings, spec)
    print(outcome.render())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from core.instrument import InstrumentSpec
from core.types import Direction, Position, Tick, Timeframe
from execution.manager import ManagementEvent, PositionManager

#: Reverse of `Timeframe.mt5_value`, so the broker double can tell which
#: timeframe a `copy_rates` call is asking for and serve the right bars.
_BY_MT5_VALUE = {frame.mt5_value: frame for frame in Timeframe}

#: The timeframe the replay steps in, and the one every coarser series is
#: built from. Everything the management layer reads (M1 shape, M5 structure,
#: H1 ATR) is a whole multiple of it.
BASE = Timeframe.M1


@dataclass(frozen=True, slots=True)
class ReplayTrade:
    """The trade as it was opened, before anything managed it."""

    symbol: str
    direction: Direction
    entry: float
    stop: float
    target: float
    volume: float
    opened_at: datetime
    #: What actually happened, when known, so the replay can be compared to it.
    actual_pnl_r: float | None = None
    actual_exit_reason: str = ""

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop)


@dataclass(frozen=True, slots=True)
class _Filled:
    """What the broker double hands back. Only the fields `manage` reads."""

    ok: bool
    filled_price: float
    filled_volume: float


@dataclass
class _ReplayBroker:
    """Just enough broker to drive `manage`, backed by history.

    Bars are served only up to the bar being replayed. Handing the manager the
    whole frame would let a rule read prices that had not happened yet, which
    is the one bug a replay must be structurally unable to have.

    Coarser timeframes are resampled from the base frame rather than fetched,
    and resampled from the *truncated* base — so the newest M5 bar at 12:03 is
    built from three M1 bars and is genuinely half-formed, exactly as MT5 would
    return it. Resampling the whole frame first and then filtering by label
    would quietly hand back a complete 12:00 to 12:05 bar at 12:03, which is
    lookahead wearing a convincing disguise.
    """

    spec_: InstrumentSpec
    frame: pd.DataFrame
    cursor: int = 0
    closed: list[tuple[int, float | None]] = field(default_factory=list)
    stops: list[float] = field(default_factory=list)

    def tick(self, symbol: str) -> Tick:
        row = self.frame.iloc[self.cursor]
        price = float(row["close"])
        # bid == ask: bar history carries no spread, and inventing one would
        # make the spread-squeeze rule fire on a number nobody measured.
        return Tick(symbol=symbol, bid=price, ask=price, time=self.now)

    @property
    def now(self) -> datetime:
        return self.frame.index[self.cursor].to_pydatetime()

    def spec(self, _symbol: str) -> InstrumentSpec:
        return self.spec_

    def copy_rates(self, _symbol: str, timeframe: int, count: int) -> list[dict[str, Any]]:
        window = self._series(timeframe, count)
        return [
            {
                "time": int(index.timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "tick_volume": int(row["tick_volume"]),
                "spread": 0,
                "real_volume": 0,
            }
            for index, row in window.iterrows()
        ]

    def _series(self, timeframe: int, count: int) -> pd.DataFrame:
        step = max(1, int(_BY_MT5_VALUE[timeframe].duration / BASE.duration))
        visible = self.frame.iloc[: self.cursor + 1]
        if step == 1:
            return visible.iloc[-count:]
        # Only the tail that can contribute to `count` coarse bars, so the cost
        # of a pass does not grow with the length of the history loaded.
        tail = visible.iloc[-(count + 1) * step :]
        coarse = tail.resample(f"{step}min", label="left", closed="left").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "tick_volume": "sum",
            }
        )
        return coarse.dropna().iloc[-count:]

    def close_position(self, position: Position, volume: float | None = None) -> _Filled:
        self.closed.append((position.ticket, volume))
        return _Filled(
            ok=True,
            filled_price=float(self.frame.iloc[self.cursor]["close"]),
            filled_volume=volume if volume is not None else position.volume,
        )

    def modify_stops(self, _position: Position, sl: float, tp: float) -> _Filled:
        del tp
        self.stops.append(sl)
        return _Filled(ok=True, filled_price=sl, filled_volume=0.0)


@dataclass
class _ReplayJournal:
    """Holds the excursion ratchet exactly as the real journal does.

    `sl` is the stop the trade was *opened* with and never moves. The manager
    reads it as the denominator of R, so letting it follow a trailing stop
    would shrink 1R every time the stop advanced and every threshold in the
    system would drift with it.
    """

    original_stop: float
    volume: float
    peak_r: float = 0.0
    trough_r: float = 0.0
    partial_taken: bool = False

    def open_trade_by_ticket(self, ticket: int) -> dict[str, Any]:
        return {
            "id": 1,
            "ticket": ticket,
            "sl": self.original_stop,
            "volume": self.volume,
            "mfe_r": self.peak_r,
        }

    def management_action_exists(self, _ticket: int, _actions: tuple[str, ...]) -> bool:
        return self.partial_taken

    def update_excursions(self, _trade_id: int, *, mae_r: float, mfe_r: float) -> None:
        self.peak_r = max(self.peak_r, mfe_r)
        self.trough_r = min(self.trough_r, mae_r)


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """One bar, and what a rule did on it."""

    at: datetime
    price: float
    r: float
    action: str
    detail: str


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    """What the management rules would have done to this trade."""

    trade: ReplayTrade
    steps: tuple[ReplayStep, ...]
    #: Volume-weighted, so a partial banked at 1.0R followed by a stop-out at
    #: break-even reads as the half-win it was rather than as a scratch.
    exit_r: float | None
    exit_reason: str
    peak_r: float
    trough_r: float
    bars: int

    @property
    def improvement_r(self) -> float | None:
        """How much better or worse than what actually happened."""
        if self.exit_r is None or self.trade.actual_pnl_r is None:
            return None
        return self.exit_r - self.trade.actual_pnl_r

    def render(self) -> str:
        lines = [
            "",
            "=" * 78,
            f"  {self.trade.symbol} {self.trade.direction.name} — management replay",
            "=" * 78,
            "",
            f"  entry {self.trade.entry:.5f}  stop {self.trade.stop:.5f}  "
            f"target {self.trade.target:.5f}",
            f"  replayed over {self.bars} bars from {self.trade.opened_at:%Y-%m-%d %H:%M}",
            "",
            f"  peak reached   {self.peak_r:+.2f}R",
            f"  worst reached  {self.trough_r:+.2f}R",
            "",
        ]
        if self.steps:
            lines.append("-" * 78)
            lines.append("  WHAT THE RULES DID")
            lines.append("-" * 78)
            lines.append("")
            for step in self.steps:
                lines.append(f"  {step.at:%H:%M}  {step.r:+6.2f}R  {step.action:<20} {step.detail}")
            lines.append("")
        else:
            lines.append("  No rule ever acted. The trade ran to its stop or target untouched,")
            lines.append("  which on a position that went into profit is the finding.")
            lines.append("")

        if self.exit_r is None:
            lines.append("  replay exit    still open when the bars ran out")
        else:
            lines.append(f"  replay exit    {self.exit_r:+.2f}R  [{self.exit_reason}]")
        if self.trade.actual_pnl_r is not None:
            lines.append(
                f"  actually got   {self.trade.actual_pnl_r:+.2f}R  "
                f"[{self.trade.actual_exit_reason or 'unknown'}]"
            )
            better = self.improvement_r
            if better is not None:
                verdict = "better" if better > 0 else "worse" if better < 0 else "the same"
                lines.append(
                    f"  difference     {better:+.2f}R — the rules would have done {verdict}"
                )
        lines.append("")
        return "\n".join(lines)


def _fill(row: pd.Series, level: float, *, downward: bool) -> float | None:
    """Where a resting level would have filled inside this bar, or None.

    `downward` is which way price has to travel to reach it — down to a long's
    stop or a short's target, up to the other two. A gap straight through the
    level fills at the open, not at the level: pretending otherwise is how a
    backtest quietly reports exits at prices that were never available.
    """
    if downward:
        if float(row["low"]) > level:
            return None
        return min(float(row["open"]), level)
    if float(row["high"]) < level:
        return None
    return max(float(row["open"]), level)


def replay_management(
    trade: ReplayTrade,
    frame: pd.DataFrame,
    settings: Any,
    spec: InstrumentSpec,
    *,
    max_bars: int = 2000,
) -> ReplayOutcome:
    """Step the real `PositionManager` over history and record what it does.

    The stop and target are checked against each bar's own high and low before
    the rules run, because the broker would have filled them first. A rule that
    "exits" after price already went through the stop is measuring a trade that
    no longer existed.
    """
    broker = _ReplayBroker(spec_=spec, frame=frame)
    journal = _ReplayJournal(original_stop=trade.stop, volume=trade.volume)
    manager = PositionManager(broker, journal, settings)  # type: ignore[arg-type]

    sign = int(trade.direction)
    risk = trade.risk
    steps: list[ReplayStep] = []
    stop = trade.stop
    volume = trade.volume
    realised_r = 0.0
    exit_r: float | None = None
    exit_reason = ""
    used = 0

    def r_at(price: float) -> float:
        return (price - trade.entry) * sign / risk

    for index in range(min(len(frame), max_bars)):
        broker.cursor = index
        used = index + 1
        row = frame.iloc[index]
        moment = frame.index[index].to_pydatetime()

        # The manager's ATR and bar caches expire on `time.monotonic`, which
        # barely moves while a replay burns through a day of bars in a second.
        # Left alone, every pass after the first would read the first bar's
        # view of the market forever. Cleared per step so each bar is a fresh
        # look, which is what a minute of wall clock buys you live.
        manager._atr_cache.clear()
        manager._bar_cache.clear()

        # The broker fills a stop or target inside the bar, before any rule of
        # ours gets a look at it. Stop first when both are touched: which came
        # first is unknowable from a bar, and the pessimistic reading is the
        # only one that cannot flatter the result.
        long = trade.direction is Direction.LONG
        stopped = _fill(row, stop, downward=long)
        if stopped is not None:
            realised_r += r_at(stopped) * (volume / trade.volume)
            exit_r, exit_reason = realised_r, "STOP"
            break
        reached = _fill(row, trade.target, downward=not long)
        if reached is not None:
            realised_r += r_at(reached) * (volume / trade.volume)
            exit_r, exit_reason = realised_r, "TARGET"
            break

        price = float(row["close"])
        r_now = r_at(price)
        moved = price - trade.entry
        position = Position(
            ticket=1,
            symbol=trade.symbol,
            direction=trade.direction,
            volume=volume,
            price_open=trade.entry,
            sl=stop,
            tp=trade.target,
            profit=spec.money_per_lot(moved) * volume * (1 if moved * sign > 0 else -1),
            swap=0.0,
            opened_at=trade.opened_at,
        )
        events: list[ManagementEvent] = manager.manage([position], moment)

        for event in events:
            steps.append(
                ReplayStep(
                    at=moment, price=price, r=r_now, action=event.action, detail=event.detail
                )
            )
            if event.action.startswith("PARTIAL_CLOSE"):
                banked = event.volume_closed or 0.0
                realised_r += r_at(event.exit_price or price) * (banked / trade.volume)
                volume = max(0.0, volume - banked)
                journal.volume = volume
                journal.partial_taken = True
                continue
            if event.exit_price is not None:
                realised_r += r_at(event.exit_price) * (volume / trade.volume)
                exit_r, exit_reason = realised_r, event.action
        if exit_r is not None:
            break
        if broker.stops:
            stop = broker.stops[-1]

    return ReplayOutcome(
        trade=trade,
        steps=tuple(steps),
        exit_r=exit_r,
        exit_reason=exit_reason or "never closed",
        peak_r=journal.peak_r,
        trough_r=journal.trough_r,
        bars=used,
    )


def frame_from_bars(bars: list[dict[str, Any]] | Any) -> pd.DataFrame:
    """Broker rate rows into the indexed frame the replay expects."""
    frame = pd.DataFrame(bars)
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame = frame.set_index("time").sort_index()
    if "tick_volume" not in frame.columns:
        frame["tick_volume"] = 1
    return frame


def bars_after(frame: pd.DataFrame, opened_at: datetime, hours: float = 24.0) -> pd.DataFrame:
    """The window a trade actually lived in."""
    start = opened_at if opened_at.tzinfo else opened_at.replace(tzinfo=UTC)
    return frame.loc[start : start + timedelta(hours=hours)]

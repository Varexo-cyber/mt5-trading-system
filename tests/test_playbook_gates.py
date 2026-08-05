"""The gates the short-horizon plays were walking straight past.

Both of the day's analysis fixes went into `ConfluenceEngine.evaluate`. A
promoted play never runs through `evaluate` — and because the swing engine
rejects most candidates, the plays are what actually reached the adviser. So
every fix applied to the path that was already quiet, and none of it to the
path that was doing the talking.

The adviser's own words, over and over: "the stop (1.7 pips) is far smaller
than even the M5 ATR (3.8 pips)", "2.4-pip stop (0.24 ATR H1)", "short entered
against the dominant higher-timeframe trend: D1 and W1 both show a clean,
sustained uptrend".
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd

from analysis.playbooks import FadeConfig, MomentumScalp, RangeFade, ScalpConfig
from core.types import MarketContext, Series, Tick, Timeframe, TradingMode

NOW = datetime(2026, 8, 5, 14, 0, tzinfo=UTC)


def series(bars: list[tuple[float, float, float, float]], timeframe: Timeframe) -> Series:
    frame = pd.DataFrame(
        {
            "open": [b[0] for b in bars],
            "high": [b[1] for b in bars],
            "low": [b[2] for b in bars],
            "close": [b[3] for b in bars],
            "tick_volume": [1] * len(bars),
        },
        index=pd.DatetimeIndex(
            [NOW - timedelta(minutes=5 * (len(bars) - i)) for i in range(len(bars))]
        ),
    )
    return Series(symbol="EURUSD", timeframe=timeframe, df=frame, fetched_at=NOW)


def impulse_context() -> MarketContext:
    """Quiet compression, a sharp impulse up, then a very shallow pullback.

    The impulse and the pullback share one 4-bar window, which is how the
    playbook reads them. The shallow pullback is the whole point: it puts the
    pivot within a hair of entry, and that is what produced the 1.7-pip stops.
    """
    bars: list[tuple[float, float, float, float]] = []
    price = 1.1000
    # A history that actually travels. The playbook will not target further than
    # this instrument is seen to move in `target_bars`, so a perfectly flat past
    # makes every target unreachable and no play is ever produced — the fixture
    # would then prove nothing at all.
    for i in range(58):
        step = 0.00030 if (i // 5) % 2 == 0 else -0.00022
        bars.append((price, price + 0.00035, price - 0.00035, price + step))
        price += step
    for _ in range(12):  # the compression the playbook requires immediately before
        bars.append((price, price + 0.00008, price - 0.00008, price))
    for _ in range(3):  # the impulse itself
        bars.append((price, price + 0.00060, price - 0.00005, price + 0.00055))
        price += 0.00055
    # Fourth bar of the window makes the high *and* holds it, retracing a few
    # ticks off the top. That is the shape that produced the 1.7-pip stops: the
    # pivot the stop anchors to is the low of this one bar, a hair below entry.
    bars.append((price, price + 0.00065, price + 0.00048, price + 0.00055))
    price += 0.00055
    minute = [(price, price + 0.00004, price - 0.00004, price) for _ in range(30)]
    return MarketContext(
        symbol="EURUSD",
        now=NOW,
        series={
            Timeframe.M5: series(bars, Timeframe.M5),
            Timeframe.M1: series(minute, Timeframe.M1),
        },
        tick=Tick(symbol="EURUSD", time=NOW, bid=price - 0.00001, ask=price + 0.00001),
    )


def scalp(**overrides) -> MomentumScalp:  # type: ignore[no-untyped-def]
    return MomentumScalp(ScalpConfig(**overrides))


def m5_atr(context: MarketContext) -> float:
    from analysis.playbooks import _atr

    return _atr(context.series[Timeframe.M5].df)


# ------------------------------------------------------------ stop floor ---


def test_the_scalp_stop_clears_one_bar_of_its_own_timeframe() -> None:
    """A stop inside the M5 ATR is not testing whether the pullback held; it is
    testing whether the next ordinary bar ticks through it."""
    context = impulse_context()
    play = scalp().propose(context, TradingMode.PAPER)
    assert play is not None
    assert play.risk >= m5_atr(context) * 1.0 - 1e-9


def test_without_the_floor_this_very_setup_produces_a_noise_stop() -> None:
    """The fixture is the real shape, not a contrived one: the impulse makes
    its high on the last bar and holds it, so the pivot the stop anchors to
    sits a hair below entry. Unfloored it comes out around half an ATR — the
    band the adviser rejected all day as "inside normal M5/M15 noise".
    """
    context = impulse_context()
    unfloored = scalp(min_stop_atr=0.0).propose(context, TradingMode.PAPER)
    floored = scalp().propose(context, TradingMode.PAPER)

    assert unfloored is not None and floored is not None
    assert unfloored.risk < m5_atr(context)
    assert floored.risk > unfloored.risk


def test_the_floor_does_not_quietly_move_the_entry_or_the_direction() -> None:
    """It widens the stop and nothing else. A floor that also shifted the entry
    would be inventing a different trade."""
    context = impulse_context()
    unfloored = scalp(min_stop_atr=0.0).propose(context, TradingMode.PAPER)
    floored = scalp().propose(context, TradingMode.PAPER)

    assert unfloored is not None and floored is not None
    assert floored.entry == unfloored.entry
    assert floored.direction is unfloored.direction


def test_the_floor_is_configured_at_one_atr() -> None:
    """Named rather than inferred: the ceiling existed and the floor did not,
    and the floor is the one that was needed."""
    assert ScalpConfig().min_stop_atr == 1.0
    assert FadeConfig().min_stop_atr == 1.0


def test_the_ceiling_still_applies_above_the_floor() -> None:
    """Widening to the floor must not be able to push a stop past the ceiling
    that keeps a scalp a scalp."""
    config = ScalpConfig()
    assert config.min_stop_atr < config.max_stop_atr


def test_a_floor_above_the_ceiling_yields_nothing_rather_than_a_swing_trade() -> None:
    """Misconfigured, it has to refuse rather than quietly emit a trade the
    swing engine should have owned."""
    play = scalp(min_stop_atr=3.0, max_stop_atr=2.5).propose(impulse_context(), TradingMode.PAPER)
    assert play is None


# ----------------------------------------------------------- range fade ---


def fade_context() -> MarketContext:
    """A range walked between its edges several times, then rejected at the top.

    Walked rather than jumped: a series that teleports from one edge to the
    other has an enormous true range, the ATR swamps the range itself, and
    `min_range_atr` refuses it before the stop is ever built.
    """
    bottom, top = 1.1000, 1.1080
    height = top - bottom
    bars: list[tuple[float, float, float, float]] = []
    previous = bottom
    for i in range(100):  # the playbook needs 80 M15 bars before it looks
        # Five full sweeps of the range across the window.
        phase = math.sin(i / 100 * 5 * 2 * math.pi)
        close = bottom + height * (0.5 + 0.5 * phase)
        bars.append((previous, max(previous, close) + 0.0002, min(previous, close) - 0.0002, close))
        previous = close
    # Final bar: a rejection wick through the top that closes back just inside.
    # Close *near* the edge on purpose — that is what leaves the stop, anchored
    # a buffer beyond the edge, sitting inside the noise that made the wick.
    bars.append((top - 0.0006, top + 0.0001, top - 0.0007, top - 0.0003))
    price = top - 0.0003
    return MarketContext(
        symbol="EURUSD",
        now=NOW,
        series={Timeframe.M15: series(bars, Timeframe.M15)},
        tick=Tick(symbol="EURUSD", time=NOW, bid=price - 0.00002, ask=price + 0.00002),
    )


def fade_atr(context: MarketContext) -> float:
    from analysis.playbooks import _atr

    return _atr(context.series[Timeframe.M15].df)


def test_the_fade_stop_clears_one_bar_too() -> None:
    context = fade_context()
    play = RangeFade(FadeConfig()).propose(context, TradingMode.PAPER)
    assert play is not None
    assert play.risk >= fade_atr(context) * 1.0 - 1e-9


def test_without_the_floor_the_fade_stop_sits_inside_the_wick() -> None:
    """A rejection that closes near the edge leaves the stop, anchored a buffer
    beyond that edge, inside the very move that produced the wick."""
    context = fade_context()
    unfloored = RangeFade(FadeConfig(min_stop_atr=0.0)).propose(context, TradingMode.PAPER)
    floored = RangeFade(FadeConfig()).propose(context, TradingMode.PAPER)

    assert unfloored is not None and floored is not None
    assert unfloored.risk < fade_atr(context)
    assert floored.risk > unfloored.risk


def test_widening_the_fade_stop_costs_reward_and_that_is_reported_honestly() -> None:
    """The floor is not free. A wider stop against the same midpoint target is
    a lower reward-to-risk, and it should read as lower rather than the target
    being quietly stretched to compensate."""
    context = fade_context()
    unfloored = RangeFade(FadeConfig(min_stop_atr=0.0)).propose(context, TradingMode.PAPER)
    floored = RangeFade(FadeConfig()).propose(context, TradingMode.PAPER)

    assert unfloored is not None and floored is not None
    assert floored.take_profit == unfloored.take_profit
    assert floored.reward_risk < unfloored.reward_risk

"""Ten detectors, chosen to be ten different bets rather than ten spellings.

The account this is for already owns nine detectors that read the same shape --
"price went up on some timeframe" -- and calls their agreement confluence. It
is not confluence. Nine readings of one fact are one reading, and the whole
scorecard behaves as if it were one, because it is.

So the list below is organised by WHO IS ON THE OTHER SIDE, not by which
indicator draws it. Two detectors belong in different sections only if a
market can make one of them right while the other is wrong:

    retest        the queue left at a level after the stops behind it are gone
    fade          a move that has spent the participants who caused it
    trend pull    a working order continuing across sessions
    squeeze       positioning compressed until it has to resolve
    session       a clock -- one participant group arriving, another leaving
    gap           an overnight repricing nobody could trade through
    false break   liquidity taken above a level with nothing behind it
    inside bar    a genuine pause, not a small candle
    range fade    an edge holding while the middle is noise
    hour drift    time of day alone, no price shape at all

`hour_drift` earns its place precisely because it reads no price pattern. If
it measures like the rest, the rest are measuring the clock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.lab.resolve import Batch

HORIZON_BARS = 96


def _channel(high: np.ndarray, low: np.ndarray, period: int):
    upper = pd.Series(high).shift(1).rolling(period).max().to_numpy()
    lower = pd.Series(low).shift(1).rolling(period).min().to_numpy()
    return upper, lower


def _empty() -> Batch:
    z = np.empty(0)
    return Batch(z.astype(int), z, z, z, False)


def _batch(rows: list[tuple[int, int, float, float]], same_bar: bool) -> Batch:
    if not rows:
        return _empty()
    arr = np.array(rows, dtype=float)
    return Batch(arr[:, 0].astype(np.int64), arr[:, 1], arr[:, 2], arr[:, 3], same_bar)


def _usable(a: np.ndarray, i: int) -> bool:
    return bool(np.isfinite(a[i]) and a[i] > 0)


# ---------------------------------------------------------------- retest ----


def level_retest(frame, a, *, period=20, tolerance=0.15, stop_beyond=0.75) -> Batch:
    """Break an N-bar channel, then buy the level it cleared.

    Counterparty: whoever is still bidding at the old edge once the stops
    behind it have been taken. Being wrong is one tick below them.
    """
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    upper, lower = _channel(high, low, period)
    rows: list[tuple[int, int, float, float]] = []
    end = len(close) - HORIZON_BARS - 1
    i = period + 15
    while i < end:
        if close[i] > upper[i]:
            direction, level = 1, upper[i]
        elif close[i] < lower[i]:
            direction, level = -1, lower[i]
        else:
            i += 1
            continue
        if not _usable(a, i):
            i += 1
            continue
        unit0 = a[i]
        for j in range(i + 1, min(i + HORIZON_BARS, end)):
            if direction > 0:
                failed = close[j] < level - stop_beyond * unit0
                touched = low[j] <= level + tolerance * unit0
            else:
                failed = close[j] > level + stop_beyond * unit0
                touched = high[j] >= level - tolerance * unit0
            # TOUCHED FIRST, AND THE ORDER IS THE WHOLE POINT. The entry sits
            # between the level and the stop, so price physically cannot reach
            # the stop without passing through the fill. When one violent bar
            # sweeps through both, testing `failed` first DROPS THE TRADE --
            # and those are exactly the worst losses, so discarding them
            # inflates the win rate of the one detector this study argues for.
            # A resting limit order would have been filled and then stopped.
            if touched:
                entry = level + direction * tolerance * unit0
                rows.append((j, direction, entry, (stop_beyond + tolerance) * unit0))
                i = j
                break
            if failed:
                break
        i += 1
    return _batch(rows, same_bar=True)


# ------------------------------------------------------------------ fade ----


def exhaustion_fade(frame, a, *, period=20, location=0.95, stop=1.0) -> Batch:
    """A bar that makes a new N-bar extreme and shuts on it. Fade it.

    Counterparty: the last buyer. A bar closing on its high after clearing a
    range has taken every resting offer and found no more.
    """
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    upper, lower = _channel(high, low, period)
    span = high - low
    with np.errstate(invalid="ignore", divide="ignore"):
        where = np.where(span > 0, (close - low) / span, 0.5)
    rows = []
    for i in range(period + 15, len(close) - HORIZON_BARS - 1):
        if not _usable(a, i):
            continue
        if close[i] > upper[i] and where[i] >= location:
            rows.append((i, -1, close[i], stop * a[i]))
        elif close[i] < lower[i] and (1.0 - where[i]) >= location:
            rows.append((i, 1, close[i], stop * a[i]))
    return _batch(rows, same_bar=False)


# ------------------------------------------------------------ trend pull ----


def trend_pullback(frame, a, *, fast=20, slow=50, depth=0.5, stop=1.0) -> Batch:
    """An established trend, entered on a dip to its fast average.

    Counterparty: nobody in particular -- this is the one bet that needs the
    market to keep doing what it was doing, which is why it is here as a
    control on the others rather than as a favourite.
    """
    close = frame["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean().to_numpy()
    ema_slow = close.ewm(span=slow, adjust=False).mean().to_numpy()
    values = close.to_numpy()
    rows = []
    for i in range(slow + 20, len(values) - HORIZON_BARS - 1):
        if not _usable(a, i):
            continue
        unit = a[i]
        up = ema_fast[i] > ema_slow[i] and ema_fast[i] > ema_fast[i - 5]
        down = ema_fast[i] < ema_slow[i] and ema_fast[i] < ema_fast[i - 5]
        if up and 0 <= (ema_fast[i] - values[i]) / unit <= depth:
            rows.append((i, 1, values[i], stop * unit))
        elif down and 0 <= (values[i] - ema_fast[i]) / unit <= depth:
            rows.append((i, -1, values[i], stop * unit))
    return _batch(rows, same_bar=False)


# --------------------------------------------------------------- squeeze ----


def squeeze_release(frame, a, *, period=20, quantile=0.25, stop=1.0) -> Batch:
    """Range compresses into the lowest quarter of its own year, then breaks.

    Counterparty: option desks and anyone short volatility. Compression is a
    positioning fact, not a price fact, which is what makes it a separate bet
    from the channel break it superficially resembles.
    """
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    upper, lower = _channel(high, low, period)
    width = pd.Series(upper - lower)
    floor = width.rolling(500, min_periods=200).quantile(quantile).to_numpy()
    rows = []
    for i in range(period + 500, len(close) - HORIZON_BARS - 1):
        if not _usable(a, i) or not np.isfinite(floor[i]):
            continue
        if not (upper[i] - lower[i]) <= floor[i]:
            continue
        if close[i] > upper[i]:
            rows.append((i, 1, close[i], stop * a[i]))
        elif close[i] < lower[i]:
            rows.append((i, -1, close[i], stop * a[i]))
    return _batch(rows, same_bar=False)


# --------------------------------------------------------------- session ----


def opening_range(frame, a, *, open_hour=13, bars=4, stop=1.0) -> Batch:
    """The first N bars after a session opens, then a break of that range.

    Counterparty: the clock. Different participants are present before and
    after an open, and this is the only detector here whose trigger is a time
    rather than a shape.
    """
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    hours = frame.index.hour.to_numpy()
    minutes = frame.index.minute.to_numpy()
    rows = []
    end = len(close) - HORIZON_BARS - 1
    starts = np.flatnonzero((hours == open_hour) & (minutes == 0))
    for start in starts:
        top = bottom = None
        if start + bars >= end:
            continue
        window = slice(start, start + bars)
        top, bottom = high[window].max(), low[window].min()
        for i in range(start + bars, min(start + bars + 12, end)):
            if not _usable(a, i):
                continue
            if close[i] > top:
                rows.append((i, 1, close[i], stop * a[i]))
                break
            if close[i] < bottom:
                rows.append((i, -1, close[i], stop * a[i]))
                break
    return _batch(rows, same_bar=False)


# ------------------------------------------------------------------- gap ----


def gap_fill(frame, a, *, minimum=0.5, stop=1.0) -> Batch:
    """A session opens away from the last close. Trade back toward it.

    Counterparty: whoever had to reprice without being able to trade. A gap is
    an adjustment made in the absence of a market, and part of it is usually
    the absence rather than the news.
    """
    close = frame["close"].to_numpy()
    opens = frame["open"].to_numpy()
    stamps = frame.index
    new_session = np.zeros(len(close), dtype=bool)
    new_session[1:] = (stamps[1:].to_numpy() - stamps[:-1].to_numpy()) > np.timedelta64(2, "h")
    rows = []
    for i in np.flatnonzero(new_session):
        if i < 30 or i >= len(close) - HORIZON_BARS - 1 or not _usable(a, i):
            continue
        gap = (opens[i] - close[i - 1]) / a[i]
        if abs(gap) < minimum:
            continue
        direction = -1 if gap > 0 else 1
        rows.append((i, direction, opens[i], stop * a[i]))
    # SAME BAR, and this is not a detail. The entry is the bar's OPEN, so the
    # whole of that bar happens after the fill -- and on a weekend gap the
    # median bar retraces 56% of the gap inside itself. Resolving from the next
    # bar skips the one where most of the trade happens, in both directions,
    # and reads +1.011R where the honest number is far lower.
    return _batch(rows, same_bar=True)


# ----------------------------------------------------------- false break ----


def false_break(frame, a, *, period=20, stop=1.0) -> Batch:
    """Price trades through a level intrabar and closes back inside.

    Counterparty: everyone whose stop was just filled. This is the opposite
    trade to `level_retest` on the same level, and the two cannot both fire on
    one bar -- one needs the close beyond, the other needs it back inside.
    """
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    upper, lower = _channel(high, low, period)
    rows = []
    for i in range(period + 15, len(close) - HORIZON_BARS - 1):
        if not _usable(a, i):
            continue
        if high[i] > upper[i] and close[i] < upper[i]:
            rows.append((i, -1, close[i], stop * a[i]))
        elif low[i] < lower[i] and close[i] > lower[i]:
            rows.append((i, 1, close[i], stop * a[i]))
    return _batch(rows, same_bar=False)


# ------------------------------------------------------------ inside bar ----


def inside_bar_break(frame, a, *, stop=1.0) -> Batch:
    """A bar wholly inside its predecessor, then a break of the mother bar.

    Counterparty: a genuine pause. Compression at a single-bar scale rather
    than the twenty-bar scale `squeeze_release` reads.
    """
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    rows = []
    for i in range(20, len(close) - HORIZON_BARS - 1):
        if not _usable(a, i):
            continue
        inside = high[i - 1] < high[i - 2] and low[i - 1] > low[i - 2]
        if not inside:
            continue
        if close[i] > high[i - 2]:
            rows.append((i, 1, close[i], stop * a[i]))
        elif close[i] < low[i - 2]:
            rows.append((i, -1, close[i], stop * a[i]))
    return _batch(rows, same_bar=False)


# ------------------------------------------------------------ range fade ----


def range_fade(frame, a, *, period=40, edge=0.1, stop=1.0) -> Batch:
    """Inside an established range, buy the floor and sell the ceiling.

    Counterparty: the market maker who wants the range to hold. Distinct from
    `exhaustion_fade`, which needs a break first; this one needs there to have
    been NO break.
    """
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    upper, lower = _channel(high, low, period)
    rows = []
    for i in range(period + 15, len(close) - HORIZON_BARS - 1):
        if not _usable(a, i):
            continue
        width = upper[i] - lower[i]
        if width <= 0 or close[i] > upper[i] or close[i] < lower[i]:
            continue
        position = (close[i] - lower[i]) / width
        if position <= edge:
            rows.append((i, 1, close[i], stop * a[i]))
        elif position >= 1.0 - edge:
            rows.append((i, -1, close[i], stop * a[i]))
    return _batch(rows, same_bar=False)


# ------------------------------------------------------------ hour drift ----


def hour_drift(frame, a, *, hour=8, stop=1.0) -> Batch:
    """Long at a fixed hour. No price shape at all.

    THE CONTROL. If this measures like the others then the others are reading
    the clock, and a detector that beats it by nothing is not a detector.
    """
    close = frame["close"].to_numpy()
    hours = frame.index.hour.to_numpy()
    minutes = frame.index.minute.to_numpy()
    rows = []
    for i in np.flatnonzero((hours == hour) & (minutes == 0)):
        if i < 30 or i >= len(close) - HORIZON_BARS - 1 or not _usable(a, i):
            continue
        rows.append((i, 1, close[i], stop * a[i]))
    return _batch(rows, same_bar=False)


CATALOGUE = {
    "level_retest": level_retest,
    "exhaustion_fade": exhaustion_fade,
    "trend_pullback": trend_pullback,
    "squeeze_release": squeeze_release,
    "opening_range": opening_range,
    "gap_fill": gap_fill,
    "false_break": false_break,
    "inside_bar_break": inside_bar_break,
    "range_fade": range_fade,
    "hour_drift": hour_drift,
}


# --------------------------------------------------------------------------
# STOP-WIDTH VARIANTS.
#
# Cost as a share of R is spread/R, so every rejection labelled "real but
# eaten by cost" is a claim about the STOP, not about the detector. range_fade
# on M1 indices measured +0.183R gross over 963,839 trades and lost to a 0.160
# cost built from a 1.0 ATR stop; at 2.0 ATR that cost halves.
#
# What is NOT known is whether the gross edge survives the wider stop. A
# detector whose edge is "price bounces soon" can be entirely destroyed by
# giving it more room, because the extra room is only ever used by the trades
# that were going to lose. That is the question these variants ask.
# --------------------------------------------------------------------------


def _widen(fn, stop):
    def variant(frame, a):
        return fn(frame, a, stop=stop)

    variant.__name__ = f"{fn.__name__}_s{stop}"
    return variant


for _base in (exhaustion_fade, trend_pullback, squeeze_release, false_break, range_fade):
    for _stop in (2.0, 3.0):
        CATALOGUE[f"{_base.__name__}_s{_stop:g}"] = _widen(_base, _stop)

# The retest is the one that already works, so it gets the opposite question:
# does a TIGHTER stop, which measured better before the floor argument, hold up
# here too?
for _tol, _beyond in ((0.15, 0.35), (0.15, 1.25), (0.30, 0.75)):

    def _retest_variant(frame, a, _t=_tol, _b=_beyond):
        return level_retest(frame, a, tolerance=_t, stop_beyond=_b)

    CATALOGUE[f"level_retest_t{_tol:g}_b{_beyond:g}"] = _retest_variant

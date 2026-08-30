"""Round three: mechanisms that are not "break a channel, buy the level".

Section two is the retest, and every survivor of the first two rounds turned
out to be a variant of it -- `retest_slow` shares 47.5% of its trades with
`retest_big_impulse`, which makes it the same section wearing a hat, not a
second one.

So this round is constrained: a detector belongs here only if it can fire when
the retest cannot, and be wrong when the retest is right. That rules out every
channel-and-level idea and leaves five families:

    swings        fractal pivots, not rolling extremes -- a swing low is where
                  the market actually turned, which is where stops actually sit
    zones         the candle before an impulse, revisited: an area, not a line
    proportion    Fibonacci retracement of a measured leg. Folklore, and cheap
                  to settle rather than argue about
    calendar      week and day boundaries rather than a rolling window
    cross-market  one instrument against its peers -- the only family here that
                  cannot be computed from a single chart, and therefore the only
                  one that is orthogonal by construction

Same rules as always: barrier-resolved only, same-bar counted as a loss, limit
fills resolved from their own bar, fill checked before failure, sigma clustered
by day, Bonferroni over the whole grid, and everything measured against the
coin-flip control rather than against theory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.lab.resolve import Batch
from scripts.lab.strategies import HORIZON_BARS, _batch
from scripts.lab.zoo import _ema, _emit, _window

ZOO3: dict = {}


def detector(name):
    def wrap(fn):
        ZOO3[name] = fn
        return fn

    return wrap


def _swings(high: np.ndarray, low: np.ndarray, span: int = 3):
    """Confirmed fractal pivots.

    A pivot is only knowable `span` bars after it prints, and both indices are
    kept apart so nothing can act on a swing before it existed. That
    distinction is the difference between a backtest and a fiction.
    """
    n = len(high)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    for i in range(span, n - span):
        window_h = high[i - span : i + span + 1]
        window_l = low[i - span : i + span + 1]
        if high[i] == window_h.max() and (window_h.argmax() == span):
            is_high[i] = True
        if low[i] == window_l.min() and (window_l.argmin() == span):
            is_low[i] = True
    return is_high, is_low


def _last_swing_prices(high, low, span=3):
    """For every bar, the most recent CONFIRMED swing high and low."""
    is_high, is_low = _swings(high, low, span)
    n = len(high)
    last_h = np.full(n, np.nan)
    last_l = np.full(n, np.nan)
    h = ll = np.nan
    for i in range(n):
        # A pivot at i-span became knowable now.
        j = i - span
        if j >= 0:
            if is_high[j]:
                h = high[j]
            if is_low[j]:
                ll = low[j]
        last_h[i] = h
        last_l[i] = ll
    return last_h, last_l


# ------------------------------------------------------------------ swings --


@detector("liquidity_sweep")
def liquidity_sweep(frame, a, *, span=3, stop=1.0):
    """Wick through the last swing low, close back above it.

    NOT `false_break`, which uses a rolling channel. A rolling low is wherever
    the window happens to start; a confirmed swing low is where the market
    actually turned, and that is where the stops of everyone who bought the
    turn are resting. The trade is that those stops get taken and nothing else
    was behind them.
    """
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    sh, sl = _last_swing_prices(high, low, span)
    keep = _window(len(close), span + 40) & np.isfinite(sh) & np.isfinite(sl)
    swept_low = (low < sl) & (close > sl)
    swept_high = (high > sh) & (close < sh)
    return _emit(keep & swept_low, keep & swept_high, close, a, stop)


@detector("swing_break_retest")
def swing_break_retest(frame, a, *, span=3, tolerance=0.15, stop_beyond=0.85):
    """A swing level broken and revisited. The retest, but on swings.

    Included precisely so the answer is known rather than assumed: if this
    matches section two it is the same trade on a different definition of
    level, and it does not get a section.
    """
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    sh, sl = _last_swing_prices(high, low, span)
    rows = []
    end = len(close) - HORIZON_BARS - 1
    i = span + 40
    while i < end:
        unit0 = a[i]
        if not (np.isfinite(unit0) and unit0 > 0):
            i += 1
            continue
        if np.isfinite(sh[i]) and close[i] > sh[i]:
            direction, level = 1, float(sh[i])
        elif np.isfinite(sl[i]) and close[i] < sl[i]:
            direction, level = -1, float(sl[i])
        else:
            i += 1
            continue
        for j in range(i + 1, min(i + HORIZON_BARS, end)):
            touched = (
                low[j] <= level + tolerance * unit0
                if direction > 0
                else high[j] >= level - tolerance * unit0
            )
            failed = (
                close[j] < level - stop_beyond * unit0
                if direction > 0
                else close[j] > level + stop_beyond * unit0
            )
            if touched:
                rows.append(
                    (
                        j,
                        direction,
                        level + direction * tolerance * unit0,
                        (stop_beyond + tolerance) * unit0,
                    )
                )
                i = j
                break
            if failed:
                break
        i += 1
    return _batch(rows, same_bar=True)


# ------------------------------------------------------------------- zones --


@detector("order_block")
def order_block(frame, a, *, impulse=1.5, tolerance=0.5, stop=1.0):
    """The last opposite-colour candle before an impulse, revisited.

    An AREA rather than a line: the body of the candle that was absorbed. The
    claim is that whoever had size left there did not get it all done, so the
    zone is defended when price comes back.
    """
    o, h, low, c = (frame[x].to_numpy() for x in ("open", "high", "low", "close"))
    rows = []
    end = len(c) - HORIZON_BARS - 1
    for i in range(50, end):
        unit = a[i]
        if not (np.isfinite(unit) and unit > 0):
            continue
        move = (c[i] - o[i]) / unit
        if abs(move) < impulse:
            continue
        direction = 1 if move > 0 else -1
        # The last candle the other way, immediately before the impulse.
        block = None
        for k in range(i - 1, max(i - 6, 0), -1):
            if (c[k] < o[k]) == (direction > 0):
                block = (min(o[k], c[k]), max(o[k], c[k]))
                break
        if block is None:
            continue
        edge = block[1] if direction > 0 else block[0]
        for j in range(i + 1, min(i + HORIZON_BARS, end)):
            touched = (
                low[j] <= edge + tolerance * unit
                if direction > 0
                else h[j] >= edge - tolerance * unit
            )
            if touched:
                rows.append((j, direction, edge + direction * tolerance * unit, stop * unit))
                break
    return _batch(rows, same_bar=True)


# -------------------------------------------------------------- proportion --


def _fib(level):
    def fn(frame, a, *, span=3, stop=1.0, tol=0.1):
        """Retracement of the last measured swing leg to a fixed proportion."""
        high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
        sh, sl = _last_swing_prices(high, low, span)
        keep = _window(len(close), span + 60) & np.isfinite(sh) & np.isfinite(sl)
        leg = sh - sl
        with np.errstate(invalid="ignore", divide="ignore"):
            # Retracement measured from the swing high downward.
            want_long = sl + leg * (1.0 - level)
            want_short = sh - leg * (1.0 - level)
            near_long = np.abs(close - want_long) / np.where(a > 0, a, np.nan) < tol
            near_short = np.abs(close - want_short) / np.where(a > 0, a, np.nan) < tol
        rising = close > _ema(close, 50)
        return _emit(keep & near_long & rising, keep & near_short & ~rising, close, a, stop)

    return fn


for _lvl in (0.382, 0.5, 0.618):
    ZOO3[f"fib_{int(_lvl * 1000)}"] = _fib(_lvl)


# ---------------------------------------------------------------- calendar --


@detector("prior_week_break")
def prior_week_break(frame, a, *, stop=1.0):
    close = frame["close"].to_numpy()
    week = frame.index.to_period("W")
    grouped = frame.groupby(week)
    hi = grouped["high"].max().shift(1).reindex(week).to_numpy()
    lo = grouped["low"].min().shift(1).reindex(week).to_numpy()
    keep = _window(len(close), 60) & np.isfinite(hi) & np.isfinite(lo)
    return _emit(keep & (close > hi), keep & (close < lo), close, a, stop)


@detector("prior_week_fade")
def prior_week_fade(frame, a, *, stop=1.0):
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    week = frame.index.to_period("W")
    grouped = frame.groupby(week)
    hi = grouped["high"].max().shift(1).reindex(week).to_numpy()
    lo = grouped["low"].min().shift(1).reindex(week).to_numpy()
    keep = _window(len(close), 60) & np.isfinite(hi) & np.isfinite(lo)
    return _emit(
        keep & (low < lo) & (close > lo), keep & (high > hi) & (close < hi), close, a, stop
    )


@detector("trend_day_continuation")
def trend_day_continuation(frame, a, *, stop=1.0):
    """Yesterday closed in the top or bottom tenth of its range. Follow it."""
    _high, _low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    day = frame.index.normalize()
    grouped = frame.groupby(day)
    hi = grouped["high"].max().shift(1).reindex(day).to_numpy()
    lo = grouped["low"].min().shift(1).reindex(day).to_numpy()
    cl = grouped["close"].last().shift(1).reindex(day).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        where = (cl - lo) / (hi - lo)
    first_bars = frame.index.hour.to_numpy() < 2
    keep = _window(len(close), 60) & np.isfinite(where) & first_bars
    return _emit(keep & (where > 0.9), keep & (where < 0.1), close, a, stop)


# ------------------------------------------------------------ session vwap --


@detector("session_vwap_reversion")
def session_vwap_reversion(frame, a, *, sigma=2.0, stop=1.0):
    """Distance from the volume-weighted average price since the day opened."""
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    volume = frame["volume"].to_numpy().astype(float)
    typical = (high + low + close) / 3.0
    day = frame.index.normalize()
    weight = pd.Series(typical * volume).groupby(day).cumsum().to_numpy()
    total = pd.Series(volume).groupby(day).cumsum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(total > 0, weight / total, np.nan)
        away = (close - vwap) / np.where(a > 0, a, np.nan)
    keep = _window(len(close), 60) & np.isfinite(away)
    return _emit(keep & (away < -sigma), keep & (away > sigma), close, a, stop)


# ------------------------------------------------------------ cross-market --


def peer_divergence(frames: dict, symbol: str, a, *, bars=12, gap=1.0, stop=1.0) -> Batch:
    """One instrument lagging its peers. The only family here needing two charts.

    Orthogonal by construction: it cannot be computed from the chart the retest
    reads, so it can be right on a day the retest is wrong. The trade pays when
    the GAP closes, not when the market goes a particular way.
    """
    frame = frames[symbol]
    close = frame["close"].to_numpy()
    own = pd.Series(close).pct_change(bars).to_numpy()
    peers = []
    for other, oframe in frames.items():
        if other == symbol:
            continue
        aligned = oframe["close"].reindex(frame.index).ffill()
        peers.append(aligned.pct_change(bars).to_numpy())
    if not peers:
        return _batch([], same_bar=False)
    basket = np.nanmean(np.vstack(peers), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        # The gap, expressed in ATR of the laggard.
        spread_atr = (basket - own) * close / np.where(a > 0, a, np.nan)
    keep = _window(len(close), 60) & np.isfinite(spread_atr)
    return _emit(keep & (spread_atr > gap), keep & (spread_atr < -gap), close, a, stop)


ALL3 = dict(ZOO3)

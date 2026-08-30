"""Round two: make the winner better, and try the mechanisms round one missed.

Two kinds of thing here.

FILTERS ON THE RETEST. It is the only detector that has survived anything, so
an extra tenth of an R on it is worth more than a third marginal strategy. Each
filter asks a different question about WHICH retests are the good ones -- was
the break convincing, was the market trending, what time was it, had the level
been tested before.

MECHANISMS ROUND ONE DID NOT HAVE. Failed-auction reversals, first-hour range
projections, consecutive-close streaks, opening-drive continuation, midpoint
reversion, and the one honest control: a coin flip with the same trade count,
which must come out at zero or the harness is broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.lab.strategies import HORIZON_BARS, _batch, _channel
from scripts.lab.zoo import ZOO, _ema, _emit, _rsi, _window

ZOO2: dict = {}


def detector(name):
    def wrap(fn):
        ZOO2[name] = fn
        return fn

    return wrap


# ---------------------------------------------------------------------------
# the retest, with one filter at a time
# ---------------------------------------------------------------------------


def _retest_rows(frame, a, period=20, tolerance=0.15, stop_beyond=0.35):
    """Every retest, plus the facts a filter might want about its break."""
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    volume = frame["volume"].to_numpy().astype(float)
    upper, lower = _channel(high, low, period)
    vbase = pd.Series(volume).rolling(30).mean().to_numpy()
    ema200 = _ema(close, 200)
    hours = frame.index.hour.to_numpy()
    out = []
    end = len(close) - HORIZON_BARS - 1
    i = max(period + 15, 220)
    while i < end:
        if close[i] > upper[i]:
            direction, level = 1, upper[i]
        elif close[i] < lower[i]:
            direction, level = -1, lower[i]
        else:
            i += 1
            continue
        unit0 = a[i]
        if not np.isfinite(unit0) or unit0 <= 0:
            i += 1
            continue
        for j in range(i + 1, min(i + HORIZON_BARS, end)):
            if direction > 0:
                failed = close[j] < level - stop_beyond * unit0
                touched = low[j] <= level + tolerance * unit0
            else:
                failed = close[j] > level + stop_beyond * unit0
                touched = high[j] >= level - tolerance * unit0
            # Touched before failed -- see the note in `strategies.py`.
            # Checking the failure first drops every trade where one bar swept
            # through the fill and the stop together, which are the worst
            # losses this detector has.
            if touched:
                out.append(
                    {
                        "j": j,
                        "direction": direction,
                        "entry": level + direction * tolerance * unit0,
                        "unit": (stop_beyond + tolerance) * unit0,
                        # how far past the level the break bar CLOSED
                        "impulse": direction * (close[i] - level) / unit0,
                        # the break bar's own size
                        "bar": (high[i] - low[i]) / unit0,
                        # volume on the break, against its own baseline
                        "vol": volume[i] / vbase[i] if vbase[i] > 0 else 1.0,
                        # was the break with the 200-bar trend
                        "with_trend": (close[i] > ema200[i]) == (direction > 0),
                        # how long the retest took
                        "wait": j - i,
                        "hour": int(hours[j]),
                    }
                )
                i = j
                break
            if failed:
                break
        i += 1
    return out


def _filtered_retest(keep, **kwargs):
    def fn(frame, a):
        rows = [r for r in _retest_rows(frame, a, **kwargs) if keep(r)]
        return _batch(
            [(r["j"], r["direction"], r["entry"], r["unit"]) for r in rows], same_bar=True
        )

    return fn


ZOO2["retest_with_trend"] = _filtered_retest(lambda r: r["with_trend"])
ZOO2["retest_against_trend"] = _filtered_retest(lambda r: not r["with_trend"])
ZOO2["retest_small_impulse"] = _filtered_retest(lambda r: r["impulse"] < 0.5)
ZOO2["retest_big_impulse"] = _filtered_retest(lambda r: r["impulse"] >= 1.0)
ZOO2["retest_high_volume"] = _filtered_retest(lambda r: r["vol"] >= 1.5)
ZOO2["retest_low_volume"] = _filtered_retest(lambda r: r["vol"] < 1.0)
ZOO2["retest_fast"] = _filtered_retest(lambda r: r["wait"] <= 6)
ZOO2["retest_slow"] = _filtered_retest(lambda r: r["wait"] > 20)
ZOO2["retest_small_bar"] = _filtered_retest(lambda r: r["bar"] < 1.5)
ZOO2["retest_big_bar"] = _filtered_retest(lambda r: r["bar"] >= 2.5)
ZOO2["retest_london"] = _filtered_retest(lambda r: 7 <= r["hour"] < 12)
ZOO2["retest_newyork"] = _filtered_retest(lambda r: 13 <= r["hour"] < 18)
ZOO2["retest_asia"] = _filtered_retest(lambda r: r["hour"] < 7 or r["hour"] >= 21)
ZOO2["retest_long_only"] = _filtered_retest(lambda r: r["direction"] > 0)
ZOO2["retest_short_only"] = _filtered_retest(lambda r: r["direction"] < 0)


# ---------------------------------------------------------------------------
# mechanisms round one did not have
# ---------------------------------------------------------------------------


@detector("failed_auction")
def failed_auction(frame, a, *, period=20, stop=1.0):
    """Break a level, then close back through it within three bars.

    Not the same as `false_break`, which needs the rejection inside one bar.
    This one lets the break stand for a while and then fail, which is the
    version that traps people who waited for confirmation.
    """
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    upper, lower = _channel(high, low, period)
    n = len(close)
    keep = _window(n, period + 20)
    broke_up = np.zeros(n, dtype=bool)
    broke_down = np.zeros(n, dtype=bool)
    for back in (1, 2, 3):
        broke_up |= np.roll(close > upper, back)
        broke_down |= np.roll(close < lower, back)
    return _emit(
        keep & broke_down & (close > lower), keep & broke_up & (close < upper), close, a, stop
    )


@detector("first_hour_projection")
def first_hour_projection(frame, a, *, stop=1.0):
    """The day's first hour sets a range; trade a break of one times it."""
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    day = frame.index.normalize()
    hour = frame.index.hour.to_numpy()
    n = len(close)
    first = pd.Series(hour < 1, index=frame.index)
    top = high[first.to_numpy()]
    if top.size == 0:
        return _batch([], same_bar=False)
    grouped = pd.DataFrame({"h": high, "l": low}, index=frame.index)[first.to_numpy()]
    hi = grouped.groupby(grouped.index.normalize())["h"].max().reindex(day).to_numpy()
    lo = grouped.groupby(grouped.index.normalize())["l"].min().reindex(day).to_numpy()
    keep = _window(n, 40) & np.isfinite(hi) & np.isfinite(lo) & (hour >= 1)
    width = hi - lo
    return _emit(keep & (close > hi + width), keep & (close < lo - width), close, a, stop)


@detector("streak_reversal_5")
def streak_reversal_5(frame, a, *, stop=1.0):
    """Five consecutive closes the same way. Fade the sixth."""
    close = frame["close"].to_numpy()
    up = np.ones(len(close), dtype=bool)
    down = np.ones(len(close), dtype=bool)
    for k in range(5):
        up &= np.roll(close, k) > np.roll(close, k + 1)
        down &= np.roll(close, k) < np.roll(close, k + 1)
    keep = _window(len(close), 40)
    return _emit(keep & down, keep & up, close, a, stop)


@detector("midpoint_reversion")
def midpoint_reversion(frame, a, *, period=50, stop=1.0):
    """Reversion to the MIDPOINT of the range rather than to a moving average.

    A mean is dragged by the trend; the midpoint of the high-low range is not,
    so the two disagree exactly when a market is trending, which is when it
    matters.
    """
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    top = pd.Series(high).rolling(period).max().to_numpy()
    bottom = pd.Series(low).rolling(period).min().to_numpy()
    mid = (top + bottom) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        away = np.where(a > 0, (close - mid) / a, 0.0)
    keep = _window(len(close), period + 20)
    return _emit(keep & (away < -2.0), keep & (away > 2.0), close, a, stop)


@detector("opening_drive")
def opening_drive(frame, a, *, stop=1.0):
    """The first bar of a session closes strongly. Follow it."""
    _o, h, lo, c = (frame[x].to_numpy() for x in ("open", "high", "low", "close"))
    stamps = frame.index
    fresh = np.zeros(len(c), dtype=bool)
    fresh[1:] = (stamps[1:].to_numpy() - stamps[:-1].to_numpy()) > np.timedelta64(2, "h")
    span = h - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        where = np.where(span > 0, (c - lo) / span, 0.5)
    keep = _window(len(c), 40) & fresh
    return _emit(keep & (where > 0.8), keep & (where < 0.2), c, a, stop)


@detector("inside_day_squeeze")
def inside_day_squeeze(frame, a, *, stop=1.0):
    """Three successively smaller bars, then a break of the largest."""
    h, lo, c = (frame[x].to_numpy() for x in ("high", "low", "close"))
    span = h - lo
    shrinking = (np.roll(span, 1) < np.roll(span, 2)) & (np.roll(span, 2) < np.roll(span, 3))
    keep = _window(len(c), 40)
    return _emit(
        keep & shrinking & (c > np.roll(h, 3)),
        keep & shrinking & (c < np.roll(lo, 3)),
        c,
        a,
        stop,
    )


@detector("rsi_divergence")
def rsi_divergence(frame, a, *, period=14, look=10, stop=1.0):
    """Price makes a new extreme, RSI does not."""
    c = frame["close"].to_numpy()
    r = _rsi(c, period)
    price_high = pd.Series(c).rolling(look).max().to_numpy()
    price_low = pd.Series(c).rolling(look).min().to_numpy()
    rsi_high = pd.Series(r).rolling(look).max().to_numpy()
    rsi_low = pd.Series(r).rolling(look).min().to_numpy()
    keep = _window(len(c), period + look + 40) & np.isfinite(r)
    bearish = (c >= price_high) & (r < rsi_high - 3)
    bullish = (c <= price_low) & (r > rsi_low + 3)
    return _emit(keep & bullish, keep & bearish, c, a, stop)


@detector("coin_flip_control")
def coin_flip_control(frame, a, *, stop=1.0):
    """THE CONTROL. Random entries, same instrument, same resolver.

    It must come out at zero. If it does not, the harness is broken and every
    other number in this study is worthless -- which is a cheaper thing to
    learn from one row than from a live account.
    """
    c = frame["close"].to_numpy()
    rng = np.random.default_rng(12345)
    keep = _window(len(c), 240)
    picked = keep & (rng.random(len(c)) < 0.02)
    side = rng.random(len(c)) < 0.5
    return _emit(picked & side, picked & ~side, c, a, stop)


ALL = {**ZOO, **ZOO2}

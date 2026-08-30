"""Everything. Every mechanism worth a look, tested the same honest way.

This file is deliberately indiscriminate: candlestick patterns, oscillators,
volatility regimes, clock effects, round numbers, streaks, prior-session
levels, volume, multi-timeframe agreement. Some of it is folklore. That is the
point -- folklore that measures is an edge and folklore that does not is
finally dead rather than merely doubted.

THE COST OF SEARCHING THIS WIDELY, stated up front because it is the whole
danger: at two hundred detectors x six timeframes x three asset classes x four
payoffs, a 3-sigma result is expected dozens of times by chance alone. So
nothing here is believed on its training half. The filter is:

    Bonferroni across the ENTIRE grid, not per family
    the holdout must agree in sign and reach 2 sigma on its own
    positive after a pessimistic spread in BOTH halves
    sigma clustered by day, so simultaneous signals across correlated
      instruments stop counting as independent trades

Everything below returns a Batch and says how it fills: `same_bar=True` for a
limit or open fill, False for an entry at a bar's close. Getting that wrong is
what made `gap_fill` read +1.011R when the truth was nothing at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.lab.resolve import Batch
from scripts.lab.strategies import HORIZON_BARS, _batch, _channel

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return (100 - 100 / (1 + gain / loss.replace(0, np.nan))).to_numpy()


def _emit(mask_long, mask_short, close, a, stop, *, entry=None, same_bar=False) -> Batch:
    """Turn two boolean masks into a Batch, dropping unusable ATR."""
    price = close if entry is None else entry
    ok = np.isfinite(a) & (a > 0) & np.isfinite(price)
    lo = np.flatnonzero(mask_long & ok)
    sh = np.flatnonzero(mask_short & ok)
    rows = [(int(i), 1, float(price[i]), float(stop * a[i])) for i in lo]
    rows += [(int(i), -1, float(price[i]), float(stop * a[i])) for i in sh]
    rows.sort()
    return _batch(rows, same_bar=same_bar)


def _window(n: int, start: int) -> np.ndarray:
    """Bars that are far enough in to have indicators and far enough from the
    end to be resolvable."""
    keep = np.zeros(n, dtype=bool)
    keep[start : n - HORIZON_BARS - 1] = True
    return keep


ZOO: dict = {}


def detector(name):
    def wrap(fn):
        ZOO[name] = fn
        return fn

    return wrap


# ---------------------------------------------------------------------------
# candlestick shapes -- folklore, measured
# ---------------------------------------------------------------------------


def _shapes(frame):
    o, h, lo, c = (frame[x].to_numpy() for x in ("open", "high", "low", "close"))
    span = h - lo
    body = np.abs(c - o)
    with np.errstate(invalid="ignore", divide="ignore"):
        upper_wick = np.where(span > 0, (h - np.maximum(o, c)) / span, 0.0)
        lower_wick = np.where(span > 0, (np.minimum(o, c) - lo) / span, 0.0)
        body_share = np.where(span > 0, body / span, 0.0)
    return o, h, lo, c, span, body_share, upper_wick, lower_wick


@detector("pin_bar")
def pin_bar(frame, a, *, stop=1.0):
    """A long rejection wick. Trade AWAY from the wick."""
    _o, _h, _lo, c, _span, body, up, dn = _shapes(frame)
    keep = _window(len(c), 30)
    return _emit(keep & (dn > 0.6) & (body < 0.3), keep & (up > 0.6) & (body < 0.3), c, a, stop)


@detector("pin_bar_at_extreme")
def pin_bar_at_extreme(frame, a, *, period=20, stop=1.0):
    """The same wick, but only where it means something: at a 20-bar edge."""
    _o, h, lo, c, _span, body, up, dn = _shapes(frame)
    upper, lower = _channel(h, lo, period)
    keep = _window(len(c), period + 20)
    return _emit(
        keep & (dn > 0.6) & (body < 0.3) & (lo <= lower),
        keep & (up > 0.6) & (body < 0.3) & (h >= upper),
        c,
        a,
        stop,
    )


@detector("engulfing")
def engulfing(frame, a, *, stop=1.0):
    o, _h, _lo, c, *_ = _shapes(frame)
    keep = _window(len(c), 30)
    prev_down = np.roll(c, 1) < np.roll(o, 1)
    prev_up = np.roll(c, 1) > np.roll(o, 1)
    swallows = (c > np.roll(o, 1)) & (o < np.roll(c, 1))
    swallowed = (c < np.roll(o, 1)) & (o > np.roll(c, 1))
    return _emit(keep & prev_down & swallows, keep & prev_up & swallowed, c, a, stop)


@detector("outside_bar")
def outside_bar(frame, a, *, stop=1.0):
    o, h, lo, c, *_ = _shapes(frame)
    keep = _window(len(c), 30)
    outside = (h > np.roll(h, 1)) & (lo < np.roll(lo, 1))
    return _emit(keep & outside & (c > o), keep & outside & (c < o), c, a, stop)


@detector("three_bar_reversal")
def three_bar_reversal(frame, a, *, stop=1.0):
    _o, _h, _lo, c, *_ = _shapes(frame)
    keep = _window(len(c), 30)
    down3 = (c < np.roll(c, 1)) & (np.roll(c, 1) < np.roll(c, 2))
    up3 = (c > np.roll(c, 1)) & (np.roll(c, 1) > np.roll(c, 2))
    # Fade the third bar of a three-bar run.
    return _emit(keep & down3, keep & up3, c, a, stop)


@detector("momentum_run_continue")
def momentum_run_continue(frame, a, *, run=4, stop=1.0):
    """Four bars the same way. Follow, rather than fade."""
    c = frame["close"].to_numpy()
    up = np.ones(len(c), dtype=bool)
    down = np.ones(len(c), dtype=bool)
    for k in range(run):
        up &= np.roll(c, k) > np.roll(c, k + 1)
        down &= np.roll(c, k) < np.roll(c, k + 1)
    keep = _window(len(c), run + 30)
    return _emit(keep & up, keep & down, c, a, stop)


@detector("momentum_run_fade")
def momentum_run_fade(frame, a, *, run=4, stop=1.0):
    c = frame["close"].to_numpy()
    up = np.ones(len(c), dtype=bool)
    down = np.ones(len(c), dtype=bool)
    for k in range(run):
        up &= np.roll(c, k) > np.roll(c, k + 1)
        down &= np.roll(c, k) < np.roll(c, k + 1)
    keep = _window(len(c), run + 30)
    return _emit(keep & down, keep & up, c, a, stop)


# ---------------------------------------------------------------------------
# oscillators
# ---------------------------------------------------------------------------


def _rsi_pair(frame, a, period, low, high, stop, fade=True):
    c = frame["close"].to_numpy()
    r = _rsi(c, period)
    keep = _window(len(c), period + 40) & np.isfinite(r)
    over_sold, over_bought = keep & (r < low), keep & (r > high)
    return (
        _emit(over_sold, over_bought, c, a, stop)
        if fade
        else _emit(over_bought, over_sold, c, a, stop)
    )


@detector("rsi14_fade")
def rsi14_fade(frame, a, *, stop=1.0):
    return _rsi_pair(frame, a, 14, 30, 70, stop)


@detector("rsi14_extreme_fade")
def rsi14_extreme_fade(frame, a, *, stop=1.0):
    return _rsi_pair(frame, a, 14, 20, 80, stop)


@detector("rsi7_extreme_fade")
def rsi7_extreme_fade(frame, a, *, stop=1.0):
    return _rsi_pair(frame, a, 7, 15, 85, stop)


@detector("rsi14_follow")
def rsi14_follow(frame, a, *, stop=1.0):
    """Overbought means strong, not stretched. The opposite bet."""
    return _rsi_pair(frame, a, 14, 30, 70, stop, fade=False)


@detector("zscore_fade")
def zscore_fade(frame, a, *, period=50, threshold=2.0, stop=1.0):
    c = frame["close"].to_numpy()
    series = pd.Series(c)
    mean = series.rolling(period).mean().to_numpy()
    sd = series.rolling(period).std().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(sd > 0, (c - mean) / sd, 0.0)
    keep = _window(len(c), period + 20) & np.isfinite(z)
    return _emit(keep & (z < -threshold), keep & (z > threshold), c, a, stop)


@detector("bollinger_touch_fade")
def bollinger_touch_fade(frame, a, *, period=20, dev=2.0, stop=1.0):
    c = frame["close"].to_numpy()
    series = pd.Series(c)
    mean = series.rolling(period).mean().to_numpy()
    sd = series.rolling(period).std().to_numpy()
    keep = _window(len(c), period + 20) & np.isfinite(sd)
    return _emit(keep & (c < mean - dev * sd), keep & (c > mean + dev * sd), c, a, stop)


@detector("ema_stretch_fade")
def ema_stretch_fade(frame, a, *, span=50, stretch=2.0, stop=1.0):
    """Price far from its own average, measured in ATR rather than sigma."""
    c = frame["close"].to_numpy()
    e = _ema(c, span)
    with np.errstate(invalid="ignore", divide="ignore"):
        away = np.where(a > 0, (c - e) / a, 0.0)
    keep = _window(len(c), span + 20)
    return _emit(keep & (away < -stretch), keep & (away > stretch), c, a, stop)


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------


@detector("macd_cross")
def macd_cross(frame, a, *, stop=1.0):
    c = frame["close"].to_numpy()
    macd = _ema(c, 12) - _ema(c, 26)
    signal = pd.Series(macd).ewm(span=9, adjust=False).mean().to_numpy()
    keep = _window(len(c), 60)
    up = (macd > signal) & (np.roll(macd, 1) <= np.roll(signal, 1))
    down = (macd < signal) & (np.roll(macd, 1) >= np.roll(signal, 1))
    return _emit(keep & up, keep & down, c, a, stop)


@detector("ema_cross")
def ema_cross(frame, a, *, fast=20, slow=50, stop=1.0):
    c = frame["close"].to_numpy()
    f, s = _ema(c, fast), _ema(c, slow)
    keep = _window(len(c), slow + 20)
    up = (f > s) & (np.roll(f, 1) <= np.roll(s, 1))
    down = (f < s) & (np.roll(f, 1) >= np.roll(s, 1))
    return _emit(keep & up, keep & down, c, a, stop)


@detector("mtf_trend_pullback")
def mtf_trend_pullback(frame, a, *, stop=1.0):
    """Slow trend decides the side; a dip to the fast average is the entry.

    The multi-timeframe idea done inside one series: EMA200 is the higher
    timeframe, EMA20 is the trigger.
    """
    c = frame["close"].to_numpy()
    slow, fast = _ema(c, 200), _ema(c, 20)
    keep = _window(len(c), 240)
    with np.errstate(invalid="ignore", divide="ignore"):
        dip = np.where(a > 0, (fast - c) / a, 0.0)
    return _emit(
        keep & (c > slow) & (dip > 0) & (dip < 0.75),
        keep & (c < slow) & (-dip > 0) & (-dip < 0.75),
        c,
        a,
        stop,
    )


@detector("turtle_break")
def turtle_break(frame, a, *, period=55, stop=2.0):
    """The classic long-horizon channel break with a wide stop."""
    h, lo, c = (frame[x].to_numpy() for x in ("high", "low", "close"))
    upper, lower = _channel(h, lo, period)
    keep = _window(len(c), period + 20)
    return _emit(keep & (c > upper), keep & (c < lower), c, a, stop)


# ---------------------------------------------------------------------------
# volatility
# ---------------------------------------------------------------------------


@detector("nr7_break")
def nr7_break(frame, a, *, stop=1.0):
    """Narrowest range of seven bars, then a break of it."""
    h, lo, c = (frame[x].to_numpy() for x in ("high", "low", "close"))
    span = pd.Series(h - lo)
    narrowest = (span == span.rolling(7).min()).to_numpy()
    prior_narrow = np.roll(narrowest, 1)
    keep = _window(len(c), 40)
    return _emit(
        keep & prior_narrow & (c > np.roll(h, 1)),
        keep & prior_narrow & (c < np.roll(lo, 1)),
        c,
        a,
        stop,
    )


@detector("atr_expansion_follow")
def atr_expansion_follow(frame, a, *, stop=1.5):
    """A bar far larger than recent ATR. Go with it."""
    o, h, lo, c, *_ = _shapes(frame)
    span = h - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(a > 0, span / a, 0.0)
    keep = _window(len(c), 40)
    big = ratio > 2.0
    return _emit(keep & big & (c > o), keep & big & (c < o), c, a, stop)


@detector("atr_expansion_fade")
def atr_expansion_fade(frame, a, *, stop=1.5):
    o, h, lo, c, *_ = _shapes(frame)
    span = h - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(a > 0, span / a, 0.0)
    keep = _window(len(c), 40)
    big = ratio > 2.0
    return _emit(keep & big & (c < o), keep & big & (c > o), c, a, stop)


# ---------------------------------------------------------------------------
# prior-session levels
# ---------------------------------------------------------------------------


def _prior_day(frame):
    day = frame.index.normalize()
    grouped = frame.groupby(day)
    highs = grouped["high"].max().shift(1)
    lows = grouped["low"].min().shift(1)
    return highs.reindex(day).to_numpy(), lows.reindex(day).to_numpy()


@detector("prior_day_break")
def prior_day_break(frame, a, *, stop=1.0):
    ph, pl = _prior_day(frame)
    c = frame["close"].to_numpy()
    keep = _window(len(c), 40) & np.isfinite(ph) & np.isfinite(pl)
    return _emit(keep & (c > ph), keep & (c < pl), c, a, stop)


@detector("prior_day_fade")
def prior_day_fade(frame, a, *, stop=1.0):
    ph, pl = _prior_day(frame)
    h, lo, c = (frame[x].to_numpy() for x in ("high", "low", "close"))
    keep = _window(len(c), 40) & np.isfinite(ph) & np.isfinite(pl)
    # Poked through and closed back inside: the level held.
    return _emit(keep & (lo < pl) & (c > pl), keep & (h > ph) & (c < ph), c, a, stop)


@detector("round_number_fade")
def round_number_fade(frame, a, *, stop=1.0):
    """Price reaches a round figure. Fade it.

    The grid is chosen from the instrument's own scale so it means the same
    thing on EURUSD at 1.10 and on the Nikkei at 22,000.
    """
    c = frame["close"].to_numpy()
    scale = 10.0 ** np.floor(np.log10(np.nanmedian(np.abs(c)))) / 100.0
    nearest = np.round(c / scale) * scale
    with np.errstate(invalid="ignore", divide="ignore"):
        distance = np.where(a > 0, np.abs(c - nearest) / a, 1.0)
    at_level = distance < 0.05
    keep = _window(len(c), 40)
    rising = c > np.roll(c, 3)
    return _emit(keep & at_level & ~rising, keep & at_level & rising, c, a, stop)


# ---------------------------------------------------------------------------
# clock
# ---------------------------------------------------------------------------


def _hour_detector(hour, direction, stop=1.0):
    def fn(frame, a, *, stop=stop):
        c = frame["close"].to_numpy()
        hours = frame.index.hour.to_numpy()
        minutes = frame.index.minute.to_numpy()
        at = (hours == hour) & (minutes == 0)
        keep = _window(len(c), 40) & at
        empty = np.zeros(len(c), dtype=bool)
        return _emit(keep if direction > 0 else empty, empty if direction > 0 else keep, c, a, stop)

    return fn


for _h in range(0, 24, 3):
    ZOO[f"hour{_h:02d}_long"] = _hour_detector(_h, 1)
    ZOO[f"hour{_h:02d}_short"] = _hour_detector(_h, -1)


@detector("weekday_monday_long")
def weekday_monday_long(frame, a, *, stop=1.0):
    c = frame["close"].to_numpy()
    at = (frame.index.dayofweek.to_numpy() == 0) & (frame.index.hour.to_numpy() == 8)
    keep = _window(len(c), 40) & at
    return _emit(keep, np.zeros(len(c), dtype=bool), c, a, stop)


@detector("session_close_reversal")
def session_close_reversal(frame, a, *, stop=1.0):
    """Fade whatever the last three hours of the US session did."""
    c = frame["close"].to_numpy()
    hours = frame.index.hour.to_numpy()
    at = (hours == 20) & (frame.index.minute.to_numpy() == 0)
    keep = _window(len(c), 60) & at
    rose = c > np.roll(c, 12)
    return _emit(keep & ~rose, keep & rose, c, a, stop)


# ---------------------------------------------------------------------------
# volume
# ---------------------------------------------------------------------------


@detector("volume_spike_follow")
def volume_spike_follow(frame, a, *, stop=1.0):
    o = frame["open"].to_numpy()
    c = frame["close"].to_numpy()
    v = frame["volume"].to_numpy().astype(float)
    baseline = pd.Series(v).rolling(30).mean().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(baseline > 0, v / baseline, 0.0)
    keep = _window(len(c), 60) & (ratio > 3.0)
    return _emit(keep & (c > o), keep & (c < o), c, a, stop)


@detector("volume_spike_fade")
def volume_spike_fade(frame, a, *, stop=1.0):
    o = frame["open"].to_numpy()
    c = frame["close"].to_numpy()
    v = frame["volume"].to_numpy().astype(float)
    baseline = pd.Series(v).rolling(30).mean().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(baseline > 0, v / baseline, 0.0)
    keep = _window(len(c), 60) & (ratio > 3.0)
    return _emit(keep & (c < o), keep & (c > o), c, a, stop)


# ---------------------------------------------------------------------------
# the retest, re-tested at other channel lengths
# ---------------------------------------------------------------------------


def _retest_variant(period, tolerance, stop_beyond):
    def fn(frame, a):
        from scripts.lab.strategies import level_retest

        return level_retest(frame, a, period=period, tolerance=tolerance, stop_beyond=stop_beyond)

    return fn


for _p in (10, 20, 40, 80):
    ZOO[f"retest_p{_p}"] = _retest_variant(_p, 0.15, 0.75)

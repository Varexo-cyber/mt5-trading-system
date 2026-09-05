"""Every candidate entry mechanism, in ONE place, shared by search and live.

WHY THIS MODULE EXISTS, and it is the same reason `section_eleven_metals`
computed its features exactly once. These functions used to live inside
`scripts/search_section_four.py`. A mechanism that a search measures and a
live section then re-implements is two implementations of one definition, and
nothing in the measured numbers says which one the account is running. That is
the defect this project has shipped more often than any other.

So: the search imports from here, `analysis/section_xaujpy.py` imports from
here, and a candidate that is measured is by construction the candidate that
trades.

Each function takes a bar frame indexed by UTC timestamp with open/high/low/
close columns and returns one direction per bar: +1 long, -1 short, 0 nothing.
They are deliberately simple and deliberately DIFFERENT from each other -- a
grid of forty variations on one idea tests one idea forty times and pays the
Bonferroni price for the privilege.

Nothing here looks at a bar it could not have seen. Every reading is built
from bars at or before the one it labels, and the harness enters at that bar's
close, so a candidate cannot be flattered by the future.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

#: Bars of history a candidate may look back over before its first signal.
WARMUP = 120
#: Bars of future a trade is given to resolve. Beyond this it is unresolved and
#: excluded rather than marked to market -- a trade that never reached either
#: barrier has not answered the question.
HORIZON = 48


# --------------------------------------------------------------------------
# candidates
#
# Each returns an array of direction per bar: +1 long, -1 short, 0 nothing.
# Deliberately simple and deliberately DIFFERENT from each other -- a grid of
# forty variations on one idea tests one idea forty times and pays the
# Bonferroni price for the privilege.
# --------------------------------------------------------------------------


def _atr(frame: pd.DataFrame, period: int = 14) -> np.ndarray:
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


def gap_continuation(frame: pd.DataFrame) -> np.ndarray:
    """An index CFD opens away from its last close; trade the direction of it.

    UNTESTABLE ON HISTDATA FX, which is why it is here. Spot FX runs
    continuously and gaps only over a weekend; index CFDs gap at every session
    boundary because the underlying was closed. The original 94 detectors could
    not see this mechanism at all.
    """
    unit = _atr(frame)
    close = frame["close"].to_numpy()
    open_ = frame["open"].to_numpy()
    gap = np.zeros(len(frame))
    gap[1:] = (open_[1:] - close[:-1]) / np.where(unit[1:] > 0, unit[1:], np.nan)
    out = np.zeros(len(frame), dtype=int)
    out[gap > 0.5] = 1
    out[gap < -0.5] = -1
    return out


def gap_fade(frame: pd.DataFrame) -> np.ndarray:
    """The same event, traded the other way. Both directions of one mechanism
    must be measured or the one that happens to pay looks like a discovery."""
    return -gap_continuation(frame)


def overnight_drift(frame: pd.DataFrame) -> np.ndarray:
    """Hold from one session's close into the next, direction of the last day.

    Index CFDs have carried a documented close-to-open drift for decades. If it
    survives this broker's costs it is the cheapest edge available; if it does
    not, that is worth knowing before anything cleverer is tried.
    """
    close = frame["close"].to_numpy()
    unit = _atr(frame)
    out = np.zeros(len(frame), dtype=int)
    if len(close) < 3:
        return out
    move = np.zeros(len(close))
    move[1:] = (close[1:] - close[:-1]) / np.where(unit[1:] > 0, unit[1:], np.nan)
    out[move > 0.3] = 1
    out[move < -0.3] = -1
    return out


def streak_reversal(frame: pd.DataFrame, length: int = 4) -> np.ndarray:
    """Four closes the same way, then trade against it."""
    close = frame["close"].to_numpy()
    up = np.zeros(len(close), dtype=int)
    up[1:] = np.sign(np.diff(close)).astype(int)
    out = np.zeros(len(close), dtype=int)
    for i in range(length, len(close)):
        window = up[i - length + 1 : i + 1]
        if np.all(window > 0):
            out[i] = -1
        elif np.all(window < 0):
            out[i] = 1
    return out


def streak_continuation(frame: pd.DataFrame, length: int = 4) -> np.ndarray:
    return -streak_reversal(frame, length)


def range_expansion(frame: pd.DataFrame) -> np.ndarray:
    """A bar twice the recent range, in its own direction.

    Distinct from `impulse_retest`: no channel, no level, no retest. It buys
    the expansion itself, which the research measured at NOTHING on FX. On a
    zero-commission index with a wider stop the cost arithmetic is different,
    and that difference is the whole question.
    """
    unit = _atr(frame)
    span = (frame["high"] - frame["low"]).to_numpy()
    body = (frame["close"] - frame["open"]).to_numpy()
    out = np.zeros(len(frame), dtype=int)
    big = span > 2.0 * unit
    out[big & (body > 0)] = 1
    out[big & (body < 0)] = -1
    return out


def inside_bar_break(frame: pd.DataFrame) -> np.ndarray:
    """A bar wholly inside its predecessor, then the next close decides."""
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    out = np.zeros(len(frame), dtype=int)
    for i in range(2, len(frame)):
        if high[i - 1] < high[i - 2] and low[i - 1] > low[i - 2]:
            if close[i] > high[i - 1]:
                out[i] = 1
            elif close[i] < low[i - 1]:
                out[i] = -1
    return out


def volatility_contraction(frame: pd.DataFrame) -> np.ndarray:
    """Trade the break out of an unusually quiet stretch, either way it goes."""
    unit = _atr(frame)
    slow = pd.Series(unit).rolling(50).mean().to_numpy()
    close = frame["close"].to_numpy()
    high = pd.Series(frame["high"]).shift(1).rolling(10).max().to_numpy()
    low = pd.Series(frame["low"]).shift(1).rolling(10).min().to_numpy()
    quiet = unit < 0.7 * slow
    out = np.zeros(len(frame), dtype=int)
    out[quiet & (close > high)] = 1
    out[quiet & (close < low)] = -1
    return out


def close_position_in_range(frame: pd.DataFrame) -> np.ndarray:
    """Where the close sits inside the bar. A close on the high after a down
    bar is a rejection; the reverse is exhaustion. One bar, no memory."""
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    span = np.where(high - low > 0, high - low, np.nan)
    position = (close - low) / span
    out = np.zeros(len(frame), dtype=int)
    out[position > 0.9] = 1
    out[position < 0.1] = -1
    return out


def close_position_fade(frame: pd.DataFrame) -> np.ndarray:
    return -close_position_in_range(frame)


def prior_day_break(frame: pd.DataFrame) -> np.ndarray:
    """The first close beyond yesterday's high or low.

    A level everyone can see, on instruments that have a real yesterday --
    which spot FX, trading through midnight, only half does.
    """
    days = frame.index.normalize()
    high = frame["high"].groupby(days).transform("max").shift(1).to_numpy()
    low = frame["low"].groupby(days).transform("min").shift(1).to_numpy()
    close = frame["close"].to_numpy()
    out = np.zeros(len(frame), dtype=int)
    fresh = np.zeros(len(frame), dtype=bool)
    fresh[1:] = days.to_numpy()[1:] != days.to_numpy()[:-1]
    seen_up = seen_down = False
    for i in range(len(frame)):
        if fresh[i]:
            seen_up = seen_down = False
        if not np.isfinite(high[i]) or not np.isfinite(low[i]):
            continue
        if close[i] > high[i] and not seen_up:
            out[i], seen_up = 1, True
        elif close[i] < low[i] and not seen_down:
            out[i], seen_down = -1, True
    return out


def prior_day_fade(frame: pd.DataFrame) -> np.ndarray:
    return -prior_day_break(frame)


#: Name -> signal function. Every mechanism appears in BOTH directions where
#: that makes sense, so a family cannot be credited for the half that happened
#: to work.
INDEX_CANDIDATES: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "gap_continuation": gap_continuation,
    "gap_fade": gap_fade,
    "overnight_drift": overnight_drift,
    "streak_reversal": streak_reversal,
    "streak_continuation": streak_continuation,
    "range_expansion": range_expansion,
    "inside_bar_break": inside_bar_break,
    "volatility_contraction": volatility_contraction,
    "close_position_in_range": close_position_in_range,
    "close_position_fade": close_position_fade,
    "prior_day_break": prior_day_break,
    "prior_day_fade": prior_day_fade,
}


# --------------------------------------------------------------------------
# GOLD INTRADAY — the family section eleven would come out of
#
# WHY A SECOND FAMILY AT ALL, and why gold specifically.
#
# The account's whole trade count already comes from gold. Over the 180-day
# replay of 3 September the four live sections took 1,236 trades and 1,083 of
# them -- 88% -- were the two gold sections. `failed_session_breakout` and
# `section_eight_trend_day_h1` contributed 153 trades and +5.45 R between
# them, which over 137 days is nothing either way.
#
# So "more trades" is a question about gold, and the twelve candidates above
# cannot answer it: they were written for index CFDs on M15 and slower, and
# ten of them key off a session boundary or a daily level that a metal
# trading 23 hours a day barely has.
#
# WHAT MAKES THIS FAMILY DIFFERENT FROM WHAT IS ALREADY LIVE, which is the
# only reason to add anything. Every live section is a MOMENTUM section:
# section six projects features onto a frozen linear model and follows it,
# section ten breaks a micro range, section seven trades a failed break, and
# section eight rides a trend day. All four are long the same underlying bet
# -- that a move continues -- so all four are red on the same days. The
# replay says so: 78 green days out of 137, and the red ones cluster.
#
# Half of what follows fades instead of follows. A mean-reverter is the only
# thing in this repo that can be green on the days the others are red, and
# that is worth more to a EUR 223 account than another momentum section
# would be, because what kills a small account is the depth of one drawdown
# rather than the height of the curve.
#
# EVERY MECHANISM IS HERE IN BOTH DIRECTIONS. Fading and following are the
# same observation read two ways, and a family that only ships the half that
# paid on the sample is a family that has fitted the sample. If `quiet_hour`
# pays in one direction and not the other, that is a finding; if it pays in
# whichever direction the search happened to try, that is noise.
# --------------------------------------------------------------------------


def _session_anchor(
    frame: pd.DataFrame, hour: int, minute: int = 0, *, span: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Per bar: the open of the day's first bar at `hour:minute`, and its age.

    Both are only defined for the `span` bars from that opening bar onward --
    `nan` and `-1` everywhere else -- so a candidate can restrict itself to a
    window without writing the day walk again.

    THE 30-MINUTE TOLERANCE IS NOT COSMETIC. "The first bar at or after
    midnight" on a Sunday is the Sunday OPEN, which on this broker can be
    22:00 or 23:00 or 01:00 depending on the week. Without the tolerance a
    midnight window silently becomes a Sunday-open window on one day in five,
    and the candidate then measures a different event from the one it names.
    """
    index = frame.index
    open_ = frame["open"].to_numpy()
    minutes = index.hour.to_numpy() * 60 + index.minute.to_numpy()
    days = index.normalize().to_numpy()
    want = hour * 60 + minute

    anchor = np.full(len(frame), np.nan)
    age = np.full(len(frame), -1, dtype=int)
    start = -1
    current_day: object = None
    for i in range(len(frame)):
        if days[i] != current_day:
            current_day, start = days[i], -1
        if start < 0 and minutes[i] >= want and minutes[i] - want <= 30:
            start = i
        if start >= 0 and i - start < span:
            anchor[i], age[i] = open_[start], i - start
    return anchor, age


def _stretch(frame: pd.DataFrame, lookback: int = 24) -> np.ndarray:
    """How far the close sits from its own recent mean, in ATR.

    The reference is a plain rolling mean of closes and NOT a session VWAP.
    A VWAP needs the volume to be real; MT5 gives tick counts on a CFD, which
    is a count of quote updates from one broker's feed and not traded size.
    Building the reference on it would look more sophisticated and would key
    the whole family off an artefact of the feed.
    """
    unit = _atr(frame)
    close = frame["close"].to_numpy()
    reference = pd.Series(close).rolling(lookback).mean().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (close - reference) / np.where(unit > 0, unit, np.nan)


def _from_stretch(
    frame: pd.DataFrame, threshold: float, hours: tuple[int, ...] | None
) -> np.ndarray:
    stretch = _stretch(frame)
    out = np.zeros(len(frame), dtype=int)
    with np.errstate(invalid="ignore"):
        out[stretch >= threshold] = -1
        out[stretch <= -threshold] = 1
    if hours is not None:
        out[~np.isin(frame.index.hour.to_numpy(), hours)] = 0
    return out


def stretch_fade(frame: pd.DataFrame) -> np.ndarray:
    """Price 1.8 ATR from its own two-hour mean; trade back toward it.

    The plainest mean reversion there is, on every hour of the day, so that
    the session-restricted versions below have something to be compared to.
    If the unrestricted version pays as well as the restricted one, the
    session story is decoration.
    """
    return _from_stretch(frame, 1.8, None)


def stretch_continuation(frame: pd.DataFrame) -> np.ndarray:
    return -stretch_fade(frame)


#: 00:00-05:59 UTC. Gold trades through it, but the desks that price it do
#: not: no European or US macro prints, no COMEX floor, and the thinnest book
#: of the 23-hour day. A stretched move in a thin book has nobody informed
#: behind it, which is the whole reason to expect it back.
#:
#: IT IS ALSO WHEN THE SPREAD IS WIDEST, so this is the candidate most likely
#: to be killed by cost rather than by direction. The report prints the cost
#: share per clock next to the result precisely so the two cannot be confused.
_QUIET_HOURS = (0, 1, 2, 3, 4, 5)


def quiet_stretch_fade(frame: pd.DataFrame) -> np.ndarray:
    """The same stretch, only in the hours when nothing informed is trading."""
    return _from_stretch(frame, 1.8, _QUIET_HOURS)


def quiet_stretch_continuation(frame: pd.DataFrame) -> np.ndarray:
    return -quiet_stretch_fade(frame)


def _drive(frame: pd.DataFrame, hour: int, *, span: int, minimum: float = 0.5) -> np.ndarray:
    """Direction of the move away from a session's opening price."""
    anchor, age = _session_anchor(frame, hour, span=span)
    unit = _atr(frame)
    close = frame["close"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        move = (close - anchor) / np.where(unit > 0, unit, np.nan)
    out = np.zeros(len(frame), dtype=int)
    live = age >= 1
    with np.errstate(invalid="ignore"):
        out[live & (move >= minimum)] = 1
        out[live & (move <= -minimum)] = -1
    return out


def london_drive(frame: pd.DataFrame) -> np.ndarray:
    """07:00 UTC. European desks arrive and gold's real day starts."""
    return _drive(frame, 7, span=12)


def london_fade(frame: pd.DataFrame) -> np.ndarray:
    return -london_drive(frame)


def comex_drive(frame: pd.DataFrame) -> np.ndarray:
    """13:00 UTC. US data lands at 13:30 and COMEX opens behind it."""
    return _drive(frame, 13, span=12)


def comex_fade(frame: pd.DataFrame) -> np.ndarray:
    return -comex_drive(frame)


def pm_fix_fade(frame: pd.DataFrame) -> np.ndarray:
    """The London PM fix window, faded.

    The 15:00 London auction is the one moment in the day when a known,
    large, price-insensitive quantity of gold changes hands at a single
    printed price. Order flow around it is mechanical rather than
    informational, which is the textbook shape of something that reverts.

    ANCHORED AT 14:00 UTC AND NOT AT THE FIX ITSELF, and the reason is
    boring: London is UTC+1 in summer and UTC+0 in winter, so "15:00 London"
    is two different UTC hours across a 180-day window. The window here is
    wide enough to contain the fix in both, and a candidate that only paid in
    one half of the year would show up as a train/holdout failure rather than
    as a discovery.
    """
    return -_drive(frame, 14, span=18, minimum=0.4)


def pm_fix_drive(frame: pd.DataFrame) -> np.ndarray:
    return -pm_fix_fade(frame)


def round_number_fade(frame: pd.DataFrame, step: float = 10.0) -> np.ndarray:
    """Approaching a ten-dollar level; trade against the approach.

    Gold is quoted in dollars and the resting book clusters on round tens --
    stops above, take-profits below, and option strikes on the hundreds. This
    is the one candidate in the family that could not exist on an FX pair,
    where the equivalent level is a pip value nobody but a machine sees, and
    it is here because it is cheap to measure and impossible to reach by
    tuning anything already in the repo.
    """
    unit = _atr(frame)
    close = frame["close"].to_numpy()
    previous = np.empty(len(close))
    previous[0] = close[0] if len(close) else 0.0
    previous[1:] = close[:-1]
    level = np.round(close / step) * step
    # THE BAND IS THE WIDER OF A VOLATILITY DISTANCE AND A PRICE DISTANCE.
    # 0.15 ATR alone is 4 cents on quiet M1 gold, which asks price to land
    # inside a band no human order sits in -- the candidate would report
    # "never fired" and read as a measured failure. A resting order at 3300
    # is at 3300, not at 3300 plus a fraction of this minute's range, so the
    # floor is a share of the STEP and the ATR term only widens it on a
    # clock fast enough to need it.
    with np.errstate(invalid="ignore"):
        band = np.maximum(0.15 * unit, 0.05 * step)
        near = np.isfinite(unit) & (np.abs(close - level) <= band)
    out = np.zeros(len(frame), dtype=int)
    out[near & (close > previous) & (close <= level)] = -1
    out[near & (close < previous) & (close >= level)] = 1
    return out


def round_number_break(frame: pd.DataFrame) -> np.ndarray:
    return -round_number_fade(frame)


def opening_range_break(frame: pd.DataFrame, build: int = 12, window: int = 48) -> np.ndarray:
    """The gold day's first hour sets a range; take its first break each way.

    Once per direction per day, deliberately: a level that has already been
    broken is not the same level, and re-entering on every bar beyond it
    counts one event as twenty.
    """
    _anchor, age = _session_anchor(frame, 0, span=window)
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    out = np.zeros(len(frame), dtype=int)
    total = len(frame)
    i = 0
    while i < total:
        if age[i] != 0:
            i += 1
            continue
        stop = min(i + window, total)
        if i + build >= stop:
            i = stop
            continue
        ceiling = float(high[i : i + build].max())
        floor = float(low[i : i + build].min())
        up = down = False
        for j in range(i + build, stop):
            if close[j] > ceiling and not up:
                out[j], up = 1, True
            elif close[j] < floor and not down:
                out[j], down = -1, True
        i = stop
    return out


def opening_range_fade(frame: pd.DataFrame) -> np.ndarray:
    return -opening_range_break(frame)


def day_range_exhaustion_fade(frame: pd.DataFrame, span: int = 20) -> np.ndarray:
    """Today has already travelled further than it usually does; fade more.

    A day's range is the most reliably mean-reverting quantity on any
    instrument -- a wide day is followed by a narrow one far more often than
    chance -- and this asks the intraday version of it: once the day has
    already spent its usual range, does the next extension pay or give back?

    Distinct from `stretch_fade`, which measures distance from a two-hour
    mean and knows nothing about the day.
    """
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    days = frame.index.normalize()
    frame_days = pd.Series(days)
    running_high = pd.Series(high).groupby(frame_days).cummax().to_numpy()
    running_low = pd.Series(low).groupby(frame_days).cummin().to_numpy()
    travelled = running_high - running_low

    full = pd.Series(high).groupby(frame_days).transform("max") - pd.Series(low).groupby(
        frame_days
    ).transform("min")
    # The median of the PREVIOUS `span` days, never today's own range: using
    # the finished range of the day being traded is look-ahead, and it is the
    # single easiest way to manufacture an edge in a study like this.
    per_day = full.groupby(frame_days).first()
    typical = per_day.shift(1).rolling(span).median()
    expected = frame_days.map(typical).to_numpy()

    # AT ITS TYPICAL RANGE, NOT AT 1.2x IT, and "near the extreme" rather
    # than exactly on it. The first version asked for both -- 1.2x the median
    # range AND a close equal to the running high to the last decimal -- and
    # fired zero times in ninety days. Two defensible-looking thresholds
    # multiplied into an impossible one, and the output would have been
    # indistinguishable from a mechanism that was measured and lost.
    unit = _atr(frame)
    out = np.zeros(len(frame), dtype=int)
    with np.errstate(invalid="ignore"):
        spent = np.isfinite(expected) & (expected > 0) & (travelled >= expected)
        edge = np.where(np.isfinite(unit), 0.25 * unit, 0.0)
        out[spent & (close >= running_high - edge)] = -1
        out[spent & (close <= running_low + edge)] = 1
    return out


def day_range_exhaustion_break(frame: pd.DataFrame) -> np.ndarray:
    return -day_range_exhaustion_fade(frame)


#: The gold-intraday grid. Sixteen cells per clock, and the Bonferroni bar in
#: the report is computed from however many of them actually fired -- so
#: adding a seventeenth idea raises the bar every other idea has to clear.
#: That is the correct incentive and it is why this dict is short.
GOLD_CANDIDATES: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "stretch_fade": stretch_fade,
    "stretch_continuation": stretch_continuation,
    "quiet_stretch_fade": quiet_stretch_fade,
    "quiet_stretch_continuation": quiet_stretch_continuation,
    "london_drive": london_drive,
    "london_fade": london_fade,
    "comex_drive": comex_drive,
    "comex_fade": comex_fade,
    "pm_fix_fade": pm_fix_fade,
    "pm_fix_drive": pm_fix_drive,
    "round_number_fade": round_number_fade,
    "round_number_break": round_number_break,
    "opening_range_break": opening_range_break,
    "opening_range_fade": opening_range_fade,
    "day_range_exhaustion_fade": day_range_exhaustion_fade,
    "day_range_exhaustion_break": day_range_exhaustion_break,
}

FAMILIES: dict[str, dict[str, Callable[[pd.DataFrame], np.ndarray]]] = {
    "index": INDEX_CANDIDATES,
    "gold": GOLD_CANDIDATES,
    "all": {**INDEX_CANDIDATES, **GOLD_CANDIDATES},
}

#: Kept so anything importing the old name still works.
CANDIDATES = INDEX_CANDIDATES

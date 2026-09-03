"""Search this broker's own bars for an edge that survives its own costs.

    python scripts/search_section_four.py --days 365

WHY THIS RUNS ON EIGHTCAP DATA AND NOT ON HISTDATA.

The original research measured 94 detectors over ten years of HistData and
shipped the two that survived a holdout. On this broker, over 1,610 live-shaped
trades and 180 days, those two came back at 49.9% -- a coin flip. The entry
carried no information at all.

So the failure was not "not enough research". It was research on the wrong
data. HistData FX bid bars cannot price this account's spread, cannot see its
commission schedule, and contain none of the session structure that index CFDs
have. Doing more of it would be repeating the mistake with more decimals.

This searches the bars the account actually trades.

WHAT IT GUARDS AGAINST, because a search is a machine for finding noise:

  * BONFERRONI. Testing 40 cells and keeping the best one finds a 2-sigma
    result about 63% of the time on pure noise. The bar rises with the size of
    the grid and is printed with the grid.
  * A HOLDOUT SPLIT BY DATE. Train on the older 60%, and a candidate must
    reach its bar on the newer 40% ON ITS OWN, in the same direction.
  * DAY-CLUSTERED SIGMA. Sixteen markets breaking on one morning are one
    observation. This was the largest single correction in the original
    research and it is the easiest one to lose.
  * A RANDOM CONTROL, measured in the same harness, on the same bars, at the
    same ratio. A bar registers a barrier when its extreme crosses it, and the
    overshoot is proportionally larger on the nearer one, so a coin flip does
    NOT read zero. Whatever it reads is subtracted from every candidate.
  * THE REAL COST. Charged from this broker's own commission schedule and
    slippage assumption, per asset class, exactly as the sizer charges it.

Nothing here is a strategy yet. It is the machine that decides whether one
exists, and it is built to come back empty.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from backtesting.replay import fetch_mt5_history
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from core.types import Timeframe
from risk.position_sizer import PositionSizer

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


@dataclass
class Trades:
    """Resolved outcomes, kept as parallel arrays so the statistics are cheap."""

    r: list[float] = field(default_factory=list)
    day: list[object] = field(default_factory=list)
    when: list[datetime] = field(default_factory=list)

    def extend(self, other: Trades) -> None:
        self.r.extend(other.r)
        self.day.extend(other.day)
        self.when.extend(other.when)

    def __len__(self) -> int:
        return len(self.r)


def resolve(
    frame: pd.DataFrame,
    signals: np.ndarray,
    *,
    stop_atr: float,
    ratio: float,
    cost_r: float,
    horizon: int = HORIZON,
) -> Trades:
    """First touch of stop or target, entry at the signal bar's close.

    THE RULES ARE THE RESEARCH'S, and the two that matter both cost the
    candidate rather than flatter it: resolution starts on the bar AFTER the
    entry bar, and a bar that spans both barriers is a LOSS because the order
    inside it is unknowable.
    """
    out = Trades()
    unit = _atr(frame)
    close = frame["close"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    index = frame.index

    for i in range(WARMUP, len(frame) - 1):
        direction = int(signals[i])
        if direction == 0 or not np.isfinite(unit[i]) or unit[i] <= 0:
            continue
        entry = close[i]
        risk = stop_atr * unit[i]
        stop = entry - direction * risk
        target = entry + direction * ratio * risk
        for j in range(i + 1, min(i + 1 + horizon, len(frame))):
            if direction > 0:
                hit_stop, hit_target = low[j] <= stop, high[j] >= target
            else:
                hit_stop, hit_target = high[j] >= stop, low[j] <= target
            if hit_stop:
                out.r.append(-1.0 - cost_r)
                break
            if hit_target:
                out.r.append(ratio - cost_r)
                break
        else:
            continue
        out.day.append(index[i].date())
        out.when.append(index[i])
    return out


def stats(trades: Trades) -> tuple[float, float, float, int]:
    """`(total R, per trade, day-clustered sigma, n)`.

    SIGMA COMES FROM THE SPREAD OF DAILY TOTALS. Counting each trade as an
    independent observation treats sixteen markets breaking on one morning as
    sixteen, and overstates significance by roughly the square root of however
    many moved together.
    """
    n = len(trades)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    by_day: dict[object, float] = {}
    for day, r in zip(trades.day, trades.r, strict=True):
        by_day[day] = by_day.get(day, 0.0) + r
    daily = np.array(list(by_day.values()), dtype=float)
    total = float(daily.sum())
    if len(daily) < 2:
        return total, total / n, 0.0, n
    se = float(daily.std(ddof=1)) * float(np.sqrt(len(daily)))
    return total, total / n, (total / se if se > 0 else 0.0), n


def _cost_share(sizer, spec, stop_price: float) -> float:
    """What a round trip costs as a fraction of the stop.

    DELEGATED TO THE SIZER, and the first version of this function is why.

    I reimplemented it here as

        pip = spec.point * 10.0
        commission_price = (per_side * 2.0) * pip / 10.0

    which is dimensionally nonsense: commission is account currency per lot,
    and multiplying it by a tenth of a pip does not convert it to price. It
    needs the instrument's pip VALUE, which depends on contract size and
    quote currency -- exactly what `spec.money_per_lot` and
    `spec.pips_to_price` already know.

    The result showed up as gold reading a 62% cost share on H1 against 0.2%
    on M30. Same instrument, same formula, and cost must FALL on a slower
    clock because the stop is wider. A number that moves 300x the wrong way
    is not a property of gold.

    `PositionSizer._cost_share` is the definition the account charges, its own
    docstring warns that "two definitions of the same cost would eventually
    disagree", and I wrote the second one anyway.
    """
    commission = sizer.settings.risk.commission_per_lot(spec.asset_class.value)
    return sizer._cost_share(spec, stop_price, commission)


def random_control(frame: pd.DataFrame, seed: int) -> np.ndarray:
    """A coin flip on the same bars, so the harness measures its own bias.

    IT DOES NOT READ ZERO, and the original research learned that the
    expensive way: random entries came back at +0.073R and +13.8 sigma at a
    3:1 target. A bar registers a barrier when its EXTREME crosses it, and the
    overshoot is proportionally larger on the nearer barrier, so the harness
    manufactures a small edge out of nothing. Whatever it reads here is
    subtracted from every candidate in the same cell.
    """
    rng = np.random.default_rng(seed)
    out = rng.choice([-1, 0, 1], size=len(frame), p=[0.15, 0.70, 0.15])
    return out.astype(int)


@dataclass
class Cell:
    """One candidate on one clock over one asset class."""

    candidate: str
    clock: str
    asset_class: str
    train: Trades = field(default_factory=Trades)
    test: Trades = field(default_factory=Trades)
    control: Trades = field(default_factory=Trades)


def bonferroni_sigma(cells: int, target_p: float = 0.05) -> float:
    """The sigma a single cell must reach when `cells` of them were tried.

    Testing forty cells and keeping the best finds a 2-sigma result about 63%
    of the time on pure noise. This is not a formality -- it is the difference
    between a search and a story.
    """
    from math import erfc, sqrt

    if cells <= 0:
        return 2.0
    per_cell = target_p / cells
    low, high = 0.0, 8.0
    for _ in range(80):
        mid = (low + high) / 2.0
        # two-sided tail
        if erfc(mid / sqrt(2.0)) > per_cell:
            low = mid
        else:
            high = mid
    return round(high, 2)


def verdict(cell: Cell, bar: float) -> tuple[bool, str]:
    """Does this cell clear every bar, and if not, which one stopped it."""
    _train_total, train_each, train_sigma, train_n = stats(cell.train)
    _test_total, test_each, test_sigma, test_n = stats(cell.test)
    _c_total, control_each, _c_sigma, control_n = stats(cell.control)

    if train_n < 150 or test_n < 100:
        return False, f"too few trades ({train_n} train / {test_n} holdout)"
    net_train = train_each - control_each
    net_test = test_each - control_each
    if net_train <= 0:
        return False, f"train {net_train:+.3f} R net of control"
    if train_sigma < bar:
        return False, f"train {train_sigma:+.2f} sigma, bar is {bar:.2f}"
    if net_test <= 0:
        return False, f"holdout {net_test:+.3f} R net of control — train only"
    if test_sigma < 2.0:
        return False, f"holdout {test_sigma:+.2f} sigma on its own"
    return True, (
        f"train {net_train:+.3f} R at {train_sigma:+.2f} sigma over {train_n}, "
        f"holdout {net_test:+.3f} R at {test_sigma:+.2f} over {test_n} "
        f"(control {control_each:+.3f} R over {control_n})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument(
        "--clocks",
        nargs="*",
        default=["M15", "M30", "H1"],
        help="timeframes to try each candidate on, space or comma separated",
    )
    parser.add_argument("--symbols", default="", help="comma list; default = the core universe")
    parser.add_argument(
        "--asset-class",
        default="",
        help="every market the scanner puts in this class, e.g. metal. Ignored "
        "when --symbols is given.",
    )
    parser.add_argument("--stop-atr", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument(
        "--family",
        choices=sorted(FAMILIES),
        default="index",
        help="which candidate grid to run: index (the original twelve), "
        "gold (the sixteen intraday metal mechanisms), or all",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=HORIZON,
        help="bars a trade is given to reach a barrier before it is discarded",
    )
    parser.add_argument(
        "--cells-already-tried",
        type=int,
        default=0,
        help=(
            "cells searched in EARLIER runs of this script, added to this run's "
            "count before the Bonferroni bar is computed. Two searches of forty "
            "cells are eighty hypotheses, and paying for forty twice is how a "
            "search launders itself into a discovery."
        ),
    )
    parser.add_argument("--csv", default="", help="write every cell's numbers here")
    parser.add_argument(
        "--database",
        default="",
        help="read the one-file SQLite research archive instead of contacting MT5",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    clocks = [
        piece.strip().upper()
        for chunk in args.clocks
        for piece in str(chunk).split(",")
        if piece.strip()
    ]

    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=True)
    dataset = None
    connector = None
    if args.database:
        from backtesting.research_dataset import ResearchDataset

        dataset = ResearchDataset(Path(args.database), read_only=True)
    else:
        connector = MT5Connector(
            settings.mt5,
            load_credentials(required=True),
            terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
        )
        connector.connect()
    try:
        from scanner.universe import UniverseScanner
        from scripts.dry_run_sections import _core_universe

        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        elif args.asset_class:
            # THE SCANNER'S OWN CLASSIFIER, not a substring match on the name.
            # Eightcap's metals are not all called XAU-something and the
            # spelling carries suffixes; asking the classifier means the
            # search finds whatever this broker actually lists rather than
            # whatever I guessed it was called. An empty result is reported
            # rather than run, because zero symbols and zero setups print the
            # same way and this repo has confused the two before.
            assert connector is not None
            wanted = args.asset_class.strip().lower()
            symbols = []
            for item in connector.symbols():
                if settings.instruments.is_ignored(item.name):
                    continue
                try:
                    found = UniverseScanner._path_class(connector.spec(item.name).path).value
                except Exception:  # noqa: BLE001 - a bad symbol is not a reason to stop
                    continue
                if found.lower() == wanted:
                    symbols.append(item.name)
            if not symbols:
                classes = sorted(
                    {
                        UniverseScanner._path_class(connector.spec(i.name).path).value
                        for i in connector.symbols()[:400]
                    }
                )
                raise SystemExit(
                    f"no symbols in asset class {wanted!r}. "
                    f"This broker's catalogue has: {', '.join(classes)}"
                )
        elif dataset is not None:
            symbols = dataset.symbols()
        else:
            assert connector is not None
            symbols = _core_universe(connector, settings)

        stored_window = dataset.window() if dataset is not None else None
        end = (
            datetime.fromisoformat(stored_window[1])
            if stored_window is not None and stored_window[1]
            else datetime.now(UTC)
        )
        start = end - timedelta(days=args.days)
        split = start + (end - start) * 0.6

        sizer = PositionSizer(settings)
        cells: dict[tuple[str, str, str], Cell] = {}
        #: (clock, asset class) -> cost share, printed with the result. It is
        #: the number the whole search turns on and it was invisible.
        costs: dict[tuple[str, str], float] = {}
        grid = FAMILIES[args.family]
        print(
            f"\nSEARCHING {len(symbols)} markets x {len(clocks)} clocks "
            f"x {len(grid)} candidates ({args.family} family), {args.days} days"
        )
        print(f"train up to {split:%Y-%m-%d}, holdout after it")
        print(f"horizon {args.horizon} bars, stop {args.stop_atr} ATR, target {args.ratio}:1\n")

        for position, symbol in enumerate(symbols, 1):
            try:
                spec = dataset.spec(symbol) if dataset is not None else connector.spec(symbol)
                asset_class = (
                    spec.asset_class.value
                    if dataset is not None
                    else UniverseScanner._path_class(spec.path).value
                )
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the run
                print(f"  [{position}/{len(symbols)}] {symbol}: no spec ({exc})")
                continue
            for clock_name in clocks:
                clock = Timeframe.parse(clock_name)
                try:
                    requested_start = start - (WARMUP + 40) * clock.duration
                    frame = (
                        dataset.frame(symbol, clock, requested_start, end)
                        if dataset is not None
                        else fetch_mt5_history(connector, symbol, clock, requested_start, end)
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{position}/{len(symbols)}] {symbol} {clock_name}: {exc}")
                    continue
                if len(frame) < WARMUP + HORIZON + 200:
                    continue
                stop_price = args.stop_atr * float(np.nanmedian(_atr(frame)))
                cost_r = _cost_share(sizer, spec, stop_price)
                costs[(clock_name, asset_class)] = cost_r

                for name, detector in grid.items():
                    key = (name, clock_name, asset_class)
                    cell = cells.setdefault(key, Cell(name, clock_name, asset_class))
                    signals = detector(frame)
                    found = resolve(
                        frame,
                        signals,
                        stop_atr=args.stop_atr,
                        ratio=args.ratio,
                        cost_r=cost_r,
                        horizon=args.horizon,
                    )
                    _split_into(found, split, cell)
                # ONE control per (clock, asset class), not per candidate: it
                # measures the HARNESS, and running twelve of them would only
                # measure the same thing twelve times with more noise.
                control_key = ("__control__", clock_name, asset_class)
                control = cells.setdefault(
                    control_key, Cell("__control__", clock_name, asset_class)
                )
                found = resolve(
                    frame,
                    random_control(frame, seed=hash((symbol, clock_name)) & 0xFFFF),
                    stop_atr=args.stop_atr,
                    ratio=args.ratio,
                    cost_r=cost_r,
                    horizon=args.horizon,
                )
                _split_into(found, split, control)
            print(f"  [{position}/{len(symbols)}] {symbol} done")
    finally:
        if dataset is not None:
            dataset.close()
        elif connector is not None:
            connector.shutdown()

    _report(cells, args, costs)


def _split_into(found: Trades, split: datetime, cell: Cell) -> None:
    for r, day, when in zip(found.r, found.day, found.when, strict=True):
        target = cell.train if when < split else cell.test
        target.r.append(r)
        target.day.append(day)
        target.when.append(when)


def _report(cells: dict, args, costs: dict | None = None) -> None:
    """What survived, and if nothing did, what stopped each one.

    A SEARCH THAT ONLY PRINTS ITS WINNERS IS A STORY. The near misses are the
    part that says whether the grid was worth running: forty cells all stopped
    at "too few trades" means the window was short, and forty stopped at
    "holdout, train only" means the search is fitting noise and no amount of
    further tuning will help.
    """
    controls = {
        (cell.clock, cell.asset_class): cell
        for cell in cells.values()
        if cell.candidate == "__control__"
    }
    real = [cell for cell in cells.values() if cell.candidate != "__control__"]
    tested = [c for c in real if len(c.train) >= 150 and len(c.test) >= 100]
    earlier = max(int(getattr(args, "cells_already_tried", 0) or 0), 0)
    counted = max(len(tested), 1) + earlier
    bar = bonferroni_sigma(counted)

    print("\n" + "=" * 78)
    print("SEARCH RESULT")
    print("=" * 78)
    print(f"  {len(real)} cells built, {len(tested)} had enough trades to judge")
    if earlier:
        print(f"  plus {earlier} cells declared from earlier searches of this project")
    print(f"  Bonferroni bar at {counted} hypotheses: {bar:.2f} sigma on train")
    print("     (2.0 would be the bar for ONE hypothesis. Keeping the best of")
    print("      many finds a 2-sigma result on pure noise most of the time.)")
    print("     The holdout bar stays 2.0, and it has to be cleared on the")
    print("     holdout's own trades -- that half was never searched over.")

    if costs:
        print("\n  WHAT A ROUND TRIP COSTS, as a share of the stop")
        for (clock, asset_class), share in sorted(costs.items(), key=lambda kv: -kv[1]):
            flag = "  <- SUSPECT" if share > 0.5 else ""
            print(f"    {asset_class:<10} {clock:<4} {share:>7.1%}{flag}")
        print("    Above ~25% nothing pays, whatever the entry does. A share")
        print("    that RISES on a slower clock is a bug, not a market.")

    print("\n  THE HARNESS'S OWN BIAS — random entries on the same bars")
    for (clock, asset_class), cell in sorted(controls.items()):
        _t, each, sigma, n = stats(cell.train)
        if n:
            print(
                f"    {asset_class:<10} {clock:<4} {each:+.4f} R over {n:>6} random trades"
                f"   ({sigma:+.2f} sigma)"
            )
    print("    Subtracted from every candidate in its own cell.")

    for cell in real:
        control = controls.get((cell.clock, cell.asset_class))
        if control is not None:
            cell.control = control.train

    passed = [(c, verdict(c, bar)) for c in real]
    winners = [(c, why) for c, (ok, why) in passed if ok]

    if winners:
        print(f"\n  SURVIVED EVERY BAR — {len(winners)} of {len(tested)}")
        for cell, why in winners:
            print(f"\n    {cell.candidate}  {cell.clock}  {cell.asset_class}")
            print(f"      {why}")
        print("\n  These are candidates for section four. Nothing is live until")
        print("  it is built into a module and measured again by history.cmd.")
    else:
        print(f"\n  NOTHING SURVIVED. {len(tested)} cells were judged and none cleared")
        print("  the bar. That is the expected outcome of an honest search and it")
        print("  is not a failure of the run.")

    # A CANDIDATE THAT NEVER FIRED IS NOT A CANDIDATE THAT FAILED, and the
    # two must not look the same. Silence means the threshold is wrong for
    # this feed, which is a fixable mistake of mine; failure means the
    # mechanism does not pay, which is an answer. A first version of this
    # report showed neither, and six of twelve candidates producing zero
    # trades would have been invisible.
    silent = sorted({c.candidate for c in real if len(c.train) + len(c.test) == 0})
    thin = sorted({c.candidate for c in real if 0 < len(c.train) + len(c.test) < 250} - set(silent))
    if silent:
        print(f"\n  NEVER FIRED — {len(silent)} candidates produced no trades at all")
        print(f"    {', '.join(silent)}")
        print("    That is a threshold that does not match this feed, not a result.")
    if thin:
        print(f"\n  TOO THIN TO JUDGE — fired, but under 250 trades: {', '.join(thin)}")

    print("\n  CLOSEST MISSES — what stopped each of the ten best")
    ranked = sorted(
        (c for c in real if len(c.train) >= 150),
        key=lambda c: -stats(c.train)[2],
    )[:10]
    for cell in ranked:
        _t, each, sigma, n = stats(cell.train)
        _ct, control_each, _cs, _cn = stats(cell.control)
        _ok, why = verdict(cell, bar)
        print(
            f"    {cell.candidate:<24} {cell.clock:<4} {cell.asset_class:<10} "
            f"{each - control_each:+.3f} R  {sigma:+5.2f} sigma  n={n:<6} {why}"
        )

    if args.csv:
        import csv as csv_module

        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv_module.writer(handle)
            writer.writerow(
                [
                    "candidate",
                    "clock",
                    "asset_class",
                    "train_n",
                    "train_r_per_trade",
                    "train_sigma",
                    "holdout_n",
                    "holdout_r_per_trade",
                    "holdout_sigma",
                    "control_r_per_trade",
                    "verdict",
                ]
            )
            for cell in real:
                _t1, train_each, train_sigma, train_n = stats(cell.train)
                _t2, test_each, test_sigma, test_n = stats(cell.test)
                _t3, control_each, _s3, _n3 = stats(cell.control)
                ok, why = verdict(cell, bar)
                writer.writerow(
                    [
                        cell.candidate,
                        cell.clock,
                        cell.asset_class,
                        train_n,
                        round(train_each, 4),
                        round(train_sigma, 2),
                        test_n,
                        round(test_each, 4),
                        round(test_sigma, 2),
                        round(control_each, 4),
                        "PASS" if ok else why,
                    ]
                )
        print(f"\n  every cell written to {path}")


if __name__ == "__main__":
    main()

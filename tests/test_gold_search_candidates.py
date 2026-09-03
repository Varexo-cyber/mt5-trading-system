"""The gold-intraday grid has to FIRE before it can be judged.

A search whose detector never triggers reports the same silence as a search
whose detector was measured and lost, and this repo has produced that
confusion at least six times. The candidates are measured against synthetic
bars here for one reason only: to prove each mechanism reaches the resolver
and produces a plausible number of trades on gold-shaped data.

NONE OF THIS MEASURES AN EDGE. The bars are a random walk with a session
shape bolted on; the R that comes out of them is noise by construction, and
these tests deliberately assert nothing about it. Whether any candidate pays
is decided by `zoek11.cmd` on the broker's own bars, and nowhere else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.search_section_four import (
    FAMILIES,
    GOLD_CANDIDATES,
    WARMUP,
    _session_anchor,
    bonferroni_sigma,
    day_range_exhaustion_fade,
    opening_range_break,
    resolve,
    round_number_fade,
    stats,
    stretch_fade,
)


def _gold_bars(days: int = 90, minutes: int = 5, seed: int = 11) -> pd.DataFrame:
    """Gold-shaped M5 bars: 23 hours a day, weekdays, a break at 22:00 UTC.

    The level and the volatility are gold's (about 3,300 dollars, roughly a
    one-percent day) so that ATR-relative thresholds and the ten-dollar round
    numbers land where they would on the real instrument. Direction is a
    random walk.
    """
    rng = np.random.default_rng(seed)
    stamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2026-01-05", tz="UTC")  # a Monday
    while len(stamps) < days * (23 * 60 // minutes):
        if day.dayofweek < 5:
            for step in range(0, 23 * 60, minutes):
                stamps.append(day + pd.Timedelta(minutes=step))
        day += pd.Timedelta(days=1)
    index = pd.DatetimeIndex(stamps)

    # A step whose size follows the session: the quiet hours really are
    # quieter, which is what the session-restricted candidates key on.
    hour = index.hour.to_numpy()
    scale = np.where((hour >= 7) & (hour < 20), 1.0, 0.45)
    step = rng.normal(0.0, 0.9, size=len(index)) * scale
    close = 3300.0 + np.cumsum(step)
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, 0.5, size=len(index))) * scale
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=index,
    )


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return _gold_bars()


@pytest.mark.parametrize("name", sorted(GOLD_CANDIDATES))
def test_every_gold_candidate_fires_on_gold_shaped_bars(name: str, bars: pd.DataFrame) -> None:
    """The failure this guards against is a threshold that never triggers.

    A three-hour search that comes back "NEVER FIRED -- 16 candidates
    produced no trades at all" has measured the author's arithmetic, not the
    market, and there is no way to tell the two apart from the output.
    """
    signals = GOLD_CANDIDATES[name](bars)
    assert len(signals) == len(bars), f"{name} returned the wrong length"
    fired = int(np.count_nonzero(signals[WARMUP:]))
    assert fired > 0, f"{name} never fired on 90 days of gold-shaped M5 bars"


@pytest.mark.parametrize("name", sorted(GOLD_CANDIDATES))
def test_no_gold_candidate_fires_on_nearly_every_bar(name: str, bars: pd.DataFrame) -> None:
    """A detector that is always on is a detector that says nothing.

    `stretch_continuation` firing on 80% of bars would take a trade on every
    bar of the day, and its R would then describe the resolver's own barrier
    bias rather than the mechanism. Half the bars is generous and still
    catches an inverted comparison.
    """
    signals = GOLD_CANDIDATES[name](bars)
    live = int(np.count_nonzero(signals[WARMUP:]))
    assert live < 0.5 * (len(bars) - WARMUP), f"{name} fired on {live} of {len(bars)} bars"


def test_each_mechanism_is_present_in_both_directions() -> None:
    """Fading and following are one observation read two ways.

    Shipping only the half that paid on the sample is the cheapest way to
    fit a sample, so the grid is checked for symmetry rather than trusted to
    stay symmetric through an edit.
    """
    pairs = [
        ("stretch_fade", "stretch_continuation"),
        ("quiet_stretch_fade", "quiet_stretch_continuation"),
        ("london_drive", "london_fade"),
        ("comex_drive", "comex_fade"),
        ("pm_fix_fade", "pm_fix_drive"),
        ("round_number_fade", "round_number_break"),
        ("opening_range_break", "opening_range_fade"),
        ("day_range_exhaustion_fade", "day_range_exhaustion_break"),
    ]
    assert {n for pair in pairs for n in pair} == set(GOLD_CANDIDATES)
    frame = _gold_bars(days=20, seed=3)
    for one, other in pairs:
        left = GOLD_CANDIDATES[one](frame)
        right = GOLD_CANDIDATES[other](frame)
        assert np.array_equal(left, -right), f"{one} and {other} are not opposites"


def test_a_stretched_close_is_faded_toward_the_mean() -> None:
    """The direction, not the count: a high close must produce a SHORT."""
    index = pd.date_range("2026-02-02 08:00", periods=200, freq="5min", tz="UTC")
    close = np.full(200, 3300.0)
    close[-1] = 3340.0  # far above its own two-hour mean
    frame = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close}, index=index
    )
    assert stretch_fade(frame)[-1] == -1


def test_the_opening_range_is_broken_once_per_direction_per_day() -> None:
    """A level that has been broken is not the same level any more.

    Re-entering on every bar beyond it counts one event as twenty, and on M1
    gold that alone would triple the trade count of the whole family.
    """
    frame = _gold_bars(days=6, seed=5)
    signals = opening_range_break(frame)
    per_day: dict[object, list[int]] = {}
    for stamp, value in zip(frame.index, signals, strict=True):
        if value:
            per_day.setdefault(stamp.normalize(), []).append(int(value))
    assert per_day, "the opening range never broke in six days"
    for day, taken in per_day.items():
        assert taken.count(1) <= 1, f"{day} broke up more than once"
        assert taken.count(-1) <= 1, f"{day} broke down more than once"


def test_day_range_exhaustion_never_reads_the_day_it_trades() -> None:
    """The reference range must come from FINISHED days only.

    Comparing today's travel against today's own finished range is
    look-ahead, and it is the single easiest way to manufacture an edge in a
    study like this: the bar that sets the day's extreme is exactly the bar
    the detector would then fire on.
    """
    frame = _gold_bars(days=40, seed=8)
    signals = day_range_exhaustion_fade(frame)
    # The first day has no previous day to take a median from, so it cannot
    # have an opinion. If it does, the reference is reading forward.
    first = frame.index.normalize() == frame.index.normalize()[0]
    assert not np.count_nonzero(signals[first])


def test_the_round_number_rule_needs_a_round_number() -> None:
    """Approaching 3300 from below is a short; drifting at 3305 is nothing."""
    index = pd.date_range("2026-02-02 08:00", periods=60, freq="5min", tz="UTC")
    close = np.linspace(3294.0, 3299.9, 60)
    frame = pd.DataFrame(
        {"open": close, "high": close + 0.2, "low": close - 0.2, "close": close}, index=index
    )
    assert round_number_fade(frame)[-1] == -1

    away = np.linspace(3304.0, 3305.5, 60)
    drifting = pd.DataFrame(
        {"open": away, "high": away + 0.2, "low": away - 0.2, "close": away}, index=index
    )
    assert round_number_fade(drifting)[-1] == 0


def test_a_sunday_open_does_not_become_the_midnight_window() -> None:
    """ "The first bar at or after midnight" is not always near midnight.

    This broker's week can open at 22:00, 23:00 or 01:00. Without the
    30-minute tolerance the midnight window silently becomes a Sunday-open
    window on one day in five, and the candidate then measures a different
    event from the one it is named after.
    """
    late = pd.date_range("2026-02-02 03:00", periods=30, freq="5min", tz="UTC")
    price = np.full(30, 3300.0)
    frame = pd.DataFrame({"open": price, "high": price, "low": price, "close": price}, index=late)
    _anchor, age = _session_anchor(frame, 0, span=12)
    assert not np.count_nonzero(age >= 0), "a 03:00 first bar was treated as midnight"

    on_time = pd.date_range("2026-02-02 00:00", periods=30, freq="5min", tz="UTC")
    punctual = frame.set_axis(on_time)
    _anchor, age = _session_anchor(punctual, 0, span=12)
    assert int(np.count_nonzero(age >= 0)) == 12


def test_the_resolver_charges_the_cost_on_both_outcomes() -> None:
    """Cost comes off a winner and is added to a loser, never skipped.

    The dry-run harness shipped for weeks with a cost model that was correct,
    tested, printed -- and never applied to a trade that was TAKEN. The same
    mistake here would flatter every cell in the grid equally, which is the
    hardest kind to notice.
    """
    frame = _gold_bars(days=30, seed=13)
    signals = stretch_fade(frame)
    free = resolve(frame, signals, stop_atr=1.0, ratio=1.0, cost_r=0.0)
    charged = resolve(frame, signals, stop_atr=1.0, ratio=1.0, cost_r=0.05)
    assert len(free) == len(charged) > 0
    assert stats(charged)[1] == pytest.approx(stats(free)[1] - 0.05, abs=1e-9)


def test_the_horizon_is_the_resolver_s_and_not_a_constant() -> None:
    """A shorter horizon discards more trades; it must not silently ignore it."""
    frame = _gold_bars(days=30, seed=17)
    signals = stretch_fade(frame)
    short = resolve(frame, signals, stop_atr=1.0, ratio=1.0, cost_r=0.0, horizon=3)
    long = resolve(frame, signals, stop_atr=1.0, ratio=1.0, cost_r=0.0, horizon=48)
    assert 0 < len(short) < len(long)


def test_the_bar_rises_with_the_size_of_the_grid() -> None:
    """Sixteen gold candidates on two clocks is not a single hypothesis."""
    one = bonferroni_sigma(1)
    grid = bonferroni_sigma(32)
    twice = bonferroni_sigma(64)
    assert one == pytest.approx(1.96, abs=0.02)
    assert grid > 3.0
    assert twice > grid


def test_the_gold_family_is_selectable_and_separate() -> None:
    """The two families stay distinct, so a run pays only for what it tried."""
    assert set(FAMILIES) == {"index", "gold", "all"}
    assert set(FAMILIES["gold"]) == set(GOLD_CANDIDATES)
    assert not set(FAMILIES["index"]) & set(GOLD_CANDIDATES)
    assert set(FAMILIES["all"]) == set(FAMILIES["index"]) | set(GOLD_CANDIDATES)

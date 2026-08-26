"""OHLCV retrieval, validation and multi-timeframe assembly.

One invariant dominates this module: **the currently forming bar is never
returned.** Every downstream module may therefore use `df.iloc[-1]` freely.
Without that guarantee, an indicator reading the live bar's close would score a
setup on a price that had not happened yet — the classic look-ahead bug that
makes a backtest look brilliant and a live account bleed.

The second job here is refusing to serve bad data. Missing bars, duplicated
timestamps, non-monotonic time, zero or inverted OHLC — all of these produce a
loud error rather than a quietly wrong indicator value. "No data" must lead to
"no trade", never to "trade on a guess".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from config.schema import DataConfig
from core.broker import MarketDataProvider
from core.clock import Clock
from core.errors import DataIntegrityError, InsufficientDataError, StaleDataError
from core.types import MarketContext, Series, Tick, Timeframe
from infra.logging import get_logger

log = get_logger(__name__)

_COLUMNS = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]


@dataclass(slots=True)
class _CacheEntry:
    series: Series
    fetched_at: datetime


class DataManager:
    """Fetches, validates and caches bars for one or more symbols.

    Caching is time-based and conservative: within `cache_ttl_seconds` the same
    frame is returned, because refetching M5 bars four times inside one cycle
    costs latency and can hand different modules slightly different views of
    the same moment.
    """

    def __init__(self, connector: MarketDataProvider, config: DataConfig, clock: Clock) -> None:
        self.connector = connector
        self.config = config
        self.clock = clock
        self._cache: dict[tuple[str, Timeframe], _CacheEntry] = {}

    # -- public API ---------------------------------------------------------

    @property
    def timeframes(self) -> tuple[Timeframe, ...]:
        return tuple(Timeframe.parse(tf) for tf in self.config.timeframes)

    def get_series(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        *,
        count: int | None = None,
        force_refresh: bool = False,
    ) -> Series:
        """Closed bars for one symbol/timeframe, validated and cached."""
        tf = Timeframe.parse(timeframe)
        bars = count or self.config.bars.get(tf.value, self.config.minimum_bars_for(tf.value))
        key = (symbol, tf)

        cached = self._cache.get(key)
        if cached is not None and len(cached.series) >= bars:
            fresh_enough = self._cache_age(cached) < self.config.cache_ttl_seconds
            if fresh_enough and not force_refresh:
                return cached.series
            # Even a forced refresh is pointless when no new bar can have closed
            # since the cache was filled. A weekly bar closes once a week; a
            # daily one once a day. Refetching 400 weekly bars for every symbol
            # on every cycle cannot return anything different, and with seven
            # timeframes across a full catalogue that waste was most of the
            # cycle: eight minutes, of which the higher timeframes contributed
            # nothing but I/O.
            if not self._bar_closed_since(cached, tf):
                return cached.series

        series = self._fetch(symbol, tf, bars)
        self._cache[key] = _CacheEntry(series=series, fetched_at=self.clock.now())
        return series

    def get_context(
        self,
        symbol: str,
        *,
        timeframes: tuple[Timeframe, ...] | None = None,
        with_tick: bool = True,
        force_refresh: bool = False,
    ) -> MarketContext:
        """Assemble the full multi-timeframe view an analysis cycle works from.

        ONE TIMEFRAME USED TO VETO THE WHOLE MARKET, and it was a dict
        comprehension that did it:

            series = {tf: self.get_series(symbol, tf) for tf in wanted}

        Seven fetches, and the first one to raise took the other six with it.
        The symbol then went into quarantine, and quarantine backs off 30
        minutes, then two hours, then four. So a hole in one M5 feed, or a
        broker carrying 180 H4 bars where 200 are asked for, removed that
        market from the account for the rest of the session -- including the
        timeframes that were complete and the modules that only read those.

        The comment above `min_bars_by_timeframe` in the overlay already
        describes this happening once: COFFEE, STLAM and SPM were "thrown out
        as having no data at all -- losing the six timeframes that actually
        drive the decision over the one that only sets background context."
        The fix applied then was to lower W1's bar count, which cured those
        three symbols and left the mechanism intact for every other one.

        THE MECHANISM IS THE BUG. The ladder was validated as a BLOCK, before
        anything asked which parts of it the decision actually needed. So the
        question is moved to where it can be answered: a timeframe named in
        `required_timeframes` is still fatal, and one that is not is left OUT
        of the context and recorded.

        THIS IS NOT "TRADE ON LESS DATA". Every module states the timeframes it
        reads and returns a neutral signal naming them when one is absent --
        `candle_momentum` has done exactly that since it was written. A missing
        timeframe therefore removes the modules that depend on it from the vote
        rather than letting them guess, which is the same fail-closed rule as
        before, applied per module instead of per market. What changes is that
        a module reading only M5 and M15 is no longer silenced because W1 has a
        hole in it.

        `meta["unavailable_timeframes"]` carries what was dropped and why, so a
        decision taken on a partial ladder says so in the journal instead of
        looking identical to one taken on a complete one.
        """
        wanted = timeframes or self.timeframes
        required = self.required_timeframes
        series: dict[Timeframe, Series] = {}
        unavailable: dict[str, str] = {}
        for tf in wanted:
            try:
                series[tf] = self.get_series(symbol, tf, force_refresh=force_refresh)
            except (InsufficientDataError, DataIntegrityError) as exc:
                if not required or tf in required:
                    raise
                unavailable[tf.value] = str(exc).strip().split("\n")[0][:200]

        tick: Tick | None = None
        if with_tick:
            tick = self.connector.tick(symbol)

        meta: dict[str, object] = {}
        if unavailable:
            meta["unavailable_timeframes"] = unavailable
            log.debug(
                "analysing without some timeframes",
                extra={
                    "event": "partial_ladder",
                    "symbol": symbol,
                    "missing": sorted(unavailable),
                },
            )
        return MarketContext(
            symbol=symbol, now=self.clock.now(), series=series, tick=tick, meta=meta
        )

    @property
    def required_timeframes(self) -> frozenset[Timeframe]:
        """Timeframes whose absence still refuses the market outright.

        Empty means every timeframe is required, which is what this class did
        before the setting existed. A config that says nothing keeps the old
        behaviour rather than quietly loosening on upgrade.
        """
        return frozenset(Timeframe.parse(tf) for tf in self.config.required_timeframes)

    def invalidate(self, symbol: str | None = None) -> None:
        """Drop cached frames. Called after a reconnect, when data may have gaps."""
        if symbol is None:
            self._cache.clear()
            return
        for key in [k for k in self._cache if k[0] == symbol]:
            del self._cache[key]

    # -- fetching -----------------------------------------------------------

    def _fetch(self, symbol: str, tf: Timeframe, bars: int) -> Series:
        # +2 covers the forming bar plus one spare, so we still end up with
        # `bars` closed bars after trimming.
        raw = self.connector.copy_rates(symbol, tf.mt5_value, bars + 2, start_pos=0)
        df = self._to_frame(raw)
        df = self._drop_forming_bar(df, tf)

        required = self.config.minimum_bars_for(tf.value)
        if len(df) < required:
            raise InsufficientDataError(
                f"{symbol} {tf}: {len(df)} closed bars available, "
                f"{required} required. The broker may not offer "
                f"this much history for this symbol/timeframe."
            )

        df = df.iloc[-bars:] if len(df) > bars else df
        self._validate(symbol, tf, df)

        series = Series(symbol=symbol, timeframe=tf, df=df, fetched_at=self.clock.now())
        log.debug(
            "series loaded",
            extra={
                "event": "data_fetch",
                "symbol": symbol,
                "timeframe": tf.value,
                "bars": len(df),
                "first": df.index[0].isoformat(),
                "last": df.index[-1].isoformat(),
            },
        )
        return series

    @staticmethod
    def _to_frame(raw: Any) -> pd.DataFrame:
        """Convert MT5's structured array to a tz-aware, time-indexed frame."""
        df = pd.DataFrame(raw)
        if "time" not in df.columns:
            raise DataIntegrityError("rates array has no `time` column")
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        for column in _COLUMNS:
            if column not in df.columns:
                df[column] = 0
        return df[_COLUMNS].astype(
            {
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "tick_volume": "int64",
                "spread": "int64",
                "real_volume": "int64",
            }
        )

    def _drop_forming_bar(self, df: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
        """Remove any bar whose close time has not passed yet.

        Done by timestamp rather than by "drop the last row", because on a
        weekend or right after a reconnect the newest bar returned may already
        be closed, and blindly discarding it throws away real information.
        """
        now = self.clock.now()
        closes_at = df.index + tf.duration
        return df[closes_at <= now]

    # -- validation ---------------------------------------------------------

    def _validate(self, symbol: str, tf: Timeframe, df: pd.DataFrame) -> None:
        """Fail loudly on anything that would corrupt downstream analysis."""
        if df.index.has_duplicates:
            duplicates = df.index[df.index.duplicated()].tolist()[:5]
            raise DataIntegrityError(f"{symbol} {tf}: duplicate bar timestamps {duplicates}")
        if not df.index.is_monotonic_increasing:
            raise DataIntegrityError(f"{symbol} {tf}: bar timestamps are not increasing")

        prices = df[["open", "high", "low", "close"]]
        if prices.isna().to_numpy().any():
            raise DataIntegrityError(f"{symbol} {tf}: NaN in OHLC data")
        if (prices <= 0).to_numpy().any():
            raise DataIntegrityError(f"{symbol} {tf}: non-positive price in OHLC data")

        inverted = df["high"] < df["low"]
        if bool(inverted.any()):
            when = df.index[inverted][0]
            raise DataIntegrityError(f"{symbol} {tf}: high < low at {when}")

        out_of_range = (
            (df["open"] > df["high"])
            | (df["open"] < df["low"])
            | (df["close"] > df["high"])
            | (df["close"] < df["low"])
        )
        if bool(out_of_range.any()):
            when = df.index[out_of_range][0]
            raise DataIntegrityError(f"{symbol} {tf}: open/close outside high/low at {when}")

        self._check_gaps(symbol, tf, df)
        self._check_staleness(symbol, tf, df)

    def _check_gaps(self, symbol: str, tf: Timeframe, df: pd.DataFrame) -> None:
        """Warn on missing bars; abort if too many are missing.

        Weekend and holiday gaps are normal and are excluded — only gaps inside
        a trading week count. A broker feed with holes in it produces indicator
        values that cannot be reproduced in a backtest, so past a threshold the
        right move is to sit the cycle out.
        """
        if len(df) < 3 or tf in (Timeframe.W1, Timeframe.MN1):
            return

        missing_bars = _missing_bars(df, tf)
        expected = max(len(df) - 1, 1)
        fraction = float(missing_bars) / expected
        if fraction > self.config.max_gap_fraction:
            raise DataIntegrityError(
                f"{symbol} {tf}: {missing_bars:.0f} bars missing inside trading weeks "
                f"({fraction:.1%} of the window, limit {self.config.max_gap_fraction:.1%}). "
                f"Refusing to analyse an incomplete series."
            )
        if missing_bars > 0:
            log.debug(
                "gaps in series",
                extra={
                    "event": "data_gaps",
                    "symbol": symbol,
                    "timeframe": tf.value,
                    "missing_bars": int(missing_bars),
                    "fraction": round(fraction, 4),
                },
            )

    def _check_staleness(self, symbol: str, tf: Timeframe, df: pd.DataFrame) -> None:
        """Reject data whose newest closed bar is too old to act on.

        Age is counted from the moment the bar **closed**, and the weekend is
        subtracted. Both matter, and getting either wrong rejects the entire
        universe on a Monday morning while the feed is perfectly healthy.

        A bar is stamped with its *open* time, so a D1 bar carries information
        up to 24 hours newer than its timestamp suggests. Measuring from the
        stamp charges a full extra bar of age to every timeframe.

        And no data arrives between Friday 21:00 and Sunday 22:00 UTC, so wall
        clock is the wrong unit: on Monday the newest closed D1 bar is Friday's,
        which is not stale, it is the most recent bar that exists. What the
        budget is really asking is "how much *trading* time passed without new
        data", which is why the closed window is discounted.

        The budget also has to come from the instrument, not from the timeframe
        alone. `stale_after_bars * tf.duration` assumes bars arrive back to back,
        which is true of spot FX and false of everything with a session. BMED —
        a Milan share — closes overnight, so its newest H4 bar was legitimately
        14 hours old at 08:25 against a 12-hour budget, and the whole instrument
        was declared a dead feed. The instrument's own recent bar spacing says
        what "recent" means for it: the ninetieth percentile captures its normal
        overnight gap, and the budget is whichever is larger.

        This does not weaken disconnection detection. A dead terminal is caught
        by the short timeframes in the same context — on a Monday morning the
        newest M15 bar should be minutes old, and if it is Friday's the
        discounted age still blows through its budget by hours. And a feed that
        stops mid-session produces an age far beyond a gap the instrument has
        ever shown, because the percentile is measured from its own history.
        """
        now = self.clock.now()
        if is_market_closed(now):
            return

        closed_at = df.index[-1].to_pydatetime() + tf.duration
        if now <= closed_at:
            return

        elapsed = now - closed_at
        age = elapsed - market_closed_overlap(closed_at, now)
        # A quarter more than the instrument's normal overnight gap. The margin
        # is deliberately narrow and was calibrated against the real case: on a
        # 16-hour overnight close, 1.25 passes a healthy Monday morning at 11:25
        # of trading age and still catches a feed that missed all of Monday at
        # 21:00. At 1.5 the missed session slips through; applying
        # `stale_after_bars` to the gap would give a Milan share a two-day
        # budget, which is not a staleness check any more.
        budget = max(tf.duration * self.config.stale_after_bars, _typical_gap(df) * 1.25)
        if age > budget:
            raise StaleDataError(
                f"{symbol} {tf}: newest closed bar closed {age} ago in trading time "
                f"({elapsed} wall clock), budget {budget}. "
                f"The terminal is likely disconnected from the broker's data feed."
            )

    def _bar_closed_since(self, entry: _CacheEntry, tf: Timeframe) -> bool:
        """Could a newer closed bar exist than the one this cache entry holds?

        The cached series already ends at the newest bar that had closed when it
        was fetched. The next one closes exactly one bar-duration later, so if
        that moment is still in the future the broker has nothing newer to give.

        Refusing the refetch is safe in the direction that matters: it can only
        return data that is already the newest available, never older. It cannot
        introduce look-ahead either, because it never reveals an unclosed bar.
        """
        next_close = entry.series.last_bar_time + tf.duration * 2
        return self.clock.now() >= next_close

    def _cache_age(self, entry: _CacheEntry) -> float:
        return (self.clock.now() - entry.fetched_at).total_seconds()


def is_market_closed(moment: datetime) -> bool:
    """Rough FX market-closed check, in UTC.

    Deliberately approximate and only used to suppress false staleness alarms.
    The authoritative session logic lives in `filters/session_filter.py`
    (Phase 3); duplicating precise rules here would mean two sources of truth.

    Closed: Friday 21:00 UTC through Sunday 22:00 UTC.
    """
    moment = moment.astimezone(UTC)
    weekday = moment.weekday()  # Monday = 0
    if weekday == 5:  # Saturday
        return True
    if weekday == 4 and moment.hour >= 21:  # Friday evening
        return True
    return bool(weekday == 6 and moment.hour < 22)  # Sunday before the open


#: A gap width counted as the instrument's schedule rather than missing data
#: once it recurs on at least this share of the days in the window. Half is
#: comfortably above "happened a few times" and comfortably below "every
#: trading day", which is what a session break actually does.
_STRUCTURAL_GAP_SHARE = 0.5

#: Friday 21:00 UTC to Sunday 22:00 UTC — the same window `is_market_closed`
#: describes, as a duration.
_WEEKEND_CLOSURE = timedelta(hours=49)


def market_closed_overlap(start: datetime, end: datetime) -> timedelta:
    """How much of `[start, end]` fell inside the weekend closure.

    Staleness budgets are expressed in bar durations, which only pass while the
    market is open. Charging a timeframe for the weekend makes every Monday
    look like a data outage.

    Same approximation as `is_market_closed`, and the same reason: the
    authoritative session rules live in `filters/session_filter.py`, and a
    second precise implementation here would be a second source of truth.
    """
    start = start.astimezone(UTC)
    end = end.astimezone(UTC)
    if end <= start:
        return timedelta()

    total = timedelta()
    # Step back far enough to catch a closure that began before `start`.
    cursor = (start - _WEEKEND_CLOSURE).replace(hour=0, minute=0, second=0, microsecond=0)
    while cursor <= end:
        if cursor.weekday() == 4:  # Friday
            closure_start = cursor.replace(hour=21)
            overlap = min(end, closure_start + _WEEKEND_CLOSURE) - max(start, closure_start)
            total += max(timedelta(), overlap)
        cursor += timedelta(days=1)
    return total


def atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder's ATR over the last `period` bars, in price units.

    Lives here rather than in `analysis/indicators.py` because Phase 1 already
    needs it: stop buffers, staleness budgets and the startup guard's
    feasibility report are all expressed in ATR, and none of those are strategy.
    """
    if len(df) < period + 1:
        raise InsufficientDataError(f"ATR({period}) needs {period + 1} bars, got {len(df)}")

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    prev_close = df["close"].shift(1).to_numpy()

    true_range = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )[1:]

    # Wilder smoothing: seed with the simple mean, then recursive average.
    seed = float(true_range[:period].mean())
    value = seed
    for tr in true_range[period:]:
        value = (value * (period - 1) + float(tr)) / period
    return value


def _typical_gap(df: pd.DataFrame) -> timedelta:
    """The largest spacing this instrument routinely shows between bars.

    The ninetieth percentile rather than the maximum: the maximum is the
    weekend, which is already discounted separately, while the ninetieth
    catches the once-a-day overnight close that a share or a future has and
    spot FX does not.
    """
    if len(df) < 10:
        return timedelta(0)
    gaps = df.index.to_series().diff().dropna()
    # Exclude weekend-sized gaps; `market_closed_overlap` already handles those,
    # and letting them in here would inflate the budget by two days.
    intraweek = gaps[gaps < pd.Timedelta(days=2)]
    if intraweek.empty:
        return timedelta(0)
    return intraweek.quantile(0.90).to_pytimedelta()


def _missing_bars(df: pd.DataFrame, tf: Timeframe) -> float:
    """Bars absent from a series, judged against the instrument's own rhythm.

    **A gap that recurs is structure; a gap that happens once is loss.** That is
    the whole rule, and it is the only one that survived contact with a real
    catalogue.

    Two earlier attempts failed for instructive reasons. Assuming FX — anything
    wider than one bar is a hole — made WHEAT H1 report 415 missing bars, 20.8%
    against a 2% limit, every one of them the market being shut; whole asset
    classes came back as corrupt data. Grouping by calendar date and comparing
    against the median day then broke on sessions that straddle midnight, which
    split one trading day across two dates and produced a run of short "days".
    And a high percentile of gap widths cannot work either: a daily break is one
    gap in nineteen on H1 and one in two hundred on M5, so no fixed quantile
    catches it on both.

    Recurrence has none of those problems. A session break appears once per
    trading day, at the same width every time. A dropped bar appears once. So
    widths that show up on a large share of the days in the window are the
    instrument's schedule and cost nothing; everything else is counted.
    """
    if tf.duration >= timedelta(days=1):
        # Daily and above: absent days are holidays, not defects.
        return 0.0

    step = tf.duration
    gaps = df.index.to_series().diff().dropna()
    # Weekends are structure too, and `market_closed_overlap` already owns them.
    intraweek = gaps[gaps < pd.Timedelta(days=2)]
    oversized = intraweek[intraweek > pd.Timedelta(step)]
    if oversized.empty:
        return 0.0

    days = max(len(set(df.index.date)), 1)
    # Round to the timeframe so two bars of the same break count as one width.
    widths = (oversized / pd.Timedelta(step)).round()
    recurrence = widths.value_counts()
    structural = {
        width for width, count in recurrence.items() if count >= _STRUCTURAL_GAP_SHARE * days
    }

    missing = 0.0
    for width, count in recurrence.items():
        if width in structural:
            continue
        missing += float((width - 1) * count)
    return missing


def expected_bars_between(start: datetime, end: datetime, tf: Timeframe) -> int:
    """Bars an FX feed should contain between two instants, weekends removed.

    Used by the warm-up check and the backtester's data-coverage report.
    """
    if end <= start:
        return 0
    total = timedelta(0)
    cursor = start
    while cursor < end:
        step = min(tf.duration, end - cursor)
        if not is_market_closed(cursor):
            total += step
        cursor += step
    return int(total / tf.duration)

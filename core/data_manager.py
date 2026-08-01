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
from core.clock import Clock
from core.errors import DataIntegrityError, InsufficientDataError, StaleDataError
from core.mt5_connector import MT5Connector
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

    def __init__(self, connector: MT5Connector, config: DataConfig, clock: Clock) -> None:
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
        bars = count or self.config.bars.get(tf.value, self.config.min_bars_required)
        key = (symbol, tf)

        cached = self._cache.get(key)
        if (
            cached is not None
            and not force_refresh
            and len(cached.series) >= bars
            and self._cache_age(cached) < self.config.cache_ttl_seconds
        ):
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
        """Assemble the full multi-timeframe view an analysis cycle works from."""
        wanted = timeframes or self.timeframes
        series = {tf: self.get_series(symbol, tf, force_refresh=force_refresh) for tf in wanted}

        tick: Tick | None = None
        if with_tick:
            tick = self.connector.tick(symbol)

        return MarketContext(symbol=symbol, now=self.clock.now(), series=series, tick=tick)

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

        if len(df) < self.config.min_bars_required:
            raise InsufficientDataError(
                f"{symbol} {tf}: {len(df)} closed bars available, "
                f"{self.config.min_bars_required} required. The broker may not offer "
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

        deltas = df.index.to_series().diff().dropna()
        step = tf.duration
        # A weekend on an FX feed shows up as a ~48-65h gap on intraday
        # timeframes; anything at or above 2 days is treated as a session break.
        session_break = pd.Timedelta(days=2)
        intraweek = deltas[deltas < session_break]
        missing_bars = ((intraweek - step) / step).clip(lower=0).sum()

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

        Skipped while the FX market is closed — a stale D1 bar at 03:00 on a
        Sunday is expected, not a fault.
        """
        now = self.clock.now()
        if is_market_closed(now):
            return

        age = now - df.index[-1].to_pydatetime()
        budget = tf.duration * self.config.stale_after_bars
        if age > budget:
            raise StaleDataError(
                f"{symbol} {tf}: newest closed bar is {age} old, budget {budget}. "
                f"The terminal is likely disconnected from the broker's data feed."
            )

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

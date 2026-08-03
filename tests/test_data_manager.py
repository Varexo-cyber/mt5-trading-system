"""Data manager: the look-ahead guarantee and the data-quality gates."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from config.schema import DataConfig, MT5Config
from core.clock import SimulatedClock
from core.data_manager import (
    DataManager,
    atr,
    expected_bars_between,
    is_market_closed,
    market_closed_overlap,
)
from core.errors import DataIntegrityError, InsufficientDataError, StaleDataError
from core.mt5_connector import MT5Connector
from core.types import Timeframe
from tests.fakes.fake_mt5 import FakeMT5, synthetic_rates

# A Wednesday, mid-London session — market definitely open.
NOW = datetime(2026, 3, 11, 14, 30, tzinfo=UTC)


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(NOW)


@pytest.fixture
def data_config() -> DataConfig:
    return DataConfig(
        timeframes=("H1", "M15"),
        bars={"H1": 300, "M15": 300},
        min_bars_required=50,
        cache_ttl_seconds=20.0,
    )


@pytest.fixture
def manager(clock: SimulatedClock, data_config: DataConfig) -> DataManager:
    fake = FakeMT5(now=NOW)
    connector = MT5Connector(MT5Config(), mt5_module=fake)
    connector.connect()
    return DataManager(connector, data_config, clock)


class TestNoLookAhead:
    """The single most important property in the whole data layer."""

    def test_forming_bar_is_never_returned(
        self, manager: DataManager, clock: SimulatedClock
    ) -> None:
        series = manager.get_series("EURUSD", Timeframe.H1)
        last_close_time = series.last_bar_time + Timeframe.H1.duration
        assert last_close_time <= clock.now()

    def test_every_returned_bar_is_closed(self, manager: DataManager) -> None:
        series = manager.get_series("EURUSD", Timeframe.M15)
        closes = series.df.index + Timeframe.M15.duration
        assert (closes <= NOW).all()

    def test_advancing_the_clock_reveals_the_next_bar(
        self, manager: DataManager, clock: SimulatedClock
    ) -> None:
        first = manager.get_series("EURUSD", Timeframe.M15).last_bar_time
        clock.advance(timedelta(minutes=30))
        second = manager.get_series("EURUSD", Timeframe.M15, force_refresh=True).last_bar_time
        assert second > first


class TestCaching:
    def test_repeat_calls_inside_the_ttl_hit_the_cache(
        self, manager: DataManager, clock: SimulatedClock
    ) -> None:
        fake = manager.connector._mt5
        assert isinstance(fake, FakeMT5)
        manager.get_series("EURUSD", Timeframe.H1)
        count = sum(1 for name, _ in fake.calls if name == "copy_rates_from_pos")

        clock.advance(timedelta(seconds=5))
        manager.get_series("EURUSD", Timeframe.H1)
        assert sum(1 for name, _ in fake.calls if name == "copy_rates_from_pos") == count

    def test_cache_expires(self, manager: DataManager, clock: SimulatedClock) -> None:
        fake = manager.connector._mt5
        assert isinstance(fake, FakeMT5)
        manager.get_series("EURUSD", Timeframe.H1)
        count = sum(1 for name, _ in fake.calls if name == "copy_rates_from_pos")

        clock.advance(timedelta(seconds=60))
        manager.get_series("EURUSD", Timeframe.H1)
        assert sum(1 for name, _ in fake.calls if name == "copy_rates_from_pos") > count

    def test_invalidate_drops_one_symbol_only(self, manager: DataManager) -> None:
        manager.get_series("EURUSD", Timeframe.H1)
        manager.get_series("USDJPY", Timeframe.H1)
        manager.invalidate("EURUSD")
        assert ("USDJPY", Timeframe.H1) in manager._cache
        assert ("EURUSD", Timeframe.H1) not in manager._cache


class TestContext:
    def test_context_carries_every_configured_timeframe(self, manager: DataManager) -> None:
        ctx = manager.get_context("EURUSD")
        assert set(ctx.series) == {Timeframe.H1, Timeframe.M15}
        assert ctx.tick is not None
        assert ctx.now == NOW

    def test_missing_timeframe_raises_a_useful_error(self, manager: DataManager) -> None:
        ctx = manager.get_context("EURUSD")
        with pytest.raises(KeyError, match="D1"):
            ctx.bars(Timeframe.D1)


class TestValidation:
    def _manager_returning(
        self, rates: np.ndarray, clock: SimulatedClock, config: DataConfig
    ) -> DataManager:
        fake = FakeMT5()
        fake.copy_rates_from_pos = lambda *a, **k: rates  # type: ignore[method-assign]
        connector = MT5Connector(MT5Config(), mt5_module=fake)
        connector.connect()
        return DataManager(connector, config, clock)

    def test_too_few_bars_is_an_error(self, clock: SimulatedClock, data_config: DataConfig) -> None:
        rates = synthetic_rates(10, Timeframe.H1.mt5_value, end=NOW)
        manager = self._manager_returning(rates, clock, data_config)
        with pytest.raises(InsufficientDataError, match="closed bars available"):
            manager.get_series("EURUSD", Timeframe.H1)

    def test_inverted_high_low_is_an_error(
        self, clock: SimulatedClock, data_config: DataConfig
    ) -> None:
        rates = synthetic_rates(200, Timeframe.H1.mt5_value, end=NOW).copy()
        rates["high"][50], rates["low"][50] = rates["low"][50], rates["high"][50]
        manager = self._manager_returning(rates, clock, data_config)
        with pytest.raises(DataIntegrityError, match="high < low"):
            manager.get_series("EURUSD", Timeframe.H1)

    def test_close_outside_the_bar_range_is_an_error(
        self, clock: SimulatedClock, data_config: DataConfig
    ) -> None:
        rates = synthetic_rates(200, Timeframe.H1.mt5_value, end=NOW).copy()
        rates["close"][30] = rates["high"][30] * 1.01
        manager = self._manager_returning(rates, clock, data_config)
        with pytest.raises(DataIntegrityError, match="outside high/low"):
            manager.get_series("EURUSD", Timeframe.H1)

    def test_non_positive_price_is_an_error(
        self, clock: SimulatedClock, data_config: DataConfig
    ) -> None:
        rates = synthetic_rates(200, Timeframe.H1.mt5_value, end=NOW).copy()
        rates["low"][10] = 0.0
        manager = self._manager_returning(rates, clock, data_config)
        with pytest.raises(DataIntegrityError, match="non-positive price"):
            manager.get_series("EURUSD", Timeframe.H1)

    def test_stale_feed_is_refused_during_market_hours(
        self, clock: SimulatedClock, data_config: DataConfig
    ) -> None:
        # Bars end two days before "now" on a Wednesday: the feed is dead.
        rates = synthetic_rates(200, Timeframe.H1.mt5_value, end=NOW - timedelta(days=2))
        manager = self._manager_returning(rates, clock, data_config)
        with pytest.raises(StaleDataError, match="disconnected"):
            manager.get_series("EURUSD", Timeframe.H1)

    def test_stale_feed_is_tolerated_over_the_weekend(self, data_config: DataConfig) -> None:
        saturday = datetime(2026, 3, 14, 10, 0, tzinfo=UTC)
        clock = SimulatedClock(saturday)
        rates = synthetic_rates(
            200, Timeframe.H1.mt5_value, end=datetime(2026, 3, 13, 20, 0, tzinfo=UTC)
        )
        manager = self._manager_returning(rates, clock, data_config)
        series = manager.get_series("EURUSD", Timeframe.H1)
        assert len(series) > 0

    def test_mondays_newest_daily_bar_is_not_stale(self, data_config: DataConfig) -> None:
        """Regression: every market was rejected on the first live Monday.

        A bar is stamped with its *open* time, so a D1 bar carries information a
        full day newer than its stamp. Add the weekend, and Friday's daily bar
        measured 3d10h on Monday morning against a 3-day budget — the whole
        universe came back DATA_UNAVAILABLE while the feed was perfectly fine.
        """
        monday = datetime(2026, 8, 3, 7, 37, 48, tzinfo=UTC)
        clock = SimulatedClock(monday)
        # The last closed D1 bar opens Friday 00:00 broker time (UTC+3).
        last_open = datetime(2026, 7, 30, 21, 0, tzinfo=UTC)
        rates = synthetic_rates(300, Timeframe.D1.mt5_value, end=last_open)
        manager = self._manager_returning(rates, clock, data_config)

        assert len(manager.get_series("EURUSD", Timeframe.D1)) > 0

    def test_a_dead_terminal_on_monday_is_still_caught(self, data_config: DataConfig) -> None:
        """Discounting the weekend must not hide a feed that stopped on Friday."""
        monday = datetime(2026, 8, 3, 7, 37, 48, tzinfo=UTC)
        clock = SimulatedClock(monday)
        last_open = datetime(2026, 7, 31, 20, 0, tzinfo=UTC)  # Friday, hours before the close
        rates = synthetic_rates(300, Timeframe.H1.mt5_value, end=last_open)
        manager = self._manager_returning(rates, clock, data_config)

        with pytest.raises(StaleDataError, match="disconnected"):
            manager.get_series("EURUSD", Timeframe.H1)

    def test_weekend_overlap_measures_the_closed_window(self) -> None:
        friday_close = datetime(2026, 7, 31, 21, 0, tzinfo=UTC)
        sunday_open = datetime(2026, 8, 2, 22, 0, tzinfo=UTC)

        # The whole closure, from a point before it to a point after it.
        assert market_closed_overlap(
            friday_close - timedelta(hours=2), sunday_open + timedelta(hours=2)
        ) == timedelta(hours=49)
        # Monday morning: only the part of the weekend that has already passed.
        assert market_closed_overlap(
            friday_close, datetime(2026, 8, 3, 7, 37, 48, tzinfo=UTC)
        ) == timedelta(days=2, hours=1)
        # A window entirely inside the trading week touches no closure.
        assert (
            market_closed_overlap(
                datetime(2026, 8, 4, 8, 0, tzinfo=UTC), datetime(2026, 8, 5, 8, 0, tzinfo=UTC)
            )
            == timedelta()
        )
        assert market_closed_overlap(sunday_open, friday_close) == timedelta()

    def test_an_overnight_close_is_not_a_dead_feed(self, data_config: DataConfig) -> None:
        """Regression: BMED, a Milan share, was declared a disconnected feed.

        "newest closed bar closed 14:25:02 ago in trading time, budget
        12:00:00" — a share shuts overnight, so on H4 its newest bar is
        routinely more than twelve hours old. The budget came from the timeframe
        alone, which assumes bars arrive back to back. That is spot FX, not a
        broker catalogue.
        """
        monday = datetime(2026, 8, 3, 8, 25, tzinfo=UTC)
        clock = SimulatedClock(monday)
        config = DataConfig(
            timeframes=("H4",), bars={"H4": 300}, min_bars_required=20, cache_ttl_seconds=20.0
        )
        # Three H4 bars a day on weekdays, then a sixteen-hour overnight gap.
        times: list[datetime] = []
        day = datetime(2026, 7, 1, tzinfo=UTC)
        while day < monday:
            if day.weekday() < 5:
                times.extend(day.replace(hour=hour) for hour in (8, 12, 16))
            day += timedelta(days=1)
        rates = _rates_at(times)
        manager = self._manager_returning(rates, clock, config)

        assert len(manager.get_series("EURUSD", Timeframe.H4)) > 0

    def test_a_share_whose_feed_died_is_still_caught(self, data_config: DataConfig) -> None:
        """Tolerating one overnight gap must not tolerate a missing session.

        Monday evening with Friday's last bar: the exchange opened and this
        instrument produced nothing all day. That is a dead feed, and the
        margin over the overnight gap has to be narrow enough to say so.
        """
        monday_evening = datetime(2026, 8, 3, 18, 0, tzinfo=UTC)
        clock = SimulatedClock(monday_evening)
        config = DataConfig(
            timeframes=("H4",), bars={"H4": 300}, min_bars_required=20, cache_ttl_seconds=20.0
        )
        times: list[datetime] = []
        day = datetime(2026, 7, 1, tzinfo=UTC)
        while day <= datetime(2026, 7, 31, tzinfo=UTC):  # last bar is Friday's
            if day.weekday() < 5:
                times.extend(day.replace(hour=hour) for hour in (8, 12, 16))
            day += timedelta(days=1)
        rates = _rates_at(times)
        manager = self._manager_returning(rates, clock, config)

        with pytest.raises(StaleDataError, match="disconnected"):
            manager.get_series("EURUSD", Timeframe.H4)

    def test_excessive_intraweek_gaps_are_refused(
        self, clock: SimulatedClock, data_config: DataConfig
    ) -> None:
        rates = synthetic_rates(200, Timeframe.H1.mt5_value, end=NOW)
        # Punch a 20-hour hole in the middle of the week.
        keep = np.ones(len(rates), dtype=bool)
        keep[100:120] = False
        manager = self._manager_returning(rates[keep], clock, data_config)
        with pytest.raises(DataIntegrityError, match="bars missing"):
            manager.get_series("EURUSD", Timeframe.H1)


class TestMarketClosed:
    @pytest.mark.parametrize(
        ("moment", "closed"),
        [
            (datetime(2026, 3, 11, 14, 0, tzinfo=UTC), False),  # Wednesday
            (datetime(2026, 3, 13, 20, 0, tzinfo=UTC), False),  # Friday 20:00
            (datetime(2026, 3, 13, 21, 30, tzinfo=UTC), True),  # Friday close
            (datetime(2026, 3, 14, 12, 0, tzinfo=UTC), True),  # Saturday
            (datetime(2026, 3, 15, 12, 0, tzinfo=UTC), True),  # Sunday morning
            (datetime(2026, 3, 15, 23, 0, tzinfo=UTC), False),  # Sunday reopen
        ],
    )
    def test_weekend_window(self, moment: datetime, closed: bool) -> None:
        assert is_market_closed(moment) is closed


class TestATR:
    def test_constant_range_gives_that_range(self) -> None:
        df = pd.DataFrame(
            {
                "high": [1.1000] * 30,
                "low": [1.0990] * 30,
                "close": [1.0995] * 30,
                "open": [1.0995] * 30,
            }
        )
        assert atr(df, period=14) == pytest.approx(0.0010, abs=1e-9)

    def test_gap_counts_as_true_range(self) -> None:
        df = pd.DataFrame(
            {
                "high": [1.1000] * 20 + [1.2000],
                "low": [1.0990] * 20 + [1.1990],
                "close": [1.0995] * 20 + [1.1995],
                "open": [1.0995] * 20 + [1.1995],
            }
        )
        # The 0.10 gap must lift the average well above the 0.0010 bar range.
        assert atr(df, period=14) > 0.0060

    def test_too_little_data_raises(self) -> None:
        df = pd.DataFrame({"high": [1.1] * 5, "low": [1.0] * 5, "close": [1.05] * 5})
        with pytest.raises(InsufficientDataError):
            atr(df, period=14)


class TestExpectedBars:
    def test_weekend_is_excluded(self) -> None:
        friday = datetime(2026, 3, 13, 12, 0, tzinfo=UTC)
        monday = datetime(2026, 3, 16, 12, 0, tzinfo=UTC)
        full = int((monday - friday) / Timeframe.H1.duration)
        assert expected_bars_between(friday, monday, Timeframe.H1) < full

    def test_reversed_range_is_zero(self) -> None:
        assert expected_bars_between(NOW, NOW - timedelta(hours=1), Timeframe.H1) == 0


def _rates_at(times: list[datetime]) -> np.ndarray:
    """Well-formed bars at exactly these open times, for session-shaped series."""
    dtype = [
        ("time", "i8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("tick_volume", "u8"),
        ("spread", "i4"),
        ("real_volume", "u8"),
    ]
    rows = [
        (int(moment.timestamp()), 10.0, 10.2, 9.8, 10.1, 500, 12, 0) for moment in sorted(times)
    ]
    return np.array(rows, dtype=dtype)

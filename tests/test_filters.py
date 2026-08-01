"""Session, spread and correlation filters, plus the chain that runs them."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.schema import (
    CorrelationFilterConfig,
    SessionFilterConfig,
    SpreadFilterConfig,
)
from core.clock import SimulatedClock
from core.instrument import InstrumentSpec
from core.types import Direction, Position, Series, Tick, Timeframe
from filters.base import Filter, FilterChain, FilterContext, FilterVerdict
from filters.correlation_filter import CorrelationFilter
from filters.session_filter import SessionFilter
from filters.spread_filter import SpreadFilter
from journal.database import Journal
from risk.reasons import Reason
from tests.fakes.fake_mt5 import eurusd_spec

# Wednesday, London/NY overlap.
NOW = datetime(2026, 3, 11, 14, 30, tzinfo=UTC)


@pytest.fixture
def spec() -> InstrumentSpec:
    return InstrumentSpec.from_mt5(eurusd_spec())


def context(
    spec: InstrumentSpec,
    *,
    now: datetime = NOW,
    spread: float = 0.00012,
    direction: Direction = Direction.LONG,
    positions: tuple[Position, ...] = (),
) -> FilterContext:
    return FilterContext(
        symbol=spec.symbol,
        spec=spec,
        now=now,
        direction=direction,
        tick=Tick(symbol=spec.symbol, time=now, bid=1.08500, ask=1.08500 + spread),
        open_positions=positions,
    )


def position(symbol: str, direction: Direction = Direction.LONG, ticket: int = 1) -> Position:
    return Position(
        ticket=ticket,
        symbol=symbol,
        direction=direction,
        volume=0.1,
        price_open=1.085,
        sl=1.083,
        tp=1.091,
        profit=0.0,
        swap=0.0,
        opened_at=NOW,
    )


# ------------------------------------------------------------------ session ---


class TestSessionFilter:
    @pytest.fixture
    def filter_(self) -> SessionFilter:
        return SessionFilter(SessionFilterConfig())

    def test_london_ny_overlap_is_tradable(
        self, filter_: SessionFilter, spec: InstrumentSpec
    ) -> None:
        verdict = filter_.check(context(spec))
        assert verdict.passed
        assert verdict.data["session"] == "london+newyork"
        assert verdict.data["session_overlap"] is True

    def test_asia_only_is_blocked_by_default(
        self, filter_: SessionFilter, spec: InstrumentSpec
    ) -> None:
        # 03:00 Wednesday: Asia is open, London and New York are not.
        moment = datetime(2026, 3, 11, 3, 0, tzinfo=UTC)
        verdict = filter_.check(context(spec, now=moment))
        assert not verdict.passed
        assert verdict.reason is Reason.OUTSIDE_TRADABLE_SESSION
        assert verdict.data["session"] == "asia"

    def test_london_alone_is_tradable(self, filter_: SessionFilter, spec: InstrumentSpec) -> None:
        moment = datetime(2026, 3, 11, 9, 0, tzinfo=UTC)
        verdict = filter_.check(context(spec, now=moment))
        assert verdict.passed
        assert verdict.data["session_overlap"] is False

    def test_rollover_window_is_blocked(self, filter_: SessionFilter, spec: InstrumentSpec) -> None:
        moment = datetime(2026, 3, 11, 21, 0, tzinfo=UTC)
        verdict = filter_.check(context(spec, now=moment))
        assert not verdict.passed
        assert verdict.reason is Reason.ROLLOVER_WINDOW

    def test_friday_evening_is_blocked(self, filter_: SessionFilter, spec: InstrumentSpec) -> None:
        moment = datetime(2026, 3, 13, 19, 30, tzinfo=UTC)  # Friday
        verdict = filter_.check(context(spec, now=moment))
        assert not verdict.passed
        assert verdict.reason is Reason.WEEKEND_EDGE
        assert "gap risk" in verdict.detail

    def test_friday_afternoon_is_fine(self, filter_: SessionFilter, spec: InstrumentSpec) -> None:
        moment = datetime(2026, 3, 13, 14, 0, tzinfo=UTC)
        assert filter_.check(context(spec, now=moment)).passed

    def test_saturday_is_market_closed(self, filter_: SessionFilter, spec: InstrumentSpec) -> None:
        moment = datetime(2026, 3, 14, 12, 0, tzinfo=UTC)
        verdict = filter_.check(context(spec, now=moment))
        assert verdict.reason is Reason.MARKET_CLOSED

    def test_sunday_before_reopen_is_market_closed(
        self, filter_: SessionFilter, spec: InstrumentSpec
    ) -> None:
        moment = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
        assert filter_.check(context(spec, now=moment)).reason is Reason.MARKET_CLOSED

    def test_wrapping_session_window(self, spec: InstrumentSpec) -> None:
        """A session crossing midnight must not silently become empty."""
        config = SessionFilterConfig(
            sessions={"overnight": ("22:00", "06:00")},
            tradable_sessions=("overnight",),
            rollover_block=("20:45", "21:15"),
            block_friday_after=None,
            block_sunday_before=None,
        )
        filter_ = SessionFilter(config)
        assert filter_.check(context(spec, now=datetime(2026, 3, 11, 23, 0, tzinfo=UTC))).passed
        assert filter_.check(context(spec, now=datetime(2026, 3, 11, 2, 0, tzinfo=UTC))).passed
        assert not filter_.check(context(spec, now=NOW)).passed

    def test_unknown_tradable_session_is_a_config_error(self) -> None:
        with pytest.raises(ValueError, match="no such session"):
            SessionFilter(SessionFilterConfig(tradable_sessions=("tokyo",)))

    def test_disabled_still_reports_the_session(self, spec: InstrumentSpec) -> None:
        filter_ = SessionFilter(SessionFilterConfig(enabled=False))
        verdict = filter_.check(context(spec, now=datetime(2026, 3, 11, 3, 0, tzinfo=UTC)))
        assert verdict.passed
        assert verdict.data["session"] == "asia"


# ------------------------------------------------------------------- spread ---


class TestSpreadFilter:
    @pytest.fixture
    def clock(self) -> SimulatedClock:
        return SimulatedClock(NOW)

    @pytest.fixture
    def journal(self, tmp_path: Path, clock: SimulatedClock) -> Journal:
        with Journal(tmp_path / "j.db", clock) as j:
            yield j

    @pytest.fixture
    def config(self) -> SpreadFilterConfig:
        return SpreadFilterConfig(
            min_observations=20, absolute_max_pips={"EURUSD": 2.0}, max_spread_multiple=2.0
        )

    @pytest.fixture
    def filter_(
        self, config: SpreadFilterConfig, journal: Journal, clock: SimulatedClock
    ) -> SpreadFilter:
        return SpreadFilter(config, journal, clock)

    def test_fallback_applies_before_the_baseline_exists(
        self, filter_: SpreadFilter, spec: InstrumentSpec
    ) -> None:
        verdict = filter_.check(context(spec, spread=0.00012))  # 1.2 pips vs 2.0 ceiling
        assert verdict.passed
        assert verdict.data["spread_baseline_source"] == "fallback"

    def test_fallback_blocks_a_wide_spread(
        self, filter_: SpreadFilter, spec: InstrumentSpec
    ) -> None:
        verdict = filter_.check(context(spec, spread=0.00030))  # 3.0 pips
        assert not verdict.passed
        assert verdict.reason is Reason.SPREAD_TOO_WIDE

    def test_unknown_symbol_without_a_fallback_is_refused(
        self, config: SpreadFilterConfig, journal: Journal, clock: SimulatedClock
    ) -> None:
        """No baseline and no configured ceiling means refuse, not guess."""
        odd = InstrumentSpec.from_mt5(eurusd_spec(name="EXOTIC"))
        verdict = SpreadFilter(config, journal, clock).check(context(odd))
        assert not verdict.passed
        assert "Refusing rather than guessing" in verdict.detail

    @staticmethod
    def _seed(filter_: SpreadFilter, values: list[float], hour: int = 14) -> None:
        """Insert observations inside a single UTC hour, spaced past the throttle."""
        base = NOW.replace(hour=hour, minute=0, second=0, microsecond=0)
        for index, value in enumerate(values):
            filter_.observe("EURUSD", value, base + timedelta(minutes=index))

    def test_learned_baseline_takes_over(self, filter_: SpreadFilter, spec: InstrumentSpec) -> None:
        self._seed(filter_, [0.8] * 25)

        median, count = filter_.baseline("EURUSD", 14)
        assert median == pytest.approx(0.8)
        assert count >= 20

        limit, source, _ = filter_.ceiling("EURUSD", 14)
        assert source == "learned"
        assert limit == pytest.approx(1.6)  # 0.8 median x 2.0 multiple

    def test_learned_baseline_blocks_between_fallback_and_median(
        self, filter_: SpreadFilter, spec: InstrumentSpec
    ) -> None:
        """1.8 pips passes the 2.0 fallback but fails a learned 1.6 ceiling."""
        self._seed(filter_, [0.8] * 25)

        verdict = filter_.check(context(spec, spread=0.00018))
        assert not verdict.passed
        assert verdict.data["spread_baseline_source"] == "learned"

    def test_median_ignores_an_outlier(self, filter_: SpreadFilter) -> None:
        """One 40-pip rollover print must not lift the whole day's ceiling."""
        self._seed(filter_, [*([0.8] * 24), 40.0])

        median, _ = filter_.baseline("EURUSD", 14)
        assert median == pytest.approx(0.8)

    def test_baselines_are_per_hour(self, filter_: SpreadFilter) -> None:
        """EURUSD at 14:00 and EURUSD at 22:00 are different instruments here."""
        self._seed(filter_, [0.8] * 25, hour=14)
        assert filter_.baseline("EURUSD", 14)[0] is not None
        assert filter_.baseline("EURUSD", 22)[0] is None

    def test_observations_are_throttled(self, filter_: SpreadFilter, clock: SimulatedClock) -> None:
        assert filter_.observe("EURUSD", 0.8, clock.now())
        assert not filter_.observe("EURUSD", 0.9, clock.now() + timedelta(seconds=10))
        assert filter_.observe("EURUSD", 0.9, clock.now() + timedelta(minutes=2))

    def test_missing_tick_blocks(self, filter_: SpreadFilter, spec: InstrumentSpec) -> None:
        """Cannot measure the spread means cannot clear it."""
        ctx = FilterContext(symbol="EURUSD", spec=spec, now=NOW, tick=None)
        assert not filter_.check(ctx).passed

    def test_broker_suffix_finds_the_fallback(
        self, config: SpreadFilterConfig, journal: Journal, clock: SimulatedClock
    ) -> None:
        suffixed = InstrumentSpec.from_mt5(eurusd_spec(name="EURUSD.pro"))
        verdict = SpreadFilter(config, journal, clock).check(context(suffixed, spread=0.00012))
        assert verdict.passed
        assert verdict.data["spread_baseline_source"] == "fallback"


# -------------------------------------------------------------- correlation ---


def make_series(symbol: str, closes: np.ndarray, start: datetime = NOW) -> Series:
    index = pd.date_range(end=start, periods=len(closes), freq="1h", tz=UTC)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "tick_volume": np.full(len(closes), 100, dtype="int64"),
            "spread": np.full(len(closes), 10, dtype="int64"),
            "real_volume": np.zeros(len(closes), dtype="int64"),
        },
        index=index,
    )
    return Series(symbol=symbol, timeframe=Timeframe.H1, df=df, fetched_at=start)


class TestCorrelationFilter:
    @pytest.fixture
    def config(self) -> CorrelationFilterConfig:
        return CorrelationFilterConfig(lookback_bars=200, max_abs_correlation=0.7)

    def _filter(
        self, config: CorrelationFilterConfig, series: dict[str, np.ndarray]
    ) -> CorrelationFilter:
        def provider(symbol: str, timeframe: Timeframe) -> Series:
            if symbol not in series:
                raise KeyError(symbol)
            return make_series(symbol, series[symbol])

        return CorrelationFilter(config, provider)

    @staticmethod
    def _walk(seed: int, n: int = 250, base: float = 1.085) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return base + np.cumsum(rng.normal(0, base * 0.0008, n))

    def test_no_open_positions_passes(
        self, config: CorrelationFilterConfig, spec: InstrumentSpec
    ) -> None:
        assert self._filter(config, {}).check(context(spec)).passed

    def test_highly_correlated_same_direction_blocks(
        self, config: CorrelationFilterConfig, spec: InstrumentSpec
    ) -> None:
        shared = self._walk(1)
        filter_ = self._filter(
            config, {"EURUSD": shared, "GBPUSD": shared * 1.2 + 0.1}  # near-perfect correlation
        )
        verdict = filter_.check(
            context(spec, direction=Direction.LONG, positions=(position("GBPUSD"),))
        )
        assert not verdict.passed
        assert verdict.reason is Reason.CORRELATED_EXPOSURE
        assert "double size" in verdict.detail

    def test_highly_correlated_opposite_direction_passes(
        self, config: CorrelationFilterConfig, spec: InstrumentSpec
    ) -> None:
        """Long EURUSD against short GBPUSD is a spread trade, not doubled risk."""
        shared = self._walk(1)
        filter_ = self._filter(config, {"EURUSD": shared, "GBPUSD": shared * 1.2 + 0.1})
        verdict = filter_.check(
            context(
                spec,
                direction=Direction.LONG,
                positions=(position("GBPUSD", Direction.SHORT),),
            )
        )
        assert verdict.passed

    def test_anticorrelated_opposite_directions_block(
        self, config: CorrelationFilterConfig, spec: InstrumentSpec
    ) -> None:
        """Long EURUSD and short USDCHF are the same short-dollar bet."""
        shared = self._walk(2)
        mirrored = 2 * float(shared[0]) - shared  # perfectly negative correlation
        filter_ = self._filter(config, {"EURUSD": shared, "USDCHF": mirrored})
        verdict = filter_.check(
            context(
                spec,
                direction=Direction.LONG,
                positions=(position("USDCHF", Direction.SHORT),),
            )
        )
        assert not verdict.passed

    def test_uncorrelated_pairs_pass(
        self, config: CorrelationFilterConfig, spec: InstrumentSpec
    ) -> None:
        filter_ = self._filter(
            config, {"EURUSD": self._walk(1), "AUDUSD": self._walk(99, base=0.65)}
        )
        verdict = filter_.check(context(spec, positions=(position("AUDUSD"),)))
        assert verdict.passed
        assert abs(verdict.data["correlations"]["AUDUSD"]) < 0.7

    def test_unmeasurable_correlation_blocks(
        self, config: CorrelationFilterConfig, spec: InstrumentSpec
    ) -> None:
        """Unknown is not the same as independent."""
        filter_ = self._filter(config, {"EURUSD": self._walk(1)})  # GBPUSD missing
        verdict = filter_.check(context(spec, positions=(position("GBPUSD"),)))
        assert not verdict.passed
        assert "unknown is not" in verdict.detail

    def test_flat_series_is_unmeasurable_not_uncorrelated(
        self, config: CorrelationFilterConfig, spec: InstrumentSpec
    ) -> None:
        flat = np.full(250, 1.085)
        filter_ = self._filter(config, {"EURUSD": self._walk(1), "GBPUSD": flat})
        assert not filter_.check(context(spec, positions=(position("GBPUSD"),))).passed

    def test_same_symbol_is_skipped(
        self, config: CorrelationFilterConfig, spec: InstrumentSpec
    ) -> None:
        """The risk manager owns that case; nothing to correlate here."""
        filter_ = self._filter(config, {"EURUSD": self._walk(1)})
        assert filter_.check(context(spec, positions=(position("EURUSD"),))).passed

    def test_direction_is_required(
        self, config: CorrelationFilterConfig, spec: InstrumentSpec
    ) -> None:
        filter_ = self._filter(config, {"EURUSD": self._walk(1)})
        ctx = FilterContext(
            symbol="EURUSD",
            spec=spec,
            now=NOW,
            direction=None,
            open_positions=(position("GBPUSD"),),
        )
        with pytest.raises(ValueError, match="needs the intended direction"):
            filter_.check(ctx)

    def test_exposure_maths(self) -> None:
        overlap = CorrelationFilter.doubles_exposure
        assert overlap(0.9, Direction.LONG, Direction.LONG) == pytest.approx(0.9)
        assert overlap(0.9, Direction.LONG, Direction.SHORT) == pytest.approx(-0.9)
        assert overlap(-0.9, Direction.LONG, Direction.SHORT) == pytest.approx(0.9)
        assert overlap(-0.9, Direction.LONG, Direction.LONG) == pytest.approx(-0.9)


# -------------------------------------------------------------------- chain ---


class _Stub(Filter):
    def __init__(self, name: str, passes: bool, **data: object) -> None:
        self.name = name
        self.passes = passes
        self.data = data
        self.calls = 0

    def check(self, ctx: FilterContext) -> FilterVerdict:
        self.calls += 1
        if self.passes:
            return FilterVerdict.allow(self.name, "ok", **self.data)
        return FilterVerdict.block(self.name, Reason.NEWS_BLACKOUT, "no", **self.data)


class TestFilterChain:
    def test_all_pass(self, spec: InstrumentSpec) -> None:
        chain = FilterChain([_Stub("a", True, session="london"), _Stub("b", True, spread=1.1)])
        verdict, data = chain.check(context(spec))
        assert verdict.passed
        assert data == {"session": "london", "spread": 1.1}

    def test_first_block_short_circuits(self, spec: InstrumentSpec) -> None:
        second = _Stub("b", True)
        chain = FilterChain([_Stub("a", False), second])
        verdict, _ = chain.check(context(spec))
        assert not verdict.passed
        assert verdict.filter_name == "a"
        assert second.calls == 0

    def test_data_from_passing_filters_survives_a_later_block(self, spec: InstrumentSpec) -> None:
        """A blocked cycle still records what the earlier filters measured."""
        chain = FilterChain([_Stub("a", True, session="london"), _Stub("b", False, spread=9.9)])
        verdict, data = chain.check(context(spec))
        assert not verdict.passed
        assert data == {"session": "london", "spread": 9.9}

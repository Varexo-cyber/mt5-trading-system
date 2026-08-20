"""The two gates that ask whether a trade has any chance of finishing.

Both exist because of the same real loss: a position opened minutes before the
system's own wind-down, held for a quarter of an hour, and closed on the clock
having paid the spread twice. Every gate that existed at the time passed it,
because every one of them asked "is now a bad moment" and none asked "is there
any point starting now".
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from config.schema import LivelinessFilterConfig, RunwayFilterConfig, SessionFilterConfig
from core.instrument import AssetClass, InstrumentSpec
from core.types import Direction, Series, Tick, Timeframe
from filters.base import FilterContext
from filters.liveliness_filter import LivelinessFilter
from filters.runway_filter import RunwayFilter
from filters.session_filter import SessionFilter
from risk.reasons import Reason
from tests.fakes.fake_mt5 import eurusd_spec

# Wednesday. The forex wind-down is 20:15 UTC, the index one 20:00.
WEDNESDAY = datetime(2026, 3, 11, tzinfo=UTC)
FRIDAY = datetime(2026, 3, 13, tzinfo=UTC)


@pytest.fixture
def spec() -> InstrumentSpec:
    return InstrumentSpec.from_mt5(eurusd_spec())


@pytest.fixture
def session() -> SessionFilter:
    return SessionFilter(SessionFilterConfig())


def at(day: datetime, hour: int, minute: int) -> datetime:
    return day.replace(hour=hour, minute=minute)


def ctx(spec: InstrumentSpec, now: datetime) -> FilterContext:
    return FilterContext(
        symbol=spec.symbol,
        spec=spec,
        now=now,
        direction=Direction.LONG,
        tick=Tick(symbol=spec.symbol, time=now, bid=1.08500, ask=1.08512),
    )


# ---------------------------------------------------------------- runway ---


class TestRunwayMeasurement:
    def test_counts_down_to_the_forex_wind_down(self, session: SessionFilter) -> None:
        assert session.minutes_of_runway(at(WEDNESDAY, 19, 30), "forex") == 45.0
        assert session.minutes_of_runway(at(WEDNESDAY, 20, 10), "forex") == 5.0

    def test_an_index_winds_down_on_its_own_earlier_clock(self, session: SessionFilter) -> None:
        """The index override is 20:00, so it always has 15 minutes less."""
        moment = at(WEDNESDAY, 19, 30)
        assert session.minutes_of_runway(moment, "index") == 30.0
        assert session.minutes_of_runway(moment, "forex") == 45.0

    def test_friday_cutoff_wins_when_it_is_earlier(self, session: SessionFilter) -> None:
        """19:00 on a Friday beats the 20:15 wind-down."""
        assert session.minutes_of_runway(at(FRIDAY, 18, 30), "forex") == 30.0
        # Same clock time on a Wednesday has the full evening left.
        assert session.minutes_of_runway(at(WEDNESDAY, 18, 30), "forex") == 105.0

    def test_a_continuous_market_has_no_deadline(self, session: SessionFilter) -> None:
        assert session.minutes_of_runway(at(WEDNESDAY, 20, 10), "crypto") is None

    def test_a_deadline_already_past_reads_as_zero_not_as_tomorrow(
        self, session: SessionFilter
    ) -> None:
        """The one wrong answer here would be 1400 minutes.

        Rolling to tomorrow's wind-down would turn the most dangerous moment of
        the day into the one with the most apparent runway.
        """
        assert session.minutes_of_runway(at(WEDNESDAY, 20, 30), "forex") == 0.0

    def test_seconds_count_against_the_runway(self, session: SessionFilter) -> None:
        moment = at(WEDNESDAY, 19, 30).replace(second=30)
        assert session.minutes_of_runway(moment, "forex") == pytest.approx(44.5)


class TestRunwayFilter:
    def test_blocks_the_entry_that_the_wind_down_would_immediately_close(
        self, spec: InstrumentSpec, session: SessionFilter
    ) -> None:
        """20:14 UTC: every other gate passes, and we flatten at 20:15.

        This is the exact moment the filter was written for.
        """
        gate = RunwayFilter(RunwayFilterConfig(), session)
        verdict = gate.check(ctx(spec, at(WEDNESDAY, 20, 14)))

        assert not verdict.passed
        assert verdict.reason is Reason.INSUFFICIENT_RUNWAY
        assert verdict.data["runway_minutes"] == 1.0

    def test_allows_a_mid_session_entry(self, spec: InstrumentSpec, session: SessionFilter) -> None:
        verdict = gate_at(session, spec, at(WEDNESDAY, 14, 30))
        assert verdict.passed
        assert verdict.data["runway_minutes"] == 345.0

    def test_exactly_the_floor_is_enough(
        self, spec: InstrumentSpec, session: SessionFilter
    ) -> None:
        """45 minutes against a 45-minute floor passes; a minute less does not."""
        assert gate_at(session, spec, at(WEDNESDAY, 19, 30)).passed
        assert not gate_at(session, spec, at(WEDNESDAY, 19, 31)).passed

    def test_an_index_is_cut_off_earlier_than_forex_at_the_same_moment(
        self, spec: InstrumentSpec, session: SessionFilter
    ) -> None:
        """19:20: forex has 55 minutes, the index has 40. Only one may enter."""
        index_spec = replace(spec, symbol="SPX500", asset_class=AssetClass.INDEX, is_forex=False)
        gate = RunwayFilter(RunwayFilterConfig(), session)
        moment = at(WEDNESDAY, 19, 20)

        assert gate.check(ctx(spec, moment)).passed
        assert not gate.check(ctx(index_spec, moment)).passed

    def test_a_continuous_market_is_never_blocked_on_runway(
        self, spec: InstrumentSpec, session: SessionFilter
    ) -> None:
        crypto = replace(spec, symbol="BTCUSD", asset_class=AssetClass.CRYPTO, is_forex=False)
        gate = RunwayFilter(RunwayFilterConfig(), session)

        verdict = gate.check(ctx(crypto, at(WEDNESDAY, 20, 14)))
        assert verdict.passed
        assert verdict.data["runway_minutes"] is None

    def test_per_class_override_applies(self, spec: InstrumentSpec, session: SessionFilter) -> None:
        config = RunwayFilterConfig(min_runway_by_class={"forex": 180.0})
        gate = RunwayFilter(config, session)

        # 17:30 leaves 165 minutes: fine by default, short of a three-hour rule.
        assert not gate.check(ctx(spec, at(WEDNESDAY, 17, 30))).passed
        assert (
            RunwayFilter(RunwayFilterConfig(), session)
            .check(ctx(spec, at(WEDNESDAY, 17, 30)))
            .passed
        )

    def test_disabled_lets_everything_through(
        self, spec: InstrumentSpec, session: SessionFilter
    ) -> None:
        gate = RunwayFilter(RunwayFilterConfig(enabled=False), session)
        assert gate.check(ctx(spec, at(WEDNESDAY, 20, 14))).passed

    def test_rejects_a_nonsensical_override(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            RunwayFilterConfig(min_runway_by_class={"forex": -5.0})


def gate_at(session: SessionFilter, spec: InstrumentSpec, moment: datetime):
    return RunwayFilter(RunwayFilterConfig(), session).check(ctx(spec, moment))


# ------------------------------------------------------------ liveliness ---


def bars(ranges: list[float], *, symbol: str = "EURUSD") -> Series:
    """A frame whose bars have exactly the given high-low ranges.

    Closes sit at the midpoint of each bar so the true range is the bar range
    and nothing else — the point of the test is the size of the moves, not the
    gaps between them.
    """
    index = pd.date_range("2026-03-11", periods=len(ranges), freq="5min", tz=UTC)
    mid = 1.08500
    rows = []
    for width in ranges:
        rows.append(
            {
                "open": mid,
                "high": mid + width / 2,
                "low": mid - width / 2,
                "close": mid,
                "tick_volume": 100,
                "spread": 12,
                "real_volume": 0,
            }
        )
    return Series(
        symbol=symbol,
        timeframe=Timeframe.M5,
        df=pd.DataFrame(rows, index=index),
        fetched_at=datetime(2026, 3, 11, 14, 30, tzinfo=UTC),
    )


def provider_for(series: Series):
    def provide(symbol: str, timeframe: Timeframe) -> Series:
        return series

    return provide


class TestLivelinessFilter:
    def test_a_market_running_at_its_normal_pace_passes(self, spec: InstrumentSpec) -> None:
        normal = [0.0010] * 200
        gate = LivelinessFilter(LivelinessFilterConfig(), provider_for(bars(normal)))

        verdict = gate.check(ctx(spec, at(WEDNESDAY, 14, 30)))
        assert verdict.passed
        assert verdict.data["activity_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_a_market_that_has_gone_to_sleep_is_blocked(self, spec: InstrumentSpec) -> None:
        """Same chart, same levels, same setup — a third of the movement."""
        dozing = [0.0010] * 194 + [0.00030] * 6
        gate = LivelinessFilter(LivelinessFilterConfig(), provider_for(bars(dozing)))

        verdict = gate.check(ctx(spec, at(WEDNESDAY, 14, 30)))
        assert not verdict.passed
        assert verdict.reason is Reason.MARKET_TOO_QUIET
        assert verdict.data["activity_ratio"] == pytest.approx(0.30, abs=0.02)

    def test_the_baseline_is_a_median_so_one_spike_does_not_poison_it(
        self, spec: InstrumentSpec
    ) -> None:
        """A news bar twenty times normal must not make the next hour read dead.

        The trap a mean baseline sets is that it distorts the reading exactly
        after the most tradeable moment of the day. The contrast is asserted
        rather than described: same data, two baselines.
        """
        spike = [0.0010] * 100 + [0.0200] * 5 + [0.0010] * 95
        series = bars(spike)
        gate = LivelinessFilter(LivelinessFilterConfig(), provider_for(series))

        verdict = gate.check(ctx(spec, at(WEDNESDAY, 14, 30)))
        assert verdict.passed
        assert verdict.data["activity_ratio"] == pytest.approx(1.0, abs=0.01)

        ranges = LivelinessFilter._true_ranges(series)
        against_mean = float(np.mean(ranges[-6:])) / float(np.mean(ranges[-120:]))
        assert against_mean == pytest.approx(0.56, abs=0.02)

    def test_a_quickening_market_is_not_blocked(self, spec: InstrumentSpec) -> None:
        waking = [0.0010] * 194 + [0.0030] * 6
        gate = LivelinessFilter(LivelinessFilterConfig(), provider_for(bars(waking)))
        assert gate.check(ctx(spec, at(WEDNESDAY, 14, 30))).passed

    def test_a_stuttering_recent_tape_is_blocked_even_when_atr_looks_normal(
        self, spec: InstrumentSpec
    ) -> None:
        """SN-shaped data: enough range, but repeated missing intraday bars."""
        series = bars([0.0010] * 200)
        index = list(series.df.index)
        for offset in range(170, 200):
            index[offset] += timedelta(minutes=(offset - 170) // 3 * 5)
        frame = series.df.copy()
        frame.index = pd.DatetimeIndex(index)
        broken = replace(series, df=frame)
        gate = LivelinessFilter(LivelinessFilterConfig(), provider_for(broken))

        verdict = gate.check(ctx(spec, at(WEDNESDAY, 14, 30)))

        assert not verdict.passed
        assert verdict.reason is Reason.MARKET_TOO_QUIET
        assert verdict.data["sparse_gap_fraction"] > 0.10

    def test_a_tape_made_mostly_of_flat_bars_is_blocked(self, spec: InstrumentSpec) -> None:
        series = bars([0.0010] * 170 + [0.0] * 18 + [0.0030] * 12)
        config = LivelinessFilterConfig(max_flat_bar_fraction=0.50)
        gate = LivelinessFilter(config, provider_for(series))

        verdict = gate.check(ctx(spec, at(WEDNESDAY, 14, 30)))

        assert not verdict.passed
        assert verdict.reason is Reason.MARKET_TOO_QUIET
        assert verdict.data["flat_bar_fraction"] > 0.50

    def test_one_session_gap_does_not_block_an_exchange_market(self, spec: InstrumentSpec) -> None:
        series = bars([0.0010] * 200)
        index = list(series.df.index)
        for offset in range(185, 200):
            index[offset] += timedelta(hours=16)
        frame = series.df.copy()
        frame.index = pd.DatetimeIndex(index)
        gapped = replace(series, df=frame)
        gate = LivelinessFilter(LivelinessFilterConfig(), provider_for(gapped))

        verdict = gate.check(ctx(spec, at(WEDNESDAY, 14, 30)))

        assert verdict.passed
        assert verdict.data["sparse_gap_fraction"] < 0.10

    def test_a_short_history_defers_rather_than_blocking(self, spec: InstrumentSpec) -> None:
        """A cold start is not evidence that the market is asleep.

        Blocking here would reproduce the failure this system has already paid
        for once: a whole day of scans and no trades, for a reason nobody could
        see in the log.
        """
        gate = LivelinessFilter(LivelinessFilterConfig(), provider_for(bars([0.0010] * 12)))

        verdict = gate.check(ctx(spec, at(WEDNESDAY, 14, 30)))
        assert verdict.passed
        assert verdict.data["activity_ratio"] is None

    def test_a_data_failure_blocks(self, spec: InstrumentSpec) -> None:
        """Unknown is not the same as fine — the house rule everywhere else."""

        def broken(symbol: str, timeframe: Timeframe) -> Series:
            raise RuntimeError("terminal not connected")

        gate = LivelinessFilter(LivelinessFilterConfig(), broken)
        verdict = gate.check(ctx(spec, at(WEDNESDAY, 14, 30)))

        assert not verdict.passed
        assert verdict.reason is Reason.DATA_UNAVAILABLE

    def test_a_flat_market_does_not_divide_by_zero(self, spec: InstrumentSpec) -> None:
        gate = LivelinessFilter(LivelinessFilterConfig(), provider_for(bars([0.0] * 200)))
        assert gate.check(ctx(spec, at(WEDNESDAY, 14, 30))).passed

    def test_disabled_lets_everything_through(self, spec: InstrumentSpec) -> None:
        dead = [0.0010] * 194 + [0.00001] * 6
        gate = LivelinessFilter(LivelinessFilterConfig(enabled=False), provider_for(bars(dead)))
        assert gate.check(ctx(spec, at(WEDNESDAY, 14, 30))).passed

    def test_windows_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="shorter than baseline_bars"):
            LivelinessFilterConfig(recent_bars=60, baseline_bars=60)

    def test_min_bars_must_cover_the_recent_window(self) -> None:
        with pytest.raises(ValueError, match="below recent_bars"):
            LivelinessFilterConfig(recent_bars=50, min_bars=10)

        with pytest.raises(ValueError, match="below quality_bars"):
            LivelinessFilterConfig(quality_bars=50, min_bars=40)


# --------------------------------------------------- reachability estimate ---


class TestTimeToTarget:
    """The maths behind `_target_is_reachable_in_time`, stated on its own.

    Net displacement over n bars scales with `sqrt(n) x ATR`, so covering a
    distance of d ATR takes `d^2` bars, not `d`. The difference is the whole
    point: a 6-ATR target is 36 bars away, not 6, and on M5 that is three
    hours rather than half of one.
    """

    @staticmethod
    def minutes(atr_units: float, efficiency: float = 1.5, bar_minutes: float = 5.0) -> float:
        return (atr_units / efficiency) ** 2 * bar_minutes

    def test_a_near_target_is_minutes_away(self) -> None:
        assert self.minutes(2.0) == pytest.approx(8.9, abs=0.1)

    def test_a_distant_target_is_hours_away(self) -> None:
        assert self.minutes(8.0) == pytest.approx(142.2, abs=0.5)

    def test_the_linear_estimate_would_understate_it_fourfold(self) -> None:
        """Why the square matters, in one assertion.

        A linear `distance / ATR` read puts the 8-ATR target 27 minutes away
        and clears it comfortably at 19:50. It is really over two hours away.
        """
        linear = 8.0 / 1.5 * 5.0
        assert linear == pytest.approx(26.7, abs=0.1)
        assert self.minutes(8.0) / linear == pytest.approx(5.3, abs=0.1)


def test_true_ranges_include_the_gap_between_bars() -> None:
    """A market that gaps is moving, even if each bar looks small."""
    index = pd.date_range("2026-03-11", periods=3, freq="5min", tz=UTC)
    df = pd.DataFrame(
        {
            "open": [1.0800, 1.0850, 1.0900],
            "high": [1.0801, 1.0851, 1.0901],
            "low": [1.0799, 1.0849, 1.0899],
            "close": [1.0800, 1.0850, 1.0900],
            "tick_volume": [1, 1, 1],
            "spread": [1, 1, 1],
            "real_volume": [0, 0, 0],
        },
        index=index,
    )
    series = Series(
        symbol="EURUSD",
        timeframe=Timeframe.M5,
        df=df,
        fetched_at=datetime(2026, 3, 11, tzinfo=UTC),
    )

    ranges = LivelinessFilter._true_ranges(series)
    # Each bar is 2 points wide but sits 50 points above the last close.
    assert np.allclose(ranges, [0.0051, 0.0051], atol=1e-6)


def test_the_wind_down_and_the_runway_floor_leave_no_gap() -> None:
    """Nothing may enter in the last 45 minutes, by either gate.

    Written as a sweep rather than a boundary case because the bug being
    prevented was a one-minute hole between two gates that each looked correct
    on its own.
    """
    session = SessionFilter(SessionFilterConfig())
    gate = RunwayFilter(RunwayFilterConfig(), session)
    spec = InstrumentSpec.from_mt5(eurusd_spec())

    moment = at(WEDNESDAY, 19, 31)
    while moment < at(WEDNESDAY, 21, 15):
        entry_ctx = ctx(spec, moment)
        blocked = not gate.check(entry_ctx).passed or not session.check(entry_ctx).passed
        assert blocked, f"{moment:%H:%M} UTC lets an entry through"
        moment += timedelta(minutes=1)


class TestADeadMarketIsNotJudgedAgainstItself:
    """Qiagen: 36.6349 / 36.6550, and every gate in the file said it was fine.

    A 2.01-cent spread while the M1 bars were a few points tall. Getting in and
    out cost more than the market moved in a minute, so the spread decided the
    outcome before the thesis had any say.

    All three existing measures let it through for the same reason: they grade a
    market against ITSELF. `min_activity_ratio` compares with the instrument's
    own recent normal, so one that is permanently dead has a dead baseline and
    scores about 1.0 — consistently untradeable reads as healthy.
    `max_flat_bar_fraction` counts only bars whose high equals their low exactly,
    and a tape that ticks once a minute has a range of one or two points, which
    is not flat by that definition.

    What was missing is the one comparison that comes from outside the chart.
    """

    @staticmethod
    def _ctx(spec: InstrumentSpec, spread: float) -> FilterContext:
        now = at(WEDNESDAY, 14, 30)
        return FilterContext(
            symbol=spec.symbol,
            spec=spec,
            now=now,
            direction=Direction.LONG,
            tick=Tick(symbol=spec.symbol, time=now, bid=1.08500, ask=1.08500 + spread),
        )

    def test_a_bar_that_does_not_cover_the_spread_is_refused(self, spec: InstrumentSpec) -> None:
        """The live shape: a steady tape, and every bar smaller than the spread."""
        steady = [0.0005] * 200
        gate = LivelinessFilter(LivelinessFilterConfig(), provider_for(bars(steady)))

        verdict = gate.check(self._ctx(spec, spread=0.0020))

        assert not verdict.passed
        assert verdict.reason is Reason.MARKET_TOO_QUIET
        assert verdict.data["bar_range_in_spreads"] == pytest.approx(0.25, abs=0.02)

    def test_the_old_measures_would_have_called_it_healthy(self, spec: InstrumentSpec) -> None:
        """Named explicitly, because this is why it needed a new measure rather
        than a tighter threshold on an existing one."""
        steady = [0.0005] * 200
        gate = LivelinessFilter(
            LivelinessFilterConfig(min_bar_range_in_spreads=0.0),
            provider_for(bars(steady)),
        )

        verdict = gate.check(self._ctx(spec, spread=0.0020))

        assert verdict.passed, "a permanently dead market scores 1.0 against itself"
        assert verdict.data["activity_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_a_market_that_moves_more_than_it_costs_still_passes(
        self, spec: InstrumentSpec
    ) -> None:
        normal = [0.0010] * 200
        gate = LivelinessFilter(LivelinessFilterConfig(), provider_for(bars(normal)))

        assert gate.check(self._ctx(spec, spread=0.00012)).passed

    def test_an_unknown_spread_does_not_block(self, spec: InstrumentSpec) -> None:
        """No quote is a different problem, and the freshness gate owns it. This
        test exists so a missing tick can never silently blank the catalogue."""
        steady = [0.0005] * 200
        gate = LivelinessFilter(LivelinessFilterConfig(), provider_for(bars(steady)))
        now = at(WEDNESDAY, 14, 30)
        blind = FilterContext(symbol=spec.symbol, spec=spec, now=now, direction=Direction.LONG)

        assert gate.check(blind).passed

    def test_the_overlay_turns_it_on(self) -> None:
        from config.loader import load_settings

        liveliness = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        ).filters.liveliness

        assert liveliness.min_bar_range_in_spreads == pytest.approx(1.0)


class TestAnOpenSharePositionIsStillManaged:
    """Dropping `stock` from the scan must not orphan what is already open.

    The scan universe decides what may be OPENED. Position management reads the
    broker's own position list, so a share bought before the change keeps its
    health reads, its stop moves and its wind-down. Worth pinning: a change that
    silently stopped protecting a live position would be far worse than the
    instrument class it was meant to remove.
    """

    def test_shares_are_scanned_but_have_to_earn_it(self) -> None:
        """Back in the scan, behind a measurement rather than behind the list."""
        from config.loader import load_settings

        settings = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        )

        assert "stock" in settings.instruments.asset_classes
        assert settings.filters.liveliness.max_gap_atr > 0

    def test_but_a_share_still_gets_flattened_before_its_close(self) -> None:
        """The wind-down stays armed as a net, not as a live gate — so an open
        share, or a re-enabled `stock`, is still not carried through a close."""
        from config.loader import load_settings

        session = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        ).filters.session

        assert "stock" in session.evening_flat_asset_classes
        assert session.evening_flat_by_class["stock"] == "15:30"

    def test_the_dead_market_test_stays_armed_too(self) -> None:
        from config.loader import load_settings

        liveliness = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        ).filters.liveliness

        assert liveliness.min_bar_range_in_spreads > 0


class TestAMarketThatJumpsCannotKeepItsStop:
    """ENR: -3.61R on an exit recorded as BROKER_SL.

    A stop costs 1.00R by definition, so 2.61R went straight through it. The
    exchange was shut and the share reopened somewhere else. Every rule in this
    system is written in R and R assumes the stop binds; an instrument that gaps
    breaks that several times a week and no management rule repairs it after the
    fact.

    Banning the asset class was the first answer and it was a proxy. The defect
    is not that ENR is a share — it is that it opened past its stop. This gate
    says so directly, which lets shares back in on behaviour and would refuse a
    forex pair that started doing the same thing.
    """

    @staticmethod
    def _series(gap_every: int, gap_size: float):  # type: ignore[no-untyped-def]
        import numpy as np
        import pandas as pd

        from core.types import Series, Timeframe

        count = 200
        close = np.full(count, 100.0)
        open_ = close.copy()
        for i in range(gap_every, count, gap_every):
            open_[i] = close[i - 1] + gap_size
        frame = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) + 0.10,
                "low": np.minimum(open_, close) - 0.10,
                "close": close,
                "tick_volume": np.full(count, 100),
                "spread": np.zeros(count),
                "real_volume": np.zeros(count),
            },
            index=pd.date_range("2026-03-01", periods=count, freq="5min", tz=UTC),
        )
        return Series("TEST", Timeframe.M5, frame, datetime(2026, 3, 11, 14, 30, tzinfo=UTC))

    def test_a_market_that_gaps_every_session_is_refused(self, spec: InstrumentSpec) -> None:
        gappy = self._series(gap_every=5, gap_size=2.0)
        gate = LivelinessFilter(LivelinessFilterConfig(), provider_for(gappy))

        verdict = gate.check(ctx(spec, at(WEDNESDAY, 14, 30)))

        assert not verdict.passed
        assert verdict.reason is Reason.MARKET_TOO_QUIET
        assert "jumps" in verdict.detail
        assert verdict.data["gap_in_atr"] > 1.0

    def test_a_market_that_holds_its_close_passes(self, spec: InstrumentSpec) -> None:
        smooth = self._series(gap_every=10_000, gap_size=0.0)
        gate = LivelinessFilter(LivelinessFilterConfig(), provider_for(smooth))

        assert gate.check(ctx(spec, at(WEDNESDAY, 14, 30))).passed

    def test_one_single_jump_does_not_ban_an_instrument(self, spec: InstrumentSpec) -> None:
        """The 99th percentile, not the maximum: a lone flash print is not a
        property of the market, and treating it as one costs a week of trading."""
        once = self._series(gap_every=190, gap_size=5.0)
        gate = LivelinessFilter(LivelinessFilterConfig(), provider_for(once))

        assert gate.check(ctx(spec, at(WEDNESDAY, 14, 30))).passed

    def test_it_can_be_switched_off(self, spec: InstrumentSpec) -> None:
        gappy = self._series(gap_every=5, gap_size=2.0)
        gate = LivelinessFilter(
            LivelinessFilterConfig(max_gap_atr=0.0, min_bar_range_in_spreads=0.0),
            provider_for(gappy),
        )

        assert gate.check(ctx(spec, at(WEDNESDAY, 14, 30))).passed


class TestTheSpreadGateSelectsInsteadOfPreselecting:
    """151 of 231 symbols were weighed out on spread before anything analysed them.

    Basis points are spread as a share of the PRICE. What breaks a trade is
    spread as a share of its RISK, and the two are unrelated: five pips against
    a ten-pip stop is fatal, five pips against a two-hundred-pip stop is noise.
    The bps gate cannot tell those apart, so it discarded good setups for the
    same reason it discarded bad ones — and it did it to two thirds of the
    catalogue, every cycle, before a single module ran.

    The right number already existed one step further on and had been running
    the whole time: `max_cost_share_of_risk`, spread plus commission against the
    trade's own risk, per order, refused as SL_TOO_TIGHT_FOR_COSTS. This is the
    division of labour those two should have had from the start — the cheap gate
    stops the absurd, the real gate prices the trade.
    """

    def test_the_absolute_gate_is_wide_enough_to_stop_preselecting(self) -> None:
        from config.loader import load_settings

        spread = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        ).filters.spread

        assert spread.max_spread_bps["forex"] >= 6.0
        assert spread.max_spread_bps["metal"] >= 20.0

    def test_but_it_is_not_switched_off(self) -> None:
        """An absurd quote is still not a market. 30 bps on forex is absurd."""
        from config.loader import load_settings

        spread = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        ).filters.spread

        assert spread.max_spread_bps["forex"] < 30.0
        assert spread.enabled

    def test_the_gate_that_actually_prices_a_trade_still_binds(self) -> None:
        """Loosening the proxy is only safe because this one runs per order."""
        from config.loader import load_settings

        risk = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        ).risk

        assert 0.0 < risk.max_cost_share_of_risk <= 0.5


class TestTheTwoGatesThePriceRecordSaysAreCostingMoney:
    """`scorecard --days 2`, 39 closed trades, on what the gates REFUSED.

    The journal shadows a blocked setup and resolves it later, so this is a
    measured counterfactual rather than an argument:

        CURRENCY_CONCENTRATION   9 blocked, 7 won, cost 9.08R
        MARKET_TOO_QUIET        10 blocked, 7 won, cost 8.46R

    Both are crude second layers sitting on top of a finer measurement, and in
    both cases the finer measurement is untouched. What is deliberately NOT
    loosened is the other half of that table — INSUFFICIENT_RUNWAY saved 10.75R
    and NEWS_BLACKOUT 7.75R, and runway is the sharper lesson of the two: 11 of
    its 16 blocked setups would have won, and it still saved money, because the
    winners were small and the losers were not.
    """

    @staticmethod
    def _live():  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH,
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml",
            env_overrides=False,
        ).filters

    def test_a_second_position_per_currency_is_allowed(self) -> None:
        """One position per currency meant a single EURUSD excluded every other
        dollar pair, including ones that move the other way."""
        assert self._live().currency_exposure.max_positions_per_currency == 2

    def test_but_the_measured_correlation_guard_still_decides(self) -> None:
        """The count was never the real protection. This is: a rolling
        correlation over 200 H1 bars, which can tell EURUSD from USDJPY where a
        count of shared currencies cannot."""
        correlation = self._live().correlation

        assert correlation.enabled
        assert correlation.max_abs_correlation <= 0.7

    def test_and_one_position_per_asset_class_is_untouched(self) -> None:
        """Four DAX shares at once stays impossible."""
        exposure = self._live().currency_exposure

        assert exposure.max_positions_per_asset_class == 1
        assert "stock" in exposure.grouped_asset_classes

    def test_a_quiet_market_is_allowed_and_a_dead_one_is_not(self) -> None:
        """`min_activity_ratio` measures a market against its OWN recent
        normal, so it cannot tell quiet from dead — a permanently dead market
        has a dead normal and scores near 1.0. The absolute test is the one
        that catches those, and it is the one left alone.
        """
        liveliness = self._live().liveliness

        assert liveliness.min_activity_ratio < 0.5  # quiet is allowed
        assert liveliness.min_bar_range_in_spreads >= 1.0  # dead is not

    def test_the_gates_that_saved_money_are_all_still_on(self) -> None:
        """The same table priced these positive, and one of them is the
        clearest lesson in it: runway blocked 16 setups of which 11 would have
        won, and still saved 10.75R."""
        filters = self._live()

        assert filters.runway.enabled
        assert filters.news.enabled
        assert filters.session.enabled
        assert filters.runway.min_runway_minutes > 0

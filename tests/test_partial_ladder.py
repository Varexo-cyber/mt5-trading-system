"""One unavailable timeframe used to refuse the whole market, for hours.

WHAT IT DID. `get_context` built its series with a dict comprehension over all
seven timeframes:

    series = {tf: self.get_series(symbol, tf) for tf in wanted}

Seven fetches, and the first to raise took the other six with it. The symbol
then went into `DataQuarantine`, which backs off 30 minutes, then two hours,
then four. So a hole in one M5 feed, or a broker carrying 180 H4 bars where 200
were asked for, removed that market from the account for the rest of the
session -- including the timeframes that were complete and the modules that
read only those.

The overlay already records this happening once: COFFEE, STLAM and SPM "thrown
out as having no data at all -- losing the six timeframes that actually drive
the decision over the one that only sets background context." The repair then
was to lower W1's bar requirement. That cured three symbols and left the
mechanism in place for every other one.

THE MECHANISM WAS THE BUG. The ladder was validated as a BLOCK, before anything
asked which parts of it the decision needed. Same shape as most of what this
system has got wrong: a check that is correct in itself, applied at a point
where nobody can yet tell whether it matters.

WHAT REPLACES IT IS NOT LOOSER. A timeframe in `required_timeframes` still
refuses the market outright -- no data is not permission. One that is not is
left out of the context, and every module that reads it returns a neutral
signal naming it. So the missing timeframe removes the modules that depend on
it from the vote instead of letting them guess: the same fail-closed rule,
applied per module rather than per market.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from config.schema import DataConfig
from core.data_manager import DataManager
from core.errors import DataIntegrityError, InsufficientDataError, StaleDataError
from core.types import Timeframe

NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


class _Clock:
    def now(self) -> datetime:
        return NOW


class _Connector:
    """A broker that cannot supply the timeframes it is told to fail."""

    def __init__(self, broken: dict[str, Exception]) -> None:
        self.broken = broken
        self.asked: list[str] = []

    def tick(self, symbol: str) -> None:
        return None


def _manager(broken: dict[str, Exception], **config: object) -> DataManager:
    fields: dict = {
        "timeframes": ("W1", "D1", "H4", "H1", "M15", "M5", "M1"),
        "bars": {"W1": 60, "D1": 130, "H4": 210, "H1": 210, "M15": 210, "M5": 210, "M1": 210},
        "min_bars_required": 200,
        "min_bars_by_timeframe": {"W1": 50, "D1": 120},
    }
    fields.update(config)
    manager = DataManager.__new__(DataManager)
    manager.config = DataConfig(**fields)  # type: ignore[arg-type]
    manager.clock = _Clock()  # type: ignore[assignment]
    manager.connector = _Connector(broken)  # type: ignore[assignment]
    manager._cache = {}

    def get_series(symbol, timeframe, *, count=None, force_refresh=False):  # type: ignore[no-untyped-def]
        tf = Timeframe.parse(timeframe)
        manager.connector.asked.append(tf.value)
        failure = broken.get(tf.value)
        if failure is not None:
            raise failure
        frame = pd.DataFrame(
            {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]},
            index=pd.DatetimeIndex([NOW]),
        )
        from core.types import Series

        return Series(symbol=symbol, timeframe=tf, df=frame, fetched_at=NOW)

    manager.get_series = get_series  # type: ignore[assignment]
    return manager


SHORT = InsufficientDataError("EURUSD W1: 8 closed bars available, 50 required.")
HOLES = DataIntegrityError("EURUSD M5: 80 bars missing inside trading weeks (5.3%).")


class TestAnOptionalTimeframeNoLongerVetoesTheMarket:
    def test_the_other_six_still_arrive(self) -> None:
        manager = _manager({"W1": SHORT}, required_timeframes=("H4", "H1", "M15", "M5", "M1"))
        context = manager.get_context("EURUSD", with_tick=False)

        assert Timeframe.W1 not in context.series
        assert len(context.series) == 6
        assert Timeframe.M5 in context.series

    def test_what_was_dropped_is_recorded_with_its_reason(self) -> None:
        """Silently analysing a partial ladder would be worse than refusing.
        The first question about a bad run is what it had in front of it."""
        manager = _manager({"W1": SHORT}, required_timeframes=("H4", "H1", "M15", "M5", "M1"))
        context = manager.get_context("EURUSD", with_tick=False)

        missing = context.meta["unavailable_timeframes"]
        assert set(missing) == {"W1"}
        assert "8 closed bars available" in missing["W1"]

    def test_the_fetch_carries_on_past_the_failure(self) -> None:
        """The comprehension stopped at the first raise. W1 is first in the
        ladder, so under the old code nothing below it was ever attempted."""
        manager = _manager({"W1": SHORT}, required_timeframes=("H4", "H1", "M15", "M5", "M1"))
        manager.get_context("EURUSD", with_tick=False)

        assert manager.connector.asked == ["W1", "D1", "H4", "H1", "M15", "M5", "M1"]


class TestARequiredTimeframeStillRefusesOutright:
    def test_no_data_is_still_not_permission(self) -> None:
        manager = _manager({"M5": HOLES}, required_timeframes=("H4", "H1", "M15", "M5", "M1"))
        with pytest.raises(DataIntegrityError):
            manager.get_context("EURUSD", with_tick=False)

    def test_an_empty_required_list_keeps_the_old_behaviour(self) -> None:
        """A config that says nothing must not loosen on upgrade. Empty means
        every timeframe is required, which is exactly what this did before."""
        manager = _manager({"W1": SHORT})
        assert manager.config.required_timeframes == ()
        with pytest.raises(InsufficientDataError):
            manager.get_context("EURUSD", with_tick=False)

    def test_a_failure_that_is_neither_structural_still_propagates(self) -> None:
        """Only the two structural failures are survivable. A stale quote is a
        fact about right now and the caller decides what to do with it; being
        swallowed here would make it invisible."""
        manager = _manager(
            {"W1": StaleDataError("EURUSD W1: newest bar is 4 days old")},
            required_timeframes=("H4", "H1", "M15", "M5", "M1"),
        )
        with pytest.raises(StaleDataError):
            manager.get_context("EURUSD", with_tick=False)


class TestTheConfigCannotBeWrongInTheExpensiveDirection:
    def test_requiring_a_timeframe_that_is_never_loaded_is_refused(self) -> None:
        """It would refuse every symbol in the catalogue for a timeframe
        nobody asked for -- a typo with the blast radius of the whole run."""
        with pytest.raises(ValueError, match=r"not in data\.timeframes"):
            DataConfig(
                timeframes=("H1", "M15"),
                bars={"H1": 210, "M15": 210},
                required_timeframes=("H4",),
            )

    def test_the_shipped_overlay_still_requires_the_intraday_ladder(self) -> None:
        """W1 and D1 set background and bias. Everything below them decides the
        trade, and that half stays hard."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        assert set(settings.data.required_timeframes) == {"H4", "H1", "M15", "M5", "M1"}


class TestNoModuleBlowsUpOnAMissingTimeframe:
    def test_every_bars_call_in_analysis_is_guarded(self) -> None:
        """`ctx.bars()` RAISES where `ctx.series.get()` returns None, so it is
        the one accessor that a missing timeframe can turn into an exception.

        The confluence engine evaluates its modules in a plain generator with
        no per-module isolation, so an unguarded KeyError there does not
        degrade one module -- it takes the whole cycle down, for every symbol,
        over one missing weekly series.

        `entry_quality` has always caught it and returns DATA_UNAVAILABLE.
        `market_structure.inspect` did not, and now reads `series.get`. This
        asserts the property rather than the current list of files, because the
        next module to reach for `bars()` will be written by someone who has
        never heard of this.
        """
        import ast
        import pathlib

        unguarded: list[str] = []
        for path in sorted(pathlib.Path("analysis").glob("*.py")):
            tree = ast.parse(path.read_text())
            protected: set[int] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    for child in ast.walk(node):
                        protected.add(id(child))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "bars"
                    and id(node) not in protected
                ):
                    unguarded.append(f"{path.name}:{node.lineno}")
        assert unguarded == []

    def test_a_structure_read_of_an_absent_timeframe_is_no_opinion(self) -> None:
        from analysis.market_structure import MarketStructure
        from config.schema import MarketStructureConfig
        from core.types import MarketContext

        module = MarketStructure(MarketStructureConfig())
        context = MarketContext(symbol="EURUSD", now=NOW, series={}, tick=None)

        assert module.inspect(context, Timeframe.D1) is None
        signal = module.analyze(context)
        assert signal.score == 0.0

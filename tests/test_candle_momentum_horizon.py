"""A one-minute trigger was being handed a twenty-four hour plan.

`candle_momentum` reads M15 for bias, M5 for confirmation and one closed M1
candle as the trigger. Nothing in it looks at a chart slower than M15. It
appeared in neither `intraday_modules` nor `quick_modules`, so
`_classify_horizon` fell through to its final `return "swing"` and the proposal
got H1 planning authority and a target twenty-four bars of H1 out.

THE WARNING WAS ALREADY WRITTEN, in the docstring of the very field it was
missing from: "adding a module without adding it here is a silent and expensive
mistake: `drift_continuation` measures eight M15 bars and was handed a swing
plan for a signal whose whole mechanism expires in about two hours." Same
sentence, one module later, and this time on a trigger eight times faster than
the one the sentence was about.

AND IT IS THE THROUGHPUT, not only the plan. A swing proposal holds its
position slot until `time_exit_hours` (24) releases it. Four of the five
modules enabled live classified as swing, so nearly every trade parked a slot
for a full day, and the slot count is the hard ceiling on trades per day. An
intraday plan targets twelve M15 bars — three hours — and frees the slot eight
times sooner.

These tests drive the classifier itself rather than reading the list back,
because the list is not the behaviour: the classifier uses a SUBSET test, so
membership only decides anything when nothing slower fired alongside.
"""

from __future__ import annotations

from analysis.confluence import ConfluenceEngine
from config.loader import DEFAULT_CONFIG_PATH, load_settings
from config.schema import ConfluenceConfig
from core.types import Signal


def _engine() -> ConfluenceEngine:
    config = load_settings(
        DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
    ).analysis.confluence
    engine = ConfluenceEngine.__new__(ConfluenceEngine)
    engine.config = config
    return engine


def _pair(module: str, score: float = 60.0, timeframe: str = "M1"):  # type: ignore[no-untyped-def]
    return (
        Signal(module, score, 0.8, reasoning="fired", details={"timeframe": timeframe}),
        1.0,
    )


class TestTheFastTriggerGetsAFastPlan:
    def test_candle_momentum_alone_is_not_a_swing_trade(self) -> None:
        """The defect, stated as the outcome it produced: a module whose
        slowest chart is M15 owning a twenty-four hour plan."""
        horizon, _family = _engine()._classify_horizon([_pair("candle_momentum")])

        assert horizon != "swing"
        assert horizon == "intraday"

    def test_the_plan_it_now_gets_expires_within_the_day(self) -> None:
        """Named against the profile rather than the string, because the string
        is only worth something if the profile behind it is actually shorter.
        Swing plans H1 x 24 bars; intraday plans M15 x 12."""
        config = _engine().config
        swing = config.horizon_profiles["swing"]
        intraday = config.horizon_profiles["intraday"]

        assert swing.planning_timeframe == "H1"
        assert swing.target_horizon_bars == 24  # 24 hours, and 24 hours of slot
        assert intraday.planning_timeframe == "M15"
        assert intraday.target_horizon_bars == 12  # three hours

    def test_it_still_defers_to_h1_structure_when_that_fired_too(self) -> None:
        """The subset test is the point and must survive this change.

        `market_structure` reading a break of structure on H1 IS a swing
        thesis, and a minute candle agreeing with it does not shrink the trade
        to three hours. Membership only decides the horizon when nothing slower
        fired alongside — which is exactly why adding a module to the list is
        safe and leaving it out is not.
        """
        horizon, _family = _engine()._classify_horizon(
            [_pair("candle_momentum"), _pair("market_structure", timeframe="H1")]
        )

        assert horizon == "swing"


class TestTheListMatchesWhatTheModuleActuallyReads:
    def test_no_chart_slower_than_the_intraday_planning_timeframe(self) -> None:
        """Asserted against the module's own configured timeframes, so the two
        move together. If someone points `candle_momentum` at H1 for bias, this
        fails and the horizon has to be reconsidered rather than silently
        becoming wrong again."""
        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        module = settings.analysis.candle_momentum

        assert {module.trigger_timeframe, module.confirm_timeframe, module.bias_timeframe} <= {
            "M1",
            "M5",
            "M15",
        }
        assert "candle_momentum" in ConfluenceConfig().intraday_modules

    def test_it_is_intraday_and_not_quick(self) -> None:
        """A deliberate line, not an oversight in the other direction. The
        quick profile plans on M5 with a thirty-minute target, and that is
        section six's own lane — which already runs this detector on M1 outside
        the vote. Inside the vote the module's SLOWEST input is M15, and the
        intraday profile plans on exactly that."""
        assert "candle_momentum" not in ConfluenceConfig().quick_modules

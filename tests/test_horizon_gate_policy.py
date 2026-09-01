"""Quick trades are judged on quick evidence without turning safety gates off."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from analysis.confluence import TradeIdea
from analysis.target_reach import ReachVerdict
from config.loader import load_settings
from core.instrument import AssetClass
from core.types import Direction, MarketContext, Series, Signal, Timeframe
from runner.service import JarvisRunner


def runner() -> JarvisRunner:
    instance = JarvisRunner.__new__(JarvisRunner)
    instance.settings = load_settings(env_overrides=False)  # type: ignore[assignment]
    return instance


def idea(*, horizon: str, family: str, signals: tuple[Signal, ...] = ()) -> TradeIdea:
    return TradeIdea(
        symbol="EURUSD.i",
        approved=True,
        direction=Direction.LONG,
        score=70.0,
        confidence=0.7,
        entry=1.10,
        stop_loss=1.099,
        take_profit=1.102,
        reason="test",
        signals=signals,
        setup_family=family,
        horizon=horizon,
        planning_timeframe="M5" if horizon == "quick" else "H1",
        expected_horizon_minutes=30 if horizon == "quick" else 1440,
    )


def context() -> MarketContext:
    now = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    series: dict[Timeframe, Series] = {}
    for timeframe, frequency in ((Timeframe.M1, "1min"), (Timeframe.M5, "5min")):
        index = pd.date_range(end=now, periods=80, freq=frequency)
        close = 1.10 + np.arange(80) * 0.00001
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.0001,
                "low": close - 0.0001,
                "close": close,
                "tick_volume": 100,
                "spread": 10,
                "real_volume": 0,
            },
            index=index,
        )
        series[timeframe] = Series("EURUSD.i", timeframe, frame, now)
    return MarketContext("EURUSD.i", now, series, None)


def test_quick_only_softens_plausible_base_rate_failures() -> None:
    service = runner()
    quick = idea(horizon="quick", family="m1_micro_breakout_m1")
    swing = idea(horizon="swing", family="trend_momentum_swing")

    plausible = ReachVerdict(200, 20.0, 27.0, 33.3)
    fantasy = ReachVerdict(200, 10.0, 27.0, 33.3)

    assert service._reach_failure_is_advisory(quick, plausible)
    assert not service._reach_failure_is_advisory(quick, fantasy)
    assert not service._reach_failure_is_advisory(swing, plausible)


def test_large_direction_disadvantage_remains_hard() -> None:
    service = runner()
    quick = idea(horizon="quick", family="m1_micro_breakout_m1")

    assert service._direction_failure_is_advisory(quick, ReachVerdict(200, 30.0, 42.0, 33.3))
    assert not service._direction_failure_is_advisory(quick, ReachVerdict(200, 20.0, 45.0, 33.3))


def test_quick_relaxation_does_not_relax_adverse_reversal_limit() -> None:
    service = runner()
    quick = idea(horizon="quick", family="m1_micro_breakout_m1")
    swing = idea(horizon="swing", family="trend_momentum_swing")
    quick_config = service._entry_quality_config_for(quick)
    swing_config = service._entry_quality_config_for(swing)

    assert quick_config.max_favourable_extension_atr["forex"] > (
        swing_config.max_favourable_extension_atr["forex"]
    )
    assert quick_config.max_last_bar_adverse_atr == swing_config.max_last_bar_adverse_atr


def test_new_m1_event_gets_its_own_review_identity() -> None:
    service = runner()
    signal = Signal(
        "m1_micro_breakout",
        75.0,
        0.8,
        details={"timeframe": "M1"},
    )
    quick = idea(
        horizon="quick",
        family="m1_micro_breakout_m1",
        signals=(signal,),
    )

    key = service._review_key(quick, context())

    assert key is not None
    assert key[4] == "M1"


def test_broad_old_veto_does_not_silence_a_new_quick_event() -> None:
    service = runner()
    quick = idea(horizon="quick", family="m1_micro_breakout_m1")
    swing = idea(horizon="swing", family="trend_momentum_swing")

    assert not service._broad_veto_memory_applies(quick)
    assert service._broad_veto_memory_applies(swing)


def test_the_replay_can_judge_in_the_account_s_own_live_mode() -> None:
    """The offline tool measured a different engine than the account runs.

    `live_enabled_modules` is only consulted when the mode is live, and the
    replay hardcoded BACKTEST. So every offline figure counted every weighted
    detector while the account counted a subset — which made the tool built to
    answer "why did nothing trade" answer it about the wrong engine.
    """
    from backtesting.replay import HistoricalContextReplay
    from core.types import TradingMode

    engine = object()

    assert HistoricalContextReplay(engine).mode is TradingMode.BACKTEST  # type: ignore[arg-type]
    assert (
        HistoricalContextReplay(engine, mode=TradingMode.MICRO_LIVE).mode  # type: ignore[arg-type]
        is TradingMode.MICRO_LIVE
    )


def test_the_live_allowlist_follows_the_measured_record() -> None:
    """Only what has an out-of-sample number may trade real money.

    Rewritten 30 August. It used to assert the section-one modules were live;
    they are not any more, and the reason is the same principle this test was
    always about. `impulse_retest` and `order_block` each carry a holdout that
    reproduced their training half; the modules that came off carry either a
    negative measurement (`trend_momentum`, -0.365R over 163 live trades) or
    none at all (`m1_micro_breakout`, `basket_divergence`).

    AND `impulse_retest` CAME OFF ON 30 AUGUST, on thirty days of this
    broker's own data:

        order_block      204 trades   54.4%   +18.00 R   EUR +85.14
        impulse_retest    48 trades   39.6%   -10.00 R   EUR -68.05

    Section two was eating three quarters of what section three earned; run
    alone, order_block was +39.5% over the window and the pair was +7.9%.

    It was NOT proven bad -- 48 trades at 39.6% is -1.44 sigma, which a
    neutral strategy produces often enough.

    AND IT WENT BACK ON, 31 August, on a measurement seven times larger:
    358 trades over 180 days, 81.3% win, +0.058 R a trade against
    order_block's +0.026, and 89 of 116 days green against 83 of 140. The
    thirty-day sample I removed it on was the smaller one and I let it weigh
    more than it should have.


        THIRD SECTION ADDED 31 AUGUST: `order_block_fast`, the same detector
        as `order_block` on M1 instead of M30, as its own instance with its own
        name, weight and breaker.

        It is the weakest evidence on the account and
        `docs/hypotheses/order_block_fast.md` says so in full: 105 trades over
        14 days, five markets, the adjacent M5 clock negative over 308 trades,
        and the run that produced the number did not charge the trades their
        spread. Live at the owner's explicit instruction with those four facts
        on the screen -- "risico's moeten genomen worden om te testen" -- under
        the strictest breaker here (30 trades, 50% loss share, 7-streak).
    """
    from config.loader import DEFAULT_CONFIG_PATH, load_settings

    settings = load_settings(
        DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
    )
    live = set(settings.analysis.confluence.live_enabled_modules)

    # The corrected broker replay on 31 August invalidated the earlier live
    # table: both sections lost after the stale-M1-price and duplicate-symbol
    # portfolio bugs were removed. They stay enabled for shadow measurement,
    # but none may use real money.
    assert live == {"walkforward_index", "failed_session_breakout"}
    # Every live module keeps a breaker. That was the real content of this
    # test and it survives the change.
    for module in live:
        assert module in settings.risk.section_breakers


class TestSectionThreeCouldNotTradeAtAll:
    """`order_block` ran 54,190 times in 24 hours and scored zero every time.

    Not "found nothing". `order_block.timeframe` is M30 and the live data
    ladder was `[W1, D1, H4, H1, M15, M5, M1]` -- no M30. So
    `ctx.series.get(Timeframe.M30)` returned None on every cycle and
    `OrderBlock.analyze` returned `Signal.neutral(... "M30 needs 117 closed
    bars")`. The section had been on `live_enabled_modules` the whole time and
    was structurally incapable of producing a signal.

    Offline it produced 1283 setups over 180 days, because the harness puts
    the section's own clock into the context. That gap between the two is what
    made "0 trades overnight" unreadable.
    """

    def _settings(self):
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

    def test_every_live_module_reads_a_timeframe_the_ladder_fetches(self) -> None:
        """The general rule, not the one instance. A module allowed to trade
        real money on a clock nobody fetches is a section that cannot fire,
        and it is silent about it."""
        settings = self._settings()
        ladder = {str(tf) for tf in settings.data.timeframes}

        for name in settings.analysis.confluence.live_enabled_modules:
            clock = getattr(settings.analysis, name).timeframe
            assert clock in ladder, (
                f"{name} reads {clock} and data.timeframes does not fetch it, "
                f"so it can only ever return a neutral signal"
            )

    def test_m30_is_fetched_and_deep_enough_for_the_block_search(self) -> None:
        settings = self._settings()
        block = settings.analysis.order_block

        assert "M30" in {str(tf) for tf in settings.data.timeframes}
        needed = block.atr_period + block.lookback_bars + block.block_search_bars + 2
        assert settings.data.bars["M30"] >= needed
        assert settings.data.bars["M30"] >= settings.data.min_bars_required

    def test_m30_is_not_required_so_a_hole_in_it_cannot_quarantine_a_market(self) -> None:
        """`required_timeframes` exists precisely so a missing frame removes
        the modules that read it from the vote instead of throwing the whole
        market away. Adding M30 there would undo that."""
        assert "M30" not in self._settings().data.required_timeframes


class TestTheEntryConfirmationExemptionReachesTheRunner:
    """It was added to the confluence engine and never reached the gate that
    actually fires.

    `_entry_is_confirmed` refused 155 of 440 setups in one live day -- 35% of
    everything that formed, the largest single gate in the funnel -- and the
    exemption list it was supposed to honour lived one layer up.
    """

    def _runner_on_the_live_config(self):
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        instance = JarvisRunner.__new__(JarvisRunner)
        instance.settings = load_settings(  # type: ignore[assignment]
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        return instance

    def _running_away_from_a_short(self) -> MarketContext:
        """Price up 10 ATR over the confirmation window: an unambiguous refusal
        for a short, so the only thing separating the two ideas below is the
        family."""
        now = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)
        series: dict[Timeframe, Series] = {}
        for timeframe, frequency in ((Timeframe.M1, "1min"), (Timeframe.M5, "5min")):
            index = pd.date_range(end=now, periods=80, freq=frequency)
            close = np.full(80, 1.10)
            close[-3:] = [1.1007, 1.1014, 1.1020]
            frame = pd.DataFrame(
                {
                    "open": close,
                    "high": close + 0.0001,
                    "low": close - 0.0001,
                    "close": close,
                    "tick_volume": 100,
                    "spread": 10,
                    "real_volume": 0,
                },
                index=index,
            )
            series[timeframe] = Series("EURUSD.i", timeframe, frame, now)
        return MarketContext("EURUSD.i", now, series, None)

    def _short(self, family: str) -> TradeIdea:
        return TradeIdea(
            symbol="EURUSD.i",
            approved=True,
            direction=Direction.SHORT,
            score=70.0,
            confidence=0.7,
            entry=1.1020,
            stop_loss=1.1030,
            take_profit=1.1000,
            reason="test",
            signals=(),
            setup_family=family,
            horizon="intraday",
            planning_timeframe="M15",
            expected_horizon_minutes=120,
        )

    def test_the_two_live_families_are_exempt_in_the_runner(self) -> None:
        """The same market, the same adverse move, two families. Only the
        exempt one is allowed through, so the test measures the exemption
        rather than the presence of a line of code."""
        service = self._runner_on_the_live_config()
        context = self._running_away_from_a_short()

        refused, adverse = service._entry_is_confirmed(context, self._short("trend_momentum_swing"))
        assert refused is False, "price ran 10 ATR against this short; the gate must hold"
        assert adverse is not None and adverse > 1.0

        # Use the setup-family strings the six shipped module/timeframe
        # instances actually produce. The exemption is a substring match; a
        # renamed clock alias that no longer contains its parent family would
        # otherwise silently bring the 35%-of-setups blocker back.
        for family in (
            "impulse_retest_m15",
            "impulse_retest_m30_m30",
            "order_block_fast_m1",
            "order_block_m15_m15",
            "order_block_m30",
            "order_block_h1_h1",
        ):
            allowed, measured = service._entry_is_confirmed(context, self._short(family))
            assert allowed is True, f"{family} is exempt and must pass the same bars"
            assert measured is None

    def test_the_third_adverse_bar_copy_honours_the_same_exemption(self) -> None:
        service = self._runner_on_the_live_config()
        context = self._running_away_from_a_short()

        ordinary = service._assess_entry_quality(
            context, self._short("trend_momentum_swing"), AssetClass.FOREX
        )
        assert ordinary.reason_code == "PULLBACK_STILL_ACTIVE"

        for family in (
            "impulse_retest_m15",
            "impulse_retest_m30_m30",
            "order_block_fast_m1",
            "order_block_m15_m15",
            "order_block_m30",
            "order_block_h1_h1",
        ):
            assessment = service._assess_entry_quality(
                context, self._short(family), AssetClass.FOREX
            )
            assert assessment.reason_code != "PULLBACK_STILL_ACTIVE", family

    def test_generic_target_reach_is_advisory_for_every_measured_retest_clock(self) -> None:
        service = self._runner_on_the_live_config()
        poor = ReachVerdict(200, 1.0, 80.0, 50.0)

        assert not service._reach_failure_is_advisory(self._short("trend_momentum_swing"), poor)
        for family in (
            "impulse_retest_m15",
            "impulse_retest_m30_m30",
            "order_block_fast_m1",
            "order_block_m15_m15",
            "order_block_m30",
            "order_block_h1_h1",
        ):
            measured = self._short(family)
            assert service._reach_failure_is_advisory(measured, poor), family
            assert service._direction_failure_is_advisory(measured, poor), family

    def test_the_exemption_is_bounded_to_the_named_families(self) -> None:
        """Every other module keeps the check. The gate was added because a
        live GBPJPY short went into a resistance break, and that reason still
        holds everywhere it was not measured away."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        exempt = set(settings.analysis.confluence.entry_timing_exempt_families)

        assert exempt == {"impulse_retest", "order_block"}
        assert settings.analysis.confluence.require_entry_confirmation is True


class TestASecondClockIsASecondModule:
    """`order_block_fast` is `order_block` on M1, and it has to be separable.

    Every registry that decides what a module may do is keyed by its NAME:
    weights, live_enabled_modules, section_breakers, and the module_scores
    rows the breaker and the scorecard read back. Two instances sharing one
    name would share a weight, share a breaker and write into each other's
    history -- so M1 could not be switched off without switching off M30, and
    a losing streak on one clock would trip the other.
    """

    def _settings(self):
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

    def test_the_instance_carries_its_own_name(self) -> None:
        from analysis.order_block import OrderBlock
        from config.schema import OrderBlockConfig

        assert OrderBlock(OrderBlockConfig()).name == "order_block"
        assert OrderBlock(OrderBlockConfig(), name="order_block_fast").name == "order_block_fast"

    def test_the_signal_is_written_under_that_name(self) -> None:
        """The journal row, the weight lookup and the breaker all key off the
        `module` field of the Signal. A class attribute here would file every
        M1 decision under the M30 section."""
        import inspect

        from analysis import order_block

        source = " ".join(inspect.getsource(order_block.OrderBlock).split())

        assert "module=self.name" in source
        assert "Signal.neutral(self.name" in source

    def test_the_runner_builds_both_instances(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from runner.service import build_analysis_modules

        built = {
            module.name
            for module in build_analysis_modules(
                load_settings(
                    DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
                )
            )
        }

        assert {
            "impulse_retest",
            "impulse_retest_m30",
            "order_block_fast",
            "order_block_m15",
            "order_block",
            "order_block_h1",
        } <= built

    def test_the_two_clocks_differ_and_nothing_else_does(self) -> None:
        """Only the clock was measured. A threshold nudged for M1 by feel would
        be measuring something other than what produced the number."""
        settings = self._settings()
        slow = settings.analysis.order_block
        fast = settings.analysis.order_block_fast

        assert (slow.timeframe, fast.timeframe) == ("M30", "M1")
        differ = {
            field
            for field in type(slow).model_fields
            if getattr(slow, field) != getattr(fast, field)
        }
        assert differ == {"timeframe"}, f"only the clock may differ, these also do: {differ}"

    def test_it_has_its_own_breaker_and_it_is_the_strictest(self) -> None:
        """The evidence is the thinnest on the account, so the automatic
        switch-off has to be the quickest."""
        breakers = self._settings().risk.section_breakers
        fast, slow = breakers["order_block_fast"], breakers["order_block"]

        assert fast.window < slow.window
        assert fast.minimum_trades < slow.minimum_trades
        assert fast.maximum_loss_share < slow.maximum_loss_share
        assert fast.losing_streak < slow.losing_streak

    def test_it_is_classified_intraday_by_exact_name(self) -> None:
        """`_classify_horizon` matches the list EXACTLY, unlike
        `entry_timing_exempt_families` which is a substring test and would have
        covered this by accident. Falling through to "swing" would give an M1
        setup H1 planning and a D1/W1 veto."""
        confluence = self._settings().analysis.confluence

        assert "order_block_fast" in confluence.intraday_modules
        assert "order_block_fast" not in confluence.quick_modules

    def test_every_enabled_clock_has_an_independent_identity_and_breaker(self) -> None:
        from runner.service import build_analysis_modules

        settings = self._settings()
        expected = {
            "impulse_retest": "M15",
            "impulse_retest_m30": "M30",
            "order_block_fast": "M1",
            "order_block_m15": "M15",
            "order_block": "M30",
            "order_block_h1": "H1",
        }
        built = {module.name: module for module in build_analysis_modules(settings)}

        assert set(settings.analysis.confluence.live_enabled_modules) == {
            "walkforward_index",
            "failed_session_breakout",
        }
        for name, timeframe in expected.items():
            assert built[name].config.enabled is True
            assert built[name].config.timeframe == timeframe
            assert name in settings.risk.section_breakers
            assert name in settings.analysis.confluence.weights
            assert name in settings.analysis.confluence.intraday_modules

    def test_the_base_config_leaves_it_off(self) -> None:
        """One owner's bet on one account at EUR 216, not a default."""
        from config.loader import load_settings

        base = load_settings(env_overrides=False)

        assert base.analysis.order_block_fast.enabled is False
        assert "order_block_fast" not in base.analysis.confluence.live_enabled_modules

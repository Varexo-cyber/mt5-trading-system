"""Confluence is deterministic, auditable, and live fails closed."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from analysis.confluence import ConfluenceEngine
from config.schema import ConfluenceConfig
from core.types import (
    Direction,
    MarketContext,
    Series,
    Signal,
    Tick,
    Timeframe,
    TradingMode,
)


class StubModule:
    def __init__(self, signal: Signal) -> None:
        self.name = signal.module
        self.signal = signal

    def analyze(self, ctx: MarketContext) -> Signal:
        return self.signal


def context(step: float = 0.0006, noise: float = 0.0, bars: int = 100) -> MarketContext:
    """A rising H1 series. `step` is the move per bar, which decides whether a
    2R target is reachable — the engine bounds the target by what this market
    actually travels, so the fixture has to have a speed.

    `noise` is what a REFUSAL test needs, and a straight line cannot supply it.
    With no noise the series never retraces, so the stop is never touched and
    every window ends a little in front. Correctly priced, that market is
    profitable at a small target, and it was only ever refused because the
    search charged a full stop-out to windows the stop never came near. A
    market that genuinely does not pay has to be able to take the stop, and a
    fixed sawtooth will not do either — a perfectly regular oscillation is
    perfectly predictable and therefore also profitable. So: a seeded random
    walk, deterministic across runs and edgeless by construction.
    """
    index = pd.date_range("2026-01-01", periods=bars, freq="1h", tz=UTC)
    drift = np.arange(bars, dtype=float) * step
    wander = np.cumsum(np.random.default_rng(4).normal(0.0, noise, bars)) if noise > 0 else 0.0
    close = pd.Series(1.10 + drift + wander, index=index)
    frame = pd.DataFrame(
        {
            "open": close - 0.00005,
            "high": close + 0.0003,
            "low": close - 0.0003,
            "close": close,
            "tick_volume": 100,
            "spread": 10,
            "real_volume": 0,
        }
    )
    now = datetime(2026, 1, 5, 5, tzinfo=UTC)
    return MarketContext(
        symbol="EURUSD",
        now=now,
        series={Timeframe.H1: Series("EURUSD", Timeframe.H1, frame, now)},
        tick=Tick("EURUSD", now, 1.1098, 1.1100),
    )


def modules() -> list[StubModule]:
    return [
        StubModule(Signal("one", 70, 0.8, invalidation_price=1.1050)),
        StubModule(Signal("two", 60, 0.7, invalidation_price=1.1060)),
        StubModule(Signal.neutral("volatility_regime")),
    ]


def config(**overrides: object) -> ConfluenceConfig:
    values = {
        "score_threshold": 40,
        "minimum_directional_modules": 2,
        "weights": {"one": 1.0, "two": 1.0, "volatility_regime": 0.0},
    }
    values.update(overrides)
    return ConfluenceConfig(**values)


def test_paper_builds_structural_trade_idea() -> None:
    idea = ConfluenceEngine(modules(), config()).evaluate(context(), TradingMode.PAPER)
    band = config()

    assert idea.approved
    assert idea.direction is not None and idea.direction.name == "LONG"
    assert idea.stop_loss < idea.entry < idea.take_profit
    # Inside the band rather than pinned to the ceiling. The search picks the
    # payable distance with the best expectancy PER BAR, so the multiple it
    # lands on is a property of the market and not of `target_r_multiple`.
    achieved = (idea.take_profit - idea.entry) / (idea.entry - idea.stop_loss)
    assert band.minimum_r_multiple - 1e-9 <= achieved <= band.target_r_multiple + 1e-9


class TestTheTargetIsChosenForWhenItArrivesNotOnlyForHowBigItIs:
    """Maximising expectancy per TRADE systematically prefers the far target,
    because a bigger target needs a lower hit rate to pay. With the ceiling at
    3.0 the search kept choosing 3R, and 3R on an H1 plan is about nine hours of
    travel under the square law — against eight or nine in a whole session.

    56 of 140 live setups in one six-hour window died on INSUFFICIENT_RUNWAY.
    Not refused on their merits: refused because the target the search chose
    could not be reached before the day ended. A session has a deadline, so the
    quantity to maximise is expectancy per unit of time.
    """

    def test_the_nearer_payable_target_is_taken(self) -> None:
        near = ConfluenceEngine(modules(), config()).evaluate(context(), TradingMode.PAPER)
        far = ConfluenceEngine(modules(), config(prefer_sooner_targets=False)).evaluate(
            context(), TradingMode.PAPER
        )

        assert near.approved and far.approved
        near_r = (near.take_profit - near.entry) / (near.entry - near.stop_loss)
        far_r = (far.take_profit - far.entry) / (far.entry - far.stop_loss)

        assert near_r < far_r

    def test_it_still_cannot_choose_an_unpayable_distance(self) -> None:
        """This changes WHICH payable distance is chosen, never whether an
        unpayable one can be. A market that reaches nothing is still refused."""
        idea = ConfluenceEngine(modules(), config()).evaluate(
            context(step=0.00002), TradingMode.PAPER
        )

        assert not idea.approved

    def test_the_note_says_how_long_the_target_should_take(self) -> None:
        """The operator reading the journal has to be able to see the trade-off
        that was made, not only the multiple that came out of it."""
        idea = ConfluenceEngine(modules(), config()).evaluate(context(), TradingMode.PAPER)

        assert "bars" in idea.reason


def test_a_target_the_market_never_reaches_is_trimmed_or_refused() -> None:
    """`entry + 2R` is arithmetic; it never asks whether the market goes there.

    On a slow instrument that produced a target reached about once a month, so
    the trade was really a bet on the stop not being hit and the reward half of
    the reward-to-risk never arrived. The distance is now also measured against
    the instrument's own favourable excursion over the horizon.
    """
    slow = ConfluenceEngine(modules(), config()).evaluate(context(step=0.00002), TradingMode.PAPER)

    assert not slow.approved
    # And it names the right complaint. On a market this slow almost nothing
    # resolves inside the horizon — neither the target nor the stop is reached
    # — so what has failed is the MEASUREMENT, not the plan. Printing "no
    # target pays on this market" there sends the next diagnosis looking for a
    # target problem that does not exist.
    assert "cannot judge" in slow.reason
    assert "ran out of time" in slow.reason


def test_a_trimmed_target_still_clears_the_minimum() -> None:
    """Between the two bounds the target shrinks rather than being refused."""
    idea = ConfluenceEngine(modules(), config()).evaluate(context(step=0.00025), TradingMode.PAPER)

    if idea.approved:
        achieved = (idea.take_profit - idea.entry) / (idea.entry - idea.stop_loss)
        assert config().minimum_r_multiple <= achieved <= config().target_r_multiple


def test_live_blocks_when_no_module_is_validated() -> None:
    idea = ConfluenceEngine(modules(), config()).evaluate(context(), TradingMode.MICRO_LIVE)

    assert not idea.approved
    assert "no modules validated for live" in idea.reason


def test_live_uses_only_explicitly_validated_modules() -> None:
    cfg = config(live_enabled_modules=("one", "two"))
    idea = ConfluenceEngine(modules(), cfg).evaluate(context(), TradingMode.MICRO_LIVE)

    assert idea.approved


def test_disagreement_blocks() -> None:
    disagreeing = modules()
    disagreeing[1] = StubModule(Signal("two", -80, 0.9, invalidation_price=1.1140))
    idea = ConfluenceEngine(disagreeing, config()).evaluate(context(), TradingMode.PAPER)

    assert not idea.approved


def test_direction_vote_uses_measured_strength_not_only_static_weight() -> None:
    reads = [
        StubModule(Signal("one", 45, 0.50, invalidation_price=1.1050)),
        StubModule(Signal("two", -90, 0.90, invalidation_price=1.1140)),
    ]
    cfg = config(
        minimum_directional_modules=1,
        minimum_agreement_ratio=0.60,
        weights={"one": 1.0, "two": 0.7},
    )

    idea = ConfluenceEngine(reads, cfg).evaluate(context(step=-0.0006), TradingMode.PAPER)

    assert idea.direction is Direction.SHORT


def test_every_live_enabled_module_has_a_hypothesis_document() -> None:
    """Pre-registration is not optional for anything spending real money.

    A module reaching live without a written mechanism — without a named
    counterparty losing money on the other side — is data mining with an
    account attached. This test is the enforcement.
    """
    from pathlib import Path

    from config.loader import DEFAULT_CONFIG_PATH, load_settings

    root = DEFAULT_CONFIG_PATH.parent.parent
    for overlay in (None, DEFAULT_CONFIG_PATH.parent / "eightcap.yaml"):
        settings = load_settings(overlay=overlay, env_overrides=False)
        for module in settings.analysis.confluence.live_enabled_modules:
            document = Path(root / "docs" / "hypotheses" / f"{module}.md")
            assert (
                document.exists()
            ), f"{module} is live-enabled but has no docs/hypotheses/{module}.md"


def test_weighted_modules_are_documented() -> None:
    """Anything carrying weight in research needs its reasoning on paper too."""

    from config.loader import DEFAULT_CONFIG_PATH, load_settings

    root = DEFAULT_CONFIG_PATH.parent.parent
    settings = load_settings(env_overrides=False)
    for module, weight in settings.analysis.confluence.weights.items():
        if weight <= 0:
            continue
        assert (root / "docs" / "hypotheses" / f"{module}.md").exists(), module
        assert (root / "docs" / "modules" / f"{module}.md").exists(), module


class TestRejectionCarriesItsScore:
    """A rejected idea must report the score it reached, not a flat zero.

    Blanking it made "the modules saw nothing" and "the modules saw something
    and the threshold is out of reach" indistinguishable in the journal — both
    land as NO_SIGNAL — which is precisely the question asked after a day with
    no trades. Twelve hours of live decisions could not answer it.
    """

    def test_a_below_threshold_score_is_reported(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from analysis.confluence import ConfluenceEngine
        from config.loader import load_settings
        from core.types import Signal

        class _Fires:
            name = "trend_momentum"

            def analyze(self, ctx):  # type: ignore[no-untyped-def]
                return Signal(
                    module="trend_momentum",
                    score=65.0,
                    # 65 * 0.7 = 45.5, under the 55 bar. Above the lone-module
                    # confidence floor on purpose: this test is about a score
                    # rejection carrying its numbers, and a detector refused for
                    # being an unconvinced loner never reaches that check.
                    confidence=0.7,
                    reasoning="test",
                    invalidation_price=1.0,
                )

        settings = load_settings(env_overrides=False)
        config = settings.analysis.confluence.model_copy(
            update={"score_threshold": 55.0, "minimum_directional_modules": 1}
        )
        engine = ConfluenceEngine([_Fires()], config)
        idea = engine.evaluate(_context(), settings.system.mode)

        assert not idea.approved
        assert "below threshold" in idea.reason
        assert idea.score > 0.0, "the score reached must survive the rejection"
        assert idea.confidence > 0.0


def _context():  # type: ignore[no-untyped-def]
    """A minimal context with enough H1 history for the engine to score."""
    from datetime import UTC, datetime

    import numpy as np
    import pandas as pd

    from core.types import MarketContext, Series, Tick, Timeframe

    now = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
    closes = np.linspace(1.10, 1.11, 200)
    index = pd.date_range(end=now, periods=200, freq=Timeframe.H1.duration, tz=UTC)
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.0005,
            "low": closes - 0.0005,
            "close": closes,
            "tick_volume": np.full(200, 100),
        },
        index=index,
    )
    return MarketContext(
        symbol="EURUSD",
        now=now,
        series={Timeframe.H1: Series("EURUSD", Timeframe.H1, frame, now)},
        tick=Tick(symbol="EURUSD", time=now, bid=1.1100, ask=1.1101),
    )


class TestATrendPremiseContradictedByAMeasuredRange:
    """`market_regime` sorts every market into trend_up, trend_down, range,
    transition or extreme. It was computed, sent to the reviewer, cited by the
    reviewer in refusal after refusal, and never read by the engine — which
    checks only `volatility_regime`, for "extreme".

    Three live refusals in one session, in the reviewer's own words: "the
    regime module explicitly flags 'range' with low efficiency (0.08 H1, 0.11
    H4) — this is chop, not a trend", and "Market_regime independently flags
    this as a range, not a trend, which undermines the trend-continuation
    premise the whole idea is built on".
    """

    def regime(self, reading: str) -> StubModule:
        return StubModule(Signal("market_regime", 0.0, 1.0, "regime", details={"regime": reading}))

    def engine(self, **overrides) -> ConfluenceEngine:  # type: ignore[no-untyped-def]
        config = ConfluenceConfig(
            score_threshold=35.0,
            minimum_directional_modules=1,
            weights={"trend_momentum": 1.0, "liquidity_sweep": 0.8, "market_regime": 0.0},
            **overrides,
        )
        return ConfluenceEngine(self.modules, config)  # type: ignore[arg-type]

    def with_modules(self, *modules: StubModule) -> None:
        self.modules = list(modules)

    def trend(self) -> StubModule:
        return StubModule(Signal("trend_momentum", 70, 0.8, invalidation_price=1.1050))

    def sweep(self) -> StubModule:
        return StubModule(Signal("liquidity_sweep", 70, 0.8, invalidation_price=1.1050))

    def test_trend_momentum_alone_is_refused_in_a_measured_range(self) -> None:
        self.with_modules(self.trend(), self.regime("range"))

        idea = self.engine().evaluate(context(), TradingMode.PAPER)

        assert idea.direction is None
        assert "measures a range" in idea.reason

    def test_the_same_setup_passes_when_the_regime_is_a_trend(self) -> None:
        self.with_modules(self.trend(), self.regime("trend_up"))

        assert self.engine().evaluate(context(), TradingMode.PAPER).direction is not None

    def test_a_range_setup_is_welcome_in_a_range(self) -> None:
        """A liquidity sweep looks for a failed break. Refusing it in a range
        would be banning the one module whose premise the regime supports.

        Asserted on the reason rather than on a direction: a sweep is an
        intraday setup and this fixture carries no M15, so it stops later for
        an unrelated cause. What matters here is that it is not THIS gate.
        """
        self.with_modules(self.sweep(), self.regime("range"))

        idea = self.engine().evaluate(context(), TradingMode.PAPER)

        assert "measures a range" not in idea.reason

    def test_company_no_longer_rescues_a_contradicted_trend_module(self) -> None:
        """The old rule was "the ONLY firing modules assert a trend", so any
        other detector joining in switched the check off completely.

        NZDJPY LONG, 18 August, is what that bought. Regime `range`. Two modules
        agreed: `trend_momentum` at +65 — describing itself as "H1 bullish with
        H4 neutral, unconfirmed by the bias timeframe" — and `impulse_break` at
        +60, which is not on the continuation list. `all(...)` was therefore
        False, the check never ran, and 3.83% of the account went long into a
        measured range. It never printed a positive tick.

        The range does not go away because a second module is standing next to
        it. The contradicted CONTRIBUTION is dropped and everything downstream
        judges what is left — which is the sweep, on its own, here.
        """
        self.with_modules(self.trend(), self.sweep(), self.regime("range"))

        idea = self.engine().evaluate(context(), TradingMode.PAPER)

        # Not refused BY this gate — the sweep's premise is not contradicted —
        # but no longer carried by the module that was.
        assert "measures a range" not in idea.reason
        assert "trend_momentum" not in idea.reason

    def test_the_setup_dies_when_only_the_contradicted_module_was_holding_it(self) -> None:
        """The live shape, reduced: drop what the regime disagrees with and see
        whether anything is actually left. Here the survivor is a lone detector
        under the confidence a lone detector needs, so the trade is refused by a
        rule that already existed rather than by a new one."""
        unsure = StubModule(Signal("liquidity_sweep", 70, 0.50, invalidation_price=1.1050))
        self.with_modules(self.trend(), unsure, self.regime("range"))

        idea = self.engine().evaluate(context(), TradingMode.PAPER)

        assert idea.direction is None
        assert "only detector pointing this way" in idea.reason

    def test_a_transition_contradicts_a_continuation_claim_too(self) -> None:
        """`transition` was read as "no measurement, no veto", and it is the
        sharper of the two objections.

        It is the classifier's leftover branch — not extreme, not a trend, not
        a range — and what it MEANS is that the two timeframes do not agree.
        `trend_momentum` INFERS a trend from EMA alignment across exactly those
        frames, so a transition contradicts it directly.

        EURAUD said it out loud on 18 August. `trend_momentum` at -65, its own
        detail reading "H1 EMA/momentum bearish with H4 neutral — unconfirmed
        by the bias timeframe", counted at full strength in a transition. The
        module named the condition and nothing read it.
        """
        self.with_modules(self.trend(), self.regime("transition"))

        idea = self.engine().evaluate(context(), TradingMode.PAPER)

        assert idea.direction is None
        assert "measures a transition" in idea.reason

    def test_a_trend_regime_still_supports_a_continuation_claim(self) -> None:
        """Only the two readings that say "there is no trend here" contradict.
        `extreme` is already refused outright at the top of `evaluate`."""
        self.with_modules(self.trend(), self.regime("trend_up"))

        assert self.engine().evaluate(context(), TradingMode.PAPER).direction is not None

    def test_a_missing_regime_module_changes_nothing(self) -> None:
        self.with_modules(self.trend())

        assert self.engine().evaluate(context(), TradingMode.PAPER).direction is not None

    def test_the_switch_turns_it_off(self) -> None:
        self.with_modules(self.trend(), self.regime("range"))

        idea = self.engine(refuse_trend_continuation_in_range=False).evaluate(
            context(), TradingMode.PAPER
        )

        assert idea.direction is not None


class TestHorizonsDoNotOutvoteEachOther:
    """The 3,894-refusals-a-day bug, and the direct cause of "why no shorts".

    The vote used to be one weight sum across every firing module, regardless
    of what clock it read. A market trending up on H4/H1 while selling hard for
    the last two hours produced trend_momentum LONG at weight 1.0 against
    drift_continuation SHORT at 0.7: direction LONG, agreement 58.8%, below the
    60% floor, whole setup discarded. Neither trade was taken.
    """

    @staticmethod
    def _split_context(*, falling: bool = True) -> MarketContext:
        """M15 as well as H1, so an intraday plan has a chart to be built on.

        `falling` has to match what the modules are being made to say. The
        entry-timing gate reads the fast charts directly, so a fixture whose
        M15 plummets while the stub modules both claim LONG is refused for
        contradicting itself — correctly, and for a reason that has nothing to
        do with the vote this class is about.
        """
        base = context()
        index = pd.date_range("2026-01-01", periods=200, freq="15min", tz=UTC)
        # Fast enough that the target clears the stop with ROOM, otherwise the
        # setup dies on target reachability instead of on the thing under test.
        # It was 0.0003, which covered exactly 1.00R inside the horizon and
        # nothing beyond it, so lifting the planning floor by a tenth of an R
        # broke four tests about the vote. A fixture tuned to a boundary tests
        # the boundary, whatever its docstring says.
        step = 0.0006 if falling else -0.0006
        close = pd.Series([1.1100 + (199 - i) * step for i in range(200)], index=index)
        frame = pd.DataFrame(
            {
                "open": close + 0.00005,
                "high": close + 0.0003,
                "low": close - 0.0003,
                "close": close,
                "tick_volume": 100,
                "spread": 10,
                "real_volume": 0,
            }
        )
        return MarketContext(
            symbol=base.symbol,
            now=base.now,
            series={**base.series, Timeframe.M15: Series("EURUSD", Timeframe.M15, frame, base.now)},
            tick=base.tick,
        )

    @staticmethod
    def _disagreeing() -> list[StubModule]:
        return [
            StubModule(Signal("trend_momentum", 65, 0.8, invalidation_price=1.1050)),
            StubModule(Signal("drift_continuation", -55, 0.8, invalidation_price=1.1115)),
            StubModule(Signal.neutral("volatility_regime")),
        ]

    @staticmethod
    def _config(**overrides: object) -> ConfluenceConfig:
        values: dict[str, object] = {
            "score_threshold": 26,
            "minimum_directional_modules": 1,
            "minimum_agreement_ratio": 0.6,
            "weights": {
                "trend_momentum": 1.0,
                "drift_continuation": 0.7,
                "volatility_regime": 0.0,
            },
        }
        values.update(overrides)
        return ConfluenceConfig(**values)

    def test_a_fast_short_survives_a_slow_long(self) -> None:
        """The whole point. 1.0 against 0.7 used to bin the setup at 58.8%
        agreement; the fast group is now coherent on its own at 100%."""
        idea = ConfluenceEngine(self._disagreeing(), self._config()).evaluate(
            self._split_context(), TradingMode.PAPER
        )

        assert idea.approved
        assert idea.direction is Direction.SHORT

    def test_it_is_planned_as_the_intraday_trade_it_is(self) -> None:
        """A two-hour thesis must not be handed H1 planning authority and a
        target a day out — that is not the trade the module found."""
        idea = ConfluenceEngine(self._disagreeing(), self._config()).evaluate(
            self._split_context(), TradingMode.PAPER
        )

        assert idea.horizon == "intraday"
        assert idea.planning_timeframe == "M15"

    def test_the_slower_reading_travels_with_it_instead_of_killing_it(self) -> None:
        """The reviewer is the right place to weigh "the hourly chart
        disagrees" — it has the whole payload and this engine does not."""
        idea = ConfluenceEngine(self._disagreeing(), self._config()).evaluate(
            self._split_context(), TradingMode.PAPER
        )

        assert "trend_momentum" in idea.reason
        assert "different horizons" in idea.reason

    def test_agreement_between_intraday_and_swing_is_unchanged(self) -> None:
        """The new isolation is specific to complete quick entry events."""
        agreeing = [
            StubModule(Signal("trend_momentum", 65, 0.8, invalidation_price=1.1050)),
            StubModule(Signal("drift_continuation", 55, 0.8, invalidation_price=1.1060)),
            StubModule(Signal.neutral("volatility_regime")),
        ]

        idea = ConfluenceEngine(agreeing, self._config()).evaluate(
            self._split_context(falling=False), TradingMode.PAPER
        )

        assert idea.approved
        assert idea.direction is Direction.LONG
        assert idea.horizon == "swing"
        assert "different horizons" not in idea.reason

    def test_a_quick_short_is_not_blended_with_an_intraday_long(self) -> None:
        modules = [
            StubModule(Signal("m1_micro_breakout", -62, 0.8, invalidation_price=1.1115)),
            StubModule(Signal("drift_continuation", 55, 0.8, invalidation_price=1.1050)),
            StubModule(Signal("trend_momentum", 65, 0.8, invalidation_price=1.1050)),
        ]
        config = self._config(
            weights={
                "m1_micro_breakout": 0.55,
                "drift_continuation": 0.7,
                "trend_momentum": 1.0,
            }
        )

        context = self._split_context()
        context = MarketContext(
            symbol=context.symbol,
            now=context.now,
            series={**context.series, Timeframe.M5: context.series[Timeframe.M15]},
            tick=context.tick,
        )
        idea = ConfluenceEngine(modules, config).evaluate(context, TradingMode.PAPER)

        assert idea.approved
        assert idea.direction is Direction.SHORT
        assert idea.horizon == "quick"
        assert idea.planning_timeframe == "M5"

    def test_a_weak_quick_flash_does_not_hide_a_qualified_swing(self) -> None:
        """Firing is detection, not automatic ownership of the symbol.

        The quick read is real enough to report but cannot clear the same
        confidence-discounted score required of every other setup. The old
        fastest-wins rule selected it anyway and rejected the whole symbol,
        hiding the independently qualified swing underneath.
        """
        modules = [
            StubModule(Signal("m1_micro_breakout", -30, 0.5, invalidation_price=1.1115)),
            StubModule(Signal("trend_momentum", 65, 0.8, invalidation_price=1.1050)),
        ]
        config = self._config(weights={"m1_micro_breakout": 0.55, "trend_momentum": 1.0})

        idea = ConfluenceEngine(modules, config).evaluate(
            self._split_context(falling=False), TradingMode.PAPER
        )

        assert idea.approved, idea.reason
        assert idea.direction is Direction.LONG
        assert idea.horizon == "swing"
        assert "did not hide a qualified slower setup" in idea.reason

    def test_a_weak_intraday_read_does_not_hide_a_qualified_opposite_swing(self) -> None:
        modules = [
            StubModule(Signal("drift_continuation", -30, 0.5, invalidation_price=1.1115)),
            StubModule(Signal("trend_momentum", 65, 0.8, invalidation_price=1.1050)),
        ]

        idea = ConfluenceEngine(modules, self._config()).evaluate(
            self._split_context(falling=False), TradingMode.PAPER
        )

        assert idea.approved, idea.reason
        assert idea.direction is Direction.LONG
        assert idea.horizon == "swing"
        assert "qualified while the opposing intraday read" in idea.reason

    def test_a_lone_group_behaves_exactly_as_before(self) -> None:
        """No swing module firing at all is the common case, and it must not
        have been disturbed by any of this."""
        only_fast = [
            StubModule(Signal("drift_continuation", -55, 0.8, invalidation_price=1.1115)),
            StubModule(Signal.neutral("volatility_regime")),
        ]

        idea = ConfluenceEngine(only_fast, self._config()).evaluate(
            self._split_context(), TradingMode.PAPER
        )

        assert idea.approved
        assert idea.direction is Direction.SHORT
        assert "different horizons" not in idea.reason

    def test_the_higher_timeframe_veto_still_stands(self) -> None:
        """The fast side winning a disagreement is not the fast side winning
        everything. An intraday trade fighting both H4 and D1 is still refused,
        and that guard is what makes the rest of this survivable."""
        from config.schema import HorizonProfileConfig

        profiles = dict(ConfluenceConfig().horizon_profiles)
        profiles["intraday"] = profiles["intraday"].model_copy(
            update={"htf_trend_timeframes": ("H1",), "minimum_htf_conflicts": 1}
        )
        assert isinstance(profiles["intraday"], HorizonProfileConfig)

        idea = ConfluenceEngine(
            self._disagreeing(), self._config(horizon_profiles=profiles)
        ).evaluate(self._split_context(), TradingMode.PAPER)

        assert not idea.approved


class TestALoneDetectorHasToBeSureOfItself:
    """HK50 SHORT: one unconvinced opinion took EUR 3.13 of a EUR 182 account.

    `impulse_break` fired alone at confidence 0.45 — the `minimum_confidence`
    floor exactly — and scored 60 x 0.45 = 27.0 against a 26.0 bar. It never
    printed a positive tick and returned -0.56R.

    For a single module the score IS `|raw| x confidence`: the weight cancels,
    because numerator and denominator both run over the agreeing modules. So the
    threshold cannot tell a detector that is sure from one that is not, and with
    only one reading there is nothing to corroborate or contradict it.

    Raising `minimum_directional_modules` to 2 would also have stopped this, and
    would have cost every lone detector reading 0.90 as well — the strongest
    single piece of evidence this engine can produce. This gate is aimed at the
    actual failure: lone AND unconvinced.
    """

    @staticmethod
    def _engine(confidence: float, *, modules: int = 1):  # type: ignore[no-untyped-def]
        from analysis.confluence import ConfluenceEngine
        from config.loader import load_settings
        from core.types import Signal

        class _Detector:
            def __init__(self, name: str) -> None:
                self.name = name

            def analyze(self, ctx):  # type: ignore[no-untyped-def]
                return Signal(
                    module=self.name,
                    score=60.0,
                    confidence=confidence,
                    reasoning="test",
                    invalidation_price=1.0,
                )

        settings = load_settings(env_overrides=False)
        names = ["trend_momentum", "market_structure"][:modules]
        config = settings.analysis.confluence.model_copy(
            update={
                "score_threshold": 26.0,
                "minimum_directional_modules": 1,
                "lone_module_minimum_confidence": 0.65,
                "weights": dict.fromkeys(names, 1.0),
            }
        )
        engine = ConfluenceEngine([_Detector(name) for name in names], config)
        return engine.evaluate(_context(), settings.system.mode)

    def test_the_hk50_shape_is_refused(self) -> None:
        idea = self._engine(confidence=0.45)

        assert not idea.approved
        assert "only detector pointing this way" in idea.reason

    def test_a_lone_detector_that_is_sure_still_trades(self) -> None:
        """The half `minimum_directional_modules: 2` would have thrown away."""
        idea = self._engine(confidence=0.90)

        assert "only detector pointing this way" not in idea.reason

    def test_two_agreeing_detectors_are_never_asked(self) -> None:
        """Corroboration is the thing the floor stands in for. Once it exists,
        the floor must not also apply — that would be paying for it twice."""
        idea = self._engine(confidence=0.45, modules=2)

        assert "only detector pointing this way" not in idea.reason

    def test_it_can_be_switched_off(self) -> None:
        from config.schema import ConfluenceConfig

        assert ConfluenceConfig(lone_module_minimum_confidence=0.0)

    def test_the_overlay_sets_it(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        overlay = DEFAULT_CONFIG_PATH.parent / "eightcap.yaml"
        settings = load_settings(overlay=overlay, env_overrides=False)

        assert settings.analysis.confluence.lone_module_minimum_confidence == 0.65
        # The blunt alternative stays off: a sure lone detector is still allowed.
        assert settings.analysis.confluence.minimum_directional_modules == 1


class TestTheTargetGoesWhereItPays:
    """A fixed multiple never asks whether the distance is worth aiming at.

    `runs` is the whole empirical distribution of how far this market travels.
    The code reduced it to one quantile, capped the plan at it, and never asked
    the question the trade turns on: how often is THIS distance reached, and
    does that beat what it costs to find out.

    At 1.2R a market must win 45.5% before costs. Whether it delivers 30% or
    60% is a property of the instrument that a multiple cannot consult, so a
    market reaching 1.2R half the time and one reaching it a fifth of the time
    were handed the same target — and a market paying handsomely at 1.0R but
    not at 1.2R was refused for missing a floor.

    Both failures cost trades and cost money at once, which is why the search
    below is not a loosening: it can only ever pick a distance whose measured
    expectancy is positive.
    """

    @staticmethod
    def _idea(step: float):  # type: ignore[no-untyped-def]
        engine = ConfluenceEngine(modules(), config(minimum_r_multiple=0.5))
        return engine.evaluate(context(step=step), TradingMode.PAPER)

    def test_a_market_that_pays_gets_a_target_and_the_arithmetic_behind_it(self) -> None:
        idea = self._idea(step=0.0004)

        assert idea.approved, idea.reason
        assert "reached first" in idea.reason
        assert "expected" in idea.reason

    def test_a_market_that_does_not_pay_is_refused_with_the_numbers(self) -> None:
        """Not a floor it missed — the measurement that decides it."""
        idea = self._idea(step=0.00002)

        assert not idea.approved
        assert "cannot judge" in idea.reason
        assert "too wide for the time the plan has" in idea.reason

    def test_the_reach_is_first_touch_and_not_the_favourable_excursion(self) -> None:
        """The measurement that makes this honest rather than merely different.

        A plain favourable excursion counts a window where price fell a full R
        and only then rallied. That trade was already stopped out, so counting
        it inflates every reach rate — most in exactly the volatile markets
        where the inflation matters. This walks each window to the first bar
        that would have taken the stop and measures only up to there.
        """
        import numpy as np
        import pandas as pd

        from analysis.confluence import ConfluenceEngine
        from core.types import Direction

        # One window: price drops through the stop, then rallies far beyond it.
        frame = pd.DataFrame(
            {
                "high": [100.0, 100.2, 100.1, 105.0, 105.0],
                "low": [100.0, 100.1, 97.0, 100.0, 100.0],
            }
        )
        closes = np.array([100.0, 100.2, 100.1, 105.0, 105.0])

        outcomes = ConfluenceEngine._first_touch_outcomes(
            frame, closes, Direction.LONG, risk=1.0, horizon=4
        )

        assert outcomes is not None
        # Without the stop the answer would be 5.0. The trade was dead first.
        assert outcomes.run[0] < 1.0
        # And the window is recorded as STOPPED, not merely as "did not reach".
        # Pricing those two alike is what refused every payable target.
        assert bool(outcomes.stopped[0])

    def test_a_frame_without_highs_falls_back_rather_than_inventing(self) -> None:
        import numpy as np
        import pandas as pd

        from analysis.confluence import ConfluenceEngine
        from core.types import Direction

        bare = pd.DataFrame({"close": [1.0, 2.0, 3.0]})

        assert (
            ConfluenceEngine._first_touch_outcomes(
                bare, np.array([1.0, 2.0, 3.0]), Direction.LONG, risk=1.0, horizon=1
            )
            is None
        )


class TestTheNzdjpyShape:
    """The live trade that showed the hole, reproduced end to end.

    NZDJPY LONG at 01:19 on 18 August. Regime `range`. `trend_momentum` at +65
    with confidence 0.68, its own detail reading "H1 EMA/momentum bullish with
    H4 neutral — unconfirmed by the bias timeframe". `impulse_break` at +60 with
    confidence 0.63. Score 41.5 against a 26.0 bar, 3.83% of the account, and it
    never printed a positive tick: best +0.00R, worst -0.36R, out at -0.33R.

    Two things let it through and both are fixed here. The range check tested
    `all(...)` over the agreeing modules, so `impulse_break` standing alongside
    switched it off entirely. And the check ran after the lone-module floor, so
    even discounting `trend_momentum` afterwards would have been too late.
    """

    @staticmethod
    def _engine():  # type: ignore[no-untyped-def]
        config = ConfluenceConfig(
            score_threshold=26.0,
            minimum_directional_modules=1,
            lone_module_minimum_confidence=0.65,
            trend_continuation_modules=("trend_momentum",),
            weights={"trend_momentum": 1.0, "impulse_break": 1.0, "market_regime": 0.0},
        )
        modules = [
            StubModule(Signal("trend_momentum", 65, 0.68, invalidation_price=1.1050)),
            StubModule(Signal("impulse_break", 60, 0.63, invalidation_price=1.1050)),
            StubModule(Signal("market_regime", 0.0, 1.0, "regime", details={"regime": "range"})),
        ]
        return ConfluenceEngine(modules, config)  # type: ignore[arg-type]

    def test_it_is_refused_now(self) -> None:
        idea = self._engine().evaluate(context(), TradingMode.PAPER)

        assert idea.direction is None

    def test_and_the_refusal_names_the_survivor_not_the_regime(self) -> None:
        """Dropping the contradicted contribution leaves `impulse_break` alone
        at 0.63, under the 0.65 a lone detector needs. The trade dies on a rule
        that already existed, which is the point: the regime check only had to
        stop counting evidence it disagrees with."""
        idea = self._engine().evaluate(context(), TradingMode.PAPER)

        assert "impulse_break is the only detector" in idea.reason
        assert "0.63" in idea.reason

    def test_the_same_two_modules_are_fine_outside_a_range(self) -> None:
        """The regime is doing the work, not a new dislike of these modules."""
        config = ConfluenceConfig(
            score_threshold=26.0,
            minimum_directional_modules=1,
            lone_module_minimum_confidence=0.65,
            trend_continuation_modules=("trend_momentum",),
            weights={"trend_momentum": 1.0, "impulse_break": 1.0, "market_regime": 0.0},
        )
        modules = [
            StubModule(Signal("trend_momentum", 65, 0.68, invalidation_price=1.1050)),
            StubModule(Signal("impulse_break", 60, 0.63, invalidation_price=1.1050)),
            StubModule(Signal("market_regime", 0.0, 1.0, "regime", details={"regime": "trend_up"})),
        ]

        idea = ConfluenceEngine(modules, config).evaluate(  # type: ignore[arg-type]
            context(), TradingMode.PAPER
        )

        assert idea.direction is not None


class TestTheTargetBandHasToContainAPayableDistance:
    """A closer target does not fix a reach problem — it creates one.

    The band was 1.00R to 1.20R, which needs a 49-54% win rate to break even
    once costs are counted. `why_no_trades` over twelve live hours shows what
    these markets actually do: AUDCHF reaches 1.00R first 26% of the time,
    CADCHF 37%, NZDCHF 6%, AUDCAD 3%. Roughly 11,800 refusals in that window
    read "no target between 1.00R and 1.20R pays on this market" — the second
    largest blockage in the entire funnel.

    None of those gaps close by aiming nearer. At 2.0R the requirement falls to
    36% and CADCHF clears it.

    Raising the ceiling is not a loosening because the distance is CHOSEN by
    measured expectancy, not set by the multiple: the search cannot pick a point
    whose expectancy is non-positive. The ceiling only says how far it may look.
    """

    @staticmethod
    def _band():  # type: ignore[no-untyped-def]
        return TestTheTargetBandHasToContainAPayableDistance._settings().analysis.confluence

    @staticmethod
    def _settings():  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml", env_overrides=False
        )

    def test_the_band_reaches_a_realistic_win_rate(self) -> None:
        """At the top of the band the required hit rate must be one these
        markets actually produce, or the band cannot contain a trade."""
        band = self._band()
        needed = (1 + 0.05) / (1 + band.target_r_multiple - 0.05)

        assert needed < 0.40, f"top of the band still needs {needed:.0%}"

    def test_the_floor_is_where_the_owner_put_it_with_the_risk_stated(self) -> None:
        """0.60, on the owner's explicit approval after both objections were put
        to him in numbers.

        Against: over 180 days and 20 markets not one detector is positive —
        trend_momentum -0.251R over 148 trades at t = -3.05, and the best row in
        the table is session_breakout at +0.145R over 33, which is still noise.
        More setups out of detectors without a measured edge is more losing.

        Also against: at these stop widths the round trip takes 27% of the risk
        on 7 pips and 63% on 3.

        He read both and accepted the risk, so this is his decision recorded
        rather than an argument re-run. What keeps it honest is that it is not a
        gate opening: the search still only picks a distance with positive
        MEASURED first-touch expectancy, and since today that calculation
        carries the whole bill — spread, commission and slippage.

        `scorecard --days 3` is the check. Trades dying on costs rather than on
        the market is the prediction coming true, and it is one line back.

        0.75 RATHER THAN 0.60, and the difference is not a change of mind about
        the risk above — the EXECUTION floor is still the 0.60 he approved. The
        planning floor has to sit above it. The search lands on its own floor
        on nearly every market, so with both numbers at 0.60 every plan reached
        the sizer at exactly 0.60R with nothing to spare, and the quote moves in
        between: NDX100 SHORT on 19 August was approved, reviewed, and then
        refused as `RR_BELOW_MINIMUM: reward:risk is 1:0.52`.

        AND ON 26 AUGUST BOTH CAME DOWN AGAIN, 0.75 -> 0.35 and 0.60 -> 0.30,
        because the first module backtest showed the floor was the defect.

        All eight detectors came back with the SAME shape over 2,192
        proposals: 54-57% win, average winner +0.68R, average loser -1.04R.
        Eight readers that agree on nothing else do not have eight separate
        problems. The direction is genuinely better than a coin; the ratio
        0.68/1.04 needs 60.5% and does not get it, because the target sat at
        0.72R against a 1.00R stop -- which is where this floor held it.

        What the give-back table says over 1,970 trades:

            target   reached   per trade
             0.30R     80%      -0.001R
             0.35R     77%      -0.002R
             0.72R     56%      -0.070R   <- where the floor held it
             1.00R     10%      -0.840R

        The engine already searches for the distance with positive measured
        expectancy. It was simply never allowed to look where the money is.
        Higher is demonstrably worse, not safer: 197 of 1,970 trades ever
        peaked above 1.00R.

        0.35 is the lowest the system permits -- the schema floor is 0.30 and
        the validator needs 0.30 x 1.15 = 0.345 of planning headroom.

        The cost objection recorded above ("27% of the risk on 7 pips") was
        right and is now acted on rather than accepted:
        `max_spread_share_of_stop` goes 0.25 -> 0.08 in the same edit. At a
        0.35R target the round trip is the whole remaining gap -- 0.040R
        leaves -0.002R a trade and 0.020R leaves +0.018R.
        """
        band = self._band()

        assert band.minimum_r_multiple == 0.35
        assert self._settings().risk.min_risk_reward == 0.30
        assert band.minimum_r_multiple >= 0.30 * (1.0 + band.target_planning_margin)
        # The cost gate is half of the same fix and must not drift back up.
        assert band.max_spread_share_of_stop <= 0.10

    def test_the_sizer_will_not_refuse_what_the_analysis_may_plan(self) -> None:
        """Two floors that must agree, and they did not: `min_risk_reward` sat
        at 2.0 above a 1.0 analysis floor in the base config, so every target
        planned between them was built, measured, scored, sized and thrown
        away. A validator on Settings refuses that combination now."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        for overlay in (None, DEFAULT_CONFIG_PATH.parent / "eightcap.yaml"):
            settings = load_settings(overlay=overlay, env_overrides=False)
            assert (
                settings.risk.min_risk_reward <= settings.analysis.confluence.minimum_r_multiple
            ), overlay

    def test_a_wider_ceiling_cannot_buy_a_losing_target(self) -> None:
        """The guarantee that makes this safe: expectancy decides, the ceiling
        only bounds the search. A market that reaches nothing is still refused."""
        idea = ConfluenceEngine(modules(), config(minimum_r_multiple=0.5)).evaluate(
            context(step=0.00002), TradingMode.PAPER
        )

        assert not idea.approved
        assert "cannot judge" in idea.reason

    def test_the_two_refusals_are_told_apart(self) -> None:
        """ "Resolved and lost" and "never resolved" are different findings.

        The first is a judgement about the plan and says fix the plan. The
        second is a statement about the window: within this horizon price
        reached neither the target nor the stop, so nothing can be concluded,
        and what needs fixing is the horizon or the width of the stop.

        Same market, same edge, only the stop widened from 4x to 32x ATR
        against a fixed 24-bar horizon: resolved windows fall 2,915 -> 7, the
        expired share climbs 2% -> 99.8%, and the expectancy goes +0.31R ->
        -0.12R purely because every expired window is charged a round trip. The
        market did not get worse; the ruler got too short. Both used to print
        the same sentence.
        """
        never = ConfluenceEngine(modules(), config()).evaluate(
            context(step=0.00002), TradingMode.PAPER
        )
        # Choppy enough that the stop is reached often, and long enough that
        # the resolved sample is a sample: this market ANSWERS the question,
        # and the answer is no.
        coin = ConfluenceEngine(modules(), config()).evaluate(
            context(step=0.0, noise=0.003, bars=800), TradingMode.PAPER
        )

        assert not never.approved and not coin.approved
        assert "cannot judge" in never.reason
        assert "pays on this market" in coin.reason
        assert "cannot judge" not in coin.reason


class TestTheLoneFloorCanBeSetPerDetector:
    """A single floor could let `fast_ema_cross` and `liquidity_sweep` through
    together or hold both back, and in the module backtest those two hold the
    worst and the best records. This is the mechanism that separates them; the
    values that go in it have to be earned by measurement.
    """

    @staticmethod
    def _engine(floors: dict[str, float] | None = None):  # type: ignore[no-untyped-def]
        from analysis.confluence import ConfluenceEngine
        from config.schema import ConfluenceConfig

        return ConfluenceEngine(
            [],
            ConfluenceConfig(
                lone_module_minimum_confidence=0.65,
                lone_module_minimum_confidence_by_module=floors or {},
            ),
        )

    def test_an_empty_table_is_the_old_behaviour_exactly(self) -> None:
        config = self._engine().config

        for module in ("liquidity_sweep", "fast_ema_cross", "anything_at_all"):
            assert config.lone_floor_for(module) == 0.65

    def test_one_detector_can_be_released_without_the_others(self) -> None:
        config = self._engine({"liquidity_sweep": 0.55}).config

        assert config.lone_floor_for("liquidity_sweep") == 0.55
        assert config.lone_floor_for("fast_ema_cross") == 0.65

    def test_one_detector_can_also_be_held_to_a_higher_bar(self) -> None:
        """The table works in both directions. `fast_ema_cross` measured -1.211R
        a trade where it was present; if that survives a larger sample, raising
        its own floor is the targeted response."""
        config = self._engine({"fast_ema_cross": 0.85}).config

        assert config.lone_floor_for("fast_ema_cross") == 0.85
        assert config.lone_floor_for("liquidity_sweep") == 0.65

    def test_a_floor_outside_zero_to_one_is_refused(self) -> None:
        import pytest as _pytest

        from config.schema import ConfluenceConfig

        with _pytest.raises(ValueError, match="between 0 and 1"):
            ConfluenceConfig(lone_module_minimum_confidence_by_module={"trend_momentum": 1.4})

    def test_every_entry_names_a_detector_with_a_live_record(self) -> None:
        """The table shipped empty and its first entry TIGHTENED, holding
        `session_breakout` above its own 0.80 ceiling until it had a number.

        It has one now, and so do the others: four days of real trades at
        +2.08, +1.30, +1.20 and +0.57 EUR apiece. That is what an entry costs,
        and it is why this asserts the table's KEYS rather than its values —
        the numbers are a judgement that will move again, the rule that only a
        measured detector gets an entry is the part that must not.
        """
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        confluence = settings.analysis.confluence
        table = confluence.lone_module_minimum_confidence_by_module

        # Every earner from the live table, and nothing else. A detector with
        # no measured record must keep the single global floor.
        assert set(table) == {
            "ema_pullback_resume",
            "impulse_break",
            "fast_ema_cross",
            "session_breakout",
        }
        # And a released detector still has to be CONVINCED. Loosening below
        # the module confidence floor would make the entry meaningless: every
        # firing would clear it and the corroboration requirement would be off
        # rather than relaxed.
        assert all(value > confluence.minimum_confidence for value in table.values())


class TestTrendMomentumIsBackOnItsNarrowedRecord:
    """Switched off on one measurement, back on because a second one disagreed
    about the part that changed.

    THE KILL, and it is not withdrawn: 20 markets, 180 days, 163 closed trades,
    -0.365R a trade, -59.45R total, t = -4.01. A real sample and a real result.

    THE SECOND MEASUREMENT, 27 August: 90 days over five markets -- the majors
    and gold -- and it is the only substantial positive population in the whole
    report:

        trend_momentum ALONE   66 trades   76% won   +0.105R   +6.94R

    The two do not contradict each other. They measure different universes, and
    this config already carries the same reasoning about `drift_continuation`:
    "the problem was never the detector, it was where it was allowed to look."
    Single-name shares left the catalogue the same day, so part of those twenty
    markets is no longer traded at all.

    WHAT THIS IS NOT: proof of an edge. Its own report scores it +0.085R against
    a coin flip and labels that inside chance. It is the best candidate in a
    report where every other live detector measured negative -- and it is armed
    with the strictest section breaker of the four, because a module going live
    against a significant negative measurement without a self-stop is not an
    experiment, it is hoping.
    """

    def _confluence(self):  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence

    def test_it_trades_live_again(self) -> None:
        assert "trend_momentum" in self._confluence().live_enabled_modules

    def test_and_it_can_stop_itself(self) -> None:
        """The condition of it being on. Eight losers in a row, or twelve of
        twenty -- against the 24% loss rate it showed in the report that put it
        back, so this fires on behaviour that is plainly different rather than
        on a bad run."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        breaker = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).risk.section_breakers["trend_momentum"]

        assert breaker.enabled
        assert breaker.losing_streak <= 8
        assert breaker.maximum_loss_share <= 0.60

    def test_but_it_is_still_measured(self) -> None:
        """Switching a module off and losing sight of it means never finding
        out whether the decision was right."""
        assert self._confluence().weights.get("trend_momentum", 0.0) > 0

    def test_the_live_engine_zeroes_it_rather_than_dropping_the_setup(self) -> None:
        """A setup that had trend_momentum AND another detector should survive
        on the other one. Only the trades it carried alone disappear — 158 of
        its 347 proposals, not all of them."""
        from analysis.confluence import ConfluenceEngine
        from config.schema import ConfluenceConfig

        config = ConfluenceConfig(
            live_enabled_modules=("liquidity_sweep",),
            weights={"trend_momentum": 1.0, "liquidity_sweep": 0.8},
        )
        engine = ConfluenceEngine([], config)

        assert "trend_momentum" in engine.config.weights
        assert "trend_momentum" not in engine.config.live_enabled_modules
        assert "liquidity_sweep" in engine.config.live_enabled_modules


class TestExtremeMeansBigAndNotMerelyRare:
    """The veto that refused a fifth of everything scanned, before any analysis.

    `volatility_regime` read a percentile of the last 100 ATRs and called the
    top five percent "extreme". That is purely relative: every market has a top
    five percent, including a dead one, so the test fires at the same rate
    whether anything is happening or not. Measured on synthetic series with
    realistic volatility clustering:

        calm, no shock at all   fires 10.4% of bars, ATR at most 1.51x median
        with a real 4x shock    fires  9.2% of bars, ATR up to 4.61x median

    Same rate, opposite situations — and it vetoes at the top of `evaluate()`,
    so nothing downstream ever gets to disagree.
    """

    @staticmethod
    def _series(scale, bars: int = 400):  # type: ignore[no-untyped-def]
        import numpy as np

        rng = np.random.default_rng(3)
        base = np.full(bars, 0.02)
        base[-20:] *= scale  # the recent stretch is `scale` times as active
        close = 100 + np.cumsum(rng.normal(0.0, 1.0, bars) * base)
        wick = np.abs(rng.normal(0.0, 1.0, bars)) * base * 0.6
        index = pd.date_range("2026-01-01", periods=bars, freq="1h", tz=UTC)
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close + wick,
                "low": close - wick,
                "close": close,
                "tick_volume": 100,
                "spread": 10,
                "real_volume": 0,
            },
            index=index,
        )
        now = index[-1].to_pydatetime()
        return MarketContext(
            symbol="EURUSD",
            now=now,
            series={Timeframe.H1: Series("EURUSD", Timeframe.H1, frame, now)},
            tick=Tick("EURUSD", now, 99.99, 100.01),
        )

    @staticmethod
    def _regime(ctx, **overrides):  # type: ignore[no-untyped-def]
        from analysis.modules import VolatilityRegime
        from config.schema import VolatilityRegimeConfig

        signal = VolatilityRegime(VolatilityRegimeConfig(**overrides)).analyze(ctx)
        return signal.details.get("regime"), signal.details.get("atr_multiple_of_median")

    def test_a_market_merely_busier_than_usual_is_not_extreme(self) -> None:
        regime, multiple = self._regime(self._series(1.3))

        assert multiple is not None and 1.0 < multiple < 2.0
        assert regime != "extreme"

    def test_a_market_that_has_genuinely_blown_out_still_is(self) -> None:
        regime, multiple = self._regime(self._series(6.0))

        assert multiple is not None and multiple >= 2.0
        assert regime == "extreme"

    def test_the_old_purely_relative_behaviour_is_one_number_away(self) -> None:
        """Kept expressible, so this is a decision and not a one-way door."""
        regime, _ = self._regime(self._series(1.3), extreme_atr_multiple=1.0)

        assert regime == "extreme"

    def test_the_reading_says_how_much_as_well_as_how_rare(self) -> None:
        """A veto this powerful has to show its working, or the next person
        reading a `NO_SIGNAL: extreme volatility regime` cannot tell whether it
        was a crash or a Tuesday."""
        from analysis.modules import VolatilityRegime
        from config.schema import VolatilityRegimeConfig

        signal = VolatilityRegime(VolatilityRegimeConfig()).analyze(self._series(6.0))

        assert "percentile" in signal.reasoning
        assert "median" in signal.reasoning


class TestARegimeThisAccountRefusesToTrade:
    """`transition` was half the book over four live days and all of the
    damage: 44 trades, -20.86 EUR, against +22.44 in `trend_up`. It is the
    classifier's leftover branch, and what it MEANS is that the timeframes
    disagree about direction — so those trades were taken into a market the
    system itself could not read.

    A hard refusal and not a score discount, because a discount only removes
    one module's contribution and the objection here is to the market rather
    than to any one reader of it.
    """

    @staticmethod
    def _modules(regime: str) -> list[StubModule]:
        return [
            *modules(),
            StubModule(Signal("market_regime", 0, 0.0, details={"regime": regime})),
        ]

    @staticmethod
    def _config(**overrides: object) -> ConfluenceConfig:
        return config(
            weights={
                "one": 1.0,
                "two": 1.0,
                "volatility_regime": 0.0,
                "market_regime": 0.0,
            },
            **overrides,
        )

    def test_a_refused_regime_is_refused_however_good_the_evidence(self) -> None:
        engine = ConfluenceEngine(
            self._modules("transition"), self._config(refused_regimes=("transition",))
        )

        idea = engine.evaluate(context(), TradingMode.PAPER)

        assert not idea.approved
        assert "transition" in idea.reason

    def test_the_same_evidence_in_a_regime_that_pays_is_taken(self) -> None:
        """Or the test above would only prove the fixture cannot trade."""
        engine = ConfluenceEngine(
            self._modules("trend_up"), self._config(refused_regimes=("transition",))
        )

        assert engine.evaluate(context(), TradingMode.PAPER).approved

    def test_range_is_refused_and_transition_is_not(self) -> None:
        """`transition` was blocked on -EUR 20.86 over 44 trades and unblocked
        again on 25 August, on the owner's instruction and on two numbers.

        The 44 trades were measured on a system that no longer exists --
        seasonality gone, the phantom index commission gone, weights raised,
        and the conviction bar moved from 26 to 60, which alone means most of
        those trades would not be taken now. And the block cost 30,382 of
        62,962 decisions in twelve hours, 48% of everything the system did,
        while transition finished 24 August as the best regime on the card at
        7 winners from 8.

        `range` took its place on the same day, and the difference is the
        evidence rather than the verdict. Transition was blocked on one summed
        number; range is negative under four detectors independently --
        drift_continuation 1 from 6, impulse_break 0 from 4, session_breakout 0
        from 2, fast_ema_cross 1 from 3 -- for -EUR 10.04 of an -EUR 11.50 two
        days. Fisher exact against the rest of the book: p = 0.0502.

        Ten trades is thin and the p-value is on the line. Pinned so both
        decisions have to be argued rather than drifted into."""
        from config.loader import load_settings

        settings = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        )

        assert settings.analysis.confluence.refused_regimes == ("range",)

    def test_an_empty_list_trades_every_regime(self) -> None:
        engine = ConfluenceEngine(self._modules("transition"), self._config())

        assert engine.evaluate(context(), TradingMode.PAPER).approved

    def test_an_unrecorded_regime_is_not_silently_blocked(self) -> None:
        """No `market_regime` signal means no reading, not a refusal — the
        fail-safe for missing data lives in the data layer, and a blocklist
        that fires on `None` would refuse every market whose classifier is
        merely disabled."""
        engine = ConfluenceEngine(modules(), self._config(refused_regimes=("transition",)))

        assert engine.evaluate(context(), TradingMode.PAPER).approved


class TestTheJournalRecordsTheWeightAModuleActuallyCarried:
    """`weights` says what a module is worth; `live_enabled_modules` says which
    modules may vote when the money is real. Live, a module off the allowlist
    is forced to zero — computed, logged, and deciding nothing.

    The engine always applied that. The journal recorded the raw `weights`
    table instead, so `trend_momentum` (weight 1.0, not live-enabled) was
    written to `module_scores` at 1.0 on every live cycle. `scorecard.py`
    credits any module with `weight > 0`, so it attributed 60 live trades to a
    detector that cast no vote in any of them — and that table is what a
    decision about which detectors to keep is made from.
    """

    @staticmethod
    def _config() -> ConfluenceConfig:
        return config(
            weights={"one": 1.0, "two": 1.0, "volatility_regime": 0.0},
            live_enabled_modules=("one",),
        )

    def test_live_zeroes_a_module_that_may_not_vote(self) -> None:
        assert self._config().effective_weights(TradingMode.MICRO_LIVE) == {
            "one": 1.0,
            "two": 0.0,
            "volatility_regime": 0.0,
        }

    def test_paper_leaves_every_weight_alone(self) -> None:
        """The allowlist gates live only. Backtests and paper have to keep
        scoring the full set or the two engines stop being comparable."""
        assert self._config().effective_weights(TradingMode.PAPER) == {
            "one": 1.0,
            "two": 1.0,
            "volatility_regime": 0.0,
        }

    def test_the_engine_scores_exactly_what_the_journal_will_record(self) -> None:
        """The bug was two places computing the same thing. This pins them to
        one answer: whatever the engine let vote is what gets written down."""
        cfg = self._config()
        engine = ConfluenceEngine(modules(), cfg)

        idea = engine.evaluate(context(), TradingMode.MICRO_LIVE)
        effective = cfg.effective_weights(TradingMode.MICRO_LIVE)

        assert not idea.approved  # `two` cannot vote, so `one` is alone
        assert effective["two"] == 0.0

    def test_a_returned_copy_cannot_edit_the_config(self) -> None:
        weights = self._config().effective_weights(TradingMode.PAPER)
        weights["one"] = 99.0

        assert self._config().weights["one"] == 1.0

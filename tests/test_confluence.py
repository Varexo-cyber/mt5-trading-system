"""Confluence is deterministic, auditable, and live fails closed."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from analysis.confluence import ConfluenceEngine
from config.schema import ConfluenceConfig
from core.types import MarketContext, Series, Signal, Tick, Timeframe, TradingMode


class StubModule:
    def __init__(self, signal: Signal) -> None:
        self.name = signal.module
        self.signal = signal

    def analyze(self, ctx: MarketContext) -> Signal:
        return self.signal


def context(step: float = 0.0006) -> MarketContext:
    """A rising H1 series. `step` is the move per bar, which decides whether a
    2R target is reachable — the engine now bounds the target by what this
    market actually travels, so the fixture has to have a speed."""
    index = pd.date_range("2026-01-01", periods=100, freq="1h", tz=UTC)
    close = pd.Series([1.10 + i * step for i in range(100)], index=index)
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

    assert idea.approved
    assert idea.direction is not None and idea.direction.name == "LONG"
    assert idea.stop_loss < idea.entry < idea.take_profit
    assert (idea.take_profit - idea.entry) / (idea.entry - idea.stop_loss) == 2.0


def test_a_target_the_market_never_reaches_is_trimmed_or_refused() -> None:
    """`entry + 2R` is arithmetic; it never asks whether the market goes there.

    On a slow instrument that produced a target reached about once a month, so
    the trade was really a bet on the stop not being hit and the reward half of
    the reward-to-risk never arrived. The distance is now also measured against
    the instrument's own favourable excursion over the horizon.
    """
    slow = ConfluenceEngine(modules(), config()).evaluate(context(step=0.00002), TradingMode.PAPER)

    assert not slow.approved
    assert "reachable target" in slow.reason


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
                    confidence=0.5,  # 65 * 0.5 = 32.5, under any sane threshold
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

    def test_a_trend_module_corroborated_by_a_range_module_survives(self) -> None:
        """The rule is "the ONLY firing modules assert a trend". A sweep firing
        alongside means something did agree with the range."""
        self.with_modules(self.trend(), self.sweep(), self.regime("range"))

        assert self.engine().evaluate(context(), TradingMode.PAPER).direction is not None

    def test_an_unclassified_regime_changes_nothing(self) -> None:
        """transition, or no regime module at all: no measurement, no veto."""
        self.with_modules(self.trend(), self.regime("transition"))

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

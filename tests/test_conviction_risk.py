"""Staking more when the reviewer is very sure, and only then.

The owner authorised this in these words: minimum 2%, up to 6-8% on a highly
convincing setup, less as the conviction falls. That is a real loosening on a
real account, so what is tested here is mostly the things that must NOT happen
as a consequence:

  - an ordinary approval must still be sized at exactly what it was before;
  - four slots at the new ceiling must not put a quarter of the account on the
    table at once;
  - a stop walked to break-even must genuinely give its budget back rather than
    merely appearing to;
  - and the anti-martingale guard must still refuse to raise risk after a loss,
    which is the one rule in this system that no authorisation can move.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from config.schema import ConvictionRiskConfig, Settings
from core.clock import SimulatedClock
from core.errors import ForbiddenStrategyError
from core.instrument import InstrumentSpec
from core.types import AccountSnapshot, Direction, Position
from journal.database import Journal
from risk.position_sizer import PositionSizer
from risk.risk_manager import RiskManager
from tests.fakes.fake_mt5 import eurusd_spec

NOW = datetime(2026, 3, 11, 14, 30, tzinfo=UTC)


@pytest.fixture
def raw() -> dict[str, Any]:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def settings_for(tmp_path: Path, raw: dict[str, Any], **overrides: Any) -> Settings:
    data = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        node: Any = data
        *path, leaf = dotted.split(".")
        for part in path:
            node = node[part]
        node[leaf] = value
    path_ = tmp_path / "config.yaml"
    path_.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_settings(path_, env_overrides=False)


@pytest.fixture
def settings(tmp_path: Path, raw: dict[str, Any]) -> Settings:
    """The live shape: micro_live, an ordinary 2% and a 6% conviction ceiling."""
    return settings_for(
        tmp_path,
        raw,
        **{
            "system.mode": "micro_live",
            "risk.risk_per_trade_pct": 2.0,
            "risk.max_risk_per_trade_pct": 6.0,
            "modes.micro_live.max_risk_per_trade_pct": 6.0,
            "risk.conviction_risk": {
                "enabled": True,
                "floor_pct": 2.0,
                "ceiling_pct": 6.0,
            },
            "risk.max_total_open_risk_pct": 12.0,
        },
    )


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(NOW)


@pytest.fixture
def journal(tmp_path: Path, clock: SimulatedClock) -> Journal:
    with Journal(tmp_path / "journal.db", clock) as j:
        yield j


@pytest.fixture
def manager(settings: Settings, journal: Journal, clock: SimulatedClock) -> RiskManager:
    return RiskManager(settings=settings, journal=journal, clock=clock)


@pytest.fixture
def spec() -> InstrumentSpec:
    return InstrumentSpec.from_mt5(eurusd_spec())


def account(equity: float) -> AccountSnapshot:
    return AccountSnapshot(
        login=1,
        server="Fake",
        currency="EUR",
        balance=equity,
        equity=equity,
        margin=0.0,
        margin_free=1e9,
        margin_level=0.0,
        leverage=500,
        is_demo=True,
        taken_at=NOW,
    )


def position(*, volume: float, entry: float = 1.0850, sl: float | None = 1.0800) -> Position:
    return Position(
        ticket=1,
        symbol="EURUSD",
        direction=Direction.LONG,
        volume=volume,
        price_open=entry,
        sl=sl,
        tp=1.0950,
        profit=0.0,
        swap=0.0,
        opened_at=NOW,
    )


class TestTheRamp:
    """What each level of reviewer confidence is worth."""

    @pytest.mark.parametrize(
        ("confidence", "expected"),
        [
            (0.00, 2.0),
            (0.55, 2.0),  # the approval bar itself buys nothing extra
            (0.70, 2.0),  # the ramp starts here
            (0.75, 3.0),
            (0.80, 4.0),
            (0.85, 5.0),
            (0.90, 6.0),
            (1.00, 6.0),  # and saturates
        ],
    )
    def test_the_authorised_shape(self, confidence: float, expected: float) -> None:
        config = ConvictionRiskConfig(enabled=True, floor_pct=2.0, ceiling_pct=6.0)

        assert config.stake_for(confidence) == pytest.approx(expected)

    def test_an_approval_that_only_just_cleared_the_bar_is_sized_as_before(self) -> None:
        """The point of the 0.70 floor. `ai.minimum_confidence` is 0.55, so
        without a gap between the two every approval would be a raise."""
        config = ConvictionRiskConfig(enabled=True, floor_pct=2.0, ceiling_pct=6.0)

        assert config.stake_for(0.56) == pytest.approx(2.0)

    def test_disabled_stakes_the_floor_whatever_the_confidence_is(self) -> None:
        config = ConvictionRiskConfig(enabled=False, floor_pct=2.0, ceiling_pct=6.0)

        assert config.stake_for(0.99) == pytest.approx(2.0)

    def test_a_ceiling_below_the_floor_is_refused_at_load(self) -> None:
        with pytest.raises(ValueError, match="ceiling is below its floor"):
            ConvictionRiskConfig(floor_pct=6.0, ceiling_pct=2.0)


class TestTheSizerHonoursTheStake:
    def test_a_bigger_stake_buys_a_bigger_position(
        self, settings: Settings, spec: InstrumentSpec
    ) -> None:
        sizer = PositionSizer(settings)
        small = sizer.size(
            spec=spec,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.0850,
            sl=1.0830,
            tp=1.0910,
            risk_pct=2.0,
        )
        large = sizer.size(
            spec=spec,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.0850,
            sl=1.0830,
            tp=1.0910,
            risk_pct=6.0,
        )

        assert small.approved and large.approved
        assert large.volume == pytest.approx(small.volume * 3, rel=0.02)

    def test_no_stake_means_the_ordinary_one(
        self, settings: Settings, spec: InstrumentSpec
    ) -> None:
        """Every caller that has no reviewer confidence to hand — and that is
        every caller but one — must keep behaving exactly as it did."""
        sizer = PositionSizer(settings)
        default = sizer.size(
            spec=spec,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.0850,
            sl=1.0830,
            tp=1.0910,
        )
        explicit = sizer.size(
            spec=spec,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.0850,
            sl=1.0830,
            tp=1.0910,
            risk_pct=settings.effective_risk_pct(),
        )

        assert default.volume == pytest.approx(explicit.volume)

    def test_the_mode_ceiling_still_clamps_an_over_large_stake(
        self, settings: Settings, spec: InstrumentSpec
    ) -> None:
        """A caller asking for more than the mode allows gets the mode's number,
        not its own. The stake is an input, never an override."""
        sizer = PositionSizer(settings)
        asked = sizer.size(
            spec=spec,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.0850,
            sl=1.0830,
            tp=1.0910,
            risk_pct=25.0,
        )
        capped = sizer.size(
            spec=spec,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.0850,
            sl=1.0830,
            tp=1.0910,
            risk_pct=settings.effective_max_risk_pct(),
        )

        assert asked.volume == pytest.approx(capped.volume)

    def test_the_anti_martingale_guard_is_untouched(
        self, settings: Settings, spec: InstrumentSpec, manager: RiskManager
    ) -> None:
        """The one rule no authorisation moves: risk may never be raised because
        of a previous loss. A conviction stake is not a recovery multiplier and
        the guard must not have been widened into accepting one.

        It crashes rather than refusing, and that is the owner's instruction
        verbatim: hardcode it as an assertion that crashes if a strategy tries
        it. A refusal could be caught and retried; this cannot.

        Note the guard is on the MULTIPLIER and not on the stake, which is why
        adding `risk_pct` did not have to touch it: conviction arrives as a
        separate argument precisely so this line keeps its meaning."""
        with pytest.raises(ValueError, match="scaling up after losses is forbidden"):
            PositionSizer(settings).size(
                spec=spec,
                equity=10_000.0,
                direction=Direction.LONG,
                entry=1.0850,
                sl=1.0830,
                tp=1.0910,
                risk_multiplier=1.5,
                risk_pct=2.0,
            )

    def test_a_conviction_stake_is_not_mistaken_for_recovery_sizing(
        self,
        tmp_path: Path,
        raw: dict[str, Any],
        spec: InstrumentSpec,
        journal: Journal,
        clock: SimulatedClock,
    ) -> None:
        """The mirror of the test above, and the reason it needed writing.

        `assert_not_forbidden` compared the intended risk against the ORDINARY
        stake. A 6% conviction trade therefore crashed the whole system with
        ForbiddenStrategyError — the feature was unusable above 2% and the
        failure looked like a martingale accusation against a trade sized
        purely on the reviewer's confidence.
        """
        settings = settings_for(
            tmp_path,
            raw,
            **{
                "system.mode": "micro_live",
                "risk.risk_per_trade_pct": 2.0,
                "risk.max_risk_per_trade_pct": 6.0,
                "modes.micro_live.max_risk_per_trade_pct": 6.0,
                "risk.conviction_risk": {
                    "enabled": True,
                    "floor_pct": 2.0,
                    "ceiling_pct": 6.0,
                },
                "risk.losing_streak_risk_multiplier": 1.0,  # as configured live
            },
        )
        manager = RiskManager(settings=settings, journal=journal, clock=clock)
        state = replace(manager.build_state(account(10_000.0)), consecutive_losses=3)
        sizing = PositionSizer(settings).size(
            spec=spec,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.0850,
            sl=1.0830,
            tp=1.0910,
            risk_pct=6.0,
        )

        assert sizing.approved
        assert sizing.intended_risk_pct == pytest.approx(6.0)
        manager.assert_not_forbidden(sizing, state)

    def test_a_losing_streak_still_shrinks_the_whole_envelope(
        self, settings: Settings, spec: InstrumentSpec, manager: RiskManager
    ) -> None:
        """The concession above is bounded. Where the losing-streak multiplier
        is actually on — it is 1.0 on this account, 0.5 in the shipped default —
        it halves the sanctioned ceiling too, and a full-conviction stake is
        refused during a streak exactly as an ordinary one would be."""
        state = replace(manager.build_state(account(10_000.0)), consecutive_losses=3)
        sizing = PositionSizer(settings).size(
            spec=spec,
            equity=10_000.0,
            direction=Direction.LONG,
            entry=1.0850,
            sl=1.0830,
            tp=1.0910,
            risk_pct=6.0,
        )

        with pytest.raises(ForbiddenStrategyError, match="martingale"):
            manager.assert_not_forbidden(sizing, state)


class TestTheAggregateCap:
    """Four slots at 6% is 24% of the account. Nothing per-trade would notice."""

    def test_an_empty_book_grants_the_full_stake(self, manager: RiskManager) -> None:
        state = manager.build_state(account(10_000.0))

        assert manager.open_risk_pct(state, lambda _: eurusd_spec_object()) == pytest.approx(0.0)
        assert manager.room_for_more_risk(
            state, 6.0, lambda _: eurusd_spec_object()
        ) == pytest.approx(6.0)

    def test_open_risk_is_measured_from_the_current_stop(self, manager: RiskManager) -> None:
        """0.10 lots of EURUSD with a 50-pip stop is EUR 50 of risk, which is
        5% of a EUR 1000 account."""
        state = manager.build_state(account(1_000.0), [position(volume=0.10)])

        assert manager.open_risk_pct(state, lambda _: eurusd_spec_object()) == pytest.approx(
            5.0, rel=0.01
        )

    def test_a_full_book_leaves_no_room(self, manager: RiskManager) -> None:
        """Two 6% positions reach the 12% ceiling with two slots still free."""
        state = manager.build_state(
            account(1_000.0),
            [
                position(volume=0.12),
                replace(position(volume=0.12), ticket=2, symbol="GBPUSD"),
            ],
        )

        assert manager.open_risk_pct(state, lambda _: eurusd_spec_object()) == pytest.approx(
            12.0, rel=0.01
        )
        assert manager.room_for_more_risk(state, 2.0, lambda _: eurusd_spec_object()) == 0.0

    def test_a_partly_loaded_book_trims_rather_than_refuses(self, manager: RiskManager) -> None:
        state = manager.build_state(account(1_000.0), [position(volume=0.16)])  # 8%

        room = manager.room_for_more_risk(state, 6.0, lambda _: eurusd_spec_object())

        assert room == pytest.approx(4.0, rel=0.01)

    def test_a_stop_at_break_even_gives_the_budget_back(self, manager: RiskManager) -> None:
        """Measured on the CURRENT stop, which is what makes the cap worth
        having rather than merely restrictive: a winner that has been made safe
        is risking nothing and should stop consuming the book's budget."""
        running = position(volume=0.10)
        banked = replace(running, sl=running.price_open)

        loaded = manager.build_state(account(1_000.0), [running])
        freed = manager.build_state(account(1_000.0), [banked])

        assert manager.open_risk_pct(loaded, lambda _: eurusd_spec_object()) > 4.0
        assert manager.open_risk_pct(freed, lambda _: eurusd_spec_object()) == pytest.approx(0.0)

    def test_a_position_with_no_stop_counts_its_whole_notional(self, manager: RiskManager) -> None:
        """Deliberately pessimistic. An unstopped position has unbounded
        downside, and the cap exists to refuse the next trade when the book is
        already loaded — not to flatter it."""
        state = manager.build_state(account(1_000.0), [position(volume=0.01, sl=None)])

        assert manager.open_risk_pct(state, lambda _: eurusd_spec_object()) > 100.0
        assert manager.room_for_more_risk(state, 2.0, lambda _: eurusd_spec_object()) == 0.0

    def test_an_unreadable_spec_never_blocks_trading(self, manager: RiskManager) -> None:
        """The cap is a limit on KNOWN exposure. A failed spec lookup is not
        evidence of a loaded book, and every other gate still stands behind it."""

        def broken(_: str) -> object:
            raise RuntimeError("no spec")

        state = manager.build_state(account(1_000.0), [position(volume=0.20)])

        assert manager.open_risk_pct(state, broken) == pytest.approx(0.0)
        assert manager.room_for_more_risk(state, 6.0, broken) == pytest.approx(6.0)

    def test_the_cap_can_be_switched_off(
        self, tmp_path: Path, raw: dict[str, Any], journal: Journal, clock: SimulatedClock
    ) -> None:
        settings = settings_for(
            tmp_path,
            raw,
            **{
                "system.mode": "micro_live",
                "risk.risk_per_trade_pct": 2.0,
                "risk.max_risk_per_trade_pct": 6.0,
                "modes.micro_live.max_risk_per_trade_pct": 6.0,
                "risk.conviction_risk": {
                    "enabled": True,
                    "floor_pct": 2.0,
                    "ceiling_pct": 6.0,
                },
                "risk.max_total_open_risk_pct": 0.0,
            },
        )
        manager = RiskManager(settings=settings, journal=journal, clock=clock)
        state = manager.build_state(account(1_000.0), [position(volume=0.50)])

        assert manager.room_for_more_risk(
            state, 6.0, lambda _: eurusd_spec_object()
        ) == pytest.approx(6.0)


def eurusd_spec_object() -> InstrumentSpec:
    return InstrumentSpec.from_mt5(eurusd_spec())


class TestTheEngineSizesTheTradeNotTheArchive:
    """With the paid reviewer switched off, whose conviction was buying the lots?

    `ai.provider: local_history` makes the adviser the nearest-neighbour
    archive, and its confidence means two different things depending on how much
    history it happens to hold:

        under min_neighbors   it passes the engine's confidence straight through
        at or over it         it returns 1 - learned_veto_rate

    The first is the engine's read of the chart in front of it. The second is a
    hit rate over comparable past setups — a real measurement, and not the one
    sizing asks for. So the stake would have stopped responding to the live
    chart the moment enough history existed, with no config change and nothing
    in the log to show the input had been swapped.

    The owner's instruction was explicit: everything comes from the engine.
    """

    @staticmethod
    def _runner(open_risk: float = 0.0):  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        from runner.service import JarvisRunner

        runner = object.__new__(JarvisRunner)
        runner.settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        runner.broker = SimpleNamespace(spec=lambda _s: None)  # type: ignore[assignment]
        runner.risk = SimpleNamespace(  # type: ignore[assignment]
            room_for_more_risk=lambda *_: 100.0,
            open_risk_pct=lambda *_: open_risk,
        )
        return runner

    @staticmethod
    def _idea(confidence: float):  # type: ignore[no-untyped-def]
        from analysis.confluence import TradeIdea
        from core.types import Direction

        return TradeIdea(
            symbol="EURUSD.i",
            approved=True,
            direction=Direction.LONG,
            score=44.0,
            confidence=confidence,
            entry=1.1000,
            stop_loss=1.0980,
            take_profit=1.1030,
            reason="test",
            signals=(),
        )

    @staticmethod
    def _advice(confidence: float):  # type: ignore[no-untyped-def]
        from advisory.providers import Advice

        return Advice(True, confidence, "test", provider="local_history", said_yes=True)

    def test_the_stake_follows_the_engine_where_the_two_disagree(self) -> None:
        """Engine sure, archive lukewarm. The chart decides the size."""
        runner = self._runner()

        stake, why = runner._conviction_stake(
            None, self._idea(confidence=0.70), self._advice(confidence=0.50)
        )

        assert stake == pytest.approx(6.0), "the archive's hit rate sized the trade"
        assert "engine conviction 0.70" in why

    def test_a_confident_archive_cannot_inflate_a_weak_setup(self) -> None:
        """The same swap in the other direction, which is the dangerous one:
        history saying 'these usually work' must not buy lots on a chart the
        engine is unsure about."""
        runner = self._runner()

        stake, why = runner._conviction_stake(
            None, self._idea(confidence=0.50), self._advice(confidence=0.95)
        )

        assert stake == pytest.approx(2.0)
        assert "engine conviction 0.50" in why

    def test_both_numbers_are_recorded_so_the_journal_shows_the_input(self) -> None:
        """A stake reason naming only the output cannot be audited later."""
        runner = self._runner()

        _, why = runner._conviction_stake(
            None, self._idea(confidence=0.60), self._advice(confidence=0.48)
        )

        assert "engine conviction 0.60" in why
        assert "adviser said 0.48" in why

    def test_the_exposure_cap_still_speaks_in_engine_terms(self) -> None:
        """The trimmed path names the engine too, or the two branches would
        report different inputs for the same decision."""
        runner = self._runner(open_risk=11.0)
        runner.risk.room_for_more_risk = lambda *_: 1.0

        stake, why = runner._conviction_stake(
            None, self._idea(confidence=0.70), self._advice(confidence=0.70)
        )

        assert stake == 0.0
        assert "engine conviction 0.70" in why


class TestAFreeAdviserIsNotRationed:
    """Setups were being refused to save money on a call that costs nothing.

    Three gates suppress a review of a setup already refused: the exact-proposal
    veto memory, the broader veto pattern, and the per-pair cooldown. All three
    justify themselves on cost, in their own words — "only ever suppresses a
    paid call", "not buying the same question again", "a replayed verdict costs
    nothing and is not rationed".

    `ai.provider: local_history` made the adviser a free nearest-neighbour
    lookup. The gates kept firing, so they saved nothing and refused trades for
    it. And the memories they read had been written by the PAID reviewer while
    it was still switched on — so a setup could be blocked by the recorded
    opinion of a model that is no longer consulted and can no longer change its
    mind. The operator named it exactly: blocked by Claude, after Claude.
    """

    @staticmethod
    def _runner(paid: bool):  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        from runner.service import JarvisRunner

        runner = object.__new__(JarvisRunner)
        runner.advisor = SimpleNamespace(uses_paid_api=paid)  # type: ignore[assignment]
        runner.settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        runner._reviews_this_cycle = 0
        return runner

    def test_a_free_adviser_reports_no_cost(self) -> None:
        assert self._runner(paid=False)._reviews_cost_money() is False

    def test_a_paid_adviser_still_reports_cost(self) -> None:
        """The rationing must come straight back if the paid reviewer returns."""
        assert self._runner(paid=True)._reviews_cost_money() is True

    def test_an_absent_marker_is_treated_as_paid(self) -> None:
        """Fails closed. An adviser that does not say is assumed to bill."""
        from types import SimpleNamespace

        runner = self._runner(paid=True)
        runner.advisor = SimpleNamespace()  # type: ignore[assignment]

        assert runner._reviews_cost_money() is True

    def test_the_review_budget_agrees_with_it(self) -> None:
        """One source for 'does this cost money', so the budget and the veto
        memories cannot disagree about whether the adviser bills."""
        assert self._runner(paid=False)._review_budget_left() is None

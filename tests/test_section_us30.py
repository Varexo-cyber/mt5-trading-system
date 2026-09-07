"""Four experimental US30 sections, and what has to hold before any replay.

WHAT THESE TESTS ARE NOT ABOUT. The detection logic is `ImpulseRetest` and
`OrderBlock`, unchanged and already covered by their own suites. These
sections CALL those detectors rather than reimplementing them, so a test that
recomputed a break here would only prove that two copies of one formula agree
-- and there is only one copy, which is the point.

What is new is the shell: a US30-only gate, one entry per setup, and a named
refusal on every path. Every test below drives the module and asserts on what
it returns.

NOTHING HERE MEASURES WHETHER THE STRATEGY PAYS. That is `us30.cmd` on the
VPS, against real Eightcap bars, and it has not been run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from analysis.section_us30 import SectionUs30
from config.schema import SectionUs30Config
from core.types import MarketContext, Series, Tick, Timeframe

BARS = 400


def _config(**over) -> SectionUs30Config:
    base = {"enabled": True, "symbol": "US30", "timeframe": "M1"}
    base.update(over)
    return SectionUs30Config(**base)


def _section(name: str = "section_us30_impulse_m1", *, broker: str = "US30.i", **over):
    return SectionUs30(name, _config(**over), broker_symbol=broker)


def _frame(clock: Timeframe, closes: np.ndarray, *, volume: np.ndarray | None = None):
    minutes = int(clock.duration.total_seconds() // 60)
    index = pd.DatetimeIndex(
        [
            datetime(2026, 4, 6, 13, 0, tzinfo=UTC) + timedelta(minutes=minutes * i)
            for i in range(len(closes))
        ]
    )
    opens = np.concatenate(([closes[0]], closes[:-1]))
    span = np.abs(closes - opens) + 3.0
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + span / 2,
            "low": np.minimum(opens, closes) - span / 2,
            "close": closes,
            "tick_volume": np.full(len(closes), 100.0) if volume is None else volume,
        },
        index=index,
    )


def _breakout(count: int = BARS) -> pd.DataFrame:
    """A flat range, one bar that CLOSES a full ATR clear of it, then a retest.

    SOLVED, NOT GUESSED. `ImpulseRetest._live_break` measures the break on the
    CLOSE against a `channel_period` high taken from the bars BEFORE it, and
    the retest against `tolerance_atr` of that same level. Three attempts at
    hand-picked numbers produced "2.04 ATR above the level it broke", so this
    builds the range, reads the level and the ATR the detector will read, and
    then places the last close where the detector needs it.
    """
    from analysis.impulse_retest import _atr

    rng = np.random.default_rng(5)
    closes = 40_000.0 + rng.normal(0, 5.0, count)
    frame = _frame(Timeframe.M1, closes)

    # The level the break bar will be judged against: the 20-bar high of the
    # bars before it, which is what `upper[i]` is.
    unit = float(_atr(frame, 14))
    level = float(frame["high"].iloc[-22:-2].max())

    closes[-2] = level + 1.6 * unit  # closes a full ATR clear -> the impulse
    closes[-1] = level + 0.05 * unit  # back to within the 0.15 ATR tolerance
    return _frame(Timeframe.M1, closes)


class _AlwaysTheSameZone:
    """A detector that fires on the same zone every bar.

    THE DEDUPE RULE IS THE WRAPPER'S, so it is tested against a stub rather
    than against whichever synthetic price path happens to make the real
    detector fire. The first version of these tests SKIPPED when the fixture
    was quiet, and a skipped test proves nothing -- least of all one of the
    four rules the brief asked for by name.
    """

    def __init__(self, level: float = 40_123.0) -> None:
        self.level = level
        self.calls = 0

    def analyze(self, ctx):  # type: ignore[no-untyped-def]
        from core.types import Signal

        self.calls += 1
        return Signal(
            module="stub",
            score=60.0,
            confidence=0.7,
            reasoning="stub setup",
            key_levels=(self.level,),
            invalidation_price=self.level - 30.0,
            details={"level": self.level},
        )


def _context(clock: Timeframe, frame, *, symbol: str = "US30.i", extra=None) -> MarketContext:
    now = frame.index[-1].to_pydatetime() + clock.duration
    price = float(frame["close"].iloc[-1])
    series = {clock: Series(symbol, clock, frame, now)}
    if extra:
        for other_clock, other in extra.items():
            series[other_clock] = Series(symbol, other_clock, other, now)
    return MarketContext(symbol, now, series, Tick(symbol, now, price - 1.0, price + 1.0))


class TestOnlyUs30AndOnlyItsOwnClock:
    def test_another_market_is_refused_by_name(self) -> None:
        section = _section()
        frame = _breakout()
        signal = section.analyze(_context(Timeframe.M1, frame, symbol="SPX500.i"))

        assert signal.score == 0.0
        assert "not this section's market" in signal.reasoning

    def test_the_suffixed_and_plain_names_are_both_this_section(self) -> None:
        """Eightcap prints `US30` on some accounts and `US30.i` on others, and
        the config says `US30`. A section that answers to only one of them is
        silent on every bar with nothing saying why -- the mismatch that has
        now disabled four separate things in this repository."""
        frame = _breakout()
        for symbol in ("US30", "US30.i"):
            signal = _section().analyze(_context(Timeframe.M1, frame, symbol=symbol))
            assert "not this section's market" not in signal.reasoning, symbol

    def test_an_m1_section_says_so_when_its_clock_is_absent(self) -> None:
        """Not a bare 'needs N closed bars', which is true and hides that the
        clock was never loaded."""
        section = _section(timeframe="M1")
        frame = _breakout()
        signal = section.analyze(_context(Timeframe.M5, frame))

        assert signal.score == 0.0
        assert "M1 bars were not loaded" in signal.reasoning

    def test_a_disabled_section_never_reads(self) -> None:
        section = _section(enabled=False)
        frame = _breakout()

        assert section.analyze(_context(Timeframe.M1, frame)).score == 0.0


class TestOneEntryPerSetup:
    """These detectors re-fire on every bar for as long as price stays in the
    zone. Correct for a detector, wrong for a section: it would take the same
    idea five times, and the duplicates are drawn from the LOSERS -- a zone
    that works is left in one bar, a zone that fails is sat on.
    """

    def _stubbed(self, level: float = 40_123.0):
        section = _section()
        section.detector = _AlwaysTheSameZone(level)
        return section

    def test_the_same_zone_is_refused_the_second_time(self) -> None:
        section = self._stubbed()
        frame = _breakout()

        first = section.analyze(_context(Timeframe.M1, frame))
        second = section.analyze(_context(Timeframe.M1, frame))

        assert first.score == 60.0
        assert second.score == 0.0
        assert "already signalled on this zone" in second.reasoning

    def test_a_new_zone_is_taken(self) -> None:
        """Without this the rule above is satisfied by a section that refuses
        everything after its first trade, which is not de-duplication."""
        section = self._stubbed()
        frame = _breakout()

        assert section.analyze(_context(Timeframe.M1, frame)).score == 60.0
        section.detector.level = 41_000.0
        assert section.analyze(_context(Timeframe.M1, frame)).score == 60.0

    def test_the_detector_is_still_consulted_on_every_bar(self) -> None:
        """The section stops the ENTRY, not the read. A section that stopped
        calling its detector would never notice the zone had changed."""
        section = self._stubbed()
        frame = _breakout()
        for _ in range(3):
            section.analyze(_context(Timeframe.M1, frame))

        assert section.detector.calls == 3

    def test_a_signal_without_a_level_is_not_deduplicated(self) -> None:
        """Refusing on an identity nothing produced would silence the section
        on a detail of its own reporting."""
        from core.types import Signal

        section = _section()
        assert section.zone_of(Signal(module="x", score=60.0, confidence=0.7)) is None

    def test_the_zone_identity_separates_direction(self) -> None:
        from core.types import Signal

        section = _section()
        up = Signal(module="x", score=60.0, confidence=0.7, key_levels=(40_000.0,))
        down = Signal(module="x", score=-60.0, confidence=0.7, key_levels=(40_000.0,))

        assert section.zone_of(up) != section.zone_of(down)

    def test_each_symbol_keeps_its_own_memory(self) -> None:
        """State keyed by section alone would let one market silence another."""
        section = self._stubbed()
        frame = _breakout()
        section._last_zone["OTHER"] = (1, 40_123.0)
        signal = section.analyze(_context(Timeframe.M1, frame))

        assert signal.score == 60.0


class TestTheParametersReachTheDetector:
    """The config is translated into the detector's own shape. A field that is
    written, documented, tested against the config and never reaches the
    detector is this repository's most repeated defect."""

    def test_the_impulse_numbers_are_the_ones_asked_for(self) -> None:
        built = _config(mechanism="impulse_retest").as_impulse_config()

        assert built.atr_period == 14
        assert built.minimum_impulse_atr == 1.0
        assert built.impulse_span_atr == 1.5
        assert built.tolerance_atr == 0.15
        assert built.stop_beyond_atr == 0.85
        assert built.lookback_bars == 96

    def test_the_order_block_numbers_are_the_ones_asked_for(self) -> None:
        built = _config(mechanism="order_block").as_order_block_config()

        assert built.atr_period == 14
        assert built.lookback_bars == 96
        assert built.minimum_impulse_atr == 1.5
        assert built.impulse_span_atr == 1.5
        assert built.block_search_bars == 5
        assert built.zone_tolerance_atr == 0.25
        assert built.stop_atr == 1.0

    def test_the_clock_reaches_the_detector(self) -> None:
        for clock in ("M1", "M5"):
            assert _config(timeframe=clock).as_impulse_config().timeframe == clock
            assert (
                _config(timeframe=clock, mechanism="order_block").as_order_block_config().timeframe
                == clock
            )

    def test_the_gold_stop_table_is_not_inherited(self) -> None:
        """The live impulse config carries a wider stop for XAUUSD, chosen for
        gold's spread. Inheriting it would give US30 a stop measured on a
        different instrument."""
        assert _config().as_impulse_config().stop_beyond_atr_by_symbol == {}
        assert _config(mechanism="order_block").as_order_block_config().stop_atr_by_symbol == {}

    def test_tuning_us30_cannot_reach_the_live_sections(self) -> None:
        from pathlib import Path

        from config.loader import load_settings

        settings = load_settings(
            overlay=Path(__file__).resolve().parents[1] / "config" / "eightcap.yaml",
            env_overrides=False,
        )
        wide = _config(retest_stop_beyond_atr=2.5).as_impulse_config()

        assert wide.stop_beyond_atr == 2.5
        assert settings.analysis.impulse_retest.stop_beyond_atr != 2.5

    def test_a_clock_nobody_asked_for_is_refused_at_load(self) -> None:
        with pytest.raises(ValueError, match="M1 or M5"):
            SectionUs30Config(timeframe="M15")


class TestTheStopIsAnAtrDistanceFromTheLevel:
    def test_a_signal_carries_a_stop_and_it_is_the_configured_multiple(self) -> None:
        section = _section()
        frame = _breakout()
        signal = section.analyze(_context(Timeframe.M1, frame))

        assert signal.score, f"the fixture produced no setup: {signal.reasoning}"
        assert signal.invalidation_price is not None
        assert signal.key_levels
        from analysis.impulse_retest import _atr

        unit = _atr(frame, 14)
        gap = abs(signal.key_levels[0] - signal.invalidation_price)
        assert gap == pytest.approx(0.85 * unit, rel=1e-6)

    def test_no_signal_can_reach_the_broker_without_a_stop(self) -> None:
        """The account's hardest rule, asserted where these sections enter."""
        for mechanism in ("impulse_retest", "order_block"):
            section = _section(mechanism=mechanism)
            signal = section.analyze(_context(Timeframe.M1, _breakout()))
            if signal.score:
                assert signal.invalidation_price is not None, mechanism


class TestTheseSectionsAreShadowAndStayThatWay:
    """`enabled` buys a replay; `live_enabled_modules` buys real money."""

    def _settings(self):  # type: ignore[no-untyped-def]
        from pathlib import Path

        from config.loader import load_settings

        return load_settings(
            overlay=Path(__file__).resolve().parents[1] / "config" / "eightcap.yaml",
            env_overrides=False,
        )

    def test_none_of_them_may_spend_real_money(self) -> None:
        settings = self._settings()
        live = set(settings.analysis.confluence.live_enabled_modules)
        for name in sorted(vars(settings.analysis)):
            if name.startswith("section_us30_"):
                assert name not in live, f"{name} reached the real-money allowlist"

    def test_they_carry_no_weight_while_they_are_off_the_allowlist(self) -> None:
        """Weight without permission misleads every research run and the
        journal, which read the raw table."""
        settings = self._settings()
        for name in sorted(vars(settings.analysis)):
            if name.startswith("section_us30_"):
                assert settings.analysis.confluence.weights[name] == 0.0, name

    def test_a_measurement_still_lifts_the_weight_on_a_copy(self) -> None:
        """Off is not deleted. A module with no weight cannot be measured, and
        that is how three M1 detectors went unjudged for months."""
        from scripts.dry_run_sections import _retimed

        settings = self._settings()
        measured = _retimed(settings, "section_us30_impulse_m1", "M1")

        assert measured.analysis.confluence.weights["section_us30_impulse_m1"] == 1.0
        assert settings.analysis.confluence.weights["section_us30_impulse_m1"] == 0.0

    def test_no_existing_live_section_was_touched(self) -> None:
        """The brief said not to change the running strategies, and the
        allowlist is where that would show first."""
        assert set(self._settings().analysis.confluence.live_enabled_modules) == {
            "failed_session_breakout",
            "section_six_gold_m5",
            "section_eight_trend_day_h1",
            "section_ten_gold_m1",
            "section_fifteen_btc_m1",
            "section_five_ndx100_m5",
        }

    def test_every_section_is_built_labelled_and_bounded(self) -> None:
        from core.trade_origin import origin_for_setup_family
        from runner.service import US30_SECTIONS, build_analysis_modules

        settings = self._settings()
        built = {module.name for module in build_analysis_modules(settings)}
        for name in US30_SECTIONS:
            assert name in built, f"{name} is configured and never built"
            assert origin_for_setup_family(name) is not None, f"{name} has no MT5 label"
            assert name in settings.risk.section_breakers, f"{name} has no breaker"

    def test_the_four_labels_are_distinct(self) -> None:
        """Two sections sharing a broker comment cannot be told apart on the
        phone, in the journal, or by the shared-symbol rule."""
        from core.trade_origin import origin_for_setup_family
        from runner.service import US30_SECTIONS

        labels = {origin_for_setup_family(name).comment for name in US30_SECTIONS}
        assert len(labels) == len(US30_SECTIONS), labels

    def test_the_replay_can_measure_them(self) -> None:
        """A section the dry run cannot see is a section this whole exercise
        cannot answer about."""
        from runner.service import US30_SECTIONS
        from scripts.dry_run_sections import _markets_each_section_needs

        markets, unbounded = _markets_each_section_needs(self._settings(), US30_SECTIONS)

        assert markets == ["US30"]
        assert not unbounded

    def test_the_launcher_uses_the_same_replay_shape_as_hoeveel(self) -> None:
        """Different gates would make the numbers incomparable with the rest
        of the book, which is the only thing they can be judged against."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        launcher = (root / "us30.cmd").read_text()

        assert "--jarvis-replay" in launcher
        assert "--section-markets" in launcher
        assert "--no-m1" not in launcher
        for name in (
            "section_us30_impulse_m1",
            "section_us30_impulse_m5",
            "section_us30_orderblock_m1",
            "section_us30_orderblock_m5",
        ):
            assert name in launcher


class TestTheJarvisGatesReachTheseSections:
    """The brief asked for the live gates, so the tests ask whether they FIRE
    on a US30 idea -- not whether the names appear in the source.

    These drive `_historical_jarvis_gate`, the same function `--jarvis-replay`
    calls for every section, with a US30-shaped setup.
    """

    def _settings(self):  # type: ignore[no-untyped-def]
        from pathlib import Path

        from config.loader import load_settings

        return load_settings(
            overlay=Path(__file__).resolve().parents[1] / "config" / "eightcap.yaml",
            env_overrides=False,
        )

    def _market(self, count: int = 400):
        """Bars an index CFD could have printed, on M1 and M5 together.

        The M1 frame is not decoration: the volume gate reads M1 and only M1,
        so a context without it would pass the gate by being unable to run it.
        """
        rng = np.random.default_rng(3)
        step = rng.normal(0, 8.0, count)
        close = 40_000.0 + np.cumsum(step)
        opens = np.concatenate(([close[0]], close[:-1]))
        spans = np.abs(step) + rng.uniform(20.0, 40.0, count)
        build = lambda minutes: pd.DataFrame(  # noqa: E731 - one shape, two clocks
            {
                "open": opens,
                "high": np.maximum(opens, close) + spans / 2,
                "low": np.minimum(opens, close) - spans / 2,
                "close": close,
                "tick_volume": np.full(count, 100.0),
            },
            index=pd.DatetimeIndex(
                [
                    datetime(2026, 4, 6, 13, 0, tzinfo=UTC) + timedelta(minutes=minutes * i)
                    for i in range(count)
                ]
            ),
        )
        return build(5), build(1)

    def _context(self, *, spread: float, last_volume: float = 100.0):
        five, one = self._market()
        one = one.copy()
        one.loc[one.index[-1], "tick_volume"] = last_volume
        now = five.index[-1].to_pydatetime() + Timeframe.M5.duration
        price = float(five["close"].iloc[-1])
        return (
            MarketContext(
                "US30",
                now,
                {
                    Timeframe.M5: Series("US30", Timeframe.M5, five, now),
                    Timeframe.M1: Series("US30", Timeframe.M1, one, now),
                },
                Tick("US30", now, price - spread / 2, price + spread / 2),
            ),
            five,
        )

    def _idea(self, five, *, width: float, ratio: float = 1.0):
        from types import SimpleNamespace

        from core.types import Direction

        entry = float(five["close"].iloc[-1])
        return SimpleNamespace(
            direction=Direction.LONG,
            entry=entry,
            stop_loss=entry - width,
            take_profit=entry + width * ratio,
            setup_family="section_us30_impulse_m5",
        )

    def _gate(self, ctx, idea):
        from types import SimpleNamespace

        from scripts.dry_run_sections import REACH_HORIZON, _historical_jarvis_gate

        spec = SimpleNamespace(
            asset_class=SimpleNamespace(value="index"),
            point=0.01,
            volume_min=0.01,
            digits=2,
        )
        return _historical_jarvis_gate(
            ctx, idea, spec, self._settings(), Timeframe.M5, horizon=REACH_HORIZON
        )

    def test_a_clean_us30_setup_passes_every_gate(self) -> None:
        """Without this the three below are satisfied by a configuration that
        refuses everything, which is not a set of gates, it is a closed market.
        """
        ctx, five = self._context(spread=1.0)
        assert self._gate(ctx, self._idea(five, width=60.0)) is None

    def test_the_spread_gate_fires_on_a_us30_idea(self) -> None:
        ctx, five = self._context(spread=12.0)
        blocked = self._gate(ctx, self._idea(five, width=60.0))

        assert blocked is not None and blocked[0] == "SPREAD_EATS_THE_STOP"

    def test_the_m1_volume_gate_fires_on_a_us30_idea(self) -> None:
        """It reads M1 and only M1, whatever clock the section runs."""
        ctx, five = self._context(spread=1.0, last_volume=900.0)
        blocked = self._gate(ctx, self._idea(five, width=60.0))

        assert blocked is not None and blocked[0] == "VOLUME_SPIKE"

    def test_a_target_this_market_does_not_reach_is_refused(self) -> None:
        ctx, five = self._context(spread=1.0)
        blocked = self._gate(ctx, self._idea(five, width=60.0, ratio=6.0))

        assert blocked is not None and blocked[0] == "TARGET_RARELY_REACHED"


class TestTheSharedPositionBookCountsTheseSections:
    """Four new sections on ONE market means four sections competing for one
    symbol slot -- with each other and with the six that spend real money.
    """

    def _trade(self, when, module, symbol="US30", direction="LONG", minutes=60):
        from scripts.dry_run_sections import Decision

        return Decision(
            when,
            symbol,
            module,
            "TRADE",
            direction=direction,
            exit_at=when + timedelta(minutes=minutes),
            pass_key=(module, "M5"),
        )

    def test_two_us30_sections_cannot_both_hold_the_same_symbol_one_way(self) -> None:
        from scripts.dry_run_sections import _under_the_slot_cap

        base = datetime(2026, 4, 6, 13, 0, tzinfo=UTC)
        offered = [
            self._trade(base, "section_us30_impulse_m1"),
            self._trade(base + timedelta(minutes=5), "section_us30_orderblock_m1"),
        ]
        allowed = _under_the_slot_cap(
            offered, slots=4, share_between_sections=False, refuse_opposite=True
        )

        assert len(allowed) == 1

    def test_the_cap_is_shared_with_the_live_sections(self) -> None:
        """A US30 section taking a slot is a slot section six does not get.
        That cost is real and it is the reason these are measured together."""
        from scripts.dry_run_sections import _under_the_slot_cap

        base = datetime(2026, 4, 6, 13, 0, tzinfo=UTC)
        offered = [
            self._trade(base, "section_six_gold_m5", symbol="XAUUSD", minutes=600),
            self._trade(base, "section_five_ndx100_m5", symbol="NDX100", minutes=600),
            self._trade(base + timedelta(minutes=1), "section_us30_impulse_m5"),
        ]
        two_slots = _under_the_slot_cap(
            offered, slots=2, share_between_sections=True, refuse_opposite=True
        )

        assert len(two_slots) == 2
        assert "section_us30_impulse_m5" not in {row.module for row in two_slots}

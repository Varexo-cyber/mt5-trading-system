"""A family that names its own exit gets that exit, not a search around it.

THIS IS THE BUG THE 30 AUGUST DRY RUN FOUND, and it is the account's recurring
defect in its purest form: a value that exists, is correct, is documented, is
tested -- and never reaches the path the code takes.

`target_r_multiple_by_family` says `impulse_retest: 1.0` and `order_block: 1.0`.
`_reachable_target` reads it into `ratio`, multiplies it into `planned` -- and
then used `planned` on ONE branch only, the fallback taken when
`_first_touch_outcomes` returns None. On the branch the code actually takes,
a search swept candidate distances from `minimum_r_multiple` up to

    ceiling = max(config.target_r_multiple, config.minimum_r_multiple)

which is the ACCOUNT's 3.0, not the family's 1.0, and picked whatever its own
model liked best. The override bounded nothing.

WHAT IT COST, straight off the dry run:

    order_block     winners averaged 1.33R against a shipped plan of 1.00
    hit rate        34% measured, against 62% in the research
    impulse_retest  winners averaged 0.98R -- correct, and the contrast is
                    the evidence: same override, different outcome, because
                    the search chose differently on different markets

The research is explicit that this is not a rounding error. The SAME entries
measured +0.279R at 1:1 and +0.016R at 3:1. Letting the search reach for payoff
does not trade the strategy worse, it trades a different strategy.

AND THE TEST THAT WAS ALREADY THERE DID NOT CATCH IT, which is the part worth
keeping in mind. `test_the_override_actually_reaches_the_target` asserts that
the STRING "target_r_multiple_by_family.items()" appears in the source. It did
appear. It still appears. The loop ran, assigned `ratio`, and its answer was
discarded four hundred lines later. A substring is not a behaviour.

So every test below calls the real method and measures the distance it comes
back with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.confluence import ConfluenceEngine
from config.loader import DEFAULT_CONFIG_PATH, load_settings
from config.schema import ConfluenceConfig, HorizonProfileConfig
from core.types import Direction, MarketContext, Series, Tick, Timeframe

ENTRY = 1.10


def _market(bars: int = 600, drift: float = 0.00004, seed: int = 11) -> pd.DataFrame:
    """A walk with a mild lean, long enough for the search to run.

    The lean matters: on a driftless walk `_first_touch_outcomes` finds nothing
    worth taking and the method returns on the None branch -- which is the ONE
    branch that always honoured the override. A test written on a flat market
    would pass against the broken code.
    """
    index = pd.date_range("2026-01-01", periods=bars, freq="15min", tz="UTC")
    rng = np.random.default_rng(seed)
    close = pd.Series(ENTRY + np.cumsum(rng.normal(drift, 0.0004, bars)), index=index)
    return pd.DataFrame(
        {"open": close, "high": close + 0.0003, "low": close - 0.0003, "close": close},
        index=index,
    )


def _context(frame: pd.DataFrame) -> MarketContext:
    when = frame.index[-1]
    series = {tf: Series("EURUSD", tf, frame, when) for tf in (Timeframe.M15, Timeframe.H1)}
    return MarketContext("EURUSD", when, series, Tick("EURUSD", when, ENTRY, ENTRY))


def _distance(engine: ConfluenceEngine, family: str, risk: float) -> float | None:
    """R-multiple of the target the engine actually returns, or None."""
    frame = _market()
    target, _note = engine._reachable_target(
        _context(frame),
        ENTRY,
        risk,
        Direction.LONG,
        profile=HorizonProfileConfig(planning_timeframe="M15", target_horizon_bars=12),
        setup_family=family,
    )
    return None if target is None else (target - ENTRY) / risk


class TestTheNamedRatioIsTheDistance:
    """One assertion, made three ways, because the failure was silent."""

    def _engine(self) -> ConfluenceEngine:
        return ConfluenceEngine(
            [],
            ConfluenceConfig(
                target_r_multiple=3.0,
                target_r_multiple_by_family={"impulse_retest": 1.0, "order_block": 1.0},
            ),
        )

    @pytest.mark.parametrize("family", ["impulse_retest", "order_block"])
    def test_a_named_family_gets_exactly_its_own_ratio(self, family: str) -> None:
        assert _distance(self._engine(), family, risk=0.0020) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("family", ["impulse_retest_m15", "order_block_m30"])
    def test_the_timeframe_suffix_still_matches(self, family: str) -> None:
        """`_classify_horizon` returns `{module}_{timeframe}` once a module is
        in `intraday_modules`, which both now are. The match is on substring
        precisely so that rename does not silently unwire the ratio."""
        assert _distance(self._engine(), family, risk=0.0020) == pytest.approx(1.0, abs=1e-6)

    def test_it_holds_at_every_stop_width(self) -> None:
        """The search's answer moves with the stop; a fixed ratio does not. If
        any of these drifts, the search is back."""
        engine = self._engine()

        for risk in (0.0005, 0.0010, 0.0020, 0.0040, 0.0080):
            assert _distance(engine, "impulse_retest", risk) == pytest.approx(1.0, abs=1e-6), risk

    def test_the_search_still_runs_for_everything_else(self) -> None:
        """This is not a change to the account's other trades. An unnamed
        family keeps the measured search, and the search does NOT return a flat
        3.0 -- if it did, this test would pass against the broken code too."""
        engine = self._engine()

        r = _distance(engine, "market_structure_swing", risk=0.0020)

        assert r is None or r != pytest.approx(1.0, abs=1e-6)

    def test_a_named_family_is_never_sent_out_past_its_ratio(self) -> None:
        """THE PROPERTY THAT ACTUALLY FAILED. Stated as an inequality so it
        also covers a future change that reintroduces a search but bounds it
        correctly."""
        engine = self._engine()

        for family in ("impulse_retest", "order_block"):
            for risk in (0.0008, 0.0020, 0.0050):
                r = _distance(engine, family, risk)
                assert r is not None and r <= 1.0 + 1e-6, (family, risk, r)


class TestTheShippedAccountAgrees:
    """The same property on the real Eightcap overlay rather than a fixture."""

    def _live(self):
        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

    def test_every_live_section_names_its_measured_target(self) -> None:
        confluence = self._live().analysis.confluence
        expected = {
            "failed_session_breakout": 1.5,
            "section_five_m5": 1.0,
            "section_six_gold_m5": 3.0,
            "section_eight_trend_day_h1": 1.0,
            "section_nine_vwap_m30": 1.5,
            "section_ten_gold_m1": 1.5,
        }

        assert set(confluence.live_enabled_modules) == set(expected)
        for module, ratio in expected.items():
            assert confluence.target_r_multiple_by_family.get(module) == pytest.approx(ratio)

    def test_the_live_config_ships_the_distance_it_names(self) -> None:
        confluence = self._live().analysis.confluence
        engine = ConfluenceEngine([], confluence)

        for module in confluence.live_enabled_modules:
            expected = confluence.target_r_multiple_by_family[module]
            assert _distance(engine, module, risk=0.0020) == pytest.approx(expected, abs=1e-6), (
                module
            )

    def test_the_account_default_is_still_three(self) -> None:
        """The override is per family. Nothing here loosens or tightens what
        the rest of the account does."""
        assert self._live().analysis.confluence.target_r_multiple == pytest.approx(3.0)


class TestBothLiveSectionsArePlannedOnAFastClock:
    """The second half of the same defect.

    Neither module was in `quick_modules` or `intraday_modules`, so
    `_classify_horizon` fell through to `return "swing"` and handed an M15
    retest H1 planning authority, a 24-bar target horizon, and the SWING higher
    timeframe veto -- D1/W1 at `minimum_htf_conflicts: 1` against intraday's
    H4/D1 at 2.

    That last number is not a detail for a break-retest. Price coming BACK to
    the level is the setup; a daily trend pointing the other way is its normal
    condition, not evidence against it. One D1 conflict refusing the trade
    refuses the strategy on the grounds of its own mechanism.
    """

    def _confluence(self):
        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence

    def test_every_live_module_is_classified_somewhere(self) -> None:
        confluence = self._confluence()
        fast = set(confluence.quick_modules) | set(confluence.intraday_modules)

        for module in confluence.live_enabled_modules:
            assert module in fast, f"{module} falls through to a swing plan"

    def test_the_classifier_actually_says_fast(self) -> None:
        """Membership is the mechanism; this is the outcome. `_classify_horizon`
        needs the WHOLE agreeing set inside the list, so a module added to the
        config but read from a stale copy would still come back swing."""
        from core.types import Signal

        confluence = self._confluence()
        engine = ConfluenceEngine([], confluence)

        for module in confluence.live_enabled_modules:
            signal = Signal(
                module=module,
                score=70.0,
                confidence=0.7,
                reasoning="",
                details={"timeframe": getattr(confluence, "timeframe", "M15")},
            )
            horizon, family = engine._classify_horizon([(signal, 1.0)])

            expected = "quick" if module in confluence.quick_modules else "intraday"
            assert horizon == expected, (module, horizon)
            assert module in family, (module, family)

    def test_the_intraday_veto_needs_two_conflicts(self) -> None:
        """What the reclassification actually buys. If this ever drops back to
        one, both sections are being refused on the D1 trend again."""
        profile = self._confluence().horizon_profiles["intraday"]

        assert profile.minimum_htf_conflicts >= 2
        assert "W1" not in profile.htf_trend_timeframes


class TestTheTimingGateDoesNotRefuseTheRetestItself:
    """FINDING SIX, and the one that actually cost every FX trade.

    Seven days on the live feed, eleven FX majors, measured by
    `scripts/signal_funnel.py`:

        impulse_retest formed 52 signals
        0.76 per pair per day, against the 0.62 the research measured
        over eleven years and 18,828 trades

    The detector is fine. The setups are there, at the rate the research says.
    The account took ZERO of them.

    `_entry_timing_conflict` refuses an entry the immediate price action is
    moving against. A break-retest buys the level BECAUSE price came back to
    it, so the adverse move IS the setup and the gate refuses every valid entry
    the strategy has.

    And it is not marginal. The gate allows 1.0 ATR of adverse travel over six
    closed bars in the ENTRY timeframe's ATR -- M5. A retest's pullback is one
    M15 ATR by construction, and an M15 ATR is roughly 1.7 M5 ATRs. Every
    qualifying retest is over the limit before it is looked at.
    """

    def _confluence(self):
        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.confluence

    def test_both_live_families_are_exempt(self) -> None:
        confluence = self._confluence()

        for module in confluence.live_enabled_modules:
            assert any(family in module for family in confluence.entry_timing_exempt_families), (
                f"{module} is still judged by the timing gate that refuses its own mechanism"
            )

    def test_the_base_config_exempts_nobody(self) -> None:
        """The gate stays fully armed for every other account and every other
        setup. This is a carve-out for two measured families, not a loosening."""
        assert (
            load_settings(env_overrides=False).analysis.confluence.entry_timing_exempt_families
            == ()
        )

    def test_the_threshold_itself_is_untouched(self) -> None:
        """Raising `entry_timing_max_adverse_atr` would have been the wrong
        repair: it loosens the gate for the trend entries it was written for,
        where eleven of the first twelve paid reviews vetoed exactly this and
        were right."""
        assert self._confluence().entry_timing_max_adverse_atr == pytest.approx(1.0)

    def test_an_exempt_family_survives_a_pullback_that_refuses_everyone_else(self) -> None:
        """THE BEHAVIOUR, not the config. Same market, same direction, same
        adverse move -- refused for an ordinary family, allowed for a retest."""
        confluence = self._confluence()
        engine = ConfluenceEngine([], confluence)
        profile = confluence.horizon_profiles["intraday"]

        # A hard pullback on the fast clocks: price falling into a long.
        #
        # The slope has to beat `entry_timing_max_adverse_atr` measured in the
        # frame's OWN ATR, and a first draft did not: a 0.0050 drift over sixty
        # bars against a 0.0010 bar range is 0.5 ATR over six bars, under the
        # 1.0 limit. The assert below exists because that fixture proved
        # nothing while looking like it proved everything.
        bars = 60
        index = pd.date_range("2026-01-01", periods=bars, freq="5min", tz="UTC")
        close = pd.Series(np.linspace(1.1200, 1.1000, bars), index=index)
        frame = pd.DataFrame(
            {"open": close, "high": close + 0.00005, "low": close - 0.00005, "close": close},
            index=index,
        )
        when = index[-1]
        series = {tf: Series("EURUSD", tf, frame, when) for tf in (Timeframe.M5, Timeframe.M1)}
        ctx = MarketContext("EURUSD", when, series, Tick("EURUSD", when, 1.1000, 1.1000))

        verdict = engine._entry_timing_conflict(
            ctx, Direction.LONG, timeframes=profile.entry_timing_timeframes
        )

        assert verdict is not None, "the fixture must be adverse or this test proves nothing"
        assert "moving against" in verdict

        # ...and the engine skips that check entirely for an exempt family.
        exempt = any(f in "impulse_retest_m15" for f in confluence.entry_timing_exempt_families)
        ordinary = any(f in "trend_momentum_swing" for f in confluence.entry_timing_exempt_families)

        assert exempt is True
        assert ordinary is False

    def test_the_exemption_is_wired_into_evaluate(self) -> None:
        """A config nothing reads is this account's signature defect, and
        `target_r_multiple_by_family` already shipped that way once -- correct,
        documented, tested by substring, and never consulted on the live path."""
        import inspect

        from analysis import confluence as module

        source = " ".join(inspect.getsource(module.ConfluenceEngine.evaluate).split())

        assert "self.config.entry_timing_exempt_families" in source
        assert "if exempt" in source

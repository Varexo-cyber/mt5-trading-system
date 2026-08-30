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

    def test_both_live_sections_name_one_to_one(self) -> None:
        confluence = self._live().analysis.confluence

        for module in confluence.live_enabled_modules:
            assert confluence.target_r_multiple_by_family.get(module) == pytest.approx(1.0), module

    def test_the_live_config_ships_the_distance_it_names(self) -> None:
        confluence = self._live().analysis.confluence
        engine = ConfluenceEngine([], confluence)

        for module in confluence.live_enabled_modules:
            assert _distance(engine, module, risk=0.0020) == pytest.approx(1.0, abs=1e-6), module

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

    def test_the_classifier_actually_says_intraday(self) -> None:
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

            assert horizon == "intraday", (module, horizon)
            assert module in family, (module, family)

    def test_the_intraday_veto_needs_two_conflicts(self) -> None:
        """What the reclassification actually buys. If this ever drops back to
        one, both sections are being refused on the D1 trend again."""
        profile = self._confluence().horizon_profiles["intraday"]

        assert profile.minimum_htf_conflicts >= 2
        assert "W1" not in profile.htf_trend_timeframes

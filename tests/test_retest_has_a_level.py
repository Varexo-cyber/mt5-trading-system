"""The "breakout retest" had no level, and bought the top instead.

`setup_lifecycle` is named for the retest and for two months did not implement
one. The tracked state held `anchor_price`, and `anchor_price` FOLLOWED THE
EXTREME:

    if sign * (executable_price - track.anchor_price) > 0.0:
        track.anchor_price = executable_price
    pullback = sign * (track.anchor_price - executable_price) / reference

So the "retest" was a pullback of 0.4 ATR from wherever the high happened to
be. On a break that ran three ATR that is an entry two and a half ATR above the
level anyone would call the retest.

    a retest   breaks 1.1000, runs to 1.1050, COMES BACK TO 1.1000,
               enters there with the stop just under the old resistance
    this       breaks 1.1000, runs to 1.1050, falls to 1.1040,
               enters there with the stop 1.5 ATR below

Two different trades, and the scorecard names which one it was:

    bought at 80-95% of its own range   14 trades   -1.65R
    bought at 95-100% (the very top)     8 trades   -0.99R
    bought at 60-80%                    21 trades   +0.66R

The only positive row is the one that is not extended. It bought extended, and
buying extended is what a retest without a level produces.

THE OWNER FOUND THIS, not the tests: "misschien weet je niet hoe je het retest
of die hele strat toepast en doe je het zo kk dom". He was right, and nothing
in this file existed to say so.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from analysis.confluence import TradeIdea
from analysis.entry_quality import EntryTimingAssessment, EntryTimingDecision
from analysis.setup_lifecycle import SetupLifecycleBook, SetupState
from core.types import Direction, Signal

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
#: One ATR, in price. Every distance below is expressed against it.
ATR = 1.0


def _book(tmp_path: Path, **changes) -> SetupLifecycleBook:  # type: ignore[no-untyped-def]
    defaults = {
        "pullback_atr": {"quick": 0.2, "intraday": 0.3, "swing": 0.4},
        "resumption_atr": {"quick": 0.05, "intraday": 0.08, "swing": 0.1},
        "expiry_minutes": {"quick": 30.0, "intraday": 240.0, "swing": 1440.0},
        "level_retest_modules": ("market_structure", "impulse_break"),
        "retest_level_atr": 0.35,
    }
    defaults.update(changes)
    return SetupLifecycleBook(tmp_path / "lifecycle.json", **defaults)  # type: ignore[arg-type]


def _idea(family: str, key_levels: tuple[float, ...] = ()) -> TradeIdea:
    return TradeIdea(
        symbol="TEST",
        approved=True,
        direction=Direction.LONG,
        score=70.0,
        confidence=0.8,
        entry=100.0,
        stop_loss=98.0,
        take_profit=104.0,
        reason="break",
        signals=(Signal("market_structure", 70.0, 0.8, reasoning="BOS", key_levels=key_levels),),
        setup_family=family,
        horizon="swing",
    )


def _late(atr: float = ATR) -> EntryTimingAssessment:
    """Timing says "too extended", which is what puts a setup into the book."""
    return EntryTimingAssessment(
        decision=EntryTimingDecision.WAIT_RETEST,
        reason_code="EXTENDED",
        detail="price is extended",
        timeframe="M5",
        reference_atr=atr,
    )


def _timely(atr: float = ATR) -> EntryTimingAssessment:
    """Timing says the price is executable. It reads distance from the
    EXTREME, so a break that ran and then eased off clears here while still
    sitting far above the level it broke."""
    return EntryTimingAssessment(
        decision=EntryTimingDecision.ENTER_NOW,
        reason_code="OK",
        detail="entry is timely",
        timeframe="M5",
        reference_atr=atr,
        last_bar_directional_atr=0.2,
    )


def _observe(  # type: ignore[no-untyped-def]
    book: SetupLifecycleBook,
    idea: TradeIdea,
    price: float,
    minutes: int,
    timing: EntryTimingAssessment | None = None,
):
    return book.observe(
        idea,
        timing or _late(),
        executable_price=price,
        now=NOW + timedelta(minutes=minutes),
    )


class TestABreakoutRetestsItsLevel:
    def test_a_shallow_dip_from_the_top_is_no_longer_a_retest(self, tmp_path: Path) -> None:
        """THE DEFECT, as the sequence that produced it. The break is seen at
        100, price runs to 103, then eases to 102.6 -- a 0.4 ATR pullback from
        the high, which the old rule accepted. It is 2.6 ATR above the level."""
        book = _book(tmp_path)
        idea = _idea("market_structure_swing")

        _observe(book, idea, 100.0, 0)  # the break: level frozen here
        _observe(book, idea, 103.0, 5)  # runs away
        verdict = _observe(book, idea, 102.6, 10)  # 0.4 ATR off the high

        assert verdict.state is SetupState.WAIT_PULLBACK
        assert "above the level it broke" in verdict.reason

    def test_coming_back_to_the_level_is(self, tmp_path: Path) -> None:
        book = _book(tmp_path)
        idea = _idea("market_structure_swing")

        _observe(book, idea, 100.0, 0)
        _observe(book, idea, 103.0, 5)
        verdict = _observe(book, idea, 100.2, 15)  # 0.2 ATR above the level

        assert verdict.state is SetupState.PULLBACK_RECEIVED
        assert "returned to within" in verdict.reason

    def test_the_level_does_not_drift_with_the_price(self, tmp_path: Path) -> None:
        """The whole repair. `anchor_price` still follows the extreme because
        the stall reads it; `level_price` must not."""
        book = _book(tmp_path)
        idea = _idea("market_structure_swing")

        _observe(book, idea, 100.0, 0)
        for minute, price in enumerate((101.0, 102.0, 103.0, 104.0), start=1):
            _observe(book, idea, price, minute)

        track = book._tracked[book.key_for(idea)]
        assert track.level_price == pytest.approx(100.0)
        assert track.anchor_price == pytest.approx(104.0)

    def test_a_short_measures_the_level_the_other_way(self, tmp_path: Path) -> None:
        """Sign errors here are invisible: a short would simply never retest,
        and the row would read as a market that did not come back."""
        book = _book(tmp_path)
        idea = _idea("market_structure_swing")
        short = replace(idea, direction=Direction.SHORT)

        _observe(book, short, 100.0, 0)
        _observe(book, short, 97.0, 5)  # runs down
        verdict = _observe(book, short, 99.8, 15)  # back up to the level

        assert verdict.state is SetupState.PULLBACK_RECEIVED


class TestTheLevelComesFromTheDetector:
    """The detection price is NOT the level.

    A setup only reaches this book once `entry_quality` has REFUSED it as
    extended -- that refusal is what "wait for a retest" means -- so the price
    recorded at detection is a break that has already run. Freezing that price
    stops the level drifting with the high and still measures the retest from
    somewhere above the level. `market_structure` publishes the swing it broke
    as the first of its `key_levels`; the range readers publish both edges.
    """

    def test_the_published_level_wins_over_the_detection_price(self, tmp_path: Path) -> None:
        book = _book(tmp_path)
        # Detected at 102 -- two ATR past the 100.0 it actually broke.
        idea = _idea("market_structure_swing", key_levels=(100.0, 97.5))

        _observe(book, idea, 102.0, 0)

        assert book._tracked[book.key_for(idea)].level_price == pytest.approx(100.0)

    def test_a_retest_of_the_detection_price_is_no_longer_enough(self, tmp_path: Path) -> None:
        """The consequence, priced: coming back to 101.8 is a 1.8 ATR-above
        entry, and 1.8 ATR above the broken level is the losing bucket."""
        book = _book(tmp_path)
        idea = _idea("market_structure_swing", key_levels=(100.0,))

        _observe(book, idea, 102.0, 0)
        verdict = _observe(book, idea, 101.8, 10)

        assert verdict.state is SetupState.WAIT_PULLBACK
        assert "1.80 ATR above the level it broke" in verdict.reason

    def test_a_level_on_the_wrong_side_is_not_a_level_this_trade_broke(
        self, tmp_path: Path
    ) -> None:
        """`key_levels` also carries the invalidation and any equal highs. A
        long that broke 100 has not broken the 104 above it, and measuring the
        retest against 104 would call every price under it a retest."""
        book = _book(tmp_path)
        idea = _idea("market_structure_swing", key_levels=(104.0, 100.0))

        _observe(book, idea, 102.0, 0)

        assert book._tracked[book.key_for(idea)].level_price == pytest.approx(100.0)

    def test_the_nearest_broken_level_is_the_one_it_retests(self, tmp_path: Path) -> None:
        """Of the levels below a long, the one it just cleared is the closest.
        An older support three ATR down is not what this break retests."""
        book = _book(tmp_path)
        idea = _idea("market_structure_swing", key_levels=(97.0, 100.0, 99.0))

        _observe(book, idea, 102.0, 0)

        assert book._tracked[book.key_for(idea)].level_price == pytest.approx(100.0)

    def test_a_short_takes_the_nearest_level_above(self, tmp_path: Path) -> None:
        book = _book(tmp_path)
        idea = replace(
            _idea("market_structure_swing", key_levels=(103.0, 100.0, 101.0)),
            direction=Direction.SHORT,
        )

        _observe(book, idea, 98.0, 0)

        assert book._tracked[book.key_for(idea)].level_price == pytest.approx(100.0)

    def test_a_detector_that_publishes_nothing_falls_back_to_the_price(
        self, tmp_path: Path
    ) -> None:
        """`m1_micro_breakout` publishes no `key_levels`. Worse than a real
        level, better than a moving one -- and it must not crash or pick a
        level belonging to some other module in the same idea."""
        book = _book(tmp_path)
        idea = _idea("market_structure_swing")

        _observe(book, idea, 102.0, 0)

        assert book._tracked[book.key_for(idea)].level_price == pytest.approx(102.0)


class TestTheRetestCanActuallyRefuseAnEntry:
    """WITHOUT THIS THE WHOLE FILE IS DECORATION.

    `observe` short-circuits to ENTER_NOW the moment `timing.passed`, before
    any state below it is consulted. So for a tracked setup the answer was
    always "enter exactly when entry_quality says the price is not extended":
    WAIT_PULLBACK could not refuse an entry timing admitted, and the retest
    decided nothing at all. Repairing where the pullback is measured from,
    without this, changes a log line and no trade.
    """

    def test_timing_clearing_far_above_the_level_does_not_enter(self, tmp_path: Path) -> None:
        book = _book(tmp_path)
        idea = _idea("market_structure_swing", key_levels=(100.0,))

        _observe(book, idea, 103.0, 0)  # refused as extended: now tracked
        verdict = _observe(book, idea, 102.5, 10, timing=_timely())

        assert verdict.state is SetupState.WAIT_PULLBACK
        assert "timing is clear but this is still 2.50 ATR above" in verdict.reason

    def test_timing_clearing_at_the_level_enters(self, tmp_path: Path) -> None:
        book = _book(tmp_path)
        idea = _idea("market_structure_swing", key_levels=(100.0,))

        _observe(book, idea, 103.0, 0)
        verdict = _observe(book, idea, 100.2, 10, timing=_timely())

        assert verdict.state is SetupState.ENTER_NOW

    def test_a_trend_module_is_not_held_back(self, tmp_path: Path) -> None:
        """The hold is for families with a level. A continuation reader has
        none, and holding it would be refusing trades for no stated reason."""
        book = _book(tmp_path)
        idea = _idea("trend_momentum_swing", key_levels=(100.0,))

        _observe(book, idea, 103.0, 0)
        verdict = _observe(book, idea, 102.5, 10, timing=_timely())

        assert verdict.state is SetupState.ENTER_NOW

    def test_a_break_that_is_timely_on_first_sight_still_enters(self, tmp_path: Path) -> None:
        """The hold only ever applies to a setup ALREADY waiting because it was
        refused as extended. A fresh break that timing likes immediately is an
        entry at the level, and it is the one bucket that made money."""
        book = _book(tmp_path)
        idea = _idea("market_structure_swing", key_levels=(100.0,))

        verdict = _observe(book, idea, 100.1, 0, timing=_timely())

        assert verdict.state is SetupState.ENTER_NOW
        assert not verdict.tracked

    def test_an_unmeasurable_distance_is_held_rather_than_waved_through(
        self, tmp_path: Path
    ) -> None:
        """No ATR, no statement about how far above the level this is. The
        house rule on this account is that missing data is a reason not to
        trade."""
        book = _book(tmp_path)
        idea = _idea("market_structure_swing", key_levels=(100.0,))

        _observe(book, idea, 103.0, 0)
        verdict = _observe(book, idea, 102.5, 10, timing=_timely(atr=0.0))

        assert verdict.state is SetupState.WAIT_PULLBACK
        assert "unmeasured" in verdict.reason


class TestEverythingElseIsUntouched:
    def test_a_trend_module_still_uses_the_pullback_from_its_extreme(self, tmp_path: Path) -> None:
        """A continuation reader has no broken level, only a move. Applying a
        level retest to it would demand a return to where the trend was first
        noticed, which is not a retest -- it is waiting for the trade to be
        over."""
        book = _book(tmp_path)
        idea = _idea("trend_momentum_swing")

        _observe(book, idea, 100.0, 0)
        _observe(book, idea, 103.0, 5)
        verdict = _observe(book, idea, 102.5, 10)  # 0.5 ATR off the high

        assert verdict.state is SetupState.PULLBACK_RECEIVED
        assert "pullback reached" in verdict.reason

    def test_the_same_prices_split_by_family(self, tmp_path: Path) -> None:
        """The two rules side by side on identical price action, because that
        is the claim: a breakout and a trend reader see the same bars and are
        supposed to want different entries."""
        book = _book(tmp_path)
        prices = ((100.0, 0), (103.0, 5), (102.5, 10))

        breakout = _idea("impulse_break_swing")
        trend = _idea("trend_momentum_swing")
        for price, minute in prices:
            broke = _observe(book, breakout, price, minute)
            trended = _observe(book, trend, price, minute)

        assert broke.state is SetupState.WAIT_PULLBACK
        assert trended.state is SetupState.PULLBACK_RECEIVED


class TestTheWiring:
    def test_the_shipped_config_names_the_breakout_families(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        entry = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.entry_quality

        assert "market_structure" in entry.lifecycle_level_retest_modules
        assert "m1_micro_breakout" in entry.lifecycle_level_retest_modules
        # And the one that must NOT be in there, or its setups stop entering.
        assert "trend_momentum" not in entry.lifecycle_level_retest_modules

    def test_the_tolerance_is_not_looser_than_the_rule_it_replaces(self) -> None:
        """0.35 ATR from the level, against the 0.40 the old rule asked from
        the extreme. This is the same distance measured from the right place,
        not a wider gate wearing a new name."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        entry = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.entry_quality

        assert entry.lifecycle_retest_level_atr <= entry.lifecycle_pullback_atr["swing"]

    def test_the_runner_passes_them_in(self) -> None:
        """A config nothing reads is the defect this whole account keeps
        producing. The book defaults `level_retest_modules` to empty, so a
        missing wire would silently leave every family on the old rule with
        every test above still green."""
        import inspect

        from runner import service

        source = " ".join(inspect.getsource(service).split())

        assert "level_retest_modules=entry.lifecycle_level_retest_modules" in source
        assert "retest_level_atr=entry.lifecycle_retest_level_atr" in source

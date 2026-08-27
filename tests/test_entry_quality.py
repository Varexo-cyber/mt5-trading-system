"""A valid direction is not permission to chase the price that confirms it."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.entry_quality import (
    EntryTimingDecision,
    assess_entry_quality,
    assess_review_drift,
)
from config.loader import load_settings
from core.instrument import AssetClass
from core.types import Direction, MarketContext, Series, Signal, Tick, Timeframe

NOW = datetime(2026, 8, 10, 13, 30, tzinfo=UTC)


def config():  # type: ignore[no-untyped-def]
    return load_settings(env_overrides=False).analysis.entry_quality


def context(closes: np.ndarray) -> MarketContext:
    index = pd.date_range("2026-08-10 06:00", periods=len(closes), freq="5min", tz=UTC)
    opens = np.r_[closes[0], closes[:-1]]
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 0.25,
            "low": np.minimum(opens, closes) - 0.25,
            "close": closes,
            "tick_volume": 100,
            "spread": 2,
            "real_volume": 0,
        },
        index=index,
    )
    return MarketContext(
        symbol="TEST",
        now=NOW,
        series={Timeframe.M5: Series("TEST", Timeframe.M5, frame, NOW)},
        tick=Tick("TEST", NOW, float(closes[-1]), float(closes[-1] + 0.02)),
    )


def quiet_history() -> np.ndarray:
    return 100.0 + np.sin(np.arange(80) / 4.0) * 0.10


def test_a_long_does_not_buy_the_end_of_a_large_m5_impulse() -> None:
    closes = quiet_history()
    closes[-4:] = [100.0, 100.9, 101.8, 102.7]

    verdict = assess_entry_quality(context(closes), Direction.LONG, AssetClass.STOCK, config())

    assert verdict.decision is EntryTimingDecision.WAIT_RETEST
    assert verdict.reason_code == "DIRECTIONAL_MOVE_OVEREXTENDED"
    assert verdict.favourable_extension_atr is not None
    assert verdict.favourable_extension_atr > 1.25


def test_a_short_does_not_sell_the_end_of_a_large_m5_impulse() -> None:
    closes = quiet_history()
    closes[-4:] = [100.0, 99.1, 98.2, 97.3]

    verdict = assess_entry_quality(context(closes), Direction.SHORT, AssetClass.STOCK, config())

    assert verdict.decision is EntryTimingDecision.WAIT_RETEST
    assert verdict.reason_code == "DIRECTIONAL_MOVE_OVEREXTENDED"


def test_an_ordinary_move_is_still_tradeable() -> None:
    closes = quiet_history()
    closes[-4:] = [100.0, 100.08, 100.16, 100.24]

    verdict = assess_entry_quality(context(closes), Direction.LONG, AssetClass.FOREX, config())

    assert verdict.decision is EntryTimingDecision.ENTER_NOW


def test_a_large_final_body_at_the_range_edge_waits_without_deleting_direction() -> None:
    closes = quiet_history()
    closes[-3:] = [100.0, 100.05, 100.85]
    live = config().model_copy(update={"directional_extreme_location": 0.80})

    verdict = assess_entry_quality(context(closes), Direction.LONG, AssetClass.FOREX, live)

    assert verdict.decision is EntryTimingDecision.WAIT_RETEST
    assert verdict.reason_code == "DIRECTIONAL_MOVE_OVEREXTENDED"
    assert "one-bar thrust at the range edge" in verdict.detail


def test_an_old_impulse_waits_for_a_fresh_resumption_bar() -> None:
    market = context(quiet_history())
    stale = Signal(
        module="impulse_break",
        score=60.0,
        confidence=0.8,
        reasoning="old impulse",
        details={"bars_since_impulse": 2},
    )

    verdict = assess_entry_quality(
        market,
        Direction.LONG,
        AssetClass.FOREX,
        config(),
        signals=(stale,),
    )

    assert verdict.decision is EntryTimingDecision.WAIT_RETEST
    assert verdict.reason_code == "STALE_IMPULSE_AWAITING_RESUMPTION"


def test_a_fresh_impulse_is_not_delayed_by_the_stale_impulse_rule() -> None:
    market = context(quiet_history())
    fresh = Signal(
        module="impulse_break",
        score=60.0,
        confidence=0.8,
        reasoning="fresh impulse",
        details={"bars_since_impulse": 1},
    )

    verdict = assess_entry_quality(
        market,
        Direction.LONG,
        AssetClass.FOREX,
        config(),
        signals=(fresh,),
    )

    assert verdict.decision is EntryTimingDecision.ENTER_NOW


def test_a_live_spike_after_the_last_closed_bar_is_not_chased() -> None:
    """The setup may be valid while its currently offered price is not.

    This is an execution wait, not a deleted setup: once the quote returns to a
    normal location the same closed-bar evidence may clear immediately.
    """
    closes = quiet_history()
    market = context(closes)

    chased = assess_entry_quality(
        market,
        Direction.LONG,
        AssetClass.CRYPTO,
        config(),
        executable_price=103.0,
    )
    retested = assess_entry_quality(
        market,
        Direction.LONG,
        AssetClass.CRYPTO,
        config(),
        executable_price=float(closes[-1] + 0.05),
    )

    assert chased.decision is EntryTimingDecision.WAIT_RETEST
    assert chased.reason_code == "DIRECTIONAL_MOVE_OVEREXTENDED"
    assert chased.executable_gap_atr is not None and chased.executable_gap_atr > 0
    assert retested.decision is EntryTimingDecision.ENTER_NOW


def test_quick_horizon_may_not_scale_away_the_live_quote_guard() -> None:
    """Quick breakouts may relax closed-bar shape, never the price offered now."""
    base = config()
    scaled = base.model_copy(
        update={
            "max_favourable_extension_atr": {
                key: value * 3.0 for key, value in base.max_favourable_extension_atr.items()
            },
            "max_single_bar_body_atr": {
                key: value * 3.0 for key, value in base.max_single_bar_body_atr.items()
            },
            "max_ema_distance_atr": {
                key: value * 3.0 for key, value in base.max_ema_distance_atr.items()
            },
        }
    )

    verdict = assess_entry_quality(
        context(quiet_history()),
        Direction.LONG,
        AssetClass.CRYPTO,
        scaled,
        executable_price=101.5,
    )

    assert verdict.decision is EntryTimingDecision.WAIT_RETEST
    assert "live quote ran" in verdict.detail


def test_a_live_short_dump_is_not_sold_at_the_bottom() -> None:
    closes = quiet_history()

    verdict = assess_entry_quality(
        context(closes),
        Direction.SHORT,
        AssetClass.CRYPTO,
        config(),
        executable_price=97.0,
    )

    assert verdict.decision is EntryTimingDecision.WAIT_RETEST


def test_a_pullback_that_is_still_falling_waits_for_the_turn() -> None:
    closes = quiet_history()
    closes[-4:] = [100.5, 100.45, 100.35, 100.0]

    verdict = assess_entry_quality(context(closes), Direction.LONG, AssetClass.FOREX, config())

    assert verdict.decision is EntryTimingDecision.WAIT_RETEST
    assert verdict.reason_code == "PULLBACK_STILL_ACTIVE"


def test_a_retest_after_an_older_impulse_can_clear_next_cycle() -> None:
    closes = quiet_history()
    closes[-8:] = [100.0, 100.8, 101.6, 102.0, 101.6, 101.3, 101.35, 101.45]

    verdict = assess_entry_quality(context(closes), Direction.LONG, AssetClass.STOCK, config())

    assert verdict.decision is EntryTimingDecision.ENTER_NOW


def test_review_price_is_bound_in_atr_not_raw_dollars() -> None:
    closes = quiet_history()
    market = context(closes)
    allowed = assess_review_drift(market, Direction.LONG, 100.0, 100.05, 20.0, config())
    stale = assess_review_drift(market, Direction.LONG, 100.0, 100.5, 20.0, config())

    assert allowed.decision is EntryTimingDecision.ENTER_NOW
    assert stale.decision is EntryTimingDecision.WAIT_RETEST


def test_a_better_fill_is_not_the_same_event_as_a_worse_one() -> None:
    """The gate took an absolute value, so it refused a discount as readily as
    a chase. For a LONG, price coming back means buying cheaper and carrying a
    shorter stop — the approval it binds is not invalidated by the fill
    improving. Whether the LEVEL is failing is a different question, owned by
    `entry_timing_max_adverse_atr` and `confirmation_max_adverse_atr`, both at
    1.00 ATR on measured evidence; this one was answering it first, at 0.25.

    Live: two of the three setups that survived every other gate in a two-hour
    window died here.
    """
    market = context(quiet_history())

    chased = assess_review_drift(market, Direction.LONG, 100.0, 100.5, 20.0, config())
    discounted = assess_review_drift(market, Direction.LONG, 100.0, 99.5, 20.0, config())

    assert chased.decision is EntryTimingDecision.WAIT_RETEST
    assert discounted.decision is EntryTimingDecision.ENTER_NOW


def test_a_short_reads_the_two_sides_the_other_way_round() -> None:
    """A sign error here is invisible in production and costs every trade, so
    the mirror case is pinned rather than assumed."""
    market = context(quiet_history())

    chased = assess_review_drift(market, Direction.SHORT, 100.0, 99.5, 20.0, config())
    discounted = assess_review_drift(market, Direction.SHORT, 100.0, 100.5, 20.0, config())

    assert chased.decision is EntryTimingDecision.WAIT_RETEST
    assert discounted.decision is EntryTimingDecision.ENTER_NOW


def test_a_collapse_during_the_review_is_still_refused() -> None:
    """Loosening the pullback side is not removing it. Past the adverse-travel
    limit the level has stopped being the level that was approved."""
    market = context(quiet_history())

    verdict = assess_review_drift(market, Direction.LONG, 100.0, 96.0, 20.0, config())

    assert verdict.decision is EntryTimingDecision.WAIT_RETEST
    assert "against the LONG" in verdict.detail


def test_slow_ai_response_never_becomes_permission_to_trade_old_analysis() -> None:
    closes = quiet_history()
    verdict = assess_review_drift(context(closes), Direction.LONG, 100.0, 100.0, 46.0, config())

    assert verdict.decision is EntryTimingDecision.WAIT_RETEST
    assert "review-age" in verdict.detail


def test_missing_entry_chart_fails_closed() -> None:
    bare = MarketContext(symbol="TEST", now=NOW, series={}, tick=None)

    verdict = assess_entry_quality(bare, Direction.LONG, AssetClass.FOREX, config())

    assert verdict.decision is EntryTimingDecision.DATA_UNAVAILABLE


class TestPlaybookVerdictTravelsWithTheRefusal:
    """One journal row must carry both readings of the same bars.

    `momentum_scalp` asks for a shallow pullback off a fresh M5 impulse.
    `entry_quality` refuses a price sitting at the extreme of its recent range.
    Whether those two descriptions land on the same bars decides whether the
    scalp theory can ever be validated honestly — a theory whose entries a
    later gate refuses cannot be measured by running it.

    Nobody could take that measurement, because whichever gate fired first
    owned the row and the other verdict was simply not written. This pins the
    carrying of the playbook verdict through to the refusal, so a day of live
    running answers it.
    """

    def test_the_candidate_carries_what_the_playbooks_saw(self) -> None:
        from runner.service import AnalysedCandidate

        assert "playbooks" in AnalysedCandidate.__dataclass_fields__

    def test_it_defaults_to_absent_rather_than_to_a_guess(self) -> None:
        """Playbooks may be switched off entirely; that is not an empty verdict."""
        from runner.service import AnalysedCandidate

        assert AnalysedCandidate.__dataclass_fields__["playbooks"].default is None

    def test_the_refusal_records_both_readings_together(self) -> None:
        """The shape the analysis depends on, asserted on the source.

        A future edit that drops `playbook_note` from the skip would silently
        restore the blind spot: every test would still pass and the number
        would quietly stop being answerable.
        """
        source = (Path(__file__).resolve().parent.parent / "runner" / "service.py").read_text(
            encoding="utf-8"
        )
        marker = source.index('"entry_quality": entry_quality.safe_dict()')
        window = source[marker : marker + 200]

        assert "playbook_note" in window, (
            "the entry-quality refusal no longer records what the playbooks saw; "
            "the scalp-versus-chase overlap becomes unmeasurable again"
        )


class TestOverExtensionIsRefusedWhereverItSitsInTheRange:
    """The four over-extension tests were each written as

        at_extreme and <limit breached>

    so the whole check only existed for a price already at 80% of its
    twelve-bar range. That left two openings, and they are the same trade seen
    from two sides: at 79% of the range no limit applied at all, and at 100% of
    the range on a calm bar every limit was clear.

    WHAT MAKES THIS A DEFECT rather than a preference. Over 1,970 closed
    trades, 197 ever peaked above +1.00R -- ten percent. A coin flip entering
    at random with the same 1R stop reaches +1R before -1R about half the time.
    Ten against fifty is not a weak edge; it is an entry taken systematically
    after the move has happened. The whole shape of the give-back table follows
    from it: 80% of trades reach 0.30R and almost none reach 1.00R.

    The limits themselves were never the problem and have not moved. 2.75 ATR
    of travel in three bars and 2.25 ATR from the EMA are generous, and a
    market past either has made its move whether or not it happens to sit at
    the top of a twelve-bar window.
    """

    def _ran_hard_then_barely_retraced(self) -> np.ndarray:
        """A six-bar climb of ten, giving back only four of it.

        The last twelve bars now form a range the price sits in the MIDDLE of
        -- 56% -- while it is still 1.56 ATR above its own EMA. Before this
        change `at_extreme` was False here, so extension, body and EMA distance
        were all skipped and the verdict was ENTER_NOW.
        """
        closes = quiet_history()
        closes[-15:-9] = np.linspace(100.0, 110.0, 6)
        closes[-9:] = 110.0 - 4.0 * np.array([0.2, 0.6, 1.0, 0.8, 0.5, 0.9, 0.6, 0.7, 0.65])
        return closes

    def test_a_market_that_ran_is_refused_even_from_the_middle_of_its_range(self) -> None:
        verdict = assess_entry_quality(
            context(self._ran_hard_then_barely_retraced()),
            Direction.LONG,
            AssetClass.STOCK,
            config(),
        )

        assert verdict.decision is EntryTimingDecision.WAIT_RETEST
        assert (
            verdict.directional_range_location < config().directional_extreme_location
        ), "the fixture has to sit BELOW the extreme or it proves nothing"
        assert verdict.ema_distance_atr > 1.5

    def test_the_same_move_retraced_properly_is_still_allowed(self) -> None:
        """The guard against over-correcting. A market that ran and then gave
        back enough to come home to its EMA is a pullback entry, which is the
        setup this system exists to take."""
        closes = quiet_history()
        closes[-15:-9] = np.linspace(100.0, 110.0, 6)
        closes[-9:] = 110.0 - 6.0 * np.array([0.2, 0.6, 1.0, 0.8, 0.5, 0.9, 0.6, 0.7, 0.65])

        verdict = assess_entry_quality(context(closes), Direction.LONG, AssetClass.STOCK, config())

        assert verdict.decision is EntryTimingDecision.ENTER_NOW

    def test_the_range_extreme_still_tightens_rather_than_enables(self) -> None:
        """`at_extreme` keeps exactly one job and it is the one it is good at.
        `max_extreme_single_bar_body_atr` is less than half
        `max_single_bar_body_atr`, because a one-bar thrust INTO the range edge
        is a different event from the same bar in the middle of a range."""
        settings = config()

        for asset in ("forex", "index", "metal", "crypto"):
            assert (
                settings.max_extreme_single_bar_body_atr[asset]
                < settings.max_single_bar_body_atr[asset]
            ), asset

    def test_a_calm_market_inside_its_range_is_still_allowed(self) -> None:
        """The guard against over-correcting. Refusing everything near a high
        would refuse every trend entry there is, which is how the same fix was
        got wrong on section six earlier the same day."""
        closes = quiet_history()

        verdict = assess_entry_quality(context(closes), Direction.LONG, AssetClass.STOCK, config())

        assert verdict.decision is EntryTimingDecision.ENTER_NOW

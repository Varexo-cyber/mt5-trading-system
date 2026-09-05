"""Section eleven: XAUJPY against XAUUSD x USDJPY.

WHAT THESE TESTS ARE FOR, and it is not the arithmetic. The gap calculation is
`analysis/legs_gap.py` and it is the same code the search measured, so a test
that recomputes it here would only prove that two copies of one formula agree.

What is genuinely new -- and what this repository has broken over and over --
is everything AROUND the read: a section that is silent when it should be, a
section that is NOT silent when it should be, and the difference between the
two being visible. Every test below is one of those.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from analysis.section_eleven_legs import (
    LEGS_META_KEY,
    MIN_LEG_BARS,
    SectionElevenLegs,
    leg_bars,
)
from config.schema import SectionElevenLegsConfig
from core.types import MarketContext, Series, Tick, Timeframe

BARS = MIN_LEG_BARS + 80
CLOCK = Timeframe.M5


def _index(count: int = BARS) -> pd.DatetimeIndex:
    start = datetime(2025, 3, 3, 8, 0, tzinfo=UTC)
    return pd.DatetimeIndex([start + timedelta(minutes=5 * i) for i in range(count)])


def _frame(values: np.ndarray, *, wobble: float = 0.4) -> pd.DataFrame:
    """OHLC around `values`, with a real range on every bar.

    A ZERO RANGE IS THE ONE THING THAT MUST NOT LEAK IN BY ACCIDENT: `alive`
    treats it as a closed market, so a fixture built from flat closes would
    silently make every test a test of the pause filter.
    """

    index = _index(len(values))
    return pd.DataFrame(
        {
            "open": values,
            "high": values + wobble,
            "low": values - wobble,
            "close": values,
            "tick_volume": np.full(len(values), 100.0),
        },
        index=index,
    )


def _legs(*, drift: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    steps = np.arange(BARS, dtype=float)
    gold = _frame(2000.0 + np.sin(steps / 7.0) * 6.0 + drift * steps)
    yen = _frame(150.0 + np.cos(steps / 11.0) * 0.5, wobble=0.02)
    return gold, yen


def _cross_from(gold: pd.DataFrame, yen: pd.DataFrame, *, offset: float = 500.0) -> pd.DataFrame:
    """A cross that tracks its legs exactly, plus a constant structural offset.

    The constant is the point: the rolling normal removes it, so a cross built
    this way reads a gap of zero rather than a gap of `offset`.
    """

    product = (gold["close"] * yen["close"]).to_numpy() + offset
    return _frame(product, wobble=20.0)


def _config(**over) -> SectionElevenLegsConfig:
    base = {"enabled": True, "gap_atr": 0.50, "minimum_gap_atr": 0.0}
    base.update(over)
    return SectionElevenLegsConfig(**base)


def _context(cross: pd.DataFrame, legs: dict | None, *, symbol: str = "XAUJPY") -> MarketContext:
    now = cross.index[-1].to_pydatetime() + CLOCK.duration
    price = float(cross["close"].iloc[-1])
    ctx = MarketContext(
        symbol,
        now,
        {CLOCK: Series(symbol, CLOCK, cross, now)},
        Tick(symbol, now, price - 0.5, price + 0.5),
    )
    if legs is not None:
        ctx.meta[LEGS_META_KEY] = legs
    return ctx


def _fresh_legs(gold: pd.DataFrame, yen: pd.DataFrame, now: datetime) -> dict:
    return {
        "XAUUSD": leg_bars("XAUUSD", gold, now),
        "USDJPY": leg_bars("USDJPY", yen, now),
    }


class TestSilenceHasToBeDistinguishable:
    """Every path that produces no trade, and each for a different reason.

    The recurring defect in this project is a section that is quiet and cannot
    say why. These do not check "no trade" as one outcome; they check that
    every separate cause reaches the same visible no-read.
    """

    def _section(self, **over) -> SectionElevenLegs:
        return SectionElevenLegs("section_eleven_xaujpy_legs_m5", _config(**over))

    def test_no_legs_in_meta_is_no_trade(self) -> None:
        gold, yen = _legs()
        cross = _cross_from(gold, yen)
        signal = self._section().analyze(_context(cross, None))
        assert signal.score == 0.0

    def test_one_leg_present_is_not_half_a_read(self) -> None:
        """Half the information is not a smaller trade, it is no trade."""
        gold, yen = _legs()
        cross = _cross_from(gold, yen)
        now = cross.index[-1].to_pydatetime()
        signal = self._section().analyze(_context(cross, {"XAUUSD": leg_bars("XAUUSD", gold, now)}))
        assert signal.score == 0.0

    def test_a_stale_leg_is_treated_as_absent(self) -> None:
        """A FROZEN LEG IS WHAT INVENTS A GAP, so old is the same as missing.

        Gold pauses daily while the yen trades on. Before that was excluded,
        two thirds of this mechanism's measured profit came out of the pause,
        and the profit was an artefact of comparing a stopped quote against a
        live one.
        """
        gold, yen = _legs()
        cross = _cross_from(gold, yen)
        now = cross.index[-1].to_pydatetime()
        stale = dict(_fresh_legs(gold, yen, now))
        old = stale["XAUUSD"]
        stale["XAUUSD"] = type(old)(old.symbol, old.frame, age_seconds=99_999.0)
        signal = self._section(max_leg_age_seconds=600.0).analyze(_context(cross, stale))
        assert signal.score == 0.0

    def test_a_short_leg_is_no_trade(self) -> None:
        gold, yen = _legs()
        cross = _cross_from(gold, yen)
        now = cross.index[-1].to_pydatetime()
        short = _fresh_legs(gold, yen.iloc[-10:], now)
        assert self._section().analyze(_context(cross, short)).score == 0.0

    def test_a_disabled_section_never_reads(self) -> None:
        gold, yen = _legs()
        cross = _cross_from(gold, yen)
        now = cross.index[-1].to_pydatetime()
        section = SectionElevenLegs("section_eleven_xaujpy_legs_m5", _config(enabled=False))
        assert section.analyze(_context(cross, _fresh_legs(gold, yen, now))).score == 0.0

    def test_another_market_is_not_this_section_s_business(self) -> None:
        gold, yen = _legs()
        cross = _cross_from(gold, yen)
        now = cross.index[-1].to_pydatetime()
        ctx = _context(cross, _fresh_legs(gold, yen, now), symbol="EURUSD")
        assert self._section().analyze(ctx).score == 0.0

    def test_a_blocked_hour_is_refused(self) -> None:
        gold, yen = _legs()
        cross = _cross_from(gold, yen)
        now = cross.index[-1].to_pydatetime()
        hour = int(cross.index[-1].hour)
        section = self._section(blocked_hours=(hour,))
        assert section.analyze(_context(cross, _fresh_legs(gold, yen, now))).score == 0.0


class TestTheBrokerSynthesisesTheCross:
    """The question that has to be answered before any backtest of this.

    Many brokers COMPUTE a cross from its legs. If Eightcap does, the gap is
    zero by construction, there is no lag, no counterparty and nothing to
    trade, and every reading is the rounding error of a multiplication. A
    section that keeps trading through that is trading noise.
    """

    def test_a_perfectly_synthetic_cross_produces_no_signal(self) -> None:
        gold, yen = _legs()
        cross = _cross_from(gold, yen)
        now = cross.index[-1].to_pydatetime()
        section = SectionElevenLegs(
            "section_eleven_xaujpy_legs_m5", _config(minimum_gap_atr=0.005, gap_atr=0.05)
        )
        signal = section.analyze(_context(cross, _fresh_legs(gold, yen, now)))
        assert signal.score == 0.0

    def test_the_floor_is_what_refuses_it_and_not_the_threshold(self) -> None:
        """Drop the floor to zero and the same bars are readable again.

        Without this the test above would pass for the wrong reason -- a
        threshold nothing ever reaches looks identical to a floor doing its
        job, and this repository has shipped that confusion repeatedly.
        """
        gold, yen = _legs()
        cross = _cross_from(gold, yen)
        section = SectionElevenLegs("section_eleven_xaujpy_legs_m5", _config(minimum_gap_atr=0.0))
        _reading, unit, median = section.reading(cross, gold, yen)
        assert unit is not None and unit > 0.0
        assert median is not None
        assert median < 0.005, f"a synthetic cross should read a near-zero gap, got {median}"


class TestALaggedQuoteIsFadedTowardItsLegs:
    """The trade itself, in both directions, on a cross that is genuinely late."""

    def _lagged(self, *, shift: int, push: float) -> tuple:
        gold, yen = _legs()
        product = (gold["close"] * yen["close"]).to_numpy()
        # The cross tracks its legs, then the last bar is pushed away from them
        # -- which is exactly what a quote that has not caught up looks like.
        late = product.copy()
        late[-shift:] = late[-shift:] + push
        cross = _frame(late, wobble=20.0)
        now = cross.index[-1].to_pydatetime()
        return cross, _fresh_legs(gold, yen, now)

    def test_a_rich_cross_is_sold(self) -> None:
        cross, legs = self._lagged(shift=1, push=400.0)
        section = SectionElevenLegs("section_eleven_xaujpy_legs_m5", _config())
        signal = section.analyze(_context(cross, legs))
        assert signal.score < 0.0
        assert signal.invalidation_price is not None
        assert signal.invalidation_price > float(cross["close"].iloc[-1])

    def test_a_cheap_cross_is_bought(self) -> None:
        cross, legs = self._lagged(shift=1, push=-400.0)
        section = SectionElevenLegs("section_eleven_xaujpy_legs_m5", _config())
        signal = section.analyze(_context(cross, legs))
        assert signal.score > 0.0
        assert signal.invalidation_price is not None
        assert signal.invalidation_price < float(cross["close"].iloc[-1])

    def test_the_signal_clears_the_confluence_threshold_on_its_own(self) -> None:
        """THE BUG THAT COST AN HOUR AND 32,407 SETUPS.

        A lone module scores exactly `|score| x confidence`. The predecessor
        shipped a hardcoded 60 against a confidence of 0.55, which is 33.0
        against a `score_threshold` of 35.0 -- two points short, on every bar,
        forever. It formed 32,407 setups over 180 days and took ZERO trades.

        This asserts the arithmetic rather than the numbers, so a later change
        to either field is caught by the property it has to keep.
        """
        from pathlib import Path

        from config.loader import load_settings

        settings = load_settings(
            overlay=Path(__file__).resolve().parents[1] / "config" / "eightcap.yaml",
            env_overrides=False,
        )
        config = settings.analysis.section_eleven_xaujpy_legs_m5
        threshold = settings.analysis.confluence.score_threshold
        assert config.score * config.confidence > threshold, (
            f"section eleven sends {config.score * config.confidence:.1f} into a "
            f"threshold of {threshold:.1f} and can never produce a trade"
        )

    def test_the_stop_is_the_configured_atr_multiple(self) -> None:
        cross, legs = self._lagged(shift=1, push=400.0)
        section = SectionElevenLegs("section_eleven_xaujpy_legs_m5", _config(stop_atr=2.0))
        wide = section.analyze(_context(cross, legs))
        narrow = SectionElevenLegs("section_eleven_xaujpy_legs_m5", _config(stop_atr=1.0)).analyze(
            _context(cross, legs)
        )
        close = float(cross["close"].iloc[-1])
        assert wide.invalidation_price is not None and narrow.invalidation_price is not None
        assert abs(wide.invalidation_price - close) == pytest.approx(
            2.0 * abs(narrow.invalidation_price - close)
        )

    def test_a_trade_without_a_stop_is_impossible_here(self) -> None:
        """The account's hardest rule, asserted at the one place it enters.

        Every signal this section can emit carries an `invalidation_price`, so
        a position from it can never reach the broker without a stop.
        """
        for push in (400.0, -400.0):
            cross, legs = self._lagged(shift=1, push=push)
            signal = SectionElevenLegs("section_eleven_xaujpy_legs_m5", _config()).analyze(
                _context(cross, legs)
            )
            assert signal.score != 0.0
            assert signal.invalidation_price is not None


class TestTheReadingIsNotAllowedToLookAhead:
    def test_a_leg_that_stopped_printing_cannot_produce_a_current_reading(self) -> None:
        """An inner join against a short leg silently hands back an OLD bar.

        The frame would then be labelled with the leg's last timestamp while
        the cross has moved on, and every bar of that difference is a gap the
        mechanism invented rather than found. That is a look-ahead of exactly
        the size of the thing being traded.
        """
        gold, yen = _legs()
        cross = _cross_from(gold, yen)
        section = SectionElevenLegs("section_eleven_xaujpy_legs_m5", _config())
        reading, unit, _median = section.reading(cross, gold.iloc[:-6], yen.iloc[:-6])
        assert reading is None and unit is None


class TestTheConfigRefusesWhatCannotWork:
    def test_a_leg_equal_to_the_cross_is_refused_at_load(self) -> None:
        with pytest.raises(ValueError, match="three different instruments"):
            SectionElevenLegsConfig(symbol="XAUJPY", base_leg="XAUJPY", quote_leg="USDJPY")

    def test_an_allow_list_inside_the_block_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="can never enter"):
            SectionElevenLegsConfig(allowed_hours=(9, 10), blocked_hours=(9, 10, 11))

    def test_an_hour_that_is_not_an_hour_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a UTC hour"):
            SectionElevenLegsConfig(blocked_hours=(25,))


class TestTheRunnerAndTheReplayAgreeOnTheKey:
    """One string, imported by three files. A string typed twice diverges."""

    def test_the_runner_writes_the_key_the_section_reads(self) -> None:
        from runner import service

        assert service.LEGS_META_KEY == LEGS_META_KEY

    def test_the_replay_writes_the_key_the_section_reads(self) -> None:
        from scripts import dry_run_sections

        assert dry_run_sections.LEGS_META_KEY == LEGS_META_KEY

    def test_the_section_is_in_the_live_module_registry(self) -> None:
        """A module absent from the registry has no weight row and no breaker.

        Which is the same silence as everything above, one level up.
        """
        from pathlib import Path

        from config.loader import load_settings
        from runner.service import build_analysis_modules

        settings = load_settings(
            overlay=Path(__file__).resolve().parents[1] / "config" / "eightcap.yaml",
            env_overrides=False,
        )
        names = {module.name for module in build_analysis_modules(settings)}
        assert "section_eleven_xaujpy_legs_m5" in names

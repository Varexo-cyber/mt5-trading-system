"""Two positions that share a currency leg are one position, sized twice.

A live account held GBPAUD short and GBPJPY short at the same time, both
losing. That reads as two bad trades and was one bad trade with a second lot
on it: a short on any GBP cross is a short on GBP, and when GBP rallied they
moved together because they were never independent.

The correlation filter sits next door and did not catch it, by design. It
measures how returns have moved over 200 hourly bars, and the AUD and JPY legs
pull those two apart enough to land under the 0.7 threshold. The GBP leg they
share is an identity, not a correlation, and no measurement weakens it.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from config.schema import CurrencyExposureConfig
from core.instrument import AssetClass, InstrumentSpec
from core.types import Direction, Position, Tick
from filters.base import FilterContext
from filters.currency_exposure import CurrencyExposureFilter, legs
from risk.reasons import Reason
from tests.fakes.fake_mt5 import eurusd_spec

NOW = datetime(2026, 8, 6, 12, 30, tzinfo=UTC)


def spec_for(symbol: str, base: str, quote: str) -> InstrumentSpec:
    return replace(
        InstrumentSpec.from_mt5(eurusd_spec()),
        symbol=symbol,
        currency_base=base,
        currency_profit=quote,
    )


SPECS = {
    "GBPAUD.i": spec_for("GBPAUD.i", "GBP", "AUD"),
    "GBPJPY.i": spec_for("GBPJPY.i", "GBP", "JPY"),
    "EURUSD.i": spec_for("EURUSD.i", "EUR", "USD"),
    "AUDUSD.i": spec_for("AUDUSD.i", "AUD", "USD"),
    "XAUUSD": replace(
        spec_for("XAUUSD", "XAU", "USD"), asset_class=AssetClass.METAL, is_forex=False
    ),
}


def held(symbol: str, direction: Direction, ticket: int = 1) -> Position:
    return Position(
        ticket=ticket,
        symbol=symbol,
        direction=direction,
        volume=0.02,
        price_open=1.9,
        sl=1.91,
        tp=1.89,
        profit=-0.95,
        swap=0.0,
        opened_at=NOW,
    )


def context(symbol: str, direction: Direction, *open_positions: Position) -> FilterContext:
    return FilterContext(
        symbol=symbol,
        spec=SPECS[symbol],
        now=NOW,
        direction=direction,
        tick=Tick(symbol=symbol, time=NOW, bid=1.9, ask=1.9001),
        open_positions=open_positions,
    )


def gate(limit: int = 1, enabled: bool = True) -> CurrencyExposureFilter:
    return CurrencyExposureFilter(
        CurrencyExposureConfig(enabled=enabled, max_positions_per_currency=limit),
        lambda symbol: SPECS[symbol],
    )


class TestLegs:
    def test_a_long_is_long_the_base_and_short_the_quote(self) -> None:
        assert legs(SPECS["GBPAUD.i"], Direction.LONG) == {"GBP": 1, "AUD": -1}

    def test_a_short_is_the_reverse(self) -> None:
        assert legs(SPECS["GBPAUD.i"], Direction.SHORT) == {"GBP": -1, "AUD": 1}

    def test_a_metal_contributes_only_its_currency_leg(self) -> None:
        """Two gold positions are not a currency concentration."""
        assert legs(SPECS["XAUUSD"], Direction.LONG) == {"USD": -1}


class TestTheLiveCase:
    def test_a_second_gbp_short_is_refused(self) -> None:
        """GBPAUD short is on. GBPJPY short would be the same bet again."""
        verdict = gate().check(
            context("GBPJPY.i", Direction.SHORT, held("GBPAUD.i", Direction.SHORT))
        )
        assert not verdict.passed
        assert verdict.reason is Reason.CURRENCY_CONCENTRATION
        assert verdict.data["currency"] == "GBP"

    def test_the_message_names_the_currency_and_the_count(self) -> None:
        verdict = gate().check(
            context("GBPJPY.i", Direction.SHORT, held("GBPAUD.i", Direction.SHORT))
        )
        assert "already short GBP on 1 open position" in verdict.detail

    def test_the_opposite_side_reduces_the_bet_and_is_allowed(self) -> None:
        """Long GBPJPY against a short GBPAUD is a hedge, not a stack."""
        assert (
            gate()
            .check(context("GBPJPY.i", Direction.LONG, held("GBPAUD.i", Direction.SHORT)))
            .passed
        )

    def test_an_unrelated_pair_passes(self) -> None:
        assert (
            gate()
            .check(context("EURUSD.i", Direction.LONG, held("GBPAUD.i", Direction.SHORT)))
            .passed
        )

    def test_the_quote_leg_counts_too(self) -> None:
        """Short GBPAUD is long AUD. A long AUDUSD would stack on that."""
        verdict = gate().check(
            context("AUDUSD.i", Direction.LONG, held("GBPAUD.i", Direction.SHORT))
        )
        assert not verdict.passed
        assert verdict.data["currency"] == "AUD"


class TestBoundaries:
    def test_an_empty_book_never_blocks(self) -> None:
        assert gate().check(context("GBPJPY.i", Direction.SHORT)).passed

    def test_a_higher_limit_permits_the_stack(self) -> None:
        assert (
            gate(limit=2)
            .check(context("GBPJPY.i", Direction.SHORT, held("GBPAUD.i", Direction.SHORT)))
            .passed
        )

    def test_disabled_lets_everything_through(self) -> None:
        assert (
            gate(enabled=False)
            .check(context("GBPJPY.i", Direction.SHORT, held("GBPAUD.i", Direction.SHORT)))
            .passed
        )

    def test_a_missing_direction_is_refused_loudly(self) -> None:
        """Which currency a cross is long is undefined without it."""
        # Needs an open book, or the empty-book shortcut answers first.
        blank = replace(
            context("GBPJPY.i", Direction.SHORT, held("GBPAUD.i", Direction.SHORT)),
            direction=None,
        )
        with pytest.raises(ValueError, match="needs the intended direction"):
            gate().check(blank)

    def test_an_unreadable_spec_does_not_end_the_cycle(self) -> None:
        """One symbol the broker will not describe must not stop the account."""

        def broken(symbol: str) -> InstrumentSpec:
            if symbol == "GBPAUD.i":
                raise RuntimeError("no spec")
            return SPECS[symbol]

        filter_ = CurrencyExposureFilter(CurrencyExposureConfig(), broken)
        assert filter_.check(
            context("GBPJPY.i", Direction.SHORT, held("GBPAUD.i", Direction.SHORT))
        ).passed


def test_standing_exposure_is_netted_across_the_book() -> None:
    """Two shorts on GBP crosses is -2 GBP, not two unrelated -1s."""
    filter_ = gate()
    totals = filter_.standing(
        context(
            "EURUSD.i",
            Direction.LONG,
            held("GBPAUD.i", Direction.SHORT, ticket=1),
            held("GBPJPY.i", Direction.SHORT, ticket=2),
        )
    )
    assert totals["GBP"] == -2
    assert totals["AUD"] == 1
    assert totals["JPY"] == 1

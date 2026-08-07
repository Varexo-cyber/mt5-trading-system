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


class TestAnIndexIsNotACurrencyBet:
    """An instrument quoted in its own base currency has no currency leg.

    `legs` wrote the base and then the quote into the same dict, so when they
    were the same code the second overwrote the first and a *long* FRA40 came
    out as a *short* on EUR. Wrong twice over: it hid the concentration when
    two European index longs were held together, and it invented one against
    any unrelated EURUSD short.

    Seen live on 7 August — FRA40 long, UK100 long and FRA40 long again inside
    twenty minutes, with this filter recording nothing worth objecting to.
    """

    FRA40 = replace(spec_for("FRA40", "EUR", "EUR"), asset_class=AssetClass.INDEX, is_forex=False)

    def test_a_long_index_is_not_a_short_on_its_own_currency(self) -> None:
        assert legs(self.FRA40, Direction.LONG) == {}

    def test_nor_is_a_short_one_a_long(self) -> None:
        assert legs(self.FRA40, Direction.SHORT) == {}

    def test_it_leaves_an_unrelated_fx_trade_alone(self) -> None:
        """The false positive. A EUR-denominated index long recorded as a EUR
        short would have refused a genuine EURUSD short standing next to it."""
        assert "EUR" not in legs(self.FRA40, Direction.LONG)

    def test_a_real_cross_still_decomposes(self) -> None:
        """The fix must not reach past the case it is for."""
        assert legs(spec_for("GBPAUD", "GBP", "AUD"), Direction.SHORT) == {"GBP": -1, "AUD": 1}

    def test_two_european_index_longs_are_still_invisible_here(self) -> None:
        """Stated so nobody reads the fix as more than it is.

        FRA40 and UK100 held long together is one bet on European equities
        sized twice, and this filter now correctly says nothing about it —
        because it is not a currency concentration. The only thing standing
        between the account and that trade is the correlation filter next
        door, and that is a measurement with a threshold rather than an
        identity. A sector limit is the missing piece, not this.
        """
        uk100 = replace(
            spec_for("UK100", "GBP", "GBP"), asset_class=AssetClass.INDEX, is_forex=False
        )
        assert legs(self.FRA40, Direction.LONG) == legs(uk100, Direction.LONG) == {}


class TestSectorConcentration:
    """The hole the index fix opened up, closed.

    Two European index longs are one bet on equities with a second lot on it,
    and the currency accounting correctly says nothing about them — they have
    no currency legs to stack. Something else has to.

    On 7 August the account held FRA40 long, then UK100 long beside it, then
    FRA40 long again, inside twenty minutes. The only thing between it and
    that book was the correlation filter, which is a measurement with a
    threshold rather than an identity.
    """

    FRA40 = replace(spec_for("FRA40", "EUR", "EUR"), asset_class=AssetClass.INDEX, is_forex=False)
    UK100 = replace(spec_for("UK100", "GBP", "GBP"), asset_class=AssetClass.INDEX, is_forex=False)

    def filter_for(self, **overrides) -> CurrencyExposureFilter:  # type: ignore[no-untyped-def]
        specs = {"FRA40": self.FRA40, "UK100": self.UK100, **SPECS}
        return CurrencyExposureFilter(
            CurrencyExposureConfig(**overrides), lambda symbol: specs[symbol]
        )

    def context(self, symbol: str, direction: Direction, open_symbols: list[tuple[str, Direction]]):  # type: ignore[no-untyped-def]
        specs = {"FRA40": self.FRA40, "UK100": self.UK100, **SPECS}
        return FilterContext(
            symbol=symbol,
            spec=specs[symbol],
            now=datetime(2026, 8, 7, 10, 16, tzinfo=UTC),
            direction=direction,
            open_positions=tuple(held(name, side) for name, side in open_symbols),
        )

    def test_a_second_index_long_is_refused(self) -> None:
        verdict = self.filter_for().check(
            self.context("UK100", Direction.LONG, [("FRA40", Direction.LONG)])
        )

        assert not verdict.passed
        assert verdict.reason is Reason.SECTOR_CONCENTRATION

    def test_the_other_side_is_still_allowed(self) -> None:
        """Shorting equities while long equities reduces the bet. That is what
        a book should be permitted to do."""
        verdict = self.filter_for().check(
            self.context("UK100", Direction.SHORT, [("FRA40", Direction.LONG)])
        )

        assert verdict.passed

    def test_an_index_beside_an_fx_position_is_fine(self) -> None:
        """Different bets. The whole point of the class grouping is that it
        does not reach past its own class."""
        verdict = self.filter_for().check(
            self.context("FRA40", Direction.LONG, [("EURUSD.i", Direction.SHORT)])
        )

        assert verdict.passed

    def test_forex_is_not_capped_by_class(self) -> None:
        """EURUSD beside USDJPY are genuinely different trades, and the
        currency legs already describe them exactly."""
        verdict = self.filter_for().check(
            self.context("AUDUSD.i", Direction.LONG, [("EURUSD.i", Direction.LONG)])
        )

        assert verdict.passed or verdict.reason is Reason.CURRENCY_CONCENTRATION

    def test_the_limit_can_be_raised(self) -> None:
        verdict = self.filter_for(max_positions_per_asset_class=2).check(
            self.context("UK100", Direction.LONG, [("FRA40", Direction.LONG)])
        )

        assert verdict.passed

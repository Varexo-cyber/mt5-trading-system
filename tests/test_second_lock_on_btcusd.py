"""The second lock on BTCUSD, and why one gate was not enough.

THE NUMBER THIS FILE EXISTS FOR. Over 180 days of Eightcap bars section
fifteen returned +EUR 15.62 on BTCUSD while
`analysis.confluence.max_spread_share_of_stop` refused 7,414 setups on that
same market worth **-2,241.89 R**. Read that pair again: the profitable
section is a trickle getting past a dam.

One number holding back a flood is not a design. Loosen it with a single YAML
edit, or find a path that never reaches it, and BTCUSD goes from fifteen euros
to catastrophic with nothing in between and nothing saying so.

So there are three layers now, and these tests hold each one on its own:

    1. `analysis.confluence.max_spread_share_of_stop`   the original gate
    2. `risk.hard_spread_ceiling_by_symbol`             the sizer's own ceiling
    3. a load-time refusal to run BTCUSD live without layer 2

Every test below SABOTAGES the layers above the one it is testing. A guard
that has only ever been exercised with its neighbours intact has not been
shown to be independent, and independence is the entire claim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.loader import load_settings
from config.schema import SYMBOLS_THAT_REQUIRE_A_SPREAD_CEILING
from core.instrument import InstrumentSpec
from core.types import Direction
from risk.position_sizer import PositionSizer
from tests.fakes.fake_mt5 import eurusd_spec

OVERLAY = Path(__file__).resolve().parents[1] / "config" / "eightcap.yaml"


def live_settings():  # type: ignore[no-untyped-def]
    return load_settings(overlay=OVERLAY, env_overrides=False)


def crypto_spec(name: str = "BTCUSD.i") -> InstrumentSpec:
    """A BTCUSD contract under the broker's suffixed name.

    SUFFIXED ON PURPOSE. The config says `BTCUSD` and every live tick says
    `BTCUSD.i`; a raw dict lookup would miss on every one of them and this
    whole file would pass while guarding nothing. That exact mismatch has
    silently disabled three separate things in this repository.
    """

    return InstrumentSpec.from_mt5(
        eurusd_spec(
            name=name,
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=0.01,
            trade_contract_size=1.0,
            path=f"Crypto\\{name}",
            currency_base="BTC",
            currency_profit="USD",
        )
    )


def outcome(settings, spec, *, spread: float, entry: float = 100_000.0, stop: float = 1_000.0):  # type: ignore[no-untyped-def]
    return (
        PositionSizer(settings)
        .size(
            spec=spec,
            equity=252.18,
            direction=Direction.LONG,
            entry=entry,
            sl=entry - stop,
            tp=entry + 3 * stop,
            spread_price=spread,
        )
        .decision
    )


def without_the_first_gate(settings):  # type: ignore[no-untyped-def]
    """The account with `max_spread_share_of_stop` opened all the way.

    This is the failure the second lock exists for, expressed as a fixture
    rather than as a worry in a comment.
    """

    confluence = settings.analysis.confluence.model_copy(update={"max_spread_share_of_stop": 1.0})
    return settings.model_copy(
        update={"analysis": settings.analysis.model_copy(update={"confluence": confluence})}
    )


class TestTheSecondLockHoldsOnItsOwn:
    def test_it_refuses_a_spread_above_the_symbol_ceiling(self) -> None:
        settings = live_settings()
        ceiling = settings.risk.hard_spread_ceiling_by_symbol["BTCUSD"]
        decision = outcome(settings, crypto_spec(), spread=1_000.0 * (ceiling + 0.02))

        assert not decision.approved
        assert decision.reason.name == "SPREAD_ABOVE_HARD_CEILING"

    def test_it_still_refuses_with_the_first_gate_wide_open(self) -> None:
        """The whole point. Layer one is the one that might be edited."""
        settings = without_the_first_gate(live_settings())
        decision = outcome(settings, crypto_spec(), spread=400.0)

        assert not decision.approved
        assert decision.reason.name == "SPREAD_ABOVE_HARD_CEILING"

    def test_it_still_refuses_with_the_cost_gate_switched_off_as_well(self) -> None:
        """`max_cost_share_of_risk` reads `if limit > 0`, so a zero silently
        removes it. Both other layers off, and BTCUSD is still refused."""
        settings = without_the_first_gate(live_settings())
        settings = settings.model_copy(
            update={"risk": settings.risk.model_copy(update={"max_cost_share_of_risk": 0.0})}
        )
        decision = outcome(settings, crypto_spec(), spread=400.0)

        assert not decision.approved
        assert decision.reason.name == "SPREAD_ABOVE_HARD_CEILING"

    def test_a_zero_ceiling_means_never_and_not_disabled(self) -> None:
        """The footgun in `max_cost_share_of_risk`, deliberately not repeated.

        There a zero switches the check off. Here it must mean the strictest
        possible ceiling, because a guard that disarms itself when someone
        types the most obvious "off" value is worse than no guard.
        """
        settings = live_settings()
        settings = settings.model_copy(
            update={
                "risk": settings.risk.model_copy(
                    update={"hard_spread_ceiling_by_symbol": {"BTCUSD": 0.0}}
                )
            }
        )
        decision = outcome(settings, crypto_spec(), spread=0.01)

        assert not decision.approved
        assert decision.reason.name == "SPREAD_ABOVE_HARD_CEILING"

    def test_an_unreadable_spread_on_a_capped_symbol_is_refused(self) -> None:
        """NO DATA IS NO TRADE, the same rule the calendar runs on. A capped
        symbol admitted because its cost could not be read is the cap failing
        OPEN, which is the one way it must never fail."""
        settings = live_settings()
        assert settings.risk.refuse_capped_symbols_without_a_spread
        decision = outcome(settings, crypto_spec(), spread=0.0)

        assert not decision.approved
        assert decision.reason.name == "SPREAD_ABOVE_HARD_CEILING"

    def test_a_tolerable_spread_is_not_refused_by_this_lock(self) -> None:
        """Without this the four tests above pass on a lock that refuses
        everything, which is not a lock, it is a closed market."""
        settings = live_settings()
        decision = outcome(settings, crypto_spec(), spread=50.0)

        assert decision.reason.name != "SPREAD_ABOVE_HARD_CEILING"

    def test_the_suffixed_and_plain_names_get_the_same_answer(self) -> None:
        settings = live_settings()
        suffixed = outcome(settings, crypto_spec("BTCUSD.i"), spread=400.0)
        plain = outcome(settings, crypto_spec("BTCUSD"), spread=400.0)

        assert suffixed.reason.name == plain.reason.name == "SPREAD_ABOVE_HARD_CEILING"

    def test_a_market_without_a_ceiling_is_untouched(self) -> None:
        """The lock is per symbol. Tightening BTCUSD must not tighten gold."""
        settings = live_settings()
        gold = InstrumentSpec.from_mt5(
            eurusd_spec(
                name="XAUUSD",
                digits=2,
                point=0.01,
                trade_tick_size=0.01,
                trade_tick_value=0.01,
                trade_contract_size=100.0,
                path="Metals\\XAUUSD",
                currency_base="XAU",
                currency_profit="USD",
            )
        )
        decision = outcome(settings, gold, spread=0.30, entry=2000.0, stop=3.0)

        assert decision.reason.name != "SPREAD_ABOVE_HARD_CEILING"


class TestTheLockCannotBeRemovedWhileTheMarketIsLive:
    def test_the_shipped_account_arms_it(self) -> None:
        settings = live_settings()
        for symbol in SYMBOLS_THAT_REQUIRE_A_SPREAD_CEILING:
            live = set(settings.analysis.confluence.live_enabled_modules)
            trades_it = any(
                symbol
                in {
                    settings.instruments.canonical_symbol(market).upper()
                    for market in (
                        tuple(
                            getattr(getattr(settings.analysis, name), "allowed_symbols", ()) or ()
                        )
                        or (
                            (getattr(getattr(settings.analysis, name), "symbol", ""),)
                            if getattr(getattr(settings.analysis, name), "symbol", "")
                            else ()
                        )
                    )
                }
                for name in live
                if getattr(settings.analysis, name, None) is not None
            )
            if trades_it:
                assert symbol in {
                    settings.instruments.canonical_symbol(name).upper()
                    for name in settings.risk.hard_spread_ceiling_by_symbol
                }

    def test_removing_it_refuses_to_load(self) -> None:
        """Load time, not startup, so no process can reach that state -- not
        from a script, not from a research run, not from a `model_copy`."""
        settings = live_settings()
        assert "section_fifteen_btc_m1" in settings.analysis.confluence.live_enabled_modules
        stripped = settings.model_copy(
            update={"risk": settings.risk.model_copy(update={"hard_spread_ceiling_by_symbol": {}})}
        )
        with pytest.raises(ValueError, match="hard_spread_ceiling_by_symbol"):
            type(settings).model_validate(stripped.model_dump())

    def test_taking_the_section_off_the_allowlist_is_the_other_way_out(self) -> None:
        """The refusal names two remedies and both have to work, or it is a
        wall rather than a choice."""
        settings = live_settings()
        confluence = settings.analysis.confluence
        without_btc = tuple(name for name in confluence.live_enabled_modules if "btc" not in name)
        candidate = settings.model_copy(
            update={
                "risk": settings.risk.model_copy(update={"hard_spread_ceiling_by_symbol": {}}),
                "analysis": settings.analysis.model_copy(
                    update={
                        "confluence": confluence.model_copy(
                            update={"live_enabled_modules": without_btc}
                        )
                    }
                ),
            }
        )
        # No exception: no live section trades a capped market any more.
        type(settings).model_validate(candidate.model_dump())

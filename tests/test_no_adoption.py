"""A symbol can be tradable by Jarvis and still not adoptable from the owner.

WHY THIS EXISTS. `observation_only_symbols` answered two different questions
with one answer:

    may Jarvis OPEN a position here?
    may Jarvis touch a position it DID NOT open here?

The owner trades gold by hand, so the second answer must stay no --
`manual_positions` is enabled and adopts magic 0, which is exactly what a
hand-placed MT5 order carries. Without the entry, Jarvis takes his own gold
trades over, manages them, and closes them.

But the same entry forbade section six from opening its own gold scalp, and
that cost more than it saved. Gold is the largest source of setups this
section has and the cheapest market it can reach:

    XAUUSD    796 setups / 30 days    cost 23.5%    <- blocked
    XAUEUR    792 setups              cost 26.5%    <- traded
    US30      480 setups              cost 44.0%    <- traded

So blocking it did not keep section six out of gold. It pushed it onto the
derived crosses, where the spread is wider and the momentum candle is often the
currency leg moving rather than the metal. The most expensive version of the
same trade.

`no_adoption_symbols` splits the two questions. These tests hold the half that
protects the owner's money, and they are deliberately blunt: every one of them
fails loudly if a refactor ever lets a magic-0 gold ticket back into the book.
"""

from __future__ import annotations

from types import SimpleNamespace

JARVIS_MAGIC = 770101
BY_HAND = 0


def _position(symbol: str, magic: int, ticket: int = 1):  # type: ignore[no-untyped-def]
    return SimpleNamespace(symbol=symbol, magic=magic, ticket=ticket)


def _runner(positions, **overrides):  # type: ignore[no-untyped-def]
    """A JarvisRunner with only what `_managed_positions` reads."""
    from config.loader import DEFAULT_CONFIG_PATH, load_settings
    from runner.service import JarvisRunner, OperationMode

    service = object.__new__(JarvisRunner)
    service.settings = load_settings(
        DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
    )
    service.operation = overrides.get("operation", OperationMode.EXPERIMENTAL_LIVE)
    service.broker = SimpleNamespace(positions=lambda: positions)  # type: ignore[assignment]
    return service


class TestTheOwnersGoldIsStillUntouchable:
    def test_a_hand_placed_gold_ticket_is_not_in_the_book(self) -> None:
        """The one that matters. Magic 0 is what MT5 gives a manual order, and
        `manual_positions` adopts magic 0 on every other symbol."""
        service = _runner([_position("XAUUSD", BY_HAND)])

        assert service._managed_positions() == []

    def test_the_same_ticket_on_any_other_symbol_is_adopted(self) -> None:
        """Proves the refusal is about the symbol and not about magic 0 in
        general -- otherwise this test passes for the wrong reason and manual
        adoption is quietly broken everywhere."""
        service = _runner([_position("EURUSD", BY_HAND)])

        assert len(service._managed_positions()) == 1

    def test_it_is_refused_before_it_ever_reaches_the_journal(self) -> None:
        """Two gates, and both are needed. `_managed_positions` decides what
        the management loop SEES; `_adopt_manual_positions` decides what gets
        written into our books as ours.

        Guarding only the first would adopt a hand-placed gold trade, record it
        as Jarvis's, and then decline to manage it -- the worst of both, and it
        would read as a bug rather than a policy.
        """
        service = _runner([])

        assert service._adoption_refused("XAUUSD")
        assert not service._adoption_refused("EURUSD")


class TestJarvisMayStillTradeItsOwnGold:
    def test_its_own_gold_position_is_managed_normally(self) -> None:
        """The whole point of the change. A section six scalp carries Jarvis's
        own magic, so the stop, the claim and the health exit all reach it."""
        service = _runner([_position("XAUUSD", JARVIS_MAGIC)])

        assert len(service._managed_positions()) == 1

    def test_both_can_be_open_at_once_and_only_one_is_ours(self) -> None:
        """The realistic case: the owner is long gold by hand while section six
        scalps it. Two tickets, same symbol, and exactly one belongs to us."""
        service = _runner(
            [
                _position("XAUUSD", BY_HAND, ticket=1),
                _position("XAUUSD", JARVIS_MAGIC, ticket=2),
            ]
        )
        managed = service._managed_positions()

        assert [position.ticket for position in managed] == [2]

    def test_opening_is_no_longer_refused_by_name(self) -> None:
        """`is_hands_off` is what the entry gate consults. Gold has to be out
        of it for section six to open anything, and out of `is_ignored` for the
        scanner to produce a context at all."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        instruments = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).instruments

        assert not instruments.is_hands_off("XAUUSD")
        assert not instruments.is_ignored("XAUUSD")
        assert instruments.refuses_adoption("XAUUSD")


class TestThePredicateItself:
    """Built off the shipped config rather than a bare `InstrumentsConfig`, so
    the broker's real suffix map and mode whitelists are the ones under test."""

    def instruments(self, **changes):  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        base = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).instruments
        return base.model_copy(update=changes)

    def test_it_matches_the_brokers_own_name_for_the_symbol(self) -> None:
        """Eightcap renames some symbols. A list that only matched the
        canonical name would silently protect nothing on a broker that
        suffixes them, and the failure is invisible: the ticket is adopted and
        nothing logs."""
        config = self.instruments(no_adoption_symbols=("XAUUSD",))

        assert config.refuses_adoption("XAUUSD")
        assert config.refuses_adoption("xauusd")
        assert config.refuses_adoption(config.broker_symbol("XAUUSD"))

    def test_an_empty_list_refuses_nothing(self) -> None:
        """The default has to be "adopt as before", or this field changes
        behaviour on every account that never set it."""
        config = self.instruments(no_adoption_symbols=())

        assert not config.refuses_adoption("XAUUSD")

    def test_it_is_not_the_same_question_as_hands_off(self) -> None:
        """If these two ever collapse back into one predicate the split is
        gone, and nobody notices until Jarvis closes a manual trade."""
        config = self.instruments(no_adoption_symbols=("XAUUSD",), observation_only_symbols=())

        assert config.refuses_adoption("XAUUSD")
        assert not config.is_hands_off("XAUUSD")

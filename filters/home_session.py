"""Is the money that moves this instrument actually awake?

The session filter next door asks one question: is the clock inside a session
this mode trades. With `tradable_sessions: [asia, london, newyork]` that is
true around twenty-two hours a day, and it is the same answer for every symbol
in the catalogue. It never asks the question that decides whether a quote means
anything, which is whether *this instrument's own* market is open.

Two live trades made the gap concrete.

  NDX100 short, opened 00:51 UTC. The Nasdaq cash market had been shut for five
  hours. What was quoting was the overnight CFD book.

  EURCAD short, opened 03:03 UTC. Frankfurt was closed and Toronto was closed.
  Neither of the two currencies in the pair had a home market open; the price
  was being made by nobody in particular.

Both passed the session filter because Asia was active, and Asia is on the
allowed list. Asia has nothing to do with either instrument.

Why this is not the spread filter's job. A thin book does not necessarily quote
a wide spread — the market maker will happily show two pips on EURCAD at three
in the morning — it quotes a price with very little behind it. The structure an
H1 chart shows was drawn by real volume during London and New York, and it is
being traded during a session that had no part in making it. The spread filter
measures what the trade costs to open. This measures whether the reasoning
behind it applies right now.

What it deliberately does not do:

  - Block on ignorance. An instrument whose home sessions cannot be worked out
    is allowed, always. Guessing at a currency nobody has mapped and silently
    refusing every trade in it is a far worse failure than letting one through.
  - Touch crypto. It genuinely trades around the clock; there is no home
    session to be closed.
  - Decide anything by itself. The verdict is also written into the payload, so
    the reviewer can see that it is judging a chart out of hours instead of
    inferring it from a timestamp.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.instrument import InstrumentSpec

#: Which session's participants set the price of each currency.
#:
#: The mapping is a fact about where a currency is used, not a measurement, so
#: it needs no lookback and no threshold. Anything absent is treated as
#: always-open by `home_sessions` returning nothing at all for it.
_CURRENCY_SESSIONS: dict[str, frozenset[str]] = {
    # Europe
    "EUR": frozenset({"london"}),
    "GBP": frozenset({"london"}),
    "CHF": frozenset({"london"}),
    "SEK": frozenset({"london"}),
    "NOK": frozenset({"london"}),
    "DKK": frozenset({"london"}),
    "PLN": frozenset({"london"}),
    "CZK": frozenset({"london"}),
    "HUF": frozenset({"london"}),
    "TRY": frozenset({"london"}),
    "ZAR": frozenset({"london"}),
    # The Americas
    "USD": frozenset({"newyork"}),
    "CAD": frozenset({"newyork"}),
    "MXN": frozenset({"newyork"}),
    "BRL": frozenset({"newyork"}),
    # Asia-Pacific
    "JPY": frozenset({"asia"}),
    "AUD": frozenset({"asia"}),
    "NZD": frozenset({"asia"}),
    "CNH": frozenset({"asia"}),
    "HKD": frozenset({"asia"}),
    "SGD": frozenset({"asia"}),
    # Metals price around the London fix and the New York futures session. Not
    # currencies, and they appear in the base of XAUUSD, so they are mapped
    # here rather than excluded.
    "XAU": frozenset({"london", "newyork"}),
    "XAG": frozenset({"london", "newyork"}),
    "XPT": frozenset({"london", "newyork"}),
    "XPD": frozenset({"london", "newyork"}),
}

#: Asset classes that have no home session because they never close.
_ALWAYS_OPEN = frozenset({"crypto"})


def home_sessions(spec: InstrumentSpec) -> frozenset[str]:
    """The sessions whose participants actually price this instrument.

    Empty means "no opinion", and every caller must read it that way rather
    than as "no session qualifies". That is the difference between a filter
    that declines to judge an unmapped instrument and one that bans it.

    An instrument quoted in its own currency — an index, where base and quote
    match — is priced by the market that currency belongs to. FRA40 in EUR is
    a bet on European equities and belongs to London; NDX100 in USD belongs to
    New York. Taking the quote currency gets both right, and it is the same
    reasoning `currency_exposure.legs` uses to conclude such an instrument has
    no currency leg at all.
    """
    if spec.asset_class.value in _ALWAYS_OPEN:
        return frozenset()
    base = spec.currency_base.upper()
    quote = spec.currency_profit.upper()
    sessions: set[str] = set()
    for code in (base, quote):
        sessions |= _CURRENCY_SESSIONS.get(code, frozenset())
    return frozenset(sessions)


def home_session_open(spec: InstrumentSpec, active: Sequence[str]) -> bool | None:
    """Whether any market that prices this instrument is currently open.

    None when there is nothing to say — an unmapped instrument, or one that
    never closes. Distinguished from False on purpose: a caller that folds the
    two together turns "I do not know" into "refuse", which is exactly the
    behaviour that makes a safety check unusable in production.
    """
    home = home_sessions(spec)
    if not home:
        return None
    return bool(home & set(active))


def describe(spec: InstrumentSpec, active: Sequence[str]) -> str:
    home = sorted(home_sessions(spec))
    running = ", ".join(active) if active else "no session"
    return (
        f"{spec.symbol} is priced in {', '.join(home)} and neither is open; "
        f"the running session is {running}. The structure on the chart was "
        f"drawn by volume that is not present now."
    )

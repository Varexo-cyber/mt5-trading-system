"""Which instruments a headline is about.

This is the part of a news layer that goes quietly wrong, and it goes wrong in
a way nothing downstream can detect. A tagger that answers "EUR" to a third of
all headlines will block a third of all EUR entries and look, from every log
and every report, exactly like a market that was busy.

So three rules hold here, and the tests hold them:

1. **Word boundaries, always.** "Goldman Sachs downgrades Ford" contains the
   substring `gold`. A naive `in` test tags it XAU and blocks gold trading on a
   story about a bank. The same trap sits in `oil` inside "toil", `yen` inside
   "Yentl", `chf` inside nothing useful, and `eur` inside "Europe" — which is
   wanted — and "neural", which is not.
2. **Say nothing rather than guess.** A headline that matches no term gets an
   empty set and reaches no instrument. Over-tagging costs skipped trades in a
   pattern nobody can see; under-tagging costs nothing that was not already
   missing before this module existed.
3. **Currencies, not symbols.** The same reasoning as
   `filters.calendar.events.symbol_currencies`: broker suffixes and non-obvious
   names ("GOLD" for XAUUSD) make symbol-name parsing wrong exactly when it
   matters. A headline is tagged with currency codes, and the instrument is
   matched through the broker's own contract spec.

What this deliberately does NOT do is read the headline's meaning. There is no
sentiment score anywhere in this package, and that is a decision rather than an
omission — see `filters.newsfeed.service.NewsPressure`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

#: Terms that identify a currency, keyed by the code the calendar already uses.
#:
#: Curated rather than generated. Each entry is a phrase whose appearance in a
#: financial headline means that currency is the subject, and the bar for
#: adding one is that it is hard to imagine it appearing otherwise. "dollar" is
#: here; "rate" is not, because every currency has rates. Central banks are
#: here because "ECB holds" never means anything but EUR.
#:
#: Ambiguity is resolved toward silence. "dollar" alone is left off the USD row
#: on purpose: Australian, Canadian, New Zealand and Singapore dollars all
#: answer to it, and a headline about the Australian dollar tagging USD would
#: put a blackout on every major. "greenback" and "us dollar" are unambiguous
#: and are what the row carries instead.
#: THE ISO CODE ITSELF IS A TERM, and leaving it out was the tagger's largest
#: hole. A live run over eighteen feeds found actionforex serving fifteen items
#: — "EUR/USD Weekly Outlook", "USD/JPY Weekly Outlook", "GBP/USD Weekly
#: Outlook" and so on — and the tagger recognised none of them. The table had
#: "euro" and "sterling" and "yen" and every central bank, and not one of the
#: three-letter codes those headlines are actually written in.
#:
#: Safe because every term is matched on word boundaries: `eur` cannot fire
#: inside "neural", `aud` cannot fire inside "audit", and the slash in a pair
#: is a non-word character, so "EUR/USD" reads as two clean tokens.
CURRENCY_TERMS: Mapping[str, tuple[str, ...]] = {
    "USD": (
        "usd",
        "us dollar",
        "u.s. dollar",
        "u.s.",
        "greenback",
        "federal reserve",
        "fed",
        "fomc",
        "powell",
        "treasury yields",
        "nonfarm",
        "non-farm",
    ),
    "EUR": (
        "eur",
        "euro",
        "eurozone",
        "euro zone",
        "european central bank",
        "ecb",
        "lagarde",
        "bundesbank",
    ),
    "GBP": (
        "gbp",
        "sterling",
        "pound",
        "uk",
        "britain",
        "british",
        "bank of england",
        "boe",
        "gilt",
        "gilts",
    ),
    "JPY": (
        "jpy",
        "yen",
        "bank of japan",
        "boj",
        "ueda",
        "jgb",
    ),
    "CHF": (
        "chf",
        "swiss franc",
        "franc",
        "swiss national bank",
        "snb",
    ),
    "AUD": (
        "aud",
        "aussie",
        "australian dollar",
        "reserve bank of australia",
        "rba",
    ),
    "NZD": (
        "nzd",
        "kiwi",
        "new zealand dollar",
        "reserve bank of new zealand",
        "rbnz",
    ),
    "CAD": (
        "cad",
        "loonie",
        "canadian dollar",
        "bank of canada",
        "boc",
    ),
    "XAU": (
        "gold",
        "bullion",
        "xau",
    ),
    "XAG": (
        "silver",
        "xag",
    ),
    # The crypto-wide terms sit on both rows on purpose. A story about
    # stablecoin regulation or an exchange failure moves BTCUSD and ETHUSD
    # together, and the live run showed why it matters: three Cointelegraph
    # items, none of which said "bitcoin" or "ethereum", all of which were
    # plainly about the asset class.
    "BTC": (
        "btc",
        "bitcoin",
        "crypto",
        "cryptocurrency",
        "stablecoin",
        "stablecoins",
        "digital asset",
        "digital assets",
    ),
    "ETH": (
        "eth",
        "ethereum",
        "ether",
        "crypto",
        "cryptocurrency",
        "stablecoin",
        "stablecoins",
        "digital asset",
        "digital assets",
    ),
    # Not currencies, and in here anyway. An index and a barrel of oil have no
    # ISO code, but `InstrumentSpec.currency_base` carries a pseudo-code for
    # exactly these and the calendar's `symbol_currencies` already passes it
    # through untouched. Putting them in the same table means one lookup and
    # one set of word-boundary rules rather than a second mechanism that would
    # drift.
    #
    # A note on what is NOT here: individual equities. "Apple misses on
    # revenue" is a real story about a real instrument, but the ticker space is
    # tens of thousands of names, half of them ordinary English words -- Gap,
    # Ford, Target, Shell, Visa -- and the substring traps this module exists
    # to avoid get very much worse. Index-level terms cover the same risk for
    # the instruments this account actually trades.
    "US500": ("s&p 500", "s&p500", "spx", "wall street", "wall st"),
    "US100": ("nasdaq", "ndx"),
    "US30": ("dow jones", "the dow"),
    "UK100": ("ftse",),
    "DE40": ("dax",),
    "FRA40": ("cac 40", "cac40"),
    "JP225": ("nikkei",),
    "AUS200": ("asx 200", "asx200"),
    # Bare "oil" as well as the grades. The live run had oilprice.com serving
    # "Oil Traders Stay Bearish Despite Deepening Middle East Disruptions" and
    # the tagger missed it, because the table only had "oil prices". Word
    # boundaries keep it out of "turmoil" and "boiler".
    "OIL": ("oil", "crude", "brent", "wti", "opec"),
    "NGAS": ("natural gas",),
}

#: Terms whose subject is every market at once, tagged onto every currency the
#: caller cares about rather than onto one.
#:
#: A headline reading "risk assets sell off as war escalates" names no currency
#: and moves all of them. Tagging it with nothing would be the tagger's most
#: expensive miss, because these are precisely the moments an automated system
#: should not be opening anything.
SYSTEMIC_TERMS: tuple[str, ...] = (
    "risk-off",
    "risk off",
    "flight to safety",
    "circuit breaker",
    "market crash",
    "flash crash",
    "emergency meeting",
    "emergency rate",
    "state of emergency",
    "declares war",
    "invasion",
    "military strike",
    "airstrike",
    "sanctions",
    "default",
    "bailout",
    "contagion",
    "bank run",
)


def _pattern(terms: Iterable[str]) -> re.Pattern[str]:
    """One alternation per group, anchored on word boundaries.

    Compiled once at import. The service tags every headline against every
    group on every refresh, and a regex rebuilt per call was the difference
    between this being free and this being noticeable on a one-core VPS.

    Longest first, so "us dollar" is preferred over a bare "dollar" if both
    were ever in the same group — alternation in Python is first-match, not
    longest-match, and the ordering is the only thing that decides it.
    """
    ordered = sorted({term.lower() for term in terms}, key=len, reverse=True)
    joined = "|".join(re.escape(term) for term in ordered)
    return re.compile(rf"(?<!\w)(?:{joined})(?!\w)")


_CURRENCY_PATTERNS: dict[str, re.Pattern[str]] = {
    code: _pattern(terms) for code, terms in CURRENCY_TERMS.items()
}
_SYSTEMIC_PATTERN = _pattern(SYSTEMIC_TERMS)


def is_systemic(text: str) -> bool:
    """Does this headline concern every market rather than one currency?"""
    return _SYSTEMIC_PATTERN.search(text.lower()) is not None


def currencies_in(text: str) -> frozenset[str]:
    """Currency codes this headline is about. Empty when it is about none.

    Empty is a real and common answer. Most of what a financial wire publishes
    is about an equity, a company or a country's politics, and reaches no pair
    this account trades.
    """
    lowered = text.lower()
    return frozenset(
        code for code, pattern in _CURRENCY_PATTERNS.items() if pattern.search(lowered)
    )

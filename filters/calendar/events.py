"""Economic calendar events, and which instruments they touch.

An event is relevant to a symbol when it hits either of the symbol's two
currencies. That comes from the broker's contract spec (`currency_base` /
`currency_profit`) rather than from parsing the symbol name, because broker
suffixes ("EURUSD.pro", "EURUSDm") and non-obvious names ("GOLD" for XAUUSD)
make name parsing quietly wrong exactly when it matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum


class Impact(IntEnum):
    """Expected market impact. Ordered so comparisons read naturally."""

    HOLIDAY = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    @classmethod
    def parse(cls, text: str) -> Impact:
        """Map a provider's impact label onto our scale.

        Unknown labels become HIGH, not LOW. An impact string we do not
        recognise is a reason to be cautious, and treating it as harmless would
        be the one failure mode the news filter exists to prevent.
        """
        normalised = text.strip().lower()
        if normalised in ("high", "red", "3"):
            return cls.HIGH
        if normalised in ("medium", "orange", "moderate", "2"):
            return cls.MEDIUM
        if normalised in ("low", "yellow", "1"):
            return cls.LOW
        if normalised in ("holiday", "non-economic", "none", "0", ""):
            return cls.HOLIDAY
        return cls.HIGH


@dataclass(frozen=True, slots=True)
class EconomicEvent:
    """One scheduled release. `when` is always tz-aware UTC."""

    when: datetime
    currency: str
    title: str
    impact: Impact
    source: str = ""
    forecast: str = ""
    previous: str = ""
    actual: str = ""

    def __post_init__(self) -> None:
        if self.when.tzinfo is None:
            raise ValueError(f"{self.title}: event time must be tz-aware")

    @property
    def key(self) -> tuple[str, str, str]:
        """Identity for de-duplication across providers.

        Rounded to the minute: two feeds routinely disagree by a few seconds on
        the same release, and treating those as distinct events would double
        every blackout window.
        """
        return (
            self.when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M"),
            self.currency.upper(),
            self.title.strip().lower(),
        )

    def is_extreme(self, keywords: tuple[str, ...]) -> bool:
        """True for the releases that move everything, not just their currency.

        Matched on the title because impact ratings do not distinguish "FOMC
        rate decision" from an ordinary high-impact print, and the two deserve
        very different blackout windows.
        """
        title = self.title.lower()
        return any(keyword.lower() in title for keyword in keywords)

    def affects(self, currencies: frozenset[str]) -> bool:
        return self.currency.upper() in currencies

    def describe(self) -> str:
        return f"{self.when:%Y-%m-%d %H:%M} UTC {self.currency} {self.impact.name} — {self.title}"


#: Bases that are priced against the dollar wherever they trade. Gold quoted in
#: yen is still gold: the yen leg is an exchange-rate wrapper around a dollar
#: market, and the events that move it are American.
_DOLLAR_DENOMINATED_BASES = frozenset({"XAU", "XAG", "XPT", "XPD", "BTC", "ETH", "LTC", "XRP"})


def symbol_currencies(currency_base: str, currency_profit: str) -> frozenset[str]:
    """Currencies whose news moves this instrument.

    THE ONE THAT MOVES GOLD IS NOT IN THE PAIR. This used to return exactly the
    two legs, and said so itself: "metals and crypto carry a pseudo-currency
    base that no calendar publishes events for... it simply never matches". That
    sentence describes a hole rather than a design. On XAUAUD the blackout was
    watching Australia while the instrument was waiting on American inflation.

    A live XAUAUD short is the case. It was working, a red-folder release moved
    gold, and the stop was taken for -1.01R and EUR 6.82 on a EUR 172 account —
    the largest single loss of the day. The calendar had the event. Nothing
    asked it, because USD is neither leg of XAUAUD. The same trade on XAUJPY
    reported its next news as 3,509 minutes away: fifty-eight hours of clear
    sky, counted over Japanese releases only.

    So a dollar-denominated base adds USD. Gold in yen is gold; the yen leg
    prices the wrapper, and CPI, payrolls and the FOMC price the metal.

    The base itself stays in the set. It still never matches, and removing it
    would be deciding which side of a pair counts on a day when a calendar
    finally does publish something for it.
    """
    base, quote = currency_base.upper(), currency_profit.upper()
    currencies = {base, quote}
    if base in _DOLLAR_DENOMINATED_BASES:
        currencies.add("USD")
    return frozenset(currencies)


def deduplicate(events: list[EconomicEvent]) -> list[EconomicEvent]:
    """Merge events that two providers both reported, keeping the higher impact.

    When feeds disagree on impact we take the more severe reading. Under-
    blocking is the expensive error here; over-blocking costs a skipped setup.
    """
    best: dict[tuple[str, str, str], EconomicEvent] = {}
    for event in events:
        existing = best.get(event.key)
        if existing is None or event.impact > existing.impact:
            best[event.key] = event
    return sorted(best.values(), key=lambda e: e.when)

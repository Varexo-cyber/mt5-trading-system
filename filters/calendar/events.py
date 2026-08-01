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


def symbol_currencies(currency_base: str, currency_profit: str) -> frozenset[str]:
    """Currencies whose news moves this instrument.

    Metals and crypto carry a pseudo-currency base (XAU, BTC) that no calendar
    publishes events for. It is kept in the set anyway rather than filtered
    out: it simply never matches, and dropping it would mean silently deciding
    which side of a pair counts.
    """
    return frozenset({currency_base.upper(), currency_profit.upper()})


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

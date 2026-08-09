"""A headline, and what a group of them says about one instrument."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

from filters.newsfeed.tagging import currencies_in, is_systemic


@dataclass(frozen=True, slots=True)
class Headline:
    """One item off a wire. `published` is always tz-aware UTC."""

    published: datetime
    title: str
    source: str
    link: str = ""

    def __post_init__(self) -> None:
        if self.published.tzinfo is None:
            raise ValueError(f"{self.title}: published time must be tz-aware")

    @property
    def key(self) -> str:
        """Identity for de-duplication across feeds.

        The title alone, normalised. Two wires carrying the same story publish
        it minutes apart with different links and slightly different
        timestamps, so keying on any of those counts one event several times —
        and this whole layer measures *how many* things are happening, which
        makes double-counting the failure that matters.

        Hashed rather than kept as text because the set of seen keys is held
        for the length of the retention window and a wire is verbose.
        """
        squashed = " ".join(self.title.lower().split())
        return hashlib.blake2b(squashed.encode("utf-8"), digest_size=16).hexdigest()

    @property
    def currencies(self) -> frozenset[str]:
        return currencies_in(self.title)

    @property
    def systemic(self) -> bool:
        return is_systemic(self.title)

    def touches(self, currencies: frozenset[str]) -> bool:
        """Is this headline about any of these currencies?

        A systemic headline touches everything. "Risk assets sell off as war
        escalates" names no currency and moves all of them, and the moment it
        appears is exactly when an automated system should not be opening
        anything.
        """
        return self.systemic or bool(self.currencies & currencies)

    def age_minutes(self, now: datetime) -> float:
        return (now.astimezone(UTC) - self.published.astimezone(UTC)).total_seconds() / 60.0


@dataclass(frozen=True, slots=True)
class NewsPressure:
    """How much is being written about one instrument, against its own normal.

    DELIBERATELY NOT SENTIMENT. There is no score here for whether the news is
    good or bad, and that is the central design decision of this package rather
    than a piece not built yet.

    The reason is latency. By the time a story is on a public RSS feed, the
    move it describes has happened — the wire is behind the tape by seconds at
    best and minutes routinely, and a retail VPS is behind the wire again.
    Trading the direction of a headline from here is buying what somebody else
    already bought. Every measurement this project has made says the entries
    are indistinguishable from a coin flip; a sentiment score would be a sixth
    coin with a confident voice.

    What survives the latency is the fact that something is happening. "Nine
    stories about the yen in twelve minutes, against a normal of one" is true
    whether or not anyone can read them, it is true before the direction is
    knowable, and it is a good reason for an automated system to keep its hands
    still. That is what this measures.
    """

    symbol: str
    currencies: frozenset[str]
    #: Headlines touching this instrument inside the recent window.
    recent: int
    #: What that window normally carries, from the same feeds over the long
    #: window. A rate, so it is comparable to `recent` directly.
    baseline: float
    #: Whether any of the recent ones were market-wide rather than currency-
    #: specific. Reported separately because it is a different kind of reason.
    systemic: bool
    window_minutes: float
    #: The newest few, for a human and for the reviewer. Not used in any
    #: arithmetic.
    latest: tuple[Headline, ...] = ()

    @property
    def multiple(self) -> float:
        """How many times its usual rate this instrument is running at.

        The baseline is floored at one story per window before dividing. Not a
        fudge: without it an instrument whose normal is 0.02 stories per window
        shows a fiftyfold spike the first time anyone mentions it, which is how
        a quiet pair ends up permanently blocked by a single routine mention.
        """
        return self.recent / max(self.baseline, 1.0)

    def describe(self) -> str:
        head = self.latest[0].title if self.latest else "no headline captured"
        return (
            f"{self.recent} headline(s) on {'/'.join(sorted(self.currencies))} in "
            f"{self.window_minutes:.0f} min against a normal of {self.baseline:.1f}"
            f"{' (market-wide story)' if self.systemic else ''} — {head}"
        )

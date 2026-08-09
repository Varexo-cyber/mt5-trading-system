"""Learning *why* setups get refused, not just which ones did.

`VetoMemory` remembers the shape of a proposal — instrument, direction, entry
and stop measured in ATR — and stays quiet while that shape holds. It is the
right memory for "you already asked me this". It is the wrong memory for what
a live account actually produced, which was five refusals of GBPCAD LONG that
were not the same proposal at all: different entries, different stops,
different cycles, and every single one refused as a counter-trend long.

The shape moved every time. The flaw never did. So the shape memory forgot,
and the account paid five cents each time to be told the same thing.

The durable fact is the reason. A reviewer that has called four GBPCAD longs
counter-trend in six hours is describing the daily chart, not those four
entries, and the fifth long will be counter-trend too. This module learns that
sentence.

Three properties keep it a memory rather than a mute button:

1. **Reasons are grouped, not matched literally.** Claude writes prose, and
   "Counter-trend long", "countertrend against D1" and "fighting the higher
   timeframe trend" are one observation in three sentences. Matching strings
   would learn nothing; matching tags learns the market.
2. **Evidence is required, and it expires.** One refusal is an anecdote about a
   moment. Silence starts at the third within the window, and the window rolls
   forward, so a pattern that stops recurring stops applying on its own.
3. **An approval erases it.** The moment the reviewer approves this instrument
   and direction, the accumulated pattern is wrong by demonstration and is
   dropped outright. That is the check that stops a bad early run from
   silencing an instrument for good.

Deliberately narrower than it could be. It only ever suppresses a *paid API
call*; it can never approve, size, or open anything, and every deterministic
gate still runs in full on every cycle.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from infra.atomic import write_json_atomic
from infra.logging import get_logger

log = get_logger(__name__)

#: Refusals of one tag needed before it silences the next question. Two is an
#: anecdote — the same market at two nearby moments. Three separate cycles
#: reaching the same conclusion is the reviewer describing the chart.
DEFAULT_MIN_OCCURRENCES = 3

#: How long a refusal counts as evidence. Six hours spans a session without
#: reaching across the D1 close that most of these tags are about; past it the
#: structure being described has usually changed.
DEFAULT_WINDOW_HOURS = 6.0

#: Tags, and the phrases that map onto them. Ordered by specificity — the first
#: match wins, so narrow tags must precede broad ones.
#:
#: Kept deliberately small. Every tag here is something that recurs on the same
#: instrument for a structural reason and is therefore predictive of the next
#: proposal; a tag for "the market looked choppy" would be neither.
_TAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "countertrend",
        ("counter-trend", "countertrend", "counter trend", "against the higher", "against d1"),
    ),
    (
        "stop-too-tight",
        ("stop is only", "stop is tight", "tight stop", "stop inside", "noise band"),
    ),
    ("stop-too-wide", ("stop is wide", "wide stop", "oversized stop")),
    ("target-unreachable", ("target is", "unreachable", "rarely reached", "never reached")),
    ("low-conviction", ("weakest", "low-conviction", "low conviction", "ranked dead last")),
    ("momentum-against", ("momentum", "selling into", "buying into")),
    ("structure-missing", ("no clear structure", "range", "chop", "no trend")),
    ("spread", ("spread",)),
    ("stale", ("stale", "quote too old", "quote age")),
)


def classify(risks: tuple[str, ...] | list[str], thesis: str = "") -> tuple[str, ...]:
    """Reduce a reviewer's prose to the tags it is really making.

    Reads the `risks` list first because it is already terse and close to a
    label; falls back to the thesis only when there are no risks, since a full
    paragraph will match half the vocabulary and dilute everything.

    Returns an empty tuple when nothing matches. That is the honest answer and
    the safe one: an unrecognised objection teaches nothing, and inventing a
    tag for it would silence future questions on evidence that was never
    understood.
    """
    text = " ".join(str(item) for item in risks).lower()
    if not text.strip():
        text = thesis.lower()
    if not text.strip():
        return ()

    found: list[str] = []
    for tag, phrases in _TAGS:
        if any(phrase in text for phrase in phrases) and tag not in found:
            found.append(tag)
    return tuple(found)


@dataclass(frozen=True, slots=True)
class Pattern:
    """A recurring objection to one instrument in one direction."""

    tag: str
    occurrences: int
    first_seen: datetime
    last_seen: datetime

    def describe(self) -> str:
        hours = (self.last_seen - self.first_seen).total_seconds() / 3600.0
        span = f"{hours:.0f}h" if hours >= 1 else f"{hours * 60:.0f}min"
        return f"{self.tag} ({self.occurrences} refusals over {span})"


@dataclass
class _Entry:
    tags: list[tuple[str, str]] = field(default_factory=list)  # (tag, iso timestamp)


class VetoPatterns:
    """What the reviewer keeps saying about an instrument, and for how long."""

    def __init__(
        self,
        path: Path,
        *,
        min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
        window_hours: float = DEFAULT_WINDOW_HOURS,
    ) -> None:
        self.path = path
        self.min_occurrences = max(1, min_occurrences)
        self.window = timedelta(hours=max(0.1, window_hours))
        self._entries: dict[str, _Entry] = {}
        self._load()

    # -- persistence -------------------------------------------------------

    @staticmethod
    def _key(symbol: str, direction: str) -> str:
        return f"{symbol}|{direction.upper()}"

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt file is not worth failing a trading session over. The
            # memory rebuilds itself from the next few refusals.
            log.warning(
                "veto pattern file unreadable; starting empty", extra={"path": str(self.path)}
            )
            return
        for key, value in (raw.get("entries") or {}).items():
            tags = [
                (str(item[0]), str(item[1]))
                for item in value.get("tags", [])
                if isinstance(item, list | tuple) and len(item) == 2
            ]
            self._entries[key] = _Entry(tags)

    def _save(self) -> None:
        payload = {
            "entries": {key: {"tags": entry.tags} for key, entry in self._entries.items()},
        }
        try:
            write_json_atomic(self.path, payload)
        except OSError:
            log.warning("could not persist veto patterns", extra={"path": str(self.path)})

    # -- recording ---------------------------------------------------------

    def remember(
        self,
        symbol: str,
        direction: str,
        *,
        risks: tuple[str, ...] | list[str],
        thesis: str,
        now: datetime,
    ) -> tuple[str, ...]:
        """File the tags behind one refusal. Returns what was recognised."""
        tags = classify(risks, thesis)
        if not tags:
            return ()
        key = self._key(symbol, direction)
        entry = self._entries.setdefault(key, _Entry())
        stamp = now.isoformat()
        entry.tags.extend((tag, stamp) for tag in tags)
        self._prune(entry, now)
        self._save()
        log.info(
            "recorded a veto pattern",
            extra={
                "event": "veto_pattern_recorded",
                "symbol": symbol,
                "direction": direction,
                "tags": list(tags),
            },
        )
        return tags

    def clear(self, symbol: str, direction: str) -> None:
        """Forget everything for this instrument and direction.

        Called on an approval. The reviewer has just said yes to exactly the
        pair the pattern claimed was hopeless, so the pattern is wrong by
        demonstration — not weakened, wrong. Anything softer leaves a bad early
        run able to silence an instrument long after the market has moved on.
        """
        if self._entries.pop(self._key(symbol, direction), None) is not None:
            self._save()

    # -- lookup ------------------------------------------------------------

    def _prune(self, entry: _Entry, now: datetime) -> None:
        cutoff = now - self.window
        entry.tags = [
            (tag, stamp) for tag, stamp in entry.tags if _parse(stamp) and _parse(stamp) > cutoff
        ]

    def established(self, symbol: str, direction: str, now: datetime) -> Pattern | None:
        """The strongest objection standing against this pair right now.

        None when there is not enough recent evidence, which is the normal
        answer and the one that costs an API call.
        """
        key = self._key(symbol, direction)
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._prune(entry, now)
        if not entry.tags:
            self._entries.pop(key, None)
            return None

        counts = Counter(tag for tag, _ in entry.tags)
        tag, occurrences = counts.most_common(1)[0]
        if occurrences < self.min_occurrences:
            return None
        stamps = sorted(_parse(stamp) for t, stamp in entry.tags if t == tag)
        stamps = [item for item in stamps if item is not None]
        if not stamps:
            return None
        return Pattern(tag, occurrences, stamps[0], stamps[-1])


def _parse(text: str) -> datetime | None:
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def readable(tag: str) -> str:
    """A tag as an operator sentence, for the log and the deck.

    A raw slug on screen explains nothing, and these lines are the only place
    the operator learns what the system has concluded about an instrument.
    """
    return {
        "countertrend": "de trade gaat tegen de hogere trend in",
        "stop-too-tight": "de stop ligt binnen de ruis",
        "stop-too-wide": "de stop is te breed voor dit account",
        "target-unreachable": "het doel wordt op deze markt zelden gehaald",
        "low-conviction": "te zwakke setup vergeleken met de rest",
        "momentum-against": "het momentum loopt tegen de richting in",
        "structure-missing": "geen bruikbare structuur, het is een range",
        "spread": "de spread is te groot voor deze stop",
        "stale": "de koers is te oud om op te handelen",
    }.get(tag, tag)

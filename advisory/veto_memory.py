"""Durable memory of refused setups, so the same question is not bought twice.

The scanner ranks the catalogue the same way every cycle, so the same handful of
instruments comes top over and over. Without a memory that meant paying Claude to
re-derive an answer already on file: one session sent SPX500 LONG three times in
two minutes and got the identical veto, word for word, all three times; an
earlier one sent EURCAD seventy-four times in twenty minutes.

A per-bar cache does not fix that on its own. It expires when a new bar closes,
and a new bar closing is not the same thing as the setup changing — a market that
grinds sideways for six hours produces six new H1 bars and six identical vetoes.
What actually changed is the question that matters, and the question is defined
by the *shape of the proposal*: which instrument, which direction, and where the
entry and stop sit. While those are the same within a fraction of ATR, the
reviewer has already answered.

Three properties make this a memory rather than a mute button:

1. **It forgets when the setup moves.** The fingerprint is measured in ATR, not
   in price, so "materially different" means the same thing on gold as on
   EURUSD. Once entry or stop shifts past the tolerance the veto no longer
   applies and the new proposal is asked afresh.
2. **It backs off further the more it is right.** A symbol that keeps producing
   refused variations of the same idea earns a longer silence each time, capped.
   That is the part that learns: repeated refusal is evidence the instrument is
   not worth the API spend right now, and the system acts on its own evidence.
3. **It survives a restart.** Written to disk, so stopping the service is not a
   way to accidentally re-buy several hundred vetoes.

Deliberately one-sided: only vetoes are remembered here. An approval that did
not become a trade is cheap to re-derive and stale approvals are exactly what
should not persist.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.clock import Clock, LiveClock
from infra.atomic import write_json_atomic
from infra.logging import get_logger

log = get_logger(__name__)

#: How far entry or stop may drift and still count as "the same setup", measured
#: in ATR of the signal timeframe. A quarter of an average bar's range is inside
#: the noise the reviewer explicitly said it was not trading; beyond it the
#: structure genuinely differs.
DEFAULT_TOLERANCE_ATR = 0.25

#: Silence after the first veto. Long enough to skip many cycles at a 30s loop,
#: short enough that a real intraday reversal is not missed.
DEFAULT_BASE_MINUTES = 90.0

#: The ceiling on escalation. A day of silence on one symbol and direction is
#: plenty; past that the market has genuinely moved on and deserves a fresh look.
DEFAULT_MAX_MINUTES = 1440.0

#: Records retained. One per symbol and direction, and the catalogue is ~850
#: instruments, so this covers a full sweep of refusals with room to spare.
DEFAULT_CAPACITY = 2000


@dataclass(frozen=True, slots=True)
class VetoRecord:
    """One refused setup, its shape, and how long it stays refused."""

    symbol: str
    direction: str
    entry: float
    stop: float
    atr: float
    thesis: str
    confidence: float
    repeats: int
    first_seen: str
    last_seen: str
    suppress_until: str

    @property
    def expires_at(self) -> datetime:
        return _parse(self.suppress_until)

    @property
    def last_seen_at(self) -> datetime:
        """When this refusal was most recently earned.

        Distinct from `expires_at`, which carries the escalating suppression.
        A caller wanting "how long since we last paid for this" must not read
        the expiry: after three repeats those are twelve hours apart.
        """
        return _parse(self.last_seen)

    def matches(self, entry: float, stop: float, tolerance_atr: float) -> bool:
        """Whether a new proposal is the same shape as the one refused.

        Scaled by the ATR recorded *with the veto*, not the current one. Using
        today's ATR would let a volatility spike silently widen the tolerance
        and swallow a setup that really had moved.
        """
        if self.atr <= 0:
            # No usable scale. Fall back to exactness rather than guessing a
            # tolerance in raw price, which means nothing across instruments.
            return entry == self.entry and stop == self.stop
        allowed = self.atr * tolerance_atr
        return abs(entry - self.entry) <= allowed and abs(stop - self.stop) <= allowed

    def describe(self, now: datetime) -> str:
        remaining = self.expires_at - now
        minutes = max(0, int(remaining.total_seconds() // 60))
        return (
            f"Claude already refused this setup {self.repeats}x "
            f"(last confidence {self.confidence:.2f}); unchanged within "
            f"{DEFAULT_TOLERANCE_ATR:.2f} ATR, so it is not being re-sent for "
            f"another {minutes} min. Original reason: {self.thesis}"
        )


class VetoMemory:
    """Remembers refusals across cycles and restarts, and forgets on change."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        tolerance_atr: float = DEFAULT_TOLERANCE_ATR,
        base_minutes: float = DEFAULT_BASE_MINUTES,
        max_minutes: float = DEFAULT_MAX_MINUTES,
        capacity: int = DEFAULT_CAPACITY,
        clock: Clock | None = None,
    ) -> None:
        self.path = path
        self.tolerance_atr = tolerance_atr
        self.base_minutes = base_minutes
        self.max_minutes = max_minutes
        self.capacity = capacity
        self.clock = clock or LiveClock()
        self._records: dict[tuple[str, str], VetoRecord] = {}
        self._load()

    # ------------------------------------------------------------------ read

    def recall(
        self,
        symbol: str,
        direction: str,
        entry: float,
        stop: float,
        now: datetime,
    ) -> VetoRecord | None:
        """The live refusal covering this proposal, or None to go and ask."""
        record = self._records.get((symbol, direction))
        if record is None:
            return None
        if now >= record.expires_at:
            # Expired: drop it so the next refusal starts a fresh escalation
            # rather than inheriting a count from hours ago.
            del self._records[(symbol, direction)]
            self._save()
            return None
        if not record.matches(entry, stop, self.tolerance_atr):
            return None
        return record

    def standing(self, symbol: str, direction: str, now: datetime) -> VetoRecord | None:
        """A live refusal for this market and side, whatever the price has done.

        `recall` answers "may this exact proposal be suppressed" and is strict
        about it -- same entry and stop within a quarter of an ATR -- because
        silencing a setup that genuinely moved would be discarding new
        evidence. On a live tick that window is almost never hit: the deck
        showed forty-five paid calls against one served from memory, thirty-two
        of them refused.

        This answers a different and much cheaper question: has the reviewer
        turned this direction down lately. Nothing may be blocked on it. It
        exists so a scarce paid review goes to a market that was not just
        refused, and it must never be wired into a gate.
        """
        record = self._records.get((symbol, direction))
        if record is None or now >= record.expires_at:
            return None
        return record

    def active(self, now: datetime) -> list[VetoRecord]:
        """Every refusal still in force, newest suppression first."""
        return sorted(
            (record for record in self._records.values() if now < record.expires_at),
            key=lambda record: record.expires_at,
            reverse=True,
        )

    # ----------------------------------------------------------------- write

    def remember(
        self,
        symbol: str,
        direction: str,
        *,
        entry: float,
        stop: float,
        atr: float,
        thesis: str,
        confidence: float,
        now: datetime,
    ) -> VetoRecord:
        """Record a refusal, escalating if this symbol keeps earning them."""
        previous = self._records.get((symbol, direction))
        repeats = 1
        first_seen = now
        if previous is not None and now < previous.expires_at + timedelta(
            minutes=self.base_minutes
        ):
            # Still within a window of the last refusal, so this is the same
            # argument continuing rather than a new one. Escalate.
            repeats = previous.repeats + 1
            first_seen = _parse(previous.first_seen)
        minutes = min(self.max_minutes, self.base_minutes * (2 ** (repeats - 1)))
        record = VetoRecord(
            symbol=symbol,
            direction=direction,
            entry=entry,
            stop=stop,
            atr=atr,
            thesis=thesis[:400],
            confidence=confidence,
            repeats=repeats,
            first_seen=_iso(first_seen),
            last_seen=_iso(now),
            suppress_until=_iso(now + timedelta(minutes=minutes)),
        )
        self._records[(symbol, direction)] = record
        self._evict(now)
        self._save()
        log.info(
            "remembering an AI veto so it is not re-purchased",
            extra={
                "event": "ai_veto_remembered",
                "symbol": symbol,
                "direction": direction,
                "repeats": repeats,
                "silent_minutes": round(minutes),
            },
        )
        return record

    def clear(self, symbol: str, direction: str) -> None:
        """Forget a refusal outright — the setup was approved, or traded."""
        if self._records.pop((symbol, direction), None) is not None:
            self._save()

    # -------------------------------------------------------------- internal

    def _evict(self, now: datetime) -> None:
        expired = [key for key, record in self._records.items() if now >= record.expires_at]
        for key in expired:
            del self._records[key]
        while len(self._records) > self.capacity:
            oldest = min(self._records, key=lambda key: self._records[key].last_seen)
            del self._records[oldest]

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning(
                "veto memory unreadable; starting empty",
                extra={"event": "veto_memory_unreadable", "path": str(self.path)},
            )
            return
        for row in payload.get("records", []) if isinstance(payload, dict) else []:
            try:
                record = VetoRecord(**row)
            except TypeError:
                # A record written by an older shape. Dropping it costs one
                # re-review; refusing to start costs the whole session.
                continue
            self._records[(record.symbol, record.direction)] = record

    def _save(self) -> None:
        if self.path is None:
            return
        write_json_atomic(
            self.path,
            {
                "version": 1,
                "written_at": _iso(self.clock.now()),
                "records": [asdict(record) for record in self._records.values()],
            },
        )


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def _parse(text: str) -> datetime:
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

"""What the system has learned so far, carried into the next decision.

Until now the learning loop had a hole in the middle. `reflect()` was called on
every closed trade, cost money, produced genuine observations — and wrote them
to `runtime/ai_reviews.jsonl`, which nothing ever read back. The reviewer began
each judgement with no idea that the last four trades on this instrument had all
been stopped out the same way. Every cycle started from zero.

This module closes that loop. It accumulates three things across the account's
whole life and hands them to the reviewer as context:

1. **Lessons** from post-trade reflections, deduplicated and counted. A lesson
   stated once is an anecdote; the same lesson arriving from four separate
   trades is a pattern, and the count is what makes the difference visible.
2. **A per-instrument scoreboard.** Realised R by symbol and direction. "Long
   SPX500 has been taken four times and lost 3.2R in total" is a harder fact
   than any narrative about it.
3. **Refusals.** How often a symbol and direction has been vetoed recently, so
   the reviewer can see it is being asked the same question again.

Three deliberate limits, because a learning system that can rewrite itself is
how an account dies:

- **It is context, never control.** Everything here becomes text in a prompt.
  No number in this file can move a risk limit, a weight, a threshold or a lot
  size — those change only by an explicit edit to config, visible in a diff.
  The reviewer may become more sceptical about an instrument; it cannot become
  more leveraged on one.
- **It is bounded.** Caps on lessons retained and symbols tracked, so the prompt
  cannot grow without limit and quietly become the dominant cost of a review.
- **It decays.** Statistics older than the retention window are dropped. A
  market regime from three months ago is not evidence about this morning, and a
  memory that never forgets converges on a permanent opinion.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.clock import Clock, LiveClock
from infra.atomic import write_json_atomic
from infra.logging import get_logger

log = get_logger(__name__)

#: Distinct lessons retained. Enough to carry real patterns, small enough that
#: the briefing stays a paragraph rather than an essay.
MAX_LESSONS = 40

#: Lessons actually sent to the reviewer, highest evidence first. The rest stay
#: on disk building up their counts for when they become significant.
BRIEFED_LESSONS = 8

#: Instruments tracked in the scoreboard.
MAX_SYMBOLS = 200

#: How long an observation counts as evidence about the present.
RETENTION_DAYS = 90

# One hundred closed trades is the project's pre-registered minimum for an
# empirical conclusion. The memory can surface observations earlier, but it
# must label them honestly so a fluent reflection from one trade cannot acquire
# the authority of a measured pattern merely by being fed back into a prompt.
MIN_STATISTICAL_SAMPLE = 100
DEVELOPING_SAMPLE = 30


@dataclass
class SymbolRecord:
    """Realised track record for one symbol and direction."""

    symbol: str
    direction: str
    trades: int = 0
    wins: int = 0
    total_r: float = 0.0
    vetoes: int = 0
    last_seen: str = ""

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    def summary(self) -> str:
        if not self.trades:
            return f"{self.symbol} {self.direction}: refused {self.vetoes}x, never traded"
        return (
            f"{self.symbol} {self.direction}: {self.trades} trades, "
            f"{self.win_rate:.0%} won, {self.total_r:+.2f}R total"
            + (f", refused {self.vetoes}x" if self.vetoes else "")
        )


@dataclass
class Lesson:
    """One observation, and how much evidence stands behind it."""

    text: str
    occurrences: int = 1
    first_seen: str = ""
    last_seen: str = ""
    symbols: list[str] = field(default_factory=list)

    def summary(self) -> str:
        seen = f" (seen {self.occurrences}x" if self.occurrences > 1 else " (seen once"
        where = f" on {', '.join(self.symbols[:4])}" if self.symbols else ""
        return f"{self.text}{seen}{where})"


class TradingMemory:
    """Durable, bounded, decaying record of what the account has been taught."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        retention_days: int = RETENTION_DAYS,
        clock: Clock | None = None,
    ) -> None:
        self.path = path
        self.retention_days = retention_days
        self.clock = clock or LiveClock()
        self._lessons: dict[str, Lesson] = {}
        self._symbols: dict[tuple[str, str], SymbolRecord] = {}
        self._closed_trades = 0
        self._total_r = 0.0
        self._processed_trade_ids: set[int] = set()
        self._reflected_trade_ids: set[int] = set()
        self._load()

    # ---------------------------------------------------------------- record

    def record_reflection(
        self,
        outcome: Mapping[str, object],
        lessons: tuple[str, ...],
        now: datetime | None = None,
        *,
        trade_id: int | None = None,
    ) -> None:
        """Fold one post-trade reflection into the accumulated record."""
        if trade_id is not None and trade_id in self._reflected_trade_ids:
            return
        moment = now or self.clock.now()
        symbol = str(outcome.get("symbol", "") or "")
        for raw in lessons:
            text = " ".join(str(raw).split())[:300]
            if len(text) < 12:
                # Too short to be an observation. "Be careful" teaches nothing
                # and would crowd out something that does.
                continue
            key = _normalise(text)
            existing = self._lessons.get(key)
            if existing is None:
                self._lessons[key] = Lesson(
                    text=text,
                    first_seen=_iso(moment),
                    last_seen=_iso(moment),
                    symbols=[symbol] if symbol else [],
                )
            else:
                existing.occurrences += 1
                existing.last_seen = _iso(moment)
                # Keep the wording of the first statement. Later paraphrases of
                # the same point are the same point, and swapping the text each
                # time would make the count look like drift.
                if symbol and symbol not in existing.symbols:
                    existing.symbols.append(symbol)
        if trade_id is not None:
            self._reflected_trade_ids.add(trade_id)
        self._prune(moment)
        self._save()

    def record_outcome(
        self,
        symbol: str,
        direction: str,
        pnl_r: float,
        now: datetime | None = None,
        *,
        trade_id: int | None = None,
    ) -> None:
        """Fold one closed trade's realised result into the scoreboard."""
        if trade_id is not None and trade_id in self._processed_trade_ids:
            return
        moment = now or self.clock.now()
        record = self._symbols.setdefault(
            (symbol, direction), SymbolRecord(symbol=symbol, direction=direction)
        )
        record.trades += 1
        record.wins += 1 if pnl_r > 0 else 0
        record.total_r += pnl_r
        record.last_seen = _iso(moment)
        self._closed_trades += 1
        self._total_r += pnl_r
        if trade_id is not None:
            self._processed_trade_ids.add(trade_id)
        self._prune(moment)
        self._save()

    def synchronize_outcomes(
        self, outcomes: Iterable[Mapping[str, object]], now: datetime | None = None
    ) -> None:
        """Rebuild realised statistics from the journal without duplicating them.

        The journal is the source of truth. Older runner versions only updated
        memory for closures they observed directly, so broker-side closures
        could exist in SQLite without reaching this file. Rebuilding at start
        repairs that drift and makes every later `record_outcome` idempotent.
        Reflection lessons and veto counts are preserved; only arithmetic that
        can be recomputed exactly is replaced.
        """
        moment = now or self.clock.now()
        cutoff = moment - timedelta(days=self.retention_days)
        for record in self._symbols.values():
            record.trades = 0
            record.wins = 0
            record.total_r = 0.0
        self._closed_trades = 0
        self._total_r = 0.0
        self._processed_trade_ids.clear()

        for outcome in outcomes:
            try:
                trade_id = int(outcome["id"])
                closed_at = _parse(str(outcome["closed_at"]))
                symbol = str(outcome["symbol"])
                direction = str(outcome["direction"])
                pnl_r = float(outcome["pnl_r"])
            except (KeyError, TypeError, ValueError):
                continue
            if closed_at < cutoff:
                continue
            record = self._symbols.setdefault(
                (symbol, direction), SymbolRecord(symbol=symbol, direction=direction)
            )
            record.trades += 1
            record.wins += int(pnl_r > 0)
            record.total_r += pnl_r
            record.last_seen = max(record.last_seen, _iso(closed_at))
            self._closed_trades += 1
            self._total_r += pnl_r
            self._processed_trade_ids.add(trade_id)
        self._prune(moment)
        self._save()

    def has_reflection(self, trade_id: int) -> bool:
        """Whether a successful structured reflection already exists."""
        return trade_id in self._reflected_trade_ids

    def synchronize_reflections(
        self, rows: Iterable[Mapping[str, object]], now: datetime | None = None
    ) -> None:
        """Rebuild lessons once per trade from the durable AI audit.

        Earlier versions could reflect the same broker closure more than once.
        Replaying the audit through the trade-ID guard removes those duplicate
        lesson counts and restores reflections that existed in JSONL but were
        never folded into memory.
        """
        material = list(rows)
        if not material:
            return
        self._lessons.clear()
        self._reflected_trade_ids.clear()
        fallback = now or self.clock.now()
        for row in material:
            outcome = row.get("outcome")
            reflection = row.get("reflection")
            if not isinstance(outcome, Mapping) or not isinstance(reflection, Mapping):
                continue
            try:
                trade_id = int(outcome["trade_id"])
            except (KeyError, TypeError, ValueError):
                continue
            raw_lessons = reflection.get("lessons", ())
            if not isinstance(raw_lessons, (list, tuple)):
                continue
            timestamp = _parse(str(row.get("timestamp", ""))) if row.get("timestamp") else fallback
            self.record_reflection(
                outcome,
                tuple(str(lesson) for lesson in raw_lessons),
                timestamp,
                trade_id=trade_id,
            )
        self._prune(fallback)
        self._save()

    def record_veto(self, symbol: str, direction: str, now: datetime | None = None) -> None:
        """Note that a proposal here was refused, for the reviewer's context."""
        moment = now or self.clock.now()
        record = self._symbols.setdefault(
            (symbol, direction), SymbolRecord(symbol=symbol, direction=direction)
        )
        record.vetoes += 1
        record.last_seen = _iso(moment)
        self._prune(moment)
        self._save()

    # ----------------------------------------------------------------- brief

    def briefing(self, symbol: str = "", direction: str = "") -> dict[str, object]:
        """The compact, prompt-ready view of everything learned so far.

        Ordered by evidence rather than recency: a lesson drawn from six trades
        outranks last night's, because the point of carrying this forward is to
        surface patterns and not to relitigate the most recent loss.
        """
        ranked = sorted(
            self._lessons.values(),
            key=lambda lesson: (lesson.occurrences, lesson.last_seen),
            reverse=True,
        )
        here = self._symbols.get((symbol, direction)) if symbol and direction else None
        worst = sorted(
            (record for record in self._symbols.values() if record.trades >= 2),
            key=lambda record: record.total_r,
        )[:5]
        if self._closed_trades >= MIN_STATISTICAL_SAMPLE:
            evidence_status = "MINIMUM_SAMPLE_REACHED"
        elif self._closed_trades >= DEVELOPING_SAMPLE:
            evidence_status = "DEVELOPING"
        else:
            evidence_status = "ANECDOTAL_ONLY"
        return {
            "closed_trades_recorded": self._closed_trades,
            "cumulative_r": round(self._total_r, 2),
            "minimum_sample": MIN_STATISTICAL_SAMPLE,
            "evidence_status": evidence_status,
            "lessons": [lesson.summary() for lesson in ranked[:BRIEFED_LESSONS]],
            "this_instrument": here.summary() if here is not None else "no history on record",
            "worst_performing": [record.summary() for record in worst],
            "guardrail": (
                "Fewer than 100 closed trades is not statistical evidence. Do not approve or "
                "veto solely because of this memory; use it only to identify a concrete feature "
                "that is independently visible in the supplied market data."
                if self._closed_trades < MIN_STATISTICAL_SAMPLE
                else "The minimum account-level sample has been reached, but instrument and "
                "regime subgroups may still be too small. Correlation is not causation."
            ),
            "note": (
                "Accumulated from this account's own closed trades. Reflections are structured "
                "research notes, not model retraining and not permission to rewrite production "
                "parameters. A symbol with no history is neutral."
            ),
        }

    def has_evidence(self) -> bool:
        """Whether there is enough here to be worth sending at all."""
        return bool(self._lessons) or self._closed_trades > 0

    # -------------------------------------------------------------- internal

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(days=self.retention_days)
        for key, lesson in list(self._lessons.items()):
            if lesson.last_seen and _parse(lesson.last_seen) < cutoff:
                del self._lessons[key]
        for key, record in list(self._symbols.items()):
            if record.last_seen and _parse(record.last_seen) < cutoff:
                del self._symbols[key]
        while len(self._lessons) > MAX_LESSONS:
            # Drop the weakest evidence, not the oldest: a single-occurrence
            # note from yesterday is less useful than a pattern from last month.
            weakest = min(
                self._lessons,
                key=lambda key: (self._lessons[key].occurrences, self._lessons[key].last_seen),
            )
            del self._lessons[weakest]
        while len(self._symbols) > MAX_SYMBOLS:
            oldest = min(self._symbols, key=lambda key: self._symbols[key].last_seen)
            del self._symbols[oldest]

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning(
                "trading memory unreadable; starting empty",
                extra={"event": "memory_unreadable", "path": str(self.path)},
            )
            return
        if not isinstance(payload, dict):
            return
        self._closed_trades = int(payload.get("closed_trades", 0))
        self._total_r = float(payload.get("total_r", 0.0))
        self._processed_trade_ids = {int(value) for value in payload.get("processed_trade_ids", [])}
        self._reflected_trade_ids = {int(value) for value in payload.get("reflected_trade_ids", [])}
        for row in payload.get("lessons", []):
            try:
                lesson = Lesson(**row)
            except TypeError:
                continue
            self._lessons[_normalise(lesson.text)] = lesson
        for row in payload.get("symbols", []):
            try:
                record = SymbolRecord(**row)
            except TypeError:
                continue
            self._symbols[(record.symbol, record.direction)] = record

    def _save(self) -> None:
        if self.path is None:
            return
        write_json_atomic(
            self.path,
            {
                "version": 2,
                "written_at": _iso(self.clock.now()),
                "closed_trades": self._closed_trades,
                "total_r": round(self._total_r, 4),
                "processed_trade_ids": sorted(self._processed_trade_ids),
                "reflected_trade_ids": sorted(self._reflected_trade_ids),
                "lessons": [asdict(lesson) for lesson in self._lessons.values()],
                "symbols": [asdict(record) for record in self._symbols.values()],
            },
        )


def _normalise(text: str) -> str:
    """Collapse a lesson to a comparison key.

    Reflections restate the same point across trades with the wording shuffled,
    so "the stop was in the noise" and "stop in noise" have to land on one key
    or the occurrence count — the thing that distinguishes a pattern from an
    anecdote — never rises above one.

    What it absorbs: case, punctuation, word order, and the filler words below.
    What it does not: synonyms. "Inside the noise" and "within normal range"
    stay separate entries. Fixing that needs an embedding, and the cost of
    getting it wrong is two lessons that should have been merged — mild, and
    much better than two distinct lessons silently collapsing into one.
    """
    lowered = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    words = [word for word in lowered.split() if word not in _FILLER]
    return " ".join(sorted(set(words)))[:200]


_FILLER = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def _parse(text: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

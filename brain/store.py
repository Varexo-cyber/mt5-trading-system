"""The Postgres brain: writes evidence, reads it back as context.

THE RULE THAT SHAPES EVERY LINE HERE: this is memory, not a risk control.
Postgres going down must never stop a trade, close a position, or take down
the one-second guard loop. Every public method swallows its own failures and
returns a neutral answer, and `failures` counts them so the operator can see a
silent outage rather than discovering it a week later in an empty table.

That is the opposite of the calendar's contract, deliberately. A missing
calendar means the system does not know whether news is coming, so it must not
trade. A missing memory means the system cannot remember what it learned, which
is a reason to be sad and not a reason to be unsafe. The account traded for
months with a JSON file that had the same properties.

THE SECOND RULE: nothing read from this database may move a risk limit, a
threshold, a weight or a lot size. Realised trades have one bounded authority:
after the configured minimum sample they may reorder setups that already
passed every entry gate. They cannot create a setup or make it larger. All
other reads become text in a prompt and rows in a report. A learning system
that can rewrite its own risk controls is how an account dies, and a remote
database that can do it is worse, because the change would not be visible in a
diff.

The connection string lives in `NEON_DATABASE_URL` in `config/.env`, which is
gitignored. It is never logged, never written to the journal, and never sent to
the dashboard or to the Claude API.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infra.logging import get_logger

log = get_logger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

#: Environment variable carrying the DSN. Read from the process environment,
#: which `config.loader.load_credentials` has already populated from
#: `config/.env` by the time anything here runs.
DSN_ENV = "NEON_DATABASE_URL"

#: Lessons handed to the reviewer, best-evidenced first. The rest stay in the
#: table accumulating sightings until they are worth saying.
BRIEFED_LESSONS = 10

#: Closed trades needed before the account is allowed to move its own exit
#: threshold. Forty is not many, and it is deliberately not fewer: a level
#: learned from fifteen trades is a level learned from one week's weather.
MIN_TRADES_TO_LEARN = 40

#: Closed trades needed on ONE side before long and short are reported apart.
#: Lower than `MIN_TRADES_TO_LEARN` because the two do different jobs: that one
#: guards a threshold the system then acts on by itself, this one only puts a
#: sentence in front of a reviewer who can weigh it against the chart. Fifteen
#: is around where a three-to-one split between the sides stops being something
#: four trades could have produced.
MIN_TRADES_TO_SPLIT_SIDES = 15

#: Trades a single detector must have found before it is graded by name.
#: Higher than the side split because the consequence is heavier: a side that
#: has lost is a sentence in a prompt, while a detector shown to lose money is
#: an argument for switching it off, and switching off the wrong one removes
#: setups permanently.
MIN_TRADES_TO_GRADE_A_MODULE = 20

#: Floor on the learned threshold, in R. Below this the profit does not clear
#: the spread and commission it costs to collect, so banking there converts a
#: small win into a small loss. `_cost_of_leaving` measures the real figure per
#: trade; this is the blunt guard that keeps the learned value in the region
#: where that measurement can even apply.
MIN_LEARNED_BANK_R = 0.15

#: Everything here runs inside a trading cycle. A database that has gone slow
#: must cost a cycle a couple of seconds, not a minute.
CONNECT_TIMEOUT = 8
STATEMENT_TIMEOUT_MS = 5_000

_PUNCTUATION = re.compile(r"[^a-z ]+")
_WHITESPACE = re.compile(r"\s+")


def lesson_key(text: str) -> str:
    """Normalised form used to group repeats of the same observation.

    Lowercased, digits and punctuation removed, whitespace collapsed. "Stopped
    out 3 pips before the reversal." and "stopped out 5 pips before the
    reversal" become one lesson with two sightings, which is the whole point:
    an evidence count is the only thing separating a pattern from an anecdote,
    and counting near-duplicates separately inflates every count equally until
    nothing stands out.
    """
    lowered = _PUNCTUATION.sub(" ", text.lower())
    return _WHITESPACE.sub(" ", lowered).strip()


def fingerprint(*parts: object) -> str:
    """Stable identity for a row the runner may write more than once.

    A retry after a network blip must not create a second decision, because
    every aggregate downstream counts rows.
    """
    joined = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()


def _signal_rows(signals: Sequence[Any]) -> list[dict[str, Any]]:
    """Flatten the module verdicts into something a query can group by.

    Only the modules that actually said something are kept. A neutral module is
    the absence of evidence, and storing forty thousand rows of "trend_momentum
    saw nothing" would make every per-module average a measure of how often the
    detector is quiet rather than of whether it is right.
    """
    rows: list[dict[str, Any]] = []
    for signal in signals or ():
        name = str(getattr(signal, "module", "") or "")
        score = float(getattr(signal, "score", 0.0) or 0.0)
        if not name or not score:
            continue
        details = getattr(signal, "details", None)
        rows.append(
            {
                "module": name,
                "score": score,
                "confidence": float(getattr(signal, "confidence", 0.0) or 0.0),
                "direction": "LONG" if score > 0 else "SHORT",
                "details": dict(details) if isinstance(details, Mapping) else {},
            }
        )
    return rows


@dataclass
class BrainStatus:
    """What the operator needs to know about the memory without opening it."""

    connected: bool
    dsn_configured: bool
    writes: int = 0
    failures: int = 0
    last_error: str = ""

    def summary(self) -> str:
        if not self.dsn_configured:
            return f"no {DSN_ENV} configured; the brain is a local file only"
        if not self.connected:
            return f"brain unreachable ({self.last_error or 'unknown'}); running without it"
        return f"brain connected, {self.writes} writes, {self.failures} failures"


@dataclass
class Lesson:
    """One observation and how much evidence stands behind it."""

    lesson: str
    sightings: int
    mean_r: float | None
    last_seen: datetime

    def summary(self) -> str:
        evidence = f"{self.sightings}x" if self.sightings > 1 else "once"
        result = "" if self.mean_r is None else f", averaging {self.mean_r:+.2f}R"
        return f"[{evidence}{result}] {self.lesson}"


@dataclass
class Scoreline:
    """What one instrument and direction has actually returned."""

    symbol: str
    direction: str
    trades: int
    total_r: float
    wins: int
    mean_kept: float | None

    def summary(self) -> str:
        kept = "" if self.mean_kept is None else f", kept {self.mean_kept:.0%} of its best"
        return (
            f"{self.direction} {self.symbol}: {self.trades} trades, "
            f"{self.wins} won, {self.total_r:+.2f}R total{kept}"
        )


@dataclass(frozen=True, slots=True)
class SideRecord:
    """What one side of the market has returned across every instrument.

    `scoreboard` groups by symbol *and* direction, which is the right shape for
    "how has gold treated us" and the wrong shape for the question this answers.
    Sixty-four trades spread across thirty instruments produce thirty lines of
    one and two trades each: individually meaningless, and the reviewer was
    being handed the top twelve of them as though they were evidence.

    Collapsed to the side alone, the same trades say something no single chart
    can show — on this account the long book lost roughly three times what the
    short book lost over a comparable number of trades. That is the largest
    measured asymmetry in the record and nothing in the payload mentioned it.

    Evidence for the reviewer to weigh, never a gate. A side that has lost is a
    reason to want more from a setup, and it is not on its own a reason to
    refuse one: the sample is small, and refusing every long outright would be
    letting one month of weather close half the market permanently.
    """

    direction: str
    trades: int
    total_r: float
    wins: int

    @property
    def mean_r(self) -> float:
        return self.total_r / self.trades if self.trades else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    def summary(self) -> str:
        return (
            f"{self.direction}: {self.trades} trades, {self.win_rate:.0%} won, "
            f"{self.total_r:+.2f}R total, {self.mean_r:+.2f}R per trade"
        )


@dataclass(frozen=True, slots=True)
class ManagementRecord:
    """What one exit rule earned against simply leaving the trade alone.

    Every closed trade is replayed against its own untouched stop and target,
    and `lift_r` is the difference. Positive means intervening beat holding;
    negative means the rule is an expensive habit.

    Grouped by the rule that actually closed the position, because "our exits
    are costing us" names nothing to stop doing. AI_CLOSE and PEAK_STALL are
    different decisions with different records, and on this account they have
    pointed in opposite directions.
    """

    action: str
    trades: int
    total_lift_r: float
    better: int
    #: Average best R the trade went on to reach AFTER this rule closed it.
    #: The number that separates "closed at +0.1R and it ran to +2R" from
    #: "closed at +0.1R and it collapsed" — the same row in every other column.
    left_on_the_table_r: float | None = None
    #: Average gap between the best exit that was available while the trade was
    #: open and what it actually took. The losing side of the same question:
    #: a short that ran straight to its stop reports -0.89R and says nothing
    #: about the moment it was only 0.2R down.
    missed_r: float | None = None

    @property
    def mean_lift_r(self) -> float:
        return self.total_lift_r / self.trades if self.trades else 0.0

    def summary(self) -> str:
        verdict = "beat holding" if self.mean_lift_r > 0 else "cost us against holding"
        after = (
            f"; the market went on to reach {self.left_on_the_table_r:+.2f}R on average "
            f"after it acted"
            if self.left_on_the_table_r is not None
            else ""
        )
        missed = (
            f"; a better exit was available and missed by {self.missed_r:.2f}R on average"
            if self.missed_r is not None and self.missed_r > 0
            else ""
        )
        return (
            f"{self.action}: {self.trades} trades, {self.better} better than leaving it "
            f"alone, {self.mean_lift_r:+.2f}R per trade — {verdict}{after}{missed}"
        )


@dataclass(frozen=True, slots=True)
class ModuleRecord:
    """What one detector has actually earned, across every trade it found.

    The question this exists to answer had no answer anywhere: `conviction`
    stored the blended score and nothing stored what produced it, so sixty-four
    closed trades were sixty-four undifferentiated data points. The only lesson
    available from them was "we are down", which names nothing to stop doing.

    Attributed to every module that scored on the decision, not only to the
    strongest. A setup is the sum of what agreed on it, and crediting one
    detector for a trade three of them found would flatter whichever happened
    to score highest.
    """

    module: str
    trades: int
    total_r: float
    wins: int

    @property
    def mean_r(self) -> float:
        return self.total_r / self.trades if self.trades else 0.0

    def summary(self) -> str:
        return (
            f"{self.module}: {self.trades} trades found, {self.wins} won, "
            f"{self.total_r:+.2f}R total, {self.mean_r:+.2f}R each"
        )


@dataclass(frozen=True, slots=True)
class EdgeCalibration:
    """Shrunk realised expectancy for ordering one already-valid setup."""

    asset_class: str
    setup_family: str
    horizon: str
    direction: str
    regime: str
    trades: int
    mean_r: float
    modifier: float
    specificity: int

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.asset_class,
            self.setup_family,
            self.horizon,
            self.direction,
            self.regime,
        )

    def summary(self) -> str:
        return (
            f"{self.trades} realised trades in this segment average {self.mean_r:+.2f}R; "
            f"bounded ranking adjustment {self.modifier:+.2f}"
        )


@dataclass(frozen=True, slots=True)
class SelectionEvidence:
    """One shrunk facet used by the second-brain candidate ranker.

    Facets stay separate on purpose.  An exact five-column segment is usually
    empty on a small account, while one account-wide LONG/SHORT average throws
    away nearly everything the journal knows.  The runner combines matching
    facets without granting any of them entry, sizing or risk authority.
    """

    dimension: str
    value: str
    direction: str
    trades: int
    wins: int
    mean_r: float
    modifier: float

    @property
    def key(self) -> tuple[str, str, str]:
        return self.dimension, self.value, self.direction

    def summary(self) -> str:
        return (
            f"{self.dimension}={self.value}: {self.trades} realised trades, "
            f"{self.wins} won, {self.mean_r:+.2f}R average, "
            f"{self.modifier:+.2f} ranking points"
        )


@dataclass(frozen=True, slots=True)
class GateScoreline:
    """Passive grade of plans refused by one gate."""

    blocked_by: str
    observations: int
    hypothetical_wins: int
    mean_r: float

    def summary(self) -> str:
        return (
            f"{self.blocked_by}: {self.observations} resolved refused plans, "
            f"{self.hypothetical_wins} would have won, averaging {self.mean_r:+.2f}R. "
            "Counterfactual evidence only, not broker fills."
        )


class Brain:
    """Postgres-backed memory. Fails soft, always.

    Lazy about connecting: constructing one costs nothing and does no I/O, so
    a system with no DSN configured behaves exactly as it did before this
    module existed. The first write opens the connection.
    """

    def __init__(
        self,
        dsn: str = "",
        *,
        account: str = "",
        enabled: bool = True,
        connect_timeout: int = CONNECT_TIMEOUT,
    ) -> None:
        self.dsn = dsn or os.getenv(DSN_ENV, "")
        self.account = account
        self.enabled = enabled and bool(self.dsn)
        self.connect_timeout = connect_timeout
        self.status = BrainStatus(connected=False, dsn_configured=bool(self.dsn))
        self._connection: Any = None
        # The guard loop and the scan cycle both write. One connection shared
        # across threads without this is a corrupted protocol stream, which
        # surfaces as unrelated nonsense from the driver several calls later.
        self._lock = threading.Lock()

    # -- plumbing ----------------------------------------------------------

    def _connect(self) -> Any:
        import psycopg  # imported here so the package is optional

        # NOT `options="-c statement_timeout=..."`. That is the obvious way to
        # bound a query and Neon's POOLED endpoint refuses the whole connection
        # over it:
        #
        #   unsupported startup parameter in options: statement_timeout
        #   Please use unpooled connection or remove this parameter
        #
        # pgbouncer cannot pass arbitrary startup parameters through to the
        # server it hands you, so it rejects rather than silently ignores. The
        # DSN in use has `-pooler` in the host, which is the right endpoint for
        # a long-running process that reconnects, so the timeout moves instead
        # of the endpoint.
        connection = psycopg.connect(
            self.dsn,
            connect_timeout=self.connect_timeout,
            autocommit=True,
        )
        # Best effort, and it says so. In pgbouncer's transaction-pooling mode
        # a bare SET may not survive to the next statement, because the next
        # one can land on a different server connection. Attempting it costs a
        # round trip and helps on the unpooled endpoint and on plain Postgres;
        # where it does not stick, `connect_timeout` and the fail-soft wrapper
        # are what keep a slow database from holding up a trading cycle.
        with contextlib.suppress(Exception):
            connection.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
        self.status.connected = True
        self.status.last_error = ""
        return connection

    def _run(self, sql: str, params: Sequence[Any] = (), *, fetch: str = "none") -> Any:
        """Execute, reconnecting once, and never raise into the caller.

        One retry rather than none because a pooled Neon connection is closed
        from the far end after an idle period, and the first statement after
        that always fails. Retrying it is the difference between "the memory
        works" and "the memory works until the market is quiet for an hour".
        """
        if not self.enabled:
            return None
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._connection is None or self._connection.closed:
                        self._connection = self._connect()
                    with self._connection.cursor() as cursor:
                        cursor.execute(sql, params)
                        if fetch == "one":
                            return cursor.fetchone()
                        if fetch == "all":
                            return cursor.fetchall()
                        return None
                except Exception as exc:  # noqa: BLE001 - memory must not raise
                    self._connection = None
                    self.status.connected = False
                    self.status.last_error = f"{type(exc).__name__}: {exc}"
                    if attempt == 2:
                        self.status.failures += 1
                        log.warning(
                            "brain write failed; continuing without it",
                            extra={
                                "event": "brain_unavailable",
                                "why": self.status.last_error,
                                "failures": self.status.failures,
                            },
                        )
                        return None
        return None

    def migrate(self) -> bool:
        """Create every table, index and view. Safe to run repeatedly."""
        if not self.enabled:
            return False
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        # A DDL failure is worth knowing about loudly, unlike a write.
        with self._lock:
            try:
                if self._connection is None or self._connection.closed:
                    self._connection = self._connect()
                with self._connection.cursor() as cursor:
                    cursor.execute(sql)
                self.status.writes += 1
                return True
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                self._connection = None
                self.status.connected = False
                self.status.last_error = f"{type(exc).__name__}: {exc}"
                self.status.failures += 1
                return False

    def close(self) -> None:
        with self._lock:
            if self._connection is not None and not self._connection.closed:
                # Shutdown must not raise either. A connection already broken
                # from the far end throws on close, and that would take down a
                # runner that was on its way out cleanly.
                with contextlib.suppress(Exception):
                    self._connection.close()
            self._connection = None
            self.status.connected = False

    # -- writing -----------------------------------------------------------

    def record_decision(
        self,
        *,
        decided_at: datetime,
        symbol: str,
        reason: str,
        mode: str,
        direction: str = "",
        detail: str = "",
        taken: bool = False,
        equity: float | None = None,
        conviction: float | None = None,
        playbook: str = "",
        entry: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        filters: Mapping[str, Any] | None = None,
        ai: Mapping[str, Any] | None = None,
        headlines: Sequence[Mapping[str, Any]] = (),
        signals: Sequence[Any] = (),
        features: Mapping[str, float] | None = None,
        setup_family: str = "",
    ) -> int | None:
        """One row for every decision, taken or refused. Returns its id.

        The refusals matter more than the trades and there are two thousand of
        them for every one. Without them "is this system refusing the right
        things" cannot be asked at all, and it is the question with the most
        money behind it right now.
        """
        import json

        verdict = dict(ai or {})
        measured = dict(filters or {})
        intelligence = measured.get("market_intelligence")
        intelligence = intelligence if isinstance(intelligence, Mapping) else {}
        asset_context = intelligence.get("asset_context")
        asset_context = asset_context if isinstance(asset_context, Mapping) else {}
        asset_class = measured.get("asset_class") or asset_context.get("asset_class")
        regime = measured.get("volatility_regime") or intelligence.get("regime")
        session = measured.get("session")
        horizon = measured.get("trade_horizon")
        planning_timeframe = measured.get("planning_timeframe")
        row = self._run(
            """
            INSERT INTO decisions (
                fingerprint, decided_at, account, mode, symbol, direction, reason,
                detail, taken, equity, conviction, playbook, asset_class, regime,
                session, horizon, planning_timeframe, entry, stop_loss,
                take_profit, filters, ai_verdict, ai_confidence, ai_reasoning,
                ai_tokens, headlines, signals, features, setup_family
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s::jsonb,
                %s::jsonb, %s::jsonb, %s
            )
            ON CONFLICT (fingerprint) DO UPDATE SET detail = EXCLUDED.detail
            RETURNING id
            """,
            (
                fingerprint(decided_at.isoformat(), self.account, symbol, direction, reason),
                decided_at,
                self.account,
                mode,
                symbol,
                direction or None,
                reason,
                detail,
                taken,
                equity,
                conviction,
                playbook or None,
                asset_class,
                regime,
                session,
                horizon,
                planning_timeframe,
                entry,
                stop_loss,
                take_profit,
                json.dumps(measured, default=str),
                verdict.get("verdict"),
                verdict.get("confidence"),
                verdict.get("reasoning"),
                verdict.get("tokens"),
                json.dumps(list(headlines), default=str),
                json.dumps(_signal_rows(signals), default=str),
                json.dumps(dict(features or {}), default=str),
                setup_family,
            ),
            fetch="one",
        )
        if row is None:
            return None
        self.status.writes += 1
        return int(row[0])

    def record_trade_opened(
        self,
        *,
        ticket: int,
        decision_id: int | None,
        symbol: str,
        direction: str,
        volume: float,
        opened_at: datetime,
        entry: float,
        stop_loss: float,
        take_profit: float | None,
        risk_money: float,
    ) -> int | None:
        row = self._run(
            """
            INSERT INTO trades (
                account, ticket, decision_id, symbol, direction, volume,
                opened_at, entry, stop_loss, take_profit, risk_money
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (account, ticket, opened_at) DO UPDATE
                SET stop_loss = EXCLUDED.stop_loss, updated_at = NOW()
            RETURNING id
            """,
            (
                self.account,
                ticket,
                decision_id,
                symbol,
                direction,
                volume,
                opened_at,
                entry,
                stop_loss,
                take_profit,
                risk_money,
            ),
            fetch="one",
        )
        if row is None:
            return None
        self.status.writes += 1
        return int(row[0])

    def record_management_outcome(
        self,
        *,
        local_trade_id: int,
        resolved_at: datetime,
        symbol: str,
        direction: str,
        exit_action: str,
        baseline_pnl_r: float,
        actual_pnl_r: float,
        after_exit_best_r: float | None = None,
        after_exit_worst_r: float | None = None,
        best_exit_r: float | None = None,
        minutes_to_best_exit: float | None = None,
    ) -> None:
        """What stepping in was worth on one closed trade.

        The replay against the untouched original stop and target already ran
        for every closed trade and already wrote its answer — to local SQLite,
        where the layer that decides hold-versus-close every second cannot read
        it. So that judgement has been made on this account for weeks with no
        idea what its own interventions have earned, while the answer sat in a
        file on the VPS.
        """
        self._run(
            """
            INSERT INTO management_outcomes (
                account, local_trade_id, resolved_at, symbol, direction,
                exit_action, baseline_pnl_r, actual_pnl_r, lift_r,
                after_exit_best_r, after_exit_worst_r,
                best_exit_r, missed_r, minutes_to_best_exit
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (account, local_trade_id) DO NOTHING
            """,
            (
                self.account,
                local_trade_id,
                resolved_at,
                symbol,
                direction,
                exit_action,
                baseline_pnl_r,
                actual_pnl_r,
                actual_pnl_r - baseline_pnl_r,
                after_exit_best_r,
                after_exit_worst_r,
                best_exit_r,
                (best_exit_r - actual_pnl_r) if best_exit_r is not None else None,
                minutes_to_best_exit,
            ),
        )
        self.status.writes += 1

    def management_records(self, minimum_trades: int = 3) -> list[ManagementRecord]:
        """Per exit rule: what it took, what holding would have paid, the gap.

        The one thing the judgement layer needs and has never been given. It is
        asked "hold or close" about once a second and has no record of whether
        closing has helped — on THIS account, with THESE stops.

        Withheld below `minimum_trades` per rule rather than reported with a
        caveat. "AI_CLOSE: 1 trade, +0.64R better" is not a weaker version of
        the finding, it is a different and false one, and an adviser handed a
        number will use it however much hedging surrounds it.
        """
        rows = self._run(
            """
            SELECT exit_action, COUNT(*), SUM(lift_r), COUNT(*) FILTER (WHERE lift_r > 0),
                   AVG(after_exit_best_r), AVG(missed_r)
            FROM management_outcomes
            WHERE account = %s AND exit_action <> ''
            GROUP BY exit_action
            HAVING COUNT(*) >= %s
            ORDER BY COUNT(*) DESC
            """,
            (self.account, minimum_trades),
            fetch="all",
        )
        if not rows:
            return []
        return [
            ManagementRecord(
                action=str(row[0]),
                trades=int(row[1]),
                total_lift_r=float(row[2] or 0.0),
                better=int(row[3]),
                left_on_the_table_r=(float(row[4]) if row[4] is not None else None),
                missed_r=(float(row[5]) if row[5] is not None else None),
            )
            for row in rows
        ]

    def supervision_examples(self, limit: int = 2000) -> list[dict[str, Any]]:
        """Past open-position judgements, in the shape the local model matches on.

        The nearest-neighbour adviser needs comparable past states and was
        reading them out of `runtime/ai_reviews.jsonl` — a file on a rented VPS
        that starts empty after a fresh clone. It requires five comparable
        states before it may act on a position, so on an empty file it holds
        everything, forever, while the account's whole history sits here.

        Rows without a feature vector are skipped rather than defaulted. They
        predate the `features` column, and matching an all-zero shape against a
        live position returns confident nonsense — an empty archive is an
        honest answer, a fabricated neighbour is not.
        """
        rows = self._run(
            """
            SELECT symbol, direction, action, confidence, r_at_the_time, features,
                   stop_fraction
            FROM supervisions
            WHERE account = %s AND features <> '{}'::JSONB AND direction <> ''
            ORDER BY asked_at DESC
            LIMIT %s
            """,
            (self.account, limit),
            fetch="all",
        )
        if not rows:
            return []
        examples: list[dict[str, Any]] = []
        for row in rows:
            features = row[5]
            if not isinstance(features, Mapping) or not features:
                continue
            examples.append(
                {
                    "symbol": str(row[0] or ""),
                    "direction": str(row[1] or ""),
                    "action": str(row[2] or ""),
                    "confidence": float(row[3] or 0.0),
                    "r_at_the_time": float(row[4] or 0.0),
                    "features": {str(k): float(v) for k, v in features.items()},
                    "stop_fraction": (float(row[6]) if row[6] is not None else None),
                }
            )
        return examples

    def entry_examples(self, limit: int = 2000) -> list[dict[str, Any]]:
        """Past setups, graded on what the market actually did about them.

        The entry matcher was reading `runtime/ai_reviews.jsonl`, where a setup's
        usefulness is the paid reviewer's own `said_yes`. That is an archive of
        OPINIONS, and it could refuse a setup the engine was sure about because a
        model which is now switched off — and cannot revise itself — once
        declined something similar.

        These rows are graded on the result. A decision that became a trade is
        useful when it closed positive; one a gate refused is useful when its
        counterfactual, the same plan left to its own stop and target, came out
        positive. Both are facts about this account rather than recollections of
        an opinion about it.

        Two LEFT JOINs and a COALESCE rather than two queries: a decision is in
        exactly one of those states and the caller wants one list. Rows with no
        resolved outcome are dropped — an unresolved setup has nothing to teach
        yet — and so are rows with no feature vector, because matching an
        all-zero shape against a live chart returns confident nonsense.
        """
        rows = self._run(
            """
            SELECT d.symbol, d.direction, d.setup_family, d.horizon, d.features,
                   COALESCE(t.pnl_r, c.pnl_r) AS realised_r,
                   d.ai_confidence, d.detail, d.reason
            FROM decisions d
            LEFT JOIN trades t
                   ON t.decision_id = d.id AND t.closed_at IS NOT NULL
            LEFT JOIN counterfactuals c
                   ON c.account = d.account
                  AND c.symbol = d.symbol
                  AND c.direction = d.direction
                  AND c.entry = d.entry
            WHERE d.account = %s
              AND d.features <> '{}'::JSONB
              AND d.direction IS NOT NULL AND d.direction <> ''
              AND COALESCE(t.pnl_r, c.pnl_r) IS NOT NULL
            ORDER BY d.decided_at DESC
            LIMIT %s
            """,
            (self.account, limit),
            fetch="all",
        )
        if not rows:
            return []
        examples: list[dict[str, Any]] = []
        for row in rows:
            features = row[4]
            if not isinstance(features, Mapping) or not features:
                continue
            examples.append(
                {
                    "symbol": str(row[0] or ""),
                    "direction": str(row[1] or ""),
                    "setup_family": str(row[2] or ""),
                    "horizon": str(row[3] or ""),
                    "features": {str(k): float(v) for k, v in features.items()},
                    # The whole reason this method exists. Positive R means the
                    # setup was worth taking whether or not anyone approved it,
                    # so a refused setup that would have won now argues FOR the
                    # next one like it instead of against it.
                    "useful": float(row[5]) > 0.0,
                    "realised_r": float(row[5]),
                    "confidence": float(row[6] or 0.0),
                    "thesis": str(row[7] or ""),
                    "blocked_by": str(row[8] or ""),
                }
            )
        return examples

    def best_exit_available(self, trade_id: int) -> tuple[float, float] | None:
        """The best R this position ever showed, and how many minutes in.

        Read from `position_path`, which is why that table exists. Its rows are
        the only record of what was on offer between the open and the close: a
        trade that ran to its stop reports -0.89R and nothing else, and the
        moment it was 0.2R down and drifting leaves no trace anywhere else.

        None when the path was never recorded — a position opened before this
        existed, or one whose samples were lost to an unreachable brain. An
        absent path is not a path with no good moment in it.
        """
        row = self._run(
            """
            SELECT MAX(r_now),
                   EXTRACT(EPOCH FROM (
                       MIN(sampled_at) FILTER (
                           WHERE r_now = (SELECT MAX(r_now) FROM position_path WHERE trade_id = %s)
                       ) - MIN(sampled_at)
                   )) / 60.0
            FROM position_path
            WHERE trade_id = %s
            """,
            (trade_id, trade_id),
            fetch="one",
        )
        if not row or row[0] is None:
            return None
        return float(row[0]), float(row[1] or 0.0)

    def record_position_path(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """The second-by-second life of open positions, in one round trip.

        Everything else here records the moments a decision was taken — opened,
        banked, closed, reviewed — so every question about MANAGEMENT has been
        answered from a trade's endpoints. "Should the stop have gone to entry
        sooner" and "how long did it sit at its high before giving up" are
        questions about the path, and the path only ever existed in local
        SQLite, on a rented VPS, in a file nobody queries across months.

        The live case that forced it: a CADCHF long showing EUR 2.82 on a
        EUR 130 account with the broker stop still twelve pips below entry.
        Whether holding that was right is answerable only against what price
        did next, second by second.

        BATCHED, AND THAT IS THE DESIGN CONSTRAINT. The guard runs about once a
        second on one vCPU, and a network round trip to Neon inside that loop
        would make the thing watching the money the slowest part of the system.
        Callers buffer and hand over a batch; this writes it in one statement
        and, like every write here, never raises into the caller.
        """
        payload = [
            (
                int(row["trade_id"]),
                row["sampled_at"],
                float(row["price"]),
                float(row["r_now"]),
                float(row["peak_r"]),
                float(row["money"]),
                row.get("stop_price"),
                row.get("stop_r"),
                bool(row.get("protected", False)),
                str(row.get("health", "") or ""),
            )
            for row in rows
            if row.get("trade_id")
        ]
        if not payload:
            return 0
        written = self._run_many(
            """
            INSERT INTO position_path (
                trade_id, sampled_at, price, r_now, peak_r, money,
                stop_price, stop_r, protected, health
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            payload,
        )
        self.status.writes += written
        return written

    def _run_many(self, sql: str, rows: Sequence[Sequence[Any]]) -> int:
        """`_run` for a batch, returning how many rows actually landed.

        Separate from `_run` rather than folded into it because the failure
        mode differs: a batch that fails has written nothing, and the caller
        needs to know that rather than assume the count it handed over. Same
        one-retry reconnect, same rule that memory never raises.
        """
        if not self.enabled or not rows:
            return 0
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._connection is None or self._connection.closed:
                        self._connection = self._connect()
                    with self._connection.cursor() as cursor:
                        cursor.executemany(sql, rows)
                    return len(rows)
                except Exception as exc:  # noqa: BLE001 - memory must not raise
                    self._connection = None
                    self.status.connected = False
                    self.status.last_error = f"{type(exc).__name__}: {exc}"
                    if attempt == 2:
                        self.status.failures += 1
                        log.warning(
                            "brain batch write failed; continuing without it",
                            extra={
                                "event": "brain_unavailable",
                                "why": self.status.last_error,
                                "rows": len(rows),
                            },
                        )
        return 0

    def record_trade_event(
        self,
        *,
        trade_id: int,
        happened_at: datetime,
        action: str,
        reason: str = "",
        r_at_action: float | None = None,
        price: float | None = None,
        money: float | None = None,
    ) -> None:
        """Every move the guard made, in order.

        This is the operator's "what happened when and why": BREAK_EVEN at
        13:42 on +0.31R, PROFIT_BANKED at 13:58 because the move stopped
        running. The journal already holds it; here it survives the machine and
        can be grouped across months.
        """
        self._run(
            """
            INSERT INTO trade_events (
                trade_id, happened_at, action, reason, r_at_action, price, money
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (trade_id, happened_at, action, reason, r_at_action, price, money),
        )
        self.status.writes += 1

    def record_trade_closed(
        self,
        *,
        ticket: int,
        closed_at: datetime,
        exit_price: float,
        exit_reason: str,
        pnl_money: float,
        pnl_r: float,
        mfe_r: float | None = None,
        mae_r: float | None = None,
    ) -> None:
        self._run(
            """
            UPDATE trades SET
                closed_at = %s, exit_price = %s, exit_reason = %s,
                pnl_money = %s, pnl_r = %s,
                mfe_r = COALESCE(%s, mfe_r), mae_r = COALESCE(%s, mae_r),
                updated_at = NOW()
            WHERE account = %s AND ticket = %s AND closed_at IS NULL
            """,
            (
                closed_at,
                exit_price,
                exit_reason,
                pnl_money,
                pnl_r,
                mfe_r,
                mae_r,
                self.account,
                ticket,
            ),
        )
        self.status.writes += 1

    def record_counterfactual(
        self,
        *,
        symbol: str,
        direction: str,
        blocked_by: str,
        opened_at: datetime,
        entry: float,
        stop_loss: float,
        take_profit: float,
        resolved_at: datetime,
        outcome: str,
        pnl_r: float,
    ) -> None:
        """Persist how one refused executable plan subsequently resolved.

        Counterfactuals grade gates and the adviser. They are intentionally in
        their own table so a hypothetical fill can never enter the realised
        calibration query by accident.
        """
        self.record_counterfactuals(
            [
                {
                    "symbol": symbol,
                    "direction": direction,
                    "blocked_by": blocked_by,
                    "opened_at": opened_at,
                    "entry": entry,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "resolved_at": resolved_at,
                    "outcome": outcome,
                    "pnl_r": pnl_r,
                }
            ]
        )

    def record_trade_history(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """Copy closed trades the local journal has and this database does not.

        WHY THIS EXISTS, and it cost real money before it did. The brain only
        ever received trades opened after it was switched on, so a journal
        holding forty-seven closed trades faced a Neon table holding twenty-two.
        `learned_bank_threshold` needs forty before it will speak, so it stayed
        silent, `bank_at_r` stayed at its configured 0.30, and a USDCAD short
        that peaked at +0.23R -- seven hundredths under the line -- was closed
        at -0.10R instead of banked. The account's own history said to take
        0.15R, and the account could not see its own history.

        Counterfactuals already had this catch-up. Realised trades, which are
        what every learned threshold is actually built on, did not.

        Returns how many rows were sent rather than how many were new: the
        unique constraint settles that server-side, and asking would cost a
        round trip to learn nothing worth acting on.
        """
        import json

        payload = [dict(row) for row in rows]
        if not payload:
            return 0
        self._run(
            """
            INSERT INTO trades (
                account, ticket, symbol, direction, volume, opened_at, entry,
                stop_loss, take_profit, risk_money, closed_at, exit_price,
                exit_reason, pnl_money, pnl_r, mfe_r, mae_r
            )
            SELECT
                %s, row.ticket, row.symbol, row.direction, row.volume,
                row.opened_at, row.entry, row.stop_loss, row.take_profit,
                row.risk_money, row.closed_at, row.exit_price, row.exit_reason,
                row.pnl_money, row.pnl_r, row.mfe_r, row.mae_r
            FROM jsonb_to_recordset(%s::jsonb) AS row(
                ticket BIGINT, symbol TEXT, direction TEXT, volume NUMERIC,
                opened_at TIMESTAMPTZ, entry NUMERIC, stop_loss NUMERIC,
                take_profit NUMERIC, risk_money NUMERIC, closed_at TIMESTAMPTZ,
                exit_price NUMERIC, exit_reason TEXT, pnl_money NUMERIC,
                pnl_r NUMERIC, mfe_r NUMERIC, mae_r NUMERIC
            )
            -- Never overwrite. A row already here was written by the runner as
            -- it happened, with its decision_id attached; a backfilled copy
            -- knows strictly less and must not replace the better record.
            ON CONFLICT (account, ticket, opened_at) DO NOTHING
            """,
            (self.account, json.dumps(payload, default=str)),
        )
        self.status.writes += len(payload)
        return len(payload)

    def record_counterfactuals(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Persist resolved refusals in one Neon round trip.

        Startup catch-up may contain hundreds of local rows. Sending those one
        statement at a time would delay the scanner for minutes on a remote
        database, so Postgres expands one JSON payload server-side.
        """
        import json

        payload: list[dict[str, Any]] = []
        for row in rows:
            opened_at = row["opened_at"]
            payload.append(
                {
                    "fingerprint": fingerprint(
                        self.account,
                        row["symbol"],
                        row["direction"],
                        row["blocked_by"],
                        opened_at.isoformat() if isinstance(opened_at, datetime) else opened_at,
                        row["entry"],
                        row["stop_loss"],
                        row["take_profit"],
                    ),
                    **dict(row),
                    "account": self.account,
                }
            )
        if not payload:
            return
        self._run(
            """
            INSERT INTO counterfactuals (
                fingerprint, account, symbol, direction, blocked_by, opened_at,
                entry, stop_loss, take_profit, resolved_at, outcome, pnl_r
            )
            SELECT
                row.fingerprint, row.account, row.symbol, row.direction,
                row.blocked_by, row.opened_at, row.entry, row.stop_loss,
                row.take_profit, row.resolved_at, row.outcome, row.pnl_r
            FROM jsonb_to_recordset(%s::jsonb) AS row(
                fingerprint TEXT,
                account TEXT,
                symbol TEXT,
                direction TEXT,
                blocked_by TEXT,
                opened_at TIMESTAMPTZ,
                entry NUMERIC,
                stop_loss NUMERIC,
                take_profit NUMERIC,
                resolved_at TIMESTAMPTZ,
                outcome TEXT,
                pnl_r NUMERIC
            )
            ON CONFLICT (fingerprint) DO UPDATE SET
                resolved_at = EXCLUDED.resolved_at,
                outcome = EXCLUDED.outcome,
                pnl_r = EXCLUDED.pnl_r,
                updated_at = NOW()
            """,
            (json.dumps(payload, default=str),),
        )
        self.status.writes += len(payload)

    def record_lessons(
        self,
        lessons: Sequence[str],
        *,
        learned_at: datetime,
        symbol: str = "",
        direction: str = "",
        pnl_r: float | None = None,
        trade_id: int | None = None,
    ) -> None:
        """One row per lesson, not one blob per reflection.

        One row each is what turns "this has now arrived from nine separate
        trades" into a GROUP BY. Stored as a blob it would be a text search
        over paragraphs, which is why the JSON memory caps out at forty.
        """
        for text in lessons:
            cleaned = " ".join(text.split())
            if not cleaned:
                continue
            self._run(
                """
                INSERT INTO lessons (
                    trade_id, learned_at, symbol, direction, lesson_key, lesson, pnl_r
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    trade_id,
                    learned_at,
                    symbol,
                    direction,
                    lesson_key(cleaned),
                    cleaned,
                    pnl_r,
                ),
            )
            self.status.writes += 1

    def record_supervision(
        self,
        *,
        trade_id: int | None,
        asked_at: datetime,
        symbol: str,
        action: str,
        confidence: float | None = None,
        reasoning: str = "",
        r_at_the_time: float | None = None,
        applied: bool = False,
        latency_ms: float | None = None,
        model: str = "",
        direction: str = "",
        features: Mapping[str, float] | None = None,
        stop_fraction: float | None = None,
    ) -> None:
        """What the reviewer said about an open position, and whether it was
        carried out.

        The loop this closes is the one the system could not see. Every other
        table records the account grading its own rules; this records it
        grading its own adviser. With `supervision_outcomes` beside it, "the
        reviewer said hold at +0.4R and the trade ended at -1R" becomes a
        query instead of a hunch.

        `applied` matters and is not the same as the action being non-hold: a
        verdict the risk layer refused is still evidence about the adviser, and
        counting it as acted-upon would credit or blame it for something that
        never happened.

        `features` is the shape of the position at the moment of the question,
        and it is what turns this table from a record of the adviser into
        material the adviser can learn from. Without it the local
        nearest-neighbour model cannot find comparable past states here, which
        is why it was reading a JSONL file on the VPS — a file that starts
        empty after a fresh clone, so the model holds everything forever while
        months of decisions sit in Postgres unread.
        """
        import json

        self._run(
            """
            INSERT INTO supervisions (
                trade_id, account, asked_at, symbol, action, confidence,
                reasoning, r_at_the_time, applied, latency_ms, model,
                direction, features, stop_fraction
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                trade_id,
                self.account,
                asked_at,
                symbol,
                action,
                confidence,
                reasoning,
                r_at_the_time,
                applied,
                int(latency_ms) if latency_ms is not None else None,
                model,
                direction,
                json.dumps(dict(features or {}), default=str),
                stop_fraction,
            ),
        )
        self.status.writes += 1

    def record_headlines(self, items: Sequence[Any]) -> int:
        """Keep wire copy past the few hours a feed carries.

        Without this, "what was being written when this trade opened" stops
        being answerable the same afternoon, and the link between news and
        outcome -- the thing worth learning -- can never be measured.
        """
        written = 0
        for item in items:
            self._run(
                """
                INSERT INTO headlines (
                    fingerprint, published_at, source, title, link, currencies, systemic
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (fingerprint) DO NOTHING
                """,
                (
                    item.key,
                    item.published,
                    item.source,
                    item.title,
                    item.link,
                    sorted(item.currencies),
                    item.systemic,
                ),
            )
            written += 1
        self.status.writes += written
        return written

    # -- reading -----------------------------------------------------------

    def lessons(self, *, limit: int = BRIEFED_LESSONS, symbol: str = "") -> list[Lesson]:
        """The best-evidenced observations, most-sighted first.

        Ordered by how many separate trades produced the lesson, not by recency.
        Something seen nine times is worth more than something seen this
        morning, and ordering by time is how a memory ends up dominated by
        whatever happened last.
        """
        rows = self._run(
            """
            SELECT MIN(lesson) AS lesson, COUNT(*) AS sightings,
                   AVG(pnl_r) AS mean_r, MAX(learned_at) AS last_seen
            FROM lessons
            WHERE (%s = '' OR symbol = %s)
            GROUP BY lesson_key
            ORDER BY sightings DESC, last_seen DESC
            LIMIT %s
            """,
            (symbol, symbol, limit),
            fetch="all",
        )
        if not rows:
            return []
        return [
            Lesson(
                lesson=str(row[0]),
                sightings=int(row[1]),
                mean_r=float(row[2]) if row[2] is not None else None,
                last_seen=row[3],
            )
            for row in rows
        ]

    def scoreboard(self, *, symbol: str = "", limit: int = 12) -> list[Scoreline]:
        """Realised R by instrument and direction, over the account's whole life."""
        rows = self._run(
            """
            SELECT symbol, direction, COUNT(*) AS trades, SUM(pnl_r) AS total_r,
                   COUNT(*) FILTER (WHERE pnl_r > 0) AS wins, AVG(kept) AS mean_kept
            FROM trade_history
            WHERE account = %s AND (%s = '' OR symbol = %s)
            GROUP BY symbol, direction
            ORDER BY ABS(COALESCE(SUM(pnl_r), 0)) DESC
            LIMIT %s
            """,
            (self.account, symbol, symbol, limit),
            fetch="all",
        )
        if not rows:
            return []
        return [
            Scoreline(
                symbol=str(row[0]),
                direction=str(row[1]),
                trades=int(row[2]),
                total_r=float(row[3] or 0.0),
                wins=int(row[4]),
                mean_kept=float(row[5]) if row[5] is not None else None,
            )
            for row in rows
        ]

    def side_records(self, *, minimum_trades: int = MIN_TRADES_TO_SPLIT_SIDES) -> list[SideRecord]:
        """Realised R per side, for the whole account, or nothing.

        Withheld below `minimum_trades` on a side rather than reported with a
        caveat. "LONG: 3 trades, -1.20R per trade" is not a weaker version of
        the finding, it is a different and false one, and a reviewer handed a
        number will use it however much hedging surrounds it. Silence is the
        honest output of an empty sample.
        """
        rows = self._run(
            """
            SELECT direction, COUNT(*), SUM(pnl_r), COUNT(*) FILTER (WHERE pnl_r > 0)
            FROM trade_history
            WHERE account = %s AND pnl_r IS NOT NULL AND direction <> ''
            GROUP BY direction
            HAVING COUNT(*) >= %s
            ORDER BY COUNT(*) DESC
            """,
            (self.account, minimum_trades),
            fetch="all",
        )
        if not rows:
            return []
        return [
            SideRecord(
                direction=str(row[0]),
                trades=int(row[1]),
                total_r=float(row[2] or 0.0),
                wins=int(row[3]),
            )
            for row in rows
        ]

    def module_records(
        self, *, minimum_trades: int = MIN_TRADES_TO_GRADE_A_MODULE
    ) -> list[ModuleRecord]:
        """Realised R per detector, or nothing.

        The join that makes the system improvable rather than merely busy:
        every closed trade, back to the decision that opened it, back to the
        modules that scored on it. Ask it after a few hundred trades and it
        says which detector to switch off.

        Withheld below `minimum_trades` per module for the same reason the side
        split is: a detector shown at four trades will be read as a verdict on
        the detector, and it is a verdict on four trades.
        """
        rows = self._run(
            """
            SELECT s->>'module' AS module,
                   COUNT(*)     AS trades,
                   SUM(t.pnl_r) AS total_r,
                   COUNT(*) FILTER (WHERE t.pnl_r > 0) AS wins
            FROM trades t
            JOIN decisions d ON d.id = t.decision_id
            CROSS JOIN LATERAL jsonb_array_elements(d.signals) AS s
            WHERE t.account = %s AND t.pnl_r IS NOT NULL
            GROUP BY s->>'module'
            HAVING COUNT(*) >= %s
            ORDER BY SUM(t.pnl_r)
            """,
            (self.account, minimum_trades),
            fetch="all",
        )
        if not rows:
            return []
        return [
            ModuleRecord(
                module=str(row[0]),
                trades=int(row[1]),
                total_r=float(row[2] or 0.0),
                wins=int(row[3]),
            )
            for row in rows
        ]

    def edge_calibrations(
        self,
        *,
        minimum_trades: int,
        shrinkage_trades: int,
        points_per_r: float,
        modifier_cap: float,
    ) -> list[EdgeCalibration]:
        """Bounded realised evidence for ordering already-valid candidates.

        The three UNION branches form a fallback ladder: exact regime first,
        then the same setup across regimes, then asset class plus direction.
        All read only broker-confirmed closed trades. A segment with fewer than
        ``minimum_trades`` is absent, which means exactly zero influence.
        """
        rows = self._run(
            """
            WITH realised AS (
                SELECT
                    COALESCE(d.asset_class, 'unknown') AS asset_class,
                    COALESCE(d.playbook, 'unknown') AS setup_family,
                    COALESCE(d.horizon, 'unknown') AS horizon,
                    t.direction,
                    COALESCE(d.regime, 'unknown') AS regime,
                    t.pnl_r
                FROM trades t
                LEFT JOIN decisions d ON d.id = t.decision_id
                WHERE t.account = %s AND t.closed_at IS NOT NULL AND t.pnl_r IS NOT NULL
            )
            SELECT asset_class, setup_family, horizon, direction, regime,
                   COUNT(*), AVG(pnl_r), 3 AS specificity
            FROM realised
            GROUP BY asset_class, setup_family, horizon, direction, regime
            HAVING COUNT(*) >= %s
            UNION ALL
            SELECT asset_class, setup_family, horizon, direction, '*',
                   COUNT(*), AVG(pnl_r), 2 AS specificity
            FROM realised
            GROUP BY asset_class, setup_family, horizon, direction
            HAVING COUNT(*) >= %s
            UNION ALL
            SELECT asset_class, '*', '*', direction, '*',
                   COUNT(*), AVG(pnl_r), 1 AS specificity
            FROM realised
            GROUP BY asset_class, direction
            HAVING COUNT(*) >= %s
            -- Direction alone, and it is the branch that actually fires on a
            -- small account. Two things converge here: the finer buckets need
            -- samples a forty-trade month cannot supply, and a trade
            -- backfilled from the local journal has no decision behind it, so
            -- its asset class reads 'unknown' and it can never match a live
            -- 'forex' candidate in the branches above. "This account is worse
            -- at longs than shorts" needs neither an asset class nor a setup
            -- family to be true, and on this book it is the single loudest
            -- thing the history says.
            UNION ALL
            SELECT '*', '*', '*', direction, '*',
                   COUNT(*), AVG(pnl_r), 0 AS specificity
            FROM realised
            GROUP BY direction
            HAVING COUNT(*) >= %s
            """,
            (
                self.account,
                minimum_trades,
                minimum_trades,
                minimum_trades,
                minimum_trades,
            ),
            fetch="all",
        )
        if not rows:
            return []
        estimates: list[EdgeCalibration] = []
        for row in rows:
            trades = int(row[5])
            mean_r = float(row[6] or 0.0)
            shrink = trades / (trades + max(1, shrinkage_trades))
            raw = mean_r * shrink * points_per_r
            modifier = max(-modifier_cap, min(modifier_cap, raw))
            estimates.append(
                EdgeCalibration(
                    asset_class=str(row[0]),
                    setup_family=str(row[1]),
                    horizon=str(row[2]),
                    direction=str(row[3]),
                    regime=str(row[4]),
                    trades=trades,
                    mean_r=mean_r,
                    modifier=round(modifier, 3),
                    specificity=int(row[7]),
                )
            )
        return sorted(estimates, key=lambda item: item.specificity, reverse=True)

    def selection_evidence(
        self,
        *,
        minimum_trades: int,
        shrinkage_trades: int,
        points_per_r: float,
        modifier_cap: float,
        outcome_floor_r: float,
        outcome_cap_r: float,
    ) -> list[SelectionEvidence]:
        """Independent realised facets for ranking already-valid candidates.

        Only broker-confirmed closed trades enter ``realised``.  Each UNION
        asks one reusable question instead of demanding a sparse exact match:
        how this setup family, horizon, detector, asset class, regime, session,
        score band or direction has actually performed on this account.

        ``pnl_r`` is clipped for this ranking estimate only.  The untouched P/L
        remains in the database and every report; clipping merely prevents one
        corrupt fill or singular outlier from teaching the selector more than
        the rest of its history combined.
        """
        rows = self._run(
            """
            WITH realised AS (
                SELECT
                    COALESCE(NULLIF(d.setup_family, ''), NULLIF(d.playbook, ''),
                             'unknown') AS setup_family,
                    COALESCE(d.horizon, 'unknown') AS horizon,
                    COALESCE(d.asset_class, 'unknown') AS asset_class,
                    COALESCE(d.regime, 'unknown') AS regime,
                    COALESCE(d.session, 'unknown') AS session,
                    CASE
                        WHEN d.conviction IS NULL THEN 'unknown'
                        ELSE (FLOOR(d.conviction / 10) * 10)::INT::TEXT
                    END AS score_band,
                    COALESCE(d.signals, '[]'::jsonb) AS signals,
                    t.direction,
                    GREATEST(%s, LEAST(%s, t.pnl_r)) AS bounded_r
                FROM trades t
                LEFT JOIN decisions d ON d.id = t.decision_id
                WHERE t.account = %s AND t.closed_at IS NOT NULL AND t.pnl_r IS NOT NULL
            ), facets AS (
                SELECT 'setup_horizon' AS dimension,
                       setup_family || '|' || horizon AS value,
                       direction, bounded_r
                FROM realised
                UNION ALL
                SELECT 'setup_family', setup_family, direction, bounded_r FROM realised
                UNION ALL
                SELECT 'horizon', horizon, direction, bounded_r FROM realised
                UNION ALL
                SELECT 'asset_class', asset_class, direction, bounded_r FROM realised
                UNION ALL
                SELECT 'regime', regime, direction, bounded_r FROM realised
                UNION ALL
                SELECT 'session', session, direction, bounded_r FROM realised
                UNION ALL
                SELECT 'score_band', score_band, direction, bounded_r FROM realised
                UNION ALL
                SELECT 'direction', '*', direction, bounded_r FROM realised
                UNION ALL
                SELECT 'detector', signal->>'module', realised.direction, bounded_r
                FROM realised
                CROSS JOIN LATERAL jsonb_array_elements(realised.signals) AS signal
                WHERE COALESCE(signal->>'module', '') <> ''
                  AND COALESCE(signal->>'direction', '') = realised.direction
            )
            SELECT dimension, value, direction, COUNT(*),
                   COUNT(*) FILTER (WHERE bounded_r > 0), AVG(bounded_r)
            FROM facets
            WHERE value <> 'unknown'
            GROUP BY dimension, value, direction
            HAVING COUNT(*) >= %s
            ORDER BY dimension, value, direction
            """,
            (
                outcome_floor_r,
                outcome_cap_r,
                self.account,
                minimum_trades,
            ),
            fetch="all",
        )
        if not rows:
            return []
        estimates: list[SelectionEvidence] = []
        for row in rows:
            trades = int(row[3])
            mean_r = float(row[5] or 0.0)
            shrink = trades / (trades + max(1, shrinkage_trades))
            raw = mean_r * shrink * points_per_r
            modifier = max(-modifier_cap, min(modifier_cap, raw))
            estimates.append(
                SelectionEvidence(
                    dimension=str(row[0]),
                    value=str(row[1]),
                    direction=str(row[2]),
                    trades=trades,
                    wins=int(row[4]),
                    mean_r=mean_r,
                    modifier=round(modifier, 3),
                )
            )
        return estimates

    def gate_scoreboard(self, *, symbol: str = "", limit: int = 12) -> list[GateScoreline]:
        """How refused executable plans resolved after the decision."""
        rows = self._run(
            """
            SELECT blocked_by, COUNT(*),
                   COUNT(*) FILTER (WHERE pnl_r > 0), AVG(pnl_r)
            FROM counterfactuals
            WHERE account = %s AND (%s = '' OR symbol = %s)
            GROUP BY blocked_by
            ORDER BY COUNT(*) DESC
            LIMIT %s
            """,
            (self.account, symbol, symbol, limit),
            fetch="all",
        )
        if not rows:
            return []
        return [
            GateScoreline(
                blocked_by=str(row[0]),
                observations=int(row[1]),
                hypothetical_wins=int(row[2]),
                mean_r=float(row[3] or 0.0),
            )
            for row in rows
        ]

    def learned_bank_threshold(self, *, minimum_trades: int = MIN_TRADES_TO_LEARN) -> float | None:
        """Where this account's own trades say profit should be taken, in R.

        THE ONE THING THE DATABASE IS ALLOWED TO MOVE, and the exception is
        narrow enough to state exactly. Everywhere else in this file the rule
        is absolute: nothing read from Postgres may change a risk limit, a
        threshold, a weight or a lot size. That rule exists to stop a learning
        system quietly making itself more dangerous.

        This one cannot. `PositionManager._worth_taking` takes the MINIMUM of
        the configured threshold and this, so the only move available is taking
        profit sooner. A position closed earlier is less exposure, never more.
        The worst case is money left on the table; the worst case of the rule
        it is constrained by is the account.

        WHAT IT MEASURES. The journal knows, for every closed trade, the best
        it ever reached and what it actually returned. The difference is what
        was handed back. Grouped by where the peak was, the question is the
        operator's own — at what point does taking it beat holding on — and
        the answer is the lowest peak band where banking would have beaten
        what the account really did.

        Lowest rather than largest, because the rule fires on the way up. A
        threshold only ever acts at the first level price crosses, so choosing
        the band with the biggest gap would be choosing a level most trades
        never reach.

        Conservative in three ways, each because the alternative is a
        threshold learned from one week's weather: it needs `minimum_trades`
        before it says anything, it is clamped so it can never demand less
        than the round trip is worth, and it returns None rather than a guess
        when the evidence is thin or the database is unreachable — which
        leaves the configured value in charge, exactly as before it existed.
        """
        rows = self._run(
            """
            SELECT
                width_bucket(mfe_r, 0.25, 2.0, 7) AS band,
                COUNT(*)      AS trades,
                AVG(mfe_r)    AS mean_peak,
                AVG(pnl_r)    AS mean_result
            FROM trade_history
            WHERE account = %s AND mfe_r > 0 AND pnl_r IS NOT NULL
            GROUP BY band
            HAVING COUNT(*) >= %s
            ORDER BY band
            """,
            (self.account, max(3, minimum_trades // 4)),
            fetch="all",
        )
        if not rows:
            return None
        if sum(int(row[1]) for row in rows) < minimum_trades:
            return None

        for _band, _trades, mean_peak, mean_result in rows:
            peak, result = float(mean_peak), float(mean_result)
            if peak > result:
                return max(MIN_LEARNED_BANK_R, round(peak, 2))
        return None

    def briefing(self, symbol: str = "", direction: str = "") -> dict[str, Any]:
        """Everything worth telling the reviewer before it judges a setup.

        Shaped to match `learning.memory.TradingMemory.briefing` so the two can
        be merged into one payload without the prompt having to know which of
        them answered.
        """
        if not self.enabled:
            return {}
        lessons = [item.summary() for item in self.lessons()]
        here = [line.summary() for line in self.scoreboard(symbol=symbol)]
        overall = [line.summary() for line in self.scoreboard()] if not symbol else []
        gates = [line.summary() for line in self.gate_scoreboard()]
        sides = self.side_records()
        detectors = self.module_records()
        management = self.management_records()
        brief: dict[str, Any] = {}
        if lessons:
            brief["lessons_from_past_trades"] = lessons
        # Ahead of the per-instrument lines, because it is the one figure here
        # drawn from the whole record rather than from a handful of trades, and
        # a payload is read in the order it is written.
        if sides:
            brief["how_each_side_has_actually_done"] = {
                "records": [record.summary() for record in sides],
                "weight": (
                    "Realised broker fills on this account, every instrument pooled. "
                    "The largest sample in this briefing and still small. Treat a losing "
                    "side as a reason to require more from the setup in front of you — "
                    "a clearer structure, a better location — and not as a standing "
                    "veto on that direction."
                ),
            }
        if detectors:
            brief["what_each_detector_has_earned"] = {
                "records": [record.summary() for record in detectors],
                "weight": (
                    "Realised R attributed to every module that scored on the trade. "
                    "The engine that produced the proposal in front of you is named in "
                    "`modules`; if it is one of the losing detectors here, that is a "
                    "reason to want more from the chart than usual."
                ),
            }
        if management:
            brief["what_stepping_in_has_earned"] = {
                "records": [record.summary() for record in management],
                "weight": (
                    "Every closed trade on this account replayed against its own "
                    "untouched stop and target. Positive means closing early beat "
                    "leaving it alone; negative means the rule is an expensive habit. "
                    "This is the only evidence here about MANAGEMENT rather than "
                    "entries, and it is the question you are being asked: a rule with "
                    "a negative record is a reason to hold a working trade rather than "
                    "bank it, and one with a positive record is a reason to act while "
                    "the money is still there."
                ),
            }
        if here:
            brief["this_instrument_so_far"] = here
        if overall:
            brief["account_scoreboard"] = overall
        if gates:
            brief["refusal_outcomes"] = gates
        # Only once there is something to attach it to. Returned on its own it
        # is a payload key echoing back what the reviewer just asked about,
        # which is tokens spent to tell a model something it already knows —
        # and worse, it makes an empty memory look like a populated one.
        if brief and direction:
            brief["direction_under_review"] = direction
        return brief


@dataclass
class NullBrain:
    """What everything gets when no DSN is configured.

    A real object rather than `None` so no caller needs an `if self.brain is
    not None` around every write. Every method is a no-op that answers the way
    an empty database would.
    """

    status: BrainStatus = field(
        default_factory=lambda: BrainStatus(connected=False, dsn_configured=False)
    )
    enabled: bool = False
    account: str = ""

    def migrate(self) -> bool:
        return False

    def close(self) -> None:
        return None

    def record_decision(self, **_: Any) -> int | None:
        return None

    def record_trade_opened(self, **_: Any) -> int | None:
        return None

    def record_trade_event(self, **_: Any) -> None:
        return None

    def record_position_path(self, _rows: Sequence[Mapping[str, Any]] = ()) -> int:
        return 0

    def record_management_outcome(self, **_: Any) -> None:
        return None

    def management_records(self, *_: Any, **__: Any) -> list[ManagementRecord]:
        return []

    def supervision_examples(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []

    def entry_examples(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []

    def best_exit_available(self, *_: Any, **__: Any) -> tuple[float, float] | None:
        return None

    def record_trade_closed(self, **_: Any) -> None:
        return None

    def record_counterfactual(self, **_: Any) -> None:
        return None

    def record_counterfactuals(self, _rows: Sequence[Mapping[str, Any]] = ()) -> None:
        return None

    def record_trade_history(self, _rows: Sequence[Mapping[str, Any]] = ()) -> int:
        return 0

    def record_lessons(self, _lessons: Sequence[str] = (), **_kwargs: Any) -> None:
        return None

    def record_supervision(self, **_: Any) -> None:
        return None

    def record_headlines(self, _items: Sequence[Any] = ()) -> int:
        return 0

    def learned_bank_threshold(self, **_: Any) -> float | None:
        return None

    def lessons(self, **_: Any) -> list[Lesson]:
        return []

    def scoreboard(self, **_: Any) -> list[Scoreline]:
        return []

    def side_records(self, **_: Any) -> list[SideRecord]:
        return []

    def module_records(self, **_: Any) -> list[ModuleRecord]:
        return []

    def edge_calibrations(self, **_: Any) -> list[EdgeCalibration]:
        return []

    def selection_evidence(self, **_: Any) -> list[SelectionEvidence]:
        return []

    def gate_scoreboard(self, **_: Any) -> list[GateScoreline]:
        return []

    def briefing(self, symbol: str = "", direction: str = "") -> dict[str, Any]:
        del symbol, direction
        return {}


def build_brain(account: str = "", *, enabled: bool = True) -> Brain | NullBrain:
    """A `Brain` when a DSN is configured and psycopg is installed, else null.

    Both fallbacks are silent by design. The DSN is optional on a developer
    machine and psycopg is an optional dependency, and neither absence is a
    reason for the trading system to refuse to start.
    """
    dsn = os.getenv(DSN_ENV, "")
    if not enabled or not dsn:
        return NullBrain()
    try:
        import psycopg  # noqa: F401
    except ImportError:
        log.warning(
            "NEON_DATABASE_URL is set but psycopg is not installed; running without the brain",
            extra={"event": "brain_driver_missing"},
        )
        return NullBrain()
    return Brain(dsn, account=account)


def utcnow() -> datetime:
    return datetime.now(UTC)

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
                ai_tokens, headlines
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s::jsonb
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
        """
        self._run(
            """
            INSERT INTO supervisions (
                trade_id, account, asked_at, symbol, action, confidence,
                reasoning, r_at_the_time, applied, latency_ms, model
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        brief: dict[str, Any] = {}
        if lessons:
            brief["lessons_from_past_trades"] = lessons
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

    def edge_calibrations(self, **_: Any) -> list[EdgeCalibration]:
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

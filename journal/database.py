"""SQLite journal: schema, connection, and the queries the risk layer needs.

This exists from day one rather than being bolted on later, because the two
things that decide whether this project ever works — "does the edge survive
out-of-sample" and "did the system do what it thought it did" — are both
answered from recorded data, and data you did not record in March cannot be
recovered in September.

Design notes:

* **Every analysis cycle is stored, including the ones that produce no trade.**
  The skipped setups are where filter-effectiveness analysis lives: if blocked
  setups were systematically profitable, a filter is too strict, and only the
  non-trade rows can show that.
* **Timestamps are ISO-8601 UTC strings.** Sortable lexicographically, readable
  in the `sqlite3` CLI without conversion, and immune to the epoch/local-time
  confusion that quietly corrupts trading databases.
* **Money and R are separate columns.** R is the unit that stays comparable as
  the account grows; money is what actually happened. Reports need both.
* **Schema changes go through `_MIGRATIONS`,** append-only. A journal you
  cannot read back is worse than no journal.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from core.clock import Clock
from infra.logging import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 4


_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        # One row per analysis cycle, trade or no trade.
        """
        CREATE TABLE analysis_cycles (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_id        TEXT    NOT NULL,
            ts              TEXT    NOT NULL,
            symbol          TEXT    NOT NULL,
            mode            TEXT    NOT NULL,
            decision        TEXT    NOT NULL,   -- TRADE | SKIP
            reason          TEXT    NOT NULL,   -- risk.reasons.Reason
            detail          TEXT    NOT NULL DEFAULT '',
            direction       TEXT,               -- LONG | SHORT | NULL
            total_score     REAL,
            score_threshold REAL,
            equity          REAL    NOT NULL,
            atr             REAL,
            spread_pips     REAL,
            session         TEXT,
            volatility_regime TEXT,
            minutes_to_news REAL,
            context_json    TEXT    NOT NULL DEFAULT '{}'
        )
        """,
        "CREATE INDEX idx_cycles_ts ON analysis_cycles(ts)",
        "CREATE INDEX idx_cycles_symbol_ts ON analysis_cycles(symbol, ts)",
        "CREATE INDEX idx_cycles_reason ON analysis_cycles(reason)",
        # Per-module contribution to that cycle's score.
        """
        CREATE TABLE module_scores (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_pk    INTEGER NOT NULL REFERENCES analysis_cycles(id) ON DELETE CASCADE,
            module      TEXT    NOT NULL,
            score       REAL    NOT NULL,
            confidence  REAL    NOT NULL,
            weight      REAL    NOT NULL,
            reasoning   TEXT    NOT NULL DEFAULT '',
            details_json TEXT   NOT NULL DEFAULT '{}'
        )
        """,
        "CREATE INDEX idx_module_scores_cycle ON module_scores(cycle_pk)",
        "CREATE INDEX idx_module_scores_module ON module_scores(module)",
        # Trades actually taken.
        """
        CREATE TABLE trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_pk        INTEGER REFERENCES analysis_cycles(id),
            ticket          INTEGER,
            magic           INTEGER NOT NULL DEFAULT 0,
            symbol          TEXT    NOT NULL,
            direction       TEXT    NOT NULL,
            volume          REAL    NOT NULL,
            entry_price     REAL    NOT NULL,
            sl              REAL    NOT NULL,
            tp              REAL    NOT NULL DEFAULT 0,
            risk_money      REAL    NOT NULL,
            risk_pct        REAL    NOT NULL,
            sl_distance_pips REAL   NOT NULL,
            planned_rr      REAL    NOT NULL DEFAULT 0,
            opened_at       TEXT    NOT NULL,
            closed_at       TEXT,
            exit_price      REAL,
            exit_reason     TEXT,
            pnl_money       REAL,
            pnl_r           REAL,
            mae_r           REAL,
            mfe_r           REAL,
            duration_seconds INTEGER,
            equity_before   REAL    NOT NULL,
            equity_after    REAL
        )
        """,
        "CREATE INDEX idx_trades_opened ON trades(opened_at)",
        "CREATE INDEX idx_trades_closed ON trades(closed_at)",
        "CREATE UNIQUE INDEX idx_trades_ticket ON trades(ticket) WHERE ticket IS NOT NULL",
        # Raw execution telemetry — the input to EXECUTION_REPORT.md.
        """
        CREATE TABLE order_attempts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id        INTEGER REFERENCES trades(id) ON DELETE CASCADE,
            ts              TEXT    NOT NULL,
            kind            TEXT    NOT NULL,   -- ENTRY | EXIT | MODIFY
            symbol          TEXT    NOT NULL,
            ok              INTEGER NOT NULL,
            retcode         INTEGER,
            retcode_name    TEXT    NOT NULL,
            broker_comment  TEXT    NOT NULL DEFAULT '',
            requested_price REAL    NOT NULL,
            filled_price    REAL    NOT NULL,
            slippage_pips   REAL    NOT NULL,
            requested_volume REAL   NOT NULL,
            filled_volume   REAL    NOT NULL,
            latency_ms      REAL    NOT NULL,
            spread_at_send  REAL    NOT NULL,
            attempts        INTEGER NOT NULL
        )
        """,
        "CREATE INDEX idx_attempts_trade ON order_attempts(trade_id)",
        "CREATE INDEX idx_attempts_retcode ON order_attempts(retcode)",
        # Break-even moves, partials, trailing steps, time and news exits.
        """
        CREATE TABLE management_actions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id    INTEGER NOT NULL REFERENCES trades(id) ON DELETE CASCADE,
            ts          TEXT    NOT NULL,
            action      TEXT    NOT NULL,
            old_sl      REAL,
            new_sl      REAL,
            old_tp      REAL,
            new_tp      REAL,
            volume_closed REAL,
            r_at_action REAL,
            note        TEXT NOT NULL DEFAULT ''
        )
        """,
        "CREATE INDEX idx_actions_trade ON management_actions(trade_id)",
        # What would have happened to setups the filters blocked.
        """
        CREATE TABLE shadow_trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_pk    INTEGER NOT NULL REFERENCES analysis_cycles(id) ON DELETE CASCADE,
            symbol      TEXT    NOT NULL,
            direction   TEXT    NOT NULL,
            blocked_by  TEXT    NOT NULL,
            entry_price REAL    NOT NULL,
            sl          REAL    NOT NULL,
            tp          REAL    NOT NULL,
            opened_at   TEXT    NOT NULL,
            resolved_at TEXT,
            outcome     TEXT,               -- TP | SL | TIMEOUT | UNRESOLVED
            pnl_r       REAL
        )
        """,
        "CREATE INDEX idx_shadow_cycle ON shadow_trades(cycle_pk)",
        "CREATE INDEX idx_shadow_unresolved ON shadow_trades(outcome) WHERE outcome IS NULL",
        # OHLCV around an entry, so a decision can be replayed later.
        """
        CREATE TABLE bar_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_pk    INTEGER NOT NULL REFERENCES analysis_cycles(id) ON DELETE CASCADE,
            symbol      TEXT    NOT NULL,
            timeframe   TEXT    NOT NULL,
            bars_json   TEXT    NOT NULL
        )
        """,
        "CREATE INDEX idx_snapshots_cycle ON bar_snapshots(cycle_pk)",
        # Equity anchors for the daily/weekly loss limits and the drawdown
        # breaker. Persisted so a restart cannot reset a loss limit.
        """
        CREATE TABLE equity_marks (
            period      TEXT    NOT NULL,   -- DAY | WEEK | PEAK
            period_key  TEXT    NOT NULL,   -- ISO timestamp of the boundary
            equity      REAL    NOT NULL,
            recorded_at TEXT    NOT NULL,
            PRIMARY KEY (period, period_key)
        )
        """,
        # Config in force, so a result can be tied to the parameters that made it.
        """
        CREATE TABLE config_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            config_hash TEXT    NOT NULL,
            config_json TEXT    NOT NULL
        )
        """,
        "CREATE INDEX idx_config_hash ON config_snapshots(config_hash)",
    ),
    2: (
        # Observed spreads, bucketed by hour of day. The spread filter learns
        # its own baseline from these rather than trusting a static threshold:
        # EURUSD at 09:00 and EURUSD at 22:00 are different instruments as far
        # as cost of entry is concerned.
        """
        CREATE TABLE spread_observations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT    NOT NULL,
            hour_utc    INTEGER NOT NULL,
            spread_pips REAL    NOT NULL,
            ts          TEXT    NOT NULL
        )
        """,
        "CREATE INDEX idx_spread_symbol_hour ON spread_observations(symbol, hour_utc)",
        "CREATE INDEX idx_spread_ts ON spread_observations(ts)",
    ),
    3: (
        # Entry intent, written *before* the order is sent.
        #
        # There was a window between `order_send` returning a ticket and the
        # journal row being inserted. A crash inside it left a real position at
        # the broker the journal had never heard of, and on restart
        # reconciliation could only read that as an orphan and close it — a
        # correctly sized, AI-approved trade destroyed by a power cut, with the
        # loss booked and no record of why the position had ever existed.
        #
        # `entry_state` closes it. The row is written as PENDING before the
        # order goes out and promoted to OPEN once the ticket is known, so a
        # position found without a journal entry can be matched back to the
        # intent that created it instead of being liquidated.
        #
        # Existing rows default to OPEN: every trade already in the journal got
        # there by completing the old path, so OPEN is the truthful value.
        "ALTER TABLE trades ADD COLUMN entry_state TEXT NOT NULL DEFAULT 'OPEN'",
        "CREATE INDEX idx_trades_entry_state ON trades(entry_state)",
    ),
    4: (
        # Collapse the equity peak to a single row.
        #
        # It was keyed by the timestamp of each write, so every cycle inserted
        # another row and `equity_peak()` took the maximum over all of them.
        # That works, and it grows without bound: at a thirty-second loop it is
        # roughly a million rows a year in a file the dashboard reads
        # constantly.
        #
        # It also made the docstring's "monotonic by construction" untrue. The
        # peak was monotonic only because the clock kept advancing and produced
        # a fresh key each time; two writes landing on one key would REPLACE,
        # and the peak could fall. A frozen clock does exactly that, which is
        # how a test caught a 20% drawdown failing to trip the breaker.
        #
        # One row, updated only upward, is what the invariant actually needs.
        # OR REPLACE, and MAX taken over every PEAK row including any existing
        # 'all-time': the collapse can then only preserve or raise the peak, and
        # re-running it is harmless. A plain INSERT would fail outright the
        # moment a row under that key already exists.
        """
        INSERT OR REPLACE INTO equity_marks (period, period_key, equity, recorded_at)
        SELECT 'PEAK', 'all-time', MAX(equity), MAX(recorded_at)
        FROM equity_marks WHERE period = 'PEAK'
        HAVING COUNT(*) > 0
        """,
        "DELETE FROM equity_marks WHERE period = 'PEAK' AND period_key != 'all-time'",
    ),
}

#: The single row the all-time equity peak lives in.
PEAK_KEY = "all-time"


class Journal:
    """Owns the SQLite connection, the schema, and read queries.

    Writes live in `journal/recorder.py`. The split is deliberate: reads are
    used by the risk layer on every cycle and must stay cheap and obvious,
    while writes have to capture a lot of detail and are noisier.
    """

    def __init__(self, path: Path | str, clock: Clock, *, day_boundary_utc: str = "21:00") -> None:
        """
        Args:
            day_boundary_utc: when a trading day rolls over, in UTC. Defaults to
                21:00, which is the FX rollover — using midnight UTC would put
                the daily loss limit's reset in the middle of the New York
                session.
        """
        self.path = Path(path)
        self.clock = clock
        self.day_boundary = _parse_hhmm(day_boundary_utc)
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> Journal:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL keeps reads from blocking the writer, which matters because the
        # risk layer reads on every cycle while the recorder is writing.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        self._conn = conn
        self._migrate()
        return self

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Journal:
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("journal is not open; call open() or use it as a context manager")
        return self._conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Explicit transaction. Rolls back on any exception."""
        conn = self.conn
        conn.execute("BEGIN")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def _migrate(self) -> None:
        conn = self.conn
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        current = row["v"] or 0

        for version in sorted(_MIGRATIONS):
            if version <= current:
                continue
            with self.transaction() as tx:
                for statement in _MIGRATIONS[version]:
                    tx.execute(statement)
                tx.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            log.info(
                "journal migrated",
                extra={"event": "journal_migration", "version": version, "path": str(self.path)},
            )

        if current > SCHEMA_VERSION:
            raise RuntimeError(
                f"journal at {self.path} is schema v{current}, but this code understands "
                f"v{SCHEMA_VERSION}. Refusing to run against a newer journal."
            )

    # -- trading-period boundaries ----------------------------------------

    def day_start(self, moment: datetime | None = None) -> datetime:
        """Start of the trading day containing `moment`."""
        moment = (moment or self.clock.now()).astimezone(UTC)
        hour, minute = self.day_boundary
        candidate = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate > moment:
            candidate -= timedelta(days=1)
        return candidate

    def week_start(self, moment: datetime | None = None) -> datetime:
        """Start of the trading week containing `moment`.

        The FX week opens Sunday evening, so the week boundary is the Sunday
        day-boundary — not Monday 00:00, which would split the Sunday-evening
        session away from the week it belongs to.
        """
        day = self.day_start(moment)
        days_since_sunday = (day.weekday() + 1) % 7  # Monday=0 -> Sunday=0
        return day - timedelta(days=days_since_sunday)

    # -- reads used by the risk layer --------------------------------------

    def trades_since(self, since: datetime) -> int:
        """Trades opened at or after `since`.

        Counted on open, not on close: three trades opened today is three
        trades today, whatever happens to them tomorrow.
        """
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM trades WHERE opened_at >= ?", (iso(since),)
        ).fetchone()
        return int(row["n"])

    def realised_pnl_since(self, since: datetime) -> float:
        """Realised money P/L from trades closed at or after `since`."""
        row = self.conn.execute(
            "SELECT COALESCE(SUM(pnl_money), 0.0) AS pnl FROM trades WHERE closed_at >= ?",
            (iso(since),),
        ).fetchone()
        return float(row["pnl"])

    def consecutive_losses(self) -> int:
        """Length of the current losing streak, most recent closed trade first.

        Break-even trades (exactly 0.0) neither extend nor reset the streak;
        they are not losses, and treating them as wins would let a flat trade
        restore full risk after two real losses.
        """
        rows = self.conn.execute(
            "SELECT pnl_money FROM trades WHERE closed_at IS NOT NULL "
            "ORDER BY closed_at DESC LIMIT 50"
        ).fetchall()
        streak = 0
        for row in rows:
            pnl = float(row["pnl_money"] or 0.0)
            if pnl < 0:
                streak += 1
            elif pnl > 0:
                break
        return streak

    def last_trade_risk_pct(self) -> float | None:
        """Risk percentage of the most recently opened trade.

        Used to catch a strategy trying to increase risk after a loss.
        """
        row = self.conn.execute(
            "SELECT risk_pct FROM trades ORDER BY opened_at DESC, id DESC LIMIT 1"
        ).fetchone()
        return None if row is None else float(row["risk_pct"])

    def open_trade_by_ticket(self, ticket: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM trades WHERE ticket = ? AND closed_at IS NULL", (ticket,)
        ).fetchone()

    def open_trades(self) -> list[sqlite3.Row]:
        """Trades believed to be live at the broker.

        Excludes PENDING rows deliberately. A pending row is an intent whose
        order may never have reached the broker, and reconciliation reads this
        list as "the journal says these should exist" — a never-sent intent in
        there would be reported as a position the broker has lost, which halts
        new risk on a phantom.
        """
        return list(
            self.conn.execute(
                "SELECT * FROM trades WHERE closed_at IS NULL AND entry_state = 'OPEN' "
                "ORDER BY opened_at"
            ).fetchall()
        )

    def pending_entries(self, since: datetime | None = None) -> list[sqlite3.Row]:
        """Intents written before an order was sent, oldest first."""
        if since is None:
            return list(
                self.conn.execute(
                    "SELECT * FROM trades WHERE entry_state = 'PENDING' AND closed_at IS NULL "
                    "ORDER BY opened_at"
                ).fetchall()
            )
        return list(
            self.conn.execute(
                "SELECT * FROM trades WHERE entry_state = 'PENDING' AND closed_at IS NULL "
                "AND opened_at >= ? ORDER BY opened_at",
                (iso(since),),
            ).fetchall()
        )

    def claim_pending_entry(
        self,
        *,
        symbol: str,
        direction: str,
        volume: float,
        ticket: int,
        entry_price: float,
        volume_tolerance: float,
        since: datetime | None = None,
        opened_at: datetime | None = None,
    ) -> int | None:
        """Bind an unexplained broker position to the intent that created it.

        Matched on what the broker can actually confirm — symbol, direction and
        volume — because the ticket is precisely the thing the crash lost. The
        entry price is not matched on: the fill differs from the intent by
        slippage, which is the normal case rather than a mismatch.

        `since` bounds how far back an intent may be claimed. Without it a
        pending row from last week could adopt a position opened by hand this
        morning, which would attach real money to the wrong plan.

        Returns the trade id on success, None when nothing matches — and the
        caller must treat None as "still an orphan", not as "adopted".
        """
        rows = self.pending_entries(since)
        for row in rows:
            if str(row["symbol"]) != symbol or str(row["direction"]) != direction:
                continue
            if abs(float(row["volume"]) - volume) > volume_tolerance:
                continue
            trade_id = int(row["id"])
            self._mark_open(trade_id, ticket=ticket, entry_price=entry_price, opened_at=opened_at)
            return trade_id
        return None

    def promote_pending_entry(
        self,
        trade_id: int,
        *,
        ticket: int,
        entry_price: float,
        opened_at: datetime | None = None,
    ) -> None:
        """Mark an intent as a live trade once the broker has confirmed it.

        The fill price is preferred and the intent price is the fallback. It
        used to be an unconditional overwrite, and a live EURGBP entry came
        back from the broker with no usable fill price — so a perfectly good
        0.857xx recorded at sizing time was replaced with 0.00000, and every
        number the postmortem derives from entry became nonsense.

        Nothing downstream noticed, because the manager works from the
        broker's own `price_open` rather than this column. The trade was
        managed correctly and only its record was wrong, which is the kind of
        corruption that survives for months.
        """
        self._mark_open(trade_id, ticket=ticket, entry_price=entry_price, opened_at=opened_at)

    def _mark_open(
        self,
        trade_id: int,
        *,
        ticket: int,
        entry_price: float,
        opened_at: datetime | None,
    ) -> None:
        """Attach a broker ticket to an intent, preserving a good entry price.

        Shared by both routes into the OPEN state — the ordinary promotion and
        the adoption of an unexplained broker position after a restart. They
        had the same unconditional overwrite written out twice, which is how
        fixing one of them and not the other has already gone wrong once in
        this codebase.
        """
        if entry_price > 0:
            self.conn.execute(
                "UPDATE trades SET ticket = ?, entry_price = ?, entry_state = 'OPEN', "
                "opened_at = ? WHERE id = ?",
                (ticket, entry_price, iso(opened_at or self.clock.now()), trade_id),
            )
        else:
            log.warning(
                "broker confirmed the entry without a usable fill price; "
                "keeping the price recorded at sizing time",
                extra={
                    "event": "entry_price_missing",
                    "trade_id": trade_id,
                    "ticket": ticket,
                    "reported": entry_price,
                },
            )
            self.conn.execute(
                "UPDATE trades SET ticket = ?, entry_state = 'OPEN', opened_at = ? WHERE id = ?",
                (ticket, iso(opened_at or self.clock.now()), trade_id),
            )
        self.conn.commit()

    def abandon_pending_entry(self, trade_id: int, reason: str) -> None:
        """Retire an intent whose order never became a position.

        Closed rather than deleted. A rejected entry is evidence — a run of them
        is how a broker-side problem becomes visible — and deleting the row
        would also orphan the `order_attempts` record that explains the refusal.
        `pnl_money` is zeroed rather than left NULL so it never reads as an
        unresolved trade in the reporting queries.
        """
        self.conn.execute(
            "UPDATE trades SET entry_state = 'ABANDONED', closed_at = ?, exit_reason = ?, "
            "pnl_money = 0.0, pnl_r = 0.0 WHERE id = ?",
            (iso(self.clock.now()), reason[:200], trade_id),
        )
        self.conn.commit()

    def management_action_exists(self, ticket: int, actions: tuple[str, ...]) -> bool:
        if not actions:
            return False
        placeholders = ",".join("?" for _ in actions)
        row = self.conn.execute(
            "SELECT 1 FROM management_actions a JOIN trades t ON t.id = a.trade_id "
            f"WHERE t.ticket = ? AND a.action IN ({placeholders}) LIMIT 1",
            (ticket, *actions),
        ).fetchone()
        return row is not None

    def update_excursions(self, trade_id: int, *, mae_r: float, mfe_r: float) -> None:
        """Ratchet MAE/MFE while a trade is open.

        Only ever moves outward (`MIN`/`MAX`), so a late retrace cannot erase
        the fact that the trade was once 2R in profit — which is exactly what
        the give-back rule needs to remember, and what a postmortem needs to
        tell a trade that was never right from one that was right and given
        back.
        """
        self.conn.execute(
            """
            UPDATE trades SET
                mae_r = MIN(COALESCE(mae_r, 0.0), ?),
                mfe_r = MAX(COALESCE(mfe_r, 0.0), ?)
            WHERE id = ?
            """,
            (mae_r, mfe_r, trade_id),
        )

    def update_open_trade_volume(self, ticket: int, volume: float) -> None:
        self.conn.execute(
            "UPDATE trades SET volume = ? WHERE ticket = ? AND closed_at IS NULL",
            (volume, ticket),
        )

    def equity_mark(self, period: str, period_key: datetime) -> float | None:
        row = self.conn.execute(
            "SELECT equity FROM equity_marks WHERE period = ? AND period_key = ?",
            (period, iso(period_key)),
        ).fetchone()
        return None if row is None else float(row["equity"])

    def set_equity_mark(self, period: str, period_key: datetime, equity: float) -> None:
        """Record a boundary equity, once. Later writes for the same key are ignored.

        `INSERT OR IGNORE` is the whole point: if the system restarts at 14:00
        after losing 2% since the 21:00 boundary, re-anchoring to the current
        equity would erase that loss and hand back a fresh daily budget.
        """
        self.conn.execute(
            "INSERT OR IGNORE INTO equity_marks (period, period_key, equity, recorded_at) "
            "VALUES (?, ?, ?, ?)",
            (period, iso(period_key), equity, iso(self.clock.now())),
        )

    def equity_peak(self) -> float | None:
        row = self.conn.execute(
            "SELECT MAX(equity) AS peak FROM equity_marks WHERE period = 'PEAK'"
        ).fetchone()
        return None if row is None or row["peak"] is None else float(row["peak"])

    def record_equity_peak(self, equity: float) -> float:
        """Ratchet the all-time equity peak upward and return the current value.

        Monotonic in the row itself, not merely in a MAX over many rows. The
        earlier version keyed each write by its own timestamp, so the peak held
        only because the clock kept advancing — two writes on one key REPLACEd,
        and the reference could fall, quietly making the circuit breaker harder
        to trip. It also added a row every cycle, forever.

        A losing day must never lower the drawdown reference. `MAX` in the
        upsert is what guarantees that, whatever the clock does.
        """
        now = iso(self.clock.now())
        self.conn.execute(
            "INSERT INTO equity_marks (period, period_key, equity, recorded_at) "
            "VALUES ('PEAK', ?, ?, ?) "
            "ON CONFLICT(period, period_key) DO UPDATE SET "
            "equity = MAX(equity, excluded.equity), recorded_at = excluded.recorded_at",
            (PEAK_KEY, equity, now),
        )
        peak = self.equity_peak()
        return equity if peak is None else max(peak, equity)

    def has_open_position_in(self, symbol: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM trades WHERE symbol = ? AND closed_at IS NULL LIMIT 1", (symbol,)
        ).fetchone()
        return row is not None

    # -- spread baseline ---------------------------------------------------

    def record_spread(self, symbol: str, spread_pips: float, moment: datetime) -> None:
        """Store one spread observation for the hour it fell in."""
        self.conn.execute(
            "INSERT INTO spread_observations (symbol, hour_utc, spread_pips, ts) VALUES (?,?,?,?)",
            (symbol, moment.astimezone(UTC).hour, spread_pips, iso(moment)),
        )

    def spread_samples(
        self,
        symbol: str,
        hour_utc: int,
        limit: int = 2000,
        *,
        weekend: bool | None = None,
    ) -> list[float]:
        """Recent hourly samples, optionally split into weekday/weekend regimes."""
        regime_sql = ""
        if weekend is True:
            regime_sql = " AND CAST(strftime('%w', ts) AS INTEGER) IN (0, 6)"
        elif weekend is False:
            regime_sql = " AND CAST(strftime('%w', ts) AS INTEGER) NOT IN (0, 6)"
        rows = self.conn.execute(
            "SELECT spread_pips FROM spread_observations WHERE symbol = ? AND hour_utc = ?"
            f"{regime_sql} ORDER BY ts DESC LIMIT ?",
            (symbol, hour_utc, limit),
        ).fetchall()
        return [float(row[0]) for row in rows]

    def last_spread_observation(self, symbol: str) -> datetime | None:
        row = self.conn.execute(
            "SELECT MAX(ts) AS ts FROM spread_observations WHERE symbol = ?", (symbol,)
        ).fetchone()
        return None if row is None or row["ts"] is None else parse_iso(row["ts"])

    def prune_spread_observations(self, before: datetime) -> int:
        """Drop observations older than `before`. Returns the number removed."""
        cursor = self.conn.execute("DELETE FROM spread_observations WHERE ts < ?", (iso(before),))
        return cursor.rowcount or 0

    # -- generic helpers ---------------------------------------------------

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, params).fetchall())

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.conn.execute(sql, params).fetchone()
        return None if row is None else row[0]


# ------------------------------------------------------------------ utils ---


def iso(moment: datetime) -> str:
    """Serialise a tz-aware datetime as a sortable UTC ISO-8601 string."""
    if moment.tzinfo is None:
        raise ValueError("refusing to store a naive datetime in the journal")
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def parse_iso(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(UTC)


def dumps(payload: Any) -> str:
    """JSON for a TEXT column, never raising on an odd value."""
    return json.dumps(payload, default=str, ensure_ascii=False)


def _parse_hhmm(text: str) -> tuple[int, int]:
    try:
        hour, minute = (int(part) for part in text.split(":", 1))
    except ValueError as exc:
        raise ValueError(f"expected HH:MM, got {text!r}") from exc
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"{text!r} is not a valid time of day")
    return hour, minute

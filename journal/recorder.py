"""Writing to the journal.

Everything the system decides passes through here. The rule is that a cycle is
recorded *before* an order is sent, so a crash between the decision and the
fill still leaves the reasoning on disk — a trade in MT5 with no matching
journal row is a reconciliation failure, and reconciliation only works if the
journal is written first.

MAE and MFE are tracked in R rather than money because that is the unit that
stays comparable as the account grows. They drive the two questions that
actually improve a system:

* MFE — are we taking profit too early? How far did winners run before turning?
* MAE — are we stopped out just before being right? How deep did eventual
  winners go against us?
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from config.schema import Settings
from core.clock import Clock
from core.types import Direction, OrderResult, Signal
from infra.logging import get_logger
from journal.database import Journal, dumps, iso
from risk.position_sizer import SizingResult
from risk.reasons import Reason

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CycleContext:
    """Market conditions at the moment of a decision.

    Every field here answers a question the postmortem module will ask —
    "do we lose more in high-ATR regimes", "are Asian-session setups worse",
    "how close to news were the losers". Collecting them is cheap; adding them
    retroactively is impossible.
    """

    symbol: str
    equity: float
    atr: float | None = None
    spread_pips: float | None = None
    session: str | None = None
    volatility_regime: str | None = None
    minutes_to_news: float | None = None
    extra: dict[str, Any] | None = None


class Recorder:
    """Writes cycles, trades, executions and management actions."""

    def __init__(self, journal: Journal, clock: Clock, settings: Settings) -> None:
        self.journal = journal
        self.clock = clock
        self.settings = settings

    # -- cycles ------------------------------------------------------------

    def record_cycle(
        self,
        *,
        cycle_id: str,
        context: CycleContext,
        reason: Reason,
        detail: str = "",
        traded: bool = False,
        direction: Direction | None = None,
        total_score: float | None = None,
        score_threshold: float | None = None,
        signals: list[Signal] | None = None,
        weights: dict[str, float] | None = None,
    ) -> int:
        """Record one analysis cycle and return its primary key.

        Called for **every** cycle, including the overwhelming majority that
        produce no trade. Those rows are not noise: they are the denominator
        for "how often does this setup appear" and the raw material for
        filter-effectiveness analysis.
        """
        now = self.clock.now()
        cursor = self.journal.conn.execute(
            """
            INSERT INTO analysis_cycles (
                cycle_id, ts, symbol, mode, decision, reason, detail, direction,
                total_score, score_threshold, equity, atr, spread_pips, session,
                volatility_regime, minutes_to_news, context_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                cycle_id,
                iso(now),
                context.symbol,
                self.settings.mode.value,
                "TRADE" if traded else "SKIP",
                str(reason),
                detail,
                direction.name if direction is not None else None,
                total_score,
                score_threshold,
                context.equity,
                context.atr,
                context.spread_pips,
                context.session,
                context.volatility_regime,
                context.minutes_to_news,
                dumps(context.extra or {}),
            ),
        )
        cycle_pk = int(cursor.lastrowid or 0)

        if signals:
            weights = weights or {}
            self.journal.conn.executemany(
                """
                INSERT INTO module_scores
                    (cycle_pk, module, score, confidence, weight, reasoning, details_json)
                VALUES (?,?,?,?,?,?,?)
                """,
                [
                    (
                        cycle_pk,
                        signal.module,
                        signal.score,
                        signal.confidence,
                        weights.get(signal.module, 0.0),
                        signal.reasoning,
                        dumps(signal.details),
                    )
                    for signal in signals
                ],
            )

        log.debug(
            "cycle recorded",
            extra={
                "event": "journal_cycle",
                "cycle_pk": cycle_pk,
                "symbol": context.symbol,
                "decision": "TRADE" if traded else "SKIP",
                "reason": str(reason),
            },
        )
        return cycle_pk

    def record_sizing(self, cycle_pk: int, sizing: SizingResult) -> None:
        """Attach the sizing arithmetic to a cycle.

        Stored in the cycle's `context_json` rather than its own table: sizing
        is one dict per cycle, and a join for it would buy nothing.
        """
        row = self.journal.conn.execute(
            "SELECT context_json FROM analysis_cycles WHERE id = ?", (cycle_pk,)
        ).fetchone()
        if row is None:  # pragma: no cover - caller error
            raise KeyError(f"no analysis cycle with id {cycle_pk}")

        context = json.loads(row["context_json"] or "{}")
        context["sizing"] = sizing.journal_row()
        self.journal.conn.execute(
            "UPDATE analysis_cycles SET context_json = ? WHERE id = ?",
            (dumps(context), cycle_pk),
        )

    def record_bar_snapshot(
        self, cycle_pk: int, symbol: str, timeframe: str, bars: list[dict[str, Any]]
    ) -> None:
        """Store OHLCV around a decision so it can be replayed later."""
        self.journal.conn.execute(
            "INSERT INTO bar_snapshots (cycle_pk, symbol, timeframe, bars_json) VALUES (?,?,?,?)",
            (cycle_pk, symbol, timeframe, dumps(bars)),
        )

    # -- trades ------------------------------------------------------------

    def record_trade_open(
        self,
        *,
        cycle_pk: int | None,
        sizing: SizingResult,
        ticket: int | None,
        entry_price: float,
        equity_before: float,
        opened_at: datetime | None = None,
        entry_state: str = "OPEN",
    ) -> int:
        """Record an opened trade and return its id.

        `entry_price` is the **filled** price, not the requested one — the
        difference is slippage and is recorded separately on the order attempt.
        Using the requested price here would quietly flatter every R
        calculation for the life of the account.

        `entry_state` is OPEN for a confirmed position. `record_entry_intent`
        below uses PENDING to write the row *before* the order is sent.
        """
        now = opened_at or self.clock.now()
        cursor = self.journal.conn.execute(
            """
            INSERT INTO trades (
                cycle_pk, ticket, magic, symbol, direction, volume, entry_price, sl, tp,
                risk_money, risk_pct, sl_distance_pips, planned_rr, opened_at, equity_before,
                entry_state
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                cycle_pk,
                ticket,
                self.settings.system.magic_number,
                sizing.symbol,
                sizing.direction.name,
                sizing.volume,
                entry_price,
                sizing.sl,
                sizing.tp,
                sizing.actual_risk_money,
                sizing.actual_risk_pct,
                sizing.sl_distance_pips,
                sizing.reward_risk,
                iso(now),
                equity_before,
                entry_state,
            ),
        )
        trade_id = int(cursor.lastrowid or 0)
        log.info(
            "trade opened" if entry_state == "OPEN" else "entry intent recorded",
            extra={
                "event": "journal_trade_open" if entry_state == "OPEN" else "journal_entry_intent",
                "trade_id": trade_id,
                "ticket": ticket,
                "symbol": sizing.symbol,
                "direction": sizing.direction.name,
                "volume": sizing.volume,
                "risk_money": round(sizing.actual_risk_money, 2),
                "risk_pct": round(sizing.actual_risk_pct, 4),
            },
        )
        return trade_id

    def record_entry_intent(
        self,
        *,
        cycle_pk: int | None,
        sizing: SizingResult,
        equity_before: float,
    ) -> int:
        """Write down what is about to be sent, before sending it.

        This is the durable half of closing the crash window. The row exists
        with everything except the ticket, so if the process dies between
        `order_send` and the confirmation the plan is still on disk and the
        resulting position can be recognised rather than liquidated.

        Committed immediately and deliberately: an intent still sitting in an
        uncommitted transaction when the power goes out is not an intent, and
        the whole point is to survive exactly that.
        """
        trade_id = self.record_trade_open(
            cycle_pk=cycle_pk,
            sizing=sizing,
            ticket=None,
            entry_price=sizing.entry,
            equity_before=equity_before,
            entry_state="PENDING",
        )
        self.journal.conn.commit()
        return trade_id

    def record_trade_close(
        self,
        trade_id: int,
        *,
        exit_price: float,
        pnl_money: float,
        exit_reason: str,
        equity_after: float,
        mae_r: float | None = None,
        mfe_r: float | None = None,
        closed_at: datetime | None = None,
    ) -> None:
        """Close a trade, deriving R from the risk recorded at open.

        R is computed against the risk that was *actually* taken, not against a
        nominal 1% — if the sizer rounded down to 0.01 lots and the real risk
        was 0.7%, then a full stop-out is -1R of 0.7%, and pretending otherwise
        makes the expectancy statistics wrong in the optimistic direction.
        """
        row = self.journal.conn.execute(
            "SELECT risk_money, opened_at FROM trades WHERE id = ?", (trade_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no trade with id {trade_id}")

        risk_money = float(row["risk_money"])
        pnl_r = pnl_money / risk_money if risk_money > 0 else 0.0

        now = closed_at or self.clock.now()
        opened = datetime.fromisoformat(row["opened_at"])
        duration = int((now - opened).total_seconds())

        # The excursions are ratcheted throughout the trade by the guard, once
        # a second. This write used to overwrite them with its own arguments,
        # which both callers leave at None — so every trade lost its peak and
        # trough at the exact moment they became history, and every postmortem
        # read "best it reached: unknown" on a trade the system had watched run
        # to 0.92R and said so in its own management log.
        #
        # `COALESCE(?, mfe_r)` keeps what is on the row when the caller has
        # nothing better. A caller that *does* pass a value still wins, which is
        # what the broker-recovery path needs when it reconstructs a closure
        # from deal history the guard never saw.
        self.journal.conn.execute(
            """
            UPDATE trades SET
                closed_at = ?, exit_price = ?, exit_reason = ?, pnl_money = ?, pnl_r = ?,
                mae_r = COALESCE(?, mae_r), mfe_r = COALESCE(?, mfe_r),
                duration_seconds = ?, equity_after = ?
            WHERE id = ?
            """,
            (
                iso(now),
                exit_price,
                exit_reason,
                pnl_money,
                pnl_r,
                mae_r,
                mfe_r,
                duration,
                equity_after,
                trade_id,
            ),
        )
        log.info(
            "trade closed",
            extra={
                "event": "journal_trade_close",
                "trade_id": trade_id,
                "exit_reason": exit_reason,
                "pnl_money": round(pnl_money, 2),
                "pnl_r": round(pnl_r, 3),
                "mae_r": mae_r,
                "mfe_r": mfe_r,
                "duration_s": duration,
            },
        )

    def update_excursions(self, trade_id: int, *, mae_r: float, mfe_r: float) -> None:
        """Ratchet MAE/MFE while a trade is open. See `Journal.update_excursions`.

        A passthrough because the position manager holds a `Journal` and not a
        `Recorder`; two copies of the same UPDATE would eventually disagree
        about which direction each column ratchets.
        """
        self.journal.update_excursions(trade_id, mae_r=mae_r, mfe_r=mfe_r)

    # -- executions and management ----------------------------------------

    def record_order_attempt(
        self, *, trade_id: int | None, kind: str, symbol: str, result: OrderResult
    ) -> None:
        """Store one `order_send` outcome, successful or not.

        Rejections are recorded too. A broker that rejects 8% of orders on
        Tuesday afternoons is a finding, and it is invisible unless the
        failures are stored alongside the successes.
        """
        self.journal.conn.execute(
            """
            INSERT INTO order_attempts (
                trade_id, ts, kind, symbol, ok, retcode, retcode_name, broker_comment,
                requested_price, filled_price, slippage_pips, requested_volume,
                filled_volume, latency_ms, spread_at_send, attempts
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                trade_id,
                iso(result.sent_at),
                kind,
                symbol,
                int(result.ok),
                result.retcode,
                result.retcode_name,
                result.comment,
                result.requested_price,
                result.filled_price,
                result.slippage_pips,
                result.requested_volume,
                result.filled_volume,
                result.latency_ms,
                result.spread_at_send,
                result.attempts,
            ),
        )

    def record_management_action(
        self,
        trade_id: int,
        *,
        action: str,
        old_sl: float | None = None,
        new_sl: float | None = None,
        old_tp: float | None = None,
        new_tp: float | None = None,
        volume_closed: float | None = None,
        r_at_action: float | None = None,
        note: str = "",
    ) -> None:
        """Record a break-even move, partial, trail step, or time/news exit."""
        self.journal.conn.execute(
            """
            INSERT INTO management_actions
                (trade_id, ts, action, old_sl, new_sl, old_tp, new_tp,
                 volume_closed, r_at_action, note)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                trade_id,
                iso(self.clock.now()),
                action,
                old_sl,
                new_sl,
                old_tp,
                new_tp,
                volume_closed,
                r_at_action,
                note,
            ),
        )

    def record_management_baseline(
        self,
        *,
        trade_id: int,
        outcome: str,
        baseline_pnl_r: float,
        actual_pnl_r: float,
        observed_at: datetime,
    ) -> None:
        """Persist the passive original-SL/TP comparison for one closed trade."""
        self.journal.conn.execute(
            "INSERT OR IGNORE INTO management_baselines "
            "(trade_id, observed_at, resolved_at, outcome, baseline_pnl_r, "
            "actual_pnl_r, lift_r) VALUES (?,?,?,?,?,?,?)",
            (
                trade_id,
                iso(observed_at),
                iso(self.clock.now()),
                outcome,
                baseline_pnl_r,
                actual_pnl_r,
                actual_pnl_r - baseline_pnl_r,
            ),
        )

    # -- shadow trades -----------------------------------------------------

    def record_shadow_trade(
        self,
        *,
        cycle_pk: int,
        symbol: str,
        direction: Direction,
        blocked_by: Reason,
        entry_price: float,
        sl: float,
        tp: float,
    ) -> int:
        """Register a setup a filter blocked, to resolve its outcome later.

        This is how "is the news filter too strict" becomes measurable rather
        than a matter of opinion. If blocked setups turn out to be
        systematically profitable, the filter is costing money and the data
        will say so.
        """
        cursor = self.journal.conn.execute(
            """
            INSERT INTO shadow_trades
                (cycle_pk, symbol, direction, blocked_by, entry_price, sl, tp, opened_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                cycle_pk,
                symbol,
                direction.name,
                str(blocked_by),
                entry_price,
                sl,
                tp,
                iso(self.clock.now()),
            ),
        )
        return int(cursor.lastrowid or 0)

    def has_unresolved_shadow_trade(self, symbol: str, direction: Direction) -> bool:
        """Avoid recording the same still-running hypothetical every cycle."""
        row = self.journal.conn.execute(
            """
            SELECT 1 FROM shadow_trades
            WHERE symbol = ? AND direction = ? AND outcome IS NULL
            LIMIT 1
            """,
            (symbol, direction.name),
        ).fetchone()
        return row is not None

    def unresolved_shadow_trades(self, limit: int = 50) -> list[Any]:
        """Return a bounded oldest-first queue for passive outcome resolution."""
        return list(
            self.journal.conn.execute(
                """
                SELECT * FROM shadow_trades
                WHERE outcome IS NULL
                ORDER BY opened_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        )

    def resolve_shadow_trade(self, shadow_id: int, *, outcome: str, pnl_r: float) -> None:
        self.journal.conn.execute(
            "UPDATE shadow_trades SET resolved_at = ?, outcome = ?, pnl_r = ? WHERE id = ?",
            (iso(self.clock.now()), outcome, pnl_r, shadow_id),
        )

    # -- config ------------------------------------------------------------

    def record_config_snapshot(self) -> str:
        """Store the config in force, keyed by hash. Returns the hash.

        Without this, a result six months old cannot be tied to the parameters
        that produced it, and the learning changelog becomes unfalsifiable.
        """
        payload = dumps(self.settings.redacted_dump())
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        existing = self.journal.conn.execute(
            "SELECT 1 FROM config_snapshots WHERE config_hash = ? LIMIT 1", (digest,)
        ).fetchone()
        if existing is None:
            self.journal.conn.execute(
                "INSERT INTO config_snapshots (ts, config_hash, config_json) VALUES (?,?,?)",
                (iso(self.clock.now()), digest, payload),
            )
            log.info(
                "config snapshot stored",
                extra={"event": "journal_config", "config_hash": digest},
            )
        return digest

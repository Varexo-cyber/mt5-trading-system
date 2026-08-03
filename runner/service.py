"""Long-running orchestration: scan, analyse, filter, size, execute, reconcile."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from advisory import (
    Advice,
    Advisor,
    AIReviewLedger,
    DisabledAdvisor,
    build_advisor,
    build_review_payload,
)
from analysis import (
    ConfluenceEngine,
    LevelReaction,
    LiquiditySweep,
    MarketStructure,
    TrendMomentum,
    VolatilityRegime,
)
from config.schema import Settings
from core.broker import Broker
from core.clock import LiveClock
from core.data_manager import DataManager
from core.errors import TradingSystemError
from core.startup import run_startup_guard
from core.types import AccountSnapshot, OrderRequest, TradingMode
from execution.manager import PositionManager
from execution.paper_broker import PaperBroker
from filters.base import FilterContext
from filters.news_filter import NewsFilter
from infra.killswitch import KillSwitch
from infra.logging import get_logger
from journal.database import Journal
from journal.recorder import CycleContext, Recorder
from learning.config_control import ShadowRecorder
from main import build_filter_chain
from monitoring.alerts import AlertSender
from monitoring.operation_ledger import OperationLedger
from monitoring.scan_activity import ScanActivityLedger
from promotion.audit import PromotionAudit
from promotion.experimental import (
    ExperimentalLiveContract,
    apply_experimental_live_limits,
    contract_path,
)
from reporting.daily_report import DailyReportGenerator
from reporting.execution_report import ExecutionReportGenerator
from reporting.weekly_report import WeeklyReportGenerator
from risk.position_sizer import PositionSizer
from risk.reasons import Reason
from risk.risk_manager import RiskManager
from scanner.universe import ScanBatch, UniverseScanner

log = get_logger(__name__)


class OperationMode(StrEnum):
    MONITOR = "monitor"
    PAPER = "paper"
    DEMO = "demo"
    LIVE = "live"
    EXPERIMENTAL_LIVE = "experimental_live"


@dataclass(frozen=True, slots=True)
class CycleSummary:
    started_at: datetime
    finished_at: datetime
    inspected: int
    rejected: int
    deep_analysed: int
    candidates: int
    trades_opened: int
    next_cursor: int
    universe_size: int


class JarvisRunner:
    """One deterministic service around an optional bounded AI adviser."""

    def __init__(
        self,
        market: Broker,
        settings: Settings,
        root: Path,
        operation: OperationMode = OperationMode.MONITOR,
        advisor: Advisor | None = None,
    ) -> None:
        self.root = root
        self.operation = operation
        self.settings = self._settings_for_operation(settings, operation)
        self.kill_switch = KillSwitch.in_dir(root, self.settings.system.kill_switch_file)
        self.broker: Broker = (
            PaperBroker(market, root / "runtime" / "paper_state.json")
            if operation is OperationMode.PAPER
            else market
        )
        self.clock = LiveClock()
        self.journal = Journal(
            root / self.settings.journal.database_path,
            self.clock,
            day_boundary_utc=self.settings.risk.day_boundary_utc,
        )
        self.data = DataManager(self.broker, self.settings.data, self.clock)
        self.engine = ConfluenceEngine(
            [
                MarketStructure(self.settings.analysis.market_structure),
                TrendMomentum(),
                LiquiditySweep(),
                LevelReaction(),
                VolatilityRegime(),
            ],
            self.settings.analysis.confluence,
        )
        self.shadow = ShadowRecorder(root)
        self.shadow_engine = self._shadow_engine()
        self.advisor = advisor or build_advisor(self.settings.ai)
        self.ai_ledger = AIReviewLedger(root / "runtime" / "ai_reviews.jsonl")
        self.cursor = self._load_cursor()
        self.operation_ledger = OperationLedger(root / "runtime" / "operation_history.json")
        self.scan_activity = ScanActivityLedger(root / "runtime" / "scan_activity.json")
        self.experimental_contract: ExperimentalLiveContract | None = None

    def connect(self) -> None:
        account = self.broker.connect()
        try:
            self._assert_account_mode(account.is_demo)
            if self.operation is OperationMode.LIVE:
                self._assert_live_armed(account.login)
            elif self.operation is OperationMode.EXPERIMENTAL_LIVE:
                contract = ExperimentalLiveContract.load(contract_path(self.root))
                contract.assert_compatible(account, self.settings)
                self._assert_ai_gate()
                self.experimental_contract = contract
        except Exception:
            self.broker.shutdown()
            raise
        self.clock.server_offset = self.broker.server_offset
        self.journal.open()
        self.recorder = Recorder(self.journal, self.clock, self.settings)
        self.risk = RiskManager(
            self.settings,
            self.journal,
            self.clock,
            self.kill_switch,
            margin_estimator=self.broker.estimate_margin,
        )
        self.filters = build_filter_chain(self.broker, self.settings, self.journal, self.clock)
        self.scanner = UniverseScanner(self.broker, self.settings, self.clock)
        self.manager = PositionManager(self.broker, self.journal, self.settings)
        self.alerts = AlertSender(self.settings.monitoring)
        self.reports = DailyReportGenerator(
            self.journal,
            self.root / self.settings.monitoring.report_directory,
            self.settings.monitoring.report_interval_minutes,
        )
        report_directory = self.root / self.settings.monitoring.report_directory
        self.weekly_reports = WeeklyReportGenerator(self.journal, report_directory, self.settings)
        self.execution_reports = ExecutionReportGenerator(
            self.journal, report_directory / "EXECUTION_REPORT.md"
        )
        self.recorder.record_config_snapshot()
        self._report_feasibility(account)
        if self.operation is OperationMode.EXPERIMENTAL_LIVE:
            self._report_promotion_evidence()
        log.info(
            "jarvis connected",
            extra={"event": "jarvis_start", "operation": self.operation.value},
        )
        self.operation_ledger.start(self.operation.value, account.login, self.clock.now())
        self.alerts.send(f"Jarvis started in {self.operation.value.upper()} mode")

    def close(self) -> None:
        self._save_cursor()
        self.operation_ledger.finish(self.clock.now())
        self.journal.close()
        self.broker.shutdown()

    def run_forever(self) -> None:
        self.connect()
        try:
            while True:
                if self.kill_switch.is_engaged():
                    remaining = self._flatten_owned_positions("operator hard STOP")
                    if not remaining:
                        log.warning("STOP engaged; owned positions flat; Jarvis service exiting")
                        break
                    log.critical(
                        "STOP engaged but owned positions remain; retrying closures",
                        extra={"event": "stop_close_retry", "positions": len(remaining)},
                    )
                    time.sleep(self.settings.system.loop_interval_seconds)
                    continue
                started = time.monotonic()
                self.run_once()
                elapsed = time.monotonic() - started
                time.sleep(max(0.0, self.settings.system.loop_interval_seconds - elapsed))
        finally:
            self.close()

    def run_once(
        self, *, batch_size: int | None = None, deep_candidates: int | None = None
    ) -> CycleSummary:
        started_at = self.clock.now()
        if self.kill_switch.is_engaged():
            self._flatten_owned_positions("operator hard STOP")
            return self._summary(started_at, ScanBatch((), (), 0, 0, self.cursor, 0), 0, 0)
        self.broker.ensure_connected()
        if isinstance(self.broker, PaperBroker):
            self._record_paper_closures(self.broker.mark_to_market())
        account = self.broker.account()
        positions = self.broker.positions(magic=self.settings.system.magic_number)
        if self._experimental_floor_tripped(account.equity, positions):
            return self._summary(started_at, ScanBatch((), (), 0, 0, self.cursor, 0), 0, 0)
        reconciliation = self.manager.reconcile(positions)
        self._record_management(reconciliation)
        if any(event.action == "BROKER_CLOSED_PENDING_HISTORY" for event in reconciliation):
            self.alerts.send(
                "CRITICAL: broker/journal closure could not be recovered; new risk halted"
            )
            self.risk.halt("broker/journal reconciliation requires deal-history recovery")
        positions = self.broker.positions(magic=self.settings.system.magic_number)
        news_filter = next(
            (item for item in self.filters.filters if isinstance(item, NewsFilter)), None
        )
        if news_filter is not None:
            self._record_management(self.manager.manage_news(positions, news_filter))
            positions = self.broker.positions(magic=self.settings.system.magic_number)
        self._record_management(self.manager.manage(positions, self.clock.now()))
        positions = self.broker.positions(magic=self.settings.system.magic_number)
        state = self.risk.build_state(account, positions)
        if self.risk.circuit_breaker_tripped(state):
            for position in positions:
                self.broker.close_position(position)
            self.alerts.send(self.risk.trip_circuit_breaker(state))
            return self._summary(started_at, ScanBatch((), (), 0, 0, self.cursor, 0), 0, 0)

        batch = self.scanner.scan(
            cursor=self.cursor,
            batch_size=batch_size if batch_size is not None else self.settings.scanner.batch_size,
            keep=(
                deep_candidates
                if deep_candidates is not None
                else self.settings.scanner.deep_candidates
            ),
        )
        self.scan_activity.record_batch(batch, started_at, self.operation.value)
        self.cursor = batch.next_cursor
        opened = 0
        deep = 0
        for candidate in batch.candidates:
            deep += 1
            if self._process_candidate(candidate.symbol, account, tuple(positions)):
                opened += 1
                account = self.broker.account()
                positions = self.broker.positions(magic=self.settings.system.magic_number)
                state = self.risk.build_state(account, positions)
                if not self.risk.check_can_trade(state).approved:
                    break
        self._save_cursor()
        summary = self._summary(started_at, batch, deep, opened)
        self._save_heartbeat(summary)
        self._log_cycle(summary, batch)
        self.operation_ledger.cycle(
            summary.finished_at,
            trades_opened=summary.trades_opened,
        )
        self.reports.maybe_generate(self.broker.account(), self.clock.now())
        self.weekly_reports.maybe_generate(self.broker.account(), self.clock.now())
        self.execution_reports.maybe_generate(self.clock.now())
        return summary

    def _process_candidate(self, symbol: str, account, positions) -> bool:  # type: ignore[no-untyped-def]
        cycle_id = str(uuid.uuid4())
        try:
            context = self.data.get_context(symbol, force_refresh=True)
            idea = self.engine.evaluate(context, self.settings.mode)
            if self.shadow_engine is not None:
                candidate = self.shadow_engine.evaluate(context, TradingMode.PAPER)
                self.shadow.record(symbol, idea, candidate, self.clock.now())
        except (TradingSystemError, ValueError) as exc:
            self._record_skip(cycle_id, symbol, account.equity, Reason.DATA_UNAVAILABLE, str(exc))
            return False
        if not idea.approved or idea.direction is None:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.NO_SIGNAL,
                idea.reason,
                signals=list(idea.signals),
            )
            return False

        spec = self.broker.spec(symbol)
        state = self.risk.build_state(account, positions)
        risk_decision = self.risk.evaluate(state, symbol, spec)
        if not risk_decision.approved:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                risk_decision.reason,
                risk_decision.detail,
                signals=list(idea.signals),
            )
            return False

        filter_verdict, filter_data = self.filters.check(
            FilterContext(
                symbol=symbol,
                spec=spec,
                now=self.clock.now(),
                direction=idea.direction,
                tick=context.tick,
                open_positions=positions,
            )
        )
        if not filter_verdict.passed:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                filter_verdict.reason,
                filter_verdict.detail,
                signals=list(idea.signals),
                extra=filter_data,
            )
            return False

        sizing = PositionSizer(self.settings).size(
            spec=spec,
            equity=account.equity,
            direction=idea.direction,
            entry=idea.entry,
            sl=spec.normalize_price(idea.stop_loss),
            tp=spec.normalize_price(idea.take_profit),
            risk_multiplier=self.risk.risk_multiplier(state),
        )
        if not sizing.approved:
            cycle_pk = self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                sizing.reason,
                sizing.decision.detail,
                signals=list(idea.signals),
            )
            self.recorder.record_sizing(cycle_pk, sizing)
            return False
        margin = self.risk.check_margin(
            state,
            symbol,
            idea.direction,
            sizing.volume,
            sizing.entry,
        )
        if not margin.approved:
            cycle_pk = self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                margin.reason,
                margin.detail,
                signals=list(idea.signals),
            )
            self.recorder.record_sizing(cycle_pk, sizing)
            return False
        self.risk.assert_not_forbidden(sizing, state)

        proposal = {
            "operation": self.operation.value,
            "account_currency": account.currency,
            "equity": account.equity,
            "margin_free": account.margin_free,
            "sizing": sizing.journal_row(),
            "filters": filter_data,
            "open_positions": len(positions),
            "quote_age_seconds": (
                max(0.0, (context.now - context.tick.time).total_seconds())
                if context.tick is not None
                else None
            ),
        }
        request_payload = build_review_payload(idea, context, proposal)
        try:
            self.ai_ledger.append(
                "pretrade_request",
                {
                    "cycle_id": cycle_id,
                    "symbol": symbol,
                    "direction": idea.direction.name,
                    "provider": (
                        "monitor"
                        if self.operation is OperationMode.MONITOR
                        else self.settings.ai.provider
                    ),
                    "model": self.settings.ai.anthropic_model or self.settings.ai.openai_model,
                    "request": request_payload,
                },
            )
        except OSError:
            advice = Advice(
                False,
                0.0,
                "AI review audit could not be persisted; trade vetoed",
                provider=self.settings.ai.provider,
                model=self.settings.ai.anthropic_model or self.settings.ai.openai_model,
                error="audit_write_failed",
            )
        else:
            ai_started = time.monotonic()
            advice = (
                Advice(
                    True,
                    1.0,
                    "Monitor mode does not request paid AI review",
                    provider="monitor",
                )
                if self.operation is OperationMode.MONITOR
                else self.advisor.review(idea, context, proposal)
            )
            ai_latency_ms = round((time.monotonic() - ai_started) * 1000, 1)
            try:
                self.ai_ledger.append(
                    "pretrade_response",
                    {
                        "cycle_id": cycle_id,
                        "symbol": symbol,
                        "direction": idea.direction.name,
                        "latency_ms": ai_latency_ms,
                        "decision": advice.safe_dict(),
                    },
                )
            except OSError:
                advice = Advice(
                    False,
                    0.0,
                    "AI review audit could not be persisted; trade vetoed",
                    provider=advice.provider,
                    model=advice.model,
                    request_id=advice.request_id,
                    error="audit_write_failed",
                )
        ai_data = {
            "ai_provider": advice.provider,
            "ai_model": advice.model,
            "ai_confidence": advice.confidence,
            "ai_risks": advice.risks,
            "ai_request_id": advice.request_id,
            "ai_error": advice.error,
        }
        if not advice.approved:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.AI_VETO,
                advice.thesis,
                signals=list(idea.signals),
                extra={**filter_data, **ai_data},
            )
            return False

        cycle_pk = self.recorder.record_cycle(
            cycle_id=cycle_id,
            context=CycleContext(symbol, account.equity, extra={**filter_data, **ai_data}),
            reason=Reason.OK,
            detail=idea.reason,
            traded=self.operation is not OperationMode.MONITOR,
            direction=idea.direction,
            total_score=idea.score,
            score_threshold=self.settings.analysis.confluence.score_threshold,
            signals=list(idea.signals),
            weights=self.settings.analysis.confluence.weights,
        )
        self.recorder.record_sizing(cycle_pk, sizing)
        if self.operation is OperationMode.MONITOR:
            self.scan_activity.record_deep_decision(
                symbol,
                "MONITOR_SIGNAL_ONLY",
                "All gates passed in monitor mode",
                "Monitor mode cannot send an order",
                self.clock.now(),
            )
            return False
        if not self._entry_still_allowed():
            self.scan_activity.record_deep_decision(
                symbol,
                "LAST_MOMENT_BLOCK",
                "Entry guard changed after analysis",
                "STOP, account contract or capital floor blocked the order",
                self.clock.now(),
            )
            return False
        request = OrderRequest(
            symbol=symbol,
            direction=idea.direction,
            volume=sizing.volume,
            sl=sizing.sl,
            tp=sizing.tp,
            reference_price=sizing.entry,
            deviation_points=self.settings.mt5.deviation_points,
            magic=self.settings.system.magic_number,
            comment=(
                "jarvis-exp-live" if self.operation is OperationMode.EXPERIMENTAL_LIVE else "jarvis"
            ),
        )
        result = self.broker.order_send(request, spec)
        if not result.ok:
            self.scan_activity.record_deep_decision(
                symbol,
                "ORDER_REJECTED",
                result.retcode_name,
                result.comment,
                self.clock.now(),
            )
            self.recorder.record_order_attempt(
                trade_id=None, kind="ENTRY", symbol=symbol, result=result
            )
            self.alerts.send(f"Order rejected: {symbol} {result.retcode_name} {result.comment}")
            return False
        trade_id = self.recorder.record_trade_open(
            cycle_pk=cycle_pk,
            sizing=sizing,
            ticket=result.position_ticket,
            entry_price=result.filled_price,
            equity_before=account.equity,
        )
        self.recorder.record_order_attempt(
            trade_id=trade_id, kind="ENTRY", symbol=symbol, result=result
        )
        self.alerts.send(
            f"Opened {symbol} {idea.direction.name} {sizing.volume:g} lots, "
            f"entry {result.filled_price:g}, SL {sizing.sl:g}, TP {sizing.tp:g}"
        )
        self.scan_activity.record_deep_decision(
            symbol,
            "TRADE_OPENED",
            "Broker confirmed the entry",
            f"Position ticket {result.position_ticket}",
            self.clock.now(),
        )
        return True

    def _record_skip(
        self,
        cycle_id: str,
        symbol: str,
        equity: float,
        reason: Reason,
        detail: str,
        *,
        signals=None,  # type: ignore[no-untyped-def]
        extra=None,  # type: ignore[no-untyped-def]
    ) -> int:
        cycle_pk = self.recorder.record_cycle(
            cycle_id=cycle_id,
            context=CycleContext(symbol, equity, extra=extra),
            reason=reason,
            detail=detail,
            signals=signals,
            weights=self.settings.analysis.confluence.weights,
        )
        self.scan_activity.record_deep_decision(
            symbol,
            "DEEP_REJECTED",
            str(reason),
            detail,
            self.clock.now(),
        )
        return cycle_pk

    def _record_paper_closures(self, events) -> None:  # type: ignore[no-untyped-def]
        for position, reason in events:
            row = self.journal.open_trade_by_ticket(position.ticket)
            if row is None:
                continue
            closed = self.broker.closed_position(position.ticket)
            if closed is None:
                self.risk.halt(f"paper closure #{position.ticket} missing from durable ledger")
                continue
            self.recorder.record_trade_close(
                int(row["id"]),
                exit_price=closed.exit_price,
                pnl_money=closed.pnl_money,
                exit_reason=reason,
                equity_after=self.broker.account().equity,
                closed_at=closed.closed_at,
            )
            self._reflect_closed_trade(
                row,
                exit_price=closed.exit_price,
                pnl_money=closed.pnl_money,
                exit_reason=reason,
                closed_at=closed.closed_at,
            )
            self.alerts.send(
                f"Paper position #{position.ticket} closed {reason}: "
                f"{closed.pnl_money:+.2f} {self.broker.account().currency}"
            )

    def _record_management(self, events) -> None:  # type: ignore[no-untyped-def]
        for event in events:
            if event.action == "BROKER_CLOSED_PENDING_HISTORY":
                self.operation_ledger.reconciliation_failure(self.clock.now())
            row = self.journal.open_trade_by_ticket(event.ticket)
            if row is None:
                continue
            trade_id = int(row["id"])
            self.recorder.record_management_action(
                trade_id,
                action=event.action,
                volume_closed=event.volume_closed,
                r_at_action=event.r_at_action,
                note=event.detail,
            )
            if event.remaining_volume is not None:
                self.journal.update_open_trade_volume(event.ticket, event.remaining_volume)
            if event.exit_price is not None and event.pnl_money is not None:
                self.recorder.record_trade_close(
                    trade_id,
                    exit_price=event.exit_price,
                    pnl_money=event.pnl_money,
                    exit_reason=event.action,
                    equity_after=self.broker.account().equity,
                    closed_at=event.closed_at,
                )
                self._reflect_closed_trade(
                    row,
                    exit_price=event.exit_price,
                    pnl_money=event.pnl_money,
                    exit_reason=event.action,
                    closed_at=event.closed_at,
                )
                self.alerts.send(
                    f"Position #{event.ticket} closed ({event.action}): "
                    f"{event.pnl_money:+.2f} {self.broker.account().currency}"
                )

    def _report_promotion_evidence(self) -> None:
        """Run the evidence audit for experimental live and report, without blocking.

        `LIVE` requires every check to pass. `EXPERIMENTAL_LIVE` deliberately
        does not: the audit needs 100 out-of-sample trades per module, and
        those cannot exist before the account has ever traded. Blocking here
        would make the experiment impossible, which defeats its purpose.

        What is not acceptable is doing that silently. The audit runs, every
        failing check is logged and alerted, and the operator sees exactly
        which evidence is missing from the money they are about to risk. The
        capital protections that do apply — account-bound contract, 1% risk,
        the 15% floor, every filter, the AI veto — are unaffected.
        """
        checks = PromotionAudit(self.root, self.settings).run()
        failures = [check for check in checks if not check.passed]
        log.warning(
            "experimental live running on incomplete evidence",
            extra={
                "event": "experimental_live_evidence",
                "checks_total": len(checks),
                "checks_failed": len(failures),
                "missing": [f"{item.name}: {item.detail}" for item in failures],
            },
        )
        if failures:
            self.alerts.send(
                f"EXPERIMENTAL LIVE armed with {len(failures)}/{len(checks)} promotion checks "
                f"still failing. This is real money on unvalidated evidence: "
                + "; ".join(f"{item.name}" for item in failures[:6])
            )

    def _report_feasibility(self, account: AccountSnapshot) -> None:
        """Log what this equity can actually express, before the first cycle.

        The autonomous path had no equivalent of `main.py --status`, so the one
        fact that governs a EUR 100 account — that 1% risk buys roughly eleven
        pips of stop at the minimum lot, and every structural stop wider than
        that is skipped — was visible only if the operator happened to run the
        diagnostic first. A run that skips every setup for an entirely
        legitimate reason looks identical to one that is broken.

        Reports; never blocks. The hard refusals live in `connect` above, and
        the position sizer enforces the arithmetic per trade regardless.
        """
        try:
            report = run_startup_guard(self.settings, self.broker, account)
        except Exception:
            log.exception("feasibility report failed", extra={"event": "feasibility_failed"})
            return

        expressible = [item.symbol for item in report.symbols if item.is_expressible]
        log.info(
            "startup feasibility",
            extra={
                "event": "feasibility_report",
                "equity": account.equity,
                "currency": account.currency,
                "risk_money": round(report.risk_money, 2),
                "expressible": expressible,
                "warnings": list(report.warnings),
                "errors": list(report.errors),
            },
        )
        if not expressible:
            self.alerts.send(
                f"No whitelisted symbol can express a trade at {report.risk_pct:.2f}% risk on "
                f"{account.equity:.2f} {account.currency}. The system will scan and analyse "
                f"but skip essentially every setup."
            )

    def _assert_live_armed(self, login: int) -> None:
        PromotionAudit(self.root, self.settings).assert_passed()
        path = self.root / "runtime" / "LIVE_ARMED.json"
        if not path.exists():
            raise RuntimeError("LIVE_NOT_ARMED: run paper/demo acceptance and create arming file")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("login") != login or payload.get("phrase") != "I_ACCEPT_LIVE_RISK":
            raise RuntimeError("LIVE_NOT_ARMED: account or confirmation phrase mismatch")

    def _assert_account_mode(self, is_demo: bool) -> None:
        if self.operation is OperationMode.DEMO and not is_demo:
            raise RuntimeError(
                "DEMO_ACCOUNT_REQUIRED: refusing to send demo orders to a live MT5 account"
            )
        if self.operation in {OperationMode.LIVE, OperationMode.EXPERIMENTAL_LIVE} and is_demo:
            raise RuntimeError("LIVE_ACCOUNT_REQUIRED: live mode cannot run on a demo account")

    def _assert_ai_gate(self) -> None:
        if isinstance(self.advisor, DisabledAdvisor):
            raise RuntimeError(  # noqa: TRY004 - runtime readiness, not caller type
                "EXPERIMENTAL_LIVE_REQUIRES_AI: configured adviser is disabled or unavailable"
            )

    def _reflect_closed_trade(
        self,
        row,  # type: ignore[no-untyped-def]
        *,
        exit_price: float,
        pnl_money: float,
        exit_reason: str,
        closed_at: datetime | None,
    ) -> None:
        if isinstance(self.advisor, DisabledAdvisor):
            return
        risk_money = float(row["risk_money"])
        outcome = {
            "trade_id": int(row["id"]),
            "ticket": int(row["ticket"]) if row["ticket"] is not None else None,
            "symbol": str(row["symbol"]),
            "direction": str(row["direction"]),
            "volume": float(row["volume"]),
            "entry_price": float(row["entry_price"]),
            "stop_loss": float(row["sl"]),
            "take_profit": float(row["tp"]),
            "planned_risk_money": risk_money,
            "planned_risk_pct": float(row["risk_pct"]),
            "planned_reward_risk": float(row["planned_rr"]),
            "exit_price": exit_price,
            "pnl_money": pnl_money,
            "pnl_r": pnl_money / risk_money if risk_money > 0 else 0.0,
            "exit_reason": exit_reason,
            "opened_at": str(row["opened_at"]),
            "closed_at": closed_at.isoformat() if closed_at is not None else None,
        }
        reflection = self.advisor.reflect(outcome)
        try:
            self.ai_ledger.append(
                "posttrade_reflection",
                {"outcome": outcome, "reflection": reflection.safe_dict()},
            )
        except OSError:
            log.exception(
                "failed to persist AI post-trade reflection",
                extra={"event": "ai_reflection_audit_failed", "trade_id": int(row["id"])},
            )

    @staticmethod
    def _settings_for_operation(settings: Settings, operation: OperationMode) -> Settings:
        if operation is OperationMode.EXPERIMENTAL_LIVE:
            return apply_experimental_live_limits(settings)
        mode = (
            TradingMode.PAPER
            if operation in {OperationMode.MONITOR, OperationMode.PAPER, OperationMode.DEMO}
            else TradingMode.MICRO_LIVE
        )
        return settings.model_copy(
            update={"system": settings.system.model_copy(update={"mode": mode})}
        )

    def _entry_still_allowed(self) -> bool:
        """Last-moment entry guard; exits and position management remain available."""
        if self.kill_switch.is_engaged():
            return False
        if self.experimental_contract is None:
            return True
        account = self.broker.account()
        if self._experimental_floor_tripped(
            account.equity,
            self.broker.positions(magic=self.settings.system.magic_number),
        ):
            return False
        self.experimental_contract.assert_compatible(account, self.settings)
        return True

    def _experimental_floor_tripped(self, equity: float, positions) -> bool:  # type: ignore[no-untyped-def]
        contract = self.experimental_contract
        if contract is None or not contract.floor_breached(equity):
            return False
        message = (
            f"EXPERIMENTAL LIVE CAPITAL STOP: equity {equity:.2f} {contract.currency} "
            f"reached floor {contract.equity_floor:.2f} from initial "
            f"{contract.initial_equity:.2f}."
        )
        remaining = self._flatten_owned_positions(message, positions=positions)
        if remaining:
            message += f" {len(remaining)} owned position(s) still require closure."
        else:
            message += " All owned positions are flat."
        self.risk.halt(message)
        self.kill_switch.engage(message)
        self.alerts.send(message)
        return True

    def _flatten_owned_positions(self, reason: str, *, positions=None):  # type: ignore[no-untyped-def]
        """Close only this system's magic-number positions and report survivors."""
        owned = (
            tuple(positions)
            if positions is not None
            else tuple(self.broker.positions(magic=self.settings.system.magic_number))
        )
        for position in owned:
            try:
                result = self.broker.close_position(position)
            except Exception:
                log.exception(
                    "exception while flattening owned position",
                    extra={"event": "flatten_exception", "ticket": position.ticket},
                )
                continue
            if not result.ok:
                log.critical(
                    "broker rejected emergency close",
                    extra={
                        "event": "flatten_rejected",
                        "ticket": position.ticket,
                        "retcode": result.retcode,
                        "comment": result.comment,
                    },
                )
        remaining = tuple(self.broker.positions(magic=self.settings.system.magic_number))
        if remaining:
            self.alerts.send(f"CRITICAL: {reason}; {len(remaining)} Jarvis position(s) still open")
        return remaining

    def _load_cursor(self) -> int:
        path = self.root / "runtime" / "runner_state.json"
        if not path.exists():
            return 0
        try:
            return int(json.loads(path.read_text(encoding="utf-8")).get("cursor", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            return 0

    def _shadow_engine(self) -> ConfluenceEngine | None:
        candidate_path = self.root / "runtime" / "shadow" / "candidate.yaml"
        if not candidate_path.exists():
            return None
        from config.loader import load_settings

        candidate = load_settings(candidate_path, env_overrides=False)
        return ConfluenceEngine(
            [
                MarketStructure(candidate.analysis.market_structure),
                TrendMomentum(),
                LiquiditySweep(),
                LevelReaction(),
                VolatilityRegime(),
            ],
            candidate.analysis.confluence,
        )

    def _save_cursor(self) -> None:
        path = self.root / "runtime" / "runner_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"cursor": self.cursor}), encoding="utf-8")

    def _log_cycle(self, summary: CycleSummary, batch: ScanBatch) -> None:
        """One line per cycle, so the console shows the system is alive.

        There was no such line. The only thing reaching the console during a
        scan was an accidental burst of connector warnings, and once those were
        fixed the process looked frozen — a full-catalogue pass takes minutes
        and printed nothing at all until it opened a trade, which on a EUR 100
        account may be never. Silence and a hang have to look different.

        The rejection breakdown is here because "659 rejected" invites exactly
        one question and answers none of it. Six hundred instruments refused
        because their exchange is shut at 08:15 UTC is the system working;
        six hundred refused on spread is a misconfigured limit. Those look
        identical in a bare count and need opposite responses.
        """
        elapsed = (summary.finished_at - summary.started_at).total_seconds()
        stages: dict[str, int] = {}
        for row in batch.inspections:
            if row.status == "REJECTED":
                stages[row.stage] = stages.get(row.stage, 0) + 1
        breakdown = ", ".join(
            f"{n} {stage}" for stage, n in sorted(stages.items(), key=lambda kv: -kv[1])
        )
        log.info(
            "cycle: %d/%d scanned, %d eligible, %d analysed, %d opened, %.0fs%s",
            summary.inspected,
            summary.universe_size,
            summary.inspected - summary.rejected,
            summary.deep_analysed,
            summary.trades_opened,
            elapsed,
            f" | rejected: {breakdown}" if breakdown else "",
            extra={
                "event": "cycle_complete",
                "inspected": summary.inspected,
                "universe_size": summary.universe_size,
                "eligible": summary.inspected - summary.rejected,
                "rejected": summary.rejected,
                "rejected_by_stage": stages,
                "deep_analysed": summary.deep_analysed,
                "trades_opened": summary.trades_opened,
                "seconds": round(elapsed, 1),
                "next_cursor": summary.next_cursor,
            },
        )

    def _save_heartbeat(self, summary: CycleSummary) -> None:
        path = self.root / "runtime" / "heartbeat.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "operation": self.operation.value,
                    "finished_at": summary.finished_at.isoformat(),
                    "inspected": summary.inspected,
                    "rejected": summary.rejected,
                    "deep_analysed": summary.deep_analysed,
                    "trades_opened": summary.trades_opened,
                    "next_cursor": summary.next_cursor,
                    "universe_size": summary.universe_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _summary(self, started: datetime, batch: ScanBatch, deep: int, opened: int) -> CycleSummary:
        return CycleSummary(
            started_at=started,
            finished_at=self.clock.now(),
            inspected=batch.inspected,
            rejected=batch.rejected,
            deep_analysed=deep,
            candidates=len(batch.candidates),
            trades_opened=opened,
            next_cursor=batch.next_cursor,
            universe_size=batch.universe_size,
        )

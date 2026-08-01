"""Long-running orchestration: scan, analyse, filter, size, execute, reconcile."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from advisory import build_advisor
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
from core.types import OrderRequest, TradingMode
from execution.manager import PositionManager
from execution.paper_broker import PaperBroker
from filters.base import FilterContext
from infra.killswitch import KillSwitch
from infra.logging import get_logger
from journal.database import Journal
from journal.recorder import CycleContext, Recorder
from main import build_filter_chain
from monitoring.alerts import AlertSender
from reporting.daily_report import DailyReportGenerator
from risk.position_sizer import PositionSizer
from risk.reasons import Reason
from risk.risk_manager import RiskManager
from scanner.universe import ScanBatch, UniverseScanner

log = get_logger(__name__)


class OperationMode(StrEnum):
    MONITOR = "monitor"
    PAPER = "paper"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class CycleSummary:
    started_at: datetime
    finished_at: datetime
    inspected: int
    deep_analysed: int
    candidates: int
    trades_opened: int
    next_cursor: int


class JarvisRunner:
    """One deterministic service around an optional bounded AI adviser."""

    def __init__(
        self,
        market: Broker,
        settings: Settings,
        root: Path,
        operation: OperationMode = OperationMode.MONITOR,
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
        self.advisor = build_advisor(self.settings.ai)
        self.cursor = self._load_cursor()

    def connect(self) -> None:
        account = self.broker.connect()
        self.clock.server_offset = self.broker.server_offset
        self.journal.open()
        self.recorder = Recorder(self.journal, self.clock, self.settings)
        self.risk = RiskManager(self.settings, self.journal, self.clock, self.kill_switch)
        self.filters = build_filter_chain(self.broker, self.settings, self.journal, self.clock)
        self.scanner = UniverseScanner(self.broker, self.settings)
        self.manager = PositionManager(self.broker, self.journal, self.settings)
        self.alerts = AlertSender(self.settings.monitoring)
        self.reports = DailyReportGenerator(
            self.journal,
            self.root / self.settings.monitoring.report_directory,
            self.settings.monitoring.report_interval_minutes,
        )
        self.recorder.record_config_snapshot()
        if self.operation is OperationMode.LIVE:
            self._assert_live_armed(account.login)
        log.info(
            "jarvis connected",
            extra={"event": "jarvis_start", "operation": self.operation.value},
        )
        self.alerts.send(f"Jarvis started in {self.operation.value.upper()} mode")

    def close(self) -> None:
        self._save_cursor()
        self.journal.close()
        self.broker.shutdown()

    def run_forever(self) -> None:
        self.connect()
        try:
            while True:
                if self.kill_switch.is_engaged():
                    log.warning("STOP engaged; Jarvis service exiting")
                    break
                started = time.monotonic()
                self.run_once()
                elapsed = time.monotonic() - started
                time.sleep(max(0.0, self.settings.system.loop_interval_seconds - elapsed))
        finally:
            self.close()

    def run_once(self, *, batch_size: int = 25, deep_candidates: int = 5) -> CycleSummary:
        started_at = self.clock.now()
        if self.kill_switch.is_engaged():
            return self._summary(started_at, ScanBatch((), 0, 0, self.cursor, 0), 0, 0)
        self.broker.ensure_connected()
        if isinstance(self.broker, PaperBroker):
            self._record_paper_closures(self.broker.mark_to_market())
        account = self.broker.account()
        positions = self.broker.positions(magic=self.settings.system.magic_number)
        reconciliation = self.manager.reconcile(positions)
        self._record_management(reconciliation)
        if any(event.action == "BROKER_CLOSED_PENDING_HISTORY" for event in reconciliation):
            self.risk.halt("broker/journal reconciliation requires deal-history recovery")
        positions = self.broker.positions(magic=self.settings.system.magic_number)
        self._record_management(self.manager.manage(positions, self.clock.now()))
        positions = self.broker.positions(magic=self.settings.system.magic_number)
        state = self.risk.build_state(account, positions)
        if self.risk.circuit_breaker_tripped(state):
            for position in positions:
                self.broker.close_position(position)
            self.risk.trip_circuit_breaker(state)
            return self._summary(started_at, ScanBatch((), 0, 0, self.cursor, 0), 0, 0)

        batch = self.scanner.scan(cursor=self.cursor, batch_size=batch_size, keep=deep_candidates)
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
        self.reports.maybe_generate(self.broker.account(), self.clock.now())
        return summary

    def _process_candidate(self, symbol: str, account, positions) -> bool:  # type: ignore[no-untyped-def]
        cycle_id = str(uuid.uuid4())
        try:
            context = self.data.get_context(symbol, force_refresh=True)
            idea = self.engine.evaluate(context, self.settings.mode)
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

        advice = self.advisor.review(idea, context)
        if not advice.approved:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.AI_VETO,
                advice.thesis,
                signals=list(idea.signals),
                extra={"ai_provider": advice.provider, "ai_risks": advice.risks},
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
        self.risk.assert_not_forbidden(sizing, state)

        cycle_pk = self.recorder.record_cycle(
            cycle_id=cycle_id,
            context=CycleContext(symbol, account.equity, extra=filter_data),
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
            comment="jarvis",
        )
        result = self.broker.order_send(request, spec)
        if not result.ok:
            self.recorder.record_order_attempt(
                trade_id=None, kind="ENTRY", symbol=symbol, result=result
            )
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
        return self.recorder.record_cycle(
            cycle_id=cycle_id,
            context=CycleContext(symbol, equity, extra=extra),
            reason=reason,
            detail=detail,
            signals=signals,
            weights=self.settings.analysis.confluence.weights,
        )

    def _record_paper_closures(self, events) -> None:  # type: ignore[no-untyped-def]
        for position, reason in events:
            row = self.journal.open_trade_by_ticket(position.ticket)
            if row is None:
                continue
            self.recorder.record_trade_close(
                int(row["id"]),
                exit_price=(position.tp if reason == "TP" else position.sl),
                pnl_money=position.profit,
                exit_reason=reason,
                equity_after=self.broker.account().equity,
            )

    def _record_management(self, events) -> None:  # type: ignore[no-untyped-def]
        for event in events:
            row = self.journal.open_trade_by_ticket(event.ticket)
            if row is None:
                continue
            trade_id = int(row["id"])
            self.recorder.record_management_action(
                trade_id,
                action=event.action,
                note=event.detail,
            )
            if event.exit_price is not None and event.pnl_money is not None:
                self.recorder.record_trade_close(
                    trade_id,
                    exit_price=event.exit_price,
                    pnl_money=event.pnl_money,
                    exit_reason=event.action,
                    equity_after=self.broker.account().equity,
                )

    def _assert_live_armed(self, login: int) -> None:
        path = self.root / "runtime" / "LIVE_ARMED.json"
        if not path.exists():
            raise RuntimeError("LIVE_NOT_ARMED: run paper/demo acceptance and create arming file")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("login") != login or payload.get("phrase") != "I_ACCEPT_LIVE_RISK":
            raise RuntimeError("LIVE_NOT_ARMED: account or confirmation phrase mismatch")

    @staticmethod
    def _settings_for_operation(settings: Settings, operation: OperationMode) -> Settings:
        mode = (
            TradingMode.PAPER
            if operation in {OperationMode.MONITOR, OperationMode.PAPER}
            else TradingMode.MICRO_LIVE
        )
        return settings.model_copy(
            update={"system": settings.system.model_copy(update={"mode": mode})}
        )

    def _load_cursor(self) -> int:
        path = self.root / "runtime" / "runner_state.json"
        if not path.exists():
            return 0
        try:
            return int(json.loads(path.read_text(encoding="utf-8")).get("cursor", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            return 0

    def _save_cursor(self) -> None:
        path = self.root / "runtime" / "runner_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"cursor": self.cursor}), encoding="utf-8")

    def _save_heartbeat(self, summary: CycleSummary) -> None:
        path = self.root / "runtime" / "heartbeat.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "operation": self.operation.value,
                    "finished_at": summary.finished_at.isoformat(),
                    "inspected": summary.inspected,
                    "deep_analysed": summary.deep_analysed,
                    "trades_opened": summary.trades_opened,
                    "next_cursor": summary.next_cursor,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _summary(self, started: datetime, batch: ScanBatch, deep: int, opened: int) -> CycleSummary:
        return CycleSummary(
            started,
            self.clock.now(),
            batch.inspected,
            deep,
            len(batch.candidates),
            opened,
            batch.next_cursor,
        )

"""Long-running orchestration: scan, analyse, filter, size, execute, reconcile."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import pandas as pd

from advisory import (
    Advice,
    Advisor,
    AIReviewLedger,
    DisabledAdvisor,
    VetoMemory,
    build_advisor,
    build_review_payload,
    build_supervision_payload,
)
from analysis import (
    ConfluenceEngine,
    LevelReaction,
    LiquiditySweep,
    MarketStructure,
    TrendMomentum,
    VolatilityRegime,
)
from analysis.confluence import TradeIdea
from analysis.playbooks import (
    FadeConfig,
    MomentumScalp,
    Playbook,
    PlaybookEngine,
    PlaybookVerdict,
    RangeFade,
    ScalpConfig,
)
from config.schema import Settings
from core.broker import Broker
from core.clock import Clock, LiveClock
from core.data_manager import DataManager
from core.errors import TradingSystemError
from core.startup import run_startup_guard
from core.types import AccountSnapshot, MarketContext, OrderRequest, Timeframe, TradingMode
from execution.manager import ManagementEvent, PositionManager
from execution.paper_broker import PaperBroker
from filters.base import FilterContext
from filters.news_filter import NewsFilter
from infra.atomic import write_json_atomic
from infra.killswitch import KillSwitch
from infra.logging import get_logger
from journal.database import Journal
from journal.recorder import CycleContext, Recorder
from learning.config_control import ShadowRecorder
from learning.memory import TradingMemory
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
from risk.posture import PostureAssessment, assess
from risk.reasons import Reason
from risk.risk_manager import RiskManager
from scanner.universe import ScanBatch, UniverseScanner

log = get_logger(__name__)

#: What is reported for a position the fast layer has not read yet — one opened
#: seconds ago, or one whose bars could not be fetched. Deliberately not
#: "healthy": "we looked and it is fine" and "we have not looked" are different
#: claims, and only one of them is true here.
_UNKNOWN_HEALTH: dict[str, object] = {
    "verdict": "unknown",
    "severity": 0.0,
    "action": "hold",
    "signals": [],
}

#: The timeframe a review is tied to. H1 is where the weighted modules read
#: their structure, so it is what defines "the same setup".
_REVIEW_TIMEFRAME = Timeframe.H1

#: AI verdicts retained. One per symbol and direction per bar of the fastest
#: timeframe, so a few hundred covers a full catalogue for several bars.
_REVIEW_CACHE_ENTRIES = 500


class OperationMode(StrEnum):
    MONITOR = "monitor"
    PAPER = "paper"
    DEMO = "demo"
    LIVE = "live"
    EXPERIMENTAL_LIVE = "experimental_live"


@dataclass(frozen=True, slots=True)
class AnalysedCandidate:
    """A setup the engine has already judged, waiting its turn to be acted on.

    Carries its own `context` so the execution phase does not refetch seven
    timeframes it already has. That makes the queue slightly stale by the time
    the last entry is reached, which is correct for the *chart* — closed bars do
    not change — and deliberately not relied on for anything else: the tick,
    the account, the open positions and every risk gate are re-read at
    execution time.
    """

    symbol: str
    cycle_id: str
    idea: TradeIdea
    context: MarketContext

    @property
    def conviction(self) -> float:
        """How strongly the engine believes this one, for ranking only.

        Score times confidence, because the two say different things: a high
        score from modules that are individually unsure is not the same claim as
        a moderate score they all agree on, and multiplying is the honest way to
        say a weakness in either weakens the whole.

        Never a gate. `score_threshold` decides what is tradeable; this only
        decides what is looked at first.
        """
        return self.idea.score * self.idea.confidence


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
        clock: Clock | None = None,
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
        # Injectable so a whole cycle can be exercised at a chosen moment. This
        # was the last hardcoded LiveClock, and while it stood no end-to-end
        # test could place itself on, say, a Monday morning — which is exactly
        # when the interesting failures happen.
        self.clock: Clock = clock or LiveClock()
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
        self.playbook_config = self.settings.analysis.playbooks
        self.playbooks = self._build_playbooks()
        self.advisor = advisor or build_advisor(self.settings.ai)
        self.ai_ledger = AIReviewLedger(root / "runtime" / "ai_reviews.jsonl")
        self.cursor = self._load_cursor()
        self.operation_ledger = OperationLedger(root / "runtime" / "operation_history.json")
        self.scan_activity = ScanActivityLedger(root / "runtime" / "scan_activity.json")
        self.experimental_contract: ExperimentalLiveContract | None = None
        # (symbol, direction, fastest-timeframe bar close) -> the verdict given.
        self._review_cache: dict[tuple[str, str, datetime], Advice] = {}
        # Refusals outlive the bar they were given on; see advisory/veto_memory.
        self.veto_memory = VetoMemory(root / "runtime" / "veto_memory.json")
        # What the account has taught itself, fed back into every review.
        self.memory = TradingMemory(root / "runtime" / "trading_memory.json")
        # The fast layer's live read, published for the deck. The manager holds
        # it in memory; the dashboard is a separate process and cannot see that.
        self.health_file = root / "runtime" / "position_health.json"
        # ticket -> when the supervisor last looked at it, so an open position
        # is reconsidered on a sane cadence rather than every thirty seconds.
        self._supervised_at: dict[int, datetime] = {}
        # Why new risk is refused, if it is. Empty means trading is permitted.
        self.blocked_reason = ""
        self.blocked_detail = ""
        # Recomputed every cycle; STEADY until the first one runs.
        self.posture: PostureAssessment = assess(consecutive_losses=0, equity=1.0, equity_peak=1.0)

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
                # The gap between cycles is not idle time — it is the time open
                # money spends unwatched. Spend it watching.
                self._guard_until(started + self.settings.system.loop_interval_seconds)
        finally:
            self.close()

    def _guard_until(self, deadline: float) -> None:
        """Watch open positions until the next full cycle is due.

        A cycle scans the whole catalogue and can take most of a minute. Before
        this, management ran once per cycle, so a trade could run to 1.6R and
        give all of it back between two consecutive looks — the rules were
        right and simply were not being asked often enough.

        Only the mechanical rules run here: break-even, the trail, the
        give-back exit. No scan, no sizing, no adviser. That is what makes a
        one-second cadence affordable — the adviser stays on its own interval
        and its cost is unchanged by anything in this loop.
        """
        interval = self.settings.system.guard_interval_seconds
        if interval <= 0:
            time.sleep(max(0.0, deadline - time.monotonic()))
            return
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(interval, remaining))
            if self.kill_switch.is_engaged():
                # The full cycle handles the flattening; returning early gets
                # us there now rather than after the rest of the interval.
                return
            self.guard_tick()

    def guard_tick(self) -> list[ManagementEvent]:
        """One cheap pass over open positions. Never opens anything.

        Failures are swallowed deliberately. This runs between cycles on a
        best-effort basis; a dropped tick or a momentary IPC hiccup must not
        take the service down, and the next full cycle re-does everything this
        does with its own error handling.
        """
        try:
            self.broker.ensure_connected()
            positions = self.broker.positions(magic=self.settings.system.magic_number)
            if not positions:
                return []
            events = self.manager.manage(
                positions, self.clock.now(), self.posture.patience_multiplier
            )
            self._record_management(events)
            self._publish_health(positions)
            return events
        except Exception as exc:  # noqa: BLE001 - see docstring
            log.warning(
                "guard tick failed: %s",
                exc,
                extra={"event": "guard_tick_failed", "error": type(exc).__name__},
            )
            return []

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
        # How the account should be carrying itself, given the last few trades.
        # Only ever tightens: less patience with a stalled trade, a higher bar
        # for a new one. Never larger size — see risk/posture.py.
        interim = self.risk.build_state(account, positions)
        self.posture = assess(
            consecutive_losses=interim.consecutive_losses,
            equity=interim.equity,
            equity_peak=interim.equity_peak,
        )
        if self.posture.is_stressed:
            log.warning(
                "trading posture is %s",
                self.posture.posture.value,
                extra={
                    "event": "posture",
                    "posture": self.posture.posture.value,
                    "consecutive_losses": self.posture.consecutive_losses,
                    "drawdown_pct": round(self.posture.drawdown_pct, 2),
                    "patience": self.posture.patience_multiplier,
                    "candidates_allowed": self.posture.max_candidates,
                },
            )
        self._record_management(
            self.manager.manage(positions, self.clock.now(), self.posture.patience_multiplier)
        )
        positions = self.broker.positions(magic=self.settings.system.magic_number)
        # The mechanical rules have had their say; now the judgement layer. It
        # runs after them deliberately — break-even and the ATR trail are
        # cheap, deterministic and always correct to apply, so they should not
        # wait on an API call, and the supervisor sees the position in the state
        # those rules left it.
        self._supervise_positions(positions)
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

        # Ask first whether a trade is possible at all. This was only checked
        # *after* one opened, so with the maximum positions already running the
        # loop still walked every candidate, sized it, and paid Claude to review
        # a trade that could not be placed no matter what the answer was. The
        # money and the seconds were spent on a question already settled.
        permission = self.risk.check_can_trade(state)
        # Recorded, not merely logged. A halt is the single most important fact
        # about a running system and it was visible only as one INFO line among
        # hundreds scrolling past — so a correctly halted account and a broken
        # one looked identical, and the dashboard showed a busy scanner next to
        # "0 analysed" with no explanation anywhere.
        self.blocked_reason = "" if permission.approved else str(permission.reason)
        self.blocked_detail = "" if permission.approved else permission.detail
        if not permission.approved:
            log.warning(
                "NEW RISK HALTED: %s - %s",
                permission.reason,
                permission.detail,
                extra={
                    "event": "cycle_no_new_risk",
                    "reason": str(permission.reason),
                    "detail": permission.detail,
                    "open_positions": len(positions),
                },
            )
            batch = ScanBatch(
                (),
                batch.inspections,
                batch.inspected,
                batch.rejected,
                batch.next_cursor,
                batch.universe_size,
            )

        # Analyse everything first, then act on the strongest.
        #
        # The loop used to take candidates in scanner order and stop as soon as
        # the position slots were full, which meant the account's two slots went
        # to whichever acceptable setup happened to be scanned first. With ~200
        # markets analysed per cycle and two slots, "acceptable and early" is a
        # much weaker filter than "best available", and the difference is not
        # subtle: the scanner's cheap pre-rank is a trend/activity heuristic
        # that knows nothing about whether the setup is any good.
        #
        # So the chart work — which is the same work either way — happens for
        # every candidate before any of it is acted on, and the queue is sorted
        # by the engine's own conviction. Everything downstream (risk, filters,
        # sizing, the AI review, the order) then runs in that order, so the
        # scarce slots and the paid reviews go to the best ideas available.
        analysed = self._analyse_batch(batch, account)
        # Every candidate now gets the full chart analysis, so this is the
        # honest count of deep work done — not the number that survived it.
        deep = len(batch.candidates)
        for rank, candidate in enumerate(analysed, start=1):
            try:
                traded = self._process_candidate(
                    candidate, account, tuple(positions), rank=rank, of=len(analysed)
                )
            except Exception:
                # One symbol must never end the run. The catalogue is 850
                # instruments of wildly different shapes and any of them can
                # surprise the analysis; the loop's job is to keep managing the
                # account, not to be perfect about every candidate.
                #
                # This is not hypothetical. A dashboard held scan_activity.json
                # open, the atomic replace was denied, and the exception came
                # out of the *skip-recording* path — so a failed telemetry write
                # about a market being skipped killed the whole service and left
                # open positions unmanaged. The write is safe now; this is the
                # backstop that should have been here anyway.
                log.exception(
                    "candidate failed; continuing with the rest of the batch",
                    extra={"event": "candidate_error", "symbol": candidate.symbol},
                )
                continue
            if traded:
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

    def _analyse_batch(self, batch: ScanBatch, account) -> list[AnalysedCandidate]:  # type: ignore[no-untyped-def]
        """Judge every candidate, then return the survivors strongest first.

        Only the parts of the decision that do not depend on evolving state
        happen here: read the charts, score the setup, and check it against the
        standing refusals. Everything that can change as trades open — risk
        gates, filters, sizing, margin — is deliberately left to the execution
        phase, so a candidate ranked third is checked against the account as it
        stands when its turn comes, not as it stood at the top of the cycle.
        """
        analysed: list[AnalysedCandidate] = []
        for candidate in batch.candidates:
            try:
                item = self._analyse_candidate(candidate.symbol, account)
            except Exception:
                log.exception(
                    "candidate analysis failed; continuing with the rest of the batch",
                    extra={"event": "candidate_error", "symbol": candidate.symbol},
                )
                continue
            if item is not None:
                analysed.append(item)
        analysed.sort(key=lambda item: item.conviction, reverse=True)
        # In a drawdown, go less far down the ranked list. The candidates are
        # already ordered by conviction, so this says "only the best few" —
        # scale-free, and it always leaves at least one reachable. It never
        # touches position size.
        allowed = self.posture.max_candidates
        if allowed is not None and len(analysed) > allowed:
            log.info(
                "%s posture: acting on the best %d of %d setups",
                self.posture.posture.value,
                allowed,
                len(analysed),
                extra={
                    "event": "posture_candidate_limit",
                    "posture": self.posture.posture.value,
                    "allowed": allowed,
                    "dropped": len(analysed) - allowed,
                },
            )
            analysed = analysed[:allowed]
        if analysed:
            log.info(
                "ranked %d tradeable setups by conviction",
                len(analysed),
                extra={
                    "event": "conviction_ranking",
                    "candidates": len(analysed),
                    "best": [
                        f"{item.symbol} {item.idea.direction.name if item.idea.direction else '?'}"
                        f" {item.conviction:.1f}"
                        for item in analysed[:5]
                    ],
                },
            )
        return analysed

    def _analyse_candidate(self, symbol: str, account) -> AnalysedCandidate | None:  # type: ignore[no-untyped-def]
        cycle_id = str(uuid.uuid4())
        try:
            context = self.data.get_context(symbol, force_refresh=True)
            idea = self.engine.evaluate(context, self.settings.mode)
            if self.shadow_engine is not None:
                candidate = self.shadow_engine.evaluate(context, TradingMode.PAPER)
                self.shadow.record(symbol, idea, candidate, self.clock.now())
        except (TradingSystemError, ValueError) as exc:
            self._record_skip(cycle_id, symbol, account.equity, Reason.DATA_UNAVAILABLE, str(exc))
            return None

        # The other theories get their own look at the same chart. The swing
        # engine reads H1 structure; these read M5 impulse and M15 range, and
        # they carry their own stop and target because a five-minute plan and
        # an hourly one are different trades, not the same trade at different
        # strengths.
        verdict = self._playbook_verdict(context)
        if verdict is not None and verdict.conflict and self.playbook_config.veto_on_conflict:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.NO_SIGNAL,
                verdict.note,
                signals=list(idea.signals),
                extra={"playbooks": verdict.summary()},
            )
            return None

        if not idea.approved or idea.direction is None:
            # The swing engine saw nothing — but a short-horizon theory may
            # have. This is the whole point of running them: a clean M5 impulse
            # with a 12-pip stop was previously invisible, and on a EUR 100
            # account it is frequently the only plan whose arithmetic even works.
            promoted = self._play_as_idea(verdict, context)
            if promoted is not None:
                idea = promoted
            else:
                self._record_skip(
                    cycle_id,
                    symbol,
                    account.equity,
                    Reason.NO_SIGNAL,
                    idea.reason,
                    signals=list(idea.signals),
                    extra={"playbooks": verdict.summary()} if verdict else None,
                )
                return None

        # Do the independent methods agree? Until now the playbooks could only
        # veto each other, so a swing engine reading H1 structure as LONG while
        # an M5 impulse theory read the same chart as SHORT sailed straight
        # through — the one case where two techniques genuinely contradict each
        # other, and the clearest evidence available that the read is
        # ambiguous. Standing aside costs a trade that was a coin flip; taking
        # it costs the spread plus the stop, more often than not.
        disagreement = self._method_disagreement(idea, verdict)
        if disagreement is not None:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.METHODS_DISAGREE,
                disagreement,
                signals=list(idea.signals),
                extra={"playbooks": verdict.summary()} if verdict else None,
            )
            return None

        # Before anything else costs money or time: has this exact proposal
        # already been refused? The gate sits here, ahead of the risk, filter
        # and sizing work, because none of that changes the answer — a setup
        # Claude declined is not going to be talked into by a fresh margin
        # calculation, and re-running it only produces another identical row in
        # the dashboard's AI ledger.
        remembered = self._remembered_veto(idea)
        if remembered is not None:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.AI_VETO,
                remembered.describe(self.clock.now()),
                signals=list(idea.signals),
                extra={
                    "ai_veto_remembered": True,
                    "ai_veto_repeats": remembered.repeats,
                    "ai_confidence": remembered.confidence,
                },
            )
            return None
        return AnalysedCandidate(symbol, cycle_id, idea, context)

    def _process_candidate(  # type: ignore[no-untyped-def]
        self,
        candidate: AnalysedCandidate,
        account,
        positions,
        *,
        rank: int = 1,
        of: int = 1,
    ) -> bool:
        symbol = candidate.symbol
        cycle_id = candidate.cycle_id
        idea = candidate.idea
        context = candidate.context
        assert idea.direction is not None  # _analyse_candidate rejects a None direction

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

        # Can this trade afford its own spread? Asked before sizing, because it
        # is the cheapest gate here and it fails most often in the evening.
        #
        # The spread filter above asks a different question — is the spread
        # unusual for this instrument at this hour — and after 21:00 the answer
        # is no, it is perfectly normal, because the baseline it learned is
        # itself an evening baseline. Meanwhile the stop did not widen. A 2-pip
        # spread against a 6-pip stop means the trade opens a third of the way
        # to being wrong and must clear the spread twice to earn anything, and
        # no amount of edge in the setup survives that. The playbooks already
        # refused on this; the confluence path did not, which is where the
        # evening stop-outs were coming from.
        affordable, share = self._spread_is_affordable(context, idea.entry, idea.stop_loss)
        if not affordable:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.SPREAD_EATS_THE_STOP,
                f"spread is {share:.0%} of the {abs(idea.entry - idea.stop_loss):.5g} stop, "
                f"above the {self.settings.analysis.confluence.max_spread_share_of_stop:.0%} limit",
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
        # What the reviewer cannot see from one chart: where this setup placed
        # among everything analysed this cycle, and how the account is carrying
        # itself. Both change the answer legitimately — the best of 187 deserves
        # a different reading from the 40th, and a drawdown is a reason to want
        # more from a setup, never a reason to want it bigger.
        briefing: dict[str, object] = {
            "standing_this_cycle": {
                "rank": rank,
                "of_tradeable_setups": of,
                "conviction": round(candidate.conviction, 1),
                "note": (
                    "Rank 1 means the engine rated this the strongest setup it found across "
                    "the whole catalogue this cycle. That is a reason to read it carefully, "
                    "not a reason to approve it — the best of a weak field is still weak."
                ),
            },
            "account_posture": self.posture.brief(),
        }
        if self.memory.has_evidence():
            briefing["learned_so_far"] = self.memory.briefing(symbol, idea.direction.name)
        # Every theory's reading of this chart, including the ones that did not
        # win. What the losing theories saw is evidence the reviewer cannot get
        # anywhere else, and it is the part most likely to change the answer.
        playbooks = self._playbook_verdict(context)
        if playbooks is not None and playbooks.plays:
            briefing["other_theories"] = playbooks.summary()
        request_payload = build_review_payload(idea, context, proposal, briefing)
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
                else self._reviewed(idea, context, proposal, briefing)
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
            # Remember it, so the next cycle does not buy this answer again.
            # Only a real verdict is worth remembering: a transport failure is
            # a veto by the fail-closed rule, but it says nothing about the
            # setup, and silencing a symbol for ninety minutes because of one
            # timeout would let a brief outage blank out the catalogue.
            if not advice.error:
                self._remember_veto(idea, context, advice)
                self.memory.record_veto(symbol, idea.direction.name, self.clock.now())
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
        # Approved: whatever was held against this symbol no longer stands.
        self.veto_memory.clear(symbol, idea.direction.name)

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
        # Write the plan down before sending it. Between `order_send` returning
        # and the journal row landing there was a window in which a crash left a
        # real position the journal had never heard of — and on restart
        # reconciliation reads exactly that as an orphan and closes it. A
        # correctly sized, AI-approved trade could be destroyed by a power cut,
        # with the loss booked and no record of why the position existed.
        #
        # With the intent on disk first, that same position can be matched back
        # to the plan that created it. See `PositionManager.reconcile`.
        trade_id = self.recorder.record_entry_intent(
            cycle_pk=cycle_pk,
            sizing=sizing,
            equity_before=account.equity,
        )
        result = self.broker.order_send(request, spec)
        if not result.ok:
            self.journal.abandon_pending_entry(
                trade_id, f"entry rejected: {result.retcode_name} {result.comment}"
            )
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
        self.journal.promote_pending_entry(
            trade_id,
            ticket=result.position_ticket,
            entry_price=result.filled_price,
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
        total_score: float | None = None,
    ) -> int:
        # The score belongs on the skip row, not only on the trade row. Without
        # it "why is nothing trading" cannot be answered from the journal: the
        # reason column says NO_SIGNAL and the number that would show whether
        # the threshold is merely strict or outright unreachable is missing.
        # The diagnostic read the empty column and reported "no setup scored
        # above zero at all" while the detail text on the same rows said
        # "confluence score 41.9 below threshold".
        cycle_pk = self.recorder.record_cycle(
            cycle_id=cycle_id,
            context=CycleContext(symbol, equity, extra=extra),
            reason=reason,
            detail=detail,
            total_score=total_score,
            score_threshold=self.settings.analysis.confluence.score_threshold,
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

    def _supervise_positions(self, positions) -> None:  # type: ignore[no-untyped-def]
        """Let the adviser manage what is already open, on a bounded cadence.

        The account previously had a strategist and no manager. Something
        decided what to open, and from that moment the position was handed to
        three fixed numbers — break even at 1R, partial at 2R, trail by ATR —
        which cannot tell a trend that is still intact from one that broke
        twenty minutes ago. Both look identical to a rule that only reads the
        R multiple.

        Rate-limited rather than run every loop, and the reason is not only
        cost. At a thirty-second interval an adviser asked continuously will
        eventually talk itself into acting on noise, and each intervention
        costs real spread. Once every `supervision_interval_minutes` is roughly
        how often a human glances at an open trade, which is the behaviour
        being modelled.
        """
        if isinstance(self.advisor, DisabledAdvisor) or self.operation is OperationMode.MONITOR:
            return
        interval = self.settings.trade_management.supervision_interval_minutes
        if interval <= 0:
            return
        now = self.clock.now()
        live = {position.ticket for position in positions}
        # Forget tickets that have closed, or the map grows for the life of the
        # process and starts throttling a ticket the broker has reused.
        self._supervised_at = {
            ticket: when for ticket, when in self._supervised_at.items() if ticket in live
        }
        for position in positions:
            last = self._supervised_at.get(position.ticket)
            if last is not None and (now - last).total_seconds() < interval * 60:
                continue
            try:
                context = self.data.get_context(position.symbol)
            except (TradingSystemError, ValueError) as exc:
                log.warning(
                    "no context for supervision; mechanical rules continue",
                    extra={
                        "event": "supervision_no_context",
                        "ticket": position.ticket,
                        "symbol": position.symbol,
                        "reason": str(exc),
                    },
                )
                continue
            payload = build_supervision_payload(
                position,
                context,
                {
                    "operation": self.operation.value,
                    "account_currency": self.broker.account().currency,
                    "account_posture": self.posture.brief(),
                    "learned_so_far": self.memory.briefing(
                        position.symbol, position.direction.name
                    ),
                    # What the per-second layer has been seeing. Without this
                    # the adviser judges a snapshot of the moment it happened
                    # to be asked and cannot know the trade has been bleeding
                    # for ten minutes — the fast layer watched all of it.
                    "mechanical_health": self._health_brief(position.ticket),
                },
            )
            self._supervised_at[position.ticket] = now
            started = time.monotonic()
            verdict = self.advisor.supervise(payload)
            latency_ms = round((time.monotonic() - started) * 1000, 1)
            try:
                self.ai_ledger.append(
                    "position_supervision",
                    {
                        "ticket": position.ticket,
                        "symbol": position.symbol,
                        "direction": position.direction.name,
                        "latency_ms": latency_ms,
                        "request": payload,
                        "decision": verdict.safe_dict(),
                    },
                )
            except OSError:
                log.exception(
                    "failed to persist supervision audit",
                    extra={"event": "supervision_audit_failed", "ticket": position.ticket},
                )
                # No audit, no action. Every other decision in this system is
                # reconstructable from the ledger and this one must be too.
                continue
            if verdict.action == "hold":
                continue
            event = self.manager.apply_supervision(position, verdict)
            if event is not None:
                self._record_management([event])

    def _publish_health(self, positions) -> None:  # type: ignore[no-untyped-def]
        """Write the current read to disk for the deck to pick up.

        The manager keeps this in memory and the dashboard is a different
        process, so without a file the operator's answer to "what does the
        system think of my open trade right now" is a fifteen-minute-old
        supervisor entry. Best-effort: a failed write costs a panel, and
        nothing here is state anyone recovers from.
        """
        payload = {
            "recorded_at": self.clock.now().isoformat(),
            "positions": [
                {
                    "ticket": position.ticket,
                    "symbol": position.symbol,
                    "direction": position.direction.name,
                    **self._health_brief(position.ticket),
                }
                for position in positions
            ],
        }
        write_json_atomic(self.health_file, payload)

    def _spread_is_affordable(
        self, context: MarketContext, entry: float, stop: float
    ) -> tuple[bool, float]:
        """Is the current spread a tolerable share of this trade's stop?

        Fails closed on a missing tick or a zero-width stop: both mean the
        question cannot be answered, and an unanswerable cost question is not a
        reason to pay it.
        """
        risk = abs(entry - stop)
        if context.tick is None or risk <= 0:
            return False, 1.0
        share = context.tick.spread / risk
        return share <= self.settings.analysis.confluence.max_spread_share_of_stop, share

    def _health_brief(self, ticket: int) -> dict[str, object]:
        """The fast layer's current read, for the adviser's payload."""
        health = self.manager.last_health.get(ticket)
        return health.summary() if health is not None else dict(_UNKNOWN_HEALTH)

    def _build_playbooks(self) -> PlaybookEngine | None:
        """Assemble the short-horizon theories, or None when they are off."""
        config = self.playbook_config
        if not config.enabled:
            return None
        chosen: list[Playbook] = []
        if config.momentum_scalp:
            chosen.append(
                MomentumScalp(ScalpConfig(max_spread_share_of_stop=config.max_spread_share_of_stop))
            )
        if config.range_fade:
            chosen.append(
                RangeFade(FadeConfig(max_spread_share_of_stop=config.max_spread_share_of_stop))
            )
        if not chosen:
            return None
        log.info(
            "short-horizon playbooks active: %s",
            ", ".join(playbook.name for playbook in chosen),
            extra={"event": "playbooks_enabled", "playbooks": [p.name for p in chosen]},
        )
        return PlaybookEngine(chosen, self.settings.analysis.confluence)

    def _playbook_verdict(self, context: MarketContext) -> PlaybookVerdict | None:
        """What every short-horizon theory saw, or None when they are off."""
        if self.playbooks is None:
            return None
        try:
            return self.playbooks.evaluate(context, self.settings.mode)
        except Exception:
            log.exception(
                "playbook evaluation failed",
                extra={"event": "playbook_error", "symbol": context.symbol},
            )
            return None

    def _method_disagreement(self, idea: TradeIdea, verdict: PlaybookVerdict | None) -> str | None:
        """Does an independent theory read this chart the opposite way?

        Returns the sentence for the journal, or None when nothing contradicts
        the idea. Only *opposing* plays count: a theory that stayed silent has
        no opinion, and silence is not disagreement — most markets have no
        short-horizon setup at any given moment, and treating that as a veto
        would refuse nearly everything.

        A play below the conviction floor is also ignored. It was not good
        enough to trade in its own right, so it is not good enough to cancel
        somebody else's trade either.
        """
        if verdict is None or not self.playbook_config.require_method_agreement:
            return None
        floor = self.playbook_config.min_conviction
        opposing = [
            play
            for play in verdict.plays
            if play.direction is not idea.direction and play.conviction >= floor
        ]
        if not opposing:
            return None
        names = ", ".join(f"{play.playbook} says {play.direction.name}" for play in opposing)
        return (
            f"methods disagree: the swing engine says {idea.direction.name} while {names}. "
            "Two techniques reading one chart in opposite directions is not a close call to "
            "settle by score — it is a market with no clear edge."
        )

    def _play_as_idea(
        self, verdict: PlaybookVerdict | None, context: MarketContext
    ) -> TradeIdea | None:
        """Turn the winning short-horizon play into a tradeable idea.

        Everything downstream — risk gates, filters, the sizer, the margin
        check, the AI review, the order — is unchanged and runs in full. The
        play only supplies the direction, entry, stop and target; it buys no
        exemption from anything.

        The conviction floor is deliberately higher than the swing engine's
        threshold. A short-horizon trade pays spread against a small stop, so
        the marginal ones are not worth taking even when the pattern is real.
        """
        if verdict is None or verdict.best is None:
            return None
        play = verdict.best
        if play.conviction < self.playbook_config.min_conviction:
            return None
        return TradeIdea(
            symbol=context.symbol,
            approved=True,
            direction=play.direction,
            score=play.conviction,
            # The playbook's own confidence is folded into its conviction, so a
            # second discount here would double-count it.
            confidence=min(1.0, play.conviction / 100.0),
            entry=play.entry,
            stop_loss=play.stop_loss,
            take_profit=play.take_profit,
            reason=f"{play.playbook}: {play.thesis}",
            signals=(),
        )

    def _remembered_veto(self, idea: TradeIdea):  # type: ignore[no-untyped-def]
        """The standing refusal covering this proposal, if there is one.

        Takes no context: the lookup is by the proposal's own shape, and the
        ATR scale it is compared against was recorded with the refusal rather
        than recomputed now. Using today's ATR would let a volatility spike
        silently widen the tolerance.
        """
        if idea.direction is None or self.operation is OperationMode.MONITOR:
            return None
        return self.veto_memory.recall(
            idea.symbol,
            idea.direction.name,
            idea.entry,
            idea.stop_loss,
            self.clock.now(),
        )

    def _remember_veto(self, idea: TradeIdea, context: MarketContext, advice: Advice) -> None:
        """File a refusal against the shape of the proposal that earned it."""
        if idea.direction is None:
            return
        self.veto_memory.remember(
            idea.symbol,
            idea.direction.name,
            entry=idea.entry,
            stop=idea.stop_loss,
            atr=self._signal_atr(context),
            thesis=advice.thesis,
            confidence=advice.confidence,
            now=self.clock.now(),
        )

    @staticmethod
    def _signal_atr(context: MarketContext) -> float:
        """ATR of the signal timeframe, the yardstick for "materially changed".

        Zero when the frame is missing rather than a guessed constant: the
        memory treats a zero as "compare exactly", which errs toward asking
        again. Erring the other way would silence a symbol on a scale nobody
        chose.
        """
        series = context.series.get(_REVIEW_TIMEFRAME)
        if series is None or len(series.df) < 15:
            return 0.0
        frame = series.df
        previous = frame["close"].shift(1)
        ranges = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous).abs(),
                (frame["low"] - previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        value = ranges.tail(14).mean()
        return 0.0 if pd.isna(value) else float(value)

    def _reviewed(
        self,
        idea: TradeIdea,
        context: MarketContext,
        proposal: dict[str, object],
        memory: dict[str, object] | None = None,
    ) -> Advice:
        """Ask Claude, unless the identical setup was already answered.

        The scanner ranks the catalogue the same way every cycle, so the same
        instrument tends to come top repeatedly. Without a memory that meant
        asking about a setup that had not changed once every loop interval: one
        live session sent EURCAD 74 times in twenty minutes, thirty seconds
        apart, and was vetoed 74 times for the same stated reason. Each of those
        was a paid call of eight to thirteen seconds spent re-deriving an answer
        already on file.

        The verdict is a function of the evidence, and the evidence is closed
        bars. So the key is the symbol, the direction, and the close time of the
        fastest analysed timeframe: while no new bar has closed there is
        genuinely nothing new to judge, and the moment one does the question is
        legitimately different and gets asked again.

        Deliberately not keyed on the tick. Price moves continuously and a
        review that expired on every tick would expire always, which is the
        behaviour being fixed. A stale entry is not a risk here either — the
        deterministic gates re-run in full on every cycle regardless, and this
        only ever replays a veto or an approval of an unchanged setup.
        """
        key = self._review_key(idea, context)
        if key is not None:
            cached = self._review_cache.get(key)
            if cached is not None:
                log.info(
                    "reusing the AI verdict for an unchanged setup",
                    extra={
                        "event": "ai_review_reused",
                        "symbol": idea.symbol,
                        "direction": idea.direction.name if idea.direction else None,
                        "approved": cached.approved,
                    },
                )
                return cached

        advice = self.advisor.review(idea, context, proposal, memory)
        if key is not None:
            if len(self._review_cache) >= _REVIEW_CACHE_ENTRIES:
                # Oldest first; insertion order is chronological.
                del self._review_cache[next(iter(self._review_cache))]
            self._review_cache[key] = advice
        return advice

    def _review_key(
        self, idea: TradeIdea, context: MarketContext
    ) -> tuple[str, str, datetime] | None:
        """Symbol, direction, and the close of the timeframe the signal lives on.

        Keyed on the *fastest* timeframe first, which was wrong: with M1 in the
        ladder a new bar closes every sixty seconds, so the cache expired every
        minute and the same instrument went back to Claude four times in three
        minutes with the same verdict. The one-minute bar is entry-timing
        context; it is not what the setup is made of.

        The signal timeframe is. While no new bar has closed on it the setup is
        the same setup, and re-asking buys a re-worded copy of an answer already
        held.
        """
        if idea.direction is None or not context.series:
            return None
        series = context.series.get(_REVIEW_TIMEFRAME)
        if series is None:
            series = context.series[min(context.series, key=lambda tf: tf.duration)]
        return (idea.symbol, idea.direction.name, series.last_bar_time)

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
        risk_money = float(row["risk_money"])
        # The realised result goes into the memory whether or not an adviser is
        # configured. It is the account's own arithmetic, not an opinion, and
        # it is the part of the record that matters most: switching the adviser
        # off should not blind the system to its own P&L.
        self.memory.record_outcome(
            str(row["symbol"]),
            str(row["direction"]),
            pnl_money / risk_money if risk_money > 0 else 0.0,
            self.clock.now(),
        )
        if isinstance(self.advisor, DisabledAdvisor):
            return
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
        # This is the step that was missing. The reflection used to be written
        # to a file nothing read, so every lesson was paid for once and then
        # discarded; folding it into the memory is what makes the next review
        # start from more than zero.
        if not reflection.error and reflection.lessons:
            self.memory.record_reflection(outcome, reflection.lessons, self.clock.now())
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
        # What happened to the ones that got the *deep* analysis. Without this
        # the line said "62 analysed, 0 opened" and stopped, which is the one
        # question an operator has and the one it did not answer. Five hours of
        # that reads exactly like a broken system and exactly like a correct
        # one, and the two need opposite responses.
        deep_reasons = self._deep_rejection_counts(summary.started_at)
        deep_breakdown = ", ".join(
            f"{n} {reason}" for reason, n in sorted(deep_reasons.items(), key=lambda kv: -kv[1])[:4]
        )
        log.info(
            "cycle: %d/%d scanned, %d eligible, %d analysed, %d opened, %.0fs%s%s",
            summary.inspected,
            summary.universe_size,
            summary.inspected - summary.rejected,
            summary.deep_analysed,
            summary.trades_opened,
            elapsed,
            f" | prescan: {breakdown}" if breakdown else "",
            f" | analysed: {deep_breakdown}" if deep_breakdown else "",
            extra={
                "event": "cycle_complete",
                "inspected": summary.inspected,
                "universe_size": summary.universe_size,
                "eligible": summary.inspected - summary.rejected,
                "rejected": summary.rejected,
                "rejected_by_stage": stages,
                "deep_analysed": summary.deep_analysed,
                "deep_rejections": deep_reasons,
                "trades_opened": summary.trades_opened,
                "seconds": round(elapsed, 1),
                "next_cursor": summary.next_cursor,
            },
        )

    def _deep_rejection_counts(self, since: datetime) -> dict[str, int]:
        """Reasons the deeply-analysed candidates were refused, this cycle.

        Read back from the journal rather than accumulated in memory, because
        the journal is where the decisions actually land and a counter that
        disagreed with it would be worse than none.
        """
        try:
            rows = self.journal.query(
                "SELECT reason, COUNT(*) AS n FROM analysis_cycles "
                "WHERE ts >= ? AND decision != 'TRADE' GROUP BY reason",
                (since.isoformat(),),
            )
        except Exception:  # noqa: BLE001 - a reporting query must never end a cycle
            return {}
        return {str(row["reason"]): int(row["n"]) for row in rows}

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
                    "posture": self.posture.brief(),
                    "blocked_reason": self.blocked_reason,
                    "blocked_detail": self.blocked_detail,
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

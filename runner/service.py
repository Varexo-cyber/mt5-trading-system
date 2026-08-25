"""Long-running orchestration: scan, analyse, filter, size, execute, reconcile."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from advisory import (
    Advice,
    Advisor,
    AIReviewLedger,
    DisabledAdvisor,
    ScoutDecision,
    VetoMemory,
    VetoPatterns,
    build_advisor,
    build_review_payload,
    build_supervision_payload,
    read_trade_reflections,
)
from advisory.local_history import entry_features, supervision_features
from advisory.scout import ScoutThrottle
from advisory.veto_patterns import readable as veto_readable
from analysis import (
    BASKET_META_KEY,
    BasketDivergence,
    CandleMomentum,
    ConfluenceEngine,
    DriftBurst,
    DriftContinuation,
    EmaPullbackResume,
    EntryTimingDecision,
    FastEmaCross,
    ImpulseBreak,
    LevelReaction,
    LifecycleDecision,
    LiquiditySweep,
    M1MicroBreakout,
    MarketObservation,
    MarketRegime,
    MarketStructure,
    MeanReversion,
    OpportunityIntelligence,
    PeerMove,
    Seasonality,
    SessionBreakout,
    SetupLifecycleBook,
    SetupState,
    TrendMomentum,
    VolatilityRegime,
    VolatilitySqueeze,
    apply_cross_market_context,
    assess_entry_quality,
    assess_opportunity,
    assess_review_drift,
    observe_market,
    scout_market_snapshot,
    supporting_families,
)
from analysis import (
    world_state as build_world_state,
)
from analysis.confluence import TradeIdea
from analysis.playbooks import (
    BreakConfig,
    FadeConfig,
    FailedBreak,
    FailedBreakConfig,
    MomentumScalp,
    Playbook,
    PlaybookEngine,
    PlaybookVerdict,
    PullbackConfig,
    RangeBreak,
    RangeFade,
    ScalpConfig,
    TrendPullback,
)
from analysis.target_reach import measure as measure_target_reach
from analysis.target_reach import measure_first_touch
from brain import SelectionEvidence, build_brain
from brain.selection import SelectionVerdict, combine_selection_evidence
from config.schema import Settings
from core.broker import Broker
from core.clock import Clock, LiveClock
from core.data_manager import DataManager, atr
from core.data_quarantine import DataQuarantine
from core.errors import DataIntegrityError, InsufficientDataError, TradingSystemError
from core.instrument import InstrumentSpec
from core.startup import run_startup_guard
from core.types import (
    AccountSnapshot,
    Direction,
    MarketContext,
    OrderRequest,
    OrderResult,
    Position,
    Signal,
    Timeframe,
    TradingMode,
)
from execution.manager import ManagementEvent, PositionManager
from execution.paper_broker import PaperBroker
from external_signals import (
    ExternalSignalEvent,
    ExternalSignalInbox,
    ExternalSignalKind,
    RioSignalParser,
    SignalReceiver,
    build_gold_follow_plan,
)
from filters.base import FilterContext, FilterVerdict
from filters.calendar.events import symbol_currencies
from filters.headline_filter import HeadlineFilter
from filters.news_filter import NewsFilter
from filters.runway_filter import RunwayFilter
from filters.session_filter import SessionFilter
from infra.atomic import write_json_atomic
from infra.killswitch import KillSwitch
from infra.logging import get_logger
from journal.database import Journal
from journal.recorder import CycleContext, Recorder
from learning.config_control import ShadowRecorder
from learning.counterfactual import resolve_counterfactuals, resolve_management_baselines
from learning.memory import TradingMemory
from main import build_filter_chain
from monitoring.alerts import AlertSender
from monitoring.operation_ledger import LEDGER_FILENAME, OperationLedger
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
from risk.position_sizer import PositionSizer, SizingResult
from risk.posture import PostureAssessment, assess
from risk.reasons import Reason, RiskDecision
from risk.risk_manager import RiskManager
from risk.section_breaker import BreakerVerdict, tripped_modules
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
    "reason": "",
    "signals": [],
}

RIO_POSITION_COMMENT = "jarvis-rio"


#: Order comment that marks a position as section six's. The lane counts its
#: own book by this rather than by a separate ledger, so a restart or a manual
#: close cannot put the two out of step.
_SCALP_COMMENT = "jarvis-scalp"


def _is_rio_position(position: Position) -> bool:
    """True only for positions opened by the authenticated Rio route."""
    return str(getattr(position, "comment", "")).casefold() == RIO_POSITION_COMMENT


#: How far inside the cost limit a widened stop is placed, as a multiplier on
#: the limit itself. Solving for exactly the limit and stopping there loses to
#: the float: the widened stop is normalised to the instrument's tick and comes
#: back a hair short, and the gate refuses it anyway.
_COST_MARGIN = 0.95

#: Fallback review timeframe for legacy ideas or incomplete test contexts. New
#: ideas key their verdict to their own planning timeframe.
_REVIEW_TIMEFRAME = Timeframe.H1


@dataclass(frozen=True, slots=True)
class _SupervisionSnapshot:
    """The material state at the last paid open-position review."""

    r_now: float
    peak_r: float
    giveback_fraction: float
    health_verdict: str
    health_severity: float
    #: Unrealised profit as a share of account equity. Carried alongside `r_now`
    #: because they are not the same question on a small account: a wide
    #: structural stop makes real money look like a small R, and every trigger
    #: below used to be written in R alone. The live CADCHF long held 2.2% of
    #: the account at 0.44R, so the 0.25R milestone ladder had last spoken at
    #: EUR 1.60 and would not speak again until EUR 3.20.
    profit_pct_of_equity: float = 0.0
    #: The BEST this trade has been worth, as a share of equity. The field above
    #: is what it is worth now, and on the way down those diverge — which is the
    #: entire case this exists for. Both milestone ladders only speak on a new
    #: high, so a trade that peaks and then drains is silent in every trigger
    #: except the give-back, and the give-back was gated on R alone.
    peak_pct_of_equity: float = 0.0
    #: Whether the entry engine's latest read on this chart opposes the position.
    #: Carried between reviews so the trigger can fire on the CHANGE rather than
    #: on the state — being on the wrong side is one event, not a condition to
    #: re-ask about every two minutes.
    engine_against: bool = False


@dataclass(frozen=True, slots=True)
class _RevalidatedEntry:
    """A paid approval rebound to fresh executable market and account state."""

    account: AccountSnapshot
    positions: tuple[Position, ...]
    context: MarketContext
    idea: TradeIdea
    sizing: SizingResult
    filter_data: dict[str, object]
    review_binding: dict[str, object]


@dataclass(frozen=True, slots=True)
class _EntryRevalidation:
    plan: _RevalidatedEntry | None
    reason: Reason
    detail: str
    extra: dict[str, object]

    @property
    def passed(self) -> bool:
        return self.plan is not None


#: How often the buffered position path is handed to the long memory, and
#: how many samples may queue before it goes early. Sixty seconds because
#: the guard samples about once a second and one round trip a minute is
#: invisible next to a network hop inside every pass; the row cap is what
#: stops four busy positions waiting a full minute.
_POSITION_PATH_FLUSH_SECONDS = 60.0
_POSITION_PATH_MAX_BUFFER = 240

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
    intelligence: OpportunityIntelligence | None = None
    #: What the short-horizon theories saw on the same chart, carried so the
    #: execution phase can record it beside whatever refused the entry.
    #:
    #: Without this the two verdicts land on different journal rows — whichever
    #: gate fires first writes the row — and the one question worth asking
    #: about them cannot be asked at all: how often does a playbook call a
    #: setup tradeable while `entry_quality` calls the same bars a chase.
    #: `momentum_scalp` wants a shallow pullback off an impulse and
    #: `entry_quality` refuses prices sitting at a range extreme, and whether
    #: those two descriptions collide in practice is a measurement nobody could
    #: take.
    playbooks: object | None = None
    market_priority_tier: int = 0
    spread_quality: float = 0.0
    cost_priority: float = 0.0
    #: How much of this setup's own reward survives its own round trip. See
    #: `JarvisRunner._after_cost_priority`.
    after_cost_priority: float = 0.0
    #: Raw scores establish eligibility upstream. Once eligible, their role in
    #: ordering is deliberately reduced so realised evidence and independent
    #: evidence families can materially change which setup is considered first.
    raw_conviction_weight: float = 1.0
    independent_family_bonus: float = 0.0

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

    @property
    def ranking_score(self) -> float:
        """Conviction plus bounded context, used only to decide who goes first."""
        modifier = (
            self.intelligence.modifier
            + self.intelligence.scout_alignment
            + self.intelligence.learned_alignment
            + self.intelligence.recent_refusals
            if self.intelligence is not None
            else 0.0
        )
        return (
            self.conviction * self.raw_conviction_weight + self.independent_family_bonus + modifier
        )

    @property
    def selection_score(self) -> float:
        """Evidence score plus what the trade keeps after paying to exist.

        `cost_priority` prefers a cheap instrument; `after_cost_priority`
        prefers a setup whose target is far enough away to be worth the toll.
        Those are different questions, and only the second one looks at the
        trade actually being proposed: an identical two-pip spread is a
        rounding error against a forty-pip target and half the winnings
        against a six-pip one.
        """
        return self.ranking_score + self.cost_priority + self.after_cost_priority

    @property
    def selection_key(self) -> tuple[int, float]:
        """Strict market lane first, then evidence and cost inside that lane."""
        return self.market_priority_tier, self.selection_score

    @property
    def market_priority_label(self) -> str:
        return {2: "core_market", 1: "preferred_asset_class"}.get(
            self.market_priority_tier, "catalogue_fallback"
        )


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


def build_analysis_modules(settings: Settings) -> list[object]:
    """Every reader the confluence engine votes with, in one place.

    THIS LIVED INLINE IN `JarvisRunner.__init__` AND THAT IS HOW SECTION SIX
    REACHED PRODUCTION BROKEN. `MomentumScalp` is the name of a playbook as
    well as of a detector, the playbook import came later in the file and won,
    and the engine was handed a playbook object with no `analyze` method. Every
    candidate of every cycle then failed analysis — not section six's
    candidates, ALL of them — and the account traded nothing while the log said
    "continuing with the rest of the batch".

    Nothing caught it because no test ever built this list. It is a function so
    that one can, and `tests/test_analysis_modules.py` now does: every reader
    answers to `analyze`, no two answer to the same name, and every name the
    config gives weight to is a reader that actually exists.
    """
    analysis = settings.analysis
    return [
        MarketStructure(analysis.market_structure),
        TrendMomentum(analysis.trend_momentum),
        LiquiditySweep(analysis.liquidity_sweep),
        LevelReaction(analysis.level_reaction),
        VolatilityRegime(analysis.volatility_regime),
        MarketRegime(analysis.market_regime),
        DriftContinuation(analysis.drift_continuation),
        FastEmaCross(analysis.fast_ema_cross),
        ImpulseBreak(analysis.impulse_break),
        EmaPullbackResume(analysis.ema_pullback_resume),
        M1MicroBreakout(analysis.m1_micro_breakout),
        # Constructed but not on `live_enabled_modules`, so live mode zeroes
        # their weight and they cannot move a trade. They are here so the
        # module backtest measures the same readers the account runs, rather
        # than a different set assembled in a script — which is how two copies
        # of a thing that must agree start to disagree.
        VolatilitySqueeze(analysis.volatility_squeeze),
        MeanReversion(analysis.mean_reversion),
        SessionBreakout(analysis.session_breakout),
        Seasonality(analysis.seasonality),
        # SECTION TWO, and the first reader here that is not looking at the
        # shape of the price series. It runs a hypothesis test and it FADES,
        # which is what makes it the only candidate that can ever be a genuine
        # second family for the nine that follow.
        DriftBurst(analysis.drift_burst),
        # SECTION FIVE, and the first reader that is MEANINGLESS on a single
        # chart. It measures a move between two M1 closes, which has no open
        # question about surviving the coarsening.
        BasketDivergence(analysis.basket_divergence),
        # SECTION SIX. The fast-bot shape, with the candle demoted from thesis
        # to trigger: M15 decides the side, M5 must not contradict, and the M1
        # close is when that agreement becomes actionable.
        #
        # NOT called `momentum_scalp`, deliberately. That name belongs to a
        # playbook with a measured record of -0.561R over 307 trades, switched
        # off on purpose. Reusing it would have put a new reader live under the
        # name of a known loser, in the same config file that documents why the
        # loser is off.
        CandleMomentum(analysis.candle_momentum),
    ]


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
        self.clock: Clock = clock or LiveClock()
        self.kill_switch = KillSwitch.in_dir(
            root,
            self.settings.system.kill_switch_file,
            self.clock,
        )
        self.broker: Broker = (
            PaperBroker(
                market,
                root / "runtime" / "paper_state.json",
                clock=self.clock,
            )
            if operation is OperationMode.PAPER
            else market
        )
        # Injectable so a whole cycle can be exercised at a chosen moment. This
        # was the last hardcoded LiveClock, and while it stood no end-to-end
        # test could place itself on, say, a Monday morning — which is exactly
        # when the interesting failures happen.
        self.journal = Journal(
            root / self.settings.journal.database_path,
            self.clock,
            day_boundary_utc=self.settings.risk.day_boundary_utc,
        )
        self.data = DataManager(self.broker, self.settings.data, self.clock)
        self.engine = ConfluenceEngine(
            build_analysis_modules(self.settings), self.settings.analysis.confluence
        )
        #: symbol -> (asset class, recent move in bp, its last bar's time).
        #: Section five's peer readings, kept across the scan so comparing
        #: indices costs one pass over bars already fetched.
        self._basket_moves: dict[str, tuple[str, float, datetime]] = {}
        #: Section breakers, recomputed once per cycle rather than per candidate.
        self._breaker_cache: dict[str, BreakerVerdict] = {}
        self._breaker_cache_cycle = -1
        self._cycle_serial = 0
        self.shadow = ShadowRecorder(root, self.settings.learning)
        self.shadow_engine = self._shadow_engine()
        self.playbook_config = self.settings.analysis.playbooks
        self.playbooks = self._build_playbooks()
        self.advisor = advisor or build_advisor(
            self.settings.ai,
            history_path=root / "runtime" / "ai_reviews.jsonl",
        )
        self.ai_ledger = AIReviewLedger(
            root / "runtime" / "ai_reviews.jsonl",
            database=self.journal,
            clock=self.clock,
        )
        self.cursor = self._load_cursor()
        self.operation_ledger = OperationLedger(root / "runtime" / LEDGER_FILENAME)
        self.scan_activity = ScanActivityLedger(root / "runtime" / "scan_activity.json")
        self.experimental_contract: ExperimentalLiveContract | None = None
        # Proposal identity -> the verdict given. The identity includes the
        # idea's own planning timeframe and material price shape; see
        # `_review_key` for why a fixed H1 key was not enough for intraday plans.
        self._review_cache: dict[tuple[object, ...], Advice] = {}
        # Paid reviews spent in the cycle currently running; reset by run_once.
        self._reviews_this_cycle = 0
        # Refusals outlive the bar they were given on; see advisory/veto_memory.
        self.veto_memory = VetoMemory(
            root / "runtime" / "veto_memory.json",
            clock=self.clock,
        )
        # Why repeated refusals happened, independent of the proposal's exact shape.
        self.veto_patterns = VetoPatterns(root / "runtime" / "veto_patterns.json")
        # What the account has taught itself, fed back into every review.
        self.memory = TradingMemory(
            root / "runtime" / "trading_memory.json",
            clock=self.clock,
        )
        # One global closed-bar view per cycle. It is prompt/dashboard context,
        # never a source of entry permission.
        self.market_intelligence_file = root / "runtime" / "market_intelligence.json"
        entry = self.settings.analysis.entry_quality
        self.setup_lifecycle = SetupLifecycleBook(
            root / "runtime" / "setup_lifecycle.json",
            pullback_atr=entry.lifecycle_pullback_atr,
            resumption_atr=entry.lifecycle_resumption_atr,
            expiry_minutes=entry.lifecycle_expiry_minutes,
        )
        self.scout_throttle = ScoutThrottle(root / "runtime" / "market_scout_state.json")
        self._cycle_observations: list[MarketObservation] = []
        self._cycle_contexts: dict[str, MarketContext] = {}
        #: symbol -> (direction the engine last read, score). Survives the cycle
        #: reset below on purpose: an open position needs the most recent read
        #: available, not only one from the cycle currently running.
        self._latest_direction: dict[str, tuple[str, float]] = {}
        self._world_state: dict[str, object] = {}
        self._last_scout = ScoutDecision(thesis="No scout call recorded yet")
        self._counterfactuals_checked_at: datetime | None = None
        # The long half of the same memory. The JSON file above keeps forty
        # lessons on one machine and forgets past its retention window; this
        # keeps every decision, every guard action and every lesson for the
        # life of the account, and survives the VPS being rebuilt. It is
        # deliberately optional and deliberately fail-soft — no write here can
        # stop a trade, and `brain.store` sets out why at length. Returns a
        # `NullBrain` when no DSN is configured, which is the developer case.
        # Scoped by account number, so a demo and a live account writing to the
        # same database never pool their statistics into one misleading total.
        self.brain = build_brain(account=os.getenv("MT5_LOGIN", "") or self.settings.mode.value)
        # The nearest-neighbour adviser was reading its archive of past
        # position judgements out of a JSONL file on this machine — one that
        # starts empty after a fresh clone. It needs five comparable states
        # before it may act, so an empty file means it holds every position
        # indefinitely while the account's whole history sits in Postgres.
        attach = getattr(self.advisor, "attach_brain", None)
        if callable(attach):
            attach(self.brain)
        self._selection_evidence: list[SelectionEvidence] = []
        self._selection_evidence_at: datetime | None = None
        self._brain_schema_ready = False
        if self.brain.enabled:
            self._brain_schema_ready = self.brain.migrate()
            if not self._brain_schema_ready:
                log.warning(
                    "brain schema could not be applied; running without long-term memory",
                    extra={"event": "brain_migrate_failed", "why": self.brain.status.last_error},
                )
        # Broker ticket -> the brain's own trade row, so a guard action can be
        # attached to the right position without a lookup on every event.
        self._brain_trades: dict[int, int | None] = {}
        # The fast layer's live read, published for the deck. The manager holds
        # it in memory; the dashboard is a separate process and cannot see that.
        self.health_file = root / "runtime" / "position_health.json"
        # ticket -> when the supervisor last looked at it, so an open position
        # is reconsidered on a sane cadence rather than every thirty seconds.
        #: Guard-pass samples of every open position, waiting for the
        #: batched flush into Neon. See `_flush_position_path`.
        self._position_path: list[dict[str, object]] = []
        self._position_path_flushed_at: datetime | None = None
        self._supervised_at: dict[int, datetime] = {}
        self._supervision_due_at: dict[int, datetime] = {}
        self._supervision_snapshots: dict[int, _SupervisionSnapshot] = {}
        # Why new risk is refused, if it is. Empty means trading is permitted.
        self.blocked_reason = ""
        self.blocked_detail = ""
        # Full-catalogue work is deliberately single-threaded because MT5's
        # Python session is global and not thread-safe. This timestamp lets the
        # scan yield to the one-second protection layer between broker reads.
        self._last_guard_pulse = time.monotonic()
        # Recomputed every cycle; STEADY until the first one runs.
        self.posture: PostureAssessment = assess(consecutive_losses=0, equity=1.0, equity_peak=1.0)
        self.external_inbox = ExternalSignalInbox(root / self.settings.external_signals.inbox_path)
        self.external_parser = RioSignalParser()
        self.external_receiver: SignalReceiver | None = None

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
        self.ai_ledger.backfill_database()
        self.recorder = Recorder(self.journal, self.clock, self.settings)
        if self._brain_schema_ready:
            self._sync_counterfactual_history()
            self._sync_trade_history()
        self.memory.synchronize_outcomes(
            self.journal.query(
                "SELECT id, symbol, direction, closed_at, "
                "CASE WHEN pnl_r IS NOT NULL THEN pnl_r "
                "WHEN risk_money > 0 THEN COALESCE(pnl_money, 0.0) / risk_money "
                "ELSE 0.0 END AS pnl_r "
                "FROM trades WHERE closed_at IS NOT NULL "
                "AND COALESCE(entry_state, 'OPEN') != 'ABANDONED'"
            ),
            self.clock.now(),
        )
        self.memory.synchronize_reflections(
            read_trade_reflections(self.ai_ledger.path), self.clock.now()
        )
        self.risk = RiskManager(
            self.settings,
            self.journal,
            self.clock,
            self.kill_switch,
            margin_estimator=self.broker.estimate_margin,
            manageability_probe=self._market_is_manageable,
        )
        self.filters = build_filter_chain(self.broker, self.settings, self.journal, self.clock)
        self.scanner = UniverseScanner(self.broker, self.settings, self.clock)
        quarantine_config = self.settings.scanner.data_quarantine
        self.quarantine = DataQuarantine(
            enabled=quarantine_config.enabled,
            initial_minutes=quarantine_config.initial_minutes,
            backoff_multiple=quarantine_config.backoff_multiple,
            max_minutes=quarantine_config.max_minutes,
        )
        # The brain is handed over so the banking rule can consult what this
        # account's own closed trades say about when to take profit. It can
        # only ever lower the threshold — see `PositionManager._worth_taking`.
        self.manager = PositionManager(self.broker, self.journal, self.settings, self.brain)
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
        self._start_external_receiver()

    def close(self) -> None:
        if self.external_receiver is not None:
            self.external_receiver.stop()
        self._save_cursor()
        self.operation_ledger.finish(self.clock.now())
        self.journal.close()
        self.broker.shutdown()

    def _start_external_receiver(self) -> None:
        config = self.settings.external_signals
        if not config.enabled:
            return
        token = os.getenv(config.token_env, "").strip()
        if not token:
            self.external_inbox.record(
                "receiver",
                "BLOCKED",
                f"{config.token_env} is missing; phone receiver was not started",
            )
            log.warning(
                "external signal receiver disabled: token is missing",
                extra={"event": "external_receiver_blocked", "token_env": config.token_env},
            )
            return
        receiver = SignalReceiver(
            self.external_inbox, config.listen_host, config.listen_port, token
        )
        try:
            receiver.start()
        except OSError as exc:
            self.external_inbox.record(
                "receiver", "BLOCKED", f"receiver could not bind: {type(exc).__name__}: {exc}"
            )
            log.warning(
                "external signal receiver failed to start",
                extra={"event": "external_receiver_failed", "error": type(exc).__name__},
            )
            return
        self.external_receiver = receiver
        self.external_inbox.record(
            "receiver",
            "LISTENING",
            f"loopback HTTP receiver listening on {config.listen_host}:{config.listen_port}",
        )

    def _process_external_signals(self) -> None:
        config = getattr(self.settings, "external_signals", None)
        if config is None or not config.enabled or not hasattr(self, "risk"):
            return
        if not hasattr(self, "external_inbox") or not hasattr(self, "external_parser"):
            return
        for path, envelope in self.external_inbox.pending():
            event = self.external_parser.parse(envelope)
            parsed = event.safe_dict()
            self.external_inbox.record(envelope.event_id, "PARSED", event.reason, parsed=parsed)
            try:
                status, detail, extra = self._apply_external_signal(event)
            except Exception as exc:
                log.exception(
                    "external signal failed; normal Jarvis loop continues",
                    extra={"event": "external_signal_error", "event_id": envelope.event_id},
                )
                status, detail, extra = (
                    "ERROR",
                    f"{type(exc).__name__}: {exc}",
                    {"parsed": parsed},
                )
            self.external_inbox.finish(path, status, detail, **extra)

    def _apply_external_signal(
        self, event: ExternalSignalEvent
    ) -> tuple[str, str, dict[str, object]]:
        raw_text = event.envelope.combined_text
        if any(token in raw_text for token in ("{not_", "{notification}")):
            return (
                "INVALID",
                "MacroDroid sent literal Magic Text instead of notification data",
                {"parsed": event.safe_dict()},
            )
        allowed_apps = tuple(
            name.casefold() for name in self.settings.external_signals.allowed_apps
        )
        if allowed_apps and not any(name in event.envelope.app.casefold() for name in allowed_apps):
            return (
                "WRONG_APP",
                f"notification app {event.envelope.app!r} is not allowed",
                {"parsed": event.safe_dict()},
            )
        if event.kind in {ExternalSignalKind.INFO, ExternalSignalKind.INVALID}:
            return event.kind.value, event.reason, {"parsed": event.safe_dict()}
        age = max(0.0, (self.clock.now() - event.envelope.received_at).total_seconds())
        if age > self.settings.external_signals.max_age_seconds and not self._is_gold_follow(event):
            return (
                "STALE",
                f"notification is {age:.0f}s old; no delayed order or management action sent",
                {"parsed": event.safe_dict()},
            )
        if event.kind is ExternalSignalKind.NEW:
            return self._open_external_signal(event)
        return self._manage_external_signal(event)

    def _is_gold_follow(self, event: ExternalSignalEvent) -> bool:
        config = self.settings.external_signals
        if not config.gold_follow_enabled or not event.symbol_alias:
            return False
        alias = event.symbol_alias.upper()
        canonical = config.symbol_aliases.get(alias, alias).upper()
        allowed = set(config.gold_follow_aliases)
        return alias in allowed or canonical == "XAUUSD"

    def _is_gold_symbol(self, symbol: str) -> bool:
        config = self.settings.external_signals
        canonical = self.settings.instruments.canonical_symbol(symbol).upper()
        aliases = {alias.upper() for alias in config.gold_follow_aliases}
        mapped = {config.symbol_aliases.get(alias, alias).upper() for alias in aliases}
        return canonical == "XAUUSD" or canonical in aliases or canonical in mapped

    def _external_filter_check(
        self, context: FilterContext, *, gold_follow: bool
    ) -> tuple[FilterVerdict, dict[str, object]]:
        if not gold_follow:
            return self.filters.check(context)
        # Rio owns Gold direction. These are execution/market facts, not a
        # second strategy opinion: calendar/headline danger, whether Gold is
        # actually open long enough, whether it is moving, and its live spread.
        required = {"news", "headlines", "session", "runway", "spread"}
        data: dict[str, object] = {}
        checked = 0
        for filter_ in self.filters.filters:
            if filter_.name not in required:
                continue
            checked += 1
            verdict = filter_.check(context)
            data.update(verdict.data)
            if not verdict.passed:
                return verdict, data
        return FilterVerdict.allow("gold_follow", f"{checked} Gold execution filters clear"), data

    def _external_symbol(self, alias: str | None) -> str | None:
        if not alias:
            return None
        config = self.settings.external_signals
        canonical = config.symbol_aliases.get(alias.upper(), alias.upper())
        preferred = self.settings.instruments.broker_symbol(canonical)
        names = {descriptor.name for descriptor in self.broker.symbols()}
        if preferred in names:
            return preferred
        if canonical in names:
            return canonical
        folded = canonical.casefold()
        return next((name for name in names if name.casefold() == folded), None)

    def _open_external_signal(
        self, event: ExternalSignalEvent
    ) -> tuple[str, str, dict[str, object]]:
        gold_follow = self._is_gold_follow(event)
        if event.direction is None or event.symbol_alias is None:
            return "INVALID", event.reason, {"parsed": event.safe_dict()}
        if not gold_follow and not event.complete_entry:
            return "INVALID", event.reason, {"parsed": event.safe_dict()}
        if self.operation is OperationMode.MONITOR:
            return (
                "MONITOR_ONLY",
                "monitor mode cannot send a broker order",
                {"parsed": event.safe_dict()},
            )
        if self.kill_switch.is_engaged() or not self._entry_still_allowed():
            return (
                "BLOCKED",
                "hard STOP, account contract or capital floor blocks entry",
                {"parsed": event.safe_dict()},
            )
        symbol = self._external_symbol(event.symbol_alias)
        if symbol is None:
            return (
                "BLOCKED",
                f"no Eightcap symbol found for {event.symbol_alias}",
                {"parsed": event.safe_dict()},
            )
        spec = self.broker.spec(symbol, refresh=True)
        context = self.data.get_context(symbol, force_refresh=True)
        tick = context.tick
        if tick is None:
            return "BLOCKED", f"{symbol} has no executable quote", {"parsed": event.safe_dict()}
        entry = tick.ask if event.direction is Direction.LONG else tick.bid
        resolution: dict[str, object] = {"gold_follow": gold_follow}
        if gold_follow:
            config = self.settings.external_signals
            timeframe = Timeframe.parse(config.gold_follow_timeframe)
            try:
                bars = context.bars(timeframe).df
            except KeyError:
                bars = self.data.get_series(symbol, timeframe).df
            try:
                plan = build_gold_follow_plan(
                    event,
                    live_entry=entry,
                    bars=bars,
                    spread=tick.spread,
                    minimum_stop_distance=spec.min_stop_distance_price,
                    atr_period=self.settings.trade_management.sl_atr_period,
                    structure_bars=config.gold_follow_structure_bars,
                    stop_atr_multiple=config.gold_follow_stop_atr_multiple,
                    structure_buffer_atr=config.gold_follow_structure_buffer_atr,
                    target_reward_risk=config.gold_follow_target_reward_risk,
                    max_entry_deviation_atr=config.gold_follow_max_entry_deviation_atr,
                    max_entry_deviation_bps=config.gold_follow_max_entry_deviation_bps,
                )
            except (InsufficientDataError, ValueError) as exc:
                return "BLOCKED", str(exc), {"symbol": symbol, "parsed": event.safe_dict()}
            stop_loss = spec.normalize_price(plan.stop_loss)
            target = spec.normalize_price(plan.take_profit)
            resolution.update(
                {
                    "measured_atr": plan.measured_atr,
                    "fallback_stop": plan.used_fallback_stop,
                    "fallback_target": plan.used_fallback_target,
                    "provider_entry": plan.provider_entry,
                    "max_entry_deviation": plan.maximum_entry_deviation,
                    "resolved_sl": stop_loss,
                    "resolved_tp": target,
                }
            )
        else:
            low = event.entry_low or entry
            high = event.entry_high or low
            tolerance = entry * self.settings.external_signals.entry_zone_tolerance_bps / 10_000
            if entry < low - tolerance or entry > high + tolerance:
                return (
                    "OUTSIDE_ENTRY_ZONE",
                    f"live price {entry:g} is outside provider zone {low:g}-{high:g}",
                    {"symbol": symbol, "parsed": event.safe_dict()},
                )
            assert event.stop_loss is not None and event.take_profits
            stop_loss = spec.normalize_price(event.stop_loss)
            # A 0.01 ticket cannot be partially split. TP1 is the first
            # executable objective; later provider messages manage the rest.
            target = spec.normalize_price(event.take_profits[0])
        signal = Signal(
            module="external_rio",
            score=100.0 * int(event.direction),
            confidence=1.0,
            reasoning="authenticated provider signal; not a Jarvis-derived market edge",
            invalidation_price=stop_loss,
            details={
                "source": event.envelope.source,
                "event_id": event.envelope.event_id,
                **resolution,
            },
        )
        cycle_id = f"external:{event.envelope.event_id}"
        account = self.broker.account()
        positions = tuple(self._managed_positions())
        state = self.risk.build_state(account, positions)
        if gold_follow:
            # Explicit Gold-only bypass: provider directions are not subject to
            # Jarvis pacing, slot, whitelist, concentration or same-strategy
            # preference gates. A second live XAUUSD ticket is the sole
            # exception because it would be averaging/grid/hedging, which is a
            # hard project prohibition rather than a trade opinion.
            if state.has_position_in(symbol):
                existing = state.position_in(symbol)
                assert existing is not None
                detail = (
                    f"{symbol}: Rio Gold ticket #{existing.ticket} is already open; "
                    "a second same-symbol ticket would average, grid or hedge"
                )
                cycle_pk = self._record_skip(
                    cycle_id,
                    symbol,
                    account.equity,
                    Reason.POSITION_ALREADY_OPEN,
                    detail,
                    signals=[signal],
                    extra={"source": "external_rio", "event": event.safe_dict()},
                    total_score=100.0,
                )
                return (
                    "BLOCKED",
                    detail,
                    {"cycle_pk": cycle_pk, "parsed": event.safe_dict()},
                )
            resolution["bypassed_strategy_gates"] = [
                "daily_weekly_trade_pacing",
                "position_slot_limit",
                "equity_symbol_whitelist",
                "concentration_correlation",
                "minimum_reward_risk",
                "ai_strategy_veto",
            ]
        else:
            permission = self.risk.check_can_trade(state)
            if not permission.approved:
                cycle_pk = self._record_skip(
                    cycle_id,
                    symbol,
                    account.equity,
                    permission.reason,
                    permission.detail,
                    signals=[signal],
                    extra={"source": "external_rio", "event": event.safe_dict()},
                    total_score=100.0,
                )
                return (
                    "BLOCKED",
                    permission.detail,
                    {"cycle_pk": cycle_pk, "parsed": event.safe_dict()},
                )
            symbol_permission = self.risk.check_symbol(
                symbol, state, spec, direction=event.direction, entry=entry
            )
            if not symbol_permission.approved:
                cycle_pk = self._record_skip(
                    cycle_id,
                    symbol,
                    account.equity,
                    symbol_permission.reason,
                    symbol_permission.detail,
                    signals=[signal],
                    extra={"source": "external_rio", "event": event.safe_dict()},
                    total_score=100.0,
                )
                return (
                    "BLOCKED",
                    symbol_permission.detail,
                    {
                        "cycle_pk": cycle_pk,
                        "parsed": event.safe_dict(),
                    },
                )
        filter_verdict, filter_data = self._external_filter_check(
            FilterContext(
                symbol=symbol,
                spec=spec,
                now=self.clock.now(),
                direction=event.direction,
                tick=tick,
                open_positions=positions,
                extra={"external_source": event.envelope.source},
            ),
            gold_follow=gold_follow,
        )
        if not filter_verdict.passed:
            cycle_pk = self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                filter_verdict.reason,
                filter_verdict.detail,
                signals=[signal],
                extra={**filter_data, "source": "external_rio", "event": event.safe_dict()},
                total_score=100.0,
            )
            return (
                "BLOCKED",
                filter_verdict.detail,
                {
                    "cycle_pk": cycle_pk,
                    "parsed": event.safe_dict(),
                },
            )
        sizing = PositionSizer(self.settings).size(
            spec=spec,
            equity=account.equity,
            direction=event.direction,
            entry=entry,
            sl=stop_loss,
            tp=target,
            risk_multiplier=self.risk.risk_multiplier(state),
            spread_price=context.tick.spread,
            enforce_minimum_rr=not gold_follow,
        )
        if not sizing.approved:
            cycle_pk = self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                sizing.reason,
                sizing.decision.detail,
                signals=[signal],
                extra={
                    **filter_data,
                    "source": "external_rio",
                    "event": event.safe_dict(),
                    "resolution": resolution,
                },
                total_score=100.0,
            )
            self.recorder.record_sizing(cycle_pk, sizing)
            return (
                "BLOCKED",
                sizing.decision.detail,
                {
                    "cycle_pk": cycle_pk,
                    "parsed": event.safe_dict(),
                },
            )
        requested = self.settings.external_signals.fixed_volume
        volume = spec.round_volume_down(min(requested, sizing.volume))
        if volume < spec.volume_min:
            return (
                "BLOCKED",
                f"safe size is below broker minimum {spec.volume_min:g}",
                {"parsed": event.safe_dict()},
            )
        if volume != sizing.volume:
            sizing = replace(
                sizing,
                volume=volume,
                actual_risk_money=sizing.actual_risk_money * volume / sizing.volume,
                actual_risk_pct=sizing.actual_risk_pct * volume / sizing.volume,
            )
        margin = self.risk.check_margin(state, symbol, event.direction, volume, entry)
        if not margin.approved:
            return "BLOCKED", margin.detail, {"parsed": event.safe_dict()}
        self.risk.assert_not_forbidden(sizing, state)
        cycle_pk = self.recorder.record_cycle(
            cycle_id=cycle_id,
            context=self._journal_cycle_context(
                symbol,
                account.equity,
                {
                    **filter_data,
                    "source": "external_rio",
                    "event": event.safe_dict(),
                    "resolution": resolution,
                },
            ),
            reason=Reason.OK,
            detail="authenticated external provider signal passed all Jarvis execution gates",
            traded=True,
            direction=event.direction,
            total_score=100.0,
            score_threshold=self.settings.analysis.confluence.score_threshold,
            signals=[signal],
            weights={"external_rio": 1.0},
        )
        self.recorder.record_sizing(cycle_pk, sizing)
        trade_id = self.recorder.record_entry_intent(
            cycle_pk=cycle_pk, sizing=sizing, equity_before=account.equity
        )
        request = OrderRequest(
            symbol=symbol,
            direction=event.direction,
            volume=volume,
            sl=sizing.sl,
            tp=sizing.tp,
            reference_price=entry,
            deviation_points=self.settings.mt5.deviation_points,
            magic=self.settings.system.magic_number,
            comment=RIO_POSITION_COMMENT,
        )
        result = self.broker.order_send(request, spec)
        self.recorder.record_order_attempt(
            trade_id=trade_id if result.ok else None,
            kind="EXTERNAL_ENTRY",
            symbol=symbol,
            result=result,
        )
        if not result.ok:
            self.journal.abandon_pending_entry(
                trade_id, f"external entry rejected: {result.retcode_name} {result.comment}"
            )
            return (
                "ORDER_REJECTED",
                f"{result.retcode_name}: {result.comment}",
                {
                    "cycle_pk": cycle_pk,
                    "parsed": event.safe_dict(),
                },
            )
        self._promote_confirmed_entry(
            trade_id=trade_id,
            result=result,
            sizing=sizing,
            spec=spec,
            equity=account.equity,
        )
        self.external_inbox.open_campaign(
            source=event.envelope.source,
            symbol_alias=self.settings.external_signals.symbol_aliases.get(
                (event.symbol_alias or symbol).upper(), (event.symbol_alias or symbol).upper()
            ),
            broker_symbol=symbol,
            ticket=int(result.position_ticket or 0),
            direction=event.direction.name,
            event_id=event.envelope.event_id,
        )
        return (
            "TRADE_OPENED",
            f"ticket {result.position_ticket} opened at {result.filled_price:g}",
            {
                "cycle_pk": cycle_pk,
                "ticket": result.position_ticket,
                "symbol": symbol,
                "parsed": event.safe_dict(),
            },
        )

    def _manage_external_signal(
        self, event: ExternalSignalEvent
    ) -> tuple[str, str, dict[str, object]]:
        alias = event.symbol_alias
        if alias:
            alias = self.settings.external_signals.symbol_aliases.get(alias.upper(), alias.upper())
        campaign = self.external_inbox.active_campaign(event.envelope.source, alias)
        if campaign is None:
            return (
                "NO_CAMPAIGN",
                "management message did not match exactly one open campaign",
                {"parsed": event.safe_dict()},
            )
        key, row = campaign
        ticket = int(row.get("ticket") or 0)
        position = next((item for item in self._managed_positions() if item.ticket == ticket), None)
        if position is None:
            self.external_inbox.close_campaign(key, "POSITION_NOT_FOUND")
            return (
                "POSITION_NOT_FOUND",
                f"campaign ticket {ticket} is no longer open",
                {"parsed": event.safe_dict()},
            )
        result = None
        if event.kind is ExternalSignalKind.MOVE_STOP:
            if event.move_stop_to is None:
                return (
                    "INVALID",
                    "move-stop message contains no price",
                    {"parsed": event.safe_dict()},
                )
            new_stop = self.broker.spec(position.symbol).normalize_price(event.move_stop_to)
            tightens = (new_stop - position.sl) * int(position.direction) > 0
            correct_side = (new_stop - self.broker.tick(position.symbol).mid) * int(
                position.direction
            ) < 0
            if not tightens or not correct_side:
                return (
                    "BLOCKED",
                    "provider stop change would widen risk or cross live price",
                    {"parsed": event.safe_dict()},
                )
            result = self.broker.modify_stops(position, sl=new_stop, tp=position.tp)
        elif event.kind is ExternalSignalKind.BOOK_PROFIT:
            spec = self.broker.spec(position.symbol)
            volume = spec.round_volume_down(
                position.volume * self.settings.external_signals.partial_close_fraction
            )
            if not self._is_gold_symbol(position.symbol):
                # Preserve the original external-signal behaviour outside the
                # explicitly requested Rio Gold follow mode.
                if volume <= 0 or volume >= position.volume:
                    return (
                        "INFO",
                        "position is too small for a broker-valid partial close",
                        {"parsed": event.safe_dict()},
                    )
                result = self.broker.close_position(position, volume=volume)
                # The common recorder below handles this ordinary route.
                return self._finish_external_management(event, position, result, ticket, key)
            if position.profit + position.swap <= 0:
                return (
                    "INFO",
                    "provider said book profit but the live ticket is not profitable",
                    {
                        "ticket": ticket,
                        "parsed": event.safe_dict(),
                    },
                )
            if volume <= 0 or volume >= position.volume:
                # Eightcap's 0.01 lot step cannot split a 0.01 position. The
                # truthful implementation of "book profit" is therefore to
                # bank the whole smallest ticket, not pretend a partial exists.
                result = self.broker.close_position(position)
                self.recorder.record_order_attempt(
                    trade_id=None,
                    kind="EXTERNAL_BOOK_PROFIT_FULL_MIN_LOT",
                    symbol=position.symbol,
                    result=result,
                )
                if result.ok:
                    self.external_inbox.close_campaign(key, "BOOK_PROFIT_FULL_MIN_LOT")
                status = "MANAGED" if result.ok else "ORDER_REJECTED"
                return (
                    status,
                    f"smallest lot banked: {result.retcode_name}: {result.comment}",
                    {
                        "ticket": ticket,
                        "parsed": event.safe_dict(),
                    },
                )
            result = self.broker.close_position(position, volume=volume)
            self.recorder.record_order_attempt(
                trade_id=None,
                kind="EXTERNAL_BOOK_PROFIT_PARTIAL",
                symbol=position.symbol,
                result=result,
            )
            if not result.ok:
                return (
                    "ORDER_REJECTED",
                    f"{result.retcode_name}: {result.comment}",
                    {
                        "ticket": ticket,
                        "parsed": event.safe_dict(),
                    },
                )
            remaining = next(
                (item for item in self._managed_positions() if item.ticket == ticket), None
            )
            if remaining is None:
                self.external_inbox.close_campaign(key, "BOOK_PROFIT_CLOSED")
                return (
                    "MANAGED",
                    "profit booked; broker reports no remaining ticket",
                    {
                        "ticket": ticket,
                        "parsed": event.safe_dict(),
                    },
                )
            tick = self.broker.tick(remaining.symbol)
            sign = int(remaining.direction)
            desired = remaining.price_open + sign * tick.spread
            correct_side = (desired - tick.mid) * sign < -spec.min_stop_distance_price
            tightens = remaining.sl <= 0 or (desired - remaining.sl) * sign > 0
            if correct_side and tightens:
                protected = self.broker.modify_stops(
                    remaining,
                    sl=spec.normalize_price(desired),
                    tp=remaining.tp,
                )
                self.recorder.record_order_attempt(
                    trade_id=None,
                    kind="EXTERNAL_BOOK_PROFIT_BREAK_EVEN",
                    symbol=remaining.symbol,
                    result=protected,
                )
                detail = (
                    "partial profit booked and remaining ticket protected at break-even-plus"
                    if protected.ok
                    else "partial profit booked; broker rejected immediate break-even protection"
                )
                return (
                    "MANAGED" if protected.ok else "PARTIAL_ONLY",
                    detail,
                    {
                        "ticket": ticket,
                        "parsed": event.safe_dict(),
                    },
                )
            return (
                "MANAGED",
                "partial profit booked; live price is too close for broker-valid BE",
                {"ticket": ticket, "parsed": event.safe_dict()},
            )
        elif event.kind in {ExternalSignalKind.CLOSE, ExternalSignalKind.CANCEL}:
            result = self.broker.close_position(position)
        elif event.kind is ExternalSignalKind.TP_HIT:
            return (
                "INFO",
                "provider target update recorded; broker SL/TP remains authoritative",
                {
                    "ticket": ticket,
                    "parsed": event.safe_dict(),
                },
            )
        if result is None:
            return "INFO", event.reason, {"ticket": ticket, "parsed": event.safe_dict()}
        self.recorder.record_order_attempt(
            trade_id=None,
            kind=f"EXTERNAL_{event.kind.value}",
            symbol=position.symbol,
            result=result,
        )
        if result.ok and event.kind in {ExternalSignalKind.CLOSE, ExternalSignalKind.CANCEL}:
            self.external_inbox.close_campaign(key, event.kind.value)
        status = "MANAGED" if result.ok else "ORDER_REJECTED"
        return (
            status,
            f"{result.retcode_name}: {result.comment}",
            {
                "ticket": ticket,
                "parsed": event.safe_dict(),
            },
        )

    def _finish_external_management(
        self,
        event: ExternalSignalEvent,
        position: Position,
        result: OrderResult,
        ticket: int,
        campaign_key: str,
    ) -> tuple[str, str, dict[str, object]]:
        """Record an ordinary, non-Gold provider management result."""
        self.recorder.record_order_attempt(
            trade_id=None,
            kind=f"EXTERNAL_{event.kind.value}",
            symbol=position.symbol,
            result=result,
        )
        if result.ok and event.kind in {ExternalSignalKind.CLOSE, ExternalSignalKind.CANCEL}:
            self.external_inbox.close_campaign(campaign_key, event.kind.value)
        return (
            "MANAGED" if result.ok else "ORDER_REJECTED",
            f"{result.retcode_name}: {result.comment}",
            {"ticket": ticket, "parsed": event.safe_dict()},
        )

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
                self._guard_until(self._guard_deadline(started))
        finally:
            self.close()

    def _guard_deadline(self, cycle_started: float) -> float:
        """When the next cycle may begin, given that positions need watching.

        The interval is a target gap between cycle *starts*, which is the right
        way to pace scanning and the wrong way to schedule protection: a cycle
        that overruns the interval leaves a deadline in the past, and the guard
        returns without a single tick.

        That is not a corner case here. Live cycles run 55 to 121 seconds
        against a 30-second interval, so the one-second layer never ran at all
        — every rule in it written, tested, deployed and never once executed on
        an open position, while the deck reported readings nine minutes old.

        So the guard gets a floor. A scan delayed by twenty seconds costs at
        most a setup; a position unwatched for two minutes costs money, and the
        fast layer exists precisely because that trade-off is not close.
        """
        interval = self.settings.system.loop_interval_seconds
        floor = self.settings.system.min_guard_seconds
        return max(cycle_started + interval, time.monotonic() + floor)

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
            self._process_external_signals()

    def _pulse_guard(self) -> bool:
        """Run due position protection inside a long single-threaded scan."""

        interval = self.settings.system.guard_interval_seconds
        if interval <= 0:
            return False
        now = time.monotonic()
        if now - self._last_guard_pulse < interval:
            return False
        self.guard_tick()
        self._process_external_signals()
        self._last_guard_pulse = time.monotonic()
        return True

    def _market_is_manageable(self, symbol: str) -> bool:
        """Can an order or a stop change on this symbol reach the venue right now?

        This is a question about the market, not about the setup. A single-name
        share whose exchange shut at 17:00 Amsterdam answers False: the position
        cannot be closed, its stop cannot be moved, and nothing about it can be
        secured until the venue reopens.

        Two signals, both read from the broker and neither guessed:

        1. `trade_mode`. Brokers that downgrade a symbol out of session to
           close-only or disabled say so here, and that is decisive.
        2. Quote age. Most CFD venues leave `trade_mode` at FULL and simply stop
           publishing. A quote older than `unmanageable_quote_age_seconds` means
           the venue has stopped quoting, which is the same thing said quietly.

        Unclear answers return True — the market is assumed open and the
        position keeps its slot. Releasing a slot is the loosening, so it
        happens on evidence, never on the absence of it.
        """
        try:
            spec = self.broker.spec(symbol)
        except Exception:  # noqa: BLE001 - an unreadable symbol keeps its slot
            return True
        if not spec.is_tradable:
            return False
        try:
            tick = self.broker.tick(symbol)
        except Exception:  # noqa: BLE001 - as above
            return True
        if tick is None:
            return True
        age = (self.clock.now() - tick.time).total_seconds()
        # A negative age means the broker's clock leads ours. That is a clock
        # problem, not a shut venue, and it must not release a slot.
        if age < 0:
            return True
        return age <= self.settings.risk.unmanageable_quote_age_seconds

    def _managed_positions(self) -> list[Position]:
        """Positions Jarvis has explicitly accepted responsibility for."""
        positions = self.broker.positions()
        allowed = {self.settings.system.magic_number}
        manual = getattr(getattr(self.settings, "trade_management", None), "manual_positions", None)
        if (
            manual is not None
            and manual.enabled
            and getattr(self, "operation", None)
            in {
                OperationMode.DEMO,
                OperationMode.LIVE,
                OperationMode.EXPERIMENTAL_LIVE,
            }
        ):
            allowed.update(manual.magic_numbers)
        return [
            position
            for position in positions
            if getattr(position, "magic", self.settings.system.magic_number) in allowed
            and not self._symbol_ignored(position.symbol)
        ]

    def _symbol_ignored(self, symbol: str) -> bool:
        """Compatibility-safe access to the complete instrument opt-out."""
        instruments = getattr(self.settings, "instruments", None)
        # `is_hands_off`, not `is_ignored`: an observation-only symbol IS
        # scanned so a paper section can read it, and must still never have an
        # open position touched.
        predicate = getattr(instruments, "is_hands_off", None)
        return bool(predicate(symbol)) if callable(predicate) else False

    @staticmethod
    def _autonomously_managed_positions(positions: list[Position]) -> list[Position]:
        """Exclude provider-owned Rio plans from Jarvis/Claude management.

        They remain owned for reconciliation, reporting, capital-floor and
        emergency-stop purposes. Normal management messages reach them through
        `_manage_external_signal`; this only removes unsolicited mechanical,
        news and AI changes to the provider's SL/TP/exit plan.
        """
        return [position for position in positions if not _is_rio_position(position)]

    def _manual_adoption_enabled(self) -> bool:
        manual = getattr(getattr(self.settings, "trade_management", None), "manual_positions", None)
        return bool(
            manual is not None
            and manual.enabled
            and getattr(self, "operation", None)
            in {OperationMode.DEMO, OperationMode.LIVE, OperationMode.EXPERIMENTAL_LIVE}
        )

    def _adopt_manual_positions(self, account: AccountSnapshot) -> list[ManagementEvent]:
        """Attach a measured plan to new owner-opened positions.

        Magic zero positions used to be invisible to both the one-second guard
        and Claude. Adoption is explicit and durable: missing protection is
        attached first, then the exact plan is written to the journal. If the
        broker refuses protection, the position is closed rather than claimed
        as managed while remaining exposed.
        """
        config = self.settings.trade_management.manual_positions
        if not config.enabled or self.operation not in {
            OperationMode.DEMO,
            OperationMode.LIVE,
            OperationMode.EXPERIMENTAL_LIVE,
        }:
            return []
        events: list[ManagementEvent] = []
        for position in self.broker.positions():
            if self._symbol_ignored(position.symbol):
                continue
            if (
                position.magic == self.settings.system.magic_number
                or position.magic not in config.magic_numbers
                or self.journal.open_trade_by_ticket(position.ticket) is not None
            ):
                continue
            try:
                protected = self._protect_manual_position(position)
            except Exception as exc:
                log.exception(
                    "manual position could not be planned",
                    extra={
                        "event": "manual_adoption_failed",
                        "ticket": position.ticket,
                        "symbol": position.symbol,
                    },
                )
                result = self.broker.close_position(position)
                events.append(
                    ManagementEvent(
                        position.ticket,
                        "MANUAL_UNPROTECTED_CLOSE" if result.ok else "MANUAL_CLOSE_REJECTED",
                        f"could not attach a measured SL/TP plan ({type(exc).__name__}); "
                        + ("position closed" if result.ok else "broker rejected emergency close"),
                        result.filled_price if result.ok else None,
                        position.profit + position.swap if result.ok else None,
                    )
                )
                continue
            sizing = self._manual_sizing(protected, account)
            trade_id = self.recorder.record_trade_open(
                cycle_pk=None,
                sizing=sizing,
                ticket=protected.ticket,
                entry_price=protected.price_open,
                equity_before=account.equity,
                opened_at=protected.opened_at,
                magic=protected.magic,
            )
            decision_id = self.brain.record_decision(
                decided_at=self.clock.now(),
                symbol=protected.symbol,
                reason="MANUAL_ADOPTED",
                mode=self.operation.value,
                direction=protected.direction.name,
                detail="owner-opened MT5 position adopted into Jarvis management",
                taken=True,
                equity=account.equity,
                entry=protected.price_open,
                stop_loss=protected.sl,
                take_profit=protected.tp,
                filters={"source": "manual_mt5", "magic": protected.magic},
            )
            self._brain_trades[protected.ticket] = self.brain.record_trade_opened(
                ticket=protected.ticket,
                decision_id=decision_id,
                symbol=protected.symbol,
                direction=protected.direction.name,
                volume=protected.volume,
                opened_at=protected.opened_at,
                entry=protected.price_open,
                stop_loss=protected.sl,
                take_profit=protected.tp,
                risk_money=sizing.actual_risk_money,
            )
            detail = (
                f"manual MT5 position adopted as trade #{trade_id}; SL {protected.sl:g}, "
                f"TP {protected.tp:g}, recorded risk {sizing.actual_risk_pct:.2f}%"
            )
            events.append(ManagementEvent(protected.ticket, "MANUAL_ADOPTED", detail))
            self.alerts.send(
                f"Jarvis adopted manual {protected.symbol} #{protected.ticket}: {detail}"
            )
        return events

    def _protect_manual_position(self, position: Position) -> Position:
        config = self.settings.trade_management.manual_positions
        spec = self.broker.spec(position.symbol)
        tick = self.broker.tick(position.symbol)
        price = tick.bid if position.direction is Direction.LONG else tick.ask
        sign = int(position.direction)
        stop_valid = position.sl > 0 and (price - position.sl) * sign > spec.min_stop_distance_price
        target_valid = (
            position.tp > 0 and (position.tp - price) * sign > spec.min_stop_distance_price
        )
        if stop_valid and target_valid:
            return position

        if stop_valid:
            sl = position.sl
            risk_from_entry = abs(position.price_open - sl)
        else:
            timeframe = Timeframe.parse(config.stop_timeframe)
            series = self.data.get_series(position.symbol, timeframe)
            measured_atr = atr(series.df, self.settings.trade_management.sl_atr_period)
            distance = max(
                measured_atr * config.stop_atr_multiple,
                spec.min_stop_distance_price * 2.0,
                tick.spread * 3.0,
            )
            sl = spec.normalize_price(price - sign * distance)
            risk_from_entry = abs(position.price_open - sl)

        if target_valid:
            tp = position.tp
        else:
            progress = (price - position.price_open) * sign
            minimum_ahead = max(spec.min_stop_distance_price * 2.0, tick.spread * 3.0)
            target_from_entry = max(
                config.target_reward_risk * risk_from_entry,
                progress + minimum_ahead,
            )
            tp = spec.normalize_price(position.price_open + sign * target_from_entry)
        if sl == position.sl and tp == position.tp:
            return position
        result = self.broker.modify_stops(position, sl=sl, tp=tp)
        if not result.ok:
            raise TradingSystemError(
                f"broker refused manual protection: {result.retcode_name} {result.comment}"
            )
        return replace(position, sl=sl, tp=tp)

    def _manual_sizing(self, position: Position, account: AccountSnapshot) -> SizingResult:
        spec = self.broker.spec(position.symbol)
        distance = abs(position.price_open - position.sl)
        commission = self.settings.risk.commission_per_lot(spec.asset_class.value)
        risk_money = (spec.money_per_lot(distance) + commission) * position.volume
        risk_pct = 100.0 * risk_money / account.equity if account.equity > 0 else 0.0
        reward_risk = abs(position.tp - position.price_open) / distance if distance > 0 else 0.0
        return SizingResult(
            decision=RiskDecision.allow("owner-opened position adopted with broker protection"),
            symbol=position.symbol,
            direction=position.direction,
            volume=position.volume,
            entry=position.price_open,
            sl=position.sl,
            tp=position.tp,
            intended_risk_money=risk_money,
            intended_risk_pct=risk_pct,
            actual_risk_money=risk_money,
            actual_risk_pct=risk_pct,
            sl_distance_price=distance,
            sl_distance_pips=spec.price_to_pips(distance),
            reward_risk=reward_risk,
            raw_volume=position.volume,
        )

    def guard_tick(self) -> list[ManagementEvent]:
        """One cheap pass over open positions. Never opens anything.

        Failures are swallowed deliberately. This runs between cycles on a
        best-effort basis; a dropped tick or a momentary IPC hiccup must not
        take the service down, and the next full cycle re-does everything this
        does with its own error handling.
        """
        positions: list = []
        try:
            self.broker.ensure_connected()
            if self._manual_adoption_enabled():
                account = self.broker.account()
                adopted = self._adopt_manual_positions(account)
                self._record_management(adopted)
            positions = self._managed_positions()
            if not positions:
                self._publish_health(positions)
                return []
            autonomous = self._autonomously_managed_positions(positions)
            events = self.manager.manage(
                autonomous, self.clock.now(), self.posture.patience_multiplier
            )
            self._record_management(events)
            self._record_position_states(autonomous)
            # The local outcome model costs no API call. Give it the same
            # event-driven opportunity on every guard pass as the mechanical
            # health reader. `_supervision_trigger` still suppresses unchanged
            # repeats, so "once a second" means attentive, not a fresh random
            # answer sixty times a minute. Paid advisers remain cycle-bound.
            advisor = getattr(self, "advisor", None)
            if (
                advisor is not None
                and not getattr(advisor, "uses_paid_api", True)
                and getattr(advisor, "supports_dynamic_management", False)
            ):
                fresh = self._autonomously_managed_positions(self._managed_positions())
                self._supervise_positions(fresh)
            return events
        except Exception as exc:  # noqa: BLE001 - see docstring
            log.warning(
                "guard tick failed: %s",
                exc,
                extra={"event": "guard_tick_failed", "error": type(exc).__name__},
            )
            return []
        finally:
            # Publish whatever we know, including after a failed pass. It used
            # to run only on the success path, so any error anywhere in
            # `manage` left the file frozen at its last good write — and a
            # frozen file is indistinguishable from a stopped Jarvis on the
            # deck. The one moment an operator most needs to see what the
            # system thinks is the moment something went wrong.
            if positions:
                self._publish_health(positions)

    def run_once(
        self, *, batch_size: int | None = None, deep_candidates: int | None = None
    ) -> CycleSummary:
        started_at = self.clock.now()
        self._reviews_this_cycle = 0
        # Section breakers are read once per cycle, not once per candidate.
        # Bumping the serial here is what expires that cache; without it the
        # first verdict of the run would be reused forever and a section that
        # tripped mid-session would keep trading until a restart.
        self._cycle_serial += 1
        if self.kill_switch.is_engaged():
            self._flatten_owned_positions("operator hard STOP")
            return self._summary(started_at, ScanBatch((), (), 0, 0, self.cursor, 0), 0, 0)
        self.broker.ensure_connected()
        if isinstance(self.broker, PaperBroker):
            self._record_paper_closures(self.broker.mark_to_market())
        account = self.broker.account()
        self._record_management(self._adopt_manual_positions(account))
        positions = self._managed_positions()
        if self._experimental_floor_tripped(account.equity, positions):
            return self._summary(started_at, ScanBatch((), (), 0, 0, self.cursor, 0), 0, 0)
        reconciliation = self.manager.reconcile(positions)
        self._record_management(reconciliation)
        reconciliation_failures = {
            "BROKER_CLOSED_PENDING_HISTORY",
            "EMERGENCY_CLOSE_REJECTED",
            "ORPHAN_CLOSE_REJECTED",
        }
        if any(event.action in reconciliation_failures for event in reconciliation):
            self.alerts.send(
                "CRITICAL: reconciliation left unresolved broker risk; new risk halted"
            )
            self.risk.halt("broker/journal reconciliation left unresolved broker risk")
        positions = self._managed_positions()
        news_filter = next(
            (item for item in self.filters.filters if isinstance(item, NewsFilter)), None
        )
        if news_filter is not None:
            autonomous = self._autonomously_managed_positions(positions)
            self._record_management(self.manager.manage_news(autonomous, news_filter))
            positions = self._managed_positions()
        # How the account should be carrying itself, given the last few trades.
        # Only ever tightens: less patience with a stalled trade, a higher bar
        # for a new one. Never larger size — see risk/posture.py.
        interim = self.risk.build_state(account, positions)
        self.posture = assess(
            consecutive_losses=interim.consecutive_losses,
            equity=interim.equity,
            equity_peak=interim.equity_peak,
            enabled=self.settings.risk.posture_throttle,
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
        # Once a cycle, not once a guard tick: the banking rule needs equity to
        # the nearest euro and a broker round trip every second to watch a
        # number that moves in cents is a poor trade.
        self.manager.equity = account.equity
        autonomous = self._autonomously_managed_positions(positions)
        self._record_management(
            self.manager.manage(autonomous, self.clock.now(), self.posture.patience_multiplier)
        )
        positions = self._managed_positions()
        # The mechanical rules have had their say; now the judgement layer. It
        # runs after them deliberately — break-even and the ATR trail are
        # cheap, deterministic and always correct to apply, so they should not
        # wait on an API call, and the supervisor sees the position in the state
        # those rules left it.
        self._supervise_positions(self._autonomously_managed_positions(positions))
        positions = self._managed_positions()
        state = self.risk.build_state(account, positions)
        if self.risk.circuit_breaker_tripped(state):
            for position in positions:
                self.broker.close_position(position)
            self.alerts.send(self.risk.trip_circuit_breaker(state))
            return self._summary(started_at, ScanBatch((), (), 0, 0, self.cursor, 0), 0, 0)

        self._process_external_signals()
        account = self.broker.account()
        positions = self._managed_positions()
        state = self.risk.build_state(account, positions)

        batch = self.scanner.scan(
            cursor=self.cursor,
            batch_size=batch_size if batch_size is not None else self.settings.scanner.batch_size,
            keep=(
                deep_candidates
                if deep_candidates is not None
                else self.settings.scanner.deep_candidates
            ),
            pulse=self._pulse_guard,
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
        position_slots_full = (
            not permission.approved
            and permission.reason is Reason.MAX_POSITIONS_REACHED
            and self.settings.trade_management.pyramiding.enabled
            and not self.settings.trade_management.pyramiding.counts_toward_position_limit
        )
        if position_slots_full:
            # Four primary ideas may be full while one of them has earned a
            # separately bounded winner scalp. Keep the free chart analysis
            # running; `_process_candidate` lets only a proven same-symbol
            # scalp bypass this one gate and rejects every new primary idea.
            self.blocked_reason = str(Reason.MAX_POSITIONS_REACHED)
            self.blocked_detail = (
                f"{permission.detail}; primary slots full, scanning only for an eligible "
                "winner scalp"
            )
        elif not permission.approved:
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
        # by the configured market-quality lane first and the engine's own
        # conviction plus spread quality inside that lane. Everything downstream
        # (risk, filters, sizing, AI review, order) then runs in that order.
        analysed = self._analyse_batch(batch, account)
        analysed = self._apply_market_scout(analysed, batch)
        # Every candidate now gets the full chart analysis, so this is the
        # honest count of deep work done — not the number that survived it.
        deep = len(batch.candidates)
        for rank, candidate in enumerate(analysed, start=1):
            if self._pulse_guard():
                # The pulse may have banked or stopped a position. Never size
                # or correlate a new order against the pre-pulse snapshot.
                account = self.broker.account()
                positions = self._managed_positions()
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
                positions = self._managed_positions()
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
        self._resolve_counterfactuals()
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
        prescan = {candidate.symbol: candidate for candidate in batch.candidates}
        self._cycle_observations = []
        self._cycle_contexts = {}
        for candidate in batch.candidates:
            self._pulse_guard()
            try:
                item = self._analyse_candidate(
                    candidate.symbol,
                    candidate.asset_class,
                    account,
                    latest_bar=candidate.latest_bar,
                )
            except Exception:
                log.exception(
                    "candidate analysis failed; continuing with the rest of the batch",
                    extra={"event": "candidate_error", "symbol": candidate.symbol},
                )
                continue
            if item is not None:
                cheap = prescan.get(item.symbol)
                if cheap is not None:
                    item = replace(
                        item,
                        market_priority_tier=cheap.priority_tier,
                        spread_quality=cheap.spread_quality,
                        cost_priority=(
                            cheap.spread_quality * self.settings.scanner.priority_spread_weight
                        ),
                        after_cost_priority=self._after_cost_priority(item),
                    )
                analysed.append(item)
        self._world_state = build_world_state(self._cycle_observations)
        self._refresh_selection_evidence()
        observations = {row.symbol: row for row in self._cycle_observations}
        routed: list[AnalysedCandidate] = []
        for item in analysed:
            observation = observations.get(item.symbol)
            intelligence = item.intelligence
            if observation is None or intelligence is None:
                routed.append(
                    replace(
                        item,
                        raw_conviction_weight=(
                            self.settings.learning.selection_raw_conviction_weight
                        ),
                    )
                )
                continue
            spec = self.broker.spec(item.symbol)
            routing = self.settings.analysis.asset_class_routing.get(
                spec.asset_class.value,
                self.settings.analysis.asset_class_routing["unknown"],
            )
            intelligence = apply_cross_market_context(
                intelligence,
                item.idea,
                observation,
                spec.asset_class,
                self._world_state,
                routing=routing,
                cap=self.settings.analysis.market_regime.ranking_modifier_cap,
            )
            selection = self._selection_brain_for(
                asset_class=spec.asset_class.value,
                setup_family=item.idea.setup_family,
                horizon=item.idea.horizon,
                direction=item.idea.direction.name if item.idea.direction else "",
                regime=observation.regime,
                score=item.idea.score,
                signals=item.idea.signals,
            )
            if selection.reasons:
                summary = selection.summary()
                reasons = (*intelligence.reasons, summary)
                intelligence = replace(
                    intelligence,
                    learned_alignment=selection.modifier,
                    reasons=reasons,
                    thesis=f"{intelligence.thesis}; {summary}",
                )
            penalty = self._recent_refusal_penalty(item)
            if penalty < 0.0:
                intelligence = replace(
                    intelligence,
                    recent_refusals=penalty,
                    reasons=(
                        *intelligence.reasons,
                        f"the reviewer has refused this direction recently ({penalty:+.1f})",
                    ),
                )
            families = supporting_families(item.idea.signals, item.idea.direction)
            family_steps = min(
                max(0, len(families) - 1),
                self.settings.learning.selection_independent_family_cap,
            )
            family_bonus = family_steps * self.settings.learning.selection_independent_family_bonus
            if families:
                intelligence = replace(
                    intelligence,
                    reasons=(
                        *intelligence.reasons,
                        f"independent evidence families: {', '.join(families)} "
                        f"({family_bonus:+.1f} ranking only)",
                    ),
                )
            routed.append(
                replace(
                    item,
                    intelligence=intelligence,
                    raw_conviction_weight=(self.settings.learning.selection_raw_conviction_weight),
                    independent_family_bonus=family_bonus,
                )
            )
        analysed = routed
        analysed.sort(key=lambda item: item.selection_key, reverse=True)
        self._publish_market_intelligence(analysed)
        # In a drawdown, go less far down the ranked list. Candidates are
        # already ordered by market lane, conviction and execution cost, so
        # this says "only the first few" and always leaves one reachable. It
        # never touches position size.
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
                "ranked %d tradeable setups by market priority and conviction",
                len(analysed),
                extra={
                    "event": "conviction_ranking",
                    "candidates": len(analysed),
                    "best": [
                        f"{item.symbol} {item.idea.direction.name if item.idea.direction else '?'}"
                        f" {item.selection_score:.1f} [{item.market_priority_label}]"
                        for item in analysed[:5]
                    ],
                },
            )
        return analysed

    def _target_reach(self, context: MarketContext, idea: TradeIdea):  # type: ignore[no-untyped-def]
        """How often this market has covered the target distance, or None.

        None whenever the question cannot be answered — the gate is off, the
        planning timeframe is missing, there is too little history, or the plan
        has no usable geometry. Never a refusal on ignorance: an instrument the
        measurement cannot reach must be judged on everything else, exactly as
        it was before this existed.
        """
        config = self.settings.analysis.confluence
        if not config.require_target_base_rate or idea.direction is None:
            return None
        risk = abs(idea.entry - idea.stop_loss)
        reward = abs(idea.take_profit - idea.entry)
        if risk <= 0 or reward <= 0:
            return None
        try:
            planning = Timeframe.parse(idea.planning_timeframe)
        except ValueError:
            return None
        series = context.series.get(planning)
        if series is None or series.df is None or series.df.empty:
            return None
        minutes = max(1, int(planning.duration.total_seconds() / 60))
        bars_ahead = max(1, math.ceil(idea.expected_horizon_minutes / minutes))
        verdict = measure_target_reach(
            series.df.tail(400),
            distance=reward,
            bars_ahead=bars_ahead,
            long=idea.direction is Direction.LONG,
            reward_risk=reward / risk,
        )
        if not verdict.measured:
            return None
        # The configured margin rides on top of break-even rather than being
        # baked into it, so the arithmetic stays legible: the floor is what the
        # plan needs to return zero, and the margin is the operator's opinion
        # about how much daylight to demand above that.
        return replace(
            verdict,
            required_pct=min(100.0, verdict.required_pct + config.target_reach_margin_pct),
        )

    def _target_survival(self, context: MarketContext, idea: TradeIdea):  # type: ignore[no-untyped-def]
        """Expectancy of this plan measured against the stop it will carry.

        Same windows and same horizon as `_target_reach`, one difference: each
        window ends at the first bar that would have taken the stop, so a
        window that fell a full R and only then rallied counts as the loss it
        would have been.

        This is the measurement the analysis engine already uses to CHOOSE the
        target — `ConfluenceEngine._reachable_target` will not propose a
        distance without positive first-touch expectancy. It was not being
        re-asked here, and by here the plan is no longer the plan the engine
        approved: `_widen_stop_for_costs` runs a few lines above and, on an
        account this size, fires on nearly every candidate. Widening lowers
        reward-to-risk AND makes the position much harder to stop out. The old
        gate saw only the first half — an unconditional reach rate, unmoved by
        the wider stop, tested against a break-even the wider stop had just
        raised — so widening could only ever count against a trade.

        Returns None whenever the question cannot be answered, so an instrument
        without the history is judged on everything else exactly as before.
        """
        config = self.settings.analysis.confluence
        if not config.first_touch_target_test or not config.require_target_base_rate:
            return None
        # THIS WAS SILENCED, AND THE REASON HAS BEEN ANSWERED.
        #
        # It used to return None unless `plan_on_cost_floor` was on. The
        # argument was that the engine chooses its target against the NARROW
        # stop while `_widen_stop_for_costs` runs a few lines above this, so
        # re-measuring the WIDE one refused a plan on geometry the analysis
        # never saw — 69 of 140 live setups in six hours.
        #
        # That was true of the arithmetic as it stood, and the arithmetic was
        # the problem. Widening does two things at once: it lowers
        # reward-to-risk AND it makes the position far harder to stop out. The
        # old form could only see the first, because it priced two outcomes and
        # counted every unresolved window as a stop-out. Now that stop-outs are
        # counted directly, a wider stop earns its credit. Measured on one
        # market, target fixed, stop widened in four steps:
        #
        #     stop     RR    unconditional      first-touch, corrected
        #     1x     0.75    73.0% / 57.1% ok   86.1% resolved  +0.30R ok
        #     1.5x   0.50    73.0% / 66.7% ok   97.0% resolved  +0.20R ok
        #     2x     0.38    73.0% / 72.7% ok  100.0% resolved  +0.13R ok
        #     3x     0.25    73.0% / 80.0% NO  100.0% resolved  +0.05R ok
        #
        # The unconditional reading is FROZEN at 73.0% down the whole column:
        # it does not know the stop exists, so widening can only ever count
        # against a trade. That is the reading currently deciding the
        # break-even branch, and it refused 48 of 58 live setups in eighteen
        # minutes while the plans it judged had a positive measured expectancy.
        #
        # `plan_on_cost_floor` stays off; it was falsified on its own evidence
        # and this does not depend on it.
        if idea.direction is None:
            return None
        risk = abs(idea.entry - idea.stop_loss)
        reward = abs(idea.take_profit - idea.entry)
        if risk <= 0 or reward <= 0:
            return None
        try:
            planning = Timeframe.parse(idea.planning_timeframe)
        except ValueError:
            return None
        series = context.series.get(planning)
        if series is None or series.df is None or series.df.empty:
            return None
        minutes = max(1, int(planning.duration.total_seconds() / 60))
        bars_ahead = max(1, math.ceil(idea.expected_horizon_minutes / minutes))
        # The live spread, in units of the stop that will carry it. Widening
        # the stop is exactly what makes this number small, and it belongs on
        # the same side of the comparison as the reward-to-risk it paid for.
        spread = context.tick.spread if context.tick else 0.0
        verdict = measure_first_touch(
            series.df.tail(400),
            distance=reward,
            risk=risk,
            bars_ahead=bars_ahead,
            long=idea.direction is Direction.LONG,
            cost_r=max(0.0, spread / risk),
        )
        if verdict is None or not verdict.measured:
            return None
        return replace(
            verdict,
            required_pct=min(100.0, verdict.required_pct + config.target_reach_margin_pct),
        )

    def _conviction_stake(self, state, idea: TradeIdea, advice: Advice) -> tuple[float, str]:
        """The risk percentage this setup has earned, and why.

        Three numbers decide it and they do different jobs. The ENGINE's own
        confidence in this setup sets the ceiling, the final adviser's
        confidence may reduce that ceiling, and the total-exposure cap decides
        whether the book can carry it. Neither later number can increase the
        first.

        With `ai.provider: local_history` the adviser is the nearest-neighbour
        archive, and its confidence means two different things depending on how
        much history it happens to hold:

          under `min_neighbors`  it passes the engine's confidence straight
                                 through, so the stake was the engine's anyway
          at or over it          it returns `1 - learned_veto_rate`, which is a
                                 statement about how often COMPARABLE PAST
                                 setups turned out useful

        The archive must never inflate a weak live chart. It may, however, cap
        a large stake when comparable evidence is weak. That is the asymmetry
        the live EURUSD loss exposed: engine conviction bought 5.74% of equity
        while the final adviser only reported 0.47 confidence. Taking the
        minimum preserves every approved trade and changes only its size.

        A stake of zero means do not open this at all. The cap trims down to
        the ordinary stake and no further: a trade squeezed into whatever
        fraction of a percent happens to be left is not a smaller version of
        the setup, it is a position too small for this account to express and
        would be refused as TRADE_SKIPPED_UNDERCAPITALIZED one step later, with
        a reason naming the account size rather than the full book.
        """
        config = self.settings.risk.conviction_risk
        engine_confidence = idea.confidence
        conviction = min(engine_confidence, advice.confidence)
        wanted = config.stake_for(conviction)
        allowed = self.risk.room_for_more_risk(state, wanted, self.broker.spec)
        if allowed >= wanted:
            return wanted, (
                f"effective conviction {conviction:.2f} stakes {wanted:.1f}% "
                f"(engine {engine_confidence:.2f}, adviser cap {advice.confidence:.2f})"
            )
        used = self.risk.open_risk_pct(state, self.broker.spec)
        if allowed + 1e-9 < config.floor_pct:
            return 0.0, (
                f"effective conviction {conviction:.2f} (engine {engine_confidence:.2f}, "
                f"adviser cap {advice.confidence:.2f}) earned {wanted:.1f}% but the book "
                f"already carries {used:.1f}% of the "
                f"{self.settings.risk.max_total_open_risk_pct:.1f}% ceiling, leaving "
                f"{allowed:.1f}% — under the {config.floor_pct:.1f}% ordinary stake"
            )
        return allowed, (
            f"effective conviction {conviction:.2f} (engine {engine_confidence:.2f}, "
            f"adviser cap {advice.confidence:.2f}) earned {wanted:.1f}% but the book "
            f"already carries {used:.1f}% of the "
            f"{self.settings.risk.max_total_open_risk_pct:.1f}% ceiling, so this "
            f"takes {allowed:.1f}%"
        )

    def _after_cost_priority(self, item: AnalysedCandidate) -> float:
        """How much of this setup's reward survives its own round trip.

        The one thing about this account that is measured rather than believed.
        Conviction was ordering the queue and two independent readings say it
        predicts nothing here — the highest-conviction bucket lost the most,
        and the 40-45 band produced no useful review at all. Meanwhile every
        live trade spent over a quarter of its risk on commission and slippage,
        and that number entered the ordering only as a spread preference that
        never looked at the target.

        So this is not a forecast, it is a subtraction. A winner returns its
        reward minus the toll; a loser returns its risk plus the toll. What is
        scored is the FRACTION of the setup's reward-to-risk that survives that
        round trip.

        The fraction rather than the net ratio itself, and the difference
        matters. Scoring the ratio saturated at a reward-to-risk of about three
        and stopped discriminating exactly where the real candidates live. It
        also smuggled in a claim nobody has tested here — that a higher
        reward-to-risk is better — when a wider target simply trades win rate
        for size and `RR_BELOW_MINIMUM` already sets the floor. The fraction
        claims nothing of the sort. It says only that between two setups, the
        one handing less of its winnings to the broker is preferable, and that
        is true whatever the win rate turns out to be.

        It is also exactly where this account bleeds. A 1.8-pip stop and a
        20-pip stop at the same reward-to-risk are not the same trade: the
        first keeps about half of what it is theoretically worth and the second
        keeps nearly all of it.

        Ordering only. It cannot approve anything, and it is bounded by the
        configured weight so it cannot vault a fallback instrument over a core
        market.
        """
        weight = self.settings.scanner.after_cost_priority_weight
        idea = item.idea
        if weight <= 0 or not idea.entry or not idea.stop_loss or not idea.take_profit:
            return 0.0
        risk = abs(idea.entry - idea.stop_loss)
        reward = abs(idea.take_profit - idea.entry)
        if risk <= 0 or reward <= 0:
            return 0.0
        try:
            spec = self.broker.spec(item.symbol)
        except Exception:  # noqa: BLE001 - ordering must never fail a cycle
            # Ordering is a preference, never a requirement. A broker hiccup
            # here must leave the queue in its previous order, not empty it.
            return 0.0
        spread = item.context.tick.spread if item.context.tick is not None else 0.0
        # Borrowed from the sizer rather than recomputed: that file states
        # plainly that two definitions of this cost would eventually disagree,
        # and the one that disagrees quietly is the one that decides trades.
        share = PositionSizer(self.settings).cost_share(spec, risk, spread)
        cost = risk * max(0.0, share)
        gross_rr = reward / risk
        net_rr = (reward - cost) / (risk + cost)
        retained = net_rr / gross_rr
        # Clamped at zero rather than allowed to go negative. A setup whose
        # target does not clear its own toll is already refused by the cost
        # gate; scoring it negative here would count one rejection twice, on
        # the rare path where that gate has not run yet.
        return max(0.0, min(weight, retained * weight))

    def _refresh_selection_evidence(self) -> None:
        """Refresh the second brain's realised facets on a bounded cadence.

        Neon is remote and this runs inside the scan loop, so one query every
        cycle would turn learning into latency. A missing database or thin
        sample returns an empty list and therefore exactly the old ordering.
        """
        config = self.settings.learning
        if not config.selection_calibration_enabled:
            self._selection_evidence = []
            return
        now = self.clock.now()
        if (
            self._selection_evidence_at is not None
            and now - self._selection_evidence_at
            < timedelta(minutes=config.selection_refresh_minutes)
        ):
            return
        self._selection_evidence_at = now
        self._selection_evidence = self.brain.selection_evidence(
            minimum_trades=config.selection_min_trades,
            shrinkage_trades=config.selection_shrinkage_trades,
            points_per_r=config.selection_points_per_r,
            modifier_cap=config.selection_modifier_cap,
            outcome_floor_r=config.selection_outcome_floor_r,
            outcome_cap_r=config.selection_outcome_cap_r,
        )

    def _recent_refusal_penalty(self, item: AnalysedCandidate) -> float:
        """Send a repeatedly refused direction to the back of the review queue.

        `veto_memory` already suppresses the *identical* proposal -- same entry
        and stop within a quarter of an ATR. On a live tick that window is
        almost never hit: the deck showed forty-five paid calls against one
        served from memory, and thirty-two of the forty-five came back VETO.
        The price drifts a few points, the setup is technically new, and the
        same argument is bought again.

        This is a different question and a cheaper one. Not "may this trade" --
        the memory already answers that, and this changes no gate -- but "who
        gets one of the three paid reviews this cycle". A symbol the reviewer
        turned down twice in the last hour is a worse use of that budget than
        one it has not seen, whatever the price has done since.

        Bounded and decaying, so a refusal fades rather than blacklisting a
        market: two points per recent refusal, four at most, and gone once the
        memory's own window expires.
        """
        direction = item.idea.direction
        if direction is None:
            return 0.0
        record = self.veto_memory.standing(item.symbol, direction.name, self.clock.now())
        if record is None:
            return 0.0
        return -min(4.0, 2.0 * record.repeats)

    def _selection_brain_for(
        self,
        *,
        asset_class: str,
        setup_family: str,
        horizon: str,
        direction: str,
        regime: str,
        score: float,
        signals: tuple[Signal, ...],
    ) -> SelectionVerdict:
        """The account's learned context for one candidate, never a gate."""
        if not direction:
            return SelectionVerdict()
        session_filter = self.filters.find(SessionFilter)
        if session_filter is None:
            session = "unknown"
        elif asset_class in session_filter.config.continuous_asset_classes:
            session = "continuous"
        elif asset_class in session_filter.config.broker_hours_asset_classes:
            session = "broker-hours"
        else:
            session = session_filter.session_label(self.clock.now())
        supporting = tuple(
            signal.module
            for signal in signals
            if signal.score and ("LONG" if signal.score > 0 else "SHORT") == direction
        )
        config = self.settings.learning
        return combine_selection_evidence(
            self._selection_evidence,
            asset_class=asset_class,
            setup_family=setup_family,
            horizon=horizon,
            direction=direction,
            regime=regime,
            session=session,
            score=score,
            detectors=supporting,
            weights=config.selection_dimension_weights,
            strength=config.selection_ensemble_strength,
            cap=config.selection_ensemble_modifier_cap,
        )

    def _apply_market_scout(
        self, analysed: list[AnalysedCandidate], batch: ScanBatch
    ) -> list[AnalysedCandidate]:
        """Let Claude nominate one market without creating a new trade gate.

        A matching nomination receives a bounded ordering bonus. WAIT,
        disagreement, low confidence, a timeout or a symbol for which the
        deterministic engine has no executable idea all leave the queue exactly
        as it was. This is the crucial asymmetry: scouting can help select, but
        it cannot make an API outage equal zero trades.
        """
        config = self.settings.ai.market_scout
        scout_method = getattr(self.advisor, "scout", None)
        if (
            not config.enabled
            or self.operation is OperationMode.MONITOR
            or isinstance(self.advisor, DisabledAdvisor)
            or not callable(scout_method)
            or not self._cycle_contexts
        ):
            self._publish_market_intelligence(analysed)
            return analysed

        observations = {row.symbol: row for row in self._cycle_observations}
        markets: list[dict[str, object]] = []
        signatures: list[str] = []
        for cheap in batch.candidates:
            context = self._cycle_contexts.get(cheap.symbol)
            observation = observations.get(cheap.symbol)
            if context is None or observation is None:
                continue
            markets.append(scout_market_snapshot(context, observation))
            signatures.append(f"{cheap.symbol}:{observation.last_h1_bar}")
            if len(markets) >= config.max_markets_per_call:
                break
        if len(markets) < 1:
            self._publish_market_intelligence(analysed)
            return analysed

        signature = hashlib.sha256("|".join(signatures).encode("utf-8")).hexdigest()[:24]
        try:
            reserved, why = self.scout_throttle.reserve(
                signature,
                self.clock.now(),
                cooldown_minutes=config.cooldown_minutes,
                max_calls_per_day=config.max_calls_per_day,
            )
        except OSError:
            log.exception("market scout throttle unavailable; deterministic queue unchanged")
            self._publish_market_intelligence(analysed)
            return analysed
        if not reserved:
            log.debug("market scout skipped: %s", why, extra={"event": "market_scout_skipped"})
            self._publish_market_intelligence(analysed)
            return analysed

        payload = {
            "global_market_state": self._world_state,
            "markets": markets,
            "deterministic_tradeable_symbols": [item.symbol for item in analysed],
            "rule": (
                "Nomination changes ordering only. It cannot create, block, resize or "
                "execute a trade."
            ),
        }
        try:
            self.ai_ledger.append(
                "market_scout_request",
                {
                    "signature": signature,
                    "provider": self.settings.ai.provider,
                    "model": self.settings.ai.anthropic_model or self.settings.ai.openai_model,
                    "request": payload,
                },
            )
        except OSError:
            log.exception("market scout request could not be audited; paid call skipped")
            self._publish_market_intelligence(analysed)
            return analysed
        started = time.monotonic()
        decision = scout_method(payload)
        latency_ms = round((time.monotonic() - started) * 1000.0, 1)
        self._last_scout = decision
        try:
            self.ai_ledger.append(
                "market_scout_response",
                {
                    "signature": signature,
                    "latency_ms": latency_ms,
                    "decision": decision.safe_dict(),
                },
            )
        except OSError:
            log.exception("market scout response audit failed; ranking result retained")

        promoted: list[AnalysedCandidate] = []
        matched = False
        for item in analysed:
            intelligence = item.intelligence
            if (
                not matched
                and intelligence is not None
                and decision.directional
                and decision.confidence >= config.minimum_confidence
                and item.symbol == decision.symbol
                and item.idea.direction is not None
                and item.idea.direction.name == decision.action
            ):
                matched = True
                intelligence = replace(
                    intelligence,
                    scout_alignment=config.ranking_bonus,
                    reasons=(
                        *intelligence.reasons,
                        f"Claude scout independently nominated this {decision.action} setup",
                    ),
                    thesis=f"{intelligence.thesis}; scout: {decision.thesis}",
                )
                item = replace(item, intelligence=intelligence)
            promoted.append(item)
        promoted.sort(key=lambda item: item.selection_key, reverse=True)
        self._publish_market_intelligence(promoted)
        return promoted

    def _publish_market_intelligence(self, analysed: list[AnalysedCandidate]) -> None:
        write_json_atomic(
            self.market_intelligence_file,
            {
                "version": 1,
                "updated_at": self.clock.now().isoformat(),
                "operation": self.operation.value,
                "world": self._world_state,
                "scout": self._last_scout.safe_dict(),
                "opportunities": [
                    {
                        "symbol": item.symbol,
                        "direction": item.idea.direction.name if item.idea.direction else None,
                        "conviction": round(item.conviction, 2),
                        "ranking_score": round(item.ranking_score, 2),
                        "selection_score": round(item.selection_score, 2),
                        "market_priority": item.market_priority_label,
                        "spread_quality": round(item.spread_quality, 4),
                        "intelligence": (
                            item.intelligence.safe_dict() if item.intelligence is not None else None
                        ),
                    }
                    for item in analysed
                ],
                "observations": [row.safe_dict() for row in self._cycle_observations],
            },
        )

    def _analyse_candidate(  # type: ignore[no-untyped-def]
        self, symbol: str, asset_class, account, latest_bar: datetime | None = None
    ) -> AnalysedCandidate | None:
        cycle_id = str(uuid.uuid4())
        now = self.clock.now()
        # Asked before the ladder is fetched, because the fetch is the cost.
        # A held symbol reaches the same verdict it reached last time; it just
        # reaches it without eight timeframes of bars.
        #
        # `latest_bar` comes from the cheap scan, which has already read it. A
        # bar newer than the one that failed means the venue has traded since,
        # so the hold is released there and then rather than at its deadline.
        hold = self.quarantine.hold_for(symbol, now, latest_bar)
        if hold is not None:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.DATA_QUARANTINED,
                hold.summary(now),
                extra={"failures": hold.failures, "retry_at": hold.until.isoformat()},
            )
            return None
        try:
            context = self.data.get_context(symbol, force_refresh=True)
            # Tell the engine what the stop is going to be before it plans
            # around it. `_widen_stop_for_costs` runs on nearly every candidate
            # on an account this size, so the analysis was choosing a target,
            # measuring how often that target is reached, and computing an
            # expectancy — all against a stop it was about to be told was too
            # narrow to trade. 1,710 refusals in two live hours read "no target
            # between 1.00R and 3.00R pays on this market", and they were
            # measured against a stop the broker would never have received.
            #
            # Nothing about the order changes: the runner widens to this same
            # distance a few steps later either way, so the executed stop, the
            # risk and the lot size are identical. What changes is that the
            # target and the reach measurement are now about that trade.
            self._attach_cost_floor(symbol, context)
            self._attach_basket_peers(symbol, asset_class, context)
            idea = self.engine.evaluate(context, self.settings.mode)
            self._cycle_contexts[symbol] = context
            # The scan reads every symbol anyway, including the ones we are
            # already in. Keeping its directional verdict here makes "the engine
            # has turned against an open position" free to detect — no second
            # fetch, no second evaluation, and the same read the entry side used
            # rather than one derived separately and able to drift from it.
            if idea.direction is not None:
                self._latest_direction[symbol] = (idea.direction.name, idea.score)
            observation = observe_market(context, asset_class, idea.signals)
            self._cycle_observations.append(observation)
            if self.shadow_engine is not None:
                candidate = self.shadow_engine.evaluate(context, TradingMode.PAPER)
                self.shadow.record(symbol, idea, candidate, self.clock.now())
        except (TradingSystemError, ValueError) as exc:
            # Only the two structural failures are remembered. A stale quote or
            # a dropped connection resolves on its own, and holding those would
            # keep a market out of the scan for hours after it came back.
            if isinstance(exc, InsufficientDataError | DataIntegrityError):
                self.quarantine.record_failure(symbol, str(exc), now, latest_bar)
            self._record_skip(cycle_id, symbol, account.equity, Reason.DATA_UNAVAILABLE, str(exc))
            return None
        # It analysed cleanly, so whatever was wrong with its history is gone.
        self.quarantine.clear(symbol)

        # The other theories get their own look at the same chart. The swing
        # engine reads H1 structure; these read M5 impulse and M15 range, and
        # they carry their own stop and target because a five-minute plan and
        # an hourly one are different trades, not the same trade at different
        # strengths.
        verdict = self._playbook_verdict(context)
        if (
            self._playbooks_may_execute()
            and verdict is not None
            and verdict.conflict
            and self.playbook_config.veto_on_conflict
        ):
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
        # Exact shape memory applies to every horizon, including quick. A new
        # closed M1 event is allowed past the broad symbol/direction cooldown,
        # but it is not a new paid question when entry and stop are still the
        # same proposal within the recorded ATR tolerance.
        # ...and only while asking again would cost something. A free adviser
        # re-reads the live chart for nothing, so replaying a stored refusal
        # buys no saving and throws away a trade. See `_reviews_cost_money`.
        remembered = self._remembered_veto(idea) if self._reviews_cost_money() else None
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
                total_score=idea.score,
            )
            return None
        routing = self.settings.analysis.asset_class_routing.get(
            asset_class.value,
            self.settings.analysis.asset_class_routing["unknown"],
        )
        intelligence = assess_opportunity(
            idea,
            observation,
            asset_class,
            cap=self.settings.analysis.market_regime.ranking_modifier_cap,
            routing=routing,
        )
        return AnalysedCandidate(symbol, cycle_id, idea, context, intelligence, verdict)

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
        existing_legs = state.positions_in(symbol)
        pyramid_config = self.settings.trade_management.pyramiding
        is_addon = bool(existing_legs)
        allow_pyramid = is_addon and pyramid_config.enabled
        if is_addon and candidate.conviction < pyramid_config.minimum_conviction:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.POSITION_ALREADY_OPEN,
                f"winner scalp conviction {candidate.conviction:.1f} is below the "
                f"{pyramid_config.minimum_conviction:.1f} scalp floor; the existing "
                "position remains managed",
                signals=list(idea.signals),
            )
            return False
        # A NEW SECTION THAT HAS STOPPED ITSELF CANNOT CARRY THIS TRADE.
        #
        # Checked here rather than by removing the module from the engine,
        # because the engine's config is built once and a breaker has to be
        # able to trip mid-session. The setup was never assessed on its merits
        # — that is the point, and the journal says so with its own reason.
        stopped = self._tripped_section(idea)
        if stopped is not None:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.SECTION_BREAKER_TRIPPED,
                stopped,
                signals=list(idea.signals),
            )
            return False
        risk_decision = self.risk.evaluate(
            state,
            symbol,
            spec,
            direction=idea.direction,
            entry=idea.entry,
            allow_pyramid=allow_pyramid,
        )
        if not risk_decision.approved:
            cycle_pk = self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                risk_decision.reason,
                risk_decision.detail,
                signals=list(idea.signals),
            )
            self._record_counterfactual(cycle_pk, idea, risk_decision.reason)
            return False

        # A same-symbol winner has already paid for its currency/sector slot.
        # Exclude only those legs from the exposure filters; every other open
        # market still constrains the book normally.
        filter_positions = (
            tuple(position for position in positions if position.symbol != symbol)
            if is_addon
            else positions
        )
        filter_verdict, filter_data = self.filters.check(
            FilterContext(
                symbol=symbol,
                spec=spec,
                now=self.clock.now(),
                direction=idea.direction,
                tick=context.tick,
                open_positions=filter_positions,
            )
        )
        if not filter_verdict.passed:
            cycle_pk = self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                filter_verdict.reason,
                filter_verdict.detail,
                signals=list(idea.signals),
                extra=filter_data,
            )
            self._record_counterfactual(cycle_pk, idea, filter_verdict.reason)
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
        # Is the market already moving the other way? Asked before the paid
        # review, because a setup price is actively contradicting is not worth
        # an opinion — and because this is the gate that would have stopped a
        # short being sent into a resistance break.
        confirmed, adverse = self._entry_is_confirmed(context, idea)
        if not confirmed:
            assert adverse is not None  # only False once it has been measured
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.AWAITING_CONFIRMATION,
                f"price has run {adverse:.2f} ATR against this {idea.direction.name} over the "
                f"last {self.settings.analysis.confluence.confirmation_bars} "
                f"{self.settings.analysis.confluence.confirmation_timeframe} bars, above the "
                f"{self.settings.analysis.confluence.confirmation_max_adverse_atr:.2f} limit; "
                f"the setup may still be right, it is early. Re-checked next cycle.",
                signals=list(idea.signals),
                extra={**filter_data, "adverse_atr": round(adverse, 2)},
            )
            return False

        entry_quality = self._assess_entry_quality(context, idea, spec.asset_class)
        lifecycle = self._observe_setup_lifecycle(context, idea, entry_quality)
        # Recorded on the same row as the refusal, and that is the whole point.
        # `momentum_scalp` asks for a shallow pullback off a fresh impulse;
        # `entry_quality` refuses a price sitting at its range extreme. Those
        # two descriptions may or may not be the same bars — nobody could say,
        # because whichever gate fired first owned the journal row and the
        # other verdict was simply absent. Written together, one day of running
        # turns the argument into a number.
        playbook_note = (
            {"playbooks": candidate.playbooks.summary()}  # type: ignore[attr-defined]
            if candidate.playbooks is not None
            else {}
        )
        if not lifecycle.may_enter:
            if entry_quality.decision is EntryTimingDecision.DATA_UNAVAILABLE:
                reason = Reason.DATA_UNAVAILABLE
            elif lifecycle.state in (
                SetupState.PULLBACK_RECEIVED,
                SetupState.WAIT_RESUMPTION,
            ):
                reason = Reason.AWAITING_CONFIRMATION
            else:
                reason = Reason.ENTRY_OVEREXTENDED
            cycle_pk = self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                reason,
                lifecycle.reason,
                signals=list(idea.signals),
                extra={
                    **filter_data,
                    "entry_quality": entry_quality.safe_dict(),
                    "setup_lifecycle": lifecycle.safe_dict(),
                    **playbook_note,
                },
            )
            if entry_quality.decision is not EntryTimingDecision.DATA_UNAVAILABLE:
                # The wait is an execution hypothesis, not a fact. Follow the
                # untouched SL/TP in shadow so future OOS evidence can show
                # whether refusing this late price actually reduced MAE or
                # merely discarded winners. No broker order is created.
                self._record_counterfactual(cycle_pk, idea, reason)
            return False

        # AND ON THE TRADES IT LET THROUGH, WHICH IS THE HALF THAT MATTERED.
        #
        # `entry_quality.safe_dict()` was written on the refusal above and
        # nowhere else, so every setup this gate REFUSED carried its numbers
        # into the journal and every trade it APPROVED carried none. That is
        # backwards: the refusals are the ones nobody has to explain.
        #
        # ETHUSD LONG, 20 August, entered at 2313.40 within two points of a
        # vertical M1 spike and never printed a positive tick — peak +0.00R,
        # -0.46R at the low. Asking why the gate allowed it produced
        # `activity_ratio` and `ai_confidence` and nothing else. Where in its
        # range that price sat, how far the move had already run, how large the
        # last bar was, how far from the EMA — all four measured, none kept.
        #
        # A gate that cannot be audited on its own decisions is a gate nobody
        # can improve, and the last three attempts to tune this one were made
        # by reading the code and guessing which sub-test had fired.
        filter_data = {
            **filter_data,
            "entry_quality": entry_quality.safe_dict(),
            "setup_lifecycle": lifecycle.safe_dict(),
        }

        # Give the stop the room the costs demand, before anything is sized.
        #
        # The cost gate in the sizer would otherwise refuse this outright, and
        # on this account it refuses nearly everything: measured against real
        # fills, all four of the live trades had stops between 1.8 and 6.3 pips
        # and every one of them spends over a quarter of its risk on commission
        # and slippage. A gate that says no to all of them is not a risk
        # control, it is an off switch.
        #
        # Widening is the honest alternative and it is what a person does. The
        # invalidation level does not move — the trade is still wrong in the
        # same place — it simply stops being sized as though the market cannot
        # breathe. What it costs is reward-to-risk, and that is already
        # measured: if the target no longer justifies the wider stop, the RR
        # gate refuses the trade a few lines further down, on the merits.
        idea = self._widen_stop_for_costs(idea, spec)

        affordable, share = self._spread_is_affordable(context, idea.entry, idea.stop_loss)
        if not affordable:
            cycle_pk = self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.SPREAD_EATS_THE_STOP,
                f"spread is {share:.0%} of the {abs(idea.entry - idea.stop_loss):.5g} stop, "
                f"above the {self.settings.analysis.confluence.max_spread_share_of_stop:.0%} limit",
                signals=list(idea.signals),
                extra=filter_data,
            )
            self._record_counterfactual(cycle_pk, idea, Reason.SPREAD_EATS_THE_STOP)
            return False

        # Is there time for this to work? The runway filter enforced a flat
        # floor without seeing the setup; now that the target is known, ask the
        # sharper question. A 5-ATR target on a market moving at its usual pace
        # needs hours, and an hour before the wind-down it is not a trade with
        # a lower probability — it is a trade that cannot complete, and the
        # only thing it reliably does is pay the spread twice.
        reachable, needed, runway = self._target_is_reachable_in_time(
            context, idea, spec.asset_class.value
        )
        if not reachable:
            assert needed is not None and runway is not None  # only False with both set
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.INSUFFICIENT_RUNWAY,
                f"target is ~{needed:.0f} min away at the current pace but only "
                f"{runway:.0f} min remain before the {spec.asset_class.value} wind-down; "
                f"the position would be closed on the clock, not on the idea",
                signals=list(idea.signals),
                extra={**filter_data, "minutes_to_target": round(needed, 1)},
            )
            return False

        # Does this market ever actually go where the target is?
        #
        # Asked here, after the stop has been widened for costs so the
        # reward-to-risk is the real one, and before a cent is spent. The
        # measurement has always been computed — for the review payload, where
        # the reviewer read it and refused the trade. Six consecutive live
        # refusals cited exactly this number: UK100 at 30.2%, CADCHF at 30.1%,
        # AUDUSD at "37.0% up against 37.5% down, essentially a coin flip", and
        # AUDSGD proposed LONG at 38.1% up against 46.8% DOWN.
        reach = self._target_reach(context, idea)
        # And the sharper version of the same question, asked against the stop
        # this order will actually carry. `reach` counts every window in which
        # price ever covered the distance, including the ones that first fell a
        # full R — so it answers "which way does this instrument travel", not
        # "does this plan pay". The plan owns a stop, and on this account that
        # stop has usually just been widened; see `SurvivalVerdict`. When the
        # history can answer it, it decides, and `reach` keeps the direction
        # comparison, which is what it is actually good for.
        survival = self._target_survival(context, idea)
        payable = survival if survival is not None else reach
        horizon_advisories: list[dict[str, object]] = []
        if payable is not None and not payable.clears_break_even:
            if self._reach_failure_is_advisory(idea, payable):
                horizon_advisories.append(
                    {
                        "type": (
                            "first_touch_expectancy_negative"
                            if survival is not None
                            else "unconditional_target_reach_below_break_even"
                        ),
                        "hard_gate": False,
                        "detail": payable.describe(),
                        "forward_pct": payable.forward_pct,
                        "required_pct": payable.required_pct,
                        "windows": payable.windows,
                    }
                )
            else:
                self._record_skip(
                    cycle_id,
                    symbol,
                    account.equity,
                    Reason.TARGET_RARELY_REACHED,
                    payable.describe(),
                    signals=list(idea.signals),
                    extra={**filter_data, "target_reach_pct": payable.forward_pct},
                    total_score=idea.score,
                )
                return False
        if (
            reach is not None
            and self.settings.analysis.confluence.require_direction_advantage
            and not reach.beats_the_other_side(
                self.settings.analysis.confluence.direction_advantage_tolerance_pct
            )
        ):
            direction_detail = (
                f"{idea.direction.name} target reached {reach.forward_pct:.1f}% of the time "
                f"against {reach.opposite_pct:.1f}% the other way; this market covers that "
                f"distance more readily against the proposal than for it"
            )
            if self._direction_failure_is_advisory(idea, reach):
                horizon_advisories.append(
                    {
                        "type": "unconditional_direction_disadvantage",
                        "hard_gate": False,
                        "detail": direction_detail,
                        "forward_pct": reach.forward_pct,
                        "opposite_pct": reach.opposite_pct,
                        "windows": reach.windows,
                    }
                )
            else:
                self._record_skip(
                    cycle_id,
                    symbol,
                    account.equity,
                    Reason.TARGET_RARELY_REACHED,
                    direction_detail,
                    signals=list(idea.signals),
                    extra={**filter_data, "target_reach_pct": reach.forward_pct},
                    total_score=idea.score,
                )
                return False

        entry_risk_multiplier = self.risk.risk_multiplier(state)
        if is_addon:
            entry_risk_multiplier *= pyramid_config.risk_multiplier
        sizing = PositionSizer(self.settings).size(
            spec=spec,
            equity=account.equity,
            direction=idea.direction,
            entry=idea.entry,
            sl=spec.normalize_price(idea.stop_loss),
            tp=spec.normalize_price(idea.take_profit),
            risk_multiplier=entry_risk_multiplier,
            # The largest single cost of being wrong on this account, and the
            # gate cannot weigh it against commission and slippage unless it is
            # handed the live number. A setup cannot exist without a tick — the
            # playbooks refuse to score one — so the zero branch is a belt on a
            # path that does not reach here.
            spread_price=context.tick.spread if context.tick else 0.0,
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
            self._record_counterfactual(cycle_pk, idea, sizing.reason)
            return False
        sizing = self._cap_scalp_volume(idea, spec, sizing)
        margin = self.risk.check_margin(
            state,
            symbol,
            idea.direction,
            sizing.volume,
            sizing.entry,
        )
        if not margin.approved and self.settings.risk.reduce_volume_to_fit_margin:
            # FIT IT INSTEAD OF REFUSING IT.
            #
            # Every analytical gate has already passed by here. The margin does
            # not stretch to the size the risk budget asked for, and the size is
            # the one thing that can change without invalidating any of it —
            # entry, stop and target stay exactly where they were, so the
            # measured expectancy that approved this trade is still the
            # expectancy of the smaller one.
            #
            # It risks LESS, never more. Rounding UP to the broker minimum is
            # forbidden outright in this system and stays forbidden: when even
            # `volume_min` will not fit, this returns nothing and the refusal
            # below stands.
            fitted = self.risk.largest_volume_within_margin(
                state,
                symbol,
                idea.direction,
                sizing.volume,
                sizing.entry,
                volume_min=spec.volume_min,
                volume_step=spec.volume_step,
            )
            if fitted > 0 and fitted < sizing.volume:
                scale = fitted / sizing.volume
                sizing = replace(
                    sizing,
                    volume=fitted,
                    actual_risk_money=sizing.actual_risk_money * scale,
                    actual_risk_pct=sizing.actual_risk_pct * scale,
                )
                log.info(
                    "volume reduced to fit margin",
                    extra={
                        "event": "volume_fitted_to_margin",
                        "symbol": symbol,
                        "requested_volume": round(fitted / scale, 4),
                        "volume": fitted,
                        "risk_pct": round(sizing.actual_risk_pct, 3),
                    },
                )
                margin = self.risk.check_margin(
                    state, symbol, idea.direction, sizing.volume, sizing.entry
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
            self._record_counterfactual(cycle_pk, idea, margin.reason)
            return False
        self.risk.assert_not_forbidden(sizing, state, allow_pyramid=allow_pyramid)

        # Monitor mode cannot send an order no matter what comes back, so a paid
        # verdict here buys nothing. Seventy-nine calls went out in one session
        # before anyone noticed, eleven of them approvals that could never have
        # become trades. The review is skipped rather than the mode being
        # forbidden: watching the engine reason without spending is exactly what
        # monitor is for, and PAPER exercises the full path including the
        # adviser when that is what is wanted.
        if self.operation is OperationMode.MONITOR:
            self.scan_activity.record_deep_decision(
                symbol,
                "MONITOR_SIGNAL_ONLY",
                "All gates passed in monitor mode",
                "Monitor mode cannot send an order, so the adviser was not asked "
                "and nothing was charged. Run PAPER to exercise the full path.",
                self.clock.now(),
            )
            self.recorder.record_cycle(
                cycle_id=cycle_id,
                context=self._journal_cycle_context(
                    symbol, account.equity, filter_data, market_context=context
                ),
                reason=Reason.OK,
                detail=idea.reason,
                traded=False,
                direction=idea.direction,
                total_score=idea.score,
                score_threshold=self.settings.analysis.confluence.score_threshold,
                signals=list(idea.signals),
                weights=self.settings.analysis.confluence.effective_weights(
                    self.settings.system.mode
                ),
            )
            return False

        proposal = {
            "operation": self.operation.value,
            "account_currency": account.currency,
            "equity": account.equity,
            "margin_free": account.margin_free,
            "sizing": sizing.journal_row(),
            "filters": filter_data,
            "open_positions": len(positions),
            "entry_role": "winner_scalp" if is_addon else "primary",
            "existing_symbol_legs": len(existing_legs),
            "setup_family": idea.setup_family,
            "trade_horizon": idea.horizon,
            "planning_timeframe": idea.planning_timeframe,
            "expected_horizon_minutes": idea.expected_horizon_minutes,
            "quote_age_seconds": (
                max(0.0, (context.now - context.tick.time).total_seconds())
                if context.tick is not None
                else None
            ),
            "market_intelligence": (
                candidate.intelligence.safe_dict() if candidate.intelligence is not None else {}
            ),
            "entry_quality": entry_quality.safe_dict(),
            "horizon_advisories": horizon_advisories,
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
                "selection_score": round(candidate.selection_score, 1),
                "market_priority": candidate.market_priority_label,
                "spread_quality": round(candidate.spread_quality, 4),
                "note": (
                    "The queue handles configured core markets first, then preferred asset "
                    "classes, then the catalogue fallback. Inside a lane, rank 1 means the "
                    "strongest setup after evidence and spread quality. Priority is a reason "
                    "to read it first, never a reason to approve a weak trade."
                ),
            },
            "account_posture": self.posture.brief(),
            "global_market_state": self._world_state,
            "trade_horizon": {
                "name": idea.horizon,
                "setup_family": idea.setup_family,
                "planning_timeframe": idea.planning_timeframe,
                "expected_minutes": idea.expected_horizon_minutes,
                "instruction": self._horizon_review_instruction(idea),
            },
        }
        if horizon_advisories:
            briefing["advisory_evidence"] = horizon_advisories
        if candidate.intelligence is not None:
            briefing["market_intelligence"] = candidate.intelligence.safe_dict()
        # Do we already know what the reviewer is going to say, and why?
        #
        # `veto_memory` above catches the identical proposal coming back. This
        # catches the case it cannot: five GBPCAD longs at five different
        # entries, refused five times as counter-trend. The shape moved every
        # time so the shape memory forgot; the flaw never moved at all.
        #
        # Only ever suppresses a paid call. Every deterministic gate has
        # already run, and an approval on this pair wipes the pattern outright.
        pattern = (
            self.veto_patterns.established(symbol, idea.direction.name, self.clock.now())
            if self.operation is not OperationMode.MONITOR and self._broad_veto_memory_applies(idea)
            # "Only ever suppresses a paid call" — so only when there is one.
            and self._reviews_cost_money()
            else None
        )
        if pattern is not None and self._cached_review(idea, context) is None:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.AI_VETO_PATTERN,
                f"{pattern.describe()}: {veto_readable(pattern.tag)}. Nothing about that "
                f"has changed, so this is not worth asking again yet — an approval on "
                f"{symbol} {idea.direction.name} clears it immediately.",
                signals=list(idea.signals),
                extra={**filter_data, "veto_pattern": pattern.tag},
                total_score=idea.score,
            )
            return False

        # Was this exact market and side paid for a few minutes ago?
        #
        # The shape memory above matches entry AND stop within a quarter of an
        # ATR, which is right for "is this literally the same proposal" and
        # misses the case that actually costs money. Live: EURCAD SHORT
        # reviewed at 10:15:36 and again at 10:18:38, both paid, both refused,
        # confidence 0.28 then 0.32. Three minutes of drift moved the entry
        # past the tolerance, so the memory read it as a new question. It was
        # not one — nothing changes in three minutes that a reviewer reading H4
        # and H1 bars would see.
        #
        # Keyed on symbol and direction alone and deliberately short, so a real
        # intraday turn is not missed. An approval clears it, like every other
        # memory here. A replayed verdict costs nothing and is not rationed.
        # "Not buying the same question again" only bites when it is bought.
        cooling = self._veto_cooldown(idea) if self._reviews_cost_money() else None
        if cooling is not None and self._cached_review(idea, context) is None:
            waited, remaining = cooling
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.AI_VETO_COOLDOWN,
                f"{symbol} {idea.direction.name} was reviewed and refused "
                f"{waited:.0f} min ago; not buying the same question again for "
                f"another {remaining:.0f} min. An approval on this pair clears it.",
                signals=list(idea.signals),
                extra={**filter_data, "veto_cooldown_minutes_left": round(remaining, 1)},
                total_score=idea.score,
            )
            return False

        # Has this cycle already spent its review budget on better ideas?
        #
        # Asked here, before the payload is built and before anything is
        # written, and only when the verdict is not already on file — a
        # replayed verdict costs nothing and must not be rationed.
        #
        # Candidates arrive in the engine's order of conviction, so whatever
        # reaches this point first is the best thing that survived every free
        # gate. Spending the budget there and stopping is what a person with a
        # fixed research budget does. The live account was doing the opposite:
        # paying five cents to be told, in the reviewer's own words, that its
        # "weakest setup of the 10 tradeable candidates" was weak.
        remaining = self._review_budget_left()
        if (
            remaining == 0
            and self.operation is not OperationMode.MONITOR
            and self._cached_review(idea, context) is None
        ):
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.AI_BUDGET_SPENT,
                f"rank {rank} of {of}; this cycle's {self.settings.ai.max_reviews_per_cycle} "
                f"paid reviews went to higher-conviction setups. Not a judgement on this "
                f"one — it was not asked about, and it is first in line next cycle.",
                signals=list(idea.signals),
                extra=filter_data,
                total_score=idea.score,
            )
            return False
        if self.memory.has_evidence():
            briefing["learned_so_far"] = self.memory.briefing(symbol, idea.direction.name)
        # The long memory on top of the local one. It reaches back past the JSON
        # file's retention window and past the forty lessons it can hold, so the
        # reviewer sees "this lesson has now arrived from nine separate trades"
        # rather than the last handful. Empty when no database is configured,
        # which leaves the payload exactly as it was.
        remembered = self.brain.briefing(symbol, idea.direction.name)
        if remembered:
            briefing["learned_over_the_account_lifetime"] = remembered
        # Every theory's reading of this chart, including the ones that did not
        # win. What the losing theories saw is evidence the reviewer cannot get
        # anywhere else, and it is the part most likely to change the answer.
        playbooks = self._playbook_verdict(context)
        if playbooks is not None and playbooks.plays:
            key = (
                "other_theories"
                if self._playbooks_may_execute()
                else "research_only_unvalidated_theories"
            )
            briefing[key] = {
                **playbooks.summary(),
                "authority": (
                    "may corroborate or contradict"
                    if self._playbooks_may_execute()
                    else "context only; negative evidence; may not create or veto this trade"
                ),
            }
        request_payload = build_review_payload(idea, context, proposal, briefing)
        # The matcher's view of this setup, worked out once here and written to
        # Neon on whichever row this becomes. It is the only thing that makes a
        # decision row comparable to a live chart later; without it the entry
        # adviser has no Postgres evidence at all and falls back to a JSONL file
        # on this machine, graded on whether a switched-off reviewer said yes.
        shape_features, shape_family, _shape_horizon = entry_features(request_payload)
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
            if (
                is_addon
                and advice.approved
                and advice.confidence < pyramid_config.minimum_ai_confidence
            ):
                advice = replace(
                    advice,
                    approved=False,
                    thesis=(
                        f"Add-on refused: Claude confidence {advice.confidence:.2f} is below "
                        f"the {pyramid_config.minimum_ai_confidence:.2f} stacking floor. "
                        f"Original verdict: {advice.thesis}"
                    ),
                )
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
            "ai_entry_timing": advice.entry_timing,
            "ai_retest_level": advice.retest_level,
            "ai_entry_boundary": advice.entry_boundary,
            "ai_chase_risk": advice.chase_risk,
        }
        decision_context = {
            **filter_data,
            **ai_data,
            "entry_role": "winner_scalp" if is_addon else "primary",
            "existing_symbol_legs": len(existing_legs),
            "setup_family": idea.setup_family,
            "trade_horizon": idea.horizon,
            "planning_timeframe": idea.planning_timeframe,
            "expected_horizon_minutes": idea.expected_horizon_minutes,
            "trade_thesis": (
                candidate.intelligence.thesis if candidate.intelligence is not None else idea.reason
            ),
            "market_intelligence": (
                candidate.intelligence.safe_dict() if candidate.intelligence is not None else {}
            ),
            "global_market_state": self._world_state,
            "market_scout": self._last_scout.safe_dict(),
        }
        if advice.waiting_for_retest:
            cycle_pk = self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.AI_WAIT_RETEST,
                advice.thesis,
                signals=list(idea.signals),
                extra=decision_context,
            )
            self._record_review_snapshots(cycle_pk, symbol, request_payload)
            self._record_counterfactual(cycle_pk, idea, Reason.AI_WAIT_RETEST)
            return False
        if not advice.approved:
            # Remember it, so the next cycle does not buy this answer again.
            # Only a real verdict is worth remembering: a transport failure is
            # a veto by the fail-closed rule, but it says nothing about the
            # setup, and silencing a symbol for ninety minutes because of one
            # timeout would let a brief outage blank out the catalogue.
            if not advice.error:
                self._remember_veto(idea, context, advice)
                self.veto_patterns.remember(
                    symbol,
                    idea.direction.name,
                    risks=advice.risks,
                    thesis=advice.thesis,
                    now=self.clock.now(),
                )
                self.memory.record_veto(symbol, idea.direction.name, self.clock.now())
            cycle_pk = self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.AI_VETO,
                advice.thesis,
                signals=list(idea.signals),
                extra=decision_context,
                total_score=idea.score,
                features=shape_features,
                setup_family=shape_family,
            )
            self._record_review_snapshots(cycle_pk, symbol, request_payload)
            self._record_counterfactual(cycle_pk, idea, Reason.AI_VETO)
            return False

        revalidation = self._revalidate_approved_entry(
            candidate=candidate,
            reviewed_idea=idea,
            reviewed_account=account,
            reviewed_positions=positions,
            was_addon=is_addon,
            advice=advice,
            latency_seconds=ai_latency_ms / 1000.0,
        )
        if not revalidation.passed:
            cycle_pk = self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                revalidation.reason,
                revalidation.detail,
                signals=list(idea.signals),
                extra={**decision_context, "post_review_revalidation": revalidation.extra},
            )
            self._record_review_snapshots(cycle_pk, symbol, request_payload)
            self._record_counterfactual(cycle_pk, idea, revalidation.reason)
            return False
        assert revalidation.plan is not None
        plan = revalidation.plan
        account = plan.account
        positions = list(plan.positions)
        context = plan.context
        idea = plan.idea
        sizing = plan.sizing
        filter_data = plan.filter_data
        existing_legs = tuple(position for position in positions if position.symbol == symbol)
        decision_context = {
            **decision_context,
            **filter_data,
            "existing_symbol_legs": len(existing_legs),
            "reviewed_entry": candidate.idea.entry,
            "executable_entry": idea.entry,
            "post_review_revalidation": plan.review_binding,
        }
        # Approved: whatever was held against this symbol no longer stands —
        # neither the refused shape nor the reason behind it. The reviewer has
        # just said yes to exactly the pair the pattern called hopeless, so the
        # pattern is wrong by demonstration rather than merely weakened.
        self.veto_memory.clear(symbol, idea.direction.name)
        self.veto_patterns.clear(symbol, idea.direction.name)

        cycle_pk = self.recorder.record_cycle(
            cycle_id=cycle_id,
            context=self._journal_cycle_context(
                symbol, account.equity, decision_context, market_context=context
            ),
            reason=Reason.OK,
            detail=idea.reason,
            traded=self.operation is not OperationMode.MONITOR,
            direction=idea.direction,
            total_score=idea.score,
            score_threshold=self.settings.analysis.confluence.score_threshold,
            signals=list(idea.signals),
            weights=self.settings.analysis.confluence.effective_weights(self.settings.system.mode),
        )
        self.recorder.record_sizing(cycle_pk, sizing)
        self._record_review_snapshots(cycle_pk, symbol, request_payload)
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
                "jarvis-scalp"
                if is_addon
                else (
                    "jarvis-exp-live"
                    if self.operation is OperationMode.EXPERIMENTAL_LIVE
                    else "jarvis"
                )
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
        self._promote_confirmed_entry(
            trade_id=trade_id,
            result=result,
            sizing=sizing,
            spec=spec,
            equity=account.equity,
        )
        # The taken side of the same record. Written after the broker confirms
        # rather than beside the intent, so the long-term memory never carries
        # a position that does not exist — the journal owns the crash window
        # and is the thing reconciliation reads.
        decision_id = self.brain.record_decision(
            decided_at=self.clock.now(),
            symbol=symbol,
            reason=str(Reason.OK),
            mode=self.settings.mode.value,
            direction=idea.direction.name,
            detail=f"opened at {result.filled_price:g}",
            taken=True,
            equity=account.equity,
            conviction=idea.score,
            # WHICH detector found this, not only how high the blend scored.
            # Without it a closed trade is one undifferentiated data point and
            # "should we still be running trend_momentum" has no answer in any
            # table.
            signals=list(idea.signals),
            playbook=idea.setup_family,
            entry=result.filled_price,
            stop_loss=sizing.sl,
            take_profit=sizing.tp,
            filters=dict(decision_context),
            # The shape the entry matcher compares on. Without it this row is
            # invisible to the adviser, which then falls back to a JSONL file
            # graded on whether a switched-off reviewer said yes.
            features=shape_features,
            setup_family=shape_family,
            ai={
                "verdict": "approved",
                "confidence": advice.confidence,
                "reasoning": advice.thesis,
            },
            headlines=self._headlines_for(symbol),
        )
        self._brain_trades[result.position_ticket] = self.brain.record_trade_opened(
            ticket=result.position_ticket,
            decision_id=decision_id,
            symbol=symbol,
            direction=idea.direction.name,
            volume=sizing.volume,
            opened_at=self.clock.now(),
            entry=result.filled_price,
            stop_loss=sizing.sl,
            take_profit=sizing.tp,
            risk_money=sizing.actual_risk_money,
        )
        self.recorder.record_order_attempt(
            trade_id=trade_id, kind="ENTRY", symbol=symbol, result=result
        )
        self.alerts.send(
            f"Opened {'winner scalp ' if is_addon else ''}{symbol} {idea.direction.name} "
            f"{sizing.volume:g} lots, "
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

    def _promote_confirmed_entry(
        self,
        *,
        trade_id: int,
        result: OrderResult,
        sizing: SizingResult,
        spec: InstrumentSpec,
        equity: float,
    ) -> None:
        """Replace intent-time risk arithmetic with the broker's real fill.

        An entry intent is deliberately written before ``order_send`` so a
        crash cannot orphan a live position. Its RR and money risk therefore
        describe the requested quote. Once the broker confirms a different
        fill, every entry-derived number must move with it; otherwise
        postmortems compare P/L with a fictional R and can teach the brain the
        wrong lesson. ETHUSD #134859872 exposed this: the displayed levels were
        roughly 2.51R while the stored intent still reported 0.78R.
        """

        fill = float(result.filled_price)
        if fill <= 0.0:
            self.journal.promote_pending_entry(
                trade_id,
                ticket=result.position_ticket,
                entry_price=fill,
            )
            return

        stop_distance = abs(fill - sizing.sl)
        reward = (sizing.tp - fill) * int(sizing.direction)
        planned_rr = max(reward, 0.0) / stop_distance if stop_distance > 0.0 else 0.0
        commission_per_lot = self.settings.risk.commission_per_lot(spec.asset_class.value)
        risk_money = (spec.money_per_lot(stop_distance) + commission_per_lot) * sizing.volume
        risk_pct = 100.0 * risk_money / equity if equity > 0.0 else 0.0
        self.journal.promote_pending_entry(
            trade_id,
            ticket=result.position_ticket,
            entry_price=fill,
            filled_risk_money=risk_money,
            filled_risk_pct=risk_pct,
            filled_sl_distance_pips=spec.price_to_pips(stop_distance),
            filled_planned_rr=planned_rr,
        )

    def _record_review_snapshots(
        self, cycle_pk: int, symbol: str, request_payload: dict[str, object]
    ) -> None:
        """Persist the exact closed candles the final reviewer was shown.

        The AI JSONL already contains these bars, but the journal's dedicated
        replay table was never populated by the live runner. Storing only
        candidates that actually reached the final gate keeps growth bounded
        while making every veto and executed plan reproducible from SQLite.
        """
        limit = self.settings.journal.snapshot_bars_before
        if limit <= 0:
            return
        timeframes = request_payload.get("timeframes")
        if not isinstance(timeframes, dict):
            return
        for timeframe, raw in timeframes.items():
            if not isinstance(raw, dict):
                continue
            bars = raw.get("closed_bars")
            if not isinstance(bars, list) or not bars:
                continue
            serializable = [bar for bar in bars[-limit:] if isinstance(bar, dict)]
            if serializable:
                self.recorder.record_bar_snapshot(cycle_pk, symbol, str(timeframe), serializable)

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
        features: dict[str, float] | None = None,
        setup_family: str = "",
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
            context=self._journal_cycle_context(symbol, equity, extra),
            reason=reason,
            detail=detail,
            total_score=total_score,
            score_threshold=self.settings.analysis.confluence.score_threshold,
            signals=signals,
            weights=self.settings.analysis.confluence.effective_weights(self.settings.system.mode),
        )
        # SECTION TWO observes here, and here specifically. `drift_burst`
        # FADES, so it fires most often on exactly the symbols the engine
        # declined — hooking it to the traded path would have measured it only
        # on the setups it had nothing to do with. This runs once per symbol
        # per cycle and cannot influence the decision above it, which is
        # already made and already written.
        # SECTION SIX runs its own lane FIRST, and only observes what it did
        # not take. A scalp recorded on paper AND opened live would grade the
        # same decision twice, once against the shadow resolver and once
        # against the account -- and the two would disagree the moment a fill
        # differed from the plan.
        if signals and self._scalp_lane_took_it(cycle_id, symbol, signals, reason, extra):
            self.scan_activity.record_deep_decision(
                symbol,
                "SCALP_OPENED",
                "section six",
                "opened through its own lane",
                self.clock.now(),
            )
            return None
        self._observe_paper_sections(cycle_pk, symbol, signals, reason, extra)
        self.scan_activity.record_deep_decision(
            symbol,
            "DEEP_REJECTED",
            str(reason),
            detail,
            self.clock.now(),
        )
        # The refusals are the bulk of the evidence, and there are roughly two
        # thousand of them for every trade taken. Without them the question
        # "is this system refusing the right things" cannot be asked at all,
        # and right now it is the question with the most money behind it.
        self.brain.record_decision(
            decided_at=self.clock.now(),
            symbol=symbol,
            reason=str(reason),
            mode=self.settings.mode.value,
            detail=detail,
            taken=False,
            equity=equity,
            conviction=total_score,
            filters=dict(extra or {}),
            signals=list(signals or ()),
            headlines=self._headlines_for(symbol),
            # Only the reviewed refusals carry one. A setup stopped at the spread
            # gate has no proposal shape to compare, and the ones that DO reach a
            # review are exactly the rows a counterfactual later grades — which
            # is what turns "we refused this" into "we were wrong to".
            features=features,
            setup_family=setup_family,
        )
        return cycle_pk

    @classmethod
    def _journal_cycle_context(
        cls,
        symbol: str,
        equity: float,
        extra: dict[str, object] | None,
        *,
        market_context: MarketContext | None = None,
    ) -> CycleContext:
        """Promote query-worthy context out of the JSON catch-all columns.

        Filter data has always been kept in `context_json`, but leaving the
        dedicated columns empty made ordinary SQL analysis silently report no
        session, spread or news distance. The JSON remains the full evidence;
        these fields are its indexed, typed projection.
        """
        data = extra or {}
        intelligence = data.get("market_intelligence")
        regime = data.get("volatility_regime")
        if regime is None and isinstance(intelligence, dict):
            regime = intelligence.get("regime")

        return CycleContext(
            symbol=symbol,
            equity=equity,
            atr=cls._signal_atr(market_context) if market_context is not None else None,
            spread_pips=_optional_float(data.get("spread_pips")),
            session=_optional_string(data.get("session")),
            volatility_regime=_optional_string(regime),
            minutes_to_news=_optional_float(data.get("minutes_to_news")),
            extra=dict(data),
        )

    def _record_counterfactual(self, cycle_pk: int, idea: TradeIdea, blocked_by: Reason) -> None:
        """Observe a rejected executable plan without changing the decision."""
        if idea.direction is None or self.recorder.has_unresolved_shadow_trade(
            idea.symbol, idea.direction
        ):
            return
        if min(idea.entry, idea.stop_loss, idea.take_profit) <= 0:
            return
        self.recorder.record_shadow_trade(
            cycle_pk=cycle_pk,
            symbol=idea.symbol,
            direction=idea.direction,
            blocked_by=blocked_by,
            entry_price=idea.entry,
            sl=idea.stop_loss,
            tp=idea.take_profit,
        )

    def _section_for_burst(
        self,
        symbol: str,
        context: MarketContext,
        signal: Signal,
    ) -> Reason | None:
        """Which band this burst belongs to, or None when it is not one.

        TWO BANDS, MEASURED RATHER THAN CHOSEN.

        Section two is the strict reading, `t >= 4.0`: over 6,000 random walks
        that threshold never once fired on noise, and it is correspondingly
        rare — about one event a week per instrument.

        Section four asks what a lower bar buys. At 3.5 the false-positive rate
        is still one in six thousand while detection of a five-bar burst goes
        from 18% to 46% and of a three-bar burst from 41% to 73% — two to three
        times the events for the same reliability.

        AND THEN THE CONDITION THAT DECIDES WHETHER ANY OF IT IS COLLECTABLE.
        Studying intraday reversals, the finding is that on the NYSE "the large
        widening of the bid-ask spread eliminates most of the profits that can
        be achieved by a contrarian strategy", while on the NASDAQ the spread
        stays almost constant and the same strategy yields significant abnormal
        profits. The reversal is real in both markets. Only one of them lets
        you keep it.

        A burst IS a liquidity shock, so the spread widening alongside it is
        the normal case and the expensive one. This compares the live spread
        against the baseline the account has learned for that symbol at that
        hour — a measurement sitting in `spread_observations` since the table
        was written, and never once read for this. Without a baseline the burst
        stays in section two: an unmeasured spread is not a stable one.
        """
        config = self.settings.analysis.drift_burst
        t_stat = abs(float(signal.details.get("t_stat", 0.0)))
        move_bp = abs(float(signal.details.get("move_bp", 0.0)))
        if t_stat < config.wide_net_t_threshold or move_bp < config.minimum_move_bp:
            return None
        if t_stat >= config.t_threshold:
            return Reason.SECTION_2_OBSERVED
        if not config.wide_net_enabled:
            return None
        ratio = self._spread_ratio(symbol, context)
        if ratio is None or ratio > config.wide_net_max_spread_ratio:
            return None
        return Reason.SECTION_4_OBSERVED

    def _spread_ratio(self, symbol: str, context: MarketContext) -> float | None:
        """Live spread over what this symbol normally costs at this hour."""
        # The chain holds its filters in a list, so this finds the one that
        # learned the baseline by name rather than assuming an attribute that
        # does not exist. `baseline` is the only thing wanted from it.
        spread_filter = next(
            (
                item
                for item in getattr(self.filters, "filters", ())
                if getattr(item, "name", "") == "spread" and hasattr(item, "baseline")
            ),
            None,
        )
        if spread_filter is None or context.tick is None:
            return None
        try:
            spec = self.broker.spec(symbol)
            live = float(spec.price_to_pips(context.tick.spread))
            median, count = spread_filter.baseline(symbol, context.now.hour)
        except Exception:  # noqa: BLE001 - a missing baseline is not a failure
            return None
        if median is None or median <= 0 or count < 5 or live <= 0:
            return None
        return live / float(median)

    #: Skip reasons that mean the market is not in a state any paper section
    #: should be pretending to trade. Recording an observation here would
    #: measure a strategy nobody would ever run.
    _NEWS_BLOCKS = frozenset(
        {
            Reason.NEWS_BLACKOUT,
            Reason.NEWS_CALENDAR_UNAVAILABLE,
            Reason.HEADLINE_PRESSURE,
            Reason.HEADLINES_UNAVAILABLE,
        }
    )

    def _observe_scalp(
        self,
        cycle_pk: int,
        symbol: str,
        signals: Sequence[Signal],
        reason: Reason,
        extra: dict | None,
    ) -> None:
        """SECTION SIX on paper: the M1 scalp, never inside a news window.

        THE BLACKOUT IS ENFORCED HERE AND NOT IN THE MODULE, and that is the
        whole point. The module reads bars; it has no calendar. A second copy
        of the news rules living next to it would be a copy that eventually
        disagrees with the real one, and the direction it would disagree in is
        "traded through a release nobody meant to trade through".

        So the observer uses the rules already in force: if this cycle was
        stopped by the news filter, the headline filter, or a calendar that
        could not be reached, no scalp is recorded. The fail-closed case is
        included deliberately — an unreachable calendar is the state the whole
        account already refuses to trade in, and a paper section that quietly
        kept going would be measuring a strategy nobody would run.
        """
        plan = self._scalp_plan(symbol, signals, reason, extra, live=False)
        if plan is None:
            return
        direction, entry, stop, target = plan
        self.recorder.record_shadow_trade(
            cycle_pk=cycle_pk,
            symbol=symbol,
            direction=direction,
            blocked_by=Reason.SECTION_6_OBSERVED,
            entry_price=entry,
            sl=stop,
            tp=target,
        )

    def _scalp_plan(
        self,
        symbol: str,
        signals: Sequence[Signal],
        reason: Reason,
        extra: dict | None,
        *,
        live: bool,
    ) -> tuple[Direction, float, float, float] | None:
        """The one decision, used by both the paper record and the live lane.

        Split out so a scalp that trades and a scalp that is only written down
        can never be different scalps. Two copies of this would drift, and the
        direction they would drift in is "the paper record grades a strategy
        the account does not run".

        `live` changes exactly one thing: which book the concurrency cap counts
        against -- open positions when trading, unresolved shadow rows when
        observing. Everything above it is identical.
        """
        config = self.settings.analysis.candle_momentum
        if not config.enabled or reason in self._NEWS_BLOCKS:
            return None
        # WHERE COMMISSION IS ZERO, AND NOWHERE ELSE.
        #
        # Derived from the commission table rather than typed out as a second
        # list, because two lists of asset classes fall out of step and the
        # direction they fall out of step in is "scalping forex at EUR 5.50 a
        # lot". A scalp's whole margin is a few spreads wide and a fixed fee
        # per lot is not a cost it can carry — the break-even hit rate goes
        # past anything a filter delivers.
        try:
            asset_class = self.broker.spec(symbol).asset_class.value
        except Exception:  # noqa: BLE001 - an unknown spec is not tradable here
            return None
        allowed = config.tradable_asset_classes
        if allowed:
            if asset_class not in allowed:
                return None
        elif self.settings.risk.commission_per_lot(asset_class) > 0:
            return None
        # The concurrency cap, counted against whichever book this scalp would
        # join. Honoured on paper too, or the record measures a position list
        # the account has no room to hold.
        if live:
            if self._open_scalp_count() >= config.max_concurrent:
                return None
        elif self.recorder.open_shadow_count(Reason.SECTION_6_OBSERVED) >= config.max_concurrent:
            return None
        context = self._cycle_contexts.get(symbol)
        if context is None or context.tick is None:
            return None
        signal = next((item for item in signals if item.module == "candle_momentum"), None)
        if signal is None or not signal.score:
            return None
        minutes = (extra or {}).get("minutes_to_news")
        if minutes is not None and float(minutes) <= config.news_clearance_minutes:
            return None
        direction = Direction.LONG if signal.score > 0 else Direction.SHORT
        if self.recorder.has_unresolved_shadow_trade(symbol, direction):
            return None

        series = context.series.get(Timeframe.parse(config.trigger_timeframe))
        if series is None or series.df.empty:
            return None
        candle = series.df.iloc[-1]
        entry = float(context.tick.ask if direction is Direction.LONG else context.tick.bid)
        span = float(candle["high"]) - float(candle["low"])
        if entry <= 0 or span <= 0:
            return None
        # IN SMALL, OUT SMALL, which is the owner's whole description of this:
        # "als het maar een haartje verkeerd gaat eruit, gaat het maar ietsjes
        # goed ook gelijk eruit". The stop is the other side of the candle that
        # triggered it — the level that says the minute was read wrong — and the
        # target is a fraction of that. A scalp that reaches for more is not a
        # scalp, it is a swing trade with a scalp's stop.
        sign = 1.0 if direction is Direction.LONG else -1.0
        stop = entry - sign * span * config.stop_candle_spans
        target = entry + sign * span * config.target_candle_spans
        if min(entry, stop, target) <= 0:
            return None
        return direction, entry, stop, target

    def _open_scalp_count(self) -> int:
        """Live scalp positions this lane is currently holding.

        Counted by the order comment rather than by a separate ledger, because
        a ledger would have to survive a restart and stay in step with manual
        closes -- and the broker's own position list already answers the
        question without either problem.
        """
        try:
            return sum(
                1
                for position in self._managed_positions()
                if str(getattr(position, "comment", "")).startswith(_SCALP_COMMENT)
            )
        except Exception:  # noqa: BLE001 - an unreadable book is not a free slot
            return self.settings.analysis.candle_momentum.max_concurrent

    def _tripped_section(self, idea: TradeIdea) -> str | None:
        """Whether a section behind this idea has switched itself off.

        The verdict is DERIVED from the journal every cycle rather than stored.
        A stored flag would have to survive restarts, stay in step with trades
        resolving out of order, and be reset by hand without anyone forgetting
        — three ways to be wrong about whether real money is switched on. This
        cannot disagree with the evidence because it is the evidence.

        Recomputed once per cycle and cached: it is one query per watched
        section and there are three of them.
        """
        breakers = self.settings.risk.section_breakers
        if not breakers or idea.direction is None:
            return None
        if self._breaker_cache_cycle != self._cycle_serial:
            self._breaker_cache = tripped_modules(self.journal.conn, breakers)
            self._breaker_cache_cycle = self._cycle_serial
        if not self._breaker_cache:
            return None
        sign = 1 if idea.direction is Direction.LONG else -1
        for signal in idea.signals:
            verdict = self._breaker_cache.get(signal.module)
            if verdict is None or not signal.score:
                continue
            if (1 if signal.score > 0 else -1) != sign:
                continue
            return (
                f"{signal.module} stopped itself: {verdict.reason}. Re-arm by "
                f"editing live_enabled_modules."
            )
        return None

    def _scalp_lane_took_it(
        self,
        cycle_id: str,
        symbol: str,
        signals: Sequence[Signal],
        reason: Reason,
        extra: dict | None,
    ) -> bool:
        """The lane, wrapped so it can never take section one down with it.

        THIS BOUNDARY WAS BOUGHT WITH A LIVE INCIDENT. A single malformed log
        call inside the lane raised after the order had already gone through,
        the exception travelled up into `_analyse_candidate`, and the account
        printed "candidate analysis failed; continuing with the rest of the
        batch" -- the same line a genuinely broken detector produces -- on
        every scalp it opened.

        A newly added second path must not be able to do that. Section six is
        an experiment at 0.01 lot; section one is the account. So anything the
        lane raises is logged and dropped here, and the swing decision that was
        already made and already written carries on untouched.

        Deliberately not silent, and deliberately not narrow: an unexpected
        exception from a path that sends live orders is worth an ERROR line
        every single time, and catching only the failures already imagined is
        how the next unimagined one gets through.
        """
        try:
            return self._run_scalp_lane(cycle_id, symbol, signals, reason, extra)
        except Exception:
            log.exception(
                "section six lane failed",
                extra={"event": "scalp_lane_failed", "symbol": symbol},
            )
            return False

    def _run_scalp_lane(
        self,
        cycle_id: str,
        symbol: str,
        signals: Sequence[Signal],
        reason: Reason,
        extra: dict | None,
    ) -> bool:
        """Section six's own route to an order, around the confluence vote.

        WHY IT HAS ONE. The confluence score is a weighted mean of (raw score x
        confidence) over the agreeing modules, and this module's ceiling is
        45 x 0.75 = 33.75 against a bar of 45. It cannot open a trade alone,
        and joining a strong reader makes matters worse rather than better:
        market_structure alone scores 70, and 56.4 with this agreeing. A scalp
        voting in a swing engine drags down every score it touches.

        That is not a threshold to tune. A scalp's evidence is small and
        short-lived because that is what a scalp IS, and a machine built to
        weigh swing evidence will always price it low.

        WHAT THIS LANE DOES NOT GET ITS OWN COPY OF, and the list matters more
        than the list of what it does: the news blackout (`_scalp_plan` refuses
        on the same `_NEWS_BLOCKS` the rest of the account uses), the kill
        switch and capital floor (`_entry_still_allowed`), the margin check,
        and the section breaker. A second copy of any of those would eventually
        disagree with the first, and the direction it would disagree in is
        "traded through something the account had already refused".

        Sized at a FIXED lot, and never above what the ordinary sizer permits.
        The fixed lot is a ceiling rather than an override, so an instrument
        whose stop is too wide for this account is still refused instead of
        forced through at a size the account cannot carry.
        """
        config = self.settings.analysis.candle_momentum
        if not config.own_lane_enabled or not self.settings.mode.is_live:
            return False
        if self.operation is OperationMode.MONITOR:
            return False
        plan = self._scalp_plan(symbol, signals, reason, extra, live=True)
        if plan is None:
            return False
        direction, entry, stop, target = plan

        # The section's own breaker, asked before anything is spent. Built from
        # the journal every cycle, so a section that stopped itself overnight is
        # still stopped now.
        signal = next((item for item in signals if item.module == "candle_momentum"), None)
        if signal is None:
            return False
        blocked = self._tripped_section(
            TradeIdea(
                symbol=symbol,
                approved=True,
                direction=direction,
                score=abs(signal.score),
                confidence=signal.confidence,
                entry=entry,
                stop_loss=stop,
                take_profit=target,
                reason="section six scalp",
                signals=(signal,),
            )
        )
        if blocked is not None:
            self._record_skip(
                cycle_id,
                symbol,
                self.broker.account().equity,
                Reason.SECTION_BREAKER_TRIPPED,
                blocked,
                signals=list(signals),
            )
            return False

        account = self.broker.account()
        spec = self.broker.spec(symbol)
        tick = getattr(self._cycle_contexts.get(symbol), "tick", None)
        spread = float(getattr(tick, "spread", 0.0) or 0.0)
        sizing = PositionSizer(self.settings).size(
            spec=spec,
            equity=account.equity,
            direction=direction,
            entry=entry,
            sl=stop,
            tp=target,
            spread_price=spread,
            # A scalp's reward-to-risk is set by the candle, not by the swing
            # engine's floor, and it is already guarded by
            # `minimum_target_spreads` in the module itself.
            enforce_minimum_rr=False,
        )
        if not sizing.approved:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                sizing.reason,
                sizing.decision.detail,
                signals=list(signals),
            )
            return False

        volume = spec.round_volume_down(min(config.fixed_lots, sizing.volume))
        if volume < spec.volume_min:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.TRADE_SKIPPED_UNDERCAPITALIZED,
                f"scalp lot {volume:g} is below the broker minimum {spec.volume_min:g}",
                signals=list(signals),
            )
            return False
        if volume != sizing.volume and sizing.volume > 0:
            share = volume / sizing.volume
            sizing = replace(
                sizing,
                volume=volume,
                actual_risk_money=sizing.actual_risk_money * share,
                actual_risk_pct=sizing.actual_risk_pct * share,
            )

        state = self.risk.build_state(account, tuple(self._managed_positions()))
        margin = self.risk.check_margin(state, symbol, direction, volume, entry)
        if not margin.approved:
            self._record_skip(
                cycle_id,
                symbol,
                account.equity,
                Reason.INSUFFICIENT_MARGIN,
                margin.detail,
                signals=list(signals),
            )
            return False
        self.risk.assert_not_forbidden(sizing, state)

        cycle_pk = self.recorder.record_cycle(
            cycle_id=cycle_id,
            context=self._journal_cycle_context(symbol, account.equity, dict(extra or {})),
            reason=Reason.OK,
            detail=f"section six scalp; {signal.reasoning}",
            traded=True,
            direction=direction,
            total_score=abs(signal.score) * signal.confidence,
            score_threshold=self.settings.analysis.confluence.score_threshold,
            signals=list(signals),
            weights={"candle_momentum": 1.0},
        )
        self.recorder.record_sizing(cycle_pk, sizing)
        # Re-asked here rather than earlier: the STOP file, the capital floor
        # and the armed contract can all change while the work above runs, and
        # this is the last moment before money moves.
        if not self._entry_still_allowed():
            return False
        trade_id = self.recorder.record_entry_intent(
            cycle_pk=cycle_pk, sizing=sizing, equity_before=account.equity
        )
        result = self.broker.order_send(
            OrderRequest(
                symbol=symbol,
                direction=direction,
                volume=volume,
                sl=spec.normalize_price(stop),
                tp=spec.normalize_price(target),
                reference_price=entry,
                deviation_points=self.settings.mt5.deviation_points,
                magic=self.settings.system.magic_number,
                comment=_SCALP_COMMENT,
            ),
            spec,
        )
        if not result.ok:
            self.journal.abandon_pending_entry(
                trade_id, f"scalp rejected: {result.retcode_name} {result.comment}"
            )
            self.recorder.record_order_attempt(
                trade_id=None, kind="ENTRY", symbol=symbol, result=result
            )
            return False
        self._promote_confirmed_entry(
            trade_id=trade_id,
            result=result,
            sizing=sizing,
            spec=spec,
            equity=account.equity,
        )
        # `extra=`, not keyword arguments. This is a stdlib logger, not
        # structlog, and it took a live crash to find out: Python 3.11 never
        # reaches the bad call because the level is off in tests, while 3.14 on
        # the VPS raises TypeError the moment INFO is enabled. The test suite
        # was green and the account was throwing a traceback on every scalp.
        log.info(
            "section six scalp opened",
            extra={
                "event": "scalp_opened",
                "symbol": symbol,
                "direction": direction.name,
                "volume": volume,
                "entry": result.filled_price,
            },
        )
        return True

    def _observe_paper_sections(
        self,
        cycle_pk: int,
        symbol: str,
        signals: Sequence[Signal] | None,
        reason: Reason = Reason.NO_SIGNAL,
        extra: dict | None = None,
    ) -> None:
        """Every section that is measured and never traded."""
        if signals:
            self._observe_scalp(cycle_pk, symbol, signals, reason, extra)
        self._observe_section_two(cycle_pk, symbol, signals)

    def _observe_section_two(
        self,
        cycle_pk: int,
        symbol: str,
        signals: Sequence[Signal] | None,
    ) -> None:
        """Record what `drift_burst` WOULD have traded, and trade none of it.

        SECTION TWO RUNS AS PAPER. The module is absent from
        `live_enabled_modules`, so the engine zeroes its weight before scoring
        and it cannot influence a single live decision. This writes the plan it
        would have taken into `shadow_trades`, where the resolver already in
        place settles it against real later prices — the same machinery that
        has been grading blocked setups for months.

        Why paper rather than a small live allocation: the research is measured
        on TICK data and this account has M1 bars. The statistic survives that
        coarsening in principle; whether it survives in practice is exactly
        what is unknown, and that question is answerable for free.

        THE PLAN COMES FROM THE BURST'S OWN NUMBERS, not from the swing engine,
        because the swing engine is not involved here. The stop sits beyond the
        burst's extreme — a fade is wrong precisely when the move keeps going,
        and that is the level which says so — and the target is half the move
        being faded. Both scale with what actually happened rather than with a
        fixed ATR multiple, which is the whole point of fading a named event.
        """
        config = self.settings.analysis.drift_burst
        if not config.enabled or not signals:
            return
        # The scan already built and kept this cycle's context for the symbol,
        # so the observer costs no second fetch and reads exactly the bars the
        # engine read rather than a set assembled separately and able to drift.
        context = self._cycle_contexts.get(symbol)
        if context is None or context.tick is None:
            return
        signal = next((item for item in signals if item.module == "drift_burst"), None)
        if signal is None:
            return
        label = self._section_for_burst(symbol, context, signal)
        if label is None:
            return
        t_stat = float(signal.details.get("t_stat", 0.0))
        direction = Direction.SHORT if t_stat > 0 else Direction.LONG
        if self.recorder.has_unresolved_shadow_trade(symbol, direction):
            return

        series = context.series.get(Timeframe.parse(config.timeframe))
        if series is None or len(series.df) < config.drift_window + 1:
            return
        window = series.df.iloc[-config.drift_window :]
        entry = float(context.tick.ask if direction is Direction.LONG else context.tick.bid)
        if entry <= 0:
            return
        extreme = float(
            window["low"].min() if direction is Direction.LONG else window["high"].max()
        )
        travelled = abs(entry - extreme)
        if travelled <= 0:
            return
        sign = 1.0 if direction is Direction.LONG else -1.0
        stop = extreme - sign * travelled * 0.10
        target = entry + sign * travelled * 0.5
        if min(entry, stop, target) <= 0:
            return
        self.recorder.record_shadow_trade(
            cycle_pk=cycle_pk,
            symbol=symbol,
            direction=direction,
            blocked_by=label,
            entry_price=entry,
            sl=stop,
            tp=target,
        )

    def _resolve_counterfactuals(self) -> None:
        """Update passive evidence every fifteen minutes, never the live policy.

        Every pass here is bookkeeping about trades that are already over. None
        of it can open, close, size or protect anything, and the live account is
        in exactly the same state whether it succeeds or not.

        So it must never be able to stop the runner, and it was. A resolver
        compared a bar's integer index against a datetime, pandas raised
        TypeError, and it travelled up through `run_once` and out of
        `run_forever` — the process died at launch, with positions open at the
        broker, because a learning pass could not read a timestamp.

        That is the wrong failure direction by a wide margin. Losing an
        excursion measurement costs one row in a table that refills on the next
        closed trade. Losing the runner costs supervision of real money. Each
        stage is therefore isolated: one that fails is logged with its
        traceback and skipped, and the others still run.
        """
        now = self.clock.now()
        if (
            self._counterfactuals_checked_at is not None
            and now - self._counterfactuals_checked_at < Timeframe.M15.duration
        ):
            return
        self._counterfactuals_checked_at = now
        stages: tuple[tuple[str, Callable[[], object]], ...] = (
            (
                "entry_counterfactuals",
                lambda: resolve_counterfactuals(
                    self.recorder,
                    self.broker,
                    now,
                    on_resolved=self._persist_counterfactual,
                ),
            ),
            (
                "management_baselines",
                lambda: resolve_management_baselines(
                    self.recorder,
                    self.broker,
                    now,
                    on_resolved=self._persist_management_outcome,
                ),
            ),
            ("shadow_book", lambda: self.shadow.resolve(self.broker, now)),
        )
        for name, stage in stages:
            try:
                stage()
            except Exception:
                # Deliberately bare. The whole point is that no exception type
                # from a passive pass may reach the trading loop, and naming a
                # list here would mean the next unanticipated one still does.
                log.exception(
                    "passive evidence pass failed; trading continues",
                    extra={"event": "counterfactual_stage_failed", "stage": name},
                )

    def _persist_management_outcome(self, row: dict[str, object]) -> None:
        """Copy one resolved hold-versus-close comparison into the Neon brain.

        The comparison already ran for every closed trade and already had its
        answer; it lived only in local SQLite, where the layer deciding
        hold-versus-close every second cannot read it. `Brain.management_records`
        reads it back into the briefing, which is what closes the loop.
        """
        try:
            self.brain.record_management_outcome(
                local_trade_id=int(row["trade_id"]),  # type: ignore[arg-type]
                resolved_at=self.clock.now(),
                symbol=str(row.get("symbol", "") or ""),
                direction=str(row.get("direction", "") or ""),
                exit_action=str(row.get("exit_action", "") or ""),
                baseline_pnl_r=float(row["baseline_pnl_r"]),  # type: ignore[arg-type]
                actual_pnl_r=float(row["actual_pnl_r"]),  # type: ignore[arg-type]
                after_exit_best_r=row.get("after_exit_best_r"),  # type: ignore[arg-type]
                after_exit_worst_r=row.get("after_exit_worst_r"),  # type: ignore[arg-type]
                **self._best_exit_available(row),
            )
        except (KeyError, TypeError, ValueError):
            # Learning is not a risk control. A malformed row costs one
            # observation, never the resolver loop.
            return

    def _best_exit_available(self, row: dict[str, object]) -> dict[str, float]:
        """The best moment this trade ever offered, read back from its path.

        The losing side of the exit question, and the side nothing measured.
        HK50 short on 14 August: peak +0.00R, took -0.89R, ran to the broker
        stop with no exit chosen by anything. Whether there was an earlier way
        out is answerable only from the second-by-second path, which is what
        `position_path` is for and the first thing that actually reads it.

        Empty rather than zero when there is no path. A position opened before
        the table existed, or one whose samples were lost, has an unknown best
        moment — and recording zero would claim every one of them was held to
        the perfect second.
        """
        brain_trade = self._brain_trades.get(int(row.get("ticket") or 0))
        if brain_trade is None:
            return {}
        try:
            best = self.brain.best_exit_available(brain_trade)
        except Exception:  # noqa: BLE001 - learning must not break the resolver
            return {}
        if best is None:
            return {}
        return {"best_exit_r": best[0], "minutes_to_best_exit": best[1]}

    def _persist_counterfactual(self, row: dict[str, object]) -> None:
        """Copy one resolved local shadow plan to the durable Neon brain."""
        self.brain.record_counterfactual(**self._normalise_counterfactual(row))

    def _normalise_counterfactual(self, row: dict[str, object]) -> dict[str, object]:
        """Map a local SQLite shadow row to the typed Neon representation."""
        opened_at = datetime.fromisoformat(str(row["opened_at"]))
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=self.clock.now().tzinfo)
        resolved_at = row["resolved_at"]
        if not isinstance(resolved_at, datetime):
            resolved_at = datetime.fromisoformat(str(resolved_at))
        if resolved_at.tzinfo is None:
            resolved_at = resolved_at.replace(tzinfo=self.clock.now().tzinfo)
        return {
            "symbol": str(row["symbol"]),
            "direction": str(row["direction"]),
            "blocked_by": str(row["blocked_by"]),
            "opened_at": opened_at,
            "entry": float(row["entry_price"]),
            "stop_loss": float(row["sl"]),
            "take_profit": float(row["tp"]),
            "resolved_at": resolved_at,
            "outcome": str(row["outcome"]),
            "pnl_r": float(row["pnl_r"]),
        }

    def _sync_counterfactual_history(self) -> None:
        """Backfill local resolved refusals after a VPS rebuild or DB setup."""
        rows = [
            self._normalise_counterfactual(dict(row))
            for row in self.recorder.resolved_shadow_trades()
        ]
        self.brain.record_counterfactuals(rows)

    def _sync_trade_history(self) -> None:
        """Give the brain the closed trades it was switched on too late to see.

        The brain only ever received trades opened after it was armed, so a
        journal holding forty-seven closed trades faced a Neon table holding
        twenty-two. Every learned threshold is built on that table and every
        one of them has a minimum sample, so the account could not read its own
        history and kept using the configured defaults instead.

        Refusals already had this catch-up; the realised trades the thresholds
        are actually made of did not.

        Failures are swallowed: the brain is memory, not a risk control, and a
        Neon hiccup at startup must not stop the account from trading.
        """
        try:
            rows = [
                self._trade_history_row(dict(row)) for row in self.journal.closed_trades_for_brain()
            ]
            sent = self.brain.record_trade_history(rows)
        except Exception:
            log.exception("could not backfill trade history", extra={"event": "brain_backfill"})
            return
        if sent:
            log.info(
                "trade history offered to the brain",
                extra={"event": "brain_backfill", "rows": sent},
            )

    @staticmethod
    def _trade_history_row(row: dict[str, object]) -> dict[str, object]:
        """One local journal trade in the shape the brain's table expects."""
        return {
            "ticket": int(row["ticket"] or 0),
            "symbol": str(row["symbol"]),
            "direction": str(row["direction"]),
            "volume": float(row["volume"] or 0.0),
            "opened_at": str(row["opened_at"]),
            "entry": float(row["entry_price"] or 0.0),
            "stop_loss": float(row["sl"] or 0.0),
            "take_profit": _optional_float(row.get("tp")),
            "risk_money": float(row["risk_money"] or 0.0),
            "closed_at": str(row["closed_at"]),
            "exit_price": _optional_float(row.get("exit_price")),
            "exit_reason": _optional_string(row.get("exit_reason")),
            "pnl_money": _optional_float(row.get("pnl_money")),
            "pnl_r": _optional_float(row.get("pnl_r")),
            "mfe_r": _optional_float(row.get("mfe_r")),
            "mae_r": _optional_float(row.get("mae_r")),
        }

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

    #: Mechanical rules whose whole effect is "the stop moved to here". These
    #: are the only examples of a stop placement this account can produce
    #: without a paid reviewer, and they are what breaks the cold start.
    _STOP_PLACEMENT_ACTIONS = frozenset({"BREAK_EVEN", "PROFIT_LOCK", "TRAIL"})

    def _teach_stop_placement(self, event) -> None:  # type: ignore[no-untyped-def]
        """Record a mechanical stop move as an example the model can learn from.

        The judgement layer may now move a stop, but only to a level a
        comparable past state recorded — and it can only record one by moving a
        stop. Three conditions in a circle, so it could never make the first
        move. With Claude switched off nothing external breaks it either.

        The deterministic rules break it. They place stops all day, at a level
        derived from the peak, and each placement is a real answer to "where
        did the stop go in a state like this". The model reads them as examples
        and can then diverge on what it sees in the market, which is the whole
        point — it starts by copying and ends by judging.

        Not circular: these are outcomes of a rule with its own reasoning, not
        the model's own past output fed back to it. And every example is graded
        later by what the trade actually did.
        """
        if event.action not in self._STOP_PLACEMENT_ACTIONS:
            return
        position = next(
            (item for item in self._managed_positions() if item.ticket == event.ticket), None
        )
        if position is None or not position.sl:
            return
        try:
            context = self.data.get_context(position.symbol)
        except Exception:  # noqa: BLE001 - teaching must never disturb the guard
            return
        if context is None or context.tick is None:
            return
        payload = build_supervision_payload(
            position, context, {"account_equity": self.broker.account().equity}
        )
        verdict = SimpleNamespace(action="tighten_stop", stop_loss=float(position.sl))
        fraction = self._stop_fraction(payload, verdict)
        if fraction is None:
            return
        self.brain.record_supervision(
            trade_id=self._brain_trades.get(event.ticket),
            asked_at=self.clock.now(),
            symbol=position.symbol,
            action="tighten_stop",
            confidence=1.0,
            reasoning=f"mechanical {event.action}: {event.detail}"[:400],
            r_at_the_time=event.r_at_action,
            applied=True,
            model="mechanical_rule",
            direction=position.direction.name,
            features=supervision_features(payload),
            stop_fraction=fraction,
        )

    def _record_management(self, events) -> None:  # type: ignore[no-untyped-def]
        for event in events:
            if event.action in {
                "BROKER_CLOSED_PENDING_HISTORY",
                "EMERGENCY_CLOSE_REJECTED",
                "ORPHAN_CLOSE_REJECTED",
            }:
                self.operation_ledger.reconciliation_failure(self.clock.now())
            row = self.journal.open_trade_by_ticket(event.ticket)
            if row is None:
                continue
            trade_id = int(row["id"])
            self.recorder.record_management_action(
                trade_id,
                action=event.action,
                old_sl=event.old_sl,
                new_sl=event.new_sl,
                old_tp=event.old_tp,
                new_tp=event.new_tp,
                volume_closed=event.volume_closed,
                r_at_action=event.r_at_action,
                note=event.detail,
            )
            # "What happened, when, and why" in the operator's words: BREAK_EVEN
            # at 13:42 on +0.31R, PROFIT_BANKED at 13:58 because the move had
            # stopped running. The journal holds this already; here it survives
            # the machine and can be grouped across months.
            brain_trade = self._brain_trades.get(event.ticket)
            if brain_trade is not None:
                self.brain.record_trade_event(
                    trade_id=brain_trade,
                    happened_at=self.clock.now(),
                    action=event.action,
                    reason=event.detail,
                    r_at_action=event.r_at_action,
                    price=event.exit_price,
                    money=event.pnl_money,
                )
            self._teach_stop_placement(event)
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

    def _record_position_states(self, positions) -> None:  # type: ignore[no-untyped-def]
        """Store the state Jarvis actually evaluated on this guard pass.

        The dashboard health file is a latest-value view and is overwritten.
        This is the append-only counterpart used for replay and learning. A
        position without a journal trade is intentionally skipped: there is no
        trustworthy entry risk against which its R path can be interpreted.
        """
        recorder = getattr(self, "recorder", None)
        journal = getattr(self, "journal", None)
        if recorder is None or journal is None:
            return
        observations = getattr(self.manager, "last_observation", {})
        observed_at = self.clock.now()
        for position in positions:
            row = journal.open_trade_by_ticket(position.ticket)
            if row is None:
                continue
            market = observations.get(position.ticket, {})
            health = (
                self._health_brief(position.ticket)
                if market.get("health_observed", False)
                else {
                    "verdict": "unknown",
                    "severity": None,
                    "action": "hold",
                    "reason": (
                        "health was not reached on this pass; an earlier management gate acted"
                    ),
                    "signals": [],
                }
            )
            recorder.record_position_state(
                trade_id=int(row["id"]),
                position=position,
                observed_at=observed_at,
                market=market,
                health=health,
            )
            # The BRAIN's trade id, not the journal's. `position_path.trade_id`
            # has a foreign key into Neon's `trades`, and feeding it the local
            # SQLite id made every insert violate it — the whole table would
            # have stayed empty while the flush reported rows written.
            brain_trade = self._brain_trades.get(position.ticket)
            if brain_trade is not None:
                self._buffer_position_path(brain_trade, position, row, market, observed_at)
        self._flush_position_path(observed_at)

    def _buffer_position_path(  # type: ignore[no-untyped-def]
        self, trade_id: int, position, row, market, observed_at
    ) -> None:
        """Queue one guard-pass sample of this position for the long memory.

        The same state `record_position_state` writes to local SQLite, held for
        the batched flush below. SQLite lives on a rented VPS and is queried by
        nobody across months; Neon is where "how did positions like this one
        actually behave" becomes answerable.

        `stop_r` and `protected` are what make this a management record rather
        than a price series: they say how much of the money on the table was
        actually safe at each instant. Without them the CADCHF question — was
        holding EUR 2.82 behind a stop below entry right — has no data behind
        it at all.
        """
        try:
            entry = float(position.price_open)
            risk = abs(entry - float(row["sl"]))
            price = float(market.get("price") or 0.0)
            if risk <= 0 or not price:
                return
            sign = int(position.direction)
            r_now = (price - entry) * sign / risk
            stop = float(position.sl or 0.0)
            health = self.manager.last_health.get(position.ticket)
            self._position_path.append(
                {
                    "trade_id": trade_id,
                    "sampled_at": observed_at,
                    "price": price,
                    "r_now": r_now,
                    "peak_r": max(float(row["mfe_r"] or 0.0), r_now),
                    "money": float(position.profit) + float(position.swap),
                    "stop_price": stop or None,
                    "stop_r": ((stop - entry) * sign / risk) if stop else None,
                    "protected": bool(stop and (stop - entry) * sign > 0),
                    "health": getattr(health, "verdict", "") if health else "",
                }
            )
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            # A sample is telemetry. Losing one must never disturb the loop
            # watching real money.
            return

    def _flush_position_path(self, now: datetime) -> None:
        """Hand the buffer to the brain, at most once per flush interval.

        Buffered rather than written per pass because the guard runs about once
        a second on one vCPU: a network round trip inside that loop would make
        the thing watching the money the slowest part of the system. A minute
        of samples across a handful of positions is a few hundred narrow rows
        in a single statement.

        Dropped rather than grown without bound when the brain is unreachable.
        The alternative is a buffer that swells for the life of the process
        while Neon is down, trading a telemetry gap for a memory leak inside
        the position guard.
        """
        if not self._position_path:
            return
        due = (
            self._position_path_flushed_at is None
            or (now - self._position_path_flushed_at).total_seconds()
            >= _POSITION_PATH_FLUSH_SECONDS
        )
        if not due and len(self._position_path) < _POSITION_PATH_MAX_BUFFER:
            return
        batch, self._position_path = self._position_path, []
        self._position_path_flushed_at = now
        try:
            self.brain.record_position_path(batch)
        except Exception:  # noqa: BLE001 - the guard must survive a sick memory
            log.debug("position path flush failed", extra={"event": "position_path_flush_failed"})

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

    def _conviction_now(self, position, context) -> dict[str, object]:  # type: ignore[no-untyped-def]
        """What the entry engine would do on this chart at this moment.

        Every other reading about an open position looks backwards. The health
        engine asks whether the thesis has been damaged; the excursions say
        where price has already been; the account record says what past exits
        earned. None of them answer the question the decision actually turns
        on, which is where this goes NEXT — and that question has two answers
        worth acting on, not one.

        A position that is down while the engine still reads the same direction
        is a trade to sit through: the loss is noise against a thesis that is
        intact, and closing it pays the spread to abandon something the system
        would open again this second. The identical position while the engine
        has flipped is a trade on the wrong side of the market, and the size of
        the loss is beside the point — waiting only adds to it.

        Those two states are indistinguishable in every other field of the
        payload, which is why the supervisor's answer to both was hold.

        Deliberately the same engine, not a second opinion: same modules, same
        weights, same live allowlist, same thresholds. Its verdict is therefore
        comparable to the one that opened the position rather than a different
        instrument disagreeing for reasons of its own.

        Never raises. A read that cannot be produced is reported as unknown,
        because a supervisor told nothing is in the state it was in before this
        existed, while a supervisor told a fabricated agreement is worse off.
        """
        try:
            idea = self.engine.evaluate(context, self.settings.mode)
        except (TradingSystemError, ValueError, KeyError):
            log.warning(
                "no fresh directional read for supervision",
                extra={"event": "supervision_conviction_failed", "symbol": position.symbol},
            )
            return {"verdict": "unknown", "why": "the engine could not read this chart"}

        held = position.direction.name
        if idea.direction is None:
            # Not agreement and not disagreement. The engine declining to open
            # anything here says the chart is unreadable, not that the position
            # is wrong, and collapsing that into "against" would turn every
            # quiet hour into a reason to close.
            return {
                "verdict": "no_setup",
                "position_holds": held,
                "why": idea.reason,
                "means": (
                    "The engine would not open anything here either way. That is not "
                    "evidence against this position; it is the absence of evidence."
                ),
            }

        now_says = idea.direction.name
        agrees = now_says == held
        return {
            "verdict": "agrees" if agrees else "against",
            "position_holds": held,
            "engine_would_take": now_says,
            "score": round(idea.score, 1),
            "confidence": round(idea.confidence, 2),
            "would_be_approved": idea.approved,
            "why": idea.reason,
            "means": (
                "The same engine that opened this trade still reads the same direction on "
                "the live chart. Being down is not by itself a reason to leave a thesis "
                "the system would act on again right now."
                if agrees
                else "The same engine that opened this trade now reads the OPPOSITE "
                "direction on the live chart. This position is on the wrong side of what "
                "the system currently believes, and how far it has already moved against "
                "us does not change that."
            ),
        }

    def _supervise_positions(self, positions) -> None:  # type: ignore[no-untyped-def]
        """Let the adviser manage what is already open, on cadence or evidence.

        The account previously had a strategist and no manager. Something
        decided what to open, and from that moment the position was handed to
        three fixed numbers — break even at 1R, partial at 2R, trail by ATR —
        which cannot tell a trend that is still intact from one that broke
        twenty minutes ago. Both look identical to a rule that only reads the
        R multiple.

        The local layer watches every guard tick. Claude is called either on its
        ordinary cadence or early when that watcher has genuinely new evidence:
        a worse health state, a new profit band, or meaningful give-back. This
        models an attentive human without asking a stochastic model the same
        question sixty times a minute until it eventually changes its mind.
        """
        if (
            isinstance(self.advisor, DisabledAdvisor)
            or not getattr(self.advisor, "supports_dynamic_management", True)
            or self.operation is OperationMode.MONITOR
        ):
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
        self._supervision_due_at = {
            ticket: when for ticket, when in self._supervision_due_at.items() if ticket in live
        }
        self._supervision_snapshots = {
            ticket: snapshot
            for ticket, snapshot in self._supervision_snapshots.items()
            if ticket in live
        }
        for position in positions:
            triggered = self._supervision_trigger(position, now)
            if triggered is None:
                continue
            trigger, snapshot = triggered
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
            # Empty when no database is configured, and an empty key is worse
            # than a missing one: it makes a blank record look like a consulted
            # one. Same guard the entry path uses.
            remembered = self.brain.briefing(position.symbol, position.direction.name)
            payload = build_supervision_payload(
                position,
                context,
                {
                    "trigger": trigger,
                    "operation": self.operation.value,
                    "account_currency": self.broker.account().currency,
                    # Without this the reviewer is told "you are 0.76 up" and
                    # has no way to know whether that is most of a good day or
                    # a rounding error. It is the difference between judging
                    # money and judging a number.
                    "account_equity": self.broker.account().equity,
                    "account_posture": self.posture.brief(),
                    # WOULD I TAKE THIS TRADE, ON THIS CHART, RIGHT NOW?
                    #
                    # The only question here that looks forward. Everything else
                    # in this payload is a post-mortem of a trade in progress:
                    # the health read says whether the original thesis has been
                    # damaged, the excursions say where price has been, the
                    # record says what past exits earned. None of them answer
                    # "where does this go next", and that is the answer the
                    # decision actually turns on — both ways. A position that is
                    # down while the engine still reads the same direction is a
                    # trade to sit through; the same position while the engine
                    # has flipped is one to leave.
                    #
                    # The engine that opened the trade is asked again on the
                    # live chart. It is the same weighted module vote, the same
                    # thresholds, the same allowlist, so its answer is
                    # comparable to the one that produced the entry rather than
                    # a second opinion from a different instrument.
                    "if_i_were_opening_now": self._conviction_now(position, context),
                    "learned_so_far": self.memory.briefing(
                        position.symbol, position.direction.name
                    ),
                    # The account's whole record, on the decision it is actually
                    # about. The entry review has had this since the brain was
                    # built; supervision never did, and supervision is the half
                    # that needed it.
                    #
                    # `what_stepping_in_has_earned` replays every closed trade
                    # against its own untouched stop and target, so it is the
                    # only evidence anywhere in this system about MANAGEMENT
                    # rather than entries. Its own text says "this is the
                    # question you are being asked" — and it was being attached
                    # to the payload that asks whether to OPEN something, while
                    # the payload deciding whether to hold or bank a live
                    # position got the local memory and nothing else.
                    #
                    # So the ten-trade replay showing that PEAK_STALL banked
                    # +0.43R where holding gave +1.92R was read by the reviewer
                    # judging entries, which cannot act on it, and never by the
                    # reviewer judging exits, which is the only one that can.
                    **({"learned_over_the_account_lifetime": remembered} if remembered else {}),
                    # What the per-second layer has been seeing. Without this
                    # the adviser judges a snapshot of the moment it happened
                    # to be asked and cannot know the trade has been bleeding
                    # for ten minutes — the fast layer watched all of it.
                    "mechanical_health": self._health_brief(position.ticket),
                    "peak_r": snapshot.peak_r,
                    "trade_record": self.journal.supervision_context(position.ticket),
                    # The one place a headline's actual words earn their cost.
                    # Everything in `filters.newsfeed` deliberately refuses to
                    # read meaning, because a regex cannot and a sentiment
                    # score at retail latency is buying what somebody else
                    # already bought. A language model reads a headline
                    # properly, and it is being asked about this position
                    # anyway, so the marginal cost is a few hundred tokens.
                    "headlines": self._headlines_for(position.symbol),
                },
            )
            self._supervised_at[position.ticket] = now
            self._supervision_snapshots[position.ticket] = snapshot
            started = time.monotonic()
            verdict = self.advisor.supervise(payload)
            latency_ms = round((time.monotonic() - started) * 1000, 1)
            requested = verdict.review_after_minutes or interval
            floor = self.settings.trade_management.supervision_min_interval_minutes
            cadence = floor if verdict.error else min(interval, max(floor, requested))
            self._supervision_due_at[position.ticket] = now + timedelta(minutes=cadence)
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
                event = None
            else:
                fresh = next(
                    (
                        item
                        for item in self._managed_positions()
                        if item.symbol == position.symbol and item.ticket == position.ticket
                    ),
                    None,
                )
                if fresh is None:
                    event = ManagementEvent(
                        position.ticket,
                        "AI_SUPERVISION_STALE",
                        "position closed or disappeared at the broker while the adviser "
                        "deliberated; no action sent",
                    )
                else:
                    event = self.manager.apply_supervision(fresh, verdict)
            # Written whether or not it was a hold, and whether or not the risk
            # layer carried it out. A "hold" that preceded a full stop-out is
            # the most informative row this table can hold, and marking a
            # refused verdict as acted-upon would credit the adviser for
            # something that never happened.
            self.brain.record_supervision(
                trade_id=self._brain_trades.get(position.ticket),
                asked_at=now,
                symbol=position.symbol,
                action=verdict.action,
                confidence=verdict.confidence,
                reasoning=verdict.reason,
                applied=event is not None,
                latency_ms=latency_ms,
                model=verdict.model,
                direction=position.direction.name,
                # The shape of the position at the moment the question was
                # asked. Without it this row grades the adviser and cannot
                # feed one: the nearest-neighbour model matches on features,
                # and a table without them is why it was reading a file on
                # this machine instead.
                features=supervision_features(payload),
                # Where a stop move actually put the stop, relative to the
                # distance from entry to price. This is what lets the local
                # model name a level later instead of deciding to protect a
                # position and having its verdict refused for carrying none.
                stop_fraction=self._stop_fraction(payload, verdict),
            )
            if event is not None:
                self._record_management([event])

    @staticmethod
    def _stop_fraction(payload, verdict) -> float | None:  # type: ignore[no-untyped-def]
        """Where this verdict put the stop, as a share of entry-to-price.

        None for anything that is not a stop move, and for a move whose level
        falls outside that span — a stop beyond the current price is not a
        placement anything should learn from.
        """
        if verdict.action != "tighten_stop" or verdict.stop_loss is None:
            return None
        try:
            entry = float(payload.get("entry_price") or 0.0)
            price = float(payload.get("price_now") or 0.0)
        except (TypeError, ValueError):
            return None
        if not entry or not price or entry == price:
            return None
        share = (float(verdict.stop_loss) - entry) / (price - entry)
        return share if 0.0 <= share <= 1.0 else None

    def _supervision_trigger(
        self,
        position,
        now: datetime,  # type: ignore[no-untyped-def]
    ) -> tuple[str, _SupervisionSnapshot] | None:
        """Escalate only a materially changed position to the paid adviser."""
        row = self.journal.open_trade_by_ticket(position.ticket)
        if row is None:
            return None
        original_stop = float(row["sl"])
        risk = abs(position.price_open - original_stop)
        if risk <= 0:
            return None
        tick = self.broker.tick(position.symbol)
        price = tick.bid if int(position.direction) > 0 else tick.ask
        r_now = (price - position.price_open) * int(position.direction) / risk
        peak_r = max(float(row["mfe_r"] or 0.0), r_now, 0.0)
        giveback = ((peak_r - r_now) / peak_r) if peak_r > 0 and r_now < peak_r else 0.0
        health = self.manager.last_health.get(position.ticket)
        equity = self.manager.equity
        money = position.profit + position.swap
        snapshot = _SupervisionSnapshot(
            r_now=r_now,
            peak_r=peak_r,
            giveback_fraction=max(0.0, giveback),
            health_verdict=health.verdict if health is not None else "unknown",
            health_severity=health.severity if health is not None else 0.0,
            profit_pct_of_equity=(100.0 * money / equity) if equity > 0 else 0.0,
            # Peak in money, from the same two numbers that produced `money`:
            # r_now and the profit at r_now fix the value of one R, and the peak
            # is that many R. No second source, so the two percentages below can
            # never disagree about what the trade is worth.
            peak_pct_of_equity=(
                (100.0 * money / equity) * (peak_r / r_now)
                if equity > 0 and r_now > 0 and money > 0
                else 0.0
            ),
        )

        previous = self._supervision_snapshots.get(position.ticket)
        last = self._supervised_at.get(position.ticket)
        if previous is None or last is None:
            return "position_opened", snapshot
        config = self.settings.trade_management
        due = self._supervision_due_at.get(
            position.ticket,
            last + timedelta(minutes=config.supervision_interval_minutes),
        )
        if now >= due:
            return "scheduled_review", snapshot
        if not config.supervision_event_driven:
            return None
        if (now - last).total_seconds() < config.supervision_min_interval_minutes * 60:
            return None

        # THE ENGINE HAS TURNED AGAINST THIS POSITION.
        #
        # Ahead of every other trigger below, because it is the only one that
        # says anything about where the trade goes NEXT. The health ladder
        # reports damage already done, the profit and loss ladders report price
        # already travelled; all three describe the past of a position and none
        # of them separates "down and about to come back" from "down and going
        # further". Those are opposite trades and they looked identical.
        #
        # Free to check: the scan evaluates every symbol each cycle, including
        # the ones already held, so this reads a verdict that has been computed
        # anyway rather than paying for a second one.
        #
        # It fires on the CHANGE, not on the state, so a position that is simply
        # on the wrong side does not re-ask every two minutes. And it fires in
        # both directions — a flip back to agreement is exactly as material,
        # because it is the moment a loser becomes worth sitting through.
        latest = self._latest_direction.get(position.symbol)
        if latest is not None:
            held = position.direction.name
            now_against = latest[0] != held
            was_against = previous.engine_against
            if now_against != was_against:
                snapshot = replace(snapshot, engine_against=now_against)
                return (
                    (
                        f"engine_turned_against:{latest[0]}_at_score_{latest[1]:.0f}"
                        if now_against
                        else f"engine_back_in_agreement:{latest[0]}"
                    ),
                    snapshot,
                )
            snapshot = replace(snapshot, engine_against=now_against)

        health_rank = {
            "unknown": -1,
            "healthy": 0,
            "watch": 1,
            "deteriorating": 2,
            "broken": 3,
        }
        if health_rank.get(snapshot.health_verdict, -1) > health_rank.get(
            previous.health_verdict, -1
        ):
            return (
                f"health_worsened:{previous.health_verdict}->{snapshot.health_verdict}",
                snapshot,
            )
        step = config.supervision_profit_step_r
        if int(max(snapshot.r_now, 0.0) / step) > int(max(previous.r_now, 0.0) / step):
            return f"new_profit_milestone:{snapshot.r_now:.2f}R", snapshot
        # The same ladder measured in money, because R is the width of the stop
        # and a wide stop puts the rungs far apart in the only unit that pays
        # for anything. The live CADCHF long sat on 2.2% of the account at
        # 0.44R: the R ladder had last spoken at 0.25R and the next rung was
        # 0.50R, so nothing asked the reviewer about it while EUR 2.82 stood
        # unprotected behind a stop below entry.
        cash_step = config.supervision_profit_step_equity_pct
        if cash_step > 0 and int(max(snapshot.profit_pct_of_equity, 0.0) / cash_step) > int(
            max(previous.profit_pct_of_equity, 0.0) / cash_step
        ):
            return (
                f"new_cash_milestone:{snapshot.profit_pct_of_equity:.2f}%_of_account",
                snapshot,
            )
        # Both ladders again, pointing the other way.
        #
        # Every trigger above wraps its reading in `max(x, 0.0)`, and that is
        # the same clip repeated four times: the losing half of the trade was
        # invisible to all of them. A position could walk from -0.1R to -0.8R —
        # most of its risk budget — and cross nothing. What was left watching it
        # was the fifteen-minute review, and a health reader that speaks about
        # structure rather than about money.
        #
        # So the deliberate look at "is this going further against me, and would
        # I rather take the smaller loss now" happened only by accident of
        # timing. It is the half where being late costs the most, and it was the
        # half nobody was asked about.
        loss_step = config.supervision_loss_step_r
        if loss_step > 0 and int(max(-snapshot.r_now, 0.0) / loss_step) > int(
            max(-previous.r_now, 0.0) / loss_step
        ):
            return f"loss_deepened:{snapshot.r_now:.2f}R", snapshot
        loss_cash = config.supervision_loss_step_equity_pct
        if loss_cash > 0 and int(max(-snapshot.profit_pct_of_equity, 0.0) / loss_cash) > int(
            max(-previous.profit_pct_of_equity, 0.0) / loss_cash
        ):
            return (
                f"loss_deepened_in_cash:{snapshot.profit_pct_of_equity:.2f}%_of_account",
                snapshot,
            )
        # Draining. Both ladders above only speak on a NEW high, so this is the
        # only trigger that fires while a trade is handing profit back — and it
        # inherited the mechanical closer's R floor, which on this account meant
        # it never spoke at all. A EUR 1.00 peak on EUR 133 is 0.42R against a
        # 0.50R arm: the give-back closer stays out of reach, the peak-stall
        # closer stays out of reach, and this, the one thing that only ASKS,
        # stayed out of reach with them. So the trade slid from EUR 1.00 to
        # EUR 0.80 to EUR 0.60 with nothing looking at it.
        #
        # Waking the reviewer is not an exit. It costs one call and returns
        # hold, close or a stop move, and the trade is held or banked on that
        # reading rather than on a threshold — which is the point. The R floor
        # still governs the rules that close positions by themselves; it has no
        # business governing the question.
        threshold = config.supervision_giveback_trigger_fraction
        # The money floor is the cash ladder's own first rung, not the capital
        # protection level. If a peak was large enough to be worth a call on the
        # way up, it is worth one on the way down; borrowing a stricter number
        # here would mean the system paid to admire a gain and then declined to
        # pay when that same gain started leaving.
        armed = snapshot.peak_r >= config.giveback_arm_r or (
            cash_step > 0 and snapshot.peak_pct_of_equity >= cash_step
        )
        if (
            armed
            and snapshot.giveback_fraction >= threshold
            and previous.giveback_fraction < threshold
        ):
            return f"profit_giveback:{snapshot.giveback_fraction:.0%}", snapshot
        return None

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

    def _revalidate_approved_entry(
        self,
        *,
        candidate: AnalysedCandidate,
        reviewed_idea: TradeIdea,
        reviewed_account: AccountSnapshot,
        reviewed_positions: list[Position],
        was_addon: bool,
        advice: Advice,
        latency_seconds: float,
    ) -> _EntryRevalidation:
        """Rebind a paid approval to the price and account that will execute it.

        The review is deliberately the slowest step in the entry path. A market
        order sent after it must not inherit the old entry, size and reward/risk
        merely because the direction stayed the same. Any material change waits
        for a new cycle; a small change is repriced and every deterministic gate
        is run again.
        """

        symbol = candidate.symbol
        original = reviewed_idea
        assert original.direction is not None

        def fail(
            reason: Reason, detail: str, extra: dict[str, object] | None = None
        ) -> _EntryRevalidation:
            return _EntryRevalidation(None, reason, detail, dict(extra or {}))

        try:
            fresh_context = self.data.get_context(symbol, force_refresh=True)
            fresh_account = self.broker.account()
            fresh_positions = self.broker.positions()
            spec = self.broker.spec(symbol)
        except (TradingSystemError, ValueError) as exc:
            return fail(Reason.DATA_UNAVAILABLE, f"post-review refresh failed: {exc}")

        if fresh_account.login != reviewed_account.login:
            return fail(
                Reason.ENTRY_STATE_CHANGED_DURING_REVIEW,
                f"account changed from {reviewed_account.login} to {fresh_account.login} "
                "during AI review",
            )
        before_tickets = {position.ticket for position in reviewed_positions}
        after_tickets = {position.ticket for position in fresh_positions}
        if before_tickets != after_tickets:
            return fail(
                Reason.ENTRY_STATE_CHANGED_DURING_REVIEW,
                "open-position set changed during AI review; rebuild slot, correlation and "
                "add-on context next cycle",
                {
                    "positions_before_review": sorted(before_tickets),
                    "positions_after_review": sorted(after_tickets),
                },
            )

        fresh_state = self.risk.build_state(fresh_account, fresh_positions)
        existing_legs = fresh_state.positions_in(symbol)
        if bool(existing_legs) != was_addon:
            return fail(
                Reason.ENTRY_STATE_CHANGED_DURING_REVIEW,
                "the proposal changed between a primary entry and an add-on during AI review",
            )
        if fresh_context.tick is None:
            return fail(Reason.DATA_UNAVAILABLE, "fresh post-review tick is missing")

        fresh_entry = (
            fresh_context.tick.ask
            if original.direction is Direction.LONG
            else fresh_context.tick.bid
        )
        drift = assess_review_drift(
            fresh_context,
            original.direction,
            original.entry,
            fresh_entry,
            latency_seconds,
            self.settings.analysis.entry_quality,
        )
        binding = {"review_drift": drift.safe_dict()}
        if not drift.passed:
            reason = (
                Reason.DATA_UNAVAILABLE
                if drift.decision is EntryTimingDecision.DATA_UNAVAILABLE
                else Reason.ENTRY_MOVED_DURING_REVIEW
            )
            return fail(reason, drift.detail, binding)

        if advice.entry_boundary is not None:
            boundary_breached = (
                original.direction is Direction.LONG and fresh_entry > advice.entry_boundary
            ) or (original.direction is Direction.SHORT and fresh_entry < advice.entry_boundary)
            if boundary_breached:
                return fail(
                    Reason.ENTRY_MOVED_DURING_REVIEW,
                    f"fresh entry {fresh_entry:.8g} crossed Claude's {original.direction.name} "
                    f"entry boundary {advice.entry_boundary:.8g}",
                    binding,
                )

        fresh_idea = replace(original, entry=fresh_entry)
        confirmed, adverse = self._entry_is_confirmed(fresh_context, fresh_idea)
        if not confirmed:
            assert adverse is not None
            return fail(
                Reason.AWAITING_CONFIRMATION,
                f"fresh price has run {adverse:.2f} ATR against the "
                f"{original.direction.name} after AI review; re-check next cycle",
                binding,
            )
        timing = self._assess_entry_quality(fresh_context, fresh_idea, spec.asset_class)
        binding["entry_quality"] = timing.safe_dict()
        lifecycle = self._observe_setup_lifecycle(fresh_context, fresh_idea, timing)
        binding["setup_lifecycle"] = lifecycle.safe_dict()
        if not lifecycle.may_enter:
            if timing.decision is EntryTimingDecision.DATA_UNAVAILABLE:
                reason = Reason.DATA_UNAVAILABLE
            elif lifecycle.state in (
                SetupState.PULLBACK_RECEIVED,
                SetupState.WAIT_RESUMPTION,
            ):
                reason = Reason.AWAITING_CONFIRMATION
            else:
                reason = Reason.ENTRY_OVEREXTENDED
            return fail(reason, lifecycle.reason, binding)

        risk_decision = self.risk.evaluate(
            fresh_state,
            symbol,
            spec,
            direction=original.direction,
            entry=fresh_entry,
            allow_pyramid=was_addon and self.settings.trade_management.pyramiding.enabled,
        )
        if not risk_decision.approved:
            return fail(risk_decision.reason, risk_decision.detail, binding)

        filter_positions = (
            tuple(position for position in fresh_positions if position.symbol != symbol)
            if was_addon
            else fresh_positions
        )
        filter_verdict, filter_data = self.filters.check(
            FilterContext(
                symbol=symbol,
                spec=spec,
                now=self.clock.now(),
                direction=original.direction,
                tick=fresh_context.tick,
                open_positions=filter_positions,
            )
        )
        if not filter_verdict.passed:
            return fail(filter_verdict.reason, filter_verdict.detail, binding)

        fresh_idea = self._widen_stop_for_costs(fresh_idea, spec)
        affordable, share = self._spread_is_affordable(
            fresh_context, fresh_idea.entry, fresh_idea.stop_loss
        )
        if not affordable:
            return fail(
                Reason.SPREAD_EATS_THE_STOP,
                f"fresh spread is {share:.0%} of the "
                f"{abs(fresh_idea.entry - fresh_idea.stop_loss):.5g} stop",
                binding,
            )
        reachable, needed, runway = self._target_is_reachable_in_time(
            fresh_context, fresh_idea, spec.asset_class.value
        )
        if not reachable:
            assert needed is not None and runway is not None
            return fail(
                Reason.INSUFFICIENT_RUNWAY,
                f"fresh target needs about {needed:.0f} min but only {runway:.0f} min remain",
                binding,
            )

        risk_multiplier = self.risk.risk_multiplier(fresh_state)
        if was_addon:
            risk_multiplier *= self.settings.trade_management.pyramiding.risk_multiplier
        # The stake is decided HERE and not at the first sizing, because this
        # is the first point at which the reviewer's confidence exists. The
        # earlier sizing runs before the question is even asked, so it can only
        # ever use the ordinary stake — which is also the right thing for it to
        # do: that pass exists to find out whether the trade is expressible at
        # all, not to decide how large it should be.
        stake, why_stake = self._conviction_stake(fresh_state, original, advice)
        if stake <= 0:
            return fail(
                Reason.MAX_POSITIONS_REACHED,
                f"the open book already fills the "
                f"{self.settings.risk.max_total_open_risk_pct:.1f}% total-risk ceiling; "
                f"no room for another position until one closes or its stop moves up",
                binding,
            )
        sizing = PositionSizer(self.settings).size(
            spec=spec,
            equity=fresh_account.equity,
            direction=original.direction,
            entry=fresh_idea.entry,
            sl=spec.normalize_price(fresh_idea.stop_loss),
            tp=spec.normalize_price(fresh_idea.take_profit),
            risk_multiplier=risk_multiplier,
            spread_price=fresh_context.tick.spread,
            risk_pct=stake,
        )
        if not sizing.approved:
            return fail(sizing.reason, sizing.decision.detail, binding)
        margin = self.risk.check_margin(
            fresh_state,
            symbol,
            original.direction,
            sizing.volume,
            sizing.entry,
        )
        if not margin.approved:
            return fail(margin.reason, margin.detail, binding)
        self.risk.assert_not_forbidden(
            sizing,
            fresh_state,
            allow_pyramid=was_addon and self.settings.trade_management.pyramiding.enabled,
        )
        return _EntryRevalidation(
            _RevalidatedEntry(
                fresh_account,
                tuple(fresh_positions),
                fresh_context,
                fresh_idea,
                sizing,
                {
                    **filter_data,
                    "entry_quality": timing.safe_dict(),
                    "setup_lifecycle": lifecycle.safe_dict(),
                    "stake_pct": round(stake, 2),
                    "stake_reason": why_stake,
                },
                binding,
            ),
            Reason.OK,
            "fresh entry revalidated",
            binding,
        )

    def _idea_has_embedded_confirmation(self, idea: TradeIdea) -> bool:
        """Whether the quick setup already proved its own closed-bar trigger."""
        config = self.settings.analysis.confluence
        return idea.horizon == "quick" and any(
            idea.setup_family.startswith(module)
            for module in config.quick_embedded_confirmation_modules
        )

    def _broad_veto_memory_applies(self, idea: TradeIdea) -> bool:
        """Broad historical refusals may not erase a genuinely new quick event."""
        config = self.settings.analysis.confluence
        return not (idea.horizon == "quick" and config.quick_events_bypass_broad_veto_memory)

    def _entry_quality_config_for(self, idea: TradeIdea):  # type: ignore[no-untyped-def]
        """Use a quick horizon's own chase tolerance without weakening reversal checks."""
        config = self.settings.analysis.entry_quality
        if idea.horizon != "quick" or config.quick_extension_multiplier == 1.0:
            return config
        multiplier = config.quick_extension_multiplier

        def scaled(values: dict[str, float]) -> dict[str, float]:
            return {name: value * multiplier for name, value in values.items()}

        return config.model_copy(
            update={
                "max_favourable_extension_atr": scaled(config.max_favourable_extension_atr),
                "max_single_bar_body_atr": scaled(config.max_single_bar_body_atr),
                "max_ema_distance_atr": scaled(config.max_ema_distance_atr),
            }
        )

    def _assess_entry_quality(self, context: MarketContext, idea: TradeIdea, asset_class):  # type: ignore[no-untyped-def]
        assert idea.direction is not None
        executable_price = None
        if context.tick is not None:
            executable_price = (
                context.tick.ask if idea.direction is Direction.LONG else context.tick.bid
            )
        return assess_entry_quality(
            context,
            idea.direction,
            asset_class,
            self._entry_quality_config_for(idea),
            executable_price=executable_price,
            signals=idea.signals,
        )

    def _observe_setup_lifecycle(
        self, context: MarketContext, idea: TradeIdea, timing
    ) -> LifecycleDecision:  # type: ignore[no-untyped-def]
        """Keep direction alive while execution timing develops across cycles."""
        config = self.settings.analysis.entry_quality
        if not config.lifecycle_enabled:
            state = SetupState.ENTER_NOW if timing.passed else SetupState.DETECTED
            return LifecycleDecision(state, timing.detail, False)
        if context.tick is None or idea.direction is None:
            return LifecycleDecision(SetupState.DETECTED, "live quote unavailable", False)
        executable_price = (
            context.tick.ask if idea.direction is Direction.LONG else context.tick.bid
        )
        bar_time = None
        with suppress(KeyError, ValueError, IndexError):
            bar_time = context.bars(Timeframe.parse(timing.timeframe)).last_bar_time
        return self.setup_lifecycle.observe(
            idea,
            timing,
            executable_price=executable_price,
            now=self.clock.now(),
            bar_time=bar_time,
        )

    def _reach_failure_is_advisory(self, idea: TradeIdea, reach) -> bool:  # type: ignore[no-untyped-def]
        """A conditional quick event may challenge a base rate, but not fantasy arithmetic."""
        config = self.settings.analysis.confluence
        floor = reach.required_pct * config.quick_reach_hard_floor_ratio
        return (
            idea.horizon == "quick"
            and config.quick_statistical_gates_are_advisory
            and reach.forward_pct >= floor
        )

    def _direction_failure_is_advisory(self, idea: TradeIdea, reach) -> bool:  # type: ignore[no-untyped-def]
        """Small slow-sample disagreement is context; a large skew stays a blocker."""
        config = self.settings.analysis.confluence
        gap = reach.opposite_pct - reach.forward_pct
        return (
            idea.horizon == "quick"
            and config.quick_statistical_gates_are_advisory
            and gap <= config.quick_direction_disadvantage_hard_gap_pct
        )

    @staticmethod
    def _horizon_review_instruction(idea: TradeIdea) -> str:
        if idea.horizon == "quick":
            return (
                "This is a quick trade: the closed M1 trigger and M5 planning structure are "
                "the decision authority. H1/H4 are context and D1/W1 are background only. "
                "Do not veto solely because a slow trend disagrees or an unconditional "
                "travel statistic is modest. Veto when M1/M5 evidence is internally false, "
                "near H1 structure directly invalidates the path, execution is impossible, "
                "or the stated stop/target does not fit the quick thesis."
            )
        if idea.horizon == "intraday":
            return (
                "Judge this intraday trade on its M5/M15/H1 path. H4 is context and D1/W1 "
                "are background, not an automatic veto unless nearer structure confirms the "
                "same contradiction."
            )
        return (
            "This is a swing trade. H1/H4/D1 structure is decision evidence and W1 is "
            "material context; slow-timeframe contradictions may invalidate the plan."
        )

    def _entry_is_confirmed(
        self, context: MarketContext, idea: TradeIdea
    ) -> tuple[bool, float | None]:
        """Is price already running against this trade at the moment of entry?

        Returns `(ok, adverse_atr)`. The second value is how far price has
        travelled the wrong way over the confirmation window, in ATR, and is
        None whenever the question could not be asked.

        The engine finds a level, forms a view, and the order goes out on the
        same tick. Nothing in between ever looks at which way price is moving
        *right now* — so a short is sent into a market climbing through the
        very level it is short against. A live GBPJPY short was that exactly:
        resistance broke upward and the system sold into the break.

        This is the cheapest form of waiting for confirmation and the only one
        that needs no state. It does not ask whether the market has proved the
        thesis right; it asks whether the market has already started proving it
        wrong. A setup refused here is not discarded — every cycle re-examines
        it, and it is taken the moment the adverse move stops.

        Fails open when it cannot measure. A missing timeframe is not evidence
        that price is running against the trade, and the analysis gates that
        judged the setup have all already passed.
        """
        config = self.settings.analysis.confluence
        if not config.require_entry_confirmation or idea.direction is None:
            return True, None
        if self._idea_has_embedded_confirmation(idea):
            return True, None
        try:
            timeframe = Timeframe.parse(config.confirmation_timeframe)
            frame = context.bars(timeframe).df
            reference = atr(frame, period=14)
        except (KeyError, TradingSystemError, ValueError):
            return True, None
        if reference <= 0 or len(frame) <= config.confirmation_bars:
            return True, None

        closes = frame["close"].to_numpy()
        travelled = float(closes[-1] - closes[-1 - config.confirmation_bars])
        # Positive means the market has moved against the intended direction.
        adverse = -travelled * int(idea.direction) / reference
        return adverse <= config.confirmation_max_adverse_atr, adverse

    def _cap_scalp_volume(self, idea: TradeIdea, spec, sizing):  # type: ignore[no-untyped-def]
        """Hold a candle scalp to a fixed lot, whatever the risk model sized.

        THE OWNER'S INSTRUCTION, VERBATIM: "doe gwn MAX 0.01 lot met die
        dingen op xauusd ... trade op candles ... GWN IN EN ERUIT".

        He is right and the sizer was wrong for this. `PositionSizer` spends
        the account's risk budget — 3% of equity — and on a scalp with a stop
        four tenths of an M1 candle wide that buys a lot of lots. The whole
        idea of this section is a position small enough that being wrong costs
        cents, taken often. Those are opposite instructions to the same
        function, so the section states its own size instead.

        What it is worth, measured against his own fill: XAUUSD 0.01 lot,
        4667.47 to 4671.25, EUR 3.24 — EUR 0.86 per dollar of gold. A stop of
        roughly a dollar is EUR 0.86 of risk on a EUR 176 account. Half a
        percent, where the sizer wanted three.

        NEVER UPWARDS. `min` against what the sizer approved, so this can only
        ever risk less. If the sizer refused outright the trade is already
        gone before this runs, and if the account is too small even for the
        broker minimum the existing TRADE_SKIPPED_UNDERCAPITALIZED refusal
        still stands — rounding up to reach a minimum is forbidden here and
        stays forbidden.
        """
        config = self.settings.analysis.candle_momentum
        cap = config.maximum_lots
        if cap <= 0 or not self._is_scalp_idea(idea):
            return sizing
        volume = spec.round_volume_down(min(cap, sizing.volume))
        if volume <= 0 or volume >= sizing.volume:
            return sizing
        scale = volume / sizing.volume
        log.info(
            "scalp held to its fixed lot",
            extra={
                "event": "scalp_volume_capped",
                "symbol": idea.symbol,
                "sized_volume": sizing.volume,
                "volume": volume,
                "risk_pct": round(sizing.actual_risk_pct * scale, 3),
            },
        )
        return replace(
            sizing,
            volume=volume,
            actual_risk_money=sizing.actual_risk_money * scale,
            actual_risk_pct=sizing.actual_risk_pct * scale,
        )

    def _is_scalp_idea(self, idea: TradeIdea) -> bool:
        """Is the candle reader the one actually carrying this trade?

        Carrying it, not merely present on it. A trade the swing engine formed
        on its own evidence is a swing trade even when the candle reader
        happened to agree, and shrinking it to a scalp's lot would quietly
        take the account out of the setups it makes its money on.

        So: this module scored, in the direction being taken, and no other
        weighted module scored harder. That is the same reading the scorecard
        and the section breaker use, which is deliberate — a trade cannot be
        a scalp for the sizer and a swing for the report.
        """
        best = 0.0
        mine = 0.0
        weights = self.settings.analysis.confluence.effective_weights(self.settings.mode)
        for signal in idea.signals:
            if not signal.score or weights.get(signal.module, 0.0) <= 0:
                continue
            if (signal.score > 0) != (idea.direction is Direction.LONG):
                continue
            strength = abs(signal.score) * weights[signal.module]
            if signal.module == "candle_momentum":
                mine = strength
            best = max(best, strength)
        return mine > 0.0 and mine >= best

    def _attach_basket_peers(
        self,
        symbol: str,
        asset_class,  # type: ignore[no-untyped-def]
        context: MarketContext,
    ) -> None:
        """Put what the OTHER indices just did where the analysis can see it.

        A module receives one MarketContext and cannot reach a second
        instrument, which is the whole reason nothing on this account has ever
        compared two markets. `context.meta` is the existing route around that
        — the cost floor arrives the same way — so this writes each peer's
        recent move there before the engine runs.

        THE READINGS ARE FROM THIS SCAN, WHICH MEANS SOME ARE A FEW SECONDS
        OLD. That is not a defect to apologise for: 845 markets cannot be read
        simultaneously by anything, and a real desk comparing indices is always
        looking at prices of slightly different ages. What matters is that the
        age is MEASURED and travels with the reading, so the module can refuse
        a peer that has gone quiet rather than silently treating a closed
        market's last print as current.

        Stored per symbol rather than recomputed, so the cost is one pass over
        a handful of index bars per cycle and not a second data fetch.
        """
        config = self.settings.analysis.basket_divergence
        if not config.enabled:
            return
        group = str(getattr(asset_class, "value", asset_class))
        if group not in config.grouped_asset_classes:
            return
        series = context.series.get(Timeframe.parse(config.timeframe))
        if series is not None and len(series.df) >= config.move_bars + 1:
            closes = series.df["close"]
            start = float(closes.iloc[-(config.move_bars + 1)])
            last = float(closes.iloc[-1])
            if start > 0 and last > 0:
                self._basket_moves[symbol] = (
                    group,
                    (last / start - 1.0) * 10_000.0,
                    series.df.index[-1].to_pydatetime(),
                )
        now = context.now
        peers = [
            PeerMove(
                symbol=other,
                move_bp=move_bp,
                age_seconds=max(0.0, (now - stamp).total_seconds()),
            )
            for other, (other_group, move_bp, stamp) in self._basket_moves.items()
            if other != symbol and other_group == group
        ]
        if peers:
            context.meta[BASKET_META_KEY] = peers

    def _attach_cost_floor(self, symbol: str, context: MarketContext) -> None:
        """Put what a round trip costs where the analysis can see it.

        TWO NUMBERS, AND THE FIRST ONE IS NOT OPTIONAL.

        `round_trip_cost_price` is commission and slippage expressed as a price
        distance, and the target search needs it to be honest. That search
        counted the spread and nothing else, on the reasoning that the sizer
        charges commission again later and counting it twice would refuse
        trades the sizer is about to accept. That reasoning holds at 1.0R,
        where commission is a small part of the reward. It fails at 0.6R, where
        the reward is 40% smaller and the commission is unchanged — which is
        exactly the distance the owner wants to trade, so the estimate would
        have been most optimistic precisely where it matters most.

        `min_stop_for_costs` is the separate, measured-and-retracted experiment
        behind `plan_on_cost_floor`, off by its own evidence.

        Silent on any failure. A symbol whose fee schedule cannot be read plans
        exactly as it did before, which is the old behaviour and not a refusal.
        """
        try:
            spec = self.broker.spec(symbol)
        except (TradingSystemError, ValueError, KeyError):
            return
        cost = self._round_trip_cost_price(spec)
        if cost > 0:
            context.meta["round_trip_cost_price"] = cost
        if not self.settings.analysis.confluence.plan_on_cost_floor:
            return
        floor = self._cost_implied_min_stop(spec)
        if floor > 0:
            context.meta["min_stop_for_costs"] = floor

    def _round_trip_cost_price(self, spec) -> float:  # type: ignore[no-untyped-def]
        """Commission and slippage on one lot, as a distance in price.

        The spread is deliberately NOT included: the caller reads that from the
        live tick, which is more accurate than any average, and adding it here
        would count it twice.

        Zero when the question cannot be answered, so a caller treats it as "no
        information" and not as "no cost".
        """
        if spec is None:
            return 0.0
        try:
            per_price_unit = spec.money_per_lot(1.0)
            cost_per_lot = self.settings.risk.commission_per_lot(
                spec.asset_class.value
            ) + spec.money_per_lot(
                spec.pips_to_price(
                    self.settings.risk.stop_slippage_pips.get(spec.asset_class.value, 0.0)
                )
            )
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return 0.0
        if per_price_unit <= 0 or cost_per_lot <= 0:
            return 0.0
        return float(cost_per_lot / per_price_unit)

    def _cost_implied_min_stop(self, spec) -> float:  # type: ignore[no-untyped-def]
        """The narrowest stop on which commission and slippage stay under the limit.

        A property of the instrument and the fee schedule, not of any one
        setup: cost per lot is fixed and price risk per lot is linear in the
        stop, so the distance at which their ratio equals the limit is solved
        rather than stepped towards.

        And then a hair past it. Aiming at exactly the limit leaves the gate
        re-testing `cost > limit` on a number that has since been through a
        price normalisation and a float division, and it loses: a nine-pip stop
        arrived back as 8.99999 pips and was refused by the very rule the
        widening exists to satisfy.

        Zero when the question does not apply — no limit configured, or a spec
        that cannot price a lot — so every caller treats it as "no floor" and
        not as "a floor of nothing".
        """
        limit = self.settings.risk.max_cost_share_of_risk
        if limit <= 0 or spec is None:
            return 0.0
        try:
            per_price_unit = spec.money_per_lot(1.0)
            cost_per_lot = self.settings.risk.commission_per_lot(
                spec.asset_class.value
            ) + spec.money_per_lot(
                spec.pips_to_price(
                    self.settings.risk.stop_slippage_pips.get(spec.asset_class.value, 0.0)
                )
            )
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return 0.0
        if per_price_unit <= 0 or cost_per_lot <= 0:
            return 0.0
        return float(cost_per_lot / (limit * _COST_MARGIN) / per_price_unit)

    def _widen_stop_for_costs(self, idea: TradeIdea, spec) -> TradeIdea:  # type: ignore[no-untyped-def]
        """Push the stop out until commission and slippage are a small part of it.

        Returns the idea unchanged when the stop is already wide enough, when
        the check is switched off, or when the direction is missing.

        The target is deliberately left where the analysis put it. Widening the
        stop lowers reward-to-risk, and that loss is the honest price of making
        the trade viable — `min_risk_reward` then decides whether it is still
        worth taking. Moving the target to preserve the ratio would be
        inventing a level the chart never offered.
        """
        limit = self.settings.risk.max_cost_share_of_risk
        if limit <= 0 or idea.direction is None:
            return idea

        risk = abs(idea.entry - idea.stop_loss)
        if risk <= 0:
            return idea
        sizer = PositionSizer(self.settings)
        commission = self.settings.risk.commission_per_lot(spec.asset_class.value)
        if sizer._cost_share(spec, risk, commission) <= limit:
            return idea

        # The distance at which cost is exactly the limit — solved rather than
        # stepped outward, so it lands on the boundary instead of near it.
        #
        # And then a hair past it. Aiming at exactly the limit leaves the gate
        # re-testing `cost > limit` on a number that has since been through a
        # price normalisation and a float division, and it loses: a nine-pip
        # stop arrived back as 8.99999 pips and was refused by the very rule
        # the widening exists to satisfy. Widening that fails to clear the gate
        # is worse than not widening, because it moves the stop as well.
        needed = self._cost_implied_min_stop(spec)
        if needed <= 0:
            return idea
        widened = idea.entry - needed * int(idea.direction)

        log.info(
            "widening the stop so the costs are not the trade",
            extra={
                "event": "stop_widened_for_costs",
                "symbol": idea.symbol,
                "from_pips": round(spec.price_to_pips(risk), 1),
                "to_pips": round(spec.price_to_pips(needed), 1),
            },
        )
        return replace(idea, stop_loss=spec.normalize_price(widened))

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

    def _target_is_reachable_in_time(
        self, context: MarketContext, idea: TradeIdea, asset_class: str
    ) -> tuple[bool, float | None, float | None]:
        """Can this target be reached before we force the position flat?

        Returns `(ok, minutes_needed, runway_minutes)`. Both numbers are None
        when the question does not apply — a continuous market, the check
        switched off, or no usable speed reading.

        The runway *filter* enforces a flat floor because it never sees the
        setup. This is the same idea with the setup in hand, and it is the
        version that matters: forty-five minutes is plenty for a target one ATR
        away and nowhere near enough for one five ATR away, and a rule that
        cannot tell those apart is either blocking good trades or letting bad
        ones through, usually both.

        Time-to-target comes from the same normalisation the health reader and
        the higher-timeframe conflict check already use. Net displacement over
        n bars scales with `sqrt(n) x ATR`, not with n, because price does not
        travel in a straight line — so covering d requires `(d / ATR)^2` bars,
        not `d / ATR`. Using the linear form is what makes a distant target
        look forty minutes away when it is really three hours away, and it is
        why an entry can look reasonable at 19:50 and be hopeless in fact.

        `travel_efficiency` divides the distance first, crediting the setup for
        the directional read we think we have. It is an assumption, stated in
        one place and configurable, rather than an optimism baked into the
        shape of the formula.
        """
        config = self.settings.filters.runway
        if not config.enabled or not config.require_reachable_target:
            return True, None, None

        runway_filter = self.filters.find(RunwayFilter)
        if runway_filter is None:
            return True, None, None
        runway = runway_filter.session.minutes_of_runway(self.clock.now(), asset_class)
        if runway is None:
            return True, None, None

        try:
            timeframe = Timeframe.parse(config.speed_timeframe)
            speed = atr(context.bars(timeframe).df, period=14)
        except (KeyError, TradingSystemError, ValueError) as exc:
            # No speed reading means no estimate. The flat floor in the filter
            # has already cleared this moment, so falling through is not the
            # same as skipping the check entirely.
            log.debug(
                "no speed reading for the reachability estimate",
                extra={"symbol": context.symbol, "reason": str(exc)},
            )
            return True, None, runway

        distance = abs(idea.take_profit - idea.entry)
        if speed <= 0 or distance <= 0:
            return True, None, runway

        atr_units = distance / speed / config.travel_efficiency
        bars_needed = atr_units**2
        minutes_needed = bars_needed * timeframe.duration.total_seconds() / 60.0
        return minutes_needed <= runway, minutes_needed, runway

    def _health_brief(self, ticket: int) -> dict[str, object]:
        """The fast layer's current read, for the adviser's payload."""
        health = self.manager.last_health.get(ticket)
        return health.summary() if health is not None else dict(_UNKNOWN_HEALTH)

    def _headlines_for(self, symbol: str) -> list[dict[str, object]]:
        """Recent wire copy touching this instrument, for the reviewer.

        Reaches into the chain's own `HeadlineFilter` rather than building a
        second `HeadlineService`. Two services would each hold their own window
        and their own fetch schedule, so the gate and the reviewer would end up
        disagreeing about what the news was — the same reasoning as
        `FilterChain.find`'s docstring, and the reason that method exists.

        Empty is the normal answer and an honest one. The layer ships disabled
        until its feeds have been verified on the machine running it, and an
        empty list reads downstream as "nothing supplied" rather than as
        "nothing is happening".
        """
        gate = self.filters.find(HeadlineFilter)
        if gate is None or not gate.config.enabled or not gate.service.is_usable():
            return []
        limit = gate.config.headlines_for_reviewer
        if limit <= 0:
            return []
        try:
            spec = self.broker.spec(symbol)
        except Exception:  # noqa: BLE001 - a missing spec must not skip supervision
            return []
        currencies = symbol_currencies(spec.currency_base, spec.currency_profit)
        now = self.clock.now()
        return [
            {
                "minutes_ago": round(item.age_minutes(now)),
                "source": item.source,
                "headline": item.title,
            }
            for item in gate.service.recent_for(currencies, limit=limit)
        ]

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
        if config.range_break:
            chosen.append(
                RangeBreak(BreakConfig(max_spread_share_of_stop=config.max_spread_share_of_stop))
            )
        if config.failed_break:
            chosen.append(
                FailedBreak(
                    FailedBreakConfig(max_spread_share_of_stop=config.max_spread_share_of_stop)
                )
            )
        if config.trend_pullback:
            chosen.append(
                TrendPullback(
                    PullbackConfig(max_spread_share_of_stop=config.max_spread_share_of_stop)
                )
            )
        if not chosen:
            return None
        # A floor above what any theory can express switches them all off while
        # reading like a tightening. "Only take nine-out-of-ten setups" is an
        # instruction nothing in this file can satisfy, because the highest
        # score any of them returns is 95 and most cap in the eighties.
        ceiling = max(playbook.max_conviction for playbook in chosen)
        if config.min_conviction > ceiling:
            log.warning(
                "conviction floor %.0f is above every playbook's ceiling (%.0f); "
                "no short-horizon theory can ever qualify",
                config.min_conviction,
                ceiling,
                extra={
                    "event": "conviction_floor_unreachable",
                    "min_conviction": config.min_conviction,
                    "highest_ceiling": ceiling,
                },
            )
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
        if (
            not self._playbooks_may_execute()
            or verdict is None
            or not self.playbook_config.require_method_agreement
        ):
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
        if not self._playbooks_may_execute() or verdict is None or verdict.best is None:
            return None
        play = verdict.best
        if play.conviction < self.playbook_config.min_conviction:
            return None
        # The higher-timeframe gate lives in the confluence engine, and a
        # promoted play never runs through `evaluate`, so it was skipping the
        # check entirely. Since the swing engine rejects most candidates, the
        # plays are what actually reached the adviser — and it spent the day
        # refusing them in exactly these words: "short entered against the
        # dominant higher-timeframe trend: D1 and W1 both show a clean,
        # sustained uptrend". A five-minute horizon is not a licence to trade
        # into a multi-week one.
        profile = self.settings.analysis.confluence.horizon_profiles["intraday"]
        against_the_tide = self.engine.higher_timeframe_conflict(
            context,
            play.direction,
            timeframes=profile.htf_trend_timeframes,
            threshold=profile.htf_trend_veto,
            minimum_conflicts=profile.minimum_htf_conflicts,
        )
        if against_the_tide is not None:
            log.info(
                "playbook play stood down: %s",
                against_the_tide,
                extra={
                    "event": "play_against_the_tide",
                    "symbol": context.symbol,
                    "playbook": play.playbook,
                },
            )
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
            setup_family=play.playbook,
            horizon="intraday",
            planning_timeframe=profile.planning_timeframe,
            expected_horizon_minutes=play.horizon_minutes,
        )

    def _playbooks_may_execute(self) -> bool:
        """Research stays visible even when negative evidence removes authority."""
        settings = getattr(self, "settings", None)
        if settings is None:  # isolated policy/unit use outside a full runner
            return True
        return not settings.mode.is_live or bool(
            getattr(self.playbook_config, "live_execution_enabled", False)
        )

    def _veto_cooldown(self, idea: TradeIdea) -> tuple[float, float] | None:
        """Minutes since this market and side were refused, and minutes left.

        `_remembered_veto` asks whether the same *proposal* is on file and is
        strict about it on purpose — silencing a setup that genuinely moved
        would be discarding new evidence. This asks the cheaper question the
        strict one cannot: was this market and side bought recently at all.

        Deliberately blind to price. That is the whole point: the case it
        exists for is the one where the price moved a little and the answer did
        not.
        """
        if not self._broad_veto_memory_applies(idea):
            return None
        minutes = self.settings.ai.veto_cooldown_minutes
        if minutes <= 0 or idea.direction is None or self.operation is OperationMode.MONITOR:
            return None
        now = self.clock.now()
        record = self.veto_memory.standing(idea.symbol, idea.direction.name, now)
        if record is None:
            return None
        waited = (now - record.last_seen_at).total_seconds() / 60.0
        if waited < 0 or waited >= minutes:
            return None
        return waited, minutes - waited

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
    def _signal_atr(context: MarketContext, *, timeframe: Timeframe = _REVIEW_TIMEFRAME) -> float:
        """ATR of the signal timeframe, the yardstick for "materially changed".

        Zero when the frame is missing rather than a guessed constant: the
        memory treats a zero as "compare exactly", which errs toward asking
        again. Erring the other way would silence a symbol on a scale nobody
        chose.
        """
        series = context.series.get(timeframe)
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
        cached = self._cached_review(idea, context)
        if cached is not None:
            return cached

        self._reviews_this_cycle += 1
        advice = self.advisor.review(idea, context, proposal, memory)
        if key is not None:
            if len(self._review_cache) >= _REVIEW_CACHE_ENTRIES:
                # Oldest first; insertion order is chronological.
                del self._review_cache[next(iter(self._review_cache))]
            self._review_cache[key] = advice
        return advice

    def _cached_review(self, idea: TradeIdea, context: MarketContext) -> Advice | None:
        """A verdict already on file for this exact setup, or None.

        Separate from `_reviewed` so the budget gate can ask "would this cost
        anything" without committing to the call. A replayed verdict is free
        and must never consume budget — charging for it would make a cheap
        cycle look expensive and starve the candidates that do need asking.
        """
        key = self._review_key(idea, context)
        if key is None:
            return None
        cached = self._review_cache.get(key)
        if cached is None:
            return None
        log.info(
            "reusing the AI verdict for an unchanged setup",
            extra={
                "event": "ai_review_reused",
                "symbol": idea.symbol,
                "direction": idea.direction.name if idea.direction else None,
                "approved": cached.approved,
            },
        )
        # Flagged, not stripped. The token counts stay on the row so the audit
        # trail still shows which call this verdict came from; what must not
        # happen is the spend report charging for them a second time.
        return replace(cached, replayed=True)

    def _reviews_cost_money(self) -> bool:
        """Whether asking the adviser again actually costs anything.

        Three separate gates refuse a setup purely to avoid buying the same
        opinion twice: the exact-proposal veto memory, the broader veto pattern,
        and the per-pair cooldown. Each says so in its own comment — "only ever
        suppresses a paid call", "not buying the same question again for another
        N min", "a replayed verdict costs nothing and is not rationed".

        With `ai.provider: local_history` the adviser is a free
        nearest-neighbour lookup, so all three were saving nothing and refusing
        trades for it. Worse, the memories they read were written by the PAID
        reviewer while it was still switched on, so setups were being blocked by
        recorded opinions of a model that is no longer consulted and can no
        longer revise them. The operator saw precisely that and named it:
        blocked by Claude, after Claude.

        `_review_budget_left` below already draws this distinction. These three
        gates were simply never brought along.
        """
        return bool(getattr(self.advisor, "uses_paid_api", True))

    def _review_budget_left(self) -> int | None:
        """Paid reviews still allowed this cycle. None means no budget is set."""
        if not self._reviews_cost_money():
            return None
        budget = self.settings.ai.max_reviews_per_cycle
        if budget <= 0:
            return None
        return max(0, budget - self._reviews_this_cycle)

    def _review_key(self, idea: TradeIdea, context: MarketContext) -> tuple[object, ...] | None:
        """Identity of the actual proposal Claude was asked to judge.

        M1 was too fast and forced a paid rewording every minute. A fixed H1 was
        then too slow: an M15 plan could materially change while the cache kept
        returning the old answer. The proposal's planning timeframe is the
        honest middle ground.

        Entry timing can deliberately be faster than the planning frame. A H1
        swing that is waiting for an M5 retest must become a new question when
        a new closed M5 bar arrives; otherwise a sound temporary WAIT remains
        cached for nearly an hour after the requested retest has happened.

        Price is bucketed at one quarter of the selected timeframe's ATR. This avoids a
        tick invalidating the cache while ensuring a materially relocated entry
        is a new question. Stop and target distances are included in ATR units,
        so two different plans on the same candle cannot share an approval.
        """
        if idea.direction is None or not context.series:
            return None
        try:
            planning = Timeframe.parse(idea.planning_timeframe)
        except ValueError:
            planning = _REVIEW_TIMEFRAME
        if idea.horizon == "quick":
            quick_frames: list[Timeframe] = []
            for signal in idea.signals:
                if not idea.setup_family.startswith(signal.module):
                    continue
                try:
                    quick_frames.append(Timeframe.parse(str(signal.details.get("timeframe", ""))))
                except ValueError:
                    continue
            if quick_frames:
                planning = min(quick_frames, key=lambda frame: frame.duration)
        entry_quality_config = getattr(
            getattr(self.settings, "analysis", None), "entry_quality", None
        )
        try:
            timing = Timeframe.parse(getattr(entry_quality_config, "timeframe", "M5"))
        except ValueError:
            timing = Timeframe.M5
        if timing.duration < planning.duration and timing in context.series:
            planning = timing
        series = context.series.get(planning)
        if series is None:
            planning = (
                _REVIEW_TIMEFRAME
                if _REVIEW_TIMEFRAME in context.series
                else min(context.series, key=lambda tf: tf.duration)
            )
            series = context.series[planning]
        atr = self._signal_atr(context, timeframe=planning)
        if atr > 0:
            entry_bucket: object = round(idea.entry / (atr * 0.25))
            stop_shape: object = round(abs(idea.entry - idea.stop_loss) / atr, 2)
            target_shape: object = round(abs(idea.take_profit - idea.entry) / atr, 2)
        else:
            entry_bucket = round(idea.entry, 8)
            stop_shape = round(idea.stop_loss, 8)
            target_shape = round(idea.take_profit, 8)
        return (
            idea.symbol,
            idea.direction.name,
            idea.setup_family,
            idea.horizon,
            planning.value,
            series.last_bar_time,
            entry_bucket,
            stop_shape,
            target_shape,
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
        risk_money = float(row["risk_money"])
        trade_id = int(row["id"])
        # The realised result goes into the memory whether or not an adviser is
        # configured. It is the account's own arithmetic, not an opinion, and
        # it is the part of the record that matters most: switching the adviser
        # off should not blind the system to its own P&L.
        realised_r = pnl_money / risk_money if risk_money > 0 else 0.0
        self.memory.record_outcome(
            str(row["symbol"]),
            str(row["direction"]),
            realised_r,
            self.clock.now(),
            trade_id=trade_id,
        )
        ticket = int(row["ticket"]) if row["ticket"] is not None else None
        if ticket is not None:
            self.brain.record_trade_closed(
                ticket=ticket,
                closed_at=closed_at or self.clock.now(),
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl_money=pnl_money,
                pnl_r=realised_r,
                mfe_r=float(row["mfe_r"]) if row["mfe_r"] is not None else None,
                mae_r=float(row["mae_r"]) if row["mae_r"] is not None else None,
            )
        if (
            isinstance(self.advisor, DisabledAdvisor)
            or not getattr(self.advisor, "supports_dynamic_management", True)
            or self.memory.has_reflection(trade_id)
        ):
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
            # -- what actually happened, rather than only how it ended --------
            #
            # Everything above is a departure and an arrival board. A trade
            # that reached +0.9R, had its stop pulled to break even, drifted
            # for forty minutes and closed flat is, in that summary, identical
            # to one that never moved — and the reflection was being asked what
            # went wrong with only that to look at. No wonder the lessons were
            # thin.
            #
            # These four are the journey.
            "best_it_ever_reached_r": float(row["mfe_r"]) if row["mfe_r"] is not None else None,
            "worst_it_ever_reached_r": float(row["mae_r"]) if row["mae_r"] is not None else None,
            "share_of_its_best_it_kept": (
                round(realised_r / float(row["mfe_r"]), 2)
                if row["mfe_r"] and float(row["mfe_r"]) > 0
                else None
            ),
            "what_the_system_did_and_when": self.journal.management_actions_for(int(row["id"])),
        }
        cycle_pk = dict(row).get("cycle_pk")
        if cycle_pk is not None:
            cycle = self.journal.conn.execute(
                "SELECT context_json FROM analysis_cycles WHERE id = ?", (int(cycle_pk),)
            ).fetchone()
            if cycle is not None:
                try:
                    outcome["entry_context"] = json.loads(str(cycle["context_json"] or "{}"))
                except json.JSONDecodeError:
                    outcome["entry_context"] = {}
        reflection = self.advisor.reflect(outcome)
        # This is the step that was missing. The reflection used to be written
        # to a file nothing read, so every lesson was paid for once and then
        # discarded; folding it into the memory is what makes the next review
        # start from more than zero.
        if not reflection.error:
            self.memory.record_reflection(
                outcome,
                reflection.lessons,
                self.clock.now(),
                trade_id=trade_id,
            )
            if reflection.lessons:
                # And into the long memory, one row per lesson rather than one blob
                # per reflection. One row each is what turns "this has now arrived
                # from nine separate trades" into a GROUP BY, which is the whole
                # difference between a pattern and an anecdote.
                self.brain.record_lessons(
                    reflection.lessons,
                    learned_at=self.clock.now(),
                    symbol=str(row["symbol"]),
                    direction=str(row["direction"]),
                    pnl_r=realised_r,
                    trade_id=self._brain_trades.get(int(row["ticket"] or 0)),
                )
                self._brain_trades.pop(int(row["ticket"] or 0), None)
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
            self._managed_positions(),
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
        """Close Jarvis entries and manual positions it explicitly adopted."""
        owned = tuple(positions) if positions is not None else tuple(self._managed_positions())
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
        remaining = tuple(self._managed_positions())
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
                TrendMomentum(candidate.analysis.trend_momentum),
                LiquiditySweep(candidate.analysis.liquidity_sweep),
                LevelReaction(candidate.analysis.level_reaction),
                VolatilityRegime(candidate.analysis.volatility_regime),
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
        self._report_dominant_blocker(summary, deep_reasons)

    def _report_dominant_blocker(self, summary: CycleSummary, reasons: dict[str, int]) -> None:
        """When nothing traded, name the one gate that refused the most.

        The breakdown above already carried this, and four separate times today
        it was read past: monitor mode, a circuit breaker disabled to zero, the
        posture throttle cutting sixteen setups to one, and a losing-streak
        halving that put every trade under the broker's minimum lot. Each time
        the counts were on screen and each time the answer was "why is it not
        trading" asked out loud.

        A list of five numbers is not an answer. The largest one, named, is.
        """
        if summary.trades_opened or not reasons:
            return
        reason, count = max(reasons.items(), key=lambda kv: kv[1])
        log.warning(
            "nothing opened this cycle; the biggest single blocker was %s (%d of %d analysed)",
            reason,
            count,
            summary.deep_analysed,
            extra={
                "event": "no_trades_dominant_reason",
                "reason": reason,
                "count": count,
                "deep_analysed": summary.deep_analysed,
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


def _optional_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)

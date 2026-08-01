"""Typed, validated configuration.

Two rules shape this module:

1. **No magic numbers in code.** Every threshold the system acts on is a field
   here and a value in `config.yaml`.
2. **The hard rules are validated, not documented.** Martingale, grid,
   averaging down, trading without a stop, and a news filter that fails open
   are all rejected at load time. You cannot enable them by editing YAML; you
   would have to edit this file, which is a deliberate, reviewable act.

Secrets never appear here. Credentials come from the environment (`config/.env`)
and are held in `MT5Credentials`, which is constructed separately and never
serialised into a log or a journal row.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.types import TradingMode

Pct = Annotated[float, Field(gt=0, le=100)]
NonNegInt = Annotated[int, Field(ge=0)]


class Base(BaseModel):
    """Strict base: unknown keys are an error, not a shrug.

    A typo like `risk_per_trade_pc` silently falling back to a default is
    exactly the class of bug that shows up as an unexplained loss.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


# ---------------------------------------------------------------- system ---


class SystemConfig(Base):
    mode: TradingMode
    #: Written into every order so we can tell our positions from manual ones.
    magic_number: int = Field(ge=1, le=2_147_483_647)
    loop_interval_seconds: float = Field(gt=0, le=3600)
    #: Filename of the manual kill switch, relative to the project root.
    kill_switch_file: str = "STOP"
    #: Refuse to start if the terminal reports a live account while the config
    #: says backtest/paper, and vice versa.
    enforce_account_mode_match: bool = True


class MT5Config(Base):
    #: Optional explicit path to terminal64.exe. Empty = let MT5 auto-discover.
    terminal_path: str = ""
    connect_timeout_ms: int = Field(default=60_000, ge=1_000, le=300_000)
    portable: bool = False

    max_connect_attempts: int = Field(default=5, ge=1, le=20)
    reconnect_backoff_seconds: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 30.0)

    order_max_attempts: int = Field(default=3, ge=1, le=10)
    order_retry_delay_ms: int = Field(default=250, ge=0, le=5_000)
    #: Maximum acceptable slippage on a market order, in points.
    deviation_points: int = Field(default=10, ge=0, le=200)
    #: Warn if a single MT5 call blocks longer than this.
    slow_call_warn_ms: float = Field(default=1_000.0, gt=0)

    @field_validator("reconnect_backoff_seconds")
    @classmethod
    def _backoff_is_increasing(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("reconnect_backoff_seconds must not be empty")
        if any(v <= 0 for v in value):
            raise ValueError("reconnect backoff values must be positive")
        if list(value) != sorted(value):
            raise ValueError("reconnect backoff must be non-decreasing")
        return value


class LoggingConfig(Base):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    console: bool = True
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    directory: str = "logs"
    filename: str = "trading.jsonl"
    max_bytes: int = Field(default=50 * 1024 * 1024, ge=1024 * 1024)
    backup_count: int = Field(default=20, ge=1)


# ------------------------------------------------------------------ data ---


class DataConfig(Base):
    #: Timeframes loaded every cycle, highest first (HTF bias -> LTF timing).
    timeframes: tuple[str, ...] = ("D1", "H4", "H1", "M15", "M5")
    #: Bars to keep per timeframe. Must cover the slowest indicator lookback
    #: plus the warm-up the backtest uses, or live and backtest disagree.
    bars: dict[str, int] = Field(
        default_factory=lambda: {"D1": 750, "H4": 1500, "H1": 2000, "M15": 2000, "M5": 2000}
    )
    #: Minimum closed bars before a timeframe is considered usable at all.
    min_bars_required: int = Field(default=200, ge=20)
    #: A timeframe is stale if its newest closed bar is older than
    #: `stale_after_bars` * bar duration. Weekend gaps are handled separately.
    stale_after_bars: float = Field(default=3.0, ge=1.0)
    #: Refetch a timeframe at most this often; between refreshes the cache is
    #: served. Set below the shortest timeframe's duration.
    cache_ttl_seconds: float = Field(default=20.0, ge=0.0)
    #: Abort the cycle if more than this fraction of expected bars is missing.
    max_gap_fraction: float = Field(default=0.02, ge=0.0, le=0.5)

    @model_validator(mode="after")
    def _bars_cover_timeframes(self) -> DataConfig:
        missing = [tf for tf in self.timeframes if tf not in self.bars]
        if missing:
            raise ValueError(f"data.bars has no entry for timeframe(s): {missing}")
        too_few = {tf: n for tf, n in self.bars.items() if n < self.min_bars_required}
        if too_few:
            raise ValueError(
                f"data.bars below min_bars_required={self.min_bars_required}: {too_few}"
            )
        return self


class InstrumentsConfig(Base):
    #: Broker-specific suffix, e.g. ".pro" or "m". Appended to every symbol.
    symbol_suffix: str = ""
    #: Tradable symbols per mode. The active mode's list is the whitelist;
    #: anything not on it is not tradable, full stop.
    whitelist: dict[str, tuple[str, ...]]
    #: Symbols that require at least this much equity (account currency).
    #: Gold and indices sit here because one minimum lot risks several percent
    #: of a small account before the setup is even considered.
    min_equity_for_symbol: dict[str, float] = Field(default_factory=dict)

    @field_validator("whitelist")
    @classmethod
    def _known_modes(cls, value: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
        valid = {m.value for m in TradingMode}
        unknown = set(value) - valid
        if unknown:
            raise ValueError(f"unknown mode(s) in instruments.whitelist: {sorted(unknown)}")
        empty = [mode for mode, syms in value.items() if not syms]
        if empty:
            raise ValueError(f"empty whitelist for mode(s): {empty}")
        return value


# ------------------------------------------------------------------ risk ---


class ForbiddenPractices(Base):
    """Explicitly disabled practices. Every field must stay False.

    Present as config so the prohibition is visible where parameters live —
    but validated so that flipping one in YAML fails the load. Changing this
    requires editing source, which shows up in a diff.
    """

    martingale: Literal[False] = False
    grid: Literal[False] = False
    averaging_down: Literal[False] = False
    recovery_lot_increase: Literal[False] = False
    trade_without_stop_loss: Literal[False] = False


class RiskConfig(Base):
    risk_per_trade_pct: Pct = 1.0
    #: Ceiling the sizer will never exceed regardless of setup quality.
    max_risk_per_trade_pct: Pct = 1.0

    max_concurrent_positions: int = Field(default=2, ge=1, le=10)
    max_trades_per_day: int = Field(default=3, ge=1, le=50)
    max_trades_per_week: int = Field(default=10, ge=1, le=200)

    #: All stated as POSITIVE percentages of equity; the manager applies sign.
    daily_loss_limit_pct: Pct = 3.0
    weekly_loss_limit_pct: Pct = 6.0
    #: Drawdown from the equity peak that flattens everything and halts until a
    #: human restarts the system.
    max_drawdown_circuit_breaker_pct: Pct = 15.0

    min_risk_reward: float = Field(default=2.0, ge=1.0)

    #: Anti-martingale: halve risk after a losing streak, restore on a win.
    losing_streak_threshold: int = Field(default=3, ge=2, le=10)
    losing_streak_risk_multiplier: float = Field(default=0.5, gt=0.0, le=1.0)

    #: Trade at a fraction of Kelly. Full Kelly is optimal only if the edge is
    #: known exactly; it never is, and the overestimate is what ruins accounts.
    kelly_fraction: float = Field(default=0.25, gt=0.0, le=0.5)

    forbidden: ForbiddenPractices = ForbiddenPractices()

    @model_validator(mode="after")
    def _coherent(self) -> RiskConfig:
        if self.risk_per_trade_pct > self.max_risk_per_trade_pct:
            raise ValueError(
                f"risk.risk_per_trade_pct ({self.risk_per_trade_pct}%) exceeds "
                f"max_risk_per_trade_pct ({self.max_risk_per_trade_pct}%)"
            )
        if self.weekly_loss_limit_pct < self.daily_loss_limit_pct:
            raise ValueError("weekly loss limit must be >= daily loss limit")
        if self.max_drawdown_circuit_breaker_pct <= self.weekly_loss_limit_pct:
            raise ValueError("circuit breaker must sit above the weekly loss limit")
        if self.max_trades_per_week < self.max_trades_per_day:
            raise ValueError("max_trades_per_week must be >= max_trades_per_day")
        return self


class ModeLimits(Base):
    """The operative limits for one mode.

    Layering rule, applied by `Settings`: **`RiskConfig` is the absolute
    ceiling, the mode is the operative value.** A mode may set a lower
    risk-per-trade, fewer trades or fewer positions than the global config; it
    may never set a higher one, and the validator rejects that at load time.

    Loss limits are the deliberate exception: `micro_live` runs a *wider* daily
    stop (4%) than the global 3%, because on a EUR 100 account a 3% daily stop
    is EUR 3 — less than two full-risk trades — and the point of that phase is
    to accumulate executions, not to protect capital that is already written
    off. It is still bounded by the weekly limit and the circuit breaker.
    """

    max_risk_per_trade_pct: Pct
    max_sl_pips: float = Field(gt=0)
    max_trades_per_day: int = Field(ge=1)
    daily_loss_limit_pct: Pct
    max_concurrent_positions: int = Field(ge=1)
    #: Log every returncode, price, slippage and latency. Costly, and the whole
    #: point of the micro-live phase.
    max_journal_detail: bool = False
    #: Refuse to start below this equity (account currency).
    min_equity: float = Field(default=0.0, ge=0.0)
    #: Refuse to start above this equity — a sign the mode should be upgraded.
    max_equity: float | None = None

    @model_validator(mode="after")
    def _equity_band(self) -> ModeLimits:
        if self.max_equity is not None and self.max_equity <= self.min_equity:
            raise ValueError("mode max_equity must exceed min_equity")
        return self


# --------------------------------------------------------------- filters ---


class NewsWindow(Base):
    minutes_before: int = Field(ge=0, le=1440)
    minutes_after: int = Field(ge=0, le=1440)


class NewsFilterConfig(Base):
    enabled: Literal[True] = True
    #: Hard minimums from the spec. Windows may be widened, never narrowed.
    high_impact: NewsWindow = NewsWindow(minutes_before=60, minutes_after=30)
    extreme_impact: NewsWindow = NewsWindow(minutes_before=120, minutes_after=60)
    #: Events treated as extreme regardless of the feed's own impact rating.
    extreme_keywords: tuple[str, ...] = (
        "FOMC",
        "Non-Farm",
        "Nonfarm",
        "NFP",
        "CPI",
        "Interest Rate Decision",
        "ECB Press Conference",
        "Fed Chair",
        "FOMC Statement",
        "Rate Statement",
    )
    #: No calendar, no trading. This is the fail-safe and it cannot be disabled.
    fail_closed: Literal[True] = True
    #: What to do with an open position when news approaches.
    open_position_action: Literal["break_even", "close", "none"] = "break_even"
    #: How long a cached calendar stays usable if every provider is down.
    max_calendar_age_minutes: int = Field(default=180, ge=15, le=1440)
    refresh_interval_minutes: int = Field(default=30, ge=5, le=720)
    providers: tuple[str, ...] = ("jblanked", "nfs_cached")

    @model_validator(mode="after")
    def _windows_meet_minimums(self) -> NewsFilterConfig:
        if self.high_impact.minutes_before < 60 or self.high_impact.minutes_after < 30:
            raise ValueError("high-impact news window may not be narrower than 60/30 minutes")
        if self.extreme_impact.minutes_before < 120 or self.extreme_impact.minutes_after < 60:
            raise ValueError("extreme-impact news window may not be narrower than 120/60 minutes")
        if len(self.providers) < 2:
            raise ValueError(
                "configure at least two calendar providers; a single source going "
                "down would otherwise stop all trading"
            )
        return self


class SessionFilterConfig(Base):
    enabled: bool = True
    #: Session boundaries in UTC. Broker server time is converted before use.
    sessions: dict[str, tuple[str, str]] = Field(
        default_factory=lambda: {
            "asia": ("00:00", "08:00"),
            "london": ("07:00", "16:00"),
            "newyork": ("12:00", "21:00"),
        }
    )
    tradable_sessions: tuple[str, ...] = ("london", "newyork")
    #: Rollover window: spreads blow out and liquidity vanishes. No entries.
    rollover_block: tuple[str, str] = ("20:45", "21:15")
    block_friday_after: str | None = "19:00"
    block_sunday_before: str | None = "23:00"


class SpreadFilterConfig(Base):
    enabled: bool = True
    #: Block entry when spread exceeds this multiple of the instrument's own
    #: median spread for that hour of day (learned from observation).
    max_spread_multiple: float = Field(default=2.0, gt=1.0)
    #: Absolute ceiling in pips as a backstop while the baseline is warming up.
    absolute_max_pips: dict[str, float] = Field(default_factory=dict)
    #: Observations needed before the learned baseline replaces the fallback.
    min_observations: int = Field(default=200, ge=20)


class CorrelationFilterConfig(Base):
    enabled: bool = True
    #: Block a second position when |rolling correlation| exceeds this and the
    #: directions imply doubled exposure to the same underlying risk.
    max_abs_correlation: float = Field(default=0.7, gt=0.0, le=1.0)
    lookback_bars: int = Field(default=200, ge=50)
    timeframe: str = "H1"


class FiltersConfig(Base):
    news: NewsFilterConfig = NewsFilterConfig()
    session: SessionFilterConfig = SessionFilterConfig()
    spread: SpreadFilterConfig = SpreadFilterConfig()
    correlation: CorrelationFilterConfig = CorrelationFilterConfig()


# ------------------------------------------------------ trade management ---


class TradeManagementConfig(Base):
    #: ATR multiple added beyond the structural level, so a spread widening or
    #: a stop hunt of ordinary size does not take us out.
    sl_atr_buffer_multiple: float = Field(default=0.5, ge=0.0, le=3.0)
    sl_atr_period: int = Field(default=14, ge=2)

    break_even_at_r: float = Field(default=1.0, gt=0.0)
    #: Offset past entry when moving to break even, in ATR multiples, to cover
    #: spread and commission. Break even at exactly entry is a small loss.
    break_even_offset_atr: float = Field(default=0.1, ge=0.0)

    partial_close_at_r: float = Field(default=1.5, gt=0.0)
    partial_close_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)

    trailing_mode: Literal["atr", "structure", "none"] = "atr"
    trailing_atr_multiple: float = Field(default=2.0, gt=0.0)

    #: Close a position that has gone nowhere. Dead capital still carries risk.
    time_exit_hours: float | None = Field(default=24.0, gt=0.0)
    time_exit_min_abs_r: float = Field(default=0.3, ge=0.0)


# --------------------------------------------------------------- journal ---


class JournalConfig(Base):
    database_path: str = "journal/trading.db"
    #: Record every analysis cycle, including the ones that produce no trade.
    #: The skipped setups are where the filter-effectiveness analysis lives.
    record_non_trades: bool = True
    #: Bars saved around an entry so the decision can be replayed later.
    snapshot_bars_before: int = Field(default=200, ge=0)
    snapshot_bars_after: int = Field(default=100, ge=0)
    #: Track what would have happened to setups the filters blocked.
    shadow_trades: bool = True


class MonitoringConfig(Base):
    healthcheck_interval_seconds: float = Field(default=60.0, gt=0)
    #: Alert if a cycle has not completed in this long.
    heartbeat_timeout_seconds: float = Field(default=300.0, gt=0)
    alerts_enabled: bool = False
    alert_channel: Literal["telegram", "discord", "none"] = "none"
    reconciliation_enabled: bool = True


# -------------------------------------------------------------- settings ---


class Settings(Base):
    """The complete, validated configuration tree."""

    system: SystemConfig
    mt5: MT5Config = MT5Config()
    logging: LoggingConfig = LoggingConfig()
    data: DataConfig = DataConfig()
    instruments: InstrumentsConfig
    risk: RiskConfig = RiskConfig()
    modes: dict[str, ModeLimits]
    filters: FiltersConfig = FiltersConfig()
    trade_management: TradeManagementConfig = TradeManagementConfig()
    journal: JournalConfig = JournalConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    @field_validator("modes")
    @classmethod
    def _known_modes(cls, value: dict[str, ModeLimits]) -> dict[str, ModeLimits]:
        valid = {m.value for m in TradingMode}
        unknown = set(value) - valid
        if unknown:
            raise ValueError(f"unknown mode(s) in `modes`: {sorted(unknown)}")
        return value

    @model_validator(mode="after")
    def _active_mode_is_configured(self) -> Settings:
        mode = self.system.mode.value
        if mode not in self.modes:
            raise ValueError(f"`modes` has no entry for the active mode {mode!r}")
        if mode not in self.instruments.whitelist:
            raise ValueError(f"`instruments.whitelist` has no entry for mode {mode!r}")
        return self

    @model_validator(mode="after")
    def _modes_stay_within_global_ceiling(self) -> Settings:
        """No mode may loosen a global risk ceiling."""
        for name, limits in self.modes.items():
            if limits.max_risk_per_trade_pct > self.risk.max_risk_per_trade_pct:
                raise ValueError(
                    f"modes.{name}.max_risk_per_trade_pct "
                    f"({limits.max_risk_per_trade_pct}%) exceeds the global ceiling "
                    f"risk.max_risk_per_trade_pct ({self.risk.max_risk_per_trade_pct}%)"
                )
            if limits.max_trades_per_day > self.risk.max_trades_per_day:
                raise ValueError(
                    f"modes.{name}.max_trades_per_day ({limits.max_trades_per_day}) "
                    f"exceeds risk.max_trades_per_day ({self.risk.max_trades_per_day})"
                )
            if limits.max_concurrent_positions > self.risk.max_concurrent_positions:
                raise ValueError(
                    f"modes.{name}.max_concurrent_positions "
                    f"({limits.max_concurrent_positions}) exceeds "
                    f"risk.max_concurrent_positions ({self.risk.max_concurrent_positions})"
                )
            # A daily stop that can never trigger before the weekly one is not
            # a daily stop; and neither may outrun the circuit breaker.
            if limits.daily_loss_limit_pct >= self.risk.weekly_loss_limit_pct:
                raise ValueError(
                    f"modes.{name}.daily_loss_limit_pct ({limits.daily_loss_limit_pct}%) "
                    f"must stay below risk.weekly_loss_limit_pct "
                    f"({self.risk.weekly_loss_limit_pct}%)"
                )
        return self

    # -- derived views ----------------------------------------------------

    @property
    def mode(self) -> TradingMode:
        return self.system.mode

    @property
    def active_limits(self) -> ModeLimits:
        return self.modes[self.system.mode.value]

    @property
    def active_whitelist(self) -> tuple[str, ...]:
        suffix = self.instruments.symbol_suffix
        return tuple(f"{sym}{suffix}" for sym in self.instruments.whitelist[self.system.mode.value])

    def effective_max_risk_pct(self) -> float:
        """Risk ceiling that applies right now (mode value, globally bounded)."""
        return self.active_limits.max_risk_per_trade_pct

    def effective_risk_pct(self) -> float:
        """Risk actually used per trade: the configured size, capped by the mode."""
        return min(self.risk.risk_per_trade_pct, self.effective_max_risk_pct())

    def effective_max_trades_per_day(self) -> int:
        return self.active_limits.max_trades_per_day

    def effective_daily_loss_limit_pct(self) -> float:
        return self.active_limits.daily_loss_limit_pct

    def effective_max_positions(self) -> int:
        return self.active_limits.max_concurrent_positions

    def symbol_allowed_at_equity(self, symbol: str, equity: float) -> tuple[bool, str]:
        """Whitelist + equity gate for one symbol.

        Returns (allowed, reason). The reason string is journalled verbatim so
        that "why did it not trade gold" is answerable months later.
        """
        if symbol not in self.active_whitelist:
            return False, f"SYMBOL_NOT_WHITELISTED_FOR_{self.mode.value.upper()}"
        suffix = self.instruments.symbol_suffix
        bare = symbol[: -len(suffix)] if suffix and symbol.endswith(suffix) else symbol
        required = self.instruments.min_equity_for_symbol.get(bare)
        if required is not None and equity < required:
            return False, f"SYMBOL_BLOCKED_EQUITY_BELOW_{required:g}"
        return True, "OK"

    def redacted_dump(self) -> dict[str, Any]:
        """Full config as plain data, safe to write to the journal.

        Credentials live outside this tree entirely, so there is nothing to
        strip — but the method exists so that stays true by contract.
        """
        return self.model_dump(mode="json")


class MT5Credentials(Base):
    """Login details, sourced from the environment only.

    `__repr__`/`__str__` are overridden so a stray f-string in a log line
    cannot leak the password.
    """

    login: int
    password: str
    server: str

    def __repr__(self) -> str:
        return f"MT5Credentials(login={self.login}, server={self.server!r}, password=***)"

    __str__ = __repr__

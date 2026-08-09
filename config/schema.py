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

from core.types import Timeframe, TradingMode

Pct = Annotated[float, Field(gt=0, le=100)]
NonNegInt = Annotated[int, Field(ge=0)]
#: A percentage that may also be exactly zero, where zero means "off".
NonNegPct = Annotated[float, Field(ge=0, le=100)]


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
    #: How often open positions are re-checked *between* full cycles, in
    #: seconds. Zero switches the guard off and leaves management on the cycle.
    #:
    #: A full cycle scans the catalogue and can take the better part of a
    #: minute; open money should not have to wait on that. The guard runs only
    #: the mechanical rules — break-even, the trail, peak give-back — which need
    #: nothing but a price and cost nothing but an IPC call, so they can run at
    #: this cadence. It never calls the adviser and never opens anything, so
    #: tightening it costs latency and nothing else.
    guard_interval_seconds: float = Field(default=1.0, ge=0.0, le=60.0)
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
    #: Read-only IPC calls may be repeated after a transient terminal transport
    #: failure. Orders have their own stricter retcode-based retry policy.
    read_max_attempts: int = Field(default=3, ge=1, le=10)
    read_retry_delay_ms: int = Field(default=50, ge=0, le=5_000)
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
    #: Per-timeframe override. 200 bars is a sensible floor for an indicator on
    #: an intraday series and an unreasonable demand on a weekly one: it asks
    #: for four years of history, which a broker simply does not carry for most
    #: share CFDs. COFFEE offered 81 weekly bars, STLAM and SPM 77, and all
    #: three were discarded as having no data at all rather than as having
    #: enough for the timeframes that actually drive the decision.
    min_bars_by_timeframe: dict[str, int] = Field(default_factory=dict)
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
        too_few = {tf: n for tf, n in self.bars.items() if n < self.minimum_bars_for(tf)}
        if too_few:
            raise ValueError(f"data.bars below the minimum for that timeframe: {too_few}")
        return self

    def minimum_bars_for(self, timeframe: str) -> int:
        """Closed bars a timeframe must supply before it is considered usable."""
        return self.min_bars_by_timeframe.get(timeframe, self.min_bars_required)


class InstrumentsConfig(Base):
    #: Broker-specific suffix, e.g. ".pro" or "m". Appended to every symbol.
    symbol_suffix: str = ""
    #: Exact broker names for exceptions to the global suffix. Some brokers
    #: suffix FX symbols but leave metals or indices unchanged.
    symbol_overrides: dict[str, str] = Field(default_factory=dict)
    #: How the tradable universe is decided.
    #:
    #: ``whitelist``  — only the symbols listed below. Blunt but predictable.
    #: ``affordable`` — any symbol the broker offers that the position sizer
    #:   can actually express at the configured risk. The whitelist becomes a
    #:   preference list for reporting rather than a hard gate.
    #:
    #: ``affordable`` is not the looser option it looks like. The whitelist is
    #: a hand-written proxy for "can this account afford this instrument"; the
    #: sizer answers that exactly, per symbol, at the current equity and the
    #: setup's actual stop distance. Replacing the proxy with the exact test
    #: widens what can be *considered* without widening what can be *risked* —
    #: anything unaffordable still returns TRADE_SKIPPED_UNDERCAPITALIZED.
    universe_mode: Literal["whitelist", "affordable"] = "whitelist"
    #: Never tradable, in either mode. For instruments excluded on grounds the
    #: arithmetic cannot see: no reliable data, exotic settlement, known bad
    #: fills. Matched on the canonical name.
    blocklist: tuple[str, ...] = ()
    #: Asset classes the scanner will look at. Empty means all of them.
    #:
    #: This is a horizon control, not a quality one. The stop is 1.5 ATR and the
    #: target twice that, so in *market hours* every class reaches its target in
    #: roughly the same time. What differs is how many hours a week the market
    #: is open to travel it: FX and metals run 120 hours, a London or Milan
    #: share 42. The same setup therefore takes about two calendar days on
    #: EURUSD and over six on CRDA — and with two position slots, a six-day
    #: trade ties up half the account's capacity for a week.
    #:
    #: Values: forex, metal, index, commodity, stock, crypto.
    asset_classes: tuple[str, ...] = ()
    #: Restrict the scan to exactly these symbols, before the suffix is applied.
    #: Empty means the whole catalogue. Use it to watch one instrument closely
    #: rather than to express a preference — everything else stops being looked
    #: at, including better setups elsewhere.
    symbols_only: tuple[str, ...] = ()
    #: Tradable symbols per mode. Under ``whitelist`` this is the hard gate;
    #: under ``affordable`` it is what the startup report describes.
    whitelist: dict[str, tuple[str, ...]]
    #: Symbols that require at least this much equity (account currency).
    #: Gold and indices sit here because one minimum lot risks several percent
    #: of a small account before the setup is even considered.
    min_equity_for_symbol: dict[str, float] = Field(default_factory=dict)

    @field_validator("asset_classes")
    @classmethod
    def _known_asset_classes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        known = {"forex", "metal", "index", "commodity", "stock", "crypto"}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(
                f"unknown instruments.asset_classes: {unknown}; known: {sorted(known)}"
            )
        return value

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

    def broker_symbol(self, canonical: str) -> str:
        """Return the broker's exact symbol name for a canonical symbol."""
        return self.symbol_overrides.get(canonical, f"{canonical}{self.symbol_suffix}")

    def canonical_symbol(self, broker_symbol: str) -> str:
        """Return the canonical name used by risk and equity-gate rules."""
        for canonical, exact in self.symbol_overrides.items():
            if exact == broker_symbol:
                return canonical
        suffix = self.symbol_suffix
        if suffix and broker_symbol.endswith(suffix):
            return broker_symbol[: -len(suffix)]
        return broker_symbol


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


#: `max_trades_per_day` / `max_trades_per_week` set to this mean "no cap".
#:
#: Safe because the count was never the binding constraint. At 2% risk against a
#: 3% daily loss limit the day halts after the second loser, so on a *bad* day
#: the counter never gets a say — the loss limit has already stopped everything.
#: The only day a count cap ever bites is a day that is going well, which is
#: precisely the day it should not.
#:
#: What still bounds an uncapped day, all of it unchanged: the daily and weekly
#: loss limits, the 15% drawdown breaker, the maximum concurrent positions, the
#: fixed risk per trade, and the posture that raises the entry bar after losses.
#: Removing the counter removes a crude proxy, not a protection.
UNLIMITED_TRADES = 0

#: `daily_loss_limit_pct` / `weekly_loss_limit_pct` set to this mean "no limit".
#:
#: These are pacing limits: they stop a bad day or week from continuing, and
#: they reset with the calendar. Switching one off is a real loosening and is
#: not the same class of change as removing the trade counter.
#:
#: What still stops the account depends on what else is switched on.
#:
#: `max_drawdown_circuit_breaker_pct` used to be the answer, and this note used
#: to say so. It can now be zero, and on this account it is: the operator turned
#: it off after it left 6.54 EUR of room on an 88 EUR balance. With it off, the
#: last unconditional stop is the fixed capital floor in
#: `promotion/experimental.py` — an absolute equity level that re-arming cannot
#: move. Everything between here and that floor is a soft limit that resets, or
#: the manual kill switch.
NO_LOSS_LIMIT = 0.0

#: Zero on the circuit breaker means "no automatic peak-to-current halt".
#: Named rather than spelled 0.0 at the comparison sites, because a bare zero
#: there reads as a threshold of nought — which would trip on every account,
#: always, and is the opposite of what it means.
NO_DRAWDOWN_BREAKER = 0.0


class RiskConfig(Base):
    risk_per_trade_pct: Pct = 1.0
    #: Ceiling the sizer will never exceed regardless of setup quality.
    max_risk_per_trade_pct: Pct = 1.0

    max_concurrent_positions: int = Field(default=2, ge=1, le=10)
    #: Trades a day, or 0 for no cap. See `UNLIMITED_TRADES` below for why 0 is
    #: a defensible setting and not a hole in the risk model.
    max_trades_per_day: int = Field(default=3, ge=0, le=50)
    max_trades_per_week: int = Field(default=10, ge=0, le=200)

    #: All stated as POSITIVE percentages of equity; the manager applies sign.
    #: 0 disables the limit — see NO_LOSS_LIMIT for what remains in force.
    daily_loss_limit_pct: NonNegPct = 3.0
    weekly_loss_limit_pct: NonNegPct = 6.0
    #: Drawdown from the equity peak that flattens everything and halts until a
    #: human restarts the system. Zero switches it off.
    #:
    #: Switching it off is a deliberate, consequential choice and the operator
    #: made it: on a EUR 88 account already 8% below its peak, the breaker had
    #: 6.54 EUR of room left, which is three losing trades, while the posture
    #: throttle was simultaneously refusing fifteen of every sixteen setups. The
    #: account could not trade its way out of the drawdown that was blocking it.
    #:
    #: What still stops this account with the breaker off: the fixed capital
    #: floor in `promotion/experimental.py` (an absolute equity level, unmoved by
    #: re-arming), the per-trade risk cap, the maximum concurrent positions, the
    #: daily and weekly loss limits if they are set, and the manual kill switch.
    #: What is gone is the automatic peak-to-current halt — nothing now stops a
    #: slow grind downward before it reaches the floor.
    max_drawdown_circuit_breaker_pct: NonNegPct = 15.0

    #: Let a losing run tighten how the account carries itself — fewer
    #: candidates per cycle, less patience with a stalled trade. See
    #: `risk/posture.py`. Off reports the same numbers and acts on none of them.
    #:
    #: Turned off deliberately. On an 88 EUR account 8.2% below its peak it was
    #: refusing fifteen of every sixteen setups, over a drawdown threshold of
    #: 8.0% that the account was 0.19 EUR the wrong side of — too defensive to
    #: take the trade that would have cleared it. A throttle that cannot be
    #: escaped by trading well is not a throttle, it is a stop.
    posture_throttle: bool = True

    min_risk_reward: float = Field(default=2.0, ge=1.0)

    #: Anti-martingale: halve risk after a losing streak, restore on a win.
    losing_streak_threshold: int = Field(default=3, ge=2, le=10)
    losing_streak_risk_multiplier: float = Field(default=0.5, gt=0.0, le=1.0)

    #: Trade at a fraction of Kelly. Full Kelly is optimal only if the edge is
    #: known exactly; it never is, and the overestimate is what ruins accounts.
    kelly_fraction: float = Field(default=0.25, gt=0.0, le=0.5)

    #: When a trading day rolls over, in UTC. Defaults to the FX rollover
    #: rather than midnight, so the daily loss limit does not reset in the
    #: middle of the New York session.
    day_boundary_utc: str = Field(default="21:00", pattern=r"^\d{2}:\d{2}$")
    #: Margin headroom multiplier. A position that consumes the last of the
    #: free margin leaves nothing for the excursion before its own stop.
    margin_safety_factor: float = Field(default=2.0, ge=1.0, le=10.0)

    forbidden: ForbiddenPractices = ForbiddenPractices()

    @model_validator(mode="after")
    def _coherent(self) -> RiskConfig:
        if self.risk_per_trade_pct > self.max_risk_per_trade_pct:
            raise ValueError(
                f"risk.risk_per_trade_pct ({self.risk_per_trade_pct}%) exceeds "
                f"max_risk_per_trade_pct ({self.max_risk_per_trade_pct}%)"
            )
        both_set = NO_LOSS_LIMIT not in (self.weekly_loss_limit_pct, self.daily_loss_limit_pct)
        if both_set and self.weekly_loss_limit_pct < self.daily_loss_limit_pct:
            raise ValueError("weekly loss limit must be >= daily loss limit")
        # The breaker is the backstop, so it must sit above any limit that can
        # fire before it. With the weekly limit off there is nothing to order
        # it against, and the breaker simply becomes the first thing to trip.
        if (
            self.weekly_loss_limit_pct != NO_LOSS_LIMIT
            and self.max_drawdown_circuit_breaker_pct != NO_DRAWDOWN_BREAKER
            and self.max_drawdown_circuit_breaker_pct <= self.weekly_loss_limit_pct
        ):
            raise ValueError("circuit breaker must sit above the weekly loss limit")
        capped = UNLIMITED_TRADES not in (self.max_trades_per_week, self.max_trades_per_day)
        if capped and self.max_trades_per_week < self.max_trades_per_day:
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
    #: 0 means no cap; see UNLIMITED_TRADES.
    max_trades_per_day: int = Field(ge=0)
    #: 0 disables it; see NO_LOSS_LIMIT.
    daily_loss_limit_pct: NonNegPct
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
    #: Tried in order; the first success wins. Names resolve in
    #: filters/calendar/providers.py::build_providers.
    providers: tuple[str, ...] = ("faireconomy", "tradingview")
    #: Where the last good fetch is cached, relative to the project root.
    cache_path: str = "data/calendar/cache.json"

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
    #: Evening wind-down, UTC. From here until the rollover block ends, no new
    #: entries *and* open positions are flattened. `None` switches it off.
    #:
    #: The rollover block alone was never enough. It stops us opening into the
    #: worst thirty minutes but says nothing about what is already on, so a
    #: position taken in a 1-pip market gets carried into a 6-pip one and is
    #: charged the difference on the way out — the stop is hit by the spread
    #: rather than by the market. On a small account with tight stops that is
    #: not a tail risk, it is what happens every single evening.
    #:
    #: 20:15 UTC is 22:15 in Amsterdam: New York is winding down, London has
    #: been shut for hours, and the book thins out well before the rollover
    #: itself. Deliberately earlier than the block, because being flat *before*
    #: the spread widens is the entire point — flattening at 20:45 would pay
    #: exactly the cost this is meant to avoid.
    evening_flat_from: str | None = "20:15"
    block_friday_after: str | None = "19:00"
    block_sunday_before: str | None = "23:00"
    #: These markets are not forced through the FX London/New York calendar.
    #: A missing/stale broker tick still blocks them later in the chain.
    continuous_asset_classes: tuple[str, ...] = ("crypto",)
    continuous_maintenance_block: tuple[str, str] = ("23:55", "00:10")
    #: Exchange/CFD markets follow the broker's actual quote rather than the FX
    #: London/New York preference. Quote freshness and spread remain hard gates,
    #: so a closed exchange is still rejected without pretending every stock,
    #: index, metal and commodity shares one FX timetable.
    broker_hours_asset_classes: tuple[str, ...] = (
        "stock",
        "index",
        "metal",
        "commodity",
        "unknown",
    )
    #: Only these asset classes are forcibly flattened by the FX evening rule.
    #: Applying it to every non-crypto CFD closed healthy stock and metal swing
    #: positions for a calendar they do not follow.
    evening_flat_asset_classes: tuple[str, ...] = ("forex",)


class SpreadFilterConfig(Base):
    enabled: bool = True
    #: Block entry when spread exceeds this multiple of the instrument's own
    #: median spread for that hour of day (learned from observation).
    max_spread_multiple: float = Field(default=2.0, gt=1.0)
    #: Absolute ceiling in pips as a backstop while the baseline is warming up.
    absolute_max_pips: dict[str, float] = Field(default_factory=dict)
    #: Cross-asset backstop. One "pip" is not comparable between BTC, AAPL
    #: and EURUSD, so unknown symbols are bounded as basis points of mid-price.
    max_spread_bps: dict[str, float] = Field(
        default_factory=lambda: {
            "forex": 2.0,
            "crypto": 5.0,
            "stock": 20.0,
            "index": 10.0,
            "metal": 10.0,
            "commodity": 20.0,
        }
    )
    #: A non-zero weekend quote can still be Friday's close. Treat age as a
    #: hard gate before interpreting the numerical spread.
    max_tick_age_seconds: dict[str, int] = Field(
        default_factory=lambda: {
            "forex": 30,
            "crypto": 30,
            "stock": 120,
            "index": 30,
            "metal": 30,
            "commodity": 60,
        }
    )
    #: Observations needed before the learned baseline replaces the fallback.
    min_observations: int = Field(default=200, ge=20)
    #: How long spread observations are kept before pruning.
    retention_days: int = Field(default=60, ge=7, le=730)


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


# ------------------------------------------------------------- analysis ---


class MarketStructureConfig(Base):
    enabled: bool = True
    signal_timeframe: str = "H1"
    bias_timeframe: str = "H4"
    internal_swing_lookback: int = Field(default=1, ge=1, le=10)
    external_swing_lookback: int = Field(default=3, ge=2, le=20)
    atr_period: int = Field(default=14, ge=2, le=200)
    bos_close_buffer_atr: float = Field(default=0.05, ge=0.0, le=1.0)
    equal_level_tolerance_atr: float = Field(default=0.10, gt=0.0, le=1.0)
    bos_score: float = Field(default=70.0, gt=0.0, le=100.0)
    minimum_confidence: float = Field(default=0.50, ge=0.0, lt=1.0)
    full_confidence_break_atr: float = Field(default=0.50, gt=0.0, le=5.0)

    @field_validator("signal_timeframe", "bias_timeframe")
    @classmethod
    def _supported_timeframe(cls, value: str) -> str:
        from core.types import Timeframe

        return Timeframe.parse(value).value

    @model_validator(mode="after")
    def _internal_is_finer_than_external(self) -> MarketStructureConfig:
        if self.internal_swing_lookback >= self.external_swing_lookback:
            raise ValueError("internal swing lookback must be smaller than external lookback")
        if self.signal_timeframe == self.bias_timeframe:
            raise ValueError("market-structure signal and bias timeframes must differ")
        return self


class TrendMomentumConfig(Base):
    bias_timeframe: str = "H4"
    signal_timeframe: str = "H1"
    fast_ema: int = Field(default=20, ge=2, le=500)
    slow_ema: int = Field(default=50, ge=3, le=1000)
    slope_lookback: int = Field(default=5, ge=1, le=100)
    atr_period: int = Field(default=14, ge=2, le=200)
    invalidation_lookback: int = Field(default=12, ge=2, le=500)
    score: float = Field(default=65.0, gt=0.0, le=100.0)
    base_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    separation_confidence_scale: float = Field(default=0.20, ge=0.0, le=5.0)
    maximum_confidence: float = Field(default=0.90, ge=0.0, le=1.0)

    @field_validator("bias_timeframe", "signal_timeframe")
    @classmethod
    def _supported_timeframe(cls, value: str) -> str:
        return Timeframe.parse(value).value

    @model_validator(mode="after")
    def _coherent(self) -> TrendMomentumConfig:
        if self.fast_ema >= self.slow_ema:
            raise ValueError("trend momentum fast EMA must be below slow EMA")
        if self.bias_timeframe == self.signal_timeframe:
            raise ValueError("trend momentum bias and signal timeframes must differ")
        if self.base_confidence > self.maximum_confidence:
            raise ValueError("trend momentum base confidence exceeds maximum")
        return self


class LiquiditySweepConfig(Base):
    primary_timeframe: str = "M15"
    fallback_timeframe: str = "H1"
    range_lookback: int = Field(default=20, ge=5, le=500)
    minimum_bars: int = Field(default=25, ge=10, le=5000)
    atr_period: int = Field(default=14, ge=2, le=200)
    minimum_depth_atr: float = Field(default=0.0, ge=0.0, le=5.0)
    score: float = Field(default=75.0, gt=0.0, le=100.0)
    base_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    depth_confidence_scale: float = Field(default=0.25, ge=0.0, le=5.0)
    maximum_confidence: float = Field(default=0.90, ge=0.0, le=1.0)

    @field_validator("primary_timeframe", "fallback_timeframe")
    @classmethod
    def _supported_timeframe(cls, value: str) -> str:
        return Timeframe.parse(value).value

    @model_validator(mode="after")
    def _coherent(self) -> LiquiditySweepConfig:
        if self.primary_timeframe == self.fallback_timeframe:
            raise ValueError("liquidity sweep primary and fallback timeframes must differ")
        if self.base_confidence > self.maximum_confidence:
            raise ValueError("liquidity sweep base confidence exceeds maximum")
        return self


class LevelReactionConfig(Base):
    timeframe: str = "H1"
    history_lookback: int = Field(default=50, ge=20, le=1000)
    minimum_bars: int = Field(default=60, ge=20, le=5000)
    support_quantile: float = Field(default=0.05, ge=0.0, lt=0.5)
    resistance_quantile: float = Field(default=0.95, gt=0.5, le=1.0)
    atr_period: int = Field(default=14, ge=2, le=200)
    proximity_atr: float = Field(default=0.35, gt=0.0, le=5.0)
    wick_ratio: float = Field(default=1.5, gt=1.0, le=20.0)
    score: float = Field(default=55.0, gt=0.0, le=100.0)
    base_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    wick_confidence_scale: float = Field(default=0.20, ge=0.0, le=5.0)
    maximum_confidence: float = Field(default=0.80, ge=0.0, le=1.0)

    @field_validator("timeframe")
    @classmethod
    def _supported_timeframe(cls, value: str) -> str:
        return Timeframe.parse(value).value

    @model_validator(mode="after")
    def _coherent(self) -> LevelReactionConfig:
        if self.support_quantile >= self.resistance_quantile:
            raise ValueError("support quantile must be below resistance quantile")
        if self.base_confidence > self.maximum_confidence:
            raise ValueError("level reaction base confidence exceeds maximum")
        return self


class VolatilityRegimeConfig(Base):
    timeframe: str = "H1"
    minimum_bars: int = Field(default=120, ge=20, le=5000)
    atr_period: int = Field(default=14, ge=2, le=200)
    percentile_lookback: int = Field(default=100, ge=20, le=2000)
    compressed_percentile: float = Field(default=0.20, ge=0.0, lt=1.0)
    extreme_percentile: float = Field(default=0.95, gt=0.0, le=1.0)

    @field_validator("timeframe")
    @classmethod
    def _supported_timeframe(cls, value: str) -> str:
        return Timeframe.parse(value).value

    @model_validator(mode="after")
    def _coherent(self) -> VolatilityRegimeConfig:
        if self.compressed_percentile >= self.extreme_percentile:
            raise ValueError("compressed volatility percentile must be below extreme")
        return self


class MarketRegimeConfig(Base):
    """Non-directional context used for ranking and Claude briefing.

    It deliberately has no veto switch and no directional weight. A regime is
    an imperfect description of recent closed bars, not permission to trade.
    The existing strategy remains the source of candidates; this layer only
    helps compare those candidates and explain the surrounding conditions.
    """

    enabled: bool = True
    fast_timeframe: str = "H1"
    slow_timeframe: str = "H4"
    efficiency_lookback: int = Field(default=24, ge=10, le=200)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_percentile_lookback: int = Field(default=100, ge=20, le=1000)
    extreme_atr_percentile: float = Field(default=0.95, ge=0.80, le=1.0)
    trend_efficiency_min: float = Field(default=0.30, ge=0.0, le=1.0)
    range_fast_efficiency_max: float = Field(default=0.20, ge=0.0, le=1.0)
    range_slow_efficiency_max: float = Field(default=0.25, ge=0.0, le=1.0)
    #: Maximum positive or negative contribution to ordering. It is never
    #: added to confluence and therefore cannot turn NO_SIGNAL into a trade.
    ranking_modifier_cap: float = Field(default=12.0, ge=0.0, le=30.0)

    @field_validator("fast_timeframe", "slow_timeframe")
    @classmethod
    def _supported_regime_timeframe(cls, value: str) -> str:
        from core.types import Timeframe

        return Timeframe.parse(value).value

    @model_validator(mode="after")
    def _thresholds_do_not_overlap(self) -> MarketRegimeConfig:
        if self.range_fast_efficiency_max >= self.trend_efficiency_min:
            raise ValueError("range fast-efficiency maximum must be below trend minimum")
        return self


class PlaybooksConfig(Base):
    """Short-horizon theories that run alongside the swing engine.

    Off by default. They are a different kind of trade with a different failure
    mode — spread is charged per entry and a tight stop pays it many times over
    a day — so enabling them is a deliberate choice, not a default.
    """

    #: Run the short-horizon playbooks at all.
    enabled: bool = False
    #: Keep unvalidated playbooks observable in live operation without giving
    #: them authority to create or veto a real-money order. Paper/backtest may
    #: still execute them so evidence continues to accumulate.
    live_execution_enabled: bool = False
    #: M5 impulse continuation. Tight stop under the impulse leg, ~1 hour.
    momentum_scalp: bool = True
    #: M15 range-extreme rejection targeting the midpoint, ~3 hours.
    range_fade: bool = True
    #: Refuse everything when two theories disagree on direction.
    #:
    #: On by default and it should stay on. Momentum continuation and range
    #: reversion reading the same bars in opposite directions is not a close
    #: call to be settled by whichever scored higher — it is a market with no
    #: edge in either direction, and the honest answer is to stand aside.
    veto_on_conflict: bool = True
    #: Refuse when a short-horizon theory contradicts the swing engine.
    #:
    #: `veto_on_conflict` only ever caught the playbooks disagreeing with each
    #: other. The swing engine reading H1 structure as LONG while an M5 impulse
    #: theory read the same chart as SHORT went straight through — which is the
    #: one case where two genuinely different techniques contradict each other,
    #: and the clearest evidence there is that the read is ambiguous.
    #:
    #: Silence is not disagreement: a theory with no setup has no opinion, and
    #: most markets have no short-horizon setup at any given moment. Only an
    #: opposing play above `min_conviction` counts — one too weak to trade in
    #: its own right is too weak to cancel somebody else's trade.
    require_method_agreement: bool = True
    #: Spread ceiling as a fraction of the stop, for every short-horizon play.
    #: The constraint that actually binds: at 0.15 a 10-pip stop tolerates
    #: 1.5 pips of spread and no more.
    max_spread_share_of_stop: float = Field(default=0.15, gt=0.0, le=0.5)
    #: A short-horizon play must beat this to be taken at all.
    min_conviction: float = Field(default=60.0, ge=0.0, le=100.0)


class ConfluenceConfig(Base):
    """Decision policy shared by paper, backtest and live execution."""

    score_threshold: float = Field(default=55.0, ge=1.0, le=100.0)
    minimum_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    minimum_directional_modules: int = Field(default=2, ge=1, le=10)
    minimum_agreement_ratio: float = Field(default=0.60, ge=0.5, le=1.0)
    target_r_multiple: float = Field(default=2.0, ge=1.0, le=10.0)
    #: The target is also bounded by how far this instrument actually travels.
    #: `entry + 2R` is arithmetic and never asks whether the market goes there;
    #: a slow instrument gets a target it reaches once a month, and the trade
    #: becomes a bet on the stop not being hit.
    target_horizon_bars: int = Field(default=24, ge=4, le=200)
    #: Percentile of favourable excursion over that horizon. Not the maximum —
    #: one violent week should not set the expectation for every trade.
    target_reach_quantile: float = Field(default=0.70, ge=0.5, le=0.99)
    #: Below this, the setup is rejected instead of sized down. Shrinking the
    #: target indefinitely buys a high hit rate with trades that cannot pay for
    #: their own spread.
    minimum_r_multiple: float = Field(default=1.0, ge=0.5, le=5.0)
    atr_stop_multiple: float = Field(default=1.5, gt=0.0, le=10.0)
    #: Floor on the stop distance, as a multiple of H1 ATR.
    #:
    #: Only the structural path needs this. `atr_stop_multiple` already places
    #: the fallback stop 1.5 ATR out, but a stop anchored to a structural level
    #: took that level plus a 0.25 ATR buffer and nothing else — so an
    #: invalidation price a few pips from entry produced a stop at a fraction of
    #: an ATR, sitting squarely inside normal M1/M5 chop. Those trades were not
    #: stopped out because the idea was wrong; they were stopped out by noise.
    #:
    #: 0.8 leaves room for a genuinely tight structural stop to stay tight while
    #: putting the floor above the band the adviser kept describing as "inside
    #: recent noise" (it was rejecting stops at 0.13 to 0.85 ATR).
    min_stop_atr: float = Field(default=0.8, ge=0.0, le=5.0)
    #: Spread ceiling as a fraction of this trade's own stop distance.
    #:
    #: A different question from the spread filter's, which asks whether the
    #: spread is unusual for this instrument at this hour. After 21:00 the
    #: answer to that is no — the evening spread *is* the evening baseline — and
    #: the trade is waved through while the stop has not widened to match. At
    #: 0.20 a 10-pip stop tolerates 2 pips of spread and no more.
    #:
    #: Looser than the playbooks' 0.15 because those are scalps, where the
    #: spread is paid against a much smaller target. Set to 0.5 to effectively
    #: disable; the field cannot be turned off entirely, because a trade that
    #: cannot pay its own spread is not a trade.
    max_spread_share_of_stop: float = Field(default=0.20, gt=0.0, le=0.5)
    #: Paper/backtest may research every module. Live execution is restricted
    #: to this independently validated subset; empty means live entries block.
    live_enabled_modules: tuple[str, ...] = ()

    #: Lower timeframes checked for entry timing once a direction is chosen.
    #: The bias comes from H4/H1; these decide whether *now* is the moment. An
    #: empty tuple disables the check.
    entry_timing_timeframes: tuple[str, ...] = ("M15", "M5")
    #: Closed bars of lower-timeframe movement considered.
    entry_timing_lookback: int = Field(default=6, ge=2, le=50)
    #: Adverse movement, in ATR, that blocks the entry. A flat lower timeframe
    #: is never an objection; only a move materially against the direction is.
    entry_timing_max_adverse_atr: float = Field(default=0.50, gt=0.0, le=5.0)

    #: Higher timeframes checked for an established trend the trade would be
    #: taken straight into. There was a timing gate below the bias and nothing
    #: above it, so shorts were proposed on indices in multi-week uptrends that
    #: had just broken to fresh highs. An empty tuple disables the check.
    htf_trend_timeframes: tuple[str, ...] = ("D1", "W1")
    #: Closed bars of higher-timeframe trend considered.
    htf_trend_lookback: int = Field(default=20, ge=5, le=200)
    #: How strong an opposing trend has to be to block the entry, in
    #: `sqrt(bars) * ATR` units — the same scale `analysis/position_health.py`
    #: measures drift on. Around 1.0 is a trend clearly beyond ordinary
    #: wandering; a flat or mildly opposed higher timeframe is not an objection.
    #:
    #: This does not ban counter-trend trades. It bans trading *into* a strong,
    #: still-accelerating one, where the setup has to be right about the turn
    #: and about its timing at the same time.
    htf_trend_veto: float = Field(default=1.0, gt=0.0, le=5.0)
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "market_structure": 1.0,
            "trend_momentum": 1.0,
            "liquidity_sweep": 0.8,
            "level_reaction": 0.7,
            "volatility_regime": 0.0,
        }
    )

    @field_validator("weights")
    @classmethod
    def _weights_are_bounded(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("analysis.confluence.weights may not be empty")
        if any(weight < 0.0 or weight > 5.0 for weight in value.values()):
            raise ValueError("analysis weights must be between 0 and 5")
        if not any(weight > 0 for weight in value.values()):
            raise ValueError("at least one analysis weight must be positive")
        return value


class AnalysisConfig(Base):
    market_structure: MarketStructureConfig = MarketStructureConfig()
    trend_momentum: TrendMomentumConfig = TrendMomentumConfig()
    liquidity_sweep: LiquiditySweepConfig = LiquiditySweepConfig()
    level_reaction: LevelReactionConfig = LevelReactionConfig()
    volatility_regime: VolatilityRegimeConfig = VolatilityRegimeConfig()
    market_regime: MarketRegimeConfig = MarketRegimeConfig()
    confluence: ConfluenceConfig = ConfluenceConfig()
    playbooks: PlaybooksConfig = PlaybooksConfig()


# ------------------------------------------------------ trade management ---


class PositionHealthProfile(Base):
    """Closed-candle health horizons for one asset class."""

    fast_timeframe: str = "M1"
    structure_timeframe: str = "M5"
    fast_bars: int = Field(default=40, ge=10, le=500)
    structure_bars: int = Field(default=60, ge=20, le=500)

    @field_validator("fast_timeframe", "structure_timeframe")
    @classmethod
    def _supported_timeframe(cls, value: str) -> str:
        normalised = value.strip().upper()
        if normalised not in {item.value for item in Timeframe}:
            raise ValueError(f"unsupported health timeframe {value!r}")
        return normalised


class TradeManagementConfig(Base):
    #: ATR multiple added beyond the structural level, so a spread widening or
    #: a stop hunt of ordinary size does not take us out.
    sl_atr_buffer_multiple: float = Field(default=0.5, ge=0.0, le=3.0)
    sl_atr_period: int = Field(default=14, ge=2)

    #: 0.6, down from 1.0, for the same reason the give-back arm moved.
    #:
    #: A full R is a large move on a EUR 88 account and most winners never see
    #: it. AUDJPY peaked at 0.57R — a euro of real profit — and this rule, set
    #: at 1.0, never moved the stop an inch before the whole gain went. A
    #: protection that only engages on the trades that were going to win anyway
    #: is not protecting anything.
    #:
    #: The cost of moving earlier is being taken out at break-even by noise that
    #: the trade would have survived. That cost is much lower than it was: with
    #: `min_stop_atr` the stop is now at least an ATR wide, so 0.6R is a real
    #: move rather than a wobble, and the offset below leaves the exit slightly
    #: profitable rather than exactly flat.
    break_even_at_r: float = Field(default=0.6, gt=0.0)
    #: Offset past entry when moving to break even, in ATR multiples, to cover
    #: spread and commission. Break even at exactly entry is a small loss.
    break_even_offset_atr: float = Field(default=0.1, ge=0.0)

    partial_close_at_r: float = Field(default=1.5, gt=0.0)
    partial_close_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)

    trailing_mode: Literal["atr", "structure", "none"] = "atr"
    trailing_atr_multiple: float = Field(default=2.0, gt=0.0)

    #: Peak give-back protection: how far a trade must have run before the rule
    #: arms, and how much of that peak it may hand back before we take what is
    #: left. Zero on either field switches it off.
    #:
    #: This is the rule a person applies without thinking — "I was up 1.5R and
    #: it's back to 0.6R, I'm out" — and the one the machine did not have. The
    #: stop and the trail both measure from price; neither of them knows the
    #: trade was ever in profit, so a spike that fully retraces inside one cycle
    #: was invisible. Deliberately measured against the recorded peak rather
    #: than a fresh high-water mark, so a restart cannot reset it.
    #:
    #: 1.0R to arm keeps it away from ordinary noise around entry, where a
    #: half-give-back is nothing at all. Above the arming level break-even has
    #: already banked the trade, so the worst case here is a small win rather
    #: than a loss.
    #: 0.5, not 1.0, and a live pair is why. AUDJPY peaked at EUR 1.00 against
    #: EUR 1.77 of risk — 0.57R — and ended at -0.28R. Arming at 1.0R meant the
    #: rule never engaged; break-even, on the same threshold, never engaged
    #: either. Both sat above anything that trade would ever reach.
    #:
    #: On a EUR 88 account a full R is a large move and most winners never get
    #: there. The bar has to be what counts as real profit *here*, which is the
    #: same level `health_secure_at_r` already uses, so the two agree.
    giveback_arm_r: float = Field(default=0.5, ge=0.0)
    #: How much of the peak may be handed back before the question is asked.
    #: Reaching it is not an exit on its own — see `_giveback_exit`, which then
    #: asks the health read whether the move is still working.
    giveback_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    #: And the part that is not a judgement. However intact the read says the
    #: move is, handing back this much of a gain is not a position worth
    #: holding. Without it, a permanently "healthy" read could ride a winner all
    #: the way back to entry, which is the exact complaint this rule answers.
    giveback_hard_fraction: float = Field(default=0.8, ge=0.0, le=1.0)

    #: The per-second read of how an open trade is actually behaving — has the
    #: structure broken, has momentum turned, is it running against us, has the
    #: spread blown out. See `analysis/position_health.py`.
    #:
    #: The mechanical rules above all look at one number, R, which says nothing
    #: about why. Two trades at +0.4R — one grinding toward target, one with the
    #: market falling away underneath it — are the same number and opposite
    #: situations, and without this the fast layer could not tell them apart.
    #:
    #: It can only ever reduce risk, and it needs two independent signals
    #: agreeing before it may close anything.
    health_enabled: bool = True
    #: Bars of the fast timeframe (M1) for momentum and the adverse run.
    health_fast_bars: int = Field(default=40, ge=10, le=500)
    #: Bars of the slower timeframe (M5) for the structure read. A swing on M1
    #: is noise wearing the word "structure".
    health_structure_bars: int = Field(default=60, ge=20, le=500)
    #: At or above this R, a deteriorating trade is banked rather than cut.
    health_secure_at_r: float = Field(default=0.5, ge=0.0)
    #: At or above this R, a single warning tightens the stop instead.
    health_tighten_at_r: float = Field(default=0.2, ge=0.0)
    #: Asset-specific horizons. Forex keeps the established M1/M5 behaviour;
    #: continuously traded and exchange products use slower confirmation so a
    #: single noisy one-minute print cannot masquerade as structural failure.
    health_profiles: dict[str, PositionHealthProfile] = Field(
        default_factory=lambda: {
            "forex": PositionHealthProfile(),
            "crypto": PositionHealthProfile(
                fast_timeframe="M5", structure_timeframe="M15", fast_bars=48, structure_bars=64
            ),
            "stock": PositionHealthProfile(
                fast_timeframe="M5", structure_timeframe="M15", fast_bars=48, structure_bars=64
            ),
            "index": PositionHealthProfile(),
            "metal": PositionHealthProfile(),
            "commodity": PositionHealthProfile(
                fast_timeframe="M5", structure_timeframe="M15", fast_bars=48, structure_bars=64
            ),
            "unknown": PositionHealthProfile(
                fast_timeframe="M5", structure_timeframe="M15", fast_bars=48, structure_bars=64
            ),
        }
    )

    #: Close a position that has gone nowhere. Dead capital still carries risk.
    time_exit_hours: float | None = Field(default=24.0, gt=0.0)
    time_exit_min_abs_r: float = Field(default=0.3, ge=0.0)

    #: How often the AI supervisor reconsiders each open position, in minutes.
    #: Zero switches it off and leaves the mechanical rules above as the whole
    #: management policy.
    #:
    #: Fifteen minutes is a deliberate middle. Asked every cycle the adviser
    #: reacts to noise and pays spread for the privilege; asked hourly it can
    #: watch a thesis break and do nothing about it for fifty minutes. This is
    #: also the dominant ongoing cost once positions are open: two positions at
    #: this interval is roughly eight reviews an hour.
    supervision_interval_minutes: float = Field(default=15.0, ge=0.0, le=240.0)

    #: The ordinary review remains deliberately slow, but a material change in
    #: the position may bring the next review forward. This is different from
    #: polling an LLM every second: the cheap local layer watches every tick and
    #: only escalates a changed situation to the expensive judgement layer.
    supervision_event_driven: bool = True
    #: Minimum spacing between two paid reviews, even when several triggers
    #: arrive together. It bounds cost and prevents one noisy candle from
    #: making the adviser repeatedly reconsider the same evidence.
    supervision_min_interval_minutes: float = Field(default=2.0, ge=0.25, le=60.0)
    #: A newly reached profit band is material evidence. The supervisor may
    #: bank it, protect it, or explicitly decide that intact structure deserves
    #: more room.
    supervision_profit_step_r: float = Field(default=0.25, gt=0.0, le=5.0)
    #: Once a profitable position gives this fraction of its recorded peak
    #: back, ask the judgement layer early. The hard mechanical give-back rule
    #: remains in force and does not wait for the API.
    supervision_giveback_trigger_fraction: float = Field(default=0.25, gt=0.0, le=1.0)

    #: How far back an unrecorded broker position may reach to claim a matching
    #: entry intent, in minutes. Beyond this it is treated as an orphan and
    #: closed.
    #:
    #: Ten minutes is generous for what it covers: the gap between `order_send`
    #: returning and the journal row committing is milliseconds, and the window
    #: only has to survive a restart. Making it long would let a stale intent
    #: adopt an unrelated position opened by hand — attaching real money to the
    #: wrong plan, which is worse than closing a position that should have
    #: lived.
    adoption_window_minutes: float = Field(default=10.0, ge=0.0, le=1440.0)

    @model_validator(mode="after")
    def _supervision_cadence_is_coherent(self) -> TradeManagementConfig:
        if (
            self.supervision_interval_minutes > 0
            and self.supervision_min_interval_minutes > self.supervision_interval_minutes
        ):
            raise ValueError(
                "trade_management.supervision_min_interval_minutes may not exceed "
                "supervision_interval_minutes"
            )
        return self


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
    report_interval_minutes: int = Field(default=15, ge=1, le=1440)
    report_directory: str = "runtime/reports"


class LearningConfig(Base):
    """Evidence required before a shadow configuration may replace production."""

    shadow_min_days: int = Field(default=30, ge=7, le=365)
    shadow_min_paired_outcomes: int = Field(default=100, ge=30, le=10_000)
    shadow_min_unique_days: int = Field(default=20, ge=5, le=365)
    shadow_min_symbols: int = Field(default=3, ge=1, le=500)
    #: A challenger must beat the champion by more than a rounding error even
    #: before uncertainty is considered. Results are measured in R so this is
    #: comparable across account sizes.
    shadow_min_expectancy_lift_r: float = Field(default=0.05, ge=0.0, le=1.0)
    shadow_confidence_level: float = Field(default=0.95, ge=0.80, le=0.999)
    shadow_resolution_hours: int = Field(default=72, ge=1, le=720)


class ScannerConfig(Base):
    """How much of the broker catalogue is inspected per cycle.

    The cheap ranking pass covers the whole catalogue; the top
    ``deep_candidates`` get full multi-timeframe analysis, and only what
    survives every deterministic gate reaches the AI reviewer.

    The staging exists to bound API cost, not analysis. Deep analysis is local
    pandas over bars already fetched and cached — a full catalogue pass measured
    one second. The LLM is the expensive stage, and it is protected by the
    deterministic gates downstream, which reject the overwhelming majority
    before a single token is spent. So ``deep_candidates`` can reasonably cover
    everything the cheap scan let through; ranking is only a way to decide what
    to drop when it cannot.
    """

    #: Symbols cheaply ranked per cycle. None = the whole catalogue.
    batch_size: int | None = Field(default=None, ge=1)
    #: Top-ranked symbols promoted to full analysis each cycle. The ceiling is
    #: the size of a large broker catalogue, so "analyse everything the cheap
    #: scan let through" is expressible. The old limit of 100 made that
    #: impossible to configure, which is a policy decision that does not belong
    #: in a schema bound.
    deep_candidates: int = Field(default=12, ge=1, le=2000)


class MarketScoutConfig(Base):
    """Bounded independent Claude scan over compact cross-market evidence."""

    enabled: bool = False
    cooldown_minutes: int = Field(default=30, ge=5, le=1440)
    max_calls_per_day: int = Field(default=24, ge=1, le=200)
    max_markets_per_call: int = Field(default=12, ge=3, le=50)
    minimum_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    #: Agreement can move a deterministic candidate earlier in the queue. It
    #: never changes eligibility and disagreement is not a veto.
    ranking_bonus: float = Field(default=8.0, ge=0.0, le=25.0)


class AIConfig(Base):
    """Optional second-opinion layer; it can veto but never bypass hard gates."""

    enabled: bool = False
    provider: Literal["openai", "anthropic", "consensus"] = "consensus"
    openai_model: str = "gpt-5.1"
    anthropic_model: str = ""
    minimum_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)
    fail_closed: Literal[True] = True
    market_scout: MarketScoutConfig = MarketScoutConfig()


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
    analysis: AnalysisConfig = AnalysisConfig()
    trade_management: TradeManagementConfig = TradeManagementConfig()
    learning: LearningConfig = LearningConfig()
    journal: JournalConfig = JournalConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    scanner: ScannerConfig = ScannerConfig()
    ai: AIConfig = AIConfig()

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
            # An uncapped global ceiling cannot be exceeded by anything, and a
            # mode asking for no cap under a capped global still has to obey it.
            if self.risk.max_trades_per_day != UNLIMITED_TRADES and (
                limits.max_trades_per_day == UNLIMITED_TRADES
                or limits.max_trades_per_day > self.risk.max_trades_per_day
            ):
                raise ValueError(
                    f"modes.{name}.max_trades_per_day "
                    f"({limits.max_trades_per_day or 'unlimited'}) exceeds "
                    f"risk.max_trades_per_day ({self.risk.max_trades_per_day})"
                )
            if limits.max_concurrent_positions > self.risk.max_concurrent_positions:
                raise ValueError(
                    f"modes.{name}.max_concurrent_positions "
                    f"({limits.max_concurrent_positions}) exceeds "
                    f"risk.max_concurrent_positions ({self.risk.max_concurrent_positions})"
                )
            # A daily stop that can never trigger before the weekly one is not
            # a daily stop; and neither may outrun the circuit breaker.
            if (
                NO_LOSS_LIMIT not in (limits.daily_loss_limit_pct, self.risk.weekly_loss_limit_pct)
                and limits.daily_loss_limit_pct >= self.risk.weekly_loss_limit_pct
            ):
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
        return tuple(
            self.instruments.broker_symbol(sym)
            for sym in self.instruments.whitelist[self.system.mode.value]
        )

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
        bare = self.instruments.canonical_symbol(symbol)
        if bare in self.instruments.blocklist or symbol in self.instruments.blocklist:
            return False, "SYMBOL_BLOCKLISTED"
        if self.instruments.universe_mode == "whitelist" and symbol not in self.active_whitelist:
            return False, f"SYMBOL_NOT_WHITELISTED_FOR_{self.mode.value.upper()}"
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

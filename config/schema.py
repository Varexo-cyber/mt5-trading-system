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
    #: Seconds of guarding that a slow cycle may never take away.
    #:
    #: The guard used to run until `cycle_start + loop_interval_seconds`, which
    #: is a deadline already in the past by the time a cycle returns: the live
    #: log shows cycles of 55 to 121 seconds against a 30-second interval, so
    #: `remaining <= 0` fired immediately and the one-second layer never ran at
    #: all. No give-back, no peak stall, no profit lock, no health reading —
    #: every one of them written, tested, deployed and never once executed on
    #: an open position. The deck showed it plainly the moment it started
    #: reporting the age of a reading: "this measurement is 9 minutes old".
    #:
    #: A scan that starts thirty seconds late costs a setup. A position left
    #: unwatched for two minutes costs money, and the whole reason the fast
    #: layer exists is that it is the cheaper of the two.
    min_guard_seconds: float = Field(default=20.0, ge=0.0, le=300.0)
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
    #: Timeframes whose absence refuses the market outright. Everything else in
    #: `timeframes` is left out of the context when the broker cannot supply it,
    #: and the modules that read it say so and score neutral.
    #:
    #: EMPTY MEANS ALL OF THEM, which is what this system did before the field
    #: existed: `get_context` built its series with a dict comprehension, so the
    #: first timeframe to raise took the other six with it and the symbol went
    #: into a quarantine that backs off to four hours. A hole in one M5 feed
    #: removed a market for the rest of the session, including the timeframes
    #: that were complete. A config that says nothing keeps the old behaviour
    #: rather than loosening silently on upgrade.
    required_timeframes: tuple[str, ...] = ()
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
        # A required timeframe that is never loaded can never be supplied, so
        # every symbol would be refused for a timeframe nobody asked for. That
        # is a typo with the blast radius of the whole catalogue, and it costs
        # one line to make it impossible.
        stray = [tf for tf in self.required_timeframes if tf not in self.timeframes]
        if stray:
            raise ValueError(
                f"data.required_timeframes names timeframe(s) not in data.timeframes: {stray}"
            )
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
    #: Complete Jarvis opt-out. Unlike ``blocklist`` (which only refuses a new
    #: entry), these symbols are absent from scanning, adoption, position
    #: management, emergency flattening and Jarvis' risk/trade counters. This
    #: is for an instrument owned by another system or by the account owner.
    ignored_symbols: tuple[str, ...] = ()
    #: Scanned so paper sections can measure them; never tradable.
    #:
    #: `ignored_symbols` is total: absent from scanning, adoption, position
    #: management, emergency flattening and the risk counters. That is right
    #: for an instrument the owner holds themselves, and it is too much when
    #: the only thing wanted is a READING. A section running on paper needs a
    #: MarketContext for the symbol, and an ignored symbol never gets one.
    #:
    #: So this list splits the four guarantees apart and keeps three of them.
    #: The symbol IS scanned, so an observer can record what a paper section
    #: would have done. It is refused at `symbol_allowed_at_equity`, so no
    #: entry can ever be opened. And it stays out of position management and
    #: out of the risk counters exactly as before, so an existing ticket the
    #: owner opened themselves is still never touched, moved, closed or
    #: counted against the account's exposure.
    #:
    #: The one thing that changes for such a symbol is that it appears in the
    #: journal as analysis. Nothing reaches the broker.
    observation_only_symbols: tuple[str, ...] = ()
    #: Symbols where Jarvis may trade its OWN book but must never take over a
    #: position somebody else opened. See `refuses_adoption` for why this is
    #: not the same question as `observation_only_symbols`, and for the gold
    #: case that made the distinction necessary.
    no_adoption_symbols: tuple[str, ...] = ()
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

    def is_observation_only(self, symbol: str) -> bool:
        """Scanned for measurement, never tradable, never managed."""
        canonical = self.canonical_symbol(symbol)
        watched = {item.upper() for item in self.observation_only_symbols}
        return symbol.upper() in watched or canonical.upper() in watched

    def is_hands_off(self, symbol: str) -> bool:
        """Whether Jarvis may touch an OPEN position in this symbol.

        Both lists say no. They differ only on whether the symbol is scanned,
        and scanning has nothing to do with whether a live ticket is managed.
        Keeping that distinction in one method rather than at each call site is
        what stops the two drifting apart later — and the direction they would
        drift is "started managing a position the owner said not to touch".
        """
        return self.is_ignored(symbol) or self.is_observation_only(symbol)

    def is_ignored(self, symbol: str) -> bool:
        """Whether Jarvis must behave as if this broker symbol does not exist."""
        canonical = self.canonical_symbol(symbol)
        ignored = {item.upper() for item in self.ignored_symbols}
        return symbol.upper() in ignored or canonical.upper() in ignored

    def refuses_adoption(self, symbol: str) -> bool:
        """Whether a position Jarvis did NOT open here must be left alone.

        WHY THIS IS SEPARATE FROM `is_hands_off`. That predicate answers two
        different questions with one answer, and the owner needs different
        answers to them:

            may Jarvis OPEN a position here?
            may Jarvis touch a position it DID NOT open here?

        Gold is the case. It sits in `observation_only_symbols` because the
        owner trades it by hand and Jarvis must never manage or close those
        tickets -- and `manual_positions` is enabled adopting magic 0, which is
        exactly what a hand-placed MT5 order carries, so without that entry
        Jarvis would take his own gold trades over.

        But the same entry also forbids section six from opening its own gold
        scalp, and gold is both the largest source of setups this section has
        and the CHEAPEST market it can reach: 796 setups in thirty days at 23.5%
        cost, against 26.5% on XAUEUR and 44% on US30. Blocking it did not stop
        section six trading gold -- it pushed it onto the derived crosses,
        XAUAUD and XAUEUR, whose spread is wider and whose "momentum candle" is
        often the currency leg moving rather than the metal.

        So: this list keeps the guarantee that matters -- a hand-placed ticket
        is never adopted, managed, or closed -- while letting the account open
        and run its OWN position in the same symbol, under its own magic
        number, with every ordinary gate and the section breaker in force.
        """
        canonical = self.canonical_symbol(symbol)
        refused = {item.upper() for item in self.no_adoption_symbols}
        return symbol.upper() in refused or canonical.upper() in refused


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


class ConvictionRiskConfig(Base):
    """Stake more when the reviewer is very sure, and only then.

    Authorised by the owner in these words: minimum 2%, up to 6-8% on a highly
    convincing setup, less as the conviction falls.

    WHAT DRIVES IT, and the choice matters more than the numbers. Not the
    engine's own conviction — that is measured on this account NOT to predict
    the outcome, twice: the "20+ over the bar" bucket was the worst of them all
    at -4.92R over 23 trades, and across 84 paid reviews the 40-45 conviction
    band produced nothing useful while 20-25 produced 33%. Scaling the stake by
    that number would put the most money on the trades the record says are the
    worst.

    So it rides on the REVIEWER's confidence: the last, most independent
    judgement in the chain, and the one number here that has not been disproven.

    IT HAS ALSO NOT BEEN PROVEN. Two approvals on the day this was written came
    back at 0.59 and 0.64 and both lost. There is no evidence yet that this
    number predicts a winner either, and until `module_records` and the
    scorecard say otherwise, staking on it is a decision rather than a finding.
    The ramp starts high for that reason: everything below `confidence_floor`
    stakes the ordinary amount, so an unremarkable approval is sized exactly as
    it was before this existed.
    """

    enabled: bool = False
    #: The stake below the ramp. What every trade used to get.
    floor_pct: Pct = 2.0
    #: The stake at maximum conviction.
    ceiling_pct: Pct = 6.0
    #: Reviewer confidence at which the ramp begins. Deliberately well above
    #: `ai.minimum_confidence` (0.55): an approval that only just cleared the
    #: bar is not a convincing setup, it is a permitted one.
    confidence_floor: float = Field(default=0.70, ge=0.0, le=1.0)
    #: Where the stake saturates.
    confidence_ceiling: float = Field(default=0.90, ge=0.0, le=1.0)

    def stake_for(self, confidence: float) -> float:
        """The risk percentage this confidence earns."""
        if not self.enabled:
            return self.floor_pct
        span = self.confidence_ceiling - self.confidence_floor
        if span <= 0:
            return self.ceiling_pct if confidence >= self.confidence_ceiling else self.floor_pct
        reach = (confidence - self.confidence_floor) / span
        reach = min(1.0, max(0.0, reach))
        return self.floor_pct + reach * (self.ceiling_pct - self.floor_pct)

    @model_validator(mode="after")
    def _coherent(self) -> ConvictionRiskConfig:
        if self.ceiling_pct < self.floor_pct:
            raise ValueError("conviction risk ceiling is below its floor")
        if self.confidence_ceiling < self.confidence_floor:
            raise ValueError("conviction confidence ceiling is below its floor")
        return self


class SectionBreakerConfig(Base):
    """Stop a new section automatically when it is going badly.

    Sections two, five and six went live with ZERO measured trades between
    them. Every number behind them is reasoning, and this is what survives
    that reasoning being wrong.

    Two rules, and the streak is the urgent one. A share of losses over a
    window catches a section that is quietly wrong; a run of consecutive
    losses catches one that is wrong RIGHT NOW, before the window has filled
    enough for the share to move. On a section taking a few trades a day that
    is the difference between stopping today and stopping next week.

    RE-ARMING IS MANUAL. A breaker that switches itself back on trips
    repeatedly on the same fault while the account pays for every cycle of it.
    Turning a section back on means editing `live_enabled_modules`, which shows
    up in a diff.
    """

    enabled: bool = True
    #: Recent closed trades this section was behind, judged as a group.
    window: int = Field(default=30, ge=3, le=500)
    #: Nothing trips before there is this much to judge. A section cannot be
    #: condemned by its first three trades — that is noise, and switching a
    #: section off on noise is the same mistake as switching one on for it.
    minimum_trades: int = Field(default=25, ge=3, le=500)
    #: Share of the window that may be losses before it stops.
    maximum_loss_share: float = Field(default=0.70, gt=0.0, le=1.0)
    #: Consecutive losses that stop it regardless of the window.
    losing_streak: int = Field(default=10, ge=2, le=100)

    @model_validator(mode="after")
    def _window_can_reach_the_minimum(self) -> SectionBreakerConfig:
        if self.minimum_trades > self.window:
            raise ValueError(
                "risk.section_breakers minimum_trades cannot exceed window — the "
                "breaker would never see enough trades to judge anything and would "
                "sit permanently disarmed while reading as armed"
            )
        return self


class RiskConfig(Base):
    risk_per_trade_pct: Pct = 1.0
    #: Ceiling the sizer will never exceed regardless of setup quality.
    max_risk_per_trade_pct: Pct = 1.0
    #: A retained, typed tombstone for an unsafe historical override. The
    #: broker minimum is not permission to exceed the stake behind a decision.
    #: `Literal[False]` makes a future YAML edit fail at startup instead of
    #: quietly reintroducing upward lot rounding.
    allow_minimum_lot_above_target: Literal[False] = False
    #: Stake by conviction. See `ConvictionRiskConfig`.
    conviction_risk: ConvictionRiskConfig = ConvictionRiskConfig()
    #: Total risk allowed across every open position at once, as a percentage
    #: of equity. 0 switches the cap off.
    #:
    #: THE GRENDEL THAT MAKES THE ONE ABOVE SURVIVABLE. Until now the only
    #: limit on simultaneous exposure was arithmetic: four slots times a fixed
    #: 2% is 8%. Let a single trade take 6% and that same arithmetic gives 24%
    #: — a quarter of the account at risk at once, on an account of a hundred
    #: and forty euros, with the daily and weekly loss limits switched off.
    #: Nobody asked for that and it is what raising the per-trade ceiling does
    #: by itself.
    #:
    #: Measured on the CURRENT stop of each open position, not the original, so
    #: a winner whose stop has moved to break-even stops consuming the budget
    #: and frees room for the next trade. That is the honest measure of what is
    #: actually at risk right now.
    max_total_open_risk_pct: float = Field(default=0.0, ge=0.0, le=100.0)

    #: Slippage a stop-out actually suffers, in pips, per asset class. A class
    #: not listed here contributes nothing, and the cost check below then rests
    #: on commission alone.
    #:
    #: Measured, not assumed. A live AUDNZD stop sat at 1.19722 and filled at
    #: 1.19705 — 1.7 pips straight through it, on a 5-pip stop. A stop is a
    #: request, not a guarantee, and the difference is a cost the risk model
    #: has to carry like any other.
    stop_slippage_pips: dict[str, float] = Field(default_factory=dict)
    #: Share of the risk that commission plus slippage may consume before a
    #: setup is refused outright. 0 switches the check off.
    #:
    #: On that AUDNZD trade the two together were 56% of 1R and it returned
    #: -1.48R on a -1.00R plan. At that ratio the strategy is not what is being
    #: tested — the cost of trading is, and it wins. Past roughly a quarter, a
    #: full stop-out cannot land inside -1.25R however good the entry was.
    max_cost_share_of_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    #: The same ceiling for the confluence path specifically, because section
    #: one's arithmetic is not the account's.
    #:
    #: Its own give-back table over 1,970 trades:
    #:
    #:     RR 0.30   80% hit   -0.001R      random walk 74%
    #:     RR 0.72   56% hit   -0.070R      random walk 56%
    #:     RR 1.00   10% hit   -0.840R      random walk 48%
    #:
    #: At the near target it wins 80% where a coin flip wins 74%, so the edge
    #: is six points of hit rate — about 0.05R, and that is the whole of it.
    #: The account-wide limit lets a trade spend a third of its risk on spread,
    #: commission and slippage: seven times the edge it is trying to collect.
    #: The measured result is the exact zero in the first row.
    #:
    #: Zero falls back to `max_cost_share_of_risk`, so an account that has not
    #: set this keeps its old behaviour rather than being tightened silently.
    section_one_max_cost_share: float = Field(default=0.0, ge=0.0, le=1.0)

    #: Broker commission per lot per side, in account currency. 0 for accounts
    #: whose cost is entirely in the spread.
    #:
    #: This is not bookkeeping, it is part of the risk. A live AUDNZD stop-out
    #: cost EUR 1.93 against a modelled 1R of EUR 1.53 — and EUR 0.33 of the
    #: EUR 0.40 gap was commission, confirmed against the deal in the terminal.
    #: Every threshold in this system is denominated in R, so an R that omits
    #: a fifth of what a loss actually costs makes the give-back arm late, the
    #: profit lock secure less than it claims, and every expectancy figure
    #: flatter the account.
    #:
    #: It also prices tight stops correctly for the first time. Commission is a
    #: fixed cost per lot, so on a 2-pip stop it is a third of the risk and on
    #: a 20-pip stop it is a rounding error. Including it makes scalp-width
    #: stops unattractive by arithmetic rather than by a threshold someone
    #: guessed.
    commission_per_lot_per_side: float = Field(default=0.0, ge=0.0)
    #: Per-asset-class overrides, for brokers that charge indices or metals
    #: differently from FX. Absent classes take the figure above.
    commission_by_asset_class: dict[str, float] = Field(default_factory=dict)

    #: Automatic stop for a new section that is going badly, by module name.
    #:
    #: A section not listed here has no breaker, which is right for the nine
    #: original readers: they have months of record and are judged as a book
    #: rather than one at a time. The new sections have none, so each carries
    #: its own.
    section_breakers: dict[str, SectionBreakerConfig] = Field(default_factory=dict)

    max_concurrent_positions: int = Field(default=2, ge=1, le=10)
    #: Equity that buys one concurrent position. 0 keeps the flat cap above.
    #:
    #: A fixed number of slots is the wrong shape for an account that is meant
    #: to grow. Two is right at EUR 100 — three simultaneous trades there means
    #: three positions the account cannot express at the minimum lot anyway —
    #: and plainly wrong at EUR 1,000, where the constraint is no longer size
    #: but the fact that somebody once typed a 2.
    #:
    #: Slots earned this way are still bounded by `max_concurrent_positions`
    #: and by the mode's own ceiling, so growth widens the account toward a
    #: limit that a human set; it never walks through it.
    equity_per_position: float = Field(default=0.0, ge=0.0)
    #: Never fewer than this, however small the account. Below two, one open
    #: trade blocks every other opportunity for as long as it runs, and a
    #: system that can hold only one position is a system that spends most of
    #: its day unable to act.
    min_concurrent_positions: int = Field(default=2, ge=1, le=10)
    #: Should a position whose market is shut still consume a position slot?
    #:
    #: A slot exists to bound how many trades the system is *running* at once.
    #: A single-name share whose exchange closed at 17:00 is not being run: it
    #: cannot be closed, its stop cannot be moved, it cannot be secured, and no
    #: management decision about it is possible until the venue reopens. Three
    #: of those held four slots hostage on a live account and left the scanner
    #: one usable slot for the entire evening.
    #:
    #: WHAT THIS DOES NOT DO, AND IT MATTERS. The frozen position keeps its
    #: risk. Releasing its slot means simultaneous risk can exceed
    #: `max_concurrent_positions` x the per-trade budget while a market is
    #: shut, and every one of those tickets becomes live again at the reopen.
    #: What still bounds it: margin (checked per order against real free
    #: margin), the correlation filter, per-symbol exposure, and every entry
    #: gate. This trades a hard ceiling for capacity, on purpose.
    #:
    #: Default False keeps the historical behaviour. It is switched on per
    #: account, in that account's config, where the choice is visible.
    release_slots_when_unmanageable: bool = False
    #: How stale a quote must be before its market counts as shut.
    #:
    #: Deliberately far above the spread filter's `max_tick_age_seconds` (120s
    #: for shares). That gate asks "is this quote fresh enough to price an
    #: entry"; this one asks "has this venue stopped quoting altogether", and
    #: answering the second question with the first would hand a slot back
    #: every time a thin share went quiet for two minutes mid-session.
    unmanageable_quote_age_seconds: int = Field(default=600, ge=60, le=86_400)
    #: Trades a day, or 0 for no cap. See `UNLIMITED_TRADES` below for why 0 is
    #: a defensible setting and not a hole in the risk model.
    max_trades_per_day: int = Field(default=3, ge=0, le=50)
    max_trades_per_week: int = Field(default=10, ge=0, le=200)

    #: All stated as POSITIVE percentages of equity; the manager applies sign.
    #: 0 disables the limit — see NO_LOSS_LIMIT for what remains in force.
    daily_loss_limit_pct: NonNegPct = 3.0
    #: The same limit stated in MONEY rather than percent. 0 disables it.
    #:
    #: A percentage moves with the account and the owner asked for a fixed
    #: number: "wnr daily verlies 20 EUR bereikt die gehele dag geen verdere
    #: trades meer, volgende dag verder". At EUR 216 that is 9.26%, and if the
    #: account doubled, the same 9.26% would be EUR 40 -- which is not what was
    #: asked for. So this is absolute, in the account currency, and it stays
    #: put whatever the equity does.
    #:
    #: Both limits are live at once and the FIRST to bite wins. Setting one
    #: does not switch the other off.
    #:
    #: Measured the way every other limit in `risk_manager` is measured: on
    #: EQUITY, so an open position 15 EUR underwater already counts toward it.
    #: That is stricter than counting only closed trades, and deliberately so
    #: -- the module docstring gives the reason, which is that realised-only
    #: lets an account sit deeply underwater at "0 EUR lost today" and keep
    #: opening risk. It stops NEW trades; it does not force-close what is
    #: already open, because closing at a limit turns a paper loss into a real
    #: one.
    daily_loss_limit_money: float = Field(default=0.0, ge=0.0)
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

    #: The floor was `ge=1.0`, which made "never earn less than the stop can
    #: cost" a law of the config rather than a choice. It is a reasonable
    #: default and it is not arithmetic: a 0.60R target with a measured 70%
    #: first-touch hit rate is a better trade than a 1.00R target reached a
    #: third of the time, and the engine's expectancy search is what decides
    #: which is which. The bound now permits the question; `Settings._coherent`
    #: keeps this in step with `analysis.confluence.minimum_r_multiple` so the
    #: two cannot drift apart the way they did before.
    #:
    #: The default drops from 2.0 to 1.0 because that validator found the pair
    #: had never agreed: 2.0 sat above a 1.0 analysis floor, so every target the
    #: engine planned between 1R and 2R was built, measured, scored, sized and
    #: then refused as RR_BELOW_MINIMUM. The live overlay had already aligned
    #: the two by hand and recorded why; the defaults had not, and nothing said
    #: so.
    min_risk_reward: float = Field(default=1.0, ge=0.3)

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
    #: When the margin will not stretch to the size the risk budget asked for,
    #: send the largest size it WILL stretch to instead of refusing the trade.
    #:
    #: On a EUR 180 account against single-share CFDs the margin gate refuses
    #: almost everything: 0.15 lots of one German share wants 169 EUR of margin,
    #: and with the safety buffer that is more than the account holds. Those
    #: setups had already cleared every analytical gate.
    #:
    #: Nothing about the trade changes to make it fit. Entry, stop and target
    #: stay exactly where they were, so the measured expectancy that approved it
    #: is still the expectancy of the smaller position — only the size moves,
    #: and it moves DOWN. That is the opposite direction from rounding up to the
    #: broker minimum, which this system forbids outright and still forbids:
    #: when not even `volume_min` fits, the trade is refused as before.
    reduce_volume_to_fit_margin: bool = True

    forbidden: ForbiddenPractices = ForbiddenPractices()

    def commission_per_lot(self, asset_class: str) -> float:
        """Round-trip commission for one lot, in account currency.

        Round trip because a stop-out pays both sides, and the risk model is
        answering "what does it cost me if this is wrong" — which is never one
        side of the trade.
        """
        per_side = self.commission_by_asset_class.get(asset_class, self.commission_per_lot_per_side)
        return 2.0 * max(0.0, per_side)

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
    #: Also require the instrument's OWN market to be open, not merely some
    #: session on the allowed list.
    #:
    #: `tradable_sessions` gives one global answer for the whole catalogue.
    #: With all three sessions enabled it is true around twenty-two hours a day
    #: and says nothing about the symbol in front of it. Two live trades showed
    #: what that permits: NDX100 short at 00:51 UTC, five hours after the
    #: Nasdaq closed, and EURCAD short at 03:03 UTC with Frankfurt and Toronto
    #: both shut. Asia was running, Asia is on the list, and Asia prices
    #: neither instrument.
    #:
    #: Off by default because it removes trades. An instrument whose home
    #: sessions cannot be determined is always allowed through — see
    #: `filters.home_session`.
    require_home_session: bool = False
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
    #: Asset classes that wind down at their own time rather than the FX one.
    #:
    #: An index does not follow the FX rollover. The US cash session closes at
    #: 20:00 UTC and the CFD spread widens from that moment, so a position held
    #: to the forex wind-down at 20:15 spends its last quarter of an hour in the
    #: widest quote of the day. A live SPX500 long sat in exactly that window.
    #:
    #: Times are UTC and must be no later than `evening_flat_from`; a class that
    #: wants to run *longer* than forex is not a wind-down override, it is a
    #: different feature, and letting it in here would silently extend exposure.
    evening_flat_by_class: dict[str, str] = Field(default_factory=lambda: {"index": "20:00"})
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

    @model_validator(mode="after")
    def _class_overrides_only_tighten(self) -> SessionFilterConfig:
        """A per-class wind-down may be earlier than the FX one, never later.

        Enforced rather than documented. An override reading 21:30 looks like a
        harmless edit and silently *extends* exposure into the rollover for a
        whole asset class — the opposite of what the setting exists for, and
        invisible until something is held through it.
        """
        if not self.evening_flat_from:
            return self
        limit = tuple(int(part) for part in self.evening_flat_from.split(":"))
        late = {
            name: when
            for name, when in self.evening_flat_by_class.items()
            if tuple(int(part) for part in when.split(":")) > limit
        }
        if late:
            raise ValueError(
                "evening_flat_by_class may only be earlier than evening_flat_from "
                f"({self.evening_flat_from}); these are later: {late}"
            )
        return self


class VolumeSpikeFilterConfig(Base):
    """Refuse a market whose last closed minute is carrying an event.

    Section six has refused these since it was written. Section one had no
    equivalent: seven of its eight live detectors never look at volume, and the
    one that does has a volume FLOOR and no ceiling -- more volume raises its
    confidence without limit, so a release printing ten times normal activity
    produces its strongest reading.

    The calendar is what protected section one, and a calendar only knows what
    is scheduled. An unscheduled headline, a central banker off script, a stop
    cascade: none of those are in it, and all of them print this candle.
    """

    enabled: bool = True
    #: At or above this multiple of the median, the minute is an event. The
    #: same figure section six uses, so the two cannot disagree about the same
    #: candle.
    extreme_multiple: float = Field(default=3.0, gt=1.0, le=50.0)
    #: How many closed M1 bars the median is taken over. Median rather than
    #: mean, because one earlier spike inside the window would lift a mean
    #: enough to hide the next one.
    lookback_bars: int = Field(default=30, ge=5, le=500)


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


class RunwayFilterConfig(Base):
    """How much time a trade must have before we force it flat."""

    enabled: bool = True
    #: Minutes of clear market required between an entry and this instrument's
    #: own wind-down. 45 is not a market constant, it is a floor: a structural
    #: target on the M5/M15 setups this system trades resolves in roughly one
    #: to three hours, and a trade with less than three quarters of an hour is
    #: not being given a chance to be right — it is being charged the spread
    #: for the privilege of being closed on the clock.
    min_runway_minutes: float = Field(default=45.0, ge=0.0, le=480.0)
    #: Per-asset-class overrides, for instruments that resolve on a different
    #: clock than FX. Absent classes take `min_runway_minutes`.
    min_runway_by_class: dict[str, float] = Field(default_factory=dict)

    @field_validator("min_runway_by_class")
    @classmethod
    def _sane_overrides(cls, value: dict[str, float]) -> dict[str, float]:
        bad = {name: minutes for name, minutes in value.items() if not 0.0 <= minutes <= 480.0}
        if bad:
            raise ValueError(f"min_runway_by_class out of range [0, 480]: {bad}")
        return value

    #: Beyond the blunt floor above, also ask whether *this* target can be
    #: reached in the time left at the market's current speed. The filter
    #: cannot ask that — it never sees the setup — so the runner does, once the
    #: levels are known.
    require_reachable_target: bool = True
    #: How much better than a pure random walk a market travels when we have a
    #: directional read on it. Net displacement over n bars is `sqrt(n) x ATR`
    #: for a random walk, so bars-to-target is `(distance / ATR)^2`; this
    #: divides the distance first. 1.0 assumes we have no edge at all and is
    #: brutally pessimistic; above ~2.5 the check stops rejecting anything.
    travel_efficiency: float = Field(default=1.5, ge=0.5, le=3.0)
    #: Timeframe whose ATR sets the market's speed for that estimate.
    speed_timeframe: str = "M5"

    def minutes_for(self, asset_class: str) -> float:
        return self.min_runway_by_class.get(asset_class, self.min_runway_minutes)


class LivelinessFilterConfig(Base):
    """When is a market too quiet to be worth entering."""

    enabled: bool = True
    timeframe: str = "M5"
    #: Bars averaged for "how fast is it moving right now".
    recent_bars: int = Field(default=6, ge=2, le=60)
    #: Bars the recent window is compared against. 120 M5 bars is ten hours —
    #: long enough to span more than one session, so the comparison is against
    #: the instrument's day rather than against the last twenty minutes of it.
    baseline_bars: int = Field(default=120, ge=30, le=1000)
    #: Below this fraction of its own baseline, the market is asleep. 0.5 is
    #: deliberately not close to 1.0: normal markets breathe, and a gate that
    #: fires on every lull is a gate that never lets anything through.
    min_activity_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    #: Bars needed before the ratio means anything at all.
    min_bars: int = Field(default=40, ge=10, le=1000)
    #: Recent bars used to judge whether the feed is continuous enough to
    #: execute. A large opening gap can make ATR/activity look healthy while
    #: the tape after it prints only once every few minutes; this second view
    #: measures that pathology directly.
    quality_bars: int = Field(default=30, ge=10, le=240)
    #: Share of recent timestamp gaps allowed to exceed one-and-a-half nominal
    #: bars. One overnight/weekend gap in the window remains harmless, while a
    #: stock CFD that repeatedly omits intraday bars is refused.
    max_sparse_gap_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    #: Share of recent bars allowed to have no high/low range at all. Flat bars
    #: are valid occasionally; a tape made mostly of them is not executable
    #: evidence, regardless of how attractive a higher-timeframe trend looks.
    max_flat_bar_fraction: float = Field(default=0.60, ge=0.0, le=1.0)
    #: How far this instrument is allowed to JUMP between bars, in ATR.
    #:
    #: A stop is a price level, and a market that opens somewhere other than
    #: where it closed does not fill it there. Every rule in this system is
    #: written in R, and R assumes the stop binds; an instrument that gaps
    #: routinely breaks that assumption several times a week and no amount of
    #: management can repair it after the fact.
    #:
    #: ENR is the case: -3.61R on an exit recorded as BROKER_SL. A stop costs
    #: 1.00R by definition, so 2.61R went straight through it. The exchange was
    #: shut and the share reopened elsewhere.
    #:
    #: This is what lets shares and other exchange products back into the scan
    #: on their behaviour rather than on their name. A market that does not jump
    #: keeps its stop and is welcome; one that jumps is refused whatever class it
    #: belongs to, and a forex pair that started gapping would be refused too.
    #:
    #: 99th percentile rather than the maximum: one flash print should not ban an
    #: instrument for a week, but a market that gaps every session has a 99th
    #: percentile made of those gaps. Zero disables the test.
    max_gap_atr: float = Field(default=1.0, ge=0.0, le=20.0)
    #: How much a typical bar must move, measured in spreads. THE ONLY ABSOLUTE
    #: TEST IN THIS FILTER, and the gap the other three left open.
    #:
    #: `min_activity_ratio` compares a market with its OWN recent normal, so a
    #: market that is always dead has a dead baseline and scores about 1.0 — it
    #: passes for being consistently untradeable. `max_flat_bar_fraction` only
    #: counts bars whose high equals their low exactly, and a tape that ticks
    #: once a minute prints a range of one or two points, which is not flat by
    #: that definition. Neither can see the pathology.
    #:
    #: Live: Qiagen (QIA) quoted 36.6349 / 36.6550 — a 2.01-cent spread — while
    #: printing M1 bars a few points tall. Getting in and out cost more than the
    #: market moved in a minute, so the trade was decided by the spread before
    #: the thesis had any say. There is no threshold elsewhere that catches this,
    #: because every other gate is relative to the instrument itself.
    #:
    #: Median rather than mean, so one news candle cannot vouch for an hour of
    #: nothing. 1.0 means "a typical minute must at least cover the round trip";
    #: zero disables the test.
    min_bar_range_in_spreads: float = Field(default=1.0, ge=0.0, le=20.0)

    @model_validator(mode="after")
    def _windows_are_ordered(self) -> LivelinessFilterConfig:
        if self.recent_bars >= self.baseline_bars:
            raise ValueError(
                f"recent_bars ({self.recent_bars}) must be shorter than baseline_bars "
                f"({self.baseline_bars}); comparing a window against itself always "
                f"reads as normal"
            )
        if self.min_bars < self.recent_bars:
            raise ValueError(
                f"min_bars ({self.min_bars}) is below recent_bars ({self.recent_bars}); "
                f"the recent window would be padded with bars that do not exist"
            )
        if self.min_bars < self.quality_bars:
            raise ValueError(
                f"min_bars ({self.min_bars}) is below quality_bars "
                f"({self.quality_bars}); continuity cannot be measured"
            )
        return self


class CurrencyExposureConfig(Base):
    """How many positions may lean the same way on one currency."""

    enabled: bool = True
    #: Positions already leaning one way on a currency before a further one is
    #: refused. One, because a two-slot account holding GBPAUD short and
    #: GBPJPY short is not diversified — it is one GBP short with a second lot
    #: on it, and it was believed to be two independent trades right up until
    #: both moved together.
    max_positions_per_currency: int = Field(default=1, ge=1, le=10)
    #: The same limit for instruments the currency decomposition cannot see.
    #:
    #: An index quoted in its own currency has no currency leg — correctly, it
    #: is a bet on equities and not on the euro — which leaves FRA40 long and
    #: UK100 long looking like two unrelated trades to the filter above. They
    #: are one risk-on bet with a second lot on it, and on 7 August the account
    #: held three European index longs inside twenty minutes.
    max_positions_per_asset_class: int = Field(default=1, ge=1, le=10)
    #: Which classes that limit applies to.
    #:
    #: Forex is absent on purpose: the currency legs already describe it
    #: exactly, and capping FX by class would refuse EURUSD beside USDJPY,
    #: which are genuinely different trades.
    grouped_asset_classes: tuple[str, ...] = ("index", "metal", "commodity", "crypto", "stock")


class LossCooldownConfig(Base):
    """How long an instrument is left alone after it has cost us money."""

    enabled: bool = True
    #: A third of the shortest playbook horizon. `momentum_scalp` expects to be
    #: finished inside an hour, so re-entering the same instrument within
    #: twenty minutes of a loss is the same idea taken twice — the chart it
    #: read has not had time to become a different chart.
    #:
    #: Not "the observed 75 seconds plus a margin". That would fix the one case
    #: in the log and nothing standing next to it.
    minutes: float = Field(default=20.0, ge=0.0, le=1440.0)
    #: The same rule after a WINNING close, and shorter.
    #:
    #: 25 August, live:
    #:
    #:     06:58:01  USDJPY short closed, +EUR 1.40
    #:     06:58:37  USDJPY short opened again
    #:     07:08:01  USDJPY short closed, -EUR 1.81 (EUR 1.32 of it commission)
    #:
    #: Thirty-six seconds. The loss cooldown said nothing because the previous
    #: close was a winner — and the mechanism this filter exists to stop has
    #: nothing to do with which way the last one went. A window of history is
    #: read, a trade consumes a few minutes of it, and the window read next is
    #: mostly the window just read. The setup is not re-evaluated so much as
    #: re-encountered. A winner consumes exactly the same minutes.
    #:
    #: Shorter than the loss window because the evidence is weaker. After a
    #: loss the market has demonstrably disagreed; after a win it has not, and
    #: a genuine continuation on a fresh leg is a real thing to allow. Five
    #: minutes stops the same M1 close being traded twice while leaving a
    #: setup that has actually moved on.
    #:
    #: Zero disables it and restores the losses-only behaviour.
    minutes_after_a_win: float = Field(default=5.0, ge=0.0, le=1440.0)


class HeadlineFilterConfig(Base):
    """Unscheduled news, as a reason to keep still. Never as a direction.

    The calendar next door knows what is coming. This knows what is happening,
    and it is the layer that catches the central bank moving between meetings
    and the story that breaks at 03:00 with the afternoon showing clear.

    There is no sentiment setting here and there will not be one.
    `filters.newsfeed.items.NewsPressure` carries the argument: by the time a
    story reaches a public feed the move it describes has happened, so trading
    its direction from a retail VPS is buying what somebody else already
    bought. What survives the latency is that something is going on, and that
    is what these numbers describe.
    """

    #: Off until `scripts/verify_newsfeed.py` has been run on the machine that
    #: will use it. None of the default feed URLs could be reached from the
    #: environment this was written in, so no response shape has been confirmed
    #: — and a layer that silently returns nothing reports a quiet market,
    #: which is the one answer it must never invent.
    enabled: bool = False
    #: Feed name to URL. Empty uses `filters.newsfeed.providers.DEFAULT_FEEDS`.
    feeds: dict[str, str] = Field(default_factory=dict)
    #: How often EACH FEED is polled -- not how often the layer checks.
    #:
    #: The distinction is the whole design. `HeadlineService` staggers the
    #: feeds across this interval, so with twenty feeds at twenty seconds
    #: something is fetched roughly every second while no single host sees
    #: more than three requests a minute. The operator asked for one-second
    #: scraping and that is what the batch does; what it does not do is hit
    #: one URL every second, which gets the VPS rate-limited and then blocked.
    #: A blocked address reports a permanently quiet market, which is the
    #: worst failure this layer has because nothing downstream can tell.
    #:
    #: Cheap on top of that: `RssHeadlineProvider` sends an ETag, so a poll
    #: that finds nothing new costs a couple of hundred bytes and no work at
    #: the far end.
    refresh_interval_seconds: float = Field(default=20.0, ge=5.0, le=3600.0)
    #: The recent window "how busy is it right now" is measured over.
    window_minutes: float = Field(default=20.0, ge=5.0, le=240.0)
    #: The long window that window is compared against. Per instrument, because
    #: EUR and NZD do not carry comparable traffic on any wire and one absolute
    #: threshold over both means blocking EUR always or NZD never.
    baseline_hours: float = Field(default=12.0, ge=1.0, le=72.0)
    #: How stale the held window may get before it stops being evidence.
    max_age_minutes: float = Field(default=30.0, ge=5.0, le=360.0)
    #: Both must hold to block: enough stories to mean anything, and enough
    #: above normal to be unusual. Either alone misfires — three headlines is
    #: nothing on EUR and a storm on NZD, and a tenfold rise from 0.1 is one
    #: routine mention.
    min_headlines: int = Field(default=3, ge=1, le=50)
    spike_multiple: float = Field(default=3.0, ge=1.0, le=20.0)
    #: A market-wide story blocks on its own. It touches every pair equally, so
    #: it never looks like a spike on any single one and the test above would
    #: wave all of them through at the worst possible moment.
    block_on_systemic: bool = True
    #: Stand down when the feeds are dark. Off by default, and the departure
    #: from "no data, no trade" is argued in `HeadlineFilter._unavailable`: the
    #: calendar's fail-closed rule is untouched, and with this layer dark the
    #: system is at the safety level it has run at all along.
    block_when_unavailable: bool = False
    #: Headlines handed to the reviewer with an open position. The one place
    #: the text itself is worth carrying — a language model reads a headline
    #: better than any regex here, and it is being asked anyway.
    headlines_for_reviewer: int = Field(default=6, ge=0, le=25)


class FiltersConfig(Base):
    news: NewsFilterConfig = NewsFilterConfig()
    headlines: HeadlineFilterConfig = HeadlineFilterConfig()
    session: SessionFilterConfig = SessionFilterConfig()
    runway: RunwayFilterConfig = RunwayFilterConfig()
    liveliness: LivelinessFilterConfig = LivelinessFilterConfig()
    spread: SpreadFilterConfig = SpreadFilterConfig()
    volume_spike: VolumeSpikeFilterConfig = VolumeSpikeFilterConfig()
    correlation: CorrelationFilterConfig = CorrelationFilterConfig()
    currency_exposure: CurrencyExposureConfig = CurrencyExposureConfig()
    loss_cooldown: LossCooldownConfig = LossCooldownConfig()


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
    #: What to do when the bias timeframe has no opinion at all.
    #:
    #: The module used to fold two opposite situations into one answer. H4
    #: trending *against* H1 is a conflict and refusing it is right. H4 flat
    #: while H1 trends is not a conflict — it is the absence of a headwind —
    #: and it was returning the same hard zero. On a live twelve hours, "no
    #: weighted directional evidence" was 18,150 of 36,331 no-signals, half of
    #: every refusal in the system, and this branch is the largest contributor
    #: to it.
    #:
    #: A discount rather than a pass, because an unconfirmed trend genuinely is
    #: worth less than a confirmed one: the signal carries this fraction of its
    #: usual confidence, must still clear `minimum_confidence`, and must still
    #: carry the score past the threshold. 0.0 restores the old behaviour.
    neutral_bias_confidence_scale: float = Field(default=0.75, ge=0.0, le=1.0)
    #: Bars of raw price drift on the signal timeframe that must not oppose the
    #: EMA reading. 0 switches the check off.
    #:
    #: THE FAILURE THIS EXISTS FOR. GBPUSD, 12 August. The engine wanted to buy
    #: it all day while it fell: 233 M15 and 111 M5 refusals, every one reading
    #: "price is moving against the long". Two got through, both lost, and the
    #: entry that lost most was taken on this module alone saying "H4 and H1
    #: EMA/momentum aligned bullish".
    #:
    #: An EMA is a position, not a direction. When a market tops out the fast
    #: EMA stays above the slow one for hours and its five-bar slope stays
    #: positive for a while after price has turned, so the module goes on
    #: proposing longs into a falling market. Its own exit note named the cause:
    #: "confirming the H1 fast-drift-down risk the entry review itself flagged
    #: as the weakest part of the thesis".
    #:
    #: This compares the EMA reading against what price has actually done over
    #: the last few bars of the same timeframe. It cannot create a short — a
    #: refused long is not evidence for one — it only stops the module from
    #: insisting on a direction the recent tape contradicts.
    drift_agreement_bars: int = Field(default=3, ge=0, le=50)

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
    #: And how many times its own median that ATR has to be before "rare"
    #: counts as "extreme". The percentile alone is purely relative and this
    #: reading VETOES the setup before any analysis runs, so on its own it
    #: refuses a fifth of everything scanned for being marginally busier than
    #: yesterday — 1,058 decisions an hour on this account.
    #:
    #: Measured on synthetic series with realistic volatility clustering, the
    #: percentile test fires at the same rate either way: 10.4% of bars on a
    #: calm series with no shock at all, 9.2% on one carrying a real 4x shock.
    #: What separates them is size — the calm series never exceeds 1.51x its
    #: own median, the shocked one reaches 4.61x.
    #:
    #: 1.0 restores the old purely-relative behaviour.
    extreme_atr_multiple: float = Field(default=2.0, ge=1.0, le=20.0)

    @field_validator("timeframe")
    @classmethod
    def _supported_timeframe(cls, value: str) -> str:
        return Timeframe.parse(value).value

    @model_validator(mode="after")
    def _coherent(self) -> VolatilityRegimeConfig:
        if self.compressed_percentile >= self.extreme_percentile:
            raise ValueError("compressed volatility percentile must be below extreme")
        return self


class FastEmaCrossConfig(Base):
    """A 9/20 EMA cross on the fast chart, for entries measured in minutes.

    Every other directional module reads slow charts: 20/50 EMAs on H4 and H1,
    a break of structure, or at the fastest a specific wick on M15. Nothing
    looked at the timeframe a day trade is actually taken on.

    A 9/20 cross on five minutes is the classic quick entry and the classic way
    to be sawn apart — two averages brushing all day in a quiet market, each
    brush a trade paying the spread. The three floors below are what separate a
    cross from a touch, and none of them is optional.
    """

    enabled: bool = True
    timeframe: str = "M5"
    fast_ema: int = Field(default=9, ge=2, le=200)
    slow_ema: int = Field(default=20, ge=3, le=400)
    atr_period: int = Field(default=14, ge=2, le=200)
    #: Retained for the config surface; the invalidation now sizes itself from
    #: the age of the cross. See `minimum_invalidation_bars` below.
    invalidation_lookback: int = Field(default=12, ge=2, le=500)
    #: Fewest bars the invalidation window looks back over, so a cross printed
    #: on the last bar still has a swing to hang a stop on rather than one bar.
    minimum_invalidation_bars: int = Field(default=3, ge=1, le=50)
    #: How far beyond the slow average the stop sits, in ATR. The thesis dies
    #: when price closes back through that average — the module's own third
    #: floor — so that is where the stop belongs; the buffer is what stops a
    #: single wick through it from counting as the close.
    invalidation_buffer_atr: float = Field(default=0.25, ge=0.0, le=2.0)
    #: How stale a cross may be and still count as an entry. Beyond this it is
    #: a state, and the state is what `trend_momentum` already reports.
    #:
    #: 6, up from 3, and this is the honest lever for "more quick entries" —
    #: the one that admits more OPPORTUNITIES rather than weaker evidence.
    #:
    #: Three M5 bars is a fifteen-minute window. A 9/20 cross happens a handful
    #: of times a day on a given symbol, and the scanner has to be looking at
    #: that symbol, on that pass, inside those fifteen minutes, with the
    #: separation floor and the price-side check both satisfied at that exact
    #: moment. Most crosses were simply never seen. Six bars is thirty minutes:
    #: double the chance of catching the same cross, with every quality floor
    #: untouched — the separation is still measured in ATR, price must still be
    #: on the right side of the slow average, and `entry_quality` still runs.
    #:
    #: Deliberately not larger. The module's whole justification is that it
    #: reports an entry rather than a state, and forty minutes after a cross
    #: that distinction is gone.
    max_bars_since_cross: int = Field(default=6, ge=0, le=100)
    #: How far apart the averages must be, in ATR. Two EMAs sitting on top of
    #: each other crossing back and forth is one market making no decision, and
    #: treating each touch as a signal is reading noise at high frequency.
    minimum_separation_atr: float = Field(default=0.15, ge=0.0, le=10.0)
    #: 50, the lowest score here. It is the fastest and least corroborated
    #: evidence in the engine and it should need help to clear the threshold.
    score: float = Field(default=50.0, gt=0.0, le=100.0)
    base_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    separation_confidence_scale: float = Field(default=0.50, ge=0.0, le=5.0)
    maximum_confidence: float = Field(default=0.80, ge=0.0, le=1.0)

    @field_validator("timeframe")
    @classmethod
    def _supported_timeframe(cls, value: str) -> str:
        return Timeframe.parse(value).value

    @model_validator(mode="after")
    def _coherent(self) -> FastEmaCrossConfig:
        if self.fast_ema >= self.slow_ema:
            raise ValueError("fast EMA must be below slow EMA")
        if self.base_confidence > self.maximum_confidence:
            raise ValueError("fast EMA base confidence exceeds maximum")
        return self


class ImpulseBreakConfig(Base):
    """One violent bar with follow-through. See analysis/impulse_break.py.

    Fills a hole found live: a GBPCAD M15 body of -1.34 ATR with a -2.3 ATR
    three-bar M5 drift fired no directional module at all, because
    `drift_continuation` needs 65% of eight bars to agree (an impulse gives
    about 25%) and `fast_ema_cross` needs a cross that a vertical move has
    already left behind.
    """

    enabled: bool = True
    timeframe: str = "M15"
    atr_period: int = Field(default=14, ge=2, le=200)
    #: The body, not the range: a wide bar with a small body is indecision.
    minimum_body_atr: float = Field(default=1.00, ge=0.1, le=10.0)
    #: How far into its own range the bar must close. Below this it is a
    #: rejection wearing a big candle, and joining a rejection is joining the
    #: reversal of the move you meant to join.
    minimum_close_location: float = Field(default=0.66, ge=0.5, le=1.0)
    #: The mechanism is about the minutes after the repricing.
    max_bars_since: int = Field(default=2, ge=0, le=20)
    #: How much of the move may already have been handed back. Half of it means
    #: the market rejected the break and the spike is the whole story.
    maximum_retracement: float = Field(default=0.50, ge=0.0, le=1.0)
    #: 60, chosen so the weakest signal this module can emit still clears the
    #: live threshold on its own: 60 x 0.45 = 27 against 26. That arithmetic is
    #: not optional — `fast_ema_cross` shipped at 50 against a 35 threshold and
    #: could never trade alone, which took a day and a live funnel to notice.
    score: float = Field(default=60.0, ge=0.0, le=100.0)
    base_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    body_confidence_scale: float = Field(default=0.25, ge=0.0, le=2.0)
    maximum_confidence: float = Field(default=0.80, ge=0.0, le=1.0)


class EmaPullbackResumeConfig(Base):
    """A discrete M5 trend re-entry after a shallow EMA-band pullback."""

    enabled: bool = True
    timeframe: str = "M5"
    fast_ema: int = Field(default=9, ge=2, le=200)
    slow_ema: int = Field(default=20, ge=3, le=400)
    atr_period: int = Field(default=14, ge=2, le=200)
    pullback_bars: int = Field(default=4, ge=2, le=20)
    slope_bars: int = Field(default=3, ge=1, le=20)
    minimum_separation_atr: float = Field(default=0.12, ge=0.0, le=5.0)
    minimum_slope_atr: float = Field(default=0.04, ge=0.0, le=5.0)
    maximum_slow_ema_breach_atr: float = Field(default=0.25, ge=0.0, le=3.0)
    m1_confirmation_bars: int = Field(default=3, ge=1, le=20)
    maximum_m1_adverse_atr: float = Field(default=0.50, ge=0.0, le=5.0)
    stop_buffer_atr: float = Field(default=0.20, ge=0.0, le=2.0)
    score: float = Field(default=58.0, ge=0.0, le=100.0)
    base_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    separation_confidence_scale: float = Field(default=0.35, ge=0.0, le=5.0)
    reclaim_confidence_scale: float = Field(default=0.15, ge=0.0, le=2.0)
    maximum_confidence: float = Field(default=0.80, ge=0.0, le=1.0)

    @field_validator("timeframe")
    @classmethod
    def _supported_timeframe(cls, value: str) -> str:
        return Timeframe.parse(value).value

    @model_validator(mode="after")
    def _coherent(self) -> EmaPullbackResumeConfig:
        if self.fast_ema >= self.slow_ema:
            raise ValueError("EMA pullback fast EMA must be below slow EMA")
        if self.base_confidence > self.maximum_confidence:
            raise ValueError("EMA pullback base confidence exceeds maximum")
        return self


class VolatilitySqueezeConfig(Base):
    """Range contracts to an extreme, then breaks out of its own compression.

    `volatility_regime` already measures this and only ever reports it as
    CONTEXT — it scores no direction and can therefore never produce a setup.
    So the one thing in the system that knows a market is coiled has no way to
    say "and it has just uncoiled, that way".

    Deliberately hard to satisfy. The compression has to be a genuine extreme
    against the instrument's own history, not merely "quiet today", and the
    expansion bar has to be large against the compression rather than against
    the ordinary range — otherwise every first bar of a London session
    qualifies.
    """

    enabled: bool = True
    timeframe: str = "M15"
    #: Bars whose range defines "compressed right now".
    compression_bars: int = Field(default=12, ge=4, le=200)
    #: History the compression is judged against. A quiet hour means nothing
    #: without knowing what this instrument's quiet usually looks like.
    percentile_lookback: int = Field(default=200, ge=50, le=1000)
    #: The compression must sit in the lowest N% of that history.
    compression_percentile: float = Field(default=0.25, ge=0.01, le=0.60)
    #: The breakout bar's range, as a multiple of the compressed range. Below
    #: this it is not an expansion, it is the next quiet bar.
    expansion_multiple: float = Field(default=1.8, ge=1.0, le=10.0)
    #: The close must finish this far through the compression's edge, in units
    #: of the compressed range, so a wick through the boundary is not a break.
    breakout_close_share: float = Field(default=0.25, ge=0.0, le=2.0)
    atr_period: int = Field(default=14, ge=2, le=200)
    #: Stop goes the other side of the compression, plus this much ATR.
    invalidation_buffer_atr: float = Field(default=0.25, ge=0.0, le=2.0)
    score: float = Field(default=55.0, ge=0.0, le=100.0)
    base_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    maximum_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    #: Confidence added per extra multiple of expansion beyond the floor.
    expansion_confidence_scale: float = Field(default=0.10, ge=0.0, le=1.0)


class MeanReversionConfig(Base):
    """A statistical extreme that has stopped extending — faded, not followed.

    Every directional module in this system is a continuation module in some
    form: a trend, a cross, a break, a drift, a resumed pullback. The single
    exception is `liquidity_sweep`, and over 180 days it is the only one that
    is not negative. That is a hint and not a proof, and the way to find out is
    to have a second reversion reader rather than a ninth continuation one.

    Distinct from the `range_fade` playbook, which needs an identified range
    with touched edges. This asks a purely statistical question: how far is
    price from its own mean, in its own standard deviations, and has the move
    stopped going.

    The stall requirement is what stops it being a knife-catcher. An extreme
    that is still extending is a trend, and fading a trend because it has gone
    far is the most expensive mistake in this family.
    """

    enabled: bool = True
    timeframe: str = "M15"
    lookback: int = Field(default=48, ge=10, le=500)
    #: Distance from the mean, in standard deviations, before it is extreme.
    entry_z: float = Field(default=2.2, ge=1.0, le=6.0)
    #: How far price must have come off the extreme before the move counts as
    #: stalled, in ATR. Zero would fade anything far from the mean, including a
    #: market still accelerating away.
    #:
    #: In ATR and not as a share of the extreme bar's own range, which is what
    #: this was first written as and which a test caught. A thin final bar makes
    #: that share meaningless: a bar spanning a tenth of an ATR whose close sits
    #: halfway down it reads as "50% retraced" while price has come off the high
    #: by a twentieth of an ATR — nothing at all. The module would have faded
    #: markets that were still running, which is the one mistake this family
    #: cannot afford.
    stall_retrace_atr: float = Field(default=0.35, ge=0.0, le=3.0)
    #: How recently the extreme printed. An extreme six bars ago that nothing
    #: has faded is a market that has re-based, not a stretched one.
    max_bars_since_extreme: int = Field(default=3, ge=1, le=20)
    atr_period: int = Field(default=14, ge=2, le=200)
    #: Stop beyond the extreme itself, plus this much ATR.
    invalidation_buffer_atr: float = Field(default=0.35, ge=0.0, le=3.0)
    score: float = Field(default=50.0, ge=0.0, le=100.0)
    base_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    maximum_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    #: Confidence added per standard deviation beyond the entry threshold.
    stretch_confidence_scale: float = Field(default=0.12, ge=0.0, le=1.0)


class SessionBreakoutConfig(Base):
    """The quiet session's range, broken when the busy one opens.

    Sessions exist in this system only as a way to REFUSE trades: a blackout, a
    wind-down, an evening flat. The fact that a market builds a range overnight
    and resolves it when London arrives has never been available as a reason to
    take one.

    Measured in the broker's own clock, because that is what the bars are
    stamped in, and the window is configured rather than inferred — a server
    three hours off UTC would otherwise silently move every session by three
    hours and nothing would look wrong.
    """

    enabled: bool = True
    timeframe: str = "M15"
    #: The window whose high and low form the range, as server-clock hours.
    range_start_hour: int = Field(default=0, ge=0, le=23)
    range_end_hour: int = Field(default=7, ge=0, le=23)
    #: How long after the range closes a break still counts as the session
    #: resolving it, rather than as an unrelated afternoon move.
    breakout_window_hours: float = Field(default=4.0, ge=0.5, le=12.0)
    #: The range must be at least this wide in ATR — a five-pip overnight range
    #: is a dead market, and breaking it means nothing.
    minimum_range_atr: float = Field(default=0.75, ge=0.0, le=10.0)
    #: ...and at most this wide. A range larger than the day's usual travel has
    #: already had its move; breaking it late is chasing.
    maximum_range_atr: float = Field(default=4.0, ge=0.5, le=20.0)
    #: Close beyond the edge by this share of the range, so a wick is not a break.
    breakout_close_share: float = Field(default=0.10, ge=0.0, le=1.0)
    atr_period: int = Field(default=14, ge=2, le=200)
    score: float = Field(default=50.0, ge=0.0, le=100.0)
    base_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    maximum_confidence: float = Field(default=0.80, ge=0.0, le=1.0)


class SeasonalityConfig(Base):
    """This instrument's own record on this weekday, not folklore.

    "Sell in May", "Monday reversal", "turn of the month" are stories. What is
    measurable is whether THIS symbol, over the daily history the broker
    actually carries, has a return on this weekday that is distinguishable from
    zero given its own noise.

    So it is written as a t-test and not as a table of beliefs. The sample floor
    and the t-threshold are the whole module: below them it returns neutral,
    which on most instruments and most days is what it will do.

    Weakest score and lowest confidence ceiling of any module here, on purpose.
    A weekday bias is a background lean, never a reason to take a trade by
    itself — and the lone-module floor already refuses it in that role.
    """

    enabled: bool = True
    timeframe: str = "D1"
    #: Daily bars considered. Two years of weekdays is roughly 100 samples per
    #: weekday, which is the least that makes a t-statistic worth printing.
    lookback_days: int = Field(default=500, ge=100, le=2000)
    #: Fewest observations of this weekday before the module will say anything.
    minimum_samples: int = Field(default=60, ge=20, le=500)
    #: How far from zero the mean return must be, in standard errors.
    minimum_t: float = Field(default=2.0, ge=1.0, le=6.0)
    score: float = Field(default=25.0, ge=0.0, le=100.0)
    base_confidence: float = Field(default=0.30, ge=0.0, le=1.0)
    maximum_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    #: Confidence added per standard error beyond the threshold.
    significance_confidence_scale: float = Field(default=0.08, ge=0.0, le=1.0)


class CandleMomentumConfig(Base):
    """SECTION SIX. The candle just moved — is everything else already there?

    The shape of the fast bots: watch M1, jump in when a candle goes, jump
    straight back out. What makes those lose is that the candle is the whole
    thesis, and a green minute happens about half the time.

    Here the candle is only the TRIGGER. M15 decides the side, M5 must not
    contradict it, and the M1 close is the moment that standing agreement
    becomes actionable. All three or nothing: two out of three is a
    disagreement, and a disagreement on a trade lasting minutes is a coin flip
    with costs attached.

    The news blackout is deliberately NOT here. This module reads bars and has
    no calendar; inventing one would put a second, divergent copy of the news
    rules beside the real one. `news_filter`, `headline_filter` and the runner's
    observer hold that.
    """

    enabled: bool = True
    trigger_timeframe: str = "M1"
    confirm_timeframe: str = "M5"
    bias_timeframe: str = "M15"
    #: Bars the trigger candle is measured against. Thirty minutes of its own
    #: recent history is enough to say whether this one is unusual.
    candle_lookback: int = Field(default=30, ge=5, le=500)
    confirm_bars: int = Field(default=6, ge=1, le=200)
    bias_bars: int = Field(default=4, ge=1, le=200)
    #: Bars of the trigger timeframe, BEFORE the trigger candle, used to ask
    #: whether M1 was already going this way.
    #:
    #: M1 used to be read as a single candle -- body, wick, volume -- and never
    #: as a direction, so a lone green minute inside a falling M1 sequence was
    #: a buy provided M5 and M15 happened to point up. The side agreed with the
    #: slower charts and disagreed with the chart the trade was taken on.
    #:
    #: The trigger candle is excluded on purpose. It is by construction a big
    #: bar and it is the newest one, so including it would pull the fit its own
    #: way and the test would confirm itself.
    #:
    #: Unlike M5 and M15 this only has to NOT CONTRADICT. Flat is allowed: the
    #: first minute of a real push often follows a quiet stretch, and refusing
    #: those would remove the setups this module exists for.
    trigger_bars: int = Field(default=5, ge=2, le=200)
    #: The body above which a candle stops being momentum and becomes an event,
    #: in multiples of the recent average body.
    #:
    #: There was a ceiling on VOLUME and none on the move itself, although the
    #: argument for the first is entirely about size relative to normal.
    #: `minimum_body_multiple` is a floor and `body_saturation_multiple` only
    #: caps the SCORE, so a bar ten times its average body passed with maximum
    #: confidence. The strongest-looking bar such a system will ever see is the
    #: one it should not trade.
    #:
    #: Compounded by the geometry: the stop is one candle span, so an outsized
    #: candle produces an outsized R, and on an account too small to express it
    #: the position lands on the broker minimum and the money risked stops
    #: matching the plan.
    #:
    #: Sits above `body_saturation_multiple` (3.5), so every candle that scores
    #: full marks is still tradable and the refusal begins beyond it.
    maximum_body_multiple: float = Field(default=5.0, gt=0.0, le=100.0)
    #: How far a confirming timeframe must actually travel across its window
    #: before it counts as pointing anywhere, in multiples of that timeframe's
    #: own average bar range.
    #:
    #: THE DEFECT THIS CLOSES. `slope_direction` compared two closes with no
    #: minimum at all, so a change of one hundred-thousandth of a percent
    #: returned +1 with the same authority as a real trend. On a market going
    #: nowhere that sign is a coin flip, which made "M1, M5 and M15 all agree"
    #: a one-in-four COINCIDENCE instead of evidence -- and the trade that
    #: followed was a lone M1 candle with nothing behind it. This module's
    #: entire thesis is that the slower charts already point the same way; on a
    #: flat chart that thesis was being satisfied by noise.
    #:
    #: 0.5 means the window must have travelled at least half of one average
    #: bar. Below that the answer is 0, which reads upstream as a disagreement
    #: and refuses. Zero here restores the old behaviour of accepting any sign.
    minimum_slope_range: float = Field(default=0.5, ge=0.0, le=10.0)
    #: Bars of the confirm timeframe used to ask where in the recent move the
    #: price already is.
    confirm_range_bars: int = Field(default=20, ge=2, le=500)
    #:
    #: RECORDED ON EVERY READING, GATED ON NOTHING -- yet. The exhaustion
    #: refusal below asks "is there room left in front of this" of the TRIGGER
    #: CANDLE, sixty seconds of evidence, and never asks it of the move the
    #: trade is joining. That gap is real.
    #:
    #: What is not established is where the line sits. Refusing above 80% of
    #: the recent range kills the module: in any clean trend the newest close
    #: IS near the recent high, because that is what a trend is, so the gate
    #: would refuse precisely the setups this module exists to take. "It
    #: stopped losing because it stopped trading" is not a fix.
    #:
    #: So the number goes into `details` and into the journal on every reading,
    #: and after a day of live scalps it can be compared between the winners
    #: and the losers. Then a threshold is a measurement. Invented today it
    #: would be the third knob turned on a hunch on this module, and the first
    #: two both had to come back out.
    #: Body as a share of the whole candle. Below this it is mostly wick, which
    #: means nobody finished the minute in control of it.
    #: Raised from 0.55. "Only when it is convincing" is not a slogan, it is
    #: these three numbers, and the reason they can be this strict is that the
    #: flipped exits no longer need a 67% hit rate to survive.
    minimum_body_share: float = Field(default=0.70, ge=0.0, le=1.0)
    #: Body against the recent average body. This is the "it actually moved"
    #: test, scaled to the instrument instead of stated in pips.
    minimum_body_multiple: float = Field(default=2.2, gt=0.0, le=20.0)
    body_saturation_multiple: float = Field(default=4.0, gt=0.0, le=50.0)
    #: WHERE THE ACCOUNT-ENDING TRADE IS REFUSED.
    #:
    #: A minute carrying this many times its own normal activity is not
    #: momentum, it is an event — a release, a headline, a stop cascade. It
    #: produces the strongest-looking candle a fast bot will ever see, and what
    #: follows is a spread that takes the account apart. The owner's
    #: instruction was explicit: when the volume is extreme, do not trade.
    extreme_volume_multiple: float = Field(default=3.0, gt=1.0, le=100.0)
    #: A candle closing this far into its own extreme is the last buyer
    #: finishing the move rather than the first one starting it.
    exhaustion_close_position: float = Field(default=0.92, ge=0.5, le=1.0)
    #: THE CANDLE MUST BE BIG ENOUGH TO PAY FOR ITSELF, in multiples of the
    #: live spread. This was promised in the module docstring and not built,
    #: which is the exact defect this codebase keeps producing: a refusal that
    #: exists in prose and not in an `if`.
    #:
    #: The arithmetic says it is the most important guard here, not a detail.
    #: With a target smaller than the stop the break-even hit rate is already
    #: 58% before a cent of cost. Adding a realistic spread:
    #:
    #:     XAUUSD, M1 range 1.50, spread 0.25    ->  67% needed
    #:     XAUUSD, M1 range 1.50, spread 0.40    ->  72%
    #:     SPX500, M1 range 1.00, spread 0.50    ->  84%   unreachable
    #:
    #: The last row is the point. On a quiet minute the spread is most of the
    #: target and no hit rate saves it. At 5 spreads of clearance the floor
    #: settles near 66%, which is at least a number a selective filter can
    #: plausibly reach.
    minimum_target_spreads: float = Field(default=8.0, ge=0.0, le=100.0)
    #: Minutes of clear air required before a release. Read from the calendar
    #: the rest of the account already uses; this module has no calendar of its
    #: own, on purpose.
    news_clearance_minutes: float = Field(default=15.0, ge=0.0, le=240.0)
    #: CUT THE LOSER FAST. KEEP THE WINNER. These two numbers were the wrong
    #: way round and the arithmetic is not close.
    #:
    #: The first shape was stop 1.1, target 0.8 — small in, small out — on the
    #: reasoning that a scalp should take what it can get. On gold at a $0.25
    #: spread that needs a 67% hit rate. Worse, the instinct taken further makes
    #: it impossible: a $0.20 target is BELOW the spread, so the net win is
    #: minus five cents and no hit rate on earth pays for it.
    #:
    #:     target   stop    net win   net loss   hit rate needed
    #:       0.20   1.65      -0.05      -1.90   impossible
    #:       1.20   1.90      +0.95      -2.15   69%
    #:       1.20   0.60      +0.95      -0.85   47%
    #:       1.20   0.40      +0.95      -0.65   41%
    #:       0.50   0.40      +0.25      -0.65   72%
    #:
    #: Both middle rows are "get out fast" and they point in opposite
    #: directions. Cutting the LOSER fast takes 69% down to 41%. Cutting the
    #: WINNER fast takes it back up to 72%, because the spread does not shrink
    #: with the target. Speed helps on one side of the trade only.
    #:
    #: So: the stop is a fraction of the trigger candle — if the minute was
    #: read wrong, that is known almost immediately — and the target is larger
    #: than the candle, because the winners are the only thing paying for all
    #: of this.
    stop_candle_spans: float = Field(default=0.4, gt=0.0, le=10.0)
    target_candle_spans: float = Field(default=1.2, gt=0.0, le=10.0)
    #: Asset classes this may trade, by the only rule that matters here:
    #: WHERE COMMISSION IS ZERO.
    #:
    #: Empty means "every class whose round-trip commission is nil", derived
    #: from `risk.commission_by_asset_class` rather than typed out. Naming
    #: classes here would be a second list to keep in step with the first, and
    #: the direction it would fall out of step is "scalping forex at EUR 5.50
    #: a lot". A scalp's whole margin is a few spreads wide; a fixed fee per
    #: lot is not a cost it can carry.
    tradable_asset_classes: tuple[str, ...] = ()

    #: Trade this section through its OWN path instead of the confluence vote.
    #:
    #: WHY IT NEEDS ONE. The confluence score is a weighted mean of (raw score
    #: x confidence) over the agreeing modules, and this module's ceiling is
    #: 45 x 0.75 = 33.75. Against a bar of 45 it can never open a trade alone,
    #: and joining a strong reader makes things worse rather than better --
    #: market_structure alone scores 70, and 56.4 with this agreeing. A scalp
    #: voting in a swing engine drags every score it touches downward.
    #:
    #: That is not a threshold to tune. A scalp's evidence is small and
    #: short-lived by construction, which is what a scalp IS, and a machine
    #: built to weigh swing evidence will always price it low. So it gets its
    #: own lane: its own entry, its own stop and target off the trigger candle,
    #: its own concurrency limit and its own breaker.
    #:
    #: What it does NOT get its own copy of: the news blackout, the kill
    #: switch, the capital floor, the margin check or the spread guard. Those
    #: are account-wide and a second copy of any of them would eventually
    #: disagree with the first.
    own_lane_enabled: bool = False
    #: What one scalp may risk, as a percentage of equity. Zero falls back to
    #: `risk.risk_per_trade_pct`.
    #:
    #: WHY THIS EXISTS. The lane used to send a FIXED 0.01 lots on every
    #: instrument, and a fixed lot means a different risk on every instrument:
    #: 0.01 of gold behind a 3.41 dollar stop is EUR 2.91, and 0.01 of an index
    #: behind its own stop is something else entirely. Sizing by risk is what a
    #: position sizer is for, and the lane was the only route to an order that
    #: did not use it.
    #:
    #: The original reason for the fixed lot was recorded here and has since
    #: expired: "risk-sizing it to 5% would put EUR 9 behind a stop measured in
    #: seconds". The stop was 0.4 of an M1 candle when that was written. It is
    #: 1.0 now -- about 3.41 dollars on gold, roughly eight spreads -- so the
    #: stop is no longer measured in seconds.
    #:
    #: 3.0 rather than the account's 5.0, because this lane is still the only
    #: live thing on the account with no measured record at all.
    risk_pct: float = Field(default=3.0, ge=0.0, le=20.0)
    #: A hard ceiling in lots regardless of what the risk model sized, or 0.0
    #: for no ceiling.
    #:
    #: It is a ceiling and never an override: a symbol whose stop is too wide
    #: for the account is still refused rather than forced through, and this
    #: cannot round a position UP to reach it.
    fixed_lots: float = Field(default=0.0, ge=0.0, le=1.0)

    #: Minutes after which a scalp takes whatever profit it has.
    #:
    #: THE THESIS HAS AN EXPIRY AND THE TARGET DID NOT. This module enters on
    #: one M1 candle inside an agreement the slower frames already had, and the
    #: hypothesis document is explicit about why the trade is short: "the flow
    #: continues while they are still working it... the trade is a few minutes
    #: long because that is how long the imbalance lasts."
    #:
    #: The exit was a price, not a clock. A live scalp on 26 August reached
    #: +EUR 1.20 against a EUR 3.00 target and came back to -EUR 0.90 against a
    #: EUR 1.01 stop -- a round trip of more than 2R, on a thesis that had
    #: expired minutes earlier. The give-back rule would have closed it at
    #: +0.60R and never saw the peak: position management samples on the cycle,
    #: and a gold candle moves further between two samples than the whole
    #: trade is worth.
    #:
    #: So the clock closes it. Once the minutes the imbalance was claimed to
    #: last have passed, a scalp in profit banks it rather than waiting for a
    #: level the move was never going to reach.
    #:
    #: Zero disables it and restores the price-only exit.
    #:
    #: DEFAULTS TO OFF. A clock was the first answer and the wrong one: the
    #: owner asked for a judgement, not a timer -- "hey dit gaat nog verder
    #: stijgen, hold; hey dit zakt al een beetje, laat me claimen". That is
    #: `scalp_claim_spreads` below. This stays only as a far backstop for a
    #: position nothing else has an opinion about.
    maximum_age_minutes: float = Field(default=0.0, ge=0.0, le=120.0)

    #: How far a scalp may come back off its high-water mark before the gain is
    #: taken, measured in SPREADS.
    #:
    #: This is the rule the owner actually described. While the move is still
    #: going his way the position is held; the moment it starts sagging the
    #: profit is claimed. Not a fraction of the peak and not a clock -- a
    #: distance, in the only unit the market charges in.
    #:
    #: In spreads rather than in R because every level in this section that was
    #: set in R or in candle spans turned out to sit inside its own costs: the
    #: profit lock at 0.08R, the stop at 3.2 spreads. A give-back of one spread
    #: is the smallest retreat that is not simply the bid and the ask taking
    #: turns.
    scalp_claim_spreads: float = Field(default=1.5, ge=0.0)
    #: The leash as a SHARE OF THE PEAK, with the spread figure above as its
    #: floor. The exit fires on whichever of the two is larger.
    #:
    #: WHY A SHARE AND NOT A FIXED DISTANCE. A fixed one-spread leash extracts
    #: more from any given peak, which is exactly why it looked right. What it
    #: also does is end the trade on the first ordinary retracement, at every
    #: height equally -- thirty cents of gold going the wrong way closes a
    #: position that was on its way to two euro. The high peaks it reads so
    #: well are peaks it stops the trade from ever reaching.
    #:
    #: So the leash grows with the move. At two spreads the trade is on a tight
    #: rein because it has proved almost nothing; at twelve it may breathe by
    #: nearly five, because a move that size pulls back that far on the way up.
    #: This is a hold rule, not a claim rule: "het is niet vroeg claimen, het
    #: is vroeg veilig stellen en dan proberen zoveel uit te halen".
    #:
    #: SAFE IS STILL SAFE, and it falls out rather than being bolted on: any
    #: share below 1.0 leaves the exit above the entry, so a scalp that has
    #: armed can no longer come back as a loss.
    scalp_giveback_share: float = Field(default=0.40, ge=0.0, lt=1.0)
    #: The peak has to have been worth something before either rule above may
    #: act. Read against the HIGH-WATER MARK and never the live price: testing
    #: it against the live price is what let a trade fall out of claim range by
    #: giving back exactly the profit the rule exists to protect.
    scalp_claim_minimum_spreads: float = Field(default=2.0, ge=0.0)
    #: How far the last closed trigger bar must run AGAINST an already-losing
    #: scalp before it is cut rather than left to the stop, in spreads.
    #:
    #: The other half of the same sentence: "hey dit staat verlies, nee man dit
    #: gaat verder zakken, eruit". Not every loser -- only one the trigger
    #: timeframe is still actively moving away from. A losing position that has
    #: stopped falling keeps its stop, because closing it pays a spread to book
    #: a loss the stop books for free.
    scalp_cut_spreads: float = Field(default=3.0, ge=0.0)
    #: How much profit is worth banking when the clock runs out, as a multiple
    #: of the spread. Below this the trade has not cleared its own round trip
    #: and closing early just pays to lose slightly less.
    stale_minimum_spreads: float = Field(default=1.5, ge=0.0)
    #: At most this many section-six positions at once, inside the account's
    #: overall concurrency cap rather than beside it. Enforced on paper too,
    #: or the record measures a book the account has no room to hold.
    max_concurrent: int = Field(default=2, ge=1, le=10)
    #: The largest position a candle scalp may take, in lots, regardless of
    #: what the risk model sized.
    #:
    #: The owner's instruction and he is right: "doe gwn MAX 0.01 lot met die
    #: dingen op xauusd ... trade op candles ... GWN IN EN ERUIT". The sizer
    #: spends 3% of equity, and on a stop four tenths of an M1 candle wide
    #: that buys a great many lots. This section's whole premise is a position
    #: small enough that being wrong costs cents, taken often — the opposite
    #: instruction to the same function.
    #:
    #: What 0.01 is worth, from his own fill: XAUUSD 4667.47 to 4671.25 paid
    #: EUR 3.24, so EUR 0.86 per dollar of gold. A stop of about a dollar
    #: risks EUR 0.86 on a EUR 176 account. Half a percent, where the sizer
    #: wanted three.
    #:
    #: Applied as a ceiling, never a floor. It can only ever reduce what the
    #: sizer approved, and rounding UP to reach the broker minimum remains
    #: forbidden. Zero disables the cap.
    maximum_lots: float = Field(default=0.01, ge=0.0, le=100.0)
    base_score: float = Field(default=45.0, ge=0.0, le=100.0)
    base_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    maximum_confidence: float = Field(default=0.75, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _bounds_are_coherent(self) -> CandleMomentumConfig:
        if self.body_saturation_multiple <= self.minimum_body_multiple:
            raise ValueError(
                "analysis.candle_momentum.body_saturation_multiple must exceed "
                "minimum_body_multiple"
            )
        if self.maximum_confidence < self.base_confidence:
            raise ValueError(
                "analysis.candle_momentum.maximum_confidence may not be below base_confidence"
            )
        return self


class OrderBlockConfig(Base):
    """SECTION THREE. Found by the search that ran after section two shipped.

    The first search produced only variants of the retest -- `retest_slow`
    shared 47.5% of its trades with section two, which is the same section
    wearing a hat. Round three was restricted to mechanisms that can fire when
    a retest cannot: fractal swings, Fibonacci proportions, week boundaries,
    session VWAP, cross-market divergence, and order blocks. Only this lived.

        FX M30, 31,376 trades with every section-two bar removed
        hit 62.1%   net +0.164R   train +0.168 / holdout +0.161
        every year positive; 114 of 114 months positive
    """

    enabled: bool = True
    #: M30. H1 measured better (+0.193R) with 40% fewer trades; M15 worse
    #: (+0.156R) with twice as many. M30 is also a different clock from
    #: section two's M15, which keeps the two from crowding the same bars.
    timeframe: str = "M30"
    atr_period: int = Field(default=14, ge=2, le=200)
    lookback_bars: int = Field(default=96, ge=4, le=500)
    #: THE THRESHOLD THAT CARRIES THE EDGE, measured on the impulse bar's BODY
    #: (close minus open), not its range:
    #:
    #:     >= 1.0 ATR   122,927 trades   net +0.083R
    #:     >= 1.5 ATR    37,049 trades   net +0.172R
    #:     >= 2.0 ATR    12,441 trades   net +0.185R
    #:
    #: A one-ATR candle runs over nobody. 2.0 is barely better on a third of
    #: the sample.
    minimum_impulse_atr: float = Field(default=1.5, gt=0.0, le=10.0)
    #: Impulse above the minimum at which the score saturates.
    impulse_span_atr: float = Field(default=1.5, gt=0.0, le=10.0)
    #: How far back to look for the last opposite-coloured candle.
    block_search_bars: int = Field(default=5, ge=1, le=50)
    #: How far INTO the block's body the limit rests. 0.25 measured +0.172R
    #: against 0.50 at +0.146R: the zone is defended at its edge, and reaching
    #: deeper into it is reaching past the orders.
    zone_tolerance_atr: float = Field(default=0.25, ge=0.0, le=2.0)
    #: 1.0 ATR from the fill. Clears `ConfluenceConfig.min_stop_atr: 0.8`
    #: untouched, so nothing widens it behind the module's back.
    stop_atr: float = Field(default=1.0, gt=0.0, le=5.0)
    #: Gold again: 0.10 ATR of spread over a 1.0 ATR stop is 20% of R and
    #: `max_spread_share_of_stop: 0.08` refuses it. At 1.5 ATR it is 6.7%.
    stop_atr_by_symbol: dict[str, float] = Field(default_factory=lambda: {"XAUUSD": 1.5})
    #: 58 and not 54. A lone module's score is |raw| x confidence, so 54 x 0.66
    #: is 35.6 against a `score_threshold` of 35.0 -- half a point of margin,
    #: and any drift in either number silently switches the section off. 58
    #: gives 38.3, the same margin section two carries.
    base_score: float = Field(default=58.0, ge=0.0, le=100.0)
    quality_score_bonus: float = Field(default=14.0, ge=0.0, le=50.0)
    #: The measured hit rate, and it must clear
    #: `lone_module_minimum_confidence` because every one of those 31,376
    #: trades was taken on this signal alone.
    base_confidence: float = Field(default=0.66, ge=0.0, le=1.0)
    quality_confidence_bonus: float = Field(default=0.18, ge=0.0, le=1.0)
    maximum_confidence: float = Field(default=0.84, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _confidence_is_ordered(self) -> OrderBlockConfig:
        if self.maximum_confidence < self.base_confidence:
            raise ValueError(
                "analysis.order_block.maximum_confidence may not be below base_confidence"
            )
        return self


class ImpulseRetestConfig(Base):
    """SECTION TWO. The one strategy here chosen by measurement.

    Ninety-four detectors over sixteen instruments and six timeframes; this is
    what survived a Bonferroni bar across the whole grid, a holdout that had to
    agree on its own, a day-clustered sigma, and a coin-flip control that
    turned out NOT to read zero.

        M15   18,828 trades   hit 67.9%   net +0.279R  (train +0.272/test +0.285)
        H1     5,235 trades   hit 67.9%   net +0.281R  (train +0.276/test +0.284)

    Positive in all eleven years, +0.181R to +0.394R, no losing year.
    """

    enabled: bool = True
    #: M15. H1 measured identically (+0.281R against +0.279R) and either is
    #: defensible; M15 gives three times the trades for the same edge.
    timeframe: str = "M15"
    channel_period: int = Field(default=20, ge=5, le=200)
    atr_period: int = Field(default=14, ge=2, le=200)
    #: 96 bars is the horizon every trade was resolved over.
    lookback_bars: int = Field(default=96, ge=4, le=500)
    #: THE FILTER THAT MAKES IT WORK. The break bar must CLOSE at least this
    #: far past the level. Without it the same retest nets roughly zero; with
    #: it, +0.28R. A break that closes a full ATR clear has taken every resting
    #: offer on the way and left a queue; one that pokes through has not.
    minimum_impulse_atr: float = Field(default=1.0, ge=0.0, le=5.0)
    #: Impulse above the minimum at which the score saturates.
    impulse_span_atr: float = Field(default=1.5, gt=0.0, le=10.0)
    #: How close to the level the fill must be. The edge is monotone in this
    #: and changes sign by 0.60 ATR out.
    tolerance_atr: float = Field(default=0.15, gt=0.0, le=1.0)
    #: 0.85 + 0.15 = 1.00 ATR of risk. 0.50 ATR measured better (+0.242R at a
    #: 2R target) but sits under `ConfluenceConfig.min_stop_atr: 0.8`, and a
    #: floor that silently widens the stop would leave this measuring one trade
    #: and sending another.
    stop_beyond_atr: float = Field(default=0.85, gt=0.0, le=3.0)
    #: A WIDER STOP WHERE THE SPREAD IS WIDER, per symbol.
    #:
    #: Gold on the owner's explicit instruction, 30 August, and it needed a
    #: measurement rather than a smaller lot. Cost in R is spread/R, which is a
    #: RATIO: halving the lot halves the win, the loss and the spread together
    #: and leaves the R-multiple exactly where it was. Minimum risk changes what
    #: a trade pays in euros, never whether it pays at all.
    #:
    #: What does change it is the denominator. XAUUSD carries about 0.10 ATR of
    #: spread against 0.04 on a major, so at the family's 1.00 ATR stop it
    #: spends 20% of R on execution -- and `risk.max_spread_share_of_stop: 0.08`
    #: refuses it, correctly. At 1.50 ATR the same spread is 6.7% and passes the
    #: gate untouched, which is the point: the gate is not being loosened for
    #: gold, gold is being given a stop it can afford.
    #:
    #: Measured on the same 1,942 trades, stop 1.50 ATR at the family's 1R
    #: target:
    #:
    #:     gross +0.214R   cost 0.133R   net +0.083R   (train +0.065/test +0.100)
    #:
    #: Positive in both halves. Smaller than FX and real.
    stop_beyond_atr_by_symbol: dict[str, float] = Field(default_factory=lambda: {"XAUUSD": 1.35})
    base_score: float = Field(default=58.0, ge=0.0, le=100.0)
    quality_score_bonus: float = Field(default=14.0, ge=0.0, le=50.0)
    #: CONFIDENCE IS THE MEASURED HIT RATE, and it has to be, because this is a
    #: STANDALONE strategy and `lone_module_minimum_confidence` is 0.65.
    #:
    #: At 0.58 the typical setup was silently unable to trade on its own. The
    #: fill sits at `level + tolerance`, so `closeness` is near zero on almost
    #: every real signal, and a median impulse leaves confidence around 0.60 --
    #: under the gate. The module would then only ever have traded when some
    #: other detector happened to agree, which is not the thing that was
    #: measured at 67.9% over 18,828 trades. A module that cannot reach the
    #: path that makes it work is the defect this whole account keeps
    #: producing, and it would have looked like the strategy simply not firing.
    #:
    #: 0.68 is not a preference. It is what the strategy hits.
    base_confidence: float = Field(default=0.68, ge=0.0, le=1.0)
    quality_confidence_bonus: float = Field(default=0.22, ge=0.0, le=1.0)
    maximum_confidence: float = Field(default=0.88, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _confidence_is_ordered(self) -> ImpulseRetestConfig:
        if self.maximum_confidence < self.base_confidence:
            raise ValueError(
                "analysis.impulse_retest.maximum_confidence may not be below base_confidence"
            )
        return self


class LevelRetestConfig(Base):
    """The only detector on this account chosen by measurement, not argument.

    Ten years of M15, eight instruments, 1.84 million bars, first-touch with
    the stop checked inside the fill bar and unresolved trades excluded:

        buying the channel break    190,505 signals   E -0.067R   -29 sigma
        buying its retest            55,582 signals   E +0.134R   +22 sigma

    Same detector, same breaks, 0.2R apart on where the trade is entered.
    """

    enabled: bool = True
    #: M15 because that is the timeframe the edge was measured on. Nothing
    #: here has been shown to hold on M1, and the three M1 modules on this
    #: account are exactly the ones that were never measured at all.
    timeframe: str = "M15"
    #: Both 20 and 40 measured well (+22 and +23 sigma at the shipped stop);
    #: 20 gives 40% more signals for a slightly smaller edge, and this account
    #: needs trades.
    channel_period: int = Field(default=20, ge=5, le=200)
    atr_period: int = Field(default=14, ge=2, le=200)
    #: How far back a break stays worth retesting. 96 M15 bars is 24 hours,
    #: the horizon the measurement resolved trades over.
    lookback_bars: int = Field(default=96, ge=4, le=500)
    #: THE PARAMETER. The edge is monotone in it and changes sign:
    #:
    #:     0.15 ATR   +11.3 points over chance   +54 sigma
    #:     0.35 ATR    +5.0 points               +20 sigma
    #:     0.60 ATR    -3.4 points               NEGATIVE
    #:
    #: Slack here is not tolerance, it is the edge being given away.
    tolerance_atr: float = Field(default=0.15, gt=0.0, le=1.0)
    #: 0.35 measured better (E +0.340 against +0.134) but sits under
    #: `ConfluenceConfig.min_stop_atr`, and a floor that silently widens the
    #: stop would leave this measuring one trade and sending another.
    #: 0.15 + 0.75 = 0.90 ATR clears the floor.
    stop_beyond_atr: float = Field(default=0.75, gt=0.0, le=3.0)
    base_score: float = Field(default=55.0, ge=0.0, le=100.0)
    closeness_score_bonus: float = Field(default=15.0, ge=0.0, le=50.0)
    base_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    closeness_confidence_bonus: float = Field(default=0.20, ge=0.0, le=1.0)
    maximum_confidence: float = Field(default=0.85, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _confidence_is_ordered(self) -> LevelRetestConfig:
        if self.maximum_confidence < self.base_confidence:
            raise ValueError(
                "analysis.level_retest.maximum_confidence may not be below base_confidence"
            )
        return self


class VwapReversionConfig(Base):
    """SECTION TWO, replacing `drift_burst`. How far is price from the day?

    VWAP is the volume-weighted average price since the session opened -- not
    an indicator drawn over the chart but a summary of it, the average price at
    which the day's business was actually done. An execution desk is measured
    against it, so a desk working a large order becomes a buyer below it and a
    seller above it. That is a real, dated, one-directional pressure, and it is
    the only defensible half of the usual "price snaps back to VWAP" telling.

    WHY THIS ACCOUNT, from its own scorecard and not from theory: every losing
    regime is a sideways one (`range` -2.04R, `transition` -1.76R) while the two
    trending ones are positive. That is exactly where trend followers should
    lose and mean reversion should work -- and the only mean-reversion reader on
    the book fired nine times in a day against `trend_momentum`'s 58,612.

    IT REPLACES SECTION TWO because `drift_burst` finished 89 observed setups at
    19% hit and -64.07R, and because a slot that has already been given the
    paper-first discipline is the honest place to put a tenth detector.
    """

    enabled: bool = True
    #: Bars per decision. M5 over a full session is roughly 280 bars, which is
    #: enough for a stable volume-weighted mean without carrying yesterday.
    timeframe: str = "M5"
    #: How much session to summarise. Deliberately a bar count and not a clock:
    #: this account trades gold, indices and FX, and their sessions open at
    #: different hours. 288 M5 bars is 24 hours, so on a continuous market it is
    #: a rolling day and on a session market it is everything since the open
    #: plus whatever preceded it -- both of which are the volume that traded.
    session_bars: int = Field(default=288, ge=30, le=2000)
    #: The stretch that is worth fading, in the session's OWN standard
    #: deviations rather than pips. Two sigma on a quiet day and two sigma on a
    #: violent one are different distances and the same statement.
    minimum_sigma: float = Field(default=2.0, ge=0.5, le=6.0)
    #: Past this the score stops rising. A five-sigma print is usually a data
    #: fault or a halt, and it must not outrank the population this was built
    #: for.
    saturation_sigma: float = Field(default=1.5, gt=0.0, le=6.0)
    #: How many bars back the stall is measured over.
    stall_bars: int = Field(default=3, ge=1, le=30)
    #: THE HALF THAT STOPS IT STANDING IN FRONT OF A TRAIN. Distance alone
    #: cannot separate a mispricing from a repricing: two sigma below VWAP on a
    #: quiet drift and two sigma below on a release are the same number with
    #: opposite futures. So the stretch must have STOPPED -- the last
    #: `stall_bars` may add no more than this much of it, in the same units.
    #: Zero would demand a perfectly flat pause; 0.25 allows a drifting one.
    maximum_extension: float = Field(default=0.25, ge=0.0, le=2.0)
    #: The score at exactly `minimum_sigma`, rising to `maximum_score` at
    #: saturation. Starts under the live threshold of 35 on purpose: a bare
    #: two-sigma stretch should need a second reader to agree, and only a real
    #: outlier speaks for itself.
    base_score: float = Field(default=30.0, ge=0.0, le=100.0)
    maximum_score: float = Field(default=70.0, ge=0.0, le=100.0)
    base_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    maximum_confidence: float = Field(default=0.75, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _score_and_confidence_rise(self) -> VwapReversionConfig:
        if self.maximum_score < self.base_score:
            raise ValueError(
                "analysis.vwap_reversion.maximum_score may not be below base_score -- "
                "a bigger stretch cannot score less than a smaller one"
            )
        if self.maximum_confidence < self.base_confidence:
            raise ValueError(
                "analysis.vwap_reversion.maximum_confidence may not be below " "base_confidence"
            )
        return self


class BasketDivergenceConfig(Base):
    """SECTION FIVE. One index stepped out of line with the others.

    The first reader here that is MEANINGLESS on a single chart, which is
    exactly why it can corroborate the nine that all read one. Equity indices
    run 80-95% together intraday; when the basket moves and one member has not,
    the thing that moved four of them has not yet moved the fifth.

    High hit rate by construction rather than by tuning: this does not forecast
    the market, it bets a gap closes. Both sides may go up, down or nowhere.

    Live-capable where sections two and four are not. Those rest on a statistic
    the research measured on TICK data, and whether it survives M1 bars is an
    open question. A move between two M1 closes has no such question.

    NOT ARBITRAGE. There is no margin here for two legs, so this trades one
    side — the laggard, toward the basket. A directional trade with a relative
    trigger. When an index decouples for a real reason the gap widens instead
    of closing, and that is where the losses live.
    """

    enabled: bool = True
    timeframe: str = "M1"
    #: Bars the comparison move is measured over. Long enough that a single
    #: print does not read as a divergence, short enough that the gap has not
    #: already closed by the time it is seen.
    move_bars: int = Field(default=15, ge=3, le=240)
    #: Fewest fresh peers before this says anything. Two, because a "basket" of
    #: one is a pair and one halted or gapping instrument would then define the
    #: whole reference.
    minimum_peers: int = Field(default=2, ge=2, le=20)
    #: How recent a peer's last bar must be.
    #:
    #: THIS IS THE SESSION GUARD, and it is deliberately a measurement rather
    #: than a calendar. FRA40 does not keep SPX500's hours, and comparing a
    #: closed market to an open one compares a stale price to a live one — it
    #: manufactures gaps that will never close because one side stopped
    #: printing hours ago. A session table can be wrong about a holiday or a
    #: half-day; a last-bar timestamp cannot.
    peer_max_age_seconds: float = Field(default=180.0, gt=0.0, le=3600.0)
    #: How far out of line before it is a setup, in basis points.
    minimum_gap_bp: float = Field(default=20.0, gt=0.0, le=1000.0)
    #: The basket itself must have moved this far, or the "gap" is this market
    #: running on its own while the peers sat still — the decoupling this
    #: module is most wrong about, and the opposite trade.
    minimum_basket_bp: float = Field(default=15.0, ge=0.0, le=1000.0)
    #: Where the score saturates.
    gap_saturation_bp: float = Field(default=60.0, gt=0.0, le=2000.0)
    base_score: float = Field(default=50.0, ge=0.0, le=100.0)
    base_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    maximum_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    #: Asset classes whose members are compared with each other. Indices only
    #: to begin with: they are the group whose co-movement is a fact about the
    #: same underlying risk rather than a correlation that happens to hold.
    grouped_asset_classes: tuple[str, ...] = ("index",)

    @model_validator(mode="after")
    def _bounds_are_coherent(self) -> BasketDivergenceConfig:
        if self.gap_saturation_bp <= self.minimum_gap_bp:
            raise ValueError(
                "analysis.basket_divergence.gap_saturation_bp must exceed minimum_gap_bp"
            )
        if self.maximum_confidence < self.base_confidence:
            raise ValueError(
                "analysis.basket_divergence.maximum_confidence may not be below " "base_confidence"
            )
        return self


class DriftBurstConfig(Base):
    """SECTION TWO. Is this move real, or is it someone in a hurry?

    Every other module here reads the SHAPE of the price series. This one runs
    a hypothesis test on it: is the local mean return larger than the local
    volatility can explain? Christensen, Oomen and Renò (Journal of
    Econometrics, 2022) call a reading above roughly 4 a DRIFT BURST, and
    measured on tick data across equities, fixed income, currencies and
    commodities they find about one per week per instrument, 25 to 200 basis
    points apiece, with TWO THIRDS followed by price reversion.

    So this module fades. That is the point of it: nine detectors follow and
    none of them can be the second family the corroboration rule needs, because
    they all read the same chart. The reversion is not a repeating pattern — it
    is payment for supplying immediacy to someone who overpaid to get out, which
    is also where the short-term reversal literature independently lands.

    IT DOES NOT TRADE. Deliberately absent from `live_enabled_modules` while its
    weight lets the offline backtest measure it, and the runner records what it
    would have done as a shadow trade the resolver settles against real later
    prices. The paper works on ticks and this account has M1 bars; whether the
    statistic survives that coarsening is the open question this exists to
    answer, and it is not a question worth answering with money.
    """

    enabled: bool = True
    #: The fastest series the account keeps. A burst is short-lived by
    #: definition, so a slower frame averages the explosive part away.
    timeframe: str = "M1"
    #: Bars the DRIFT is measured over. Short, because a burst is short.
    drift_window: int = Field(default=10, ge=2, le=200)
    #: Bars the VOLATILITY is measured over, and it must be much longer.
    #:
    #: TWO WINDOWS, AND THE SECOND IS NOT A REFINEMENT. Built with one window
    #: the test could not see the events it exists for: measured over 3,000
    #: synthetic paths, a 20-bar burst fired 68% of the time while a 5-bar
    #: burst covering 127 basis points — harder and faster — fired 0.0%. Never
    #: once. The burst was inflating the very denominator meant to judge it.
    #:
    #: With the volatility taken over 120 bars that are mostly the calm BEFORE
    #: the move, the same 5-bar burst fires 96-100%. Separate bandwidths are
    #: what the paper does; collapsing them was the bug.
    volatility_window: int = Field(default=120, ge=20, le=2000)
    #: Half the drift kernel's weight sits in the last N bars. A burst is LOCAL,
    #: and flat weighting inside the drift window mixes the explosion with
    #: whatever preceded it.
    half_life_bars: float = Field(default=5.0, gt=0.0, le=200.0)
    #: Autocovariance lags in the volatility correction.
    #:
    #: Not cosmetic. Bid-ask bounce makes consecutive returns negatively
    #: autocorrelated, which inflates a naive variance — and an inflated
    #: denominator SHRINKS the statistic, so the test would go blind exactly on
    #: the noisiest instruments, which are the ones bursting most often.
    noise_lags: int = Field(default=2, ge=0, le=20)
    #: Where a burst starts, and it is CALIBRATED rather than borrowed.
    #:
    #: Over 4,000 pure random walks at these settings the statistic reached a
    #: 99th percentile of 2.58 and a maximum of 3.55. So 4.0 is a false-positive
    #: rate under one in four thousand on noise — and real bursts of 75bp and
    #: up fire 96-100% of the time. The paper's threshold is also near 4, which
    #: is reassuring, but this one was measured on THIS estimator and M1 bars
    #: rather than inherited from tick data.
    #:
    #: Caveat worth keeping: one in four thousand per reading is not one in
    #: four thousand per day. Across 845 instruments read every minute the
    #: false positives add up, which is one more reason this observes before it
    #: trades.
    t_threshold: float = Field(default=4.0, ge=2.0, le=12.0)
    #: Where the score saturates. Between the threshold and here the fade grows
    #: from `base_score` to 100.
    t_saturation: float = Field(default=8.0, ge=3.0, le=30.0)
    #: A burst too small to be the paper's event is not the paper's event.
    #:
    #: The statistic is a RATIO, so a dead-quiet instrument that twitches
    #: produces a large t on a two-basis-point move. The research describes
    #: 25-200bp events; under this floor the reversion has nothing in it to
    #: collect and the spread takes what is left.
    minimum_move_bp: float = Field(default=25.0, ge=0.0, le=1000.0)
    base_score: float = Field(default=45.0, ge=0.0, le=100.0)
    base_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    maximum_confidence: float = Field(default=0.80, ge=0.0, le=1.0)

    # -- SECTION FOUR: the same statistic, cast wider and gated on cost -----
    #
    # `t_threshold` is deliberately strict, which makes the event rare — about
    # one a week per instrument. Section four asks what a lower bar buys, and
    # the answer was measured over 6,000 random walks rather than guessed:
    #
    #     threshold   false positive on noise   5-bar burst   3-bar burst
    #         4.0     0 in 6,000                    18%           41%
    #         3.5     1 in 6,000                    46%           73%
    #         3.0     1 in 333                      72%           91%
    #
    # 3.5 is where two to three times the events cost essentially nothing in
    # reliability. Below it the false positives start to matter.
    #
    # And then the condition that decides whether any of it is collectable.
    # Studying intraday reversals: on the NYSE "the large widening of the
    # bid-ask spread eliminates most of the profits that can be achieved by a
    # contrarian strategy", while on the NASDAQ the spread stays almost
    # constant and the same strategy yields significant abnormal profits. The
    # reversal exists in both markets; only one of them lets you keep it.
    #
    # A burst IS a liquidity shock, so the spread widening with it is the
    # normal case and the expensive one. This keeps the bursts where the
    # spread held against its own learned baseline for that symbol at that
    # hour — which is a measurement this account has been collecting all along
    # in `spread_observations` and never once read for this purpose.
    wide_net_enabled: bool = True
    wide_net_t_threshold: float = Field(default=3.5, ge=2.0, le=12.0)
    #: Current spread over the learned baseline. Above this the burst is
    #: recorded as section two only: the move is real, the profit is not.
    wide_net_max_spread_ratio: float = Field(default=1.25, ge=1.0, le=10.0)

    @model_validator(mode="after")
    def _saturation_above_threshold(self) -> DriftBurstConfig:
        if self.wide_net_t_threshold > self.t_threshold:
            raise ValueError(
                "analysis.drift_burst.wide_net_t_threshold may not exceed t_threshold — "
                "the wide net is the LOWER bar, and inverting them would make section "
                "four the stricter of the two while still being labelled the wider"
            )
        if self.volatility_window <= self.drift_window:
            raise ValueError(
                "analysis.drift_burst.volatility_window must exceed drift_window, or the "
                "burst inflates the very denominator meant to judge it"
            )
        if self.t_saturation <= self.t_threshold:
            raise ValueError(
                "analysis.drift_burst.t_saturation must exceed t_threshold, or the "
                "score has no range to grow across"
            )
        if self.maximum_confidence < self.base_confidence:
            raise ValueError(
                "analysis.drift_burst.maximum_confidence may not be below base_confidence"
            )
        return self


class M1MicroBreakoutConfig(Base):
    """A discrete closed-M1 range break aligned with M5 structure."""

    enabled: bool = True
    atr_period: int = Field(default=14, ge=2, le=200)
    base_bars: int = Field(default=6, ge=3, le=30)
    volume_lookback: int = Field(default=30, ge=10, le=200)
    m5_fast_ema: int = Field(default=9, ge=2, le=100)
    m5_slow_ema: int = Field(default=20, ge=3, le=200)
    m5_slope_bars: int = Field(default=3, ge=1, le=20)
    minimum_m5_separation_atr: float = Field(default=0.10, ge=0.0, le=5.0)
    minimum_m5_slope_atr: float = Field(default=0.03, ge=0.0, le=5.0)
    maximum_base_width_atr: float = Field(default=2.25, gt=0.0, le=10.0)
    minimum_break_atr: float = Field(default=0.05, ge=0.0, le=2.0)
    minimum_body_atr: float = Field(default=0.45, ge=0.0, le=5.0)
    minimum_close_location: float = Field(default=0.70, ge=0.5, le=1.0)
    minimum_volume_ratio: float = Field(default=1.20, ge=0.0, le=10.0)
    stop_buffer_atr: float = Field(default=0.20, ge=0.0, le=2.0)
    score: float = Field(default=62.0, ge=0.0, le=100.0)
    base_confidence: float = Field(default=0.48, ge=0.0, le=1.0)
    body_confidence_scale: float = Field(default=0.12, ge=0.0, le=2.0)
    volume_confidence_scale: float = Field(default=0.08, ge=0.0, le=2.0)
    maximum_confidence: float = Field(default=0.82, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _coherent(self) -> M1MicroBreakoutConfig:
        if self.m5_fast_ema >= self.m5_slow_ema:
            raise ValueError("M1 breakout fast M5 EMA must be below slow M5 EMA")
        if self.base_confidence > self.maximum_confidence:
            raise ValueError("M1 breakout base confidence exceeds maximum")
        return self


class DriftContinuationConfig(Base):
    """Join a move that is already happening, in the direction it is going.

    The hole this fills, found on a live day: `trend_momentum` runs on 20/50
    EMAs over H4 and H1, so it goes quiet long before a market turns and turns
    long after the move began; `liquidity_sweep` needs a wick through a 20-bar
    extreme on the very last candle and is a reversal pattern anyway. Between
    the slow module and the rare one sits an hour of clean one-way drift, and
    on 12 August GBPUSD spent the day in exactly that state while the engine
    tried 344 times to buy it and never once proposed a short.
    """

    enabled: bool = True
    #: M15 rather than H1: an hour of drift is four bars here and one there,
    #: and one bar cannot be tested for consistency at all.
    timeframe: str = "M15"
    lookback_bars: int = Field(default=8, ge=3, le=200)
    atr_period: int = Field(default=14, ge=2, le=200)
    #: Net movement over the window, in ATR, before this says anything. A
    #: market that has drifted a tenth of an ATR has not moved, it has breathed.
    minimum_drift_atr: float = Field(default=1.0, ge=0.0, le=20.0)
    #: Where confidence saturates. Twice the floor, so an ordinary qualifying
    #: move sits in the middle of the range rather than at its top.
    confident_drift_atr: float = Field(default=2.0, gt=0.0, le=40.0)
    #: Share of bars in the window that must close with the move.
    #:
    #: The condition that keeps this out of chop, and the reason it is not
    #: simply the refused long turned upside down. A market that ends the hour
    #: lower having gone up, down, up and down has the same net drift as one
    #: that ground steadily lower; only the second is going somewhere. The live
    #: exit note that named this failure put it as "4 of 5 closing adverse".
    minimum_consistency: float = Field(default=0.65, ge=0.0, le=1.0)
    #: Below the swing modules' 65-75. A drift is real evidence and it is the
    #: weakest kind here: it says the move happened, not that it continues.
    score: float = Field(default=55.0, gt=0.0, le=100.0)
    base_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    scale_confidence_by: float = Field(default=0.40, ge=0.0, le=2.0)
    maximum_confidence: float = Field(default=0.85, ge=0.0, le=1.0)

    @field_validator("timeframe")
    @classmethod
    def _supported_timeframe(cls, value: str) -> str:
        return Timeframe.parse(value).value

    @model_validator(mode="after")
    def _coherent(self) -> DriftContinuationConfig:
        if self.confident_drift_atr < self.minimum_drift_atr:
            raise ValueError("confident drift must not be below the minimum drift")
        if self.base_confidence > self.maximum_confidence:
            raise ValueError("drift base confidence exceeds maximum")
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
    #: H1 trend, M15 pullback, entry on the turn back. ~4 hours.
    #:
    #: The most ordinary thing a market does — go one way, rest, go on — and
    #: nothing saw it. The scalp wants compression right before the move, which
    #: a trend running for hours does not have; the fade wants a range, and a
    #: trend is the absence of one.
    trend_pullback: bool = True
    #: A break of the M15 range that could not hold and came back inside. ~2.5h.
    #:
    #: Cannot fire on the same bar as `range_break`, by construction: that one
    #: needs a close outside the edge, this one a close back inside. They read
    #: one event and disagree only about how it ended.
    failed_break: bool = True
    #: M15 range break, targeting a measured move, ~2 hours.
    #:
    #: The counterpart to `range_fade`, and worth having on even if it never
    #: opened a trade. Without it the system is not neutral about which of the
    #: two is happening: it fades every edge, including the ones that are
    #: giving way. With it, a genuine break has both theories pointing opposite
    #: ways and `veto_on_conflict` stands them both down.
    range_break: bool = True
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


class HorizonProfileConfig(Base):
    """Planning authority for one expected holding horizon.

    A short M15 reversal and an H1 swing may point the same way without being
    the same trade.  This keeps their noise floor, realistic travel window and
    higher-timeframe burden explicit instead of silently borrowing every rule
    from H1.
    """

    planning_timeframe: str = "H1"
    target_horizon_bars: int = Field(default=24, ge=2, le=200)
    htf_trend_timeframes: tuple[str, ...] = ("D1", "W1")
    minimum_htf_conflicts: int = Field(default=1, ge=1, le=5)
    htf_trend_veto: float = Field(default=1.0, gt=0.0, le=5.0)
    entry_timing_timeframes: tuple[str, ...] = ("M15", "M5")

    @field_validator("planning_timeframe")
    @classmethod
    def _planning_timeframe_is_supported(cls, value: str) -> str:
        Timeframe.parse(value)
        return value.upper()

    @field_validator("htf_trend_timeframes", "entry_timing_timeframes")
    @classmethod
    def _timeframes_are_supported(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for timeframe in value:
            Timeframe.parse(timeframe)
        return tuple(timeframe.upper() for timeframe in value)

    @model_validator(mode="after")
    def _conflict_count_fits_the_frames(self) -> HorizonProfileConfig:
        if self.htf_trend_timeframes and self.minimum_htf_conflicts > len(
            self.htf_trend_timeframes
        ):
            raise ValueError(
                "minimum_htf_conflicts may not exceed the number of htf_trend_timeframes"
            )
        return self


class EntryQualityConfig(Base):
    """Whether a valid direction is still offered at a tradeable price.

    Confluence answers *which way*. This policy answers the separate execution
    question *is now still a sensible moment*. ATR-normalised limits keep the
    same meaning across FX, crypto, indices and single-stock CFDs while the
    per-asset mappings acknowledge their different short-horizon behaviour.
    """

    enabled: bool = True
    #: Missing entry-timing data cannot prove that a market order is timely.
    fail_closed: Literal[True] = True
    timeframe: str = "M5"
    extension_bars: int = Field(default=3, ge=1, le=20)
    range_lookback_bars: int = Field(default=12, ge=4, le=100)
    #: How far up its own recent range a long may be entered. 0.88 refused any
    #: entry in the top 12% of the last twelve M5 bars.
    #:
    #: 0.95, and the owner's complaint names this gate exactly: gold went up
    #: like a rocket and nothing was taken. A market breaking out IS at the top
    #: of its recent range — that is what a breakout is — so a gate reading
    #: "near the high" as "too late" refuses every impulse it is shown and
    #: keeps only the quiet ones. `scorecard.py` priced it: ENTRY_OVEREXTENDED
    #: blocked 3 setups, 2 of them winners, and cost 3.00R.
    #:
    #: The chase protection that remains is the one measured in ATR rather than
    #: in percentile — `max_favourable_extension_atr` below — which asks how far
    #: price has actually run rather than merely where it sits.
    directional_extreme_location: float = Field(default=0.95, ge=0.5, le=1.0)
    ema_period: int = Field(default=20, ge=3, le=200)
    #: How far price may have already travelled the right way, in ATR, before
    #: the entry counts as a chase. Raised alongside the percentile above: with
    #: that gate loosened this is the one still doing the work, and 1.25 ATR on
    #: FX was cutting into ordinary impulse moves. The absolute protection —
    #: a single bar's body, and distance from the EMA — is unchanged below.
    max_favourable_extension_atr: dict[str, float] = Field(
        default_factory=lambda: {
            "forex": 1.75,
            "crypto": 2.00,
            "stock": 1.75,
            "index": 1.85,
            "metal": 1.85,
            "commodity": 1.85,
            "unknown": 1.50,
        }
    )
    max_single_bar_body_atr: dict[str, float] = Field(
        default_factory=lambda: {
            "forex": 1.00,
            "crypto": 1.30,
            "stock": 1.20,
            "index": 1.20,
            "metal": 1.20,
            "commodity": 1.20,
            "unknown": 1.00,
        }
    )
    #: A directional extreme plus a large final body is a special case. The
    #: ordinary body ceiling above must allow real breakouts; this lower limit
    #: says only that buying the top (or selling the bottom) of a one-bar thrust
    #: should wait for the same setup to be offered again.
    max_extreme_single_bar_body_atr: dict[str, float] = Field(
        default_factory=lambda: {
            "forex": 0.75,
            "crypto": 1.10,
            "stock": 1.00,
            "index": 1.00,
            "metal": 1.10,
            "commodity": 1.00,
            "unknown": 0.80,
        }
    )
    max_ema_distance_atr: dict[str, float] = Field(
        default_factory=lambda: {
            "forex": 1.35,
            "crypto": 1.60,
            "stock": 1.50,
            "index": 1.50,
            "metal": 1.50,
            "commodity": 1.50,
            "unknown": 1.20,
        }
    )
    #: Maximum favourable movement from the latest CLOSED entry-timeframe bar
    #: to the executable quote. This is deliberately separate from the three
    #: closed-bar extension limits above. A breakout may legitimately close
    #: far from its EMA; a market order another ATR beyond that already-closed
    #: evidence is simply paying late. Quick setups do not scale this limit.
    max_live_favourable_gap_atr: dict[str, float] = Field(
        default_factory=lambda: {
            "forex": 0.75,
            "crypto": 1.00,
            "stock": 0.90,
            "index": 0.90,
            "metal": 1.00,
            "commodity": 1.00,
            "unknown": 0.75,
        }
    )
    #: A pullback that is still moving materially against the proposal has not
    #: become a retest yet. It is reconsidered after the next closed M5 bar.
    max_last_bar_adverse_atr: float = Field(default=0.20, ge=0.0, le=2.0)
    #: An impulse is still directional evidence after this many bars, but it is
    #: no longer entry timing. It must be followed by a newly closed resumption
    #: bar rather than bought or sold solely because an older impulse existed.
    stale_impulse_after_bars: int = Field(default=1, ge=0, le=20)
    stale_impulse_min_resumption_atr: float = Field(default=0.05, ge=0.0, le=2.0)
    #: The AI reviewed one exact price shape. More drift than this invalidates
    #: that review; it never becomes permission to chase the new price.
    #:
    #: This is the CHASING half: price ran on in the proposed direction, so the
    #: fill is worse than the one that was approved and the reward-to-risk with
    #: it. Staying tight here is the whole point of the gate.
    max_review_price_drift_atr: float = Field(default=0.25, gt=0.0, le=2.0)
    #: The other half, which used to share the number above because the gate
    #: took an absolute value. Price moving AGAINST the direction during the
    #: review does not make the fill worse — it makes it better, and tightens
    #: the stop distance the order will carry.
    #:
    #: What it can mean is that the level is failing, and that question already
    #: has two owners: `entry_timing_max_adverse_atr` and
    #: `confirmation_max_adverse_atr`, both deliberately set to 1.00 ATR on
    #: measured evidence after half an ATR was found to be ordinary breathing
    #: on a fast chart. A third, stricter opinion standing in front of both of
    #: them answered a question it was not built for. Live: two of the three
    #: setups that survived every other gate in a two-hour window died here.
    max_review_pullback_drift_atr: float = Field(default=1.00, gt=0.0, le=3.0)
    max_review_latency_seconds: float = Field(default=45.0, gt=1.0, le=300.0)
    #: A breakout is expected to sit farther from its short EMA than a swing
    #: entry. This multiplier only relaxes the three chase thresholds for a
    #: `quick` idea; the adverse-last-bar check and post-review drift binding
    #: remain unchanged.
    quick_extension_multiplier: float = Field(default=1.25, ge=1.0, le=3.0)

    #: Preserve a valid directional setup across cycles when only its entry is
    #: late. The setup waits for a real pullback and then a newly closed bar
    #: resuming in its direction; it is not silently discarded and rediscovered.
    lifecycle_enabled: bool = True
    #: THE FAMILIES THAT HAVE A LEVEL TO COME BACK TO.
    #:
    #: A breakout retests the resistance it just cleared. A trend-continuation
    #: reader has no level -- only a move -- so for those the
    #: pullback-from-extreme below is the right measure and is unchanged.
    #:
    #: For two months there was one rule for both: wait for a pullback of
    #: `lifecycle_pullback_atr` from the running high. On a break that ran
    #: three ATR that is an entry two and a half ATR above the level anyone
    #: would call the retest -- the textbook trade enters AT the broken
    #: resistance with the stop just under it, and this entered near the top
    #: with the stop an ATR and a half below. Different trades, and the
    #: scorecard says which one it was:
    #:
    #:     bought at 80-95% of its own range   14 trades   -1.65R
    #:     bought at 95-100%                    8 trades   -0.99R
    #:     bought at 60-80%                    21 trades   +0.66R
    #:
    #: The only positive row is the one that is NOT extended. It bought
    #: extended, and buying extended is what a retest without a level produces.
    #:
    #: Matched on substring: `setup_family` is `{module}_{timeframe}`.
    lifecycle_level_retest_modules: tuple[str, ...] = (
        "market_structure",
        "impulse_break",
        "session_breakout",
        "m1_micro_breakout",
        "volatility_squeeze",
        "range_break",
    )
    #: How close to the broken level counts as having retested it, in ATR.
    #:
    #: 0.35 -> 0.15 ON MEASUREMENT, and this is the correction to the value
    #: shipped hours earlier on reasoning alone.
    #:
    #: Ten years of M15 on eight instruments, the same break entered at
    #: different distances from the level it cleared:
    #:
    #:     within 0.15 ATR   +11.3 points over chance   +54 sigma
    #:     within 0.35 ATR    +5.0 points               +20 sigma
    #:     within 0.60 ATR    -3.4 points               NEGATIVE
    #:
    #: Monotone across three channel lengths and 27 configurations, and it
    #: changes sign. 0.35 was picked to sit inside the 0.40 the old pullback
    #: rule asked for -- a defensible argument that happens to give away more
    #: than half the edge. The distance is not a tolerance around the strategy,
    #: it IS the strategy.
    #:
    #: Not zero: a retest demanding the exact tick never fills, and the level
    #: is an estimate.
    lifecycle_retest_level_atr: float = Field(default=0.15, ge=0.0, le=3.0)
    lifecycle_pullback_atr: dict[str, float] = Field(
        default_factory=lambda: {"quick": 0.20, "intraday": 0.30, "swing": 0.40}
    )
    lifecycle_resumption_atr: dict[str, float] = Field(
        default_factory=lambda: {"quick": 0.05, "intraday": 0.08, "swing": 0.10}
    )
    lifecycle_expiry_minutes: dict[str, float] = Field(
        default_factory=lambda: {"quick": 30.0, "intraday": 240.0, "swing": 1440.0}
    )

    @field_validator("timeframe")
    @classmethod
    def _entry_timeframe_is_supported(cls, value: str) -> str:
        Timeframe.parse(value)
        return value.upper()

    @field_validator(
        "max_favourable_extension_atr",
        "max_single_bar_body_atr",
        "max_extreme_single_bar_body_atr",
        "max_ema_distance_atr",
        "max_live_favourable_gap_atr",
    )
    @classmethod
    def _asset_limits_are_complete(cls, value: dict[str, float]) -> dict[str, float]:
        expected = {"forex", "crypto", "stock", "index", "metal", "commodity", "unknown"}
        missing = expected - set(value)
        extra = set(value) - expected
        if missing or extra:
            raise ValueError(
                f"entry-quality asset limits require exactly {sorted(expected)}; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if any(limit <= 0.0 or limit > 10.0 for limit in value.values()):
            raise ValueError("entry-quality ATR limits must be above zero and at most 10")
        return value

    @field_validator(
        "lifecycle_pullback_atr", "lifecycle_resumption_atr", "lifecycle_expiry_minutes"
    )
    @classmethod
    def _lifecycle_horizons_are_complete(cls, value: dict[str, float]) -> dict[str, float]:
        expected = {"quick", "intraday", "swing"}
        if set(value) != expected:
            raise ValueError(f"setup lifecycle requires exactly {sorted(expected)}")
        if any(limit <= 0.0 for limit in value.values()):
            raise ValueError("setup lifecycle limits must be above zero")
        return value


class AssetClassRoutingConfig(Base):
    """Bounded ranking preferences for one market microstructure.

    These numbers never create a setup. They decide which already executable
    setup is considered first when position slots or AI calls are scarce.
    """

    module_affinity: dict[str, float] = Field(default_factory=dict)
    trend_alignment_bonus: float = Field(default=4.0, ge=0.0, le=10.0)
    range_reversal_bonus: float = Field(default=3.0, ge=0.0, le=10.0)
    transition_penalty: float = Field(default=1.0, ge=0.0, le=10.0)
    extreme_penalty: float = Field(default=6.0, ge=0.0, le=15.0)
    countertrend_penalty: float = Field(default=3.0, ge=0.0, le=10.0)
    cross_market_bonus: float = Field(default=3.0, ge=0.0, le=10.0)
    cross_market_penalty: float = Field(default=2.0, ge=0.0, le=10.0)
    cross_market_majority: float = Field(default=0.62, ge=0.5, le=0.9)

    @field_validator("module_affinity")
    @classmethod
    def _affinities_are_bounded(cls, value: dict[str, float]) -> dict[str, float]:
        if any(modifier < -10.0 or modifier > 10.0 for modifier in value.values()):
            raise ValueError("asset-class module affinities must be between -10 and 10")
        return value


class ConfluenceConfig(Base):
    """Decision policy shared by paper, backtest and live execution."""

    #: Reuse module analysis until a new bar closes. Every module reads only
    #: closed bars — no tick, no wall clock — so identical frames give
    #: identical signals and this returns the same answer, not an approximation
    #: of it. The escape hatch exists because a future module that broke that
    #: purity would be very hard to spot from the outside.
    cache_signals_per_bar: bool = True
    score_threshold: float = Field(default=55.0, ge=1.0, le=100.0)
    #: What each agreeing module beyond the first adds to the score, as a
    #: fraction. The whole premium is capped by `max_corroboration_bonus`.
    #:
    #: WHY THIS EXISTS. The score is a weighted MEAN over the agreeing modules,
    #: so a second reader that AGREES could only pull the average toward its
    #: own value. Against a threshold of 45:
    #:
    #:     market_structure alone                        70.0
    #:     market_structure + candle_momentum agreeing    51.9
    #:
    #: Two readers pointing the same way scored eighteen points LOWER than one
    #: reader alone, in an engine named for confluence.
    #:
    #: The selection effect is the damage. At a fixed bar, a score that falls
    #: as agreement rises does not just undervalue corroborated setups — it
    #: systematically picks the ones where exactly one detector is loud and the
    #: rest are silent. All eight detectors then measured 54-57% with an
    #: average win of 0.68R against an average loss of 1.04R: one shape, and
    #: barely distinguishable from a coin.
    #:
    #: Zero restores the old arithmetic exactly, so an account that has not set
    #: this keeps its behaviour rather than being changed by an upgrade.
    corroboration_bonus_per_module: float = Field(default=0.0, ge=0.0, le=1.0)
    #: The ceiling on that premium. Bounded so a wide agreement cannot swamp
    #: the quality of the reading itself: five weak modules nodding along must
    #: not outscore one strong, well-corroborated pair.
    max_corroboration_bonus: float = Field(default=0.0, ge=0.0, le=2.0)
    minimum_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    minimum_directional_modules: int = Field(default=2, ge=1, le=10)
    #: What a SINGLE detector must be sure of before it may carry a trade alone.
    #:
    #: With `minimum_directional_modules: 1` a lone module needs nothing but the
    #: score, and for one module the score IS `|raw| x confidence` — the weight
    #: cancels, because numerator and denominator both run over the agreeing
    #: modules. So a detector firing at the bare `minimum_confidence` floor
    #: produces a score it cannot distinguish itself from a genuinely convinced
    #: one, and the threshold cannot tell them apart either.
    #:
    #: HK50 SHORT on 17 August is the case. `impulse_break` alone at confidence
    #: 0.45 — the floor exactly — scored 60 x 0.45 = 27.0 against a 26.0 bar,
    #: took EUR 3.13 of a EUR 182 account, never printed a single positive tick
    #: and returned -0.56R. One unconvinced opinion is not evidence, and raising
    #: the score bar to catch it would have cost every three-module setup in the
    #: same band as well.
    #:
    #: Aimed at the actual defect instead: lone AND unconvinced. A lone detector
    #: that is genuinely sure still trades, which is the half that
    #: `minimum_directional_modules: 2` would have thrown away with it.
    #:
    #: Only consulted when exactly one module agrees. Zero disables it.
    lone_module_minimum_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    #: Per-detector overrides of that floor. EMPTY, so the single value above
    #: still governs everything and nothing changes until a detector earns an
    #: entry here.
    #:
    #: One floor for eight detectors was a blunt instrument. Measured offline
    #: over 90 days on five symbols, dropping the global floor from 0.65 to
    #: 0.55 more than doubles the setups formed — 71 to 153 over identical
    #: bars. But of the roughly 1,174 refusals it releases, 473 are
    #: `fast_ema_cross` and 414 are `liquidity_sweep`, and in the module
    #: backtest those two hold the worst and the best records respectively:
    #: -1.211R a trade where fast_ema_cross was present, +0.119R where
    #: liquidity_sweep was. A single number cannot let one through and hold the
    #: other back, so it was letting both or neither.
    #:
    #: EARNED MEANS A NUMBER. An entry here is a claim that a specific detector
    #: is trustworthy on its own, and the only thing that can support it is
    #: `scripts/backtest_modules.py` with enough trades to matter —
    #: liquidity_sweep's own t is 0.59 on 21 lone trades, which is noise, and it
    #: needs around 268 to establish itself. Test a candidate value offline
    #: first:
    #:
    #:     python scripts/why_no_setups.py --days 90 \
    #:         --set lone_module_minimum_confidence_by_module=liquidity_sweep:0.55
    lone_module_minimum_confidence_by_module: dict[str, float] = Field(default_factory=dict)
    minimum_agreement_ratio: float = Field(default=0.60, ge=0.5, le=1.0)
    #: Refuse a target this market does not actually reach often enough for the
    #: plan's own reward-to-risk to break even.
    #:
    #: The measurement already existed and was computed only to be printed in
    #: the review payload, where the reviewer read it and refused the trade.
    #: Six consecutive live refusals cited exactly this number: UK100 at 30.2%,
    #: CADCHF at 30.1%, AUDUSD at "37.0% up against 37.5% down, essentially a
    #: coin flip". Five cents a time to be told something the engine had in
    #: hand before it asked.
    #:
    #: Arithmetic rather than opinion: reach rate is an upper bound on win
    #: rate, because a trade cannot win without the market travelling to its
    #: target. Below the break-even hit rate the plan cannot work even before
    #: the stop, the spread and the commission are counted.
    #:
    #: Named `base_rate` and not `reachable`, because `filters.runway` already
    #: has a `require_reachable_target` that asks a completely different
    #: question — whether there is TIME before the wind-down. Two settings with
    #: one name in two sections is an operator turning off the wrong one at
    #: two in the morning.
    #: Refuse a trend-continuation setup while the regime classifier measures
    #: a range.
    #:
    #: `market_regime` sorts every market into trend_up, trend_down, range,
    #: transition or extreme. It is computed, sent to the reviewer, cited by
    #: the reviewer in refusal after refusal, and never read by the engine —
    #: which checks only `volatility_regime`, for "extreme". Three live
    #: refusals in one session, in the reviewer's own words:
    #:
    #:   "the regime module explicitly flags 'range' with low efficiency
    #:    (0.08 H1, 0.11 H4) — this is chop, not a trend"
    #:   "Market_regime independently flags this as a range, not a trend,
    #:    which undermines the trend-continuation premise the whole idea is
    #:    built on"
    #:
    #: NOT a module count, which the reviewer is explicitly told to ignore and
    #: correctly does. A zero from a module looking for something else is the
    #: absence of evidence; "range" is a measurement that contradicts the
    #: premise. Only continuation setups are caught — a liquidity sweep is a
    #: range setup and belongs in a range.
    refuse_trend_continuation_in_range: bool = True
    #: Which modules assert that a trend is continuing, and therefore have
    #: their premise contradicted by a measured range. Named rather than
    #: inferred, so a new module has to be classified deliberately.
    #:
    #: `drift_continuation` and `fast_ema_cross` were in this list for one day
    #: and it cost 6,726 refusals in 24 hours — 3,813 of them the drift module,
    #: which is the only module on this account that can find a short in a
    #: falling market. Over the same period the engine proposed a long 10,453
    #: times and refused each one for the reason that price was moving against
    #: it, while the module that would have said "sell" was being vetoed here.
    #:
    #: The classification was wrong. `trend_momentum` INFERS a trend from 20/50
    #: EMA alignment on H4 and H1 — an inference, and one a measured range
    #: genuinely contradicts. `drift_continuation` MEASURES the move itself over
    #: its own eight M15 bars and requires 65% of them to agree;
    #: `fast_ema_cross` measures a separation in ATR on M5. A range on H1 or H4
    #: contradicts neither: it is a different question about a longer window,
    #: and an hour of clean one-way travel inside a daily range is the ordinary
    #: shape of an intraday move rather than a paradox.
    #:
    #: What still keeps them out of chop is their own evidence — the drift
    #: module's consistency floor, the cross module's separation floor — and
    #: `entry_quality` downstream.
    #: `drift_continuation` was tried on this list again on 18 August and taken
    #: straight back off. Two losing trades had it firing in a transition, which
    #: is not evidence: I had no count of how many WINNERS were taken in the
    #: same regimes, so "six of six losers were in range or transition" says
    #: nothing until the denominator is known. Against that stood a measured
    #: 3,813 refusals in one day and the note above that this is the only module
    #: on the account that can find a short in a falling market — which is the
    #: thing the owner has asked for most.
    trend_continuation_modules: tuple[str, ...] = ("trend_momentum",)
    #: Regimes in which a continuation claim is contradicted rather than merely
    #: unsupported. Both are measurements that say "there is no trend here";
    #: `trend_up` and `trend_down` obviously support one, and `extreme` is
    #: already refused outright at the top of `evaluate`.
    continuation_contradicting_regimes: tuple[str, ...] = ("range", "transition")
    #: Regimes this account refuses outright, whatever the modules say.
    #:
    #: Measured over 90 closed trades and four days, by what `market_regime`
    #: read at entry:
    #:
    #:     transition   44 trades  59% won  -20.86 EUR   -0.47 a trade
    #:     trend_down   11 trades  55% won  -11.75 EUR   -1.07 a trade
    #:     trend_up     21 trades  90% won  +22.44 EUR   +1.07 a trade
    #:     range         6 trades 100% won   +9.57 EUR   +1.59 a trade
    #:     extreme       8 trades 100% won   +9.25 EUR   +1.16 a trade
    #:
    #: `transition` is half the book and all of the damage. It is the
    #: classifier's leftover branch and what it MEANS is that the two
    #: timeframes disagree about direction, so half these trades were taken
    #: into a market the system could not read. At 22 trades it was +2.41 EUR
    #: and was deliberately left alone; at 44 it is -20.86.
    #:
    #: `trend_down` is NOT on this list, and the omission is deliberate. It
    #: loses money over these four days, but it is a real trend and not chop:
    #: refusing it refuses every short in a falling market, which is the one
    #: thing this account has asked for most. Eleven trades is also far too
    #: thin to condemn a direction — `transition` looked profitable at 22.
    #: The scorecard now splits regime by direction, so when the sample is big
    #: enough this can be narrowed to whichever side actually loses instead of
    #: closed wholesale.
    #:
    #: A hard refusal rather than a discount, because the objection is to the
    #: market and not to any one module reading it. Empty trades every regime.
    refused_regimes: tuple[str, ...] = ()
    #: Modules whose evidence lives on a fast chart. When nothing but these
    #: fired, the plan is an intraday one.
    #:
    #: This was hardcoded to `liquidity_sweep` alone, and adding a module
    #: without adding it here is a silent and expensive mistake:
    #: `drift_continuation` measures eight M15 bars and was handed a swing
    #: plan — H1 planning authority and a target twenty-four hours out — for a
    #: signal whose whole mechanism expires in about two hours.
    #:
    #: AND IT HAPPENED AGAIN, to `candle_momentum`, which is the module this
    #: warning was written for and the one it was not applied to. Its bias is
    #: M15, its confirmation is M5 and its trigger is a single closed M1
    #: candle. Nothing in it reads a chart slower than M15. It was in neither
    #: list, so `_classify_horizon` fell through to `return "swing"` and gave a
    #: one-minute trigger H1 planning authority and a target twenty-four hours
    #: out — the exact sentence above, one module later.
    #:
    #: That is not only a wrong plan, it is the throughput. A swing proposal
    #: holds its position slot for a full day (`time_exit_hours: 24`), and four
    #: of the five modules live on this account classified as swing, so nearly
    #: every trade parked a slot for an etmaal. An intraday plan expires in
    #: three hours and frees the slot eight times sooner.
    intraday_modules: tuple[str, ...] = (
        "liquidity_sweep",
        "drift_continuation",
        "fast_ema_cross",
        "impulse_break",
        "ema_pullback_resume",
        "m1_micro_breakout",
        "candle_momentum",
        # AND A THIRD TIME, to `basket_divergence`, where it is worst of the
        # three. That module reads ONE timeframe, M1, and its own docstring
        # states the mechanism: "the trade pays when the GAP closes". A gap
        # between two indices measured over M1 closes in minutes. It was given
        # H1 planning authority and a target twenty-four hours out, by which
        # time the gap it was entered on had long since closed and what was
        # left was an unhedged directional index position with no thesis.
        "basket_divergence",
        # AND A FOURTH AND FIFTH TIME, to the only two modules on this account
        # that trade real money.
        #
        # `impulse_retest` reads M15 and `order_block` reads M30. Neither was in
        # either list, so both fell through to `return "swing"` and were handed
        # H1 planning authority, a 24-bar target horizon, and the swing HTF
        # veto — which is D1/W1 at `minimum_htf_conflicts: 1`, against
        # intraday's H4/D1 at 2.
        #
        # That last number is not a detail for these two. A break-retest buys
        # the level BECAUSE price came back to it; a daily trend pointing the
        # other way is the normal condition of the setup, not evidence against
        # it. A single D1 conflict vetoing it refuses the strategy on the
        # grounds of its own mechanism, and the 30 August dry run refused 99.2%
        # of everything it saw.
        #
        # Both were measured on a fixed exit within a handful of bars of the
        # entry. A 24-bar H1 horizon is a day and a half.
        "impulse_retest",
        "impulse_retest_m30",
        "order_block",
        # AND A SIXTH TIME, for the M1 copy of section three. `_classify_horizon`
        # matches this list EXACTLY -- unlike `entry_timing_exempt_families`,
        # which is a substring test and would have covered it by accident. A
        # new instance therefore has to be named here or it falls through to
        # "swing" and gets H1 planning and a D1/W1 veto on an M1 setup, which
        # is the exact defect the two entries above were added to fix.
        "order_block_fast",
        "order_block_m15",
        "order_block_h1",
        "walkforward_index",
    )
    #: Complete M5/M1 theses. These receive a genuinely quick planning horizon
    #: instead of being stretched into the three-hour intraday profile.
    quick_modules: tuple[str, ...] = (
        "fast_ema_cross",
        "ema_pullback_resume",
        "m1_micro_breakout",
    )
    #: These quick modules already prove immediate direction inside their own
    #: closed-bar definition. Repeating the generic M5 adverse-drift gate would
    #: ask the same question twice. Fresh-price and chase checks still run.
    quick_embedded_confirmation_modules: tuple[str, ...] = (
        "fast_ema_cross",
        "ema_pullback_resume",
        "m1_micro_breakout",
    )
    #: Unconditional travel frequency is not the win rate conditional on a
    #: specific M1/M5 event. For quick plans it is evidence sent to Claude;
    #: swing and intraday plans keep the hard arithmetic gate.
    quick_statistical_gates_are_advisory: bool = True
    #: Even a quick event cannot rescue a decorative target. The unconditional
    #: reach rate may be treated as context only when it is at least this share
    #: of the plan's arithmetic break-even requirement.
    quick_reach_hard_floor_ratio: float = Field(default=0.50, ge=0.0, le=1.0)
    #: A small unconditional disadvantage can be timing noise for an M1/M5
    #: event; a very large one remains a hard contradiction.
    quick_direction_disadvantage_hard_gap_pct: float = Field(default=20.0, ge=0.0, le=100.0)
    #: A new closed quick event may pass broad symbol/direction veto patterns.
    #: Exact proposal memory remains active: materially unchanged entry/stop
    #: geometry must never buy the same Claude answer twice.
    quick_events_bypass_broad_veto_memory: bool = True
    require_target_base_rate: bool = True
    #: Judge the plan against the stop it will actually carry instead of an
    #: unconditional reach rate that ignores the stop entirely. It matters
    #: because `_widen_stop_for_costs` changes the plan after the engine
    #: approved it: on a small account the stop is pushed out on nearly every
    #: candidate, which lowers reward-to-risk and, at the very same time, makes
    #: the position much harder to stop out. Only the first of those two ever
    #: reached the gate, so widening — the thing that exists to make
    #: small-account trades viable at all — could only count against a trade.
    first_touch_target_test: bool = True
    #: Fewest windows that must have RESOLVED — reached the target or taken the
    #: stop — before a distance may be chosen on its measured rate.
    #:
    #: The break-even test is applied to the resolved windows, which is the
    #: population it was derived for, and that population shrinks fast as the
    #: distance grows. On a 3,000-bar series a 3R target resolves 36 windows
    #: out of 2,976: a rate read off 36 samples carries an error bar wide
    #: enough to clear any requirement by luck, and the trade it admits is one
    #: whose target is touched half a percent of the time.
    #:
    #: 30, and deliberately low, because it is a backstop and not the gate.
    #: Capping the credit on an expired window at zero already refuses these
    #: distances on their own arithmetic: on a strongly trending series the
    #: expectancy crosses zero at 2.4R with 227 windows still resolving, and is
    #: -0.15R by the time only 36 do. This catches the freak reading the
    #: arithmetic happens to like, nothing more. A higher figure starts
    #: refusing markets for having a short history rather than a bad edge.
    #: Zero switches the sample floor off entirely.
    target_minimum_resolved_windows: int = Field(default=30, ge=0, le=100_000)
    #: How far above `risk.min_risk_reward` the target search must start.
    #:
    #: THE TWO FLOORS BEING EQUAL IS A COIN FLIP, AND IT RAN LIVE. The search
    #: lands on its own floor on essentially every market, so with both numbers
    #: at 0.60 every plan arrives at the sizer at exactly 0.60R — and the quote
    #: moves between the two. NDX100 SHORT, 19 August: approved, reviewed, then
    #: `RR_BELOW_MINIMUM: reward:risk is 1:0.52, below the required 1:0.60`.
    #: Nothing was wrong with the setup. It was planned with no room.
    #:
    #: 0.15 means a plan can lose a seventh of its reward to a moved quote and
    #: still be sized. Zero restores the old knife edge and is allowed only so
    #: the behaviour stays expressible.
    target_planning_margin: float = Field(default=0.15, ge=0.0, le=2.0)
    #: Let the analysis plan on the cost-implied stop instead of only the
    #: chart's. OFF, and turned off by its own evidence rather than by taste.
    #:
    #: The argument for it was that 1,710 refusals in two live hours read "no
    #: target between 1.00R and 3.00R pays on this market" and were measured
    #: against a stop the runner was about to widen. One hour with it on, and
    #: normalised per hour against the two hours before:
    #:
    #:     no target pays     855/h -> 881/h    +3%, did not move
    #:     setups formed       52/h ->  37/h   -28%
    #:     runway failures      6%  ->  16%   of the setups that did form
    #:
    #: The number it was built to reduce did not move, and the wider stop put
    #: the target farther away, which the runway gate then refused. Two of
    #: those three could be the hour of day; the flat one cannot be, and it is
    #: the one the whole argument rested on.
    #:
    #: The code and its tests stay. Turning it back on means measuring it
    #: against a clean baseline on the same session, not another guess.
    plan_on_cost_floor: bool = False
    #: Percentage points demanded ABOVE break-even. Small on purpose: reach
    #: counts up and down moves independently over the same windows, so it is
    #: not the probability of hitting the target before the stop and must not
    #: be read as one. It is a floor for what cannot work, not a forecast.
    target_reach_margin_pct: float = Field(default=3.0, ge=0.0, le=40.0)
    #: Refuse when this instrument covers the target distance MORE often in the
    #: opposite direction. Live: AUDSGD proposed LONG at 38.1% up against 46.8%
    #: down over the same horizon. The direction came from an EMA and the
    #: target from a multiplier, and nothing ever compared the two.
    require_direction_advantage: bool = True
    #: Percentage points the other side must be better by before that counts as
    #: a disadvantage. Zero — a bare `forward >= opposite` — refused ASX200 at
    #: 47.4% against 49.0% and EURAUD at 35.3% against 35.8%, gaps of 0.63 and
    #: 0.21 standard errors on the measurement itself. 127 refusals an hour on
    #: live data, all of them reading an error bar and calling its sign
    #: evidence. The measured standard error is used when it is larger than
    #: this, so a thin sample cannot slip past on a fixed number.
    direction_advantage_tolerance_pct: float = Field(default=5.0, ge=0.0, le=40.0)
    #: Ceiling on how far the target-reach window stretches when the fee
    #: schedule floors the stop wider than the chart asked for. A wider stop
    #: makes 1R a longer distance, and distance costs TIME as its square — the
    #: same law the runway check uses. Left unbounded the square law runs away
    #: and the window stops describing a trade anyone would sit in.
    max_cost_horizon_stretch: float = Field(default=4.0, ge=1.0, le=25.0)
    #: Size the measurement window to the distance it has to watch, instead of
    #: giving every plan the profile's flat number of bars.
    #:
    #: Covering a distance d takes (d / speed)^2 bars, so a target at 0.75R of
    #: an eight-ATR stop needs 36 bars and got 24. Past six ATR of stop every
    #: window expires with neither the target nor the stop touched, and an
    #: expired window is charged a round trip, so the expectancy converges on
    #: minus the cost no matter how good the market is. 1,683 refusals an hour
    #: read "no target pays" about plans that were never measured.
    #:
    #: A structural swing stop reaching the last swing low is routinely eight
    #: to sixteen ATR, so this was not an edge case; it was most of them.
    #:
    #: It cannot invent a trade. The measurement still has to come back
    #: positive, and `filters.runway` still refuses a plan that cannot finish
    #: before the session does. All it buys is the chance to be measured.
    fit_horizon_to_the_plan: bool = True
    #: The ceiling on that, as a multiple of the profile's own bars. The square
    #: law grows fast and an unbounded window stops describing a trade anyone
    #: would sit in — at 24 bars of base this allows up to 576, which covers a
    #: stop of about thirty ATR and is already further than the runway check
    #: will let a session-bound plan travel.
    max_plan_horizon_stretch: float = Field(default=24.0, ge=1.0, le=200.0)
    #: Choose the target with the best expectancy PER BAR rather than per trade.
    #:
    #: Maximising per trade systematically prefers the far target, because a
    #: bigger target needs a lower hit rate to pay. With `target_r_multiple` at
    #: 3.0 the search kept choosing 3R, and 3R on an H1 plan is roughly nine
    #: hours of travel under the square law — against eight or nine hours in an
    #: entire session. 56 of 140 live setups in one six-hour window died on
    #: INSUFFICIENT_RUNWAY: not refused on their merits, refused because the
    #: target the search chose could not be reached before the day ended.
    #:
    #: A session has a deadline, so the quantity being maximised has to be
    #: expectancy per unit of TIME. 1R reached inside an hour beats 3R needing
    #: nine hours that do not exist, and the money is in the account either way.
    #:
    #: Still a search over distances with positive expectancy — this changes
    #: which of the payable distances is chosen, never whether an unpayable one
    #: can be.
    prefer_sooner_targets: bool = True
    target_r_multiple: float = Field(default=2.0, ge=1.0, le=10.0)
    #: PER-FAMILY TARGET, because one number cannot be right for two different
    #: trades. `impulse_retest` was measured at every ratio from 0.75 to 3.0
    #: and its optimum is 1:1 -- 68% hit, +0.28R net -- while at 3:1 the same
    #: setups net +0.016R. Sending it out under the account's 3.0 would take a
    #: measured +0.28R strategy and trade it at nearly zero.
    #:
    #: The reachability cap still applies on top: the engine takes the smaller
    #: of this and what the instrument actually travels in the horizon, so this
    #: is a ceiling on ambition, not a promise.
    target_r_multiple_by_family: dict[str, float] = Field(default_factory=dict)
    #: The target is also bounded by how far this instrument actually travels.
    #: `entry + 2R` is arithmetic and never asks whether the market goes there;
    #: a slow instrument gets a target it reaches once a month, and the trade
    #: becomes a bet on the stop not being hit.
    target_horizon_bars: int = Field(default=24, ge=4, le=200)
    #: Percentile of favourable excursion over that horizon. Not the maximum —
    #: one violent week should not set the expectation for every trade.
    target_reach_quantile: float = Field(default=0.70, ge=0.5, le=0.99)
    #: The shortest target the search may CONSIDER. Not a quality gate: the
    #: search only ever picks a distance with positive measured first-touch
    #: expectancy, so lowering this adds candidates and cannot make the chosen
    #: target worse. What it does is stop the engine refusing a market outright
    #: because nothing between 1.00R and 3.00R paid — the single largest
    #: refusal in the system, 2,138 of them over 90 days on five symbols, ahead
    #: of every lone-module refusal combined.
    #:
    #: 0.60, down from 1.00, so a quick trade with a high hit rate is allowed
    #: to exist. At 0.60R with a 10% round trip the break-even hit rate is 69%,
    #: which is demanding and reachable; at 1.00R the engine was not permitted
    #: to ask the question at all.
    #:
    #: TWO THINGS HAD TO BE TRUE FIRST and neither was.
    #:
    #: `risk.min_risk_reward` refuses anything below it one layer later, so
    #: lowering this alone changes nothing and the two drifting apart is a
    #: documented past failure — confluence trimmed to 0.9R while the sizer
    #: refused everything under 1.1R. A validator on RiskConfig now refuses that
    #: combination outright rather than leaving it to be rediscovered.
    #:
    #: And the expectancy search counted only the spread, treating commission as
    #: the sizer's problem. That is defensible at 1.00R and wrong below it: the
    #: reward shrinks while the commission does not, so the estimate was most
    #: optimistic exactly where this floor now allows the search to go. The
    #: runner supplies the full round trip through `round_trip_cost_price`.
    minimum_r_multiple: float = Field(default=1.15, ge=0.3, le=5.0)
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
    #: 1.0, up from 0.50, and mostly to stop this system holding two different
    #: opinions about one question. `confirmation_max_adverse_atr` asks exactly
    #: the same thing one stage later — has price already started proving this
    #: trade wrong — and was moved to 1.0 this morning on measured evidence
    #: (AWAITING_CONFIRMATION blocked 11 setups, 6 of them winners, cost 5.20R).
    #: Leaving its twin at half that meant the looser gate never got a say.
    #:
    #: The cost of the tighter number, one live hour: 195 refusals on M1 against
    #: a long, 175 on M5 against a long, 131 and 112 the same for shorts. 613 an
    #: hour, second only to "no module fired at all", and half an ATR over six
    #: bars is ordinary breathing on a fast chart.
    #:
    #: What it still catches is what it was built for: the GBPJPY short sent
    #: into a resistance break was 1.2 ATR of adverse travel and is still
    #: refused, on both gates.
    entry_timing_max_adverse_atr: float = Field(default=1.00, gt=0.0, le=5.0)

    #: Setup families the timing gate above may not judge, because for them the
    #: adverse move IS the setup.
    #:
    #: A break-retest buys the level precisely because price came back to it.
    #: `_entry_timing_conflict` refuses entries the immediate price action is
    #: moving against, so for these families it refuses every valid entry there
    #: is -- and it did. Seven days on the live feed, eleven FX majors:
    #: `impulse_retest` formed 52 signals, 0.76 per pair per day against the
    #: 0.62 measured over eleven years of research. The account took NONE of
    #: them.
    #:
    #: The arithmetic is not marginal. The gate allows 1.0 ATR of adverse
    #: travel over six closed bars, measured in the entry timeframe's own ATR
    #: (M5). A retest's pullback is one M15 ATR by construction and an M15 ATR
    #: is about 1.7 M5 ATRs, so every qualifying retest is over the limit
    #: before it is looked at, and a MORE decisive break is further over.
    #:
    #: Raising the limit is the wrong repair: it would loosen the gate for the
    #: trend entries it was written for, where eleven of the first twelve paid
    #: reviews vetoed exactly this and were right. The distinction is the setup,
    #: not the threshold.
    #:
    #: Matched on substring against `setup_family`, the same way
    #: `target_r_multiple_by_family` is.
    entry_timing_exempt_families: tuple[str, ...] = ()
    #: Standalone strategies whose closed-bar trigger, stop and exit were
    #: measured as one indivisible plan.  These families do not borrow the
    #: generic HTF/timing policy or have their structural stop padded: doing
    #: either changes the strategy that was tested.  Account, cost, exposure,
    #: news and broker safety checks still run in the execution service.
    #: Matched on substring because setup families carry their clock suffix.
    strategy_owned_entry_families: tuple[str, ...] = ()
    #: Strategies whose target probability was measured directly on their own
    #: entries. The generic rolling reach statistic remains visible to the
    #: reviewer for these families, but may not veto the setup before review.
    #: Matched on substring because setup families carry their clock suffix.
    target_reach_advisory_families: tuple[str, ...] = ()

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

    #: Horizon-specific authority. The legacy fields above remain the default
    #: swing policy and keep old overlays/backtests reproducible. Live routing
    #: uses these named profiles once present.
    #: THE LADDER, WITH NO RUNGS MISSING.
    #:
    #: Each profile plans on one timeframe, refuses a trade taken into a strong
    #: trend on the ones ABOVE it, and checks the immediate move on the ones
    #: BELOW. The three ladders used to skip rungs, and the skipped ones were
    #: not obscure:
    #:
    #:     swing     planned H1,  vetoed on D1/W1     -- H4 never consulted
    #:     intraday  planned M15, vetoed on H4/D1     -- H1 never consulted
    #:     quick     planned M5,  vetoed on H1/H4     -- M15 never consulted
    #:
    #: A swing trade could therefore be taken straight into an H4 downtrend
    #: without anything objecting, because the check jumped from H1 to D1.
    #:
    #: The gaps are closed and `minimum_htf_conflicts` scales with the rung
    #: count, so this is BROADER rather than STRICTER: two conflicts out of
    #: three is the same share of the ladder as two out of two was not -- it is
    #: looser per timeframe and wider in what it looks at, which is the point.
    #: More places to notice a conflict, no single one able to veto alone
    #: except on swing, where the ladder is longest and the trade slowest.
    horizon_profiles: dict[str, HorizonProfileConfig] = Field(
        default_factory=lambda: {
            "swing": HorizonProfileConfig(
                htf_trend_timeframes=("H4", "D1", "W1"),
                minimum_htf_conflicts=2,
                entry_timing_timeframes=("M15", "M5", "M1"),
            ),
            "intraday": HorizonProfileConfig(
                planning_timeframe="M15",
                target_horizon_bars=12,
                htf_trend_timeframes=("H1", "H4", "D1"),
                minimum_htf_conflicts=2,
                htf_trend_veto=1.0,
                entry_timing_timeframes=("M5", "M1"),
            ),
            "quick": HorizonProfileConfig(
                planning_timeframe="M5",
                target_horizon_bars=6,
                htf_trend_timeframes=("M15", "H1", "H4"),
                minimum_htf_conflicts=2,
                htf_trend_veto=1.0,
                entry_timing_timeframes=("M1",),
            ),
        }
    )

    @field_validator("horizon_profiles")
    @classmethod
    def _required_horizons_exist(
        cls, value: dict[str, HorizonProfileConfig]
    ) -> dict[str, HorizonProfileConfig]:
        missing = {"swing", "intraday", "quick"} - set(value)
        if missing:
            raise ValueError(f"analysis.confluence.horizon_profiles missing {sorted(missing)}")
        return value

    #: Refuse to enter while price is actively running the other way.
    #:
    #: The engine finds a level, forms a view, and the order goes out on the
    #: same tick — so a short can be sent into a market that is climbing
    #: through the very level it is short against. A live GBPJPY short was
    #: exactly that: the operator watched resistance break upward and the
    #: system sold into it, because nothing between "this is a good level" and
    #: "send the order" ever looked at which way price was going right then.
    #:
    #: This is the cheapest form of waiting for confirmation, and the only one
    #: that needs no state: not "has the market proved me right", but "has it
    #: already started proving me wrong". A setup refused here is not gone — it
    #: is re-examined every cycle, and taken the moment the move against it
    #: stops.
    require_entry_confirmation: bool = True
    #: Timeframe the immediate move is read on. Fast enough to be about *now*
    #: rather than about the setup, which the analysis has already judged.
    confirmation_timeframe: str = "M5"
    #: Closed bars of it to measure across.
    confirmation_bars: int = Field(default=3, ge=1, le=50)
    #: How far price may travel against the entry over those bars before the
    #: trade waits, in ATR of that timeframe.
    #:
    #: Measured in ATR so it means the same on gold as on EURGBP.
    #:
    #: 1.0, up from 0.5, and this is the one loosening on the board with a
    #: price tag already attached to it. `scorecard.py` measures what each gate
    #: blocked and what those setups went on to do: AWAITING_CONFIRMATION
    #: blocked 11 setups, 6 of them winners, and cost 5.20R — the most
    #: expensive gate in the table by a factor of nearly two.
    #:
    #: Half a bar was too tight. On this account the ordinary wobble around an
    #: entry is most of half an ATR, so the gate was firing on noise and calling
    #: it a failing level. At 1.0 a full adverse bar still waves through, which
    #: is exactly the situation this exists to catch — the GBPJPY short sent
    #: into a resistance break was around 1.4 ATR — so the protection that
    #: motivated it survives while the false positives do not.
    #:
    #: Re-read the same scorecard row in a few days. If the blocked-and-won
    #: count stops climbing, this was right; if the losers climb instead, 0.75.
    confirmation_max_adverse_atr: float = Field(default=1.0, gt=0.0, le=3.0)

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "market_structure": 1.0,
            "trend_momentum": 1.0,
            "liquidity_sweep": 0.8,
            "level_reaction": 0.7,
            "volatility_regime": 0.0,
            "ema_pullback_resume": 0.55,
            "m1_micro_breakout": 0.55,
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

    @field_validator("lone_module_minimum_confidence_by_module")
    @classmethod
    def _lone_floors_are_bounded(cls, value: dict[str, float]) -> dict[str, float]:
        if any(floor < 0.0 or floor > 1.0 for floor in value.values()):
            raise ValueError("lone-module floors must be between 0 and 1")
        return value

    def lone_floor_for(self, module: str) -> float:
        """The confidence a lone `module` needs, per-detector or global.

        A detector with no entry gets the single global value, which is how
        every detector behaves until one has earned an override — so the empty
        table is exactly the old behaviour and not a special case of it.
        """
        return self.lone_module_minimum_confidence_by_module.get(
            module, self.lone_module_minimum_confidence
        )

    def effective_weights(self, mode: TradingMode) -> dict[str, float]:
        """The weight each module ACTUALLY carries in `mode`.

        `weights` says what a module is worth; `live_enabled_modules` says
        which modules are allowed to vote at all when the money is real. Live,
        a module missing from the allowlist is forced to zero — it is computed,
        it is logged, and it decides nothing.

        The engine has always applied that. The journal never did: it recorded
        the raw `weights` table, so `trend_momentum` (weight 1.0, not on the
        live allowlist) was written to `module_scores` at 1.0 for every live
        cycle. `scorecard.py` credits any module with `weight > 0`, so 60 live
        trades were attributed to a detector that cast no vote in any of them
        — and the detector table is exactly what a decision about which
        detectors to keep gets made from.

        One method, called by both, so the engine's answer and the journal's
        answer cannot drift apart again.
        """
        if not mode.is_live:
            return dict(self.weights)
        allowed = set(self.live_enabled_modules)
        return {
            module: (weight if module in allowed else 0.0)
            for module, weight in self.weights.items()
        }


class WalkforwardIndexConfig(Base):
    """SECTION FOUR: frozen cost-aware H1 index reversal model."""

    enabled: bool = False
    timeframe: str = "H1"
    allowed_symbols: tuple[str, ...] = ("SPX500",)
    threshold: float = Field(default=0.075, gt=0.0, le=3.0)
    stop_atr: float = Field(default=1.0, gt=0.0, le=5.0)
    base_score: float = Field(default=62.0, ge=0.0, le=100.0)
    confidence: float = Field(default=0.72, ge=0.0, le=1.0)


class FailedSessionBreakoutConfig(Base):
    """SECTION SEVEN: cost-aware M5 SPX500 failed session break."""

    enabled: bool = False
    timeframe: str = "M5"
    allowed_symbols: tuple[str, ...] = ("SPX500",)
    range_start_hour: int = Field(default=7, ge=0, le=23)
    range_end_hour: int = Field(default=9, ge=1, le=23)
    trade_window_hours: float = Field(default=3.0, ge=0.5, le=12.0)
    stop_range_share: float = Field(default=0.5, gt=0.0, le=3.0)
    score: float = Field(default=70.0, ge=0.0, le=100.0)
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)


class SectionFiveM5Config(Base):
    """SECTION FIVE: frozen nonlinear M5 NDX100 model."""

    enabled: bool = False
    timeframe: str = "M5"
    allowed_symbols: tuple[str, ...] = ("NDX100",)
    threshold: float = Field(default=0.15, gt=0.0, le=3.0)
    stop_atr: float = Field(default=1.0, gt=0.0, le=5.0)
    score: float = Field(default=68.0, ge=0.0, le=100.0)
    confidence: float = Field(default=0.78, ge=0.0, le=1.0)


class SectionSixModelConfig(Base):
    """One frozen monthly walk-forward model inside section six."""

    enabled: bool = False
    timeframe: str = "M5"
    polarity: int = Field(default=1, ge=-1, le=1)
    threshold: float = Field(default=0.075, gt=0.0, le=3.0)
    stop_atr: float = Field(default=1.0, gt=0.0, le=5.0)
    long_only: bool = False
    session_start_hour_utc: int | None = Field(default=None, ge=0, le=23)
    session_end_hour_utc: int | None = Field(default=None, ge=0, le=23)
    score: float = Field(default=70.0, ge=0.0, le=100.0)
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def complete_session_window(self) -> SectionSixModelConfig:
        if (self.session_start_hour_utc is None) != (self.session_end_hour_utc is None):
            raise ValueError("section six session needs both start and end UTC hours")
        if (
            self.session_start_hour_utc is not None
            and self.session_start_hour_utc == self.session_end_hour_utc
        ):
            raise ValueError("section six session start and end cannot be equal")
        return self


class SectionEightTrendDayConfig(Base):
    """SECTION EIGHT: SPX500 prior-day continuation on H1."""

    enabled: bool = False
    timeframe: str = "H1"
    allowed_symbols: tuple[str, ...] = ("SPX500",)
    lower_close_location: float = Field(default=0.10, ge=0.0, lt=0.5)
    upper_close_location: float = Field(default=0.90, gt=0.5, le=1.0)
    entry_end_hour_utc: int = Field(default=2, ge=1, le=23)
    stop_atr: float = Field(default=1.0, gt=0.0, le=5.0)
    score: float = Field(default=70.0, ge=0.0, le=100.0)
    confidence: float = Field(default=0.76, ge=0.0, le=1.0)


class SectionNineSessionVwapConfig(Base):
    """SECTION NINE: USDJPY UTC-session VWAP reversion on M30."""

    enabled: bool = False
    timeframe: str = "M30"
    allowed_symbols: tuple[str, ...] = ("USDJPY.i",)
    minimum_displacement_atr: float = Field(default=2.0, gt=0.0, le=10.0)
    stop_atr: float = Field(default=1.0, gt=0.0, le=5.0)
    score: float = Field(default=70.0, ge=0.0, le=100.0)
    confidence: float = Field(default=0.76, ge=0.0, le=1.0)


class AnalysisConfig(Base):
    market_structure: MarketStructureConfig = MarketStructureConfig()
    trend_momentum: TrendMomentumConfig = TrendMomentumConfig()
    liquidity_sweep: LiquiditySweepConfig = LiquiditySweepConfig()
    level_reaction: LevelReactionConfig = LevelReactionConfig()
    volatility_regime: VolatilityRegimeConfig = VolatilityRegimeConfig()
    market_regime: MarketRegimeConfig = MarketRegimeConfig()
    drift_continuation: DriftContinuationConfig = DriftContinuationConfig()
    fast_ema_cross: FastEmaCrossConfig = FastEmaCrossConfig()
    impulse_break: ImpulseBreakConfig = ImpulseBreakConfig()
    ema_pullback_resume: EmaPullbackResumeConfig = EmaPullbackResumeConfig()
    m1_micro_breakout: M1MicroBreakoutConfig = M1MicroBreakoutConfig()
    volatility_squeeze: VolatilitySqueezeConfig = VolatilitySqueezeConfig()
    mean_reversion: MeanReversionConfig = MeanReversionConfig()
    session_breakout: SessionBreakoutConfig = SessionBreakoutConfig()
    seasonality: SeasonalityConfig = SeasonalityConfig()
    drift_burst: DriftBurstConfig = DriftBurstConfig()
    impulse_retest: ImpulseRetestConfig = ImpulseRetestConfig()
    #: Additional live clock for the same measured detector. Kept disabled in
    #: the base config because live permission belongs to the account overlay.
    impulse_retest_m30: ImpulseRetestConfig = ImpulseRetestConfig(enabled=False)
    order_block: OrderBlockConfig = OrderBlockConfig()
    #: THE SAME DETECTOR ON A SECOND CLOCK, as its own module.
    #:
    #: A module is one instance reading one timeframe, and every registry that
    #: matters -- `weights`, `live_enabled_modules`, `section_breakers`,
    #: `module_scores` -- is keyed by its name. So a section cannot run on two
    #: clocks by widening a field; it needs a second instance with a second
    #: name, and then the breaker, the weight and the journal all follow it
    #: separately. Which is the point: M30 and M1 are two strategies with two
    #: cost structures and they must be able to be switched off apart.
    #:
    #: Left disabled in the base config. This is the owner's decision about
    #: THIS account at EUR 216, on a 14-day measurement of 105 trades, taken
    #: knowing it is thin -- "risico's moeten genomen worden om te testen".
    order_block_fast: OrderBlockConfig = OrderBlockConfig(enabled=False)
    order_block_m15: OrderBlockConfig = OrderBlockConfig(enabled=False)
    order_block_h1: OrderBlockConfig = OrderBlockConfig(enabled=False)
    level_retest: LevelRetestConfig = LevelRetestConfig()
    vwap_reversion: VwapReversionConfig = VwapReversionConfig()
    basket_divergence: BasketDivergenceConfig = BasketDivergenceConfig()
    candle_momentum: CandleMomentumConfig = CandleMomentumConfig()
    walkforward_index: WalkforwardIndexConfig = WalkforwardIndexConfig()
    failed_session_breakout: FailedSessionBreakoutConfig = FailedSessionBreakoutConfig()
    section_five_m5: SectionFiveM5Config = SectionFiveM5Config()
    section_six_gold_m5: SectionSixModelConfig = SectionSixModelConfig()
    section_six_spx_h1: SectionSixModelConfig = SectionSixModelConfig(timeframe="H1")
    section_eight_trend_day_h1: SectionEightTrendDayConfig = SectionEightTrendDayConfig()
    section_nine_vwap_m30: SectionNineSessionVwapConfig = SectionNineSessionVwapConfig()
    confluence: ConfluenceConfig = ConfluenceConfig()
    entry_quality: EntryQualityConfig = EntryQualityConfig()
    playbooks: PlaybooksConfig = PlaybooksConfig()
    asset_class_routing: dict[str, AssetClassRoutingConfig] = Field(
        default_factory=lambda: {
            "forex": AssetClassRoutingConfig(
                module_affinity={"trend_momentum": 2.0, "liquidity_sweep": 2.0}
            ),
            "stock": AssetClassRoutingConfig(
                module_affinity={"market_structure": 2.0, "trend_momentum": 2.0}
            ),
            "crypto": AssetClassRoutingConfig(
                module_affinity={"trend_momentum": 3.0, "liquidity_sweep": 1.0}
            ),
            "index": AssetClassRoutingConfig(
                module_affinity={"market_structure": 2.0, "trend_momentum": 2.0}
            ),
            "metal": AssetClassRoutingConfig(
                module_affinity={"trend_momentum": 2.0, "liquidity_sweep": 2.0}
            ),
            "commodity": AssetClassRoutingConfig(
                module_affinity={"market_structure": 2.0, "trend_momentum": 3.0}
            ),
            "unknown": AssetClassRoutingConfig(),
        }
    )


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


class PyramidingConfig(Base):
    """A fresh, smaller entry added only after the existing idea has worked."""

    enabled: bool = False
    #: Original position plus later legs. A separate setting controls whether
    #: the later winner-scalp tickets consume primary-idea slots; their own
    #: risk, margin and this per-symbol ceiling always remain binding.
    max_legs_per_symbol: int = Field(default=2, ge=1, le=4)
    #: Every existing leg must have moved this far in its own recorded R before
    #: another is permitted. That is the mechanical line between pyramiding a
    #: winner and averaging down a loser.
    #:
    #: Keep it above `trade_management.bank_at_r`, or the two rules fight: the
    #: banking rule closes a stalling winner at that level, so a scalp floor
    #: underneath it can only ever fire in the sliver between the two, and the
    #: guard checks banking every second while the scanner looks for add-ons
    #: once a cycle. Above it, a leg that is still open past the banking level
    #: is one the pace check judged to be still running, which is exactly the
    #: leg worth adding to.
    min_existing_r: float = Field(default=0.25, ge=0.0, le=3.0)
    #: Refuse to stack on a leg whose broker stop still sits behind its entry.
    #:
    #: This is what bounds a scalp campaign. With the original leg closed to
    #: loss, the worst case is the add-on's own quarter-size risk instead of
    #: every leg's full risk arriving together. Checked against the position
    #: the broker holds rather than the plan in the journal, because a journal
    #: that says break-even and a broker that never received the modification
    #: is precisely the state this exists to refuse.
    require_stop_beyond_entry: bool = True
    #: The fresh full-market analysis must independently clear this bar.
    minimum_conviction: float = Field(default=75.0, ge=0.0, le=100.0)
    #: Add-on risk as a fraction of the ordinary per-trade budget. It may only
    #: reduce size; values above one would turn confidence into leverage.
    risk_multiplier: float = Field(default=0.5, gt=0.0, le=1.0)
    #: An ordinary approval is not enough to stack the same thesis.
    minimum_ai_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    #: At most this many symbols may carry winner scalp legs simultaneously.
    #: One keeps the extra attention concentrated on a single proven idea.
    max_active_symbols: int = Field(default=1, ge=1, le=4)
    #: When false, proven-winner scalp legs are extra tickets belonging to the
    #: original trade idea rather than new primary ideas. Their risk, margin,
    #: per-symbol leg ceiling and every entry filter still apply.
    counts_toward_position_limit: bool = True


class ManualPositionManagementConfig(Base):
    """Adopt owner-opened MT5 positions into the same management ledger."""

    enabled: bool = False
    magic_numbers: tuple[int, ...] = (0,)
    stop_timeframe: str = "M15"
    stop_atr_multiple: float = Field(default=1.5, gt=0.0, le=10.0)
    target_reward_risk: float = Field(default=1.5, ge=1.0, le=10.0)

    @field_validator("stop_timeframe")
    @classmethod
    def _manual_timeframe_is_supported(cls, value: str) -> str:
        normalised = value.strip().upper()
        if normalised not in {item.value for item in Timeframe}:
            raise ValueError(f"unsupported manual stop timeframe {value!r}")
        return normalised

    @field_validator("magic_numbers")
    @classmethod
    def _manual_magics_are_external(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("manual magic_numbers must not be empty")
        if any(item < 0 for item in value):
            raise ValueError("manual magic_numbers must be non-negative")
        return value


class TradeManagementConfig(Base):
    pyramiding: PyramidingConfig = PyramidingConfig()
    manual_positions: ManualPositionManagementConfig = ManualPositionManagementConfig()
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
    #: Strategies whose measured edge uses the original broker SL/TP unchanged.
    #: Exact MT5 comments prevent unrelated positions from inheriting this.
    fixed_exit_comments: tuple[str, ...] = ()

    #: Minutes the peak may stand still before a profitable trade is banked.
    #: 0 switches the rule off.
    #:
    #: Every other exit here measures how much has been *given back*, which
    #: means every one of them can only act after the money has already gone.
    #: This one measures the thing a person actually watches: the trade stopped
    #: making new highs. A move that is working prints a new high every few
    #: minutes; one that has sat at the same level for six is not pausing, it
    #: is finished, and what follows is the retrace.
    #:
    #: Six minutes is roughly a M5 bar plus confirmation — long enough that an
    #: ordinary pullback inside a live move does not trigger it, short enough
    #: to still be near the high when it does.
    peak_stall_minutes: float = Field(default=6.0, ge=0.0, le=240.0)
    #: THE SAME DEFECT AS THE TIME EXIT, ONE SCREEN AWAY IN THE SAME FILE.
    #:
    #: The comment above states this rule's own design: "roughly a M5 bar plus
    #: confirmation". M5 is the QUICK profile's planning timeframe. The rule was
    #: built for a thirty-minute trade and then applied, as a wall-clock
    #: constant, to a trade planned on H1 with a target twenty-four hours out —
    #: where one bar is sixty minutes and four minutes of no new high is not a
    #: stall, it is a trade breathing.
    #:
    #: The account's own replay says what that cost, and it is written twice in
    #: `execution/manager.py`: PEAK_STALL banked +0.54R where leaving the
    #: position alone returned +1.17R, a lift of -0.64R. The recorded response
    #: was to make it wait longer, and the file's own comment on that says it
    #: "changes when it fires and not what it pays". Correct — because the wait
    #: was still being measured against the wrong clock.
    #:
    #: So the wait is at least this share of the trade's OWN plan.
    #:
    #:     quick     30 min plan  ->   4.5 min
    #:     intraday    3 h  plan  ->  27   min
    #:     swing      24 h  plan  ->   3.6 h
    #:
    #: `peak_stall_minutes` stays as the floor, so a position with no recorded
    #: plan — an adopted one — behaves exactly as it does today, and the rule
    #: can only ever fire LATER than it used to, never sooner. That direction
    #: is the safety argument: waiting longer to close a profitable position
    #: risks money left on the table, and the account has a measured -0.64R
    #: saying that is the cheaper of the two mistakes here.
    peak_stall_share_of_horizon: float = Field(default=0.15, ge=0.0, le=1.0)
    #: Profit required before a stalled peak means anything. Below this the
    #: trade has not made enough to be worth protecting and the noise band is
    #: wide enough that "no new high" says nothing.
    peak_stall_arm_r: float = Field(default=0.6, gt=0.0)
    #: How close to the peak the price must still be. Past this the money is
    #: already gone and the give-back rule owns the decision; this rule exists
    #: to leave *near the high*, not to confirm a retrace that has happened.
    peak_stall_near_peak: float = Field(default=0.7, gt=0.0, le=1.0)

    #: From this peak R, walk the stop up behind the trade instead of leaving
    #: it parked at break-even.
    #:
    #: Between `break_even_at_r` (0.6) and `partial_close_at_r` (1.5) the stop
    #: did not move at all: a trade could run to 1.4R and hand every cent of it
    #: back to a break-even stop, having been demonstrably right for hours.
    #:
    #: The give-back rule watches the same ground from inside our own loop, and
    #: this is deliberately redundant with it. A stop lives at the broker. It
    #: survives the VPS rebooting, the terminal dropping its connection, and
    #: this process dying at three in the morning — none of which the give-back
    #: rule survives. Protection that depends on our code still running is not
    #: protection at the moment it is most needed.
    #: 0.7, not 1.0. A live NZDCAD long peaked at 0.92R and stopped out at
    #: 0.13R for 22 cents: the lock armed at 1.0R and never fired, the
    #: give-back with conviction allowed an 80% drain to 0.18R, and the
    #: break-even stop sat below both at 0.13R and won the race. Three rules
    #: aimed at that trade and the crudest of them decided it. Arming below
    #: break-even's own reach is what stops that happening again.
    profit_lock_from_r: float = Field(default=0.7, gt=0.0)
    #: Share of the *peak* excursion the stop secures. 0.5 at a 2R peak leaves
    #: the stop at 1R. Deliberately not close to 1.0: a stop tucked right under
    #: the high is stopped out by ordinary noise, and strangling a winner is
    #: the single most expensive habit available here.
    profit_lock_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    #: How many spreads must fit between the peak and the stop the lock places.
    #:
    #: The fraction above is a share of the PEAK, so the gap it leaves shrinks
    #: with the peak: 0.6 at a 0.20R peak leaves 0.08R, and on an index with a
    #: nine-point stop that is under a point — inside the spread. A stop there
    #: is taken by the next tick whatever the market does. That is not a
    #: stop-out, it is a fee.
    #:
    #: 24 August measured it: 11 of 25 trades closed on BROKER_SL, all 11 in
    #: profit, +1.22R between them. 0.11R each — the arming minimum, to the
    #: cent. Eleven trades reached the lock and not one survived it, and
    #: exactly one trade in 25 ever reached its take-profit.
    #:
    #: Two rather than one, and the two is a CHOICE, not a measurement: one
    #: spread only covers the round trip, and price crosses its own book
    #: without the market going anywhere. The next scorecard grades it — read
    #: `DID MOVING THE STOP PAY`, where a stop-moved lift still negative means
    #: this number is too small.
    #:
    #: Zero restores the old behaviour: lock wherever the fraction says.
    profit_lock_minimum_spreads: float = Field(default=2.0, ge=0.0)

    #: A wide structural stop makes a material amount of money look small in R,
    #: and every protective rule here is written in R.
    #:
    #: The live case: CADCHF long, entry 0.58542, stop 0.58422, price 0.58595.
    #: EUR 2.82 of profit on a EUR 130 account — over two percent of everything
    #: — and only 0.44R, because the stop was twelve pips wide. Break-even
    #: (0.6R), peak-stall (0.6R) and the profit lock (0.7R) were all out of
    #: reach, so the whole amount sat exposed behind a stop still below entry.
    #: The operator saw that in one glance and was right.
    #:
    #: When the PEAK is worth at least this share of equity, the existing peak
    #: lock and peak-stall judgement may arm below their ordinary R floors.
    #: Nothing else changes: the lock still secures `profit_lock_fraction` of
    #: the peak at the broker rather than closing anything, and peak-stall
    #: still needs its full wait with price still near the high.
    #:
    #: THIS IS NOT the old blind cash-banking rule, which is off and staying
    #: off — that closed a winner the instant a number went green and was
    #: measured losing to the plan nine times in ten. This one never closes a
    #: moving trade for reaching an amount. It moves a stop.
    #:
    #: Zero disables the account-relative route entirely.
    capital_protection_at_equity_pct: float = Field(default=1.0, ge=0.0, le=10.0)

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

    #: Leave when the spread is this share of the room left to the stop.
    #:
    #: A stop does not trigger on the price you watch: a short closes at the
    #: ask, a long at the bid, and both move toward the stop when the book
    #: thins — with the market perfectly still. A live NZDJPY sell had bid
    #: 92.845 against a stop at 92.904: 5.9 pips of room, and six pips of
    #: spread would have taken it out having never once gone wrong.
    #:
    #: At 0.75 the quote owns three quarters of what is left. Both exits pay
    #: the spread, but this one happens at a price we chose, and on a trade
    #: near flat it turns a full stop-out into roughly break-even.
    spread_squeeze_share: float = Field(default=0.75, ge=0.0)
    #: ...but only while price has not already carried the trade most of the
    #: way to the stop. Past this the room is small because the trade is
    #: losing, not because the quote is wide, and the stop is doing its job.
    #: Without this the rule degenerates into closing every loser a moment
    #: early, which changes the risk profile for no gain.
    spread_squeeze_min_r: float = Field(default=-0.5)
    #: ...and not once the trade is meaningfully in profit. THE RULE PAYS THE
    #: VERY COST IT IS TRYING TO AVOID.
    #:
    #: Closing at market during a blown-out quote crosses that spread with
    #: certainty. Being stopped by it costs the same spread only IF price
    #: actually travels to the stop. So on a losing trade already leaning on its
    #: stop the trade is worth making — a probable cost swapped for a certain,
    #: smaller one, which is what the NZDJPY case in `_spread_squeeze_exit`
    #: describes. On a winner with room to spare it is the reverse: a certain
    #: cost paid to avoid a possible one, on a position that is working.
    #:
    #: The account's own replay says so. `SPREAD_SQUEEZE` closed eight of
    #: twenty-two trades on 17 August and every one of them scored a negative
    #: lift against its own untouched stop and target: it banked +0.38R where
    #: leaving the position alone returned +1.02R. The blowout passed, as
    #: blowouts do, and the trade carried on without us.
    #:
    #: Zero means "only at or below break-even", which is the shape the rule's
    #: own justification describes. Raise it to let the squeeze reach into
    #: profit again; a large value restores the previous behaviour.
    spread_squeeze_max_r: float = Field(default=0.0)

    #: WHAT MADE THE SQUEEZE WORTH KEEPING. Three tests it did not have.
    #:
    #: The rule was measured negative eleven times out of eleven — eight in the
    #: 17 August replay, every one worse than leaving the position alone, and
    #: three live afterwards (EURAUD -0.35R, NDX100 -0.44R, FRA40 -0.38R). The
    #: phenomenon it exists for is real: a short is closed at the ask and a long
    #: at the bid, so a thinning book can take a stop with the market perfectly
    #: still. What was wrong was how fast it believed it.
    #:
    #: 1. PERSISTENCE. Every one of those eleven blowouts passed and the trade
    #:    carried on without us. Acting on the first reading turns a temporary
    #:    quote into a certain loss, and waiting costs nothing — the stop is
    #:    there throughout, and a quote that really will take it is still wide
    #:    half a minute later. Both an elapsed time and a reading count, because
    #:    the guard tick's rate is not guaranteed: time so a fast loop cannot
    #:    fire on a burst, count so a slow one cannot fire on a single sample.
    spread_squeeze_persist_seconds: float = Field(default=30.0, ge=0.0, le=3_600.0)
    #: 2. ABNORMALITY. The old test asked only whether the spread was large
    #:    against the room left to the stop, and those are two different
    #:    complaints — a wide quote is one, a near stop is the other, and only
    #:    the first argues for leaving. A market that is simply always this wide
    #:    tripped it every tick once the stop came close, which is exactly what
    #:    happened after the tightener pulled the stop in. The reference is the
    #:    spread THIS position has been living with, so the question asked is
    #:    the right one: is the quote wider now than it has been while I held
    #:    this?
    spread_squeeze_abnormal_multiple: float = Field(default=2.0, ge=1.0, le=50.0)
    #: 3. EVIDENCE. Below this many readings nothing can be called abnormal, and
    #:    the safe answer is to hold: the stop is still there and it is what the
    #:    size was set against.
    spread_squeeze_min_samples: int = Field(default=10, ge=2, le=10_000)
    #: How many readings the comparison is drawn from. At a one-second guard
    #: tick this is two minutes of quote history, which is the right scale for
    #: "wider than usual" without a morning's calm dragging the median down.
    spread_squeeze_history: int = Field(default=120, ge=2, le=10_000)

    #: How long a freshly opened position is left alone by the DISCRETIONARY
    #: exits. Minutes.
    #:
    #: There was no such period at all: the management chain ran from the first
    #: tick after the fill, so a trade opened on a full multi-timeframe
    #: analysis was immediately subject to rules written for a trade that has
    #: been running. FRA40 SHORT on 19 August was opened with a correct stop
    #: and target and closed four minutes later at -0.38R, having peaked at
    #: +0.07R. Nothing about the market had happened; a quote had.
    #:
    #: It gates ONLY the rules that are meaningless without a history, and the
    #: distinction is the whole point. `_peak_stall_exit` looks for a peak that
    #: has not formed and `_giveback_exit` measures a give-back from that same
    #: absent peak — neither is a judgement about the trade, both are arithmetic
    #: on a number that does not exist yet.
    #:
    #: WHAT IT DELIBERATELY DOES NOT GATE. A trade that is genuinely broken
    #: after four minutes may leave, and should: a structure break is a
    #: discrete event and is as real in minute four as in hour four. The health
    #: read already carries its own bar — two independent families have to
    #: corroborate before it acts — and that bar, not a clock, is what should
    #: decide. Blocking it would mean holding a position the system can see is
    #: broken, which is a worse failure than the one this fixes.
    #:
    #: Nor does it touch the guard rails: the broker's stop loss stands
    #: throughout and remains the risk the position was sized against, the
    #: evening flatten and the session-decay exit are deadlines rather than
    #: opinions, and `tighten` still moves the stop, because reducing risk
    #: early is never the mistake this addresses.
    #:
    #: 10 minutes: two M5 bars, and long enough for a peak to mean something
    #: when one forms. Zero restores the old behaviour of measuring a
    #: give-back from the first tick after the fill.
    min_discretionary_exit_minutes: float = Field(default=10.0, ge=0.0, le=1_440.0)

    #: How much room, measured in spreads, a tightened stop must keep between
    #: itself and the price that triggers it.
    #:
    #: `HEALTH_TIGHTEN` halves the distance to the stop on a losing trade and
    #: had no floor, so every run halved it again — three leave an eighth.
    #: FRA40 SHORT on 19 August took exactly three inside one minute, all on
    #: the same reading, and was left with 0.89 of room against a spread of
    #: 1.2. The quote alone was wider than the entire distance to the stop.
    #:
    #: A stop inside one spread of price is not a stop. It is a fill at a worse
    #: price waiting to happen, and the position had stopped being a trade
    #: before any rule closed it.
    #:
    #: 2.0 is one spread to cross and one to breathe. It clamps rather than
    #: refuses, so the risk reduction the reading earned still happens; the
    #: existing "must improve on the current stop" test then declines the move
    #: outright once the floor catches up, which is what ends the ratchet.
    #: Zero restores the unbounded halving.
    min_stop_room_spreads: float = Field(default=2.0, ge=0.0, le=50.0)
    #: The same floor expressed as a share of the trade's own risk, and the
    #: larger of the two applies.
    #:
    #: Spreads alone are not a floor everywhere. On FRA40, 32 pips of spread
    #: makes two of them 64 pips of real protection; on EURAUD the spread is
    #: 0.10 pips, two of them is 0.2 pips, and the ratchet ran with nothing in
    #: its way. EURAUD LONG on 20 August took seven tightenings inside one
    #: minute, severity climbing 0.46 -> 0.68 and never reaching the 0.75 an
    #: exit needs. Its worst excursion was -0.60R against a real stop at
    #: -1.00R, so the market never came near it — and it closed at -0.66R on
    #: the tightened stop. The rules manufactured a loss the position had not
    #: taken.
    #:
    #: 0.15R is roughly a seventh of the distance the trade was sized against,
    #: which is enough room for ordinary noise and still lets a genuine
    #: reduction happen. Zero falls back to the spread floor alone.
    min_stop_room_r: float = Field(default=0.15, ge=0.0, le=1.0)
    #: How many tightenings in a row on a reading that will not improve before
    #: the position is closed instead.
    #:
    #: The exit threshold needs two families to agree, and a reading that never
    #: gets there can repeat forever while the stop is walked into the price.
    #: EURAUD LONG on 20 August took SEVEN tightenings inside one minute —
    #: severity 0.46, 0.64, 0.66, 0.67, 0.68, 0.68, 0.68, never reaching the
    #: 0.75 an exit needs — and closed on the walked-up stop at -0.66R, worse
    #: than its own deepest excursion of -0.60R. Its real stop was never
    #: touched.
    #:
    #: Seven tightenings ARE an exit, just a worse one: the position leaves at
    #: whatever the stop was dragged to, plus slippage, at a moment the market
    #: picks rather than one we do. A reading strong enough to move the stop
    #: that many times is strong enough to close on.
    #:
    #: 3, because that is the point at which the halving has taken most of the
    #: room anyway. The run only counts while the reading refuses to improve
    #: and any `hold` clears it, so a trade that steadies is not punished for a
    #: bad moment. Zero restores tightening without limit.
    #: Close on the second INDEPENDENT warning instead of tightening again.
    #:
    #: `HealthIntervention` already stops the same evidence acting twice, which
    #: is what walked EURAUD's stop into the price seven times in one minute.
    #: It does not answer what should happen when the evidence genuinely
    #: broadens, and there the answer is not a third stop move.
    #:
    #: An exit needs two families to reach 0.75 in ONE reading. Two families
    #: arriving minutes apart never sum, so a trade can deteriorate on
    #: trajectory, then on drift as well, and still only ever earn tighter
    #: stops. EURAUD LONG on 20 August did exactly that — `adverse_excursion`,
    #: then `momentum_turned, adverse_run, adverse_excursion` — and closed on
    #: the walked-up stop at -0.66R, worse than its own deepest excursion of
    #: -0.60R, on a trade whose real stop at -1.00R was never touched.
    #:
    #: Leaving at market at a moment we choose beats leaving at whatever the
    #: stop was dragged to, plus slippage, at a moment the market chooses.
    #: `_worth_paying_to_leave` still applies, so a position whose stop is
    #: nearer than the exit costs is left to the stop.
    exit_on_second_independent_warning: bool = True
    #: MAY A HEALTH READING CLOSE A TRADE AT ALL.
    #:
    #: The rule pre-registered its own kill condition in `config/eightcap.yaml`
    #: in these words: "blijft `lift` negatief, dan is er geen instelling die
    #: dit repareert". It has now been measured at three settings and the lift
    #: was negative at all three:
    #:
    #:     broken_at 0.65   13 trades   0 won   -EUR 20,33   lift -0.40R
    #:     broken_at 0.75   12 trades   0 won   -EUR 36,82   lift -0.06R
    #:     broken_at 0.85   34 trades   0 won   -EUR 45,89   lift -0.42R
    #:
    #: Zero winners in fifty-nine closes is not by itself damning -- the rule
    #: fires on trades whose thesis broke, so losers are what it should be
    #: catching. The counterfactual column is what damns it:
    #:
    #:     stepping in was   -0.26R
    #:     doing nothing was +0.14R
    #:
    #: Doing nothing was PROFITABLE. And the damage is concentrated in one
    #: sequence -- tighten the stop, then close anyway:
    #:
    #:     stop moved, then closed   6 trades   lift -0.80R
    #:     stop untouched            2 trades   lift +0.82R
    #:
    #: False downgrades an "exit" reading to "hold" and journals it as
    #: HEALTH_EXIT_SUPPRESSED with the R it would have closed at, so the
    #: counterfactual keeps being measured on live trades rather than going
    #: dark. Tightening and profit-securing are untouched; this is only the
    #: close. The stop the trade was opened with does the work instead.
    health_exit_closes: bool = True

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
    #: A losing trade whose path and at least one independent chart family both
    #: contradict the entry thesis may be cut before the original full stop.
    #: This only reduces risk; it never widens a stop or creates a position.
    thesis_invalidation_at_r: float = Field(default=0.35, ge=0.0, le=1.0)
    #: Combined severity at which the reading becomes "broken" and the position
    #: is closed rather than merely tightened.
    #:
    #: This was a constant in `analysis/position_health.py`, which meant the one
    #: knob controlling how early a failing trade gets cut could only be moved
    #: by editing source. It is the second time this exact complaint has been
    #: acted on — the owner already raised the `trajectory` family weight from
    #: 0.30 to 0.50 "after watching trades run from a manageable loss to the
    #: full stop with every reader watching and none able to speak".
    #:
    #: The floor of 0.55 is enforced in `assess` and is not a preference. The
    #: heaviest single family is `trajectory` at 0.50, so any value above that
    #: keeps "no one family can close a trade alone" true by arithmetic rather
    #: than by convention. The corroboration rule demotes a lone family's exit
    #: to a stop tightening as well, so the guarantee is doubled.
    health_broken_at: float = Field(default=0.75, ge=0.55, le=1.0)
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

    #: Take a sum worth taking, unless the move is clearly still running.
    #:
    #: The operator's rule, in their words: on a EUR 100 account, 50 to 90 cents
    #: is a fine amount to bank; on EUR 1000 it is five to twenty euro. That is
    #: the same thing said twice — a share of the account — so it is expressed
    #: as one, and it scales without anybody editing it after a deposit.
    #:
    #: The posture is deliberately inverted from every other rule in this file.
    #: The rest hold by default and act on evidence of trouble. This one *banks*
    #: by default and holds only on evidence the move is still running, because
    #: "prima bedragen om te stationen" is the whole point: a profit you can see
    #: beats a bigger one you are hoping for.
    #:
    #: BE HONEST ABOUT THE ARITHMETIC. `bank_at_r` holds this to 0.3R whatever
    #: the lot rounded to. Banking at 0.3R against a 1R stop needs a win rate
    #: near 77% to break even, and the backtest measured 24-33%. This does
    #: not fix a negative edge and it is not meant to; it stops handing back
    #: what the account has already earned. Whether it helps is measurable with
    #: `backtest.cmd --targets`, which sweeps exactly this question.
    bank_enabled: bool = True
    #: Share of account equity that counts as worth taking.
    bank_at_equity_pct: float = Field(default=0.6, ge=0.0, le=10.0)
    #: The same sentence said in R, and the trade banks at whichever is lower.
    #:
    #: Needed because the equity share alone is not stable from trade to trade.
    #: The sizer rounds the lot *down* to the broker's step, so the risk really
    #: carried sits under the intended 2% and varies by instrument — 0.66% to
    #: 1.95% of equity across the last twenty live trades. A fixed 0.6%-of-
    #: equity threshold therefore lands anywhere between 0.31R and 0.91R, and at
    #: the top of that range it is not a banking rule: it demands nearly the
    #: whole trade. See `PositionManager._worth_taking`.
    bank_at_r: float = Field(default=0.3, ge=0.0, le=3.0)
    #: Let the account's own history lower `bank_at_r`, never raise it.
    #:
    #: Default on because the mechanism is sound and the direction is safe --
    #: it can only take profit sooner, which is less exposure. Whether taking
    #: profit sooner is *better* is a separate question, and the answer lives
    #: in `management_baselines`: every banked trade replayed against its own
    #: untouched stop and target. Turn this off wherever that baseline says
    #: early banking gives back more than it protects.
    use_learned_bank_threshold: bool = True
    #: How hard price must be coming BACK against the position to earn a hold,
    #: in random-walk units — see `analysis.position_health.drift_score`.
    #:
    #: Renamed from `bank_still_running_drift`, and the rename is the finding.
    #: The old field held the trade while the move was running our way, which
    #: is the intuitive rule and is measurably backwards.
    #: `backtest.cmd --exits --days 90` reports what an extra minute of
    #: patience was worth at every in-profit moment, split by pace, and a
    #: running move is negative at all seven profit levels on thousands of
    #: observations each. A hard run is exhaustion. The state where waiting
    #: pays is a retrace, which the old rule treated as a reason to leave.
    #:
    #: Still not zero, for the same reason as before inverted: "not obviously
    #: running" is the absence of news, and holding needs the presence of it.
    bank_while_retracing_drift: float = Field(default=0.5, ge=0.0, le=3.0)

    #: Bank a profit whose target the session can no longer deliver.
    #:
    #: No threshold to set, deliberately. The rule asks two measured questions
    #: — is the remaining distance reachable in the time left, and is what is
    #: on the table worth more than the spread and commission it costs to
    #: collect — and an R figure chosen in advance can answer neither. The
    #: version this replaced fired at 0.3R, which was a number I made up.
    session_decay_enabled: bool = True

    #: Close a position that has gone nowhere. Dead capital still carries risk.
    time_exit_hours: float | None = Field(default=24.0, gt=0.0)
    time_exit_min_abs_r: float = Field(default=0.3, ge=0.0)
    #: Past the deadline, also bank a profitable trade whose *peak* never
    #: reached this. Between `time_exit_min_abs_r` (0.3) and the give-back's
    #: arming point (0.5R) sat a gap nothing owned: a position on +0.4R after
    #: a day and a half was too profitable for the time exit and never good
    #: enough for the give-back, so it stayed on indefinitely, paying swap for
    #: one of two slots it was not using. Judged on the peak rather than the
    #: current price, because a trade that ran to 2R and came back has proved
    #: something — that one belongs to the give-back rule, not to this one.
    time_exit_stale_peak_r: float = Field(default=1.0, ge=0.0)

    #: MEASURE THE DEADLINE AGAINST THE PLAN, NOT AGAINST A CONSTANT.
    #:
    #: `time_exit_hours` was one number for every trade the system has ever
    #: taken, and the engine has decided a per-proposal horizon the whole time:
    #: `quick` is thirty minutes, `intraday` three hours, `swing` twenty-four.
    #: The confluence engine's own comment on that number says "everything
    #: downstream derives its window from this". The exit did not derive
    #: anything from it — `execution/manager.py` did not contain the word
    #: horizon — so a thirty-minute idea sat in a position slot for a day.
    #:
    #: That is one defect with two consequences, which is why it is worth this
    #: much comment. THE SLOT is what caps trades per day: a position released
    #: after its own three-hour plan frees the slot eight times sooner than one
    #: released after twenty-four. And THE TRADE is decided by whatever moves
    #: the market in the hours it is held, so a detector that claimed to read
    #: the next thirty minutes and was then judged over twenty-four hours was
    #: measured almost entirely on noise it never claimed to see. That is a
    #: mechanism for "no detector beat a coin flip" which does not require any
    #: of them to be worthless.
    #:
    #: NOTHING ABOUT SWING CHANGES. Its plan is H1 x 24 bars = 1,440 minutes =
    #: exactly the 24.0 that `time_exit_hours` already held, so at a multiple of
    #: 1.0 a swing trade keeps the deadline it has always had. The two numbers
    #: were the same number all along; only one of them was ever applied.
    time_exit_uses_plan_horizon: bool = True
    #: Rope on top of the plan's own length, because expiring exactly on the
    #: target's own bar is a coin flip about one bar rather than a policy. 1.5
    #: gives a three-hour plan four and a half hours before the deadline can
    #: fire, and the deadline is not itself an exit: `_time_exit_verdict` still
    #: only closes a trade that went nowhere or peaked below
    #: `time_exit_stale_peak_r`. A trade that is working is never closed by it.
    time_exit_horizon_multiple: float = Field(default=1.5, gt=0.0, le=10.0)
    #: A floor under the derived deadline, so a very short plan cannot produce
    #: a deadline inside the spread's own noise. Thirty minutes is the quick
    #: profile's whole horizon; below that the number stops being a plan.
    time_exit_minimum_hours: float = Field(default=0.5, ge=0.0)

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
    #: The same ladder in money, as a share of equity, and it exists because R
    #: is the width of the stop rather than a unit of value.
    #:
    #: The live CADCHF long held EUR 2.82 on a EUR 130 account — 2.2% of
    #: everything — at 0.44R, because its stop was twelve pips wide. The R
    #: ladder had last spoken at 0.25R and the next rung was 0.50R, so nothing
    #: asked the supervisor about it while that money stood behind a stop still
    #: below the entry price. Half a percent of the account is a rung the
    #: operator can feel; a quarter of an R on a wide stop is not.
    #:
    #: This only decides WHEN the reviewer is asked. It closes nothing, moves
    #: nothing, and every other gate is unchanged. Zero disables it.
    supervision_profit_step_equity_pct: float = Field(default=0.5, ge=0.0, le=10.0)
    #: Once a profitable position gives this fraction of its recorded peak
    #: back, ask the judgement layer early. The hard mechanical give-back rule
    #: remains in force and does not wait for the API.
    supervision_giveback_trigger_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    #: The same two ladders, on the losing side. Every trigger above clips the
    #: negative half away with `max(x, 0.0)`, so a position walking from -0.1R
    #: to -0.8R crossed nothing and nobody was asked about it. Between the
    #: fifteen-minute review and the health reader — which speaks about
    #: structure, not about the money — a trade could lose most of its risk
    #: budget without one deliberate look.
    #:
    #: That is the wrong half to be quiet on. A gain handed back is a gain; a
    #: loss deepening is the account. The question these ask is the one a person
    #: asks out loud: is this going further against me, and would I rather take
    #: the smaller loss now? The answer is the reviewer's — nothing here closes
    #: anything, and the stop remains the only thing that does so by itself.
    supervision_loss_step_r: float = Field(default=0.25, ge=0.0, le=5.0)
    #: The money rung on the losing side. Deliberately tighter than the profit
    #: ladder: a missed gain can be re-entered tomorrow and a loss cannot be
    #: un-lost, so the losing side is worth asking about sooner.
    supervision_loss_step_equity_pct: float = Field(default=0.35, ge=0.0, le=10.0)

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
    #: Realised broker trades may reorder already-valid candidates once the
    #: segment has enough evidence. Never changes eligibility, prices or size.
    selection_calibration_enabled: bool = True
    #: Trades a segment needs before it may speak at all.
    #:
    #: The floor goes down to ten because the shrinkage below is the real
    #: protection and it is continuous: a ten-trade segment is multiplied by
    #: 10/(10+shrinkage), so a thin sample earns a proportionally tiny voice
    #: rather than being silenced by a cliff and then, one trade later, given a
    #: full one. A hard floor on top of shrinkage is belt-and-braces that in
    #: practice just switched the whole mechanism off on any account trading
    #: less than a few hundred times a month.
    selection_min_trades: int = Field(default=40, ge=10, le=10_000)
    #: How fast a segment earns its full voice. Lower means a small account's
    #: own evidence counts sooner; the cap still bounds the total effect.
    selection_shrinkage_trades: int = Field(default=80, ge=5, le=10_000)
    #: Ranking points per R of measured expectancy, after shrinkage. Has to be
    #: large enough that the result is visible against the spread of confluence
    #: scores, or the calibration runs and changes no ordering at all.
    selection_points_per_r: float = Field(default=6.0, ge=0.0, le=40.0)
    selection_modifier_cap: float = Field(default=4.0, ge=0.0, le=10.0)
    #: The ensemble combines reusable outcome facets instead of waiting for one
    #: sparse exact segment.  It remains ranking-only and separately bounded.
    selection_ensemble_strength: float = Field(default=1.5, ge=0.0, le=3.0)
    selection_ensemble_modifier_cap: float = Field(default=6.0, ge=0.0, le=10.0)
    #: Raw confluence establishes eligibility, but must not drown out realised
    #: evidence once candidates are ordered. This affects ordering only.
    selection_raw_conviction_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    #: Reward genuinely different observations, not three variants of one EMA.
    selection_independent_family_bonus: float = Field(default=1.5, ge=0.0, le=5.0)
    selection_independent_family_cap: int = Field(default=3, ge=1, le=8)
    selection_outcome_floor_r: float = Field(default=-1.0, ge=-5.0, le=0.0)
    selection_outcome_cap_r: float = Field(default=2.0, ge=0.1, le=10.0)
    selection_dimension_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "setup_horizon": 1.5,
            "setup_family": 1.0,
            "detector": 1.0,
            "regime": 1.0,
            "session": 0.8,
            "score_band": 0.8,
            "asset_class": 0.6,
            "horizon": 0.6,
            "direction": 0.5,
        }
    )
    selection_refresh_minutes: int = Field(default=15, ge=1, le=1440)

    @model_validator(mode="after")
    def validate_selection_brain(self) -> LearningConfig:
        allowed = {
            "setup_horizon",
            "setup_family",
            "detector",
            "regime",
            "session",
            "score_band",
            "asset_class",
            "horizon",
            "direction",
        }
        unknown = set(self.selection_dimension_weights) - allowed
        if unknown:
            raise ValueError(f"unknown selection_dimension_weights: {sorted(unknown)}")
        if any(value < 0.0 for value in self.selection_dimension_weights.values()):
            raise ValueError("selection_dimension_weights may not be negative")
        if self.selection_outcome_floor_r >= self.selection_outcome_cap_r:
            raise ValueError("selection outcome floor must be below its cap")
        return self


class DataQuarantineConfig(Base):
    """Skip the timeframe ladder for symbols the broker cannot supply.

    Two failures are structural rather than momentary: too little history
    ("8 closed bars available, 50 required") and holes inside trading weeks.
    Both are properties of the broker's feed for that symbol, so re-deriving
    them every cycle for an 800-symbol catalogue on one vCPU is pure waste.

    This changes nothing about what may be traded. A held symbol was already
    refused and is still refused; only the cost of saying so falls. The
    candidate set can shrink here and never grow, so no spread, session,
    liveliness or risk rule is reachable differently because of it.

    The hold is released early by the symbol's own next bar, so a market that
    opens again tomorrow does not wait out a clock. These minutes are only the
    backstop for when the scan cannot read a bar at all.
    """

    enabled: bool = True
    #: First hold after one bad fetch. Short, because a single gap may be a
    #: momentary feed problem rather than a missing year of history.
    initial_minutes: float = Field(default=30.0, gt=0.0, le=10_080.0)
    #: Each repeat failure multiplies the hold. A symbol offering eight weekly
    #: bars where fifty are needed does not need re-checking every half hour.
    backoff_multiple: float = Field(default=4.0, ge=1.0, le=100.0)
    #: The ceiling. Hours, not days: a fetch every cycle down to one an hour
    #: already removes 98.3% of the waste, and stretching that to a day removes
    #: 99.93%. Those last seven-hundredths of a percentage point do not buy
    #: enough to be worth a day of blindness to a market.
    max_minutes: float = Field(default=240.0, gt=0.0, le=43_200.0)

    @model_validator(mode="after")
    def _ceiling_above_floor(self) -> DataQuarantineConfig:
        if self.max_minutes < self.initial_minutes:
            raise ValueError("data_quarantine.max_minutes is below initial_minutes")
        return self


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

    #: Stop re-fetching the ladder for symbols the broker has no usable history
    #: for. Ordering and every gate are untouched; a held symbol was already
    #: producing "no trade" and still does.
    data_quarantine: DataQuarantineConfig = Field(default_factory=lambda: DataQuarantineConfig())
    #: Symbols cheaply ranked per cycle. None = the whole catalogue.
    batch_size: int | None = Field(default=None, ge=1)
    #: Keep the liquid lanes on the fast clock while the remainder of a large
    #: catalogue rotates in bounded batches. Without this, an 850-symbol pass
    #: can take three minutes and a one-candle M1 event can appear and expire
    #: before its symbol is inspected again. Priority does not change a gate;
    #: it only changes how often a configured market is looked at.
    priority_every_cycle: bool = False
    #: Top-ranked symbols promoted to full analysis each cycle. The ceiling is
    #: the size of a large broker catalogue, so "analyse everything the cheap
    #: scan let through" is expressible. The old limit of 100 made that
    #: impossible to configure, which is a policy decision that does not belong
    #: in a schema bound.
    deep_candidates: int = Field(default=12, ge=1, le=2000)
    #: Selection lanes, highest-quality markets first. A preferred symbol is
    #: lane 2, a preferred asset class lane 1, and everything else remains a
    #: lane-0 fallback. This changes ordering only: every configured symbol is
    #: still inspected and must pass the same analysis, filters and risk gates.
    priority_asset_classes: tuple[
        Literal["forex", "crypto", "stock", "index", "metal", "commodity"], ...
    ] = ()
    priority_symbols: tuple[str, ...] = ()
    #: Low transaction cost may reorder candidates inside one lane, but can
    #: never promote a fallback instrument ahead of a preferred market.
    priority_spread_weight: float = Field(default=0.0, ge=0.0, le=25.0)
    #: How strongly to prefer the setup that keeps more of what it wins.
    #:
    #: The ranking was dominated by conviction — score times confidence, worth
    #: 16 to 76 points — with cost entering only as `priority_spread_weight`,
    #: capped at 10 and blind to the target. Two measurements on this account
    #: say conviction does not predict the outcome at all: the "20+ over the
    #: bar" bucket was the worst at -4.92R over 23 trades, and among 84 paid
    #: reviews the 40-45 conviction band produced nothing useful while 20-25
    #: produced 33%. So the order was being set, almost entirely, by noise.
    #:
    #: What is NOT noise is the toll. Measured on real fills, every one of the
    #: live trades spent over a quarter of its risk on commission and slippage.
    #: This scores the arithmetic the toll implies — how much reward-to-risk
    #: survives a round trip — which is not a forecast but a subtraction.
    #:
    #: Ordering only, like everything else in the selection score: it decides
    #: who is examined first and who gets a scarce paid review. It cannot
    #: approve a setup any gate refused. 0.0 switches it off.
    after_cost_priority_weight: float = Field(default=12.0, ge=0.0, le=50.0)
    #: Stop re-measuring a market whose spread is not close to the limit.
    #:
    #: A catalogue scan of 845 markets found 124 crypto pairs blocked on
    #: spread with a class median of 1,581 bps — sixteen percent of the price,
    #: seventy-nine times the cap. Those are not markets that tighten by lunch;
    #: they are quoted that way. Fetching a contract and a tick for each of them
    #: every rotation, on one vCPU, is time not spent on the markets that can
    #: actually be traded.
    #:
    #: This changes nothing about what may be traded. Every symbol held here was
    #: already refused for exactly this reason and is still refused; the refusal
    #: is simply remembered for a few hours instead of re-derived. The candidate
    #: set can shrink here and never grow.
    #:
    #: The multiple is what keeps it honest. A market a hair over the cap —
    #: XAGUSD at 8.40 against 8.0, BNBUSD at 24.99 against 20 — can tighten
    #: within the hour and is never held: it is re-measured every rotation like
    #: everything else. Only what is far past the line is put down.
    wide_spread_park_multiple: float = Field(default=2.0, ge=1.0, le=1000.0)
    #: How long such a market is left alone. 0.0 switches the whole thing off
    #: and every symbol is measured every rotation again.
    wide_spread_park_hours: float = Field(default=6.0, ge=0.0, le=168.0)

    @field_validator("priority_symbols")
    @classmethod
    def _priority_symbols_are_normalised(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(dict.fromkeys(symbol.strip().upper() for symbol in value if symbol.strip()))
        if len(cleaned) != len(value):
            raise ValueError("scanner.priority_symbols must be non-empty and unique")
        return cleaned


class ExternalSignalsConfig(Base):
    """Authenticated phone notifications entering the ordinary order pipeline."""

    enabled: bool = False
    source: str = "rio"
    listen_host: str = "127.0.0.1"
    listen_port: int = Field(default=8765, ge=1024, le=65_535)
    token_env: str = "RIO_SIGNAL_TOKEN"
    inbox_path: str = "runtime/external_signals"
    max_age_seconds: int = Field(default=180, ge=15, le=3600)
    fixed_volume: float = Field(default=0.01, gt=0.0, le=100.0)
    partial_close_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    entry_zone_tolerance_bps: float = Field(default=2.0, ge=0.0, le=100.0)
    gold_follow_enabled: bool = False
    gold_follow_aliases: tuple[str, ...] = ("GOLD", "XAUUSD")
    gold_follow_timeframe: str = "M5"
    gold_follow_structure_bars: int = Field(default=24, ge=5, le=200)
    gold_follow_stop_atr_multiple: float = Field(default=1.5, ge=0.5, le=5.0)
    gold_follow_structure_buffer_atr: float = Field(default=0.25, ge=0.0, le=2.0)
    gold_follow_target_reward_risk: float = Field(default=1.2, ge=0.5, le=5.0)
    gold_follow_max_entry_deviation_atr: float = Field(default=0.75, ge=0.0, le=5.0)
    gold_follow_max_entry_deviation_bps: float = Field(default=8.0, ge=0.0, le=100.0)
    allowed_apps: tuple[str, ...] = ("Rio Traders",)
    symbol_aliases: dict[str, str] = Field(default_factory=lambda: {"GOLD": "XAUUSD"})

    @field_validator("source", "token_env", "inbox_path")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("external signal text values must not be empty")
        return cleaned

    @field_validator("symbol_aliases")
    @classmethod
    def _normalise_aliases(cls, value: dict[str, str]) -> dict[str, str]:
        aliases = {
            str(alias).strip().upper(): str(symbol).strip()
            for alias, symbol in value.items()
            if str(alias).strip() and str(symbol).strip()
        }
        if len(aliases) != len(value):
            raise ValueError("external_signals.symbol_aliases contains an empty alias")
        return aliases

    @field_validator("gold_follow_aliases")
    @classmethod
    def _normalise_gold_follow_aliases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        aliases = tuple(dict.fromkeys(alias.strip().upper() for alias in value if alias.strip()))
        if not aliases:
            raise ValueError("external_signals.gold_follow_aliases must not be empty")
        return aliases

    @field_validator("gold_follow_timeframe")
    @classmethod
    def _gold_follow_timeframe_is_supported(cls, value: str) -> str:
        cleaned = value.strip().upper()
        Timeframe.parse(cleaned)
        return cleaned


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
    provider: Literal["openai", "anthropic", "consensus", "local_history"] = "consensus"
    openai_model: str = "gpt-5.1"
    anthropic_model: str = ""
    minimum_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)
    fail_closed: Literal[True] = True
    market_scout: MarketScoutConfig = MarketScoutConfig()
    #: The cost-free adviser only repeats a veto when several sufficiently
    #: similar, non-replayed Claude reviews agree. Unknown setups fall back to
    #: the deterministic Jarvis gates instead of pretending the archive knows.
    local_history_min_neighbors: int = Field(default=5, ge=2, le=100)
    local_history_max_distance: float = Field(default=0.55, gt=0.0, le=2.0)
    local_history_veto_rate: float = Field(default=0.80, ge=0.5, le=1.0)
    #: Paid pre-trade reviews allowed per cycle. 0 removes the budget.
    #:
    #: Candidates reach the reviewer in the engine's own order of conviction,
    #: so a budget is spent on the best setups that survived every free gate
    #: and then stops. Without one, a cycle pays to be told that its ninth,
    #: tenth and eleventh-best ideas are weak — which is what the reviewer
    #: actually said, in those words, on a live account.
    #:
    #: Deliberately a budget rather than a rank cut-off. Candidates are
    #: processed in rank order and a low rank is only *reached* because the
    #: better ones were rejected earlier, so "never review below rank 3" would
    #: silently stop trading altogether on any day the top three keep failing
    #: a deterministic gate. A budget cannot do that: whatever reaches the
    #: reviewer first is by construction the best thing still standing.
    max_reviews_per_cycle: int = Field(default=3, ge=0, le=50)
    #: Minutes before the same market and side may be paid for again.
    #:
    #: The shape memory next to this one matches on entry AND stop within a
    #: quarter of an ATR, which is right for "is this literally the same
    #: proposal" and misses the case that costs money. Live: EURCAD SHORT
    #: reviewed at 10:15:36 and again at 10:18:38, both paid, both refused,
    #: with confidence 0.28 and 0.32. Three minutes of drift moved the entry
    #: past the tolerance so the memory called it a new question. It was not a
    #: new question — nothing about a market changes in three minutes that a
    #: reviewer looking at H4 and H1 bars would notice.
    #:
    #: Keyed on symbol and direction alone, deliberately, and short enough that
    #: a genuine intraday turn is not missed: twenty minutes is a third of an
    #: H1 bar. An approval clears it immediately, as every other memory here
    #: does. 0 switches it off.
    veto_cooldown_minutes: float = Field(default=20.0, ge=0.0, le=240.0)


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
    external_signals: ExternalSignalsConfig = ExternalSignalsConfig()
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
    def _the_two_target_floors_agree(self) -> Settings:
        """The analysis may not plan a target the sizer will refuse.

        `analysis.confluence.minimum_r_multiple` is the shortest distance the
        target search may consider; `risk.min_risk_reward` refuses anything
        under it one layer later. When the first is below the second, every
        target chosen in the gap is built, measured, scored, sized and then
        thrown away as RR_BELOW_MINIMUM.

        That is not hypothetical. It shipped once already — the overlay note on
        `min_risk_reward` records confluence trimming to 0.9R while the sizer
        refused everything under 1.1R — and the fix was to align the two by
        hand, which is a fix that survives exactly until somebody edits one of
        them. Two numbers that must agree and are only kept in step by memory
        will come apart again.
        """
        floor = self.analysis.confluence.minimum_r_multiple
        margin = self.analysis.confluence.target_planning_margin
        needed = self.risk.min_risk_reward * (1.0 + margin)
        if floor + 1e-9 < needed:
            raise ValueError(
                f"analysis.confluence.minimum_r_multiple ({floor}) leaves no room above "
                f"risk.min_risk_reward ({self.risk.min_risk_reward}): the target search "
                f"lands on its own floor on nearly every market, so a plan built at "
                f"{floor}R reaches the sizer with nothing to spare and any adverse tick "
                f"between the two refuses it as RR_BELOW_MINIMUM. With a "
                f"{margin:.0%} planning margin the floor must be at least {needed:.2f}. "
                f"Raise it, lower risk.min_risk_reward, or set "
                f"analysis.confluence.target_planning_margin to 0 to accept the knife edge."
            )
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
            broker_symbol
            for sym in self.instruments.whitelist[self.system.mode.value]
            if not self.instruments.is_ignored(broker_symbol := self.instruments.broker_symbol(sym))
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

    def effective_max_positions(self, equity: float | None = None) -> int:
        """How many positions may be open at once, given what the account holds.

        The mode's ceiling is absolute and is never exceeded — this only decides
        where between the floor and that ceiling the account currently sits.
        Passing no equity returns the ceiling, which is what the startup banner
        and the profile printout want: the shape of the mode, not today's
        balance.

        Slots are earned in whole steps of `equity_per_position`, so the number
        moves when the account genuinely grows rather than drifting with every
        floating tick.
        """
        ceiling = self.active_limits.max_concurrent_positions
        step = self.risk.equity_per_position
        if equity is None or step <= 0:
            return ceiling
        earned = int(max(0.0, equity) // step)
        return max(1, min(ceiling, max(self.risk.min_concurrent_positions, earned)))

    def symbol_allowed_at_equity(self, symbol: str, equity: float) -> tuple[bool, str]:
        """Whitelist + equity gate for one symbol.

        Returns (allowed, reason). The reason string is journalled verbatim so
        that "why did it not trade gold" is answerable months later.
        """
        bare = self.instruments.canonical_symbol(symbol)
        if self.instruments.is_ignored(symbol):
            return False, "SYMBOL_IGNORED"
        if self.instruments.is_observation_only(symbol):
            return False, "SYMBOL_OBSERVATION_ONLY"
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

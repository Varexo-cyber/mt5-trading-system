"""Machine-readable reasons for every accept/reject decision.

These strings land verbatim in the journal, so they are part of the data model
rather than log prose. Six months from now the question "why did it not trade
gold in March" has to be answerable with a `GROUP BY reason`, and that only
works if the vocabulary is closed and stable.

Renaming a member is a breaking change to the journal. Add a new one instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Reason(StrEnum):
    """Why a trade was allowed, or the specific gate that stopped it."""

    OK = "OK"
    NO_SIGNAL = "NO_SIGNAL"
    #: Independent techniques read the same chart in opposite directions. Not a
    #: close call to be settled by whichever scored higher — a market with no
    #: clear edge, where the honest answer is to stand aside.
    METHODS_DISAGREE = "METHODS_DISAGREE"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    #: The broker's history for this symbol was already found unusable — too
    #: few bars, or holes inside trading weeks — so the timeframe ladder was
    #: not fetched again this cycle. Same outcome as DATA_UNAVAILABLE and kept
    #: separate on purpose: one is a market that was examined and refused, the
    #: other a market not examined because examining it has a known answer.
    #: Reads as a saving in the report rather than as a fresh rejection.
    DATA_QUARANTINED = "DATA_QUARANTINED"
    AI_VETO = "AI_VETO"
    #: The reviewer has already refused this instrument and direction several
    #: times for the same underlying reason, and nothing about that reason has
    #: changed. Different from AI_VETO: this trade was not judged, it was
    #: recognised. An approval on the same pair wipes the pattern.
    AI_VETO_PATTERN = "AI_VETO_PATTERN_KNOWN"
    #: This cycle has spent its paid-review budget on higher-conviction
    #: candidates. Not a judgement on the setup — it was never asked about.
    #: It comes back next cycle, and if the better ideas have cleared out by
    #: then, it gets the budget.
    AI_BUDGET_SPENT = "AI_REVIEW_BUDGET_SPENT"
    #: This market and side were paid for and refused a few minutes ago. Unlike
    #: AI_VETO_PATTERN this needs no repeated reason and no matching shape --
    #: only that the question was recently bought. Live case: EURCAD SHORT
    #: reviewed twice in three minutes, both refused, because three minutes of
    #: price drift moved the proposal past the shape memory's tolerance.
    AI_VETO_COOLDOWN = "AI_VETO_COOLDOWN"
    LIVE_NOT_ARMED = "LIVE_NOT_ARMED"

    # -- sizing: the account cannot express this trade --------------------
    #: Computed lot is below the broker's minimum. The system skips instead of
    #: rounding up, because rounding up silently multiplies the intended risk.
    UNDERCAPITALIZED = "TRADE_SKIPPED_UNDERCAPITALIZED"
    #: The structural stop is wider than the mode allows for this account.
    SL_TOO_WIDE_FOR_ACCOUNT = "SL_TOO_WIDE_FOR_ACCOUNT"
    #: The stop sits inside the broker's minimum stop distance.
    SL_TOO_TIGHT_FOR_BROKER = "SL_TOO_TIGHT_FOR_BROKER"
    #: The stop is legal at the broker and still too narrow to be worth taking:
    #: commission and the slippage a stop-out actually suffers would be a large
    #: share of the risk. A live AUDNZD stop-out on a 5-pip stop returned
    #: -1.48R, of which 0.56R was pure cost.
    SL_TOO_TIGHT_FOR_COSTS = "SL_TOO_TIGHT_FOR_COSTS"
    #: Even the minimum lot would risk more than the configured ceiling.
    RISK_EXCEEDS_CAP = "RISK_EXCEEDS_CAP"
    #: Stop missing, on the wrong side of entry, or equal to it.
    INVALID_STOP = "INVALID_STOP"
    #: Price is running against the trade at the moment of entry. Not a
    #: judgement on the setup — the setup may be right and simply early. It is
    #: re-examined every cycle and taken as soon as the adverse move stops.
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    #: Direction may be sound, but the current market order would chase a move
    #: already stretched to the edge of its recent M5 range. Rechecked every
    #: cycle; a pullback/retest can clear it without changing the thesis.
    ENTRY_OVEREXTENDED = "ENTRY_OVEREXTENDED"
    #: Claude agreed with the direction but explicitly judged the current price
    #: too extended for a market order. Different from AI_VETO: the setup is not
    #: remembered as bad and is offered again when its price shape changes.
    AI_WAIT_RETEST = "AI_WAIT_RETEST"
    #: The executable quote or elapsed time changed materially while the paid
    #: review was in flight. The approval belongs to the old snapshot, not the
    #: new price, so the setup returns next cycle instead of being chased.
    ENTRY_MOVED_DURING_REVIEW = "ENTRY_MOVED_DURING_REVIEW"
    #: Manual activity or another fill changed the account's open-position set
    #: while Claude was judging it. The slot, correlation and add-on context in
    #: that approval are stale, so a fresh cycle must form a new proposal.
    ENTRY_STATE_CHANGED_DURING_REVIEW = "ENTRY_STATE_CHANGED_DURING_REVIEW"
    #: Reward-to-risk on structural levels is below the minimum.
    RR_BELOW_MINIMUM = "RR_BELOW_MINIMUM"

    # -- instrument gates --------------------------------------------------
    SYMBOL_NOT_WHITELISTED = "SYMBOL_NOT_WHITELISTED"
    SYMBOL_BLOCKED_BY_EQUITY = "SYMBOL_BLOCKED_BY_EQUITY"
    SYMBOL_NOT_TRADABLE = "SYMBOL_NOT_TRADABLE"

    # -- filters -----------------------------------------------------------
    #: Inside a high-impact news blackout window.
    NEWS_BLACKOUT = "NEWS_BLACKOUT"
    #: No provider answered and the cache is stale. No data means no trade.
    NEWS_CALENDAR_UNAVAILABLE = "NEWS_CALENDAR_UNAVAILABLE"
    #: Unscheduled news. The calendar showed a clear window while the wires ran
    #: well above this instrument's own normal rate, or a market-wide story is
    #: out. Not a view on direction — a reason to keep still.
    HEADLINE_PRESSURE = "HEADLINE_PRESSURE"
    #: The headline feeds are dark and the operator chose to stand down without
    #: them. Off by default: the calendar's own fail-closed rule is untouched
    #: and still covers everything it always covered.
    HEADLINES_UNAVAILABLE = "HEADLINES_UNAVAILABLE"
    OUTSIDE_TRADABLE_SESSION = "OUTSIDE_TRADABLE_SESSION"
    #: The clock is inside a session this mode trades, and it is not a session
    #: that prices THIS instrument. `OUTSIDE_TRADABLE_SESSION` asks one global
    #: question for the whole catalogue; this asks whether the market that
    #: actually makes the price is open. NDX100 shorted at 00:51 UTC and
    #: EURCAD at 03:03 both passed the global check on Asia being active.
    HOME_SESSION_CLOSED = "HOME_SESSION_CLOSED"
    #: Daily rollover: spreads blow out and liquidity vanishes.
    ROLLOVER_WINDOW = "ROLLOVER_WINDOW"
    #: The evening wind-down before the rollover, when the book thins out and
    #: spreads widen. Wider than the rollover block, and it closes positions as
    #: well as refusing entries.
    EVENING_WIND_DOWN = "EVENING_WIND_DOWN"
    #: Friday close or Sunday reopen — thin books, gap risk over the weekend.
    WEEKEND_EDGE = "WEEKEND_EDGE"
    #: Not enough time left before this instrument's own forced exit for the
    #: trade to reach anything. The wind-down and the rollover only say "not
    #: inside this window" — they say nothing about the minute before it, which
    #: is how a position gets opened at 20:14 and flattened by us at 20:15.
    INSUFFICIENT_RUNWAY = "INSUFFICIENT_RUNWAY"
    #: The target sits at a distance this instrument does not cover often
    #: enough for the plan's own reward-to-risk to break even, or covers more
    #: readily in the opposite direction. Measured over its own recent history
    #: on the planning timeframe. Different from INSUFFICIENT_RUNWAY, which
    #: asks whether there is TIME; this asks whether the market goes there.
    TARGET_RARELY_REACHED = "TARGET_RARELY_REACHED"
    #: The market has gone to sleep: recent bar ranges are far below this
    #: instrument's own recent normal. A target measured in ATR is unreachable
    #: when the market has stopped producing ATR, and the spread is paid anyway.
    MARKET_TOO_QUIET = "MARKET_TOO_QUIET"
    MARKET_CLOSED = "MARKET_CLOSED"
    STALE_QUOTE = "STALE_QUOTE"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    #: The spread is too large a share of *this trade's own stop*, which is a
    #: different question from whether the spread is unusual. In the evening a
    #: spread can be entirely typical for the hour — so the hour-of-day baseline
    #: passes it — and still eat a third of a small stop. That trade starts a
    #: third of the way to being wrong and has to clear the spread twice before
    #: it earns anything.
    SPREAD_EATS_THE_STOP = "SPREAD_EATS_THE_STOP"
    #: A second position that would double the same underlying currency risk.
    CORRELATED_EXPOSURE = "CORRELATED_EXPOSURE"
    #: Two positions sharing a directional currency leg. Different from
    #: CORRELATED_EXPOSURE, which measures how returns have moved together:
    #: a GBP short is a GBP short by identity, and no measurement weakens it.
    CURRENCY_CONCENTRATION = "CURRENCY_CONCENTRATION"
    #: This instrument took money off us a moment ago and nothing has changed
    #: since. Live pair: GBPNZD short closed at a loss at 18:53:43, GBPNZD
    #: short reopened at 18:54:58, lost again. The read that produced the first
    #: was wrong, and seventy-five seconds is not new information.
    LOSS_COOLDOWN = "LOSS_COOLDOWN"
    #: Two positions leaning the same way on one asset class. The currency
    #: decomposition is structurally blind to these — an index quoted in its
    #: own currency has no currency leg — so FRA40 long beside UK100 long is
    #: one bet on equities with a second lot on it and nothing said so.
    SECTOR_CONCENTRATION = "SECTOR_CONCENTRATION"

    # -- exposure limits ---------------------------------------------------
    MAX_POSITIONS_REACHED = "MAX_POSITIONS_REACHED"
    MAX_TRADES_PER_DAY = "MAX_TRADES_PER_DAY"
    MAX_TRADES_PER_WEEK = "MAX_TRADES_PER_WEEK"
    #: A position in this symbol already exists. Adding to it would be
    #: averaging down or gridding, both of which are forbidden outright.
    POSITION_ALREADY_OPEN = "POSITION_ALREADY_OPEN"
    INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"
    MARGIN_ESTIMATE_FAILED = "MARGIN_ESTIMATE_FAILED"

    # -- loss limits -------------------------------------------------------
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT_HIT"
    WEEKLY_LOSS_LIMIT = "WEEKLY_LOSS_LIMIT_HIT"
    #: Drawdown from the equity peak. Flattens everything and halts until a
    #: human restarts the system.
    CIRCUIT_BREAKER = "MAX_DRAWDOWN_CIRCUIT_BREAKER"

    # -- control -----------------------------------------------------------
    KILL_SWITCH = "KILL_SWITCH_ENGAGED"
    SYSTEM_HALTED = "SYSTEM_HALTED"

    # -- observed, never acted on -------------------------------------------
    #: SECTION TWO, running as paper with no money behind it.
    #:
    #: Not a gate and not a refusal, which is why it sits apart from everything
    #: above. `drift_burst` cannot reach an order — it is absent from
    #: `live_enabled_modules`, so live mode zeroes its weight before the engine
    #: sees it. This label exists so the shadow recorder registers what it
    #: WOULD have done and the existing resolver settles that against real
    #: later prices. That is the only way to find out whether the statistic
    #: survives being computed on M1 bars instead of the paper's tick data.
    #:
    #: The scorecard reports it in its own section rather than beside the
    #: gates: a gate's blocked setups measure what REFUSING costs, and this
    #: measures what a detector nobody trades would have earned. Reading the
    #: second as the first would make it look like a filter is losing money.
    SECTION_2_OBSERVED = "SECTION_2_OBSERVED"

    @property
    def is_halt(self) -> bool:
        """True for reasons that stop the whole system, not just one trade.

        The distinction matters: a skipped setup is normal and expected, while
        a halt is an event that has to reach a human. A missing calendar counts
        as a halt: it stops every symbol, not just this one, and if it persists
        the system is silently doing nothing all day.
        """
        return self in _HALTING


_HALTING = frozenset(
    {
        Reason.NEWS_CALENDAR_UNAVAILABLE,
        Reason.DAILY_LOSS_LIMIT,
        Reason.WEEKLY_LOSS_LIMIT,
        Reason.CIRCUIT_BREAKER,
        Reason.KILL_SWITCH,
        Reason.SYSTEM_HALTED,
    }
)


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Verdict of one risk gate, with the numbers that produced it.

    `detail` is written for a human reading the weekly report; `reason` is what
    the queries group by. Both are always populated, including on approval —
    an approved trade needs its numbers recorded just as much as a rejected one.
    """

    approved: bool
    reason: Reason
    detail: str = ""

    @classmethod
    def allow(cls, detail: str = "") -> RiskDecision:
        return cls(approved=True, reason=Reason.OK, detail=detail)

    @classmethod
    def block(cls, reason: Reason, detail: str) -> RiskDecision:
        if reason is Reason.OK:
            raise ValueError("cannot block with reason OK")
        return cls(approved=False, reason=reason, detail=detail)

    def __bool__(self) -> bool:
        return self.approved

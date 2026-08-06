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
    OUTSIDE_TRADABLE_SESSION = "OUTSIDE_TRADABLE_SESSION"
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

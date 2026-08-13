"""Fail-closed LLM second opinions over bounded, structured market evidence."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import pandas as pd

from analysis.confluence import TradeIdea
from analysis.target_reach import break_even_rate
from config.schema import AIConfig
from core.types import MarketContext, Timeframe
from infra.logging import get_logger

log = get_logger(__name__)

# Claude Sonnet 5 and the Opus 4.7+ family reject `temperature`, `top_p` and
# `top_k` outright, and run adaptive thinking whenever `thinking` is omitted.
# Both facts matter here. Passing `temperature=0` returns HTTP 400, and because
# this adviser is fail-closed a 400 does not look like a bug — it looks like a
# veto, so the system simply never trades and nothing appears broken.
#
# Thinking tokens count against `max_tokens`. A budget sized for the JSON answer
# alone truncates the reply, which surfaces as stop_reason="max_tokens" and is
# likewise indistinguishable from a veto. The budget below is therefore far
# larger than the ~200-token verdict needs; the effort hint, not the ceiling, is
# what actually bounds the spend.
_MAX_TOKENS = 4_000

# A veto over a small, fully structured payload is a bounded judgement, not a
# multi-step reasoning problem. `medium` on Sonnet 5 is roughly Sonnet 4.6 at
# `high`, and keeps the round trip inside `ai.timeout_seconds` — a timeout is
# another silent veto.
_EFFORT = "medium"

_SCHEMA_NOTE = " Return only one JSON object matching the schema."


def _cached(instructions: str, ttl: str | None = None) -> list[dict[str, object]]:
    """The system prompt as one cacheable block.

    These instructions are byte-identical on every call and are the largest
    fixed part of the request — roughly 2,800 tokens for a review and 1,400 for
    a supervision — and every single call was paying full price to send them
    again. A cache read costs a tenth of that, and the break-even is two
    requests.

    The prefix must stay frozen for this to work at all: nothing here may
    interpolate a timestamp, a symbol or an account number, because a single
    changed byte invalidates the whole entry and leaves only the 1.25x write
    premium. The per-trade payload goes in `messages`, after this block, where
    it belongs.

    `ttl` selects the one-hour cache. Worth it only where calls are spaced
    further apart than five minutes but still regular — supervision runs every
    fifteen, so the default TTL would expire between every pair of calls and
    the write premium would be paid each time with no read to show for it.
    """
    block: dict[str, object] = {
        "type": "text",
        "text": instructions + _SCHEMA_NOTE,
        "cache_control": (
            {"type": "ephemeral"} if ttl is None else {"type": "ephemeral", "ttl": ttl}
        ),
    }
    return [block]


def _usage(message: Any) -> dict[str, int]:
    """Token counts for one call, for the spend ledger.

    Recorded rather than estimated. "Am I burning credit" was previously
    answerable only by opening the Anthropic console, and the number that
    matters most — how much of the prompt is being served from cache — is not
    visible anywhere else at all.

    Note that `input_tokens` is the *uncached remainder*, not the prompt size:
    the total is input + cache_creation + cache_read. Summing only the first
    would report a caching win that is really a measurement error.
    """
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    return {
        field: int(getattr(usage, field, 0) or 0)
        for field in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    }


@dataclass(frozen=True, slots=True)
class Advice:
    approved: bool
    confidence: float
    thesis: str
    risks: tuple[str, ...] = ()
    provider: str = "disabled"
    model: str = ""
    request_id: str = ""
    error: str = ""
    #: What the adviser actually said, before the confidence threshold was
    #: applied. `approved` is `said_yes and confidence >= minimum`, so the two
    #: differ whenever a setup was approved without enough conviction — and the
    #: raw answer was previously discarded, leaving the journal and the deck
    #: unable to tell "this is a bad trade" from "this is fine, I am just not
    #: sure enough". Those call for opposite responses: one is the system
    #: working, the other is a threshold that may be set too high.
    said_yes: bool = False
    #: The threshold `confidence` was compared against, recorded alongside it so
    #: a row remains readable after the setting changes.
    threshold: float = 0.0
    #: Token counts for this call. Carried on the verdict because that is what
    #: reaches the audit ledger; nothing else in the pipeline sees the response.
    usage: dict[str, int] = field(default_factory=dict)
    #: True when this verdict came from the review cache rather than the API.
    #:
    #: A replay is a real decision and belongs in the audit trail, but it cost
    #: nothing. Without this flag it carries the *original* call's token counts
    #: into the ledger, and the spend report bills them a second time: nine
    #: rows on the deck where five were paid, and a per-call figure computed
    #: over the wrong denominator. The operator then reads a cost they are not
    #: incurring and reaches for a fix that is not needed.
    replayed: bool = False
    #: Directional agreement and executable timing are separate. WAIT_RETEST
    #: means the thesis may be sound but this exact market order is too late;
    #: unlike a veto it must not poison the symbol/direction memory.
    entry_timing: str = "ENTER_NOW"
    retest_level: float | None = None
    #: Highest acceptable market entry for a LONG, lowest for a SHORT. Advisory
    #: only: deterministic fresh-price checks remain authoritative.
    entry_boundary: float | None = None
    chase_risk: str = ""

    @property
    def below_threshold(self) -> bool:
        """Approved on the merits, refused for want of conviction."""
        return self.said_yes and not self.approved and not self.waiting_for_retest

    @property
    def waiting_for_retest(self) -> bool:
        return self.entry_timing == "WAIT_RETEST"

    def safe_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScoutDecision:
    """One independent cross-market nomination, never an execution command."""

    action: str = "WAIT"
    symbol: str = ""
    confidence: float = 0.0
    thesis: str = ""
    counter_thesis: str = ""
    invalidation_price: float | None = None
    target_price: float | None = None
    patterns: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    wait_for: str = ""
    provider: str = "disabled"
    model: str = ""
    request_id: str = ""
    error: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def directional(self) -> bool:
        return self.action in {"LONG", "SHORT"} and bool(self.symbol)

    def safe_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Reflection:
    summary: str
    lessons: tuple[str, ...] = ()
    process_flags: tuple[str, ...] = ()
    provider: str = "disabled"
    model: str = ""
    request_id: str = ""
    error: str = ""

    def safe_dict(self) -> dict[str, object]:
        return asdict(self)


#: What a supervisor is allowed to do to a position that is already open,
#: ordered from least to most protective. Every one of these reduces exposure
#: or banks profit; there is deliberately no "widen the stop", no "add to the
#: position" and no "reverse" — see `Supervision.is_risk_reducing` for why that
#: is enforced rather than merely documented.
#:
#: The order is load-bearing: `ConsensusAdvisor.supervise` resolves disagreement
#: by taking the highest-indexed answer. A tightened stop still leaves full size
#: exposed to a gap, which is why banking part of the position outranks it.
SUPERVISION_ACTIONS = ("hold", "pull_target_in", "tighten_stop", "partial_close", "close")


@dataclass(frozen=True, slots=True)
class Supervision:
    """A verdict on one *open* position, restricted to de-risking moves.

    Pre-trade review answers "should this be opened". This answers "should this
    stay open, and on what terms" — the question a human actually spends their
    day on, and the one the system previously answered with nothing but a fixed
    break-even rule and an ATR trail.

    The action vocabulary is closed and every member reduces risk. That is the
    whole safety argument: an adviser that can be wrong is given a set of moves
    where being wrong costs an exit that was not needed, never an exposure that
    should not exist. `is_risk_reducing` re-checks the concrete prices against
    the live position before anything is sent, because the vocabulary alone
    cannot stop "tighten_stop" carrying a stop that is actually further away.
    """

    action: str = "hold"
    reason: str = ""
    confidence: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    close_fraction: float | None = None
    provider: str = "disabled"
    model: str = ""
    request_id: str = ""
    error: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    #: Explicitly separate the state of the original idea from the action. A
    #: profitable trade can have an intact thesis and still deserve protection;
    #: a losing trade can have an invalidated thesis and deserve an exit.
    thesis_state: str = "uncertain"
    urgency: str = "routine"
    evidence: tuple[str, ...] = ()
    #: The model may ask to see the trade sooner than the routine cadence. The
    #: runner clamps this to its configured cost/noise floor and ceiling.
    review_after_minutes: float | None = None

    def safe_dict(self) -> dict[str, object]:
        return asdict(self)

    def is_risk_reducing(
        self,
        *,
        direction_sign: int,
        current_sl: float,
        current_tp: float,
        price_now: float,
    ) -> bool:
        """Whether this verdict can only shrink exposure on this position.

        Called immediately before execution with the position as it stands now,
        not as it stood when the question was asked. A stop that was an
        improvement thirty seconds ago can be on the wrong side of price by the
        time the answer arrives, and sending it would either be rejected by the
        broker or — worse — accepted as a stop behind the market.
        """
        if self.action in {"hold", "close", "partial_close"}:
            return True
        if self.action == "tighten_stop":
            if self.stop_loss is None:
                return False
            # Closer to price in the direction of the trade, and not already
            # through it: a stop on the far side of the current price is an
            # instant market exit dressed up as a stop move.
            moved_up = (self.stop_loss - current_sl) * direction_sign > 0
            still_behind = (price_now - self.stop_loss) * direction_sign > 0
            return moved_up and still_behind
        if self.action == "pull_target_in":
            if self.take_profit is None:
                return False
            # Nearer than the existing target and still ahead of price. Pushing
            # a target further out is how a winning trade becomes a losing one,
            # so it is refused regardless of the argument attached to it.
            nearer = (current_tp - self.take_profit) * direction_sign > 0
            ahead = (self.take_profit - price_now) * direction_sign > 0
            return nearer and ahead
        return False


class Advisor(Protocol):
    def scout(self, market_state: Mapping[str, object]) -> ScoutDecision: ...

    def review(
        self,
        idea: TradeIdea,
        context: MarketContext,
        proposal: Mapping[str, object] | None = None,
        memory: Mapping[str, object] | None = None,
    ) -> Advice: ...

    def reflect(self, outcome: Mapping[str, object]) -> Reflection: ...

    def supervise(self, position_state: Mapping[str, object]) -> Supervision: ...


class DisabledAdvisor:
    def scout(self, _market_state: Mapping[str, object]) -> ScoutDecision:
        return ScoutDecision(thesis="AI advisory disabled", provider="disabled")

    def review(
        self,
        _idea: TradeIdea,
        _context: MarketContext,
        _proposal: Mapping[str, object] | None = None,
        _memory: Mapping[str, object] | None = None,
    ) -> Advice:
        return Advice(True, 1.0, "AI advisory disabled", provider="disabled")

    def reflect(self, _outcome: Mapping[str, object]) -> Reflection:
        return Reflection("AI advisory disabled", provider="disabled")

    def supervise(self, _position_state: Mapping[str, object]) -> Supervision:
        # "hold" and not "close": with the adviser switched off the mechanical
        # rules in PositionManager are the whole management policy, and they
        # are sound on their own. Closing here would mean disabling the adviser
        # silently liquidates the book.
        return Supervision("hold", "AI advisory disabled", provider="disabled")


class OpenAIAdvisor:
    def __init__(self, config: AIConfig) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=config.timeout_seconds)
        self.model = config.openai_model
        self.minimum = config.minimum_confidence

    def scout(self, market_state: Mapping[str, object]) -> ScoutDecision:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=_SCOUT_INSTRUCTIONS,
                input=_dumps({"market_scout": dict(market_state)}),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "market_scout",
                        "strict": True,
                        "schema": _SCOUT_SCHEMA,
                    }
                },
            )
            return _parse_scout(
                response.output_text,
                "openai",
                self.model,
                getattr(response, "_request_id", "") or "",
            )
        except Exception as exc:  # noqa: BLE001 - scouting is non-blocking
            return _failed_scout("openai", self.model, exc)

    def review(
        self,
        idea: TradeIdea,
        context: MarketContext,
        proposal: Mapping[str, object] | None = None,
        memory: Mapping[str, object] | None = None,
    ) -> Advice:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=_REVIEW_INSTRUCTIONS,
                input=_dumps(build_review_payload(idea, context, proposal, memory)),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "trade_review",
                        "strict": True,
                        "schema": _REVIEW_SCHEMA,
                    }
                },
            )
            return _parse_review(
                response.output_text,
                "openai",
                self.model,
                self.minimum,
                getattr(response, "_request_id", "") or "",
            )
        except Exception as exc:  # noqa: BLE001 - an unavailable adviser must veto, not crash
            return _failed_advice("openai", self.model, exc)

    def supervise(self, position_state: Mapping[str, object]) -> Supervision:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=_SUPERVISION_INSTRUCTIONS,
                input=_dumps({"open_position": dict(position_state)}),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "position_supervision",
                        "strict": True,
                        "schema": _SUPERVISION_SCHEMA,
                    }
                },
            )
            return _parse_supervision(
                response.output_text,
                "openai",
                self.model,
                getattr(response, "_request_id", "") or "",
            )
        except Exception as exc:  # noqa: BLE001 - an unreachable adviser must not touch the book
            return _failed_supervision("openai", self.model, _safe_error(exc))

    def reflect(self, outcome: Mapping[str, object]) -> Reflection:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=_REFLECTION_INSTRUCTIONS,
                input=_dumps(dict(outcome)),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "trade_reflection",
                        "strict": True,
                        "schema": _REFLECTION_SCHEMA,
                    }
                },
            )
            return _parse_reflection(
                response.output_text,
                "openai",
                self.model,
                getattr(response, "_request_id", "") or "",
            )
        except Exception as exc:  # noqa: BLE001 - reflection cannot affect already-closed risk
            return _failed_reflection("openai", self.model, exc)


class AnthropicAdvisor:
    def __init__(self, config: AIConfig) -> None:
        import anthropic

        if not config.anthropic_model:
            raise RuntimeError("ai.anthropic_model must be configured")
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            timeout=config.timeout_seconds,
            max_retries=1,
        )
        self.model = config.anthropic_model
        self.minimum = config.minimum_confidence

    def scout(self, market_state: Mapping[str, object]) -> ScoutDecision:
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                system=_cached(_SCOUT_INSTRUCTIONS),
                output_config={
                    "effort": _EFFORT,
                    "format": {
                        "type": "json_schema",
                        "schema": _anthropic_schema(_SCOUT_SCHEMA),
                    },
                },
                messages=[
                    {
                        "role": "user",
                        "content": _dumps(
                            {"schema": _SCOUT_SCHEMA, "market_scout": dict(market_state)}
                        ),
                    }
                ],
            )
            request_id = getattr(message, "_request_id", "") or ""
            stop_reason = getattr(message, "stop_reason", "") or "unknown"
            if stop_reason != "end_turn":
                return ScoutDecision(
                    thesis="Claude scout response did not finish normally",
                    provider="anthropic",
                    model=self.model,
                    request_id=request_id,
                    error=f"incomplete_response:{stop_reason}",
                )
            text = "".join(
                block.text for block in message.content if getattr(block, "type", "") == "text"
            )
            return _parse_scout(
                text,
                "anthropic",
                self.model,
                request_id,
                _usage(message),
            )
        except Exception as exc:  # noqa: BLE001 - scouting is non-blocking
            return _failed_scout("anthropic", self.model, exc)

    def review(
        self,
        idea: TradeIdea,
        context: MarketContext,
        proposal: Mapping[str, object] | None = None,
        memory: Mapping[str, object] | None = None,
    ) -> Advice:
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                system=_cached(_REVIEW_INSTRUCTIONS),
                output_config={
                    "effort": _EFFORT,
                    "format": {
                        "type": "json_schema",
                        "schema": _anthropic_schema(_REVIEW_SCHEMA),
                    },
                },
                messages=[
                    {
                        "role": "user",
                        "content": _dumps(
                            {
                                "schema": _REVIEW_SCHEMA,
                                "trade_candidate": build_review_payload(
                                    idea, context, proposal, memory
                                ),
                            }
                        ),
                    }
                ],
            )
            request_id = getattr(message, "_request_id", "") or ""
            stop_reason = getattr(message, "stop_reason", "") or "unknown"
            if stop_reason != "end_turn":
                return Advice(
                    False,
                    0.0,
                    "Claude response did not finish normally",
                    provider="anthropic",
                    model=self.model,
                    request_id=request_id,
                    # Naming the reason matters: "max_tokens" means raise the
                    # budget, "refusal" means the payload tripped a safeguard.
                    # A bare "incomplete_response" hides which one it was.
                    error=f"incomplete_response:{stop_reason}",
                )
            # Adaptive thinking is on by default, so `content` also carries
            # thinking blocks. Select on the block type rather than on the
            # presence of a `.text` attribute.
            text = "".join(
                block.text for block in message.content if getattr(block, "type", "") == "text"
            )
            return _parse_review(
                text,
                "anthropic",
                self.model,
                self.minimum,
                request_id,
                _usage(message),
            )
        except Exception as exc:  # noqa: BLE001 - API/auth/timeout/rate limit must all veto
            return _failed_advice("anthropic", self.model, exc)

    def reflect(self, outcome: Mapping[str, object]) -> Reflection:
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                system=_REFLECTION_INSTRUCTIONS
                + " Return only one JSON object matching the schema.",
                output_config={
                    "effort": _EFFORT,
                    "format": {
                        "type": "json_schema",
                        "schema": _anthropic_schema(_REFLECTION_SCHEMA),
                    },
                },
                messages=[
                    {
                        "role": "user",
                        "content": _dumps(
                            {"schema": _REFLECTION_SCHEMA, "closed_trade": dict(outcome)}
                        ),
                    }
                ],
            )
            request_id = getattr(message, "_request_id", "") or ""
            stop_reason = getattr(message, "stop_reason", "") or "unknown"
            if stop_reason != "end_turn":
                return Reflection(
                    "Claude reflection did not finish normally",
                    provider="anthropic",
                    model=self.model,
                    request_id=request_id,
                    error=f"incomplete_response:{stop_reason}",
                )
            # Adaptive thinking is on by default, so `content` also carries
            # thinking blocks. Select on the block type rather than on the
            # presence of a `.text` attribute.
            text = "".join(
                block.text for block in message.content if getattr(block, "type", "") == "text"
            )
            return _parse_reflection(text, "anthropic", self.model, request_id)
        except Exception as exc:  # noqa: BLE001 - trade is already closed; retain the failure only
            return _failed_reflection("anthropic", self.model, exc)

    def supervise(self, position_state: Mapping[str, object]) -> Supervision:
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                system=_cached(_SUPERVISION_INSTRUCTIONS, ttl="1h"),
                output_config={
                    "effort": _EFFORT,
                    "format": {
                        "type": "json_schema",
                        "schema": _anthropic_schema(_SUPERVISION_SCHEMA),
                    },
                },
                messages=[
                    {
                        "role": "user",
                        "content": _dumps(
                            {
                                "schema": _SUPERVISION_SCHEMA,
                                "open_position": dict(position_state),
                            }
                        ),
                    }
                ],
            )
            request_id = getattr(message, "_request_id", "") or ""
            stop_reason = getattr(message, "stop_reason", "") or "unknown"
            if stop_reason != "end_turn":
                return _failed_supervision(
                    "anthropic", self.model, f"incomplete_response:{stop_reason}"
                )
            text = "".join(
                block.text for block in message.content if getattr(block, "type", "") == "text"
            )
            return _parse_supervision(text, "anthropic", self.model, request_id, _usage(message))
        except Exception as exc:  # noqa: BLE001 - an unreachable adviser must not touch the book
            return _failed_supervision("anthropic", self.model, _safe_error(exc))


class ConsensusAdvisor:
    def __init__(self, advisors: list[Advisor]) -> None:
        self.advisors = advisors

    def scout(self, market_state: Mapping[str, object]) -> ScoutDecision:
        answers = [advisor.scout(market_state) for advisor in self.advisors]
        directional = [answer for answer in answers if answer.directional and not answer.error]
        if not directional:
            return ScoutDecision(
                thesis=" | ".join(answer.thesis for answer in answers if answer.thesis),
                provider="consensus",
                error=" | ".join(answer.error for answer in answers if answer.error),
            )
        sides = {(answer.symbol, answer.action) for answer in directional}
        if len(sides) != 1:
            return ScoutDecision(
                thesis="scouts disagree; no nomination promoted",
                counter_thesis=" | ".join(answer.thesis for answer in directional),
                provider="consensus",
            )
        strongest = max(directional, key=lambda answer: answer.confidence)
        return ScoutDecision(
            action=strongest.action,
            symbol=strongest.symbol,
            confidence=min(answer.confidence for answer in directional),
            thesis=" | ".join(answer.thesis for answer in directional),
            counter_thesis=" | ".join(answer.counter_thesis for answer in directional),
            invalidation_price=strongest.invalidation_price,
            target_price=strongest.target_price,
            patterns=tuple(pattern for answer in directional for pattern in answer.patterns),
            risks=tuple(risk for answer in directional for risk in answer.risks),
            wait_for=" | ".join(answer.wait_for for answer in directional if answer.wait_for),
            provider="consensus",
        )

    def review(
        self,
        idea: TradeIdea,
        context: MarketContext,
        proposal: Mapping[str, object] | None = None,
        memory: Mapping[str, object] | None = None,
    ) -> Advice:
        answers = [advisor.review(idea, context, proposal, memory) for advisor in self.advisors]
        approved = bool(answers) and all(answer.approved for answer in answers)
        confidence = min((answer.confidence for answer in answers), default=0.0)
        return Advice(
            approved,
            confidence,
            " | ".join(answer.thesis for answer in answers),
            tuple(risk for answer in answers for risk in answer.risks),
            "consensus",
            error=" | ".join(answer.error for answer in answers if answer.error),
        )

    def reflect(self, outcome: Mapping[str, object]) -> Reflection:
        answers = [advisor.reflect(outcome) for advisor in self.advisors]
        return Reflection(
            " | ".join(answer.summary for answer in answers),
            tuple(lesson for answer in answers for lesson in answer.lessons),
            tuple(flag for answer in answers for flag in answer.process_flags),
            "consensus",
            error=" | ".join(answer.error for answer in answers if answer.error),
        )

    def supervise(self, position_state: Mapping[str, object]) -> Supervision:
        """The most protective answer wins.

        Unanimity is the right rule for opening a trade, because there the
        cautious answer is "do not". Here every available action is already
        cautious, so requiring agreement would mean the *least* protective
        adviser decides — one saying "close" and one saying "hold" would hold.
        Taking the strongest de-risking answer keeps the fail-safe direction
        pointing the same way it does everywhere else in the system.
        """
        answers = [advisor.supervise(position_state) for advisor in self.advisors]
        if not answers:
            return Supervision("hold", "no advisers configured", provider="consensus")
        ranked = sorted(answers, key=lambda answer: SUPERVISION_ACTIONS.index(answer.action))
        strongest = ranked[-1]
        return Supervision(
            strongest.action,
            " | ".join(answer.reason for answer in answers if answer.reason),
            strongest.confidence,
            strongest.stop_loss,
            strongest.take_profit,
            strongest.close_fraction,
            "consensus",
            error=" | ".join(answer.error for answer in answers if answer.error),
            thesis_state=strongest.thesis_state,
            urgency=strongest.urgency,
            evidence=tuple(item for answer in answers for item in answer.evidence),
            review_after_minutes=strongest.review_after_minutes,
        )


def build_advisor(config: AIConfig) -> Advisor:
    if not config.enabled:
        return DisabledAdvisor()
    try:
        if config.provider == "openai":
            return OpenAIAdvisor(config)
        if config.provider == "anthropic":
            return AnthropicAdvisor(config)
        return ConsensusAdvisor([OpenAIAdvisor(config), AnthropicAdvisor(config)])
    except (ImportError, KeyError) as exc:
        missing = type(exc).__name__
        raise RuntimeError(
            f"AI enabled but provider dependency/credential is missing ({missing})"
        ) from exc


#: Closed bars sent per timeframe. Three was the original figure and it made
#: the review impossible: nobody can judge whether a target is reachable, or
#: whether price is running into a level, from three candles. The higher frames
#: need fewer bars to show structure; the lower ones are where the entry lives.
#:
#: Every timeframe still goes, deliberately — the adviser has to see the monthly
#: picture and the last half hour in one payload, and dropping a rung is how a
#: verdict ends up missing the trend it was standing in.
#:
#: The counts are another matter, and they were the single largest line in the
#: bill: 412 bars is roughly 11,000 tokens on every call, against a system
#: prompt of 2,800. They had been tuned for coverage without noticing that
#: coverage is what the *slower* rung is for. 80 M15 bars is twenty hours, which
#: H1 was already showing, which H4 was already showing. The lower timeframes
#: exist for recent detail and only need to reach back far enough to overlap the
#: rung above them.
#:
#: 260 bars now, about 3,700 tokens off every call, and the ladder still has no
#: gap: 30 minutes of M1 inside 3 hours of M5, inside 8 hours of M15, inside 2
#: days of H1, inside 7 days of H4, inside 40 days of D1, inside 140 of W1,
#: inside a year of MN1.
_BARS_SENT = {"MN1": 12, "W1": 20, "D1": 40, "H4": 42, "H1": 48, "M15": 32, "M5": 36, "M1": 30}
_BARS_SENT_DEFAULT = 40


#: What one row of `rows` means. Sent once per timeframe instead of repeating
#: five JSON keys on every bar.
_BAR_COLUMNS = "open,high,low,close,tick_volume"


def _chart(series: Any, count: int) -> dict[str, object]:
    """One timeframe as a compact table: the same bars, half the tokens.

    The obvious encoding — one JSON object per bar with named keys and an ISO
    timestamp — spends most of its bytes on the same eight strings repeated
    forty times. Measured on a 48-bar H1 block: 4,317 characters as objects
    against 2,157 as rows, and bars are 91% of the whole request. Nothing is
    dropped; the same numbers arrive at half the price.

    Per-bar timestamps go because they carry no information. `Series` is
    gap-checked and strictly increasing by construction, so the bars are
    contiguous at a known interval — given the timeframe and the last bar's
    close time, every other timestamp is arithmetic. The two ends are still
    sent, both because the model should not have to infer the window it is
    looking at and because a mismatch between them and `bars_sent` is a bug
    that ought to be visible.
    """
    frame = series.df
    recent = frame.tail(count)
    rows = [
        [
            round(float(row["open"]), 6),
            round(float(row["high"]), 6),
            round(float(row["low"]), 6),
            round(float(row["close"]), 6),
            int(row.get("tick_volume", 0)),
        ]
        for _, row in recent.iterrows()
    ]
    first = recent.index[0] if len(recent) else None
    return {
        "columns": _BAR_COLUMNS,
        "bar_interval": series.timeframe.value,
        "oldest_bar_opened": _iso(first),
        "last_closed_bar": series.last_bar_time.isoformat(),
        "rows": rows,
        "bars_sent": len(rows),
        "history_available": len(series),
        "atr14": round(_atr(frame), 6),
        "range_high": round(float(recent["high"].max()), 6) if len(recent) else None,
        "range_low": round(float(recent["low"].min()), 6) if len(recent) else None,
    }


def _iso(index: Any) -> str | None:
    if index is None:
        return None
    moment = index.to_pydatetime() if hasattr(index, "to_pydatetime") else index
    return moment.isoformat().replace("+00:00", "Z")


def _atr(frame: Any, period: int = 14) -> float:
    """Wilder-style ATR over the last `period` bars, in price units."""
    if len(frame) < 2:
        return 0.0
    previous = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = ranges.tail(period).mean()
    return 0.0 if pd.isna(value) else float(value)


def _reachability(frame: Any, distance: float, bars_ahead: int) -> dict[str, object]:
    """How often this market has actually travelled `distance` in `bars_ahead`.

    A target is not realistic because the risk-reward arithmetic says so; it is
    realistic if this instrument routinely moves that far in that time. The
    engine sets the target at twice the stop with no reference to whether the
    market ever goes there, which is how a share ends up with a target it needs
    six calendar days to reach. Measured from its own recent history, the
    reviewer can see that instead of guessing.
    """
    closes = frame["close"].to_numpy()
    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    windows = len(closes) - bars_ahead
    if windows <= 0 or distance <= 0:
        return {"windows_examined": 0}

    up = down = 0
    for start in range(windows):
        ahead = slice(start + 1, start + 1 + bars_ahead)
        if highs[ahead].max() - closes[start] >= distance:
            up += 1
        if closes[start] - lows[ahead].min() >= distance:
            down += 1
    return {
        "windows_examined": windows,
        "bars_ahead": bars_ahead,
        "moved_up_that_far_pct": round(100.0 * up / windows, 1),
        "moved_down_that_far_pct": round(100.0 * down / windows, 1),
    }


def _reach_reference(
    history: dict[str, object],
    direction_key: str,
    stop_distance: float,
    target_distance: float,
) -> dict[str, object]:
    """The bar a reach rate has to clear, and the noise on the measurement.

    Both numbers already existed — `analysis.target_reach` computes them for
    the engine's own gate — and neither was ever put in front of the reviewer,
    which left it comparing a percentage to nothing. See the call site for the
    live GBPCAD veto that cost.
    """
    windows = int(history.get("windows_examined") or 0)
    forward = history.get(direction_key)
    if windows <= 0 or not isinstance(forward, int | float) or stop_distance <= 0:
        return {}
    reference: dict[str, object] = {
        "break_even_reach_pct": round(break_even_rate(target_distance / stop_distance), 1),
        "reach_standard_error_pct": round(
            100.0 * math.sqrt(max(0.0, (forward / 100.0) * (1 - forward / 100.0)) / windows), 1
        ),
    }
    return reference


def _direction_vote(value: float, threshold: float) -> int:
    """Return -1/0/+1 only when a measured move clears its noise floor."""
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def _timeframe_evidence(series: Any, count: int, direction_sign: int) -> dict[str, object]:
    """Compact measurements over the exact closed bars sent to the reviewer.

    This deliberately does not predict. It saves Claude from having to perform
    arithmetic over hundreds of compact rows and, crucially, labels the one
    interpretive field as a deterministic read rather than a market fact.
    """
    recent = series.df.tail(count)
    if len(recent) < 2:
        return {
            "status": "insufficient_closed_bars",
            "closed_bars_measured": len(recent),
            "proposal_alignment": "unavailable",
        }

    closes = recent["close"].astype(float)
    atr = _atr(recent)
    last = recent.iloc[-1]
    slow_bars = min(10, len(recent) - 1)
    fast_bars = min(3, len(recent) - 1)
    ema_span = min(20, len(recent))
    ema = closes.ewm(span=ema_span, adjust=False).mean()
    ema_slope_bars = min(5, len(ema) - 1)
    window_high = float(recent["high"].max())
    window_low = float(recent["low"].min())
    window_range = window_high - window_low
    candle_range = float(last["high"] - last["low"])
    candle_body = float(last["close"] - last["open"])
    upper_wick = float(last["high"] - max(last["open"], last["close"]))
    lower_wick = float(min(last["open"], last["close"]) - last["low"])

    def in_atr(value: float) -> float | None:
        return round(value / atr, 3) if atr else None

    slow_drift = in_atr(float(closes.iloc[-1] - closes.iloc[-1 - slow_bars]))
    fast_drift = in_atr(float(closes.iloc[-1] - closes.iloc[-1 - fast_bars]))
    close_vs_ema = in_atr(float(closes.iloc[-1] - ema.iloc[-1]))
    ema_slope = in_atr(float(ema.iloc[-1] - ema.iloc[-1 - ema_slope_bars]))
    measurements = [
        (slow_drift, 0.15),
        (fast_drift, 0.08),
        (close_vs_ema, 0.10),
        (ema_slope, 0.05),
    ]
    votes = [
        _direction_vote(value, threshold) for value, threshold in measurements if value is not None
    ]
    positive = votes.count(1)
    negative = votes.count(-1)
    if positive >= 2 and positive > negative:
        lean = "up"
        lean_sign = 1
    elif negative >= 2 and negative > positive:
        lean = "down"
        lean_sign = -1
    else:
        lean = "mixed_or_flat"
        lean_sign = 0
    if not direction_sign or not lean_sign:
        alignment = "mixed_or_neutral"
    elif lean_sign == direction_sign:
        alignment = "supports_proposal"
    else:
        alignment = "opposes_proposal"

    volume = recent.get("tick_volume")
    median_volume = float(volume.tail(20).median()) if volume is not None else 0.0
    return {
        "status": "measured",
        "closed_bars_measured": len(recent),
        "last_close": round(float(closes.iloc[-1]), 6),
        "atr14": round(atr, 6),
        "slow_drift": {"bars": slow_bars, "atr": slow_drift},
        "fast_drift": {"bars": fast_bars, "atr": fast_drift},
        "ema": {
            "span": ema_span,
            "close_distance_atr": close_vs_ema,
            "slope_bars": ema_slope_bars,
            "slope_atr": ema_slope,
        },
        "close_location_in_sent_range_pct": (
            round(100.0 * (float(closes.iloc[-1]) - window_low) / window_range, 1)
            if window_range
            else None
        ),
        "last_candle": {
            "body_atr": in_atr(candle_body),
            "range_atr": in_atr(candle_range),
            "upper_wick_atr": in_atr(upper_wick),
            "lower_wick_atr": in_atr(lower_wick),
        },
        "last_tick_volume_vs_20bar_median": (
            round(float(last.get("tick_volume", 0)) / median_volume, 2) if median_volume else None
        ),
        "deterministic_lean": lean,
        "proposal_alignment": alignment,
    }


def _evidence_brief(idea: TradeIdea, context: MarketContext) -> dict[str, object]:
    """Facts Claude can verify against the exact compact bars in the payload."""
    sign = int(idea.direction) if idea.direction is not None else 0
    measured = {
        timeframe.value: _timeframe_evidence(
            series,
            _BARS_SENT.get(timeframe.value, _BARS_SENT_DEFAULT),
            sign,
        )
        for timeframe, series in context.series.items()
    }
    supporting = [
        timeframe
        for timeframe, facts in measured.items()
        if facts.get("proposal_alignment") == "supports_proposal"
    ]
    opposing = [
        timeframe
        for timeframe, facts in measured.items()
        if facts.get("proposal_alignment") == "opposes_proposal"
    ]
    neutral = [
        timeframe
        for timeframe, facts in measured.items()
        if facts.get("proposal_alignment") in {"mixed_or_neutral", "unavailable"}
    ]
    return {
        "provenance": {
            "bar_source": "the exact closed OHLCV rows in timeframes below",
            "calculation": "deterministic local arithmetic; no LLM and no imputation",
            "live_tick_separate": True,
            "warning": (
                "deterministic_lean is a labelled summary of measurements, not a forecast; "
                "Claude must verify it against the rows and may disagree"
            ),
            "alignment_method": (
                "descriptive vote over 10-bar drift, 3-bar drift, close-versus-EMA and EMA "
                "slope; it is an engine inference, not a prediction"
            ),
        },
        "trade_question": {
            "direction": idea.direction.name if idea.direction else None,
            "setup_family": idea.setup_family,
            "horizon": idea.horizon,
            "planning_timeframe": idea.planning_timeframe,
            "expected_horizon_minutes": idea.expected_horizon_minutes,
        },
        "directional_balance": {
            "supports_proposal": supporting,
            "opposes_proposal": opposing,
            "mixed_or_unavailable": neutral,
            "rule": (
                "Higher timeframes are context for short trades, not automatic vetoes; judge their "
                "importance against the stated planning timeframe and horizon."
            ),
        },
        "timeframe_measurements": measured,
    }


def build_review_payload(
    idea: TradeIdea,
    context: MarketContext,
    proposal: Mapping[str, object] | None,
    memory: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the exact secret-free trade candidate sent to an AI reviewer.

    Carries enough of each chart to actually be read. The original three bars
    per timeframe left the reviewer nothing to work from but the engine's own
    summary, so every answer restated the module's reasoning back — it could
    not see whether price was running into a level, or whether the target was
    somewhere this market ever goes.
    """
    timeframes: dict[str, object] = {
        timeframe.value: _chart(series, _BARS_SENT.get(timeframe.value, _BARS_SENT_DEFAULT))
        for timeframe, series in context.series.items()
    }

    tick = None
    if context.tick is not None:
        tick = {
            "time": context.tick.time.isoformat(),
            "bid": context.tick.bid,
            "ask": context.tick.ask,
            "spread": context.tick.spread,
        }

    # Is this target somewhere the market goes *within this plan's horizon*?
    # An M15 scalp and an H1 swing cannot honestly share a 24-H1-bar yardstick.
    target_check: dict[str, object] = {}
    try:
        planning_timeframe = Timeframe.parse(idea.planning_timeframe)
    except ValueError:
        planning_timeframe = Timeframe.H1
    signal = context.series.get(planning_timeframe)
    if signal is not None and idea.entry:
        atr = _atr(signal.df)
        timeframe_minutes = max(1, int(planning_timeframe.duration.total_seconds() / 60))
        bars_ahead = max(1, math.ceil(idea.expected_horizon_minutes / timeframe_minutes))
        stop_distance = abs(idea.entry - idea.stop_loss) if idea.stop_loss else 0.0
        target_distance = abs(idea.take_profit - idea.entry) if idea.take_profit else 0.0
        history = _reachability(signal.df.tail(400), target_distance, bars_ahead=bars_ahead)
        direction_key = (
            "moved_up_that_far_pct"
            if idea.direction is not None and int(idea.direction) > 0
            else "moved_down_that_far_pct"
        )
        target_check = {
            "timeframe": planning_timeframe.value,
            "expected_horizon_minutes": idea.expected_horizon_minutes,
            "bars_ahead": bars_ahead,
            "atr14": round(atr, 6),
            "stop_distance": round(stop_distance, 6),
            "target_distance": round(target_distance, 6),
            "stop_in_atr": round(stop_distance / atr, 2) if atr else None,
            "target_in_atr": round(target_distance / atr, 2) if atr else None,
            "spread_as_pct_of_stop": (
                round(100.0 * context.tick.spread / stop_distance, 1)
                if context.tick is not None and stop_distance
                else None
            ),
            "history": history,
            "proposed_direction_reach_pct": history.get(direction_key),
            # The bar that percentage has to clear, and the noise on it.
            #
            # Both were computed and neither was ever sent. On a live GBPCAD
            # SHORT the reviewer read 43.1% with no reference point, called it
            # "only 43.1%, barely above the 34.6% up-rate ... closer to a coin
            # flip", and vetoed. Against the actual bar — 33.3% for a 2R plan —
            # it cleared by 9.8 points, and its 8.5-point lead over the other
            # direction is 3.4 standard errors. It was not close to a coin flip
            # in either sense. The reviewer was eyeballing a percentage with
            # nothing to compare it to, which is not a judgement anyone can
            # make, and the engine had both numbers in hand before it asked.
            **_reach_reference(history, direction_key, stop_distance, target_distance),
            "note": (
                "history measures every available rolling window on the planning timeframe "
                "using this proposal's expected horizon. It reports how often price travelled "
                "at least the target distance, not how often this strategy would have won. "
                "Compare proposed_direction_reach_pct against break_even_reach_pct, not "
                "against the opposite direction: below the break-even figure the plan cannot "
                "pay for itself whatever the other side does. reach_standard_error_pct is the "
                "noise on these percentages — a gap smaller than it is not a difference."
            ),
        }
    elif signal is None:
        target_check = {
            "timeframe": planning_timeframe.value,
            "status": "planning_timeframe_unavailable",
        }

    payload: dict[str, object] = {
        "symbol": idea.symbol,
        "direction": idea.direction.name if idea.direction else None,
        "score": idea.score,
        "confidence": idea.confidence,
        "entry": idea.entry,
        "stop_loss": idea.stop_loss,
        "take_profit": idea.take_profit,
        "target_realism": target_check,
        "evidence_brief": _evidence_brief(idea, context),
        "modules": [asdict(signal) for signal in idea.signals],
        "timeframes": timeframes,
        "decision_tick": tick,
        "executable_proposal": dict(proposal or {}),
        "rule": "Veto or approve only. Never change size, stop, target, risk, or hard filters.",
    }
    # Only present once the account has actually taught it something. An empty
    # "here is what you have learned" block is worse than none: it invites the
    # reviewer to invent significance from nothing.
    if memory is not None:
        payload["learned_so_far"] = memory
    return payload


def build_supervision_payload(
    position: Any,
    context: MarketContext,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the secret-free state of one open position for the supervisor.

    Same chart depth as the pre-trade review, for the same reason: the question
    "has the thesis broken" cannot be answered from a P&L number. What differs
    is the framing — the numbers here are all relative to the position (R
    multiple, age, distance to stop and target in ATR) because that is what the
    decision turns on, not the absolute price.
    """
    timeframes: dict[str, object] = {
        timeframe.value: _chart(series, _BARS_SENT.get(timeframe.value, _BARS_SENT_DEFAULT))
        for timeframe, series in context.series.items()
    }

    sign = int(position.direction)
    price = 0.0
    if context.tick is not None:
        price = context.tick.bid if sign > 0 else context.tick.ask
    # R must stay anchored to the risk accepted at entry. Once the mechanical
    # layer moves the live stop to break-even, measuring against that stop makes
    # the denominator collapse and turns an ordinary move into a fictitious
    # 20R gain. The original plan is journal-backed and survives restarts.
    supplied = dict(extra or {})
    trade_record = supplied.get("trade_record")
    trade_record = trade_record if isinstance(trade_record, Mapping) else {}
    original_plan = trade_record.get("original_plan")
    original_plan = original_plan if isinstance(original_plan, Mapping) else {}
    try:
        original_stop = float(original_plan.get("stop_loss", position.sl) or position.sl)
    except (TypeError, ValueError):
        original_stop = float(position.sl)
    risk = abs(position.price_open - original_stop) if original_stop else 0.0
    r_now = ((price - position.price_open) * sign / risk) if (risk and price) else None
    signal = context.series.get(Timeframe.H1)
    atr = _atr(signal.df) if signal is not None else 0.0

    try:
        peak_r = float(supplied.get("peak_r", 0.0) or 0.0)
    except (TypeError, ValueError):
        peak_r = 0.0
    giveback_r = max(0.0, peak_r - (r_now or 0.0)) if peak_r > 0 else 0.0
    giveback_fraction = giveback_r / peak_r if peak_r > 0 else 0.0
    # What the position is worth in money, at the three prices that matter.
    #
    # The payload used to carry `unrealised_money` and the account currency and
    # nothing else, so the reviewer was told "you are 0.76 up" with no way to
    # know whether that is a real result or a rounding error. On a hundred-euro
    # account it is most of a good day; on ten thousand it is noise. Judging it
    # needs the size of the account, and judging whether to keep waiting needs
    # to know what is still on the table against what is already in hand.
    money = position.profit + position.swap
    equity = float((extra or {}).get("account_equity") or 0.0)
    per_unit = None
    moved = (price - position.price_open) * sign
    if price and abs(moved) > 1e-12:
        per_unit = money / moved
    to_target = (
        round(abs(position.tp - price) * per_unit, 2)
        if per_unit and position.tp and price
        else None
    )
    at_stop = (
        round((position.sl - position.price_open) * sign * per_unit, 2)
        if per_unit and position.sl
        else None
    )

    return {
        "symbol": position.symbol,
        "direction": position.direction.name,
        "ticket": position.ticket,
        "volume_lots": position.volume,
        "entry_price": position.price_open,
        "current_stop": position.sl,
        "current_target": position.tp,
        "price_now": price,
        "unrealised_money": round(money, 2),
        #: The three numbers the "should I take it" question actually turns on.
        "account_equity": round(equity, 2) if equity else None,
        "unrealised_pct_of_account": (round(money / equity * 100, 2) if equity else None),
        "money_still_to_win_if_target_hit": to_target,
        "money_if_the_current_stop_is_hit": at_stop,
        "unrealised_r": round(r_now, 2) if r_now is not None else None,
        "initial_risk_distance": round(risk, 6),
        "original_stop": original_stop,
        "peak_unrealised_r": round(peak_r, 2),
        "profit_given_back_r": round(giveback_r, 2),
        "profit_given_back_fraction": round(giveback_fraction, 3),
        "spread_as_fraction_of_initial_risk": (
            round(context.tick.spread / risk, 3) if context.tick is not None and risk else None
        ),
        "distance_to_stop_in_atr": (
            round(abs(price - position.sl) / atr, 2) if atr and price and position.sl else None
        ),
        "distance_to_target_in_atr": (
            round(abs(position.tp - price) / atr, 2) if atr and price and position.tp else None
        ),
        "opened_at": position.opened_at.isoformat(),
        "age_hours": round((context.now - position.opened_at).total_seconds() / 3600.0, 2),
        "h1_atr14": round(atr, 6),
        "timeframes": timeframes,
        "context": supplied,
        "rule": (
            "You may only hold, tighten the stop, pull the target in, close part, or close all. "
            "Widening a stop, extending a target, adding size and reversing are all refused."
        ),
    }


def _parse_scout(
    text: str,
    provider: str,
    model: str,
    request_id: str,
    usage: dict[str, int] | None = None,
) -> ScoutDecision:
    try:
        payload = json.loads(text.strip())
        action = str(payload["action"]).upper().strip()
        if action not in {"WAIT", "LONG", "SHORT"}:
            raise ValueError("invalid scout action")  # noqa: TRY301
        symbol = str(payload["symbol"]).strip()
        confidence = float(payload["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence outside [0, 1]")  # noqa: TRY301
        thesis = str(payload["thesis"]).strip()
        counter = str(payload["counter_thesis"]).strip()
        patterns = payload["patterns"]
        risks = payload["risks"]
        wait_for = str(payload["wait_for"]).strip()
        if not isinstance(patterns, list) or not isinstance(risks, list):
            raise TypeError("patterns/risks must be arrays")  # noqa: TRY301
        if action != "WAIT" and (not symbol or not thesis or not counter):
            raise ValueError("directional scout decision needs symbol and both theses")  # noqa: TRY301
        if action == "WAIT":
            symbol = ""
        return ScoutDecision(
            action=action,
            symbol=symbol,
            confidence=confidence,
            thesis=thesis,
            counter_thesis=counter,
            invalidation_price=_optional_float(payload.get("invalidation_price")),
            target_price=_optional_float(payload.get("target_price")),
            patterns=tuple(str(item) for item in patterns),
            risks=tuple(str(item) for item in risks),
            wait_for=wait_for,
            provider=provider,
            model=model,
            request_id=request_id,
            usage=dict(usage or {}),
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return ScoutDecision(
            thesis=f"Invalid {provider} market-scout response; ignored",
            provider=provider,
            model=model,
            request_id=request_id,
            error="invalid_response",
        )


def _parse_review(
    text: str,
    provider: str,
    model: str,
    minimum: float,
    request_id: str,
    usage: dict[str, int] | None = None,
) -> Advice:
    try:
        payload = json.loads(text.strip())
        approve = payload["approve"]
        if not isinstance(approve, bool):
            raise TypeError("approve must be boolean")  # noqa: TRY301
        confidence = float(payload["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence outside [0, 1]")  # noqa: TRY301
        thesis = str(payload["thesis"]).strip()
        risks = payload["risks"]
        if not thesis or not isinstance(risks, list):
            raise TypeError("thesis/risks have invalid types")  # noqa: TRY301
        entry_timing = (
            str(payload.get("entry_timing", "ENTER_NOW" if approve else "VETO")).upper().strip()
        )
        if entry_timing not in {"ENTER_NOW", "WAIT_RETEST", "VETO"}:
            raise ValueError("invalid entry_timing")  # noqa: TRY301
        # `entry_timing` carries the more precise decision. Be tolerant when a
        # provider interprets the older `approve` boolean as "send an order
        # now" and therefore returns false alongside WAIT_RETEST: that is still
        # a temporary timing wait, never a thesis veto worth memorising.
        if entry_timing == "WAIT_RETEST":
            approve = True
        elif entry_timing == "VETO" or not approve:
            approve = False
            entry_timing = "VETO"
        chase_risk = str(payload.get("chase_risk", "")).strip()
        return Advice(
            approve and entry_timing == "ENTER_NOW" and confidence >= minimum,
            confidence,
            thesis,
            tuple(str(item) for item in risks),
            provider,
            model,
            request_id,
            said_yes=approve,
            threshold=minimum,
            usage=dict(usage or {}),
            entry_timing=entry_timing,
            retest_level=_optional_float(payload.get("retest_level")),
            entry_boundary=_optional_float(payload.get("entry_boundary")),
            chase_risk=chase_risk,
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return Advice(
            False,
            0.0,
            f"Invalid {provider} review; trade vetoed",
            provider=provider,
            model=model,
            request_id=request_id,
            error="invalid_response",
        )


def _parse_reflection(text: str, provider: str, model: str, request_id: str) -> Reflection:
    try:
        payload = json.loads(text.strip())
        summary = str(payload["summary"]).strip()
        lessons = payload["lessons"]
        process_flags = payload["process_flags"]
        if not summary or not isinstance(lessons, list) or not isinstance(process_flags, list):
            raise TypeError("reflection fields have invalid types")  # noqa: TRY301
        return Reflection(
            summary,
            tuple(str(item) for item in lessons),
            tuple(str(item) for item in process_flags),
            provider,
            model,
            request_id,
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return Reflection(
            f"Invalid {provider} reflection",
            provider=provider,
            model=model,
            request_id=request_id,
            error="invalid_response",
        )


def _parse_supervision(
    text: str,
    provider: str,
    model: str,
    request_id: str,
    usage: dict[str, int] | None = None,
) -> Supervision:
    try:
        payload = json.loads(text.strip())
        action = str(payload["action"]).strip()
        if action not in SUPERVISION_ACTIONS:
            raise ValueError(f"unknown action {action!r}")  # noqa: TRY301
        confidence = float(payload["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence outside [0, 1]")  # noqa: TRY301
        reason = str(payload["reason"]).strip()
        if not reason:
            raise ValueError("reason is empty")  # noqa: TRY301
        fraction = _optional_float(payload.get("close_fraction"))
        if action == "partial_close" and not (fraction and 0.0 < fraction < 1.0):
            # A partial close without a usable fraction is not a partial close.
            # Downgrading to "hold" rather than guessing keeps a malformed reply
            # from silently becoming a full exit.
            raise ValueError("partial_close needs a fraction in (0, 1)")  # noqa: TRY301
        thesis_state = str(payload.get("thesis_state", "uncertain")).strip().lower()
        if thesis_state not in {"intact", "weakened", "invalidated", "uncertain"}:
            raise ValueError("unknown thesis state")  # noqa: TRY301
        urgency = str(payload.get("urgency", "routine")).strip().lower()
        if urgency not in {"routine", "next_close", "immediate"}:
            raise ValueError("unknown urgency")  # noqa: TRY301
        raw_evidence = payload.get("evidence", ())
        if not isinstance(raw_evidence, (list, tuple)):
            raise TypeError("evidence is not a list")  # noqa: TRY301
        evidence = tuple(str(item).strip() for item in raw_evidence if str(item).strip())
        if len(evidence) > 6:
            raise ValueError("too many evidence items")  # noqa: TRY301
        review_after = _optional_float(payload.get("review_after_minutes"))
        if review_after is not None and not 0.25 <= review_after <= 240.0:
            raise ValueError("review cadence outside supported range")  # noqa: TRY301
        return Supervision(
            action,
            reason,
            confidence,
            _optional_float(payload.get("stop_loss")),
            _optional_float(payload.get("take_profit")),
            fraction,
            provider,
            model,
            request_id,
            usage=dict(usage or {}),
            thesis_state=thesis_state,
            urgency=urgency,
            evidence=evidence,
            review_after_minutes=review_after,
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return _failed_supervision(provider, model, "invalid_response", request_id)


def _optional_float(value: object) -> float | None:
    """Coerce a nullable numeric field, treating unusable values as absent.

    The schema permits null so the model can say "this field does not apply to
    my action". A zero means the same thing here — MT5 uses 0.0 for "no stop"
    and "no target" — so it is folded into absent rather than executed as a
    price.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if number == 0.0 else number


def _failed_supervision(provider: str, model: str, error: str, request_id: str = "") -> Supervision:
    """Fail to `hold` — the opposite direction from the pre-trade adviser.

    `review` fails closed to a veto because there the safe answer is "do not
    open". Here the position already exists and the mechanical rules in
    PositionManager are still running on it, so the safe answer is "change
    nothing". Failing to `close` would mean an expired API key liquidates the
    book, and failing to any price-bearing action would mean acting on a number
    that was never received.
    """
    _log_advisory_failure("position_supervision", provider, model, error)
    return Supervision(
        "hold",
        f"{provider} supervision unavailable; mechanical management continues",
        provider=provider,
        model=model,
        request_id=request_id,
        error=error,
    )


def _failed_advice(provider: str, model: str, exc: Exception) -> Advice:
    error = _safe_error(exc)
    _log_advisory_failure("pretrade_review", provider, model, error)
    return Advice(
        False,
        0.0,
        f"{provider} unavailable; trade vetoed",
        provider=provider,
        model=model,
        error=error,
    )


def _failed_reflection(provider: str, model: str, exc: Exception) -> Reflection:
    error = _safe_error(exc)
    _log_advisory_failure("posttrade_reflection", provider, model, error)
    return Reflection(
        f"{provider} reflection unavailable",
        provider=provider,
        model=model,
        error=error,
    )


def _failed_scout(provider: str, model: str, exc: Exception) -> ScoutDecision:
    error = _safe_error(exc)
    _log_advisory_failure("market_scout", provider, model, error)
    return ScoutDecision(
        thesis=f"{provider} market scout unavailable; deterministic pipeline continues",
        provider=provider,
        model=model,
        error=error,
    )


def _log_advisory_failure(operation: str, provider: str, model: str, error: str) -> None:
    """Make an API failure identifiable without ever printing its request."""
    log.warning(
        "%s failed: %s",
        operation,
        error,
        extra={
            "event": "ai_advisory_failed",
            "operation": operation,
            "provider": provider,
            "model": model,
            "safe_error": error,
        },
    )


def _safe_error(exc: Exception) -> str:
    """Return diagnostics without serialising request payloads or headers."""
    status = getattr(exc, "status_code", None)
    suffix = f":http_{status}" if isinstance(status, int) else ""
    return f"{type(exc).__name__}{suffix}{_request_shape_detail(exc, status)}"


def _request_shape_detail(exc: Exception, status: object) -> str:
    """Return the API's own message for request-shape errors, and nothing else.

    A fail-closed adviser turns every failure into a veto, so a bare
    `BadRequestError:http_400` is indistinguishable from a considered "no" and
    the system silently stops trading. For 400 and 404 the API replies with the
    name of the offending field or model, which is exactly what is needed to fix
    it and contains no credential — the key travels in a header, never the body.

    Deliberately limited to those two statuses. A 401, 403, 429 or 5xx is fully
    described by its status, and their messages can carry organisation detail
    that has no business in a trade journal.
    """
    if status not in {400, 404}:
        return ""
    body = getattr(exc, "body", None)
    message = ""
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping):
            message = str(error.get("message", ""))
    if not message:
        return ""
    return ":" + " ".join(message.split())[:300]


def _dumps(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), default=str, ensure_ascii=False)


def _anthropic_schema(value: Any) -> Any:
    """Remove unsupported raw-schema constraints; local parsing enforces them.

    The Anthropic Python helper does the same transformation when using its
    typed parse API. This project sends raw schemas because provider outputs
    share one parser, so the transformation has to happen here as well.
    """
    if isinstance(value, dict):
        return {
            key: _anthropic_schema(item)
            for key, item in value.items()
            if key
            not in {
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "multipleOf",
                "minLength",
                "maxLength",
                "pattern",
                "maxItems",
                "uniqueItems",
                "minProperties",
                "maxProperties",
            }
        }
    if isinstance(value, list):
        return [_anthropic_schema(item) for item in value]
    return value


_REVIEW_INSTRUCTIONS = """You are the second of two independent opinions on a small real-money
trade. The first is a deterministic analysis engine; you are the other. A trade is taken only if
you also agree, so a veto is free and an approval is not — but a reviewer that never approves is
not a second opinion, it is an off switch.

HOW THE ENGINE WORKS, so you judge it against the right standard. Five modules score every market
and most score zero on any given one, because each looks for a different market state:
market_structure wants a break of structure, liquidity_sweep wants a failed break that closes back
inside, trend_momentum wants aligned EMAs. A market is rarely two of those at once. A zero from a
module is that module saying "this is not my setup" — it is not evidence against the trade, and it
is not disagreement. One module firing cleanly is the normal shape of a real signal, and the
operator has accepted single-module signals by design. Do not veto on the count of contributing
modules, or because a score is not corroborated by modules that look for something else entirely.

YOU HAVE THE CHARTS. Every timeframe from the weekly down to the one-minute arrives as closed
OHLC bars — dozens per frame, not a summary — with each frame's ATR and its high and low over
that window. Each frame is a table: `columns` names the fields, and `rows` holds one array per
bar in that order, oldest first. Bars are contiguous at `bar_interval`, so `oldest_bar_opened`
and `last_closed_bar` fix the window and every bar in between follows from the interval. Read
them. The higher frames say whether there is a trend and where the structure
is; H1 and M15 say whether the entry sits at a sensible place in it; M5 and M1 say whether price
is moving toward the entry or away from it right now. An answer that could have been written from
the module scores alone means the bars went unread.

YOU ALSO HAVE A MEASURED EVIDENCE BRIEF. `evidence_brief.timeframe_measurements` is deterministic
arithmetic over those exact same closed rows: drift in ATR, location inside the sent range,
EMA distance and slope, candle anatomy and relative tick volume. It exists to prevent arithmetic
mistakes and repetitive paraphrasing; it is not a second signal and its `deterministic_lean` is
explicitly an inference, not a forecast. Verify it against the rows. Keep three categories apart:
measured fact, engine inference, and unavailable data. Never turn one into another and never invent
a pattern, event, level or probability that is not in the payload.

JUDGE THE STOP AND THE TARGET AS PLACEMENTS. `target_realism` gives you the stop and target in
ATR, the spread as a percentage of the stop, and — measured from this instrument's own recent
history — how often it has actually travelled the target distance within THIS proposal's stated
horizon on THIS proposal's planning timeframe. `proposed_direction_reach_pct` is a base-rate travel
measurement, not a strategy win rate.
Judge it against `break_even_reach_pct`, which is the rate this plan's own reward-to-risk needs to
break even, and NOT against the opposite direction's rate — a 2R plan needs 33%, so 43% clears it
comfortably even where the other direction sits at 35%. Below the break-even figure the target is
decorative whatever the other side does. `reach_standard_error_pct` is the noise on these numbers:
a gap smaller than it is not a difference, and calling one a coin flip when it leads by several
times its own error is a misreading, not caution.
Veto a stop sitting inside ordinary noise, a stop or target on the wrong side of an obvious level
in the bars you were given, and a spread eating a large share of the stop.

WHAT TO ACTUALLY JUDGE. Read the closed bars yourself and check the proposal against them. Veto
when the bars CONTRADICT the claim — a long whose lower timeframes are selling into the entry, a
"trend" that is a range, a stop sitting where price has already traded through, a target with
structure in the way, a quote too old for the level, a rationale that only restates its own
indicator. Approve when the bars SUPPORT the direction and the stop and target are sensibly
placed for it.

ENTRY DIRECTION IS NOT ENTRY TIMING. Answer whether you would place a MARKET ORDER at the supplied
entry now, not merely whether the market may eventually move that way. `ENTER_NOW` means both the
direction and this price are sound. `WAIT_RETEST` means the directional thesis remains plausible
but the price is stretched, the pullback is still active, or a breakout needs to hold/retest.
`VETO` means the underlying trade is not sound. A WAIT is not a softer veto and must name the
concrete price behaviour that would make the entry timely.

For schema consistency set `approve=true` for both `ENTER_NOW` and `WAIT_RETEST`: in the latter you
approve the directional thesis but explicitly do not approve a market order now. Set
`approve=false` only with `VETO`. Jarvis treats `entry_timing` as the execution instruction.

Do not call a large direction-aligned candle automatic confirmation. A 2.5 ATR M5 body, a close at
the extreme of its range, or price far from its short EMA is first evidence of chase/exhaustion;
it needs a specific reason why buying or selling *after* that move still offers favourable entry
asymmetry. Higher-timeframe trend decides directional context, never permission to buy the local
top or sell the local bottom. Rank says which candidate arrived first, not that its current price
has edge. A target reach percentage is travel frequency, not win probability and not proof that
the target is reached before the stop.

THE BAR IS "SUPPORTED", NOT "PERFECT". This is the failure mode to avoid, and it is not
hypothetical: an earlier version of this prompt produced seventy-four consecutive vetoes, every
one citing something real and none of it disqualifying — resistance somewhere above the target, a
pullback in the last three bars, a range that was slightly tight. Every trade that has ever been
taken had objections like those available. A reviewer who lists them and concludes "veto" is not
being careful, it is refusing to answer the question, and a gate that never opens is the same as
having no strategy at all.

So: name a risk only if it would make you decline the trade, and if it would not, approve and
mention it as a risk. Do not treat the absence of confirmation as evidence against. Do not
require the setup to be the best available, only sound. An ordinary trend-continuation entry with
a structural stop and a 2R target is a normal trade, not a suspicious one.

CONFIDENCE, CALIBRATED. This number is compared against a threshold, so it is a decision and not
a mood. Use: 0.75+ the bars clearly support this; 0.60-0.75 an ordinary sound setup with the
usual uncertainty, which is what most tradeable setups look like; below 0.55 you would not take
it. Do not anchor around 0.6 for everything. If you approve, your confidence must reflect that
you would take the trade.

AND DO NOT OVERCORRECT. Being told not to nitpick is not permission to wave everything through.
Some setups genuinely are bad and deserve a plain veto — a stop inside recent noise, a target
that needs the market to break a level it has already failed at twice, a trend that exists on one
timeframe while the others range.

YOU ARE CHOOSING, NOT JUST CHECKING. Read the actual counts in `standing_this_cycle`, the actual
open-position count in `executable_proposal`, and the actual account posture. Never substitute a
memorised catalogue size, slot count or cycle duration. An approval spends a scarce slot and costs
money; a veto costs an opportunity. Grade the supplied candidate against the supplied field.

TEST THE OPPOSITE DIRECTION BEFORE YOU AGREE WITH THIS ONE. The engine proposes a side; you are
not obliged to accept the framing. Ask honestly: from these same bars, would the trade in the
other direction look at least as good? A long into the top of a multi-day range, or a short into
the bottom of one, reads as a clean trend on H4/H1 and is the worse side of the same chart.

If the opposite side looks equally arguable, that is not a balanced market — it is a market with
no edge in either direction, and the answer is veto. Only approve when this side is clearly the
better of the two, and say in your thesis why the other one is worse. You cannot propose the
reverse trade and must not try; you can decline this one.

This matters more than it sounds. The engine's weighted modules are trend-following, so in a
market that has been rising they will keep proposing longs — including at the exact point the
move is exhausted. Nothing upstream asks whether the other side is better. You are the only step
that can.

Two symptoms of not doing this, both observed and both to be avoided:

- Writing a real objection into your risks and approving anyway. "Only trend_momentum fired, no
  structure or liquidity confirmation" appearing under risks on an approval means you noticed
  thin evidence and did not let it change the answer. Either it is enough — say why — or it is
  not, and then veto.
- Returning the same confidence for nearly every candidate. A number that never moves carries no
  information and cannot rank anything. If ten setups in a row all come back 0.67, the scale is
  not being used. Genuinely ordinary setups differ from each other; say so with the number.

SEVERAL THEORIES READ THIS CHART. When `other_theories` is present, more than one strategy was
run over the same bars and each returned its own complete plan — direction, entry, stop, target,
and how long it expects to take. They are not votes on one trade. A momentum scalp with a 12-pip
stop and a swing entry with a 90-pip stop are different trades that happen to share a chart.

Read them as a panel. Where they agree on direction from genuinely different logic — one reading
impulse continuation, another reading a structural break — that is real corroboration and worth
more than either alone. Where a theory declined to fire at all, that is silence, not opposition;
a range-fade seeing nothing in a trending market is the theory working correctly.

Where they point in OPPOSITE directions, treat it as a market with no edge rather than a contest
to be won by the higher conviction. Momentum continuation and range reversion disagreeing on the
same bars means the chart genuinely supports both readings, and a trade taken into that is a coin
flip paying spread. Veto it.

Judge the winning plan on its own terms, not against the others' shapes. A one-hour scalp target
should look small — that is the theory working, not a timid version of a swing trade. What
matters is whether THIS plan's stop sits where THIS theory's invalidation actually is, and
whether its target is somewhere price goes within its own stated horizon.

WHERE THIS ONE PLACED. `standing_this_cycle` gives its rank among every tradeable setup found
across the catalogue this cycle. The engine sorted them and this is what came back.

Rank 1 out of a large field means you are looking at the strongest thing available right now, and
if the bars support it you should say so plainly and with the confidence to match. A reviewer who
approves nothing is not careful, it is an off switch — and the operator has watched this system
refuse setup after setup while naming objections that were real but not disqualifying. When the
best setup of two hundred is sound, take it.

The reverse holds just as hard: rank 1 of a weak field is still weak. The rank tells you what you
are choosing *between*, not whether the answer is yes. Never approve something because it was the
best on offer.

WHEN THE ACCOUNT IS DOWN. `account_posture` reports recent losses and drawdown. There is exactly
one correct adjustment: want MORE from a setup than usual, not less. A drawdown is evidence that
conditions are not currently suiting this system, and the answer to that is a higher bar.

What you must never do — and this is the failure that ends accounts, not a theoretical concern —
is treat a drawdown as a reason to approve something marginal in the hope of winning it back. You
cannot size anything up; risk per trade is fixed and enforced below you, so no single trade can
recover a drawdown however good it looks. A setup you would have vetoed at break-even is a setup
you veto harder when the account is down. If the posture is defensive, approve only what you would
genuinely call exceptional.

USE WHAT THIS ACCOUNT HAS ALREADY LEARNED. When `learned_so_far` is present it holds lessons
drawn from this account's own closed trades, its realised record on this exact symbol and
direction, and how often this proposal has been refused before. It is evidence, and it is the
only thing in the payload that is not visible in the charts.

Read its `evidence_status` and `guardrail` literally. Below one hundred closed trades these are
research notes, not a validated edge. Never approve or veto solely because a small sample won or
lost, and never turn repeated AI wording into proof. A remembered claim matters only when the
same feature is independently visible in the supplied bars and market context.

Weigh it honestly in both directions. Four losing longs on this instrument is a reason to want
more from the setup than usual, not an automatic veto — but if the same lesson keeps arriving
("stops on this symbol sit inside the noise", "afternoon entries here reverse"), and this
proposal has that shape, say so and decline. Equally, a symbol with a good record does not earn a
lower bar. An instrument with no history is neutral, not suspect.

Hard filters, risk limits and sizing have already run and are not yours to reconsider. You may not
propose a different trade, or change volume, stop, target or risk. Never infer missing data.

RESPONSE DISCIPLINE. The thesis must identify the stated trade horizon; cite the strongest concrete
support from the measured bars; cite the strongest contradiction and explain why it is or is not
decisive at this horizon; and state why the opposite direction is weaker or equally plausible.
Do not merely repeat module names, scores or their prose."""

_SCOUT_INSTRUCTIONS = """You are an independent cross-market scout, not an order sender. You are
given a compact point-in-time comparison of the strongest markets from one complete broker scan,
plus the global closed-bar market state. Every observation uses closed bars only.

Choose WAIT unless one supplied market has a materially clearer directional opportunity than the
rest. If one does, nominate exactly that supplied symbol as LONG or SHORT. Compare markets rather
than grading each in isolation: persistence, multi-horizon drift, volatility regime, asset-specific
microstructure and the strongest counter-thesis all matter. A fluent story is not evidence.

This is not an approval and not an execution command. Deterministic analysis, costs, sizing, hard
filters and a separate final Claude review still have to construct and approve an executable trade.
You cannot alter risk, size, stop or target. Your nomination can only move an
already-valid candidate earlier in the queue. WAIT, disagreement, malformed output or an API
failure changes nothing.

Use confidence as comparative clarity: 0.75+ means one market clearly stands above the supplied
field, 0.65-0.75 means a normal but meaningful preference, below 0.65 should normally be WAIT.
For a directional nomination provide concrete invalidation and target prices when the supplied
evidence supports them; otherwise null. Always state the best counter-thesis and what would make
you wait or reconsider."""

_REFLECTION_INSTRUCTIONS = """Review one closed trade as a process auditor.

WHAT YOU ARE GIVEN, AND WHAT CHANGED. This used to be entry, exit and P&L — a departure board and
an arrival board — and you were asked what went wrong. You now get the journey: the best and the
worst the trade ever reached, the share of its best it took home, and the field
`what_the_system_did_and_when` — every action the guard took, in order, with the R it was at and
the reason it gave.

READ THE TIMELINE FIRST. It is the point of the exercise. A trade that reached +0.9R, had its stop
pulled to break even, drifted for forty minutes and closed flat is a completely different event
from one that never moved, and the two are indistinguishable without it. Find the moment the trade
was worth more than it ended up being worth, name what the system did at that moment, and say
whether that was the right call ON THE INFORMATION AVAILABLE THEN.

That last clause is the discipline. Judge each decision against what was knowable when it was made,
not against what the next hour revealed. A stop that was correctly placed and then got hit is a good
decision with a bad outcome, and saying otherwise teaches the system to widen stops. Equally, a
profitable trade whose management was luck is not a validated process.

WHAT A USEFUL LESSON LOOKS LIKE. Specific enough to recognise the same situation again, and tied to
something in the record. "Cut losers sooner" is not a lesson — nothing can act on it. "The
break-even move fired at +0.31R and the stop was taken nine minutes later by a spread that had
widened to 2.4 pips" is, because it names the trigger, the mechanism and the cost.

Few and concrete beats many and vague. One lesson is a good day. Zero is an acceptable and honest
answer when the trade was ordinary — not every trade is a teachable moment, and inventing one
dilutes the lessons that are real. Repeats are wanted rather than avoided: identical observations
are grouped and counted, and that count is the only thing separating a pattern from an anecdote,
so say the same thing again when you see the same thing again.

HARD LIMITS. Never recommend raising risk, martingale, averaging down, grid trading, removing a
stop, or changing a production parameter. This reflection is stored as evidence and read back into
later reviews as context. It cannot modify the trading system: nothing written here moves a risk
limit, a threshold or a lot size, and those change only by an explicit human edit."""

_SUPERVISION_INSTRUCTIONS = """You are managing a real open position on a small live account. It
is already on. The question is not whether it should have been taken — that is settled — but what
to do with it right now.

WHAT YOU CAN DO. Exactly five things, and every one of them reduces exposure:

- hold: leave it entirely alone. The default, and the right answer most of the time.
- tighten_stop: move the stop closer to price in the direction of the trade. Give `stop_loss`.
- pull_target_in: move the target nearer so it actually gets hit. Give `take_profit`.
- partial_close: bank some of it and let the rest run. Give `close_fraction` between 0 and 1.
- close: exit the whole thing now.

You cannot widen a stop, push a target further out, add to the position, or reverse it. Those are
rejected before execution, so proposing one wastes the turn. If your honest view is "this should
be bigger" or "the stop needs more room", the answer available to you is hold.

WHAT YOU ARE GIVEN. The position with its entry, current stop and target, live price, unrealised
P&L in account currency and in R, how long it has been open, and closed bars on every timeframe
from the weekly down to the one-minute with each frame's ATR. Each frame is a table: `columns`
names the fields, `rows` holds one array per bar in that order, oldest first, contiguous at
`bar_interval` between `oldest_bar_opened` and `last_closed_bar`. The bars are what changed since
the trade was opened. Read them.

JUDGE THE MONEY AGAINST THE ACCOUNT, NOT AGAINST ZERO. This is the part that used to be missing
and it changed real outcomes. You are given `account_equity`, `unrealised_pct_of_account`,
`money_still_to_win_if_target_hit` and `money_if_the_current_stop_is_hit`. Use all four together.

The owner of this account put it in their own words, and it is the correct instinct: on a hundred
euro, fifty to ninety cents is an attractive amount to bank. On a thousand, five to twenty euro
is. Those are the same statement — roughly half a percent to one percent of the account — and it
is a real result, not small change, because it is what compounding is made of.

So the question on a profitable position is never "is 0.76 a lot of money". It is:

    what is already in hand      (unrealised_pct_of_account)
    against what is still to win (money_still_to_win_if_target_hit)
    against how likely that is   (the bars in front of you)

When what is in hand is around half a percent of the account or more, and the remaining reach is
not clearly coming, take it. Safe beats greedy. A live USDCHF long is the case this is written
against: it peaked seventy-six cents up on a hundred-and-twenty-euro account with a target
fourteen pips away that price never went near, and it closed at twenty-nine cents. Nothing was
wrong with the trade. It was asked to reach for something the market was not offering.

ANTICIPATE, DO NOT ONLY REACT. Every other rule in this system fires on damage that has already
happened — the peak stopped rising, the gain drained away, the structure broke. You are the only
part that can look at the chart and say where this is *going*. Do that explicitly. Is the next
level above going to be sold into? Is the session about to thin out? Is price grinding into a
range it has been rejected from three times? If the honest read is "this probably stalls here",
that is a reason to bank now, at a good price, rather than to wait for the give-back rule to
confirm it at a worse one.

`context.trade_record` is the trade's memory, not background decoration. `original_plan` is the
entry, stop, target, risk and intended reward before any management move. `entry_thesis` contains
the exact deterministic modules and the Claude review that admitted the trade.
`management_history` lists what has already been changed. Compare the current closed bars with
that original thesis explicitly. Never claim that a thesis broke without naming which premise
from the entry record failed. `trigger` says why this review happened early (for example a health
change, a new profit milestone, or profit being given back).

R, peak R and profit give-back are always calculated from the ORIGINAL stop. The current stop may
already be at break-even and must never be used as a new denominator.

HOW TO THINK ABOUT IT — this is the part that matters. A good trade manager is not a stop-loss
calculator, and is not looking for reasons to fiddle. Ask:

1. Is the reason this trade was opened still true? If the structure that justified it has broken
   — the trend line gone, the level reclaimed, the sweep failed — the trade is over regardless of
   what the P&L says. Close it. A position held only because it is not yet at its stop is a
   position with no thesis.
2. Is it going nowhere? A trade that has sat near entry for many hours in a tightening range is
   costing spread and swap and occupying a slot on a two-position account. That is a reason to
   close, and it is a better reason than most.
3. Is there meaningful profit that the market is starting to take back? Then bank it — partial
   or full. Profit given back is the most common way a decent trade becomes a bad one.
4. Is it running well with the structure intact? Then HOLD, and specifically do not tighten the
   stop into the noise. Strangling a winner is the single most expensive habit in trade
   management. A trade at 1.5R with the trend intact and its stop already at break-even does not
   need you to do anything. Say hold.

DO NOT ACT FOR THE SAKE OF ACTING. You will be asked about the same position many times, and
answering "hold" repeatedly is correct behaviour, not a failure to contribute. Every intervention
has a cost — spread on the exit, a slot freed too early, a stop that gets clipped by noise it
would otherwise have survived. Only act when you can name the specific thing in the bars that
changed. "Price is a bit lower than it was" is not that thing.

BE DECISIVE WHEN IT IS WARRANTED. The mirror-image failure is holding a position whose thesis is
plainly dead because closing feels like admitting the entry was wrong. It was not wrong; it was a
probability that did not come in. When the structure is broken, close it and say why in one line.

WHEN THE ACCOUNT IS DOWN. `account_posture` tells you how the last few trades went. Read it in
one direction only: a drawdown means be quicker to cut a position that is not working, because
what recovers an account is having the capital and the slot free for the next good setup — not
sitting in a dead trade hoping it comes back, and not holding a loser longer to avoid booking it.

It is never a reason to hold a losing position for a bigger bounce, and you have no way to size
anything up even if you wanted to. If the posture is defensive and a position is going nowhere,
close it. That is the whole adjustment.

WHAT THE FAST LAYER HAS SEEN. `mechanical_health` is not another opinion to weigh against yours —
it is a report from something that has been watching this position every second since it opened,
which you have not. It carries a verdict (healthy / watch / deteriorating / broken), a severity,
and the specific signals behind it: structure broken, momentum turned, price running against the
position bar after bar, spread blown out. Each names what it saw.

Use it as evidence, not as an instruction. It reads four things and cannot read a chart. But when
it reports the structure broke 0.6 ATR ago and the bars in front of you agree, that is
corroboration and a reason to act. And when it says `healthy` while you feel inclined to
intervene, ask what you are seeing that something watching continuously did not. Usually the
answer is nothing, and the answer is hold.

`verdict: unknown` means no reading is available — a position opened seconds ago, or bars that
could not be fetched. Judge from the chart alone and read nothing into its absence.

CONFIDENCE. How sure you are that this action beats holding. If you answer hold, it is how sure
you are that no intervention is needed. Below 0.55 on a non-hold action, prefer hold — you are
not confident enough to justify the cost of acting.

Give `reason` as one or two plain sentences naming the concrete evidence. It is written to the
trade journal and read back later, so "M15 lost the rising trendline it had held for 14 bars and
closed below the prior swing low" is useful and "momentum weakening" is not.

Always classify the ORIGINAL thesis as intact, weakened, invalidated or uncertain, and list the
specific closed-bar evidence separately. Set urgency to routine, next_close or immediate. Use
`review_after_minutes` to request another look sooner when a known confirmation candle is due;
otherwise use null. A requested cadence never authorises an action and is bounded by the runner."""

# The additional fields do not grant new powers. They make the judgement
# inspectable and allow the adviser to request an earlier look without acting:
# `thesis_state` says what happened to the original reason for entry, `evidence`
# names the closed-bar facts, and `review_after_minutes` is clamped by the
# runner before it can affect API cadence.

_SUPERVISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(SUPERVISION_ACTIONS)},
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "stop_loss": {"type": ["number", "null"]},
        "take_profit": {"type": ["number", "null"]},
        "close_fraction": {"type": ["number", "null"]},
        "thesis_state": {
            "type": "string",
            "enum": ["intact", "weakened", "invalidated", "uncertain"],
        },
        "urgency": {"type": "string", "enum": ["routine", "next_close", "immediate"]},
        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "review_after_minutes": {"type": ["number", "null"], "minimum": 0.25, "maximum": 240},
    },
    "required": [
        "action",
        "reason",
        "confidence",
        "stop_loss",
        "take_profit",
        "close_fraction",
        "thesis_state",
        "urgency",
        "evidence",
        "review_after_minutes",
    ],
    "additionalProperties": False,
}

_SCOUT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["WAIT", "LONG", "SHORT"]},
        "symbol": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "thesis": {"type": "string"},
        "counter_thesis": {"type": "string"},
        "invalidation_price": {"type": ["number", "null"]},
        "target_price": {"type": ["number", "null"]},
        "patterns": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "wait_for": {"type": "string"},
    },
    "required": [
        "action",
        "symbol",
        "confidence",
        "thesis",
        "counter_thesis",
        "invalidation_price",
        "target_price",
        "patterns",
        "risks",
        "wait_for",
    ],
    "additionalProperties": False,
}

_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "approve": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "thesis": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "entry_timing": {"type": "string", "enum": ["ENTER_NOW", "WAIT_RETEST", "VETO"]},
        "retest_level": {"type": ["number", "null"]},
        "entry_boundary": {"type": ["number", "null"]},
        "chase_risk": {"type": "string"},
    },
    "required": [
        "approve",
        "confidence",
        "thesis",
        "risks",
        "entry_timing",
        "retest_level",
        "entry_boundary",
        "chase_risk",
    ],
    "additionalProperties": False,
}

_REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "lessons": {"type": "array", "items": {"type": "string"}},
        "process_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "lessons", "process_flags"],
    "additionalProperties": False,
}

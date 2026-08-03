"""Fail-closed LLM second opinions over bounded, structured market evidence."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from analysis.confluence import TradeIdea
from config.schema import AIConfig
from core.types import MarketContext

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


class Advisor(Protocol):
    def review(
        self,
        idea: TradeIdea,
        context: MarketContext,
        proposal: Mapping[str, object] | None = None,
    ) -> Advice: ...

    def reflect(self, outcome: Mapping[str, object]) -> Reflection: ...


class DisabledAdvisor:
    def review(
        self,
        _idea: TradeIdea,
        _context: MarketContext,
        _proposal: Mapping[str, object] | None = None,
    ) -> Advice:
        return Advice(True, 1.0, "AI advisory disabled", provider="disabled")

    def reflect(self, _outcome: Mapping[str, object]) -> Reflection:
        return Reflection("AI advisory disabled", provider="disabled")


class OpenAIAdvisor:
    def __init__(self, config: AIConfig) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=config.timeout_seconds)
        self.model = config.openai_model
        self.minimum = config.minimum_confidence

    def review(
        self,
        idea: TradeIdea,
        context: MarketContext,
        proposal: Mapping[str, object] | None = None,
    ) -> Advice:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=_REVIEW_INSTRUCTIONS,
                input=_dumps(build_review_payload(idea, context, proposal)),
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

    def review(
        self,
        idea: TradeIdea,
        context: MarketContext,
        proposal: Mapping[str, object] | None = None,
    ) -> Advice:
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                system=_REVIEW_INSTRUCTIONS + " Return only one JSON object matching the schema.",
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
                                "trade_candidate": build_review_payload(idea, context, proposal),
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


class ConsensusAdvisor:
    def __init__(self, advisors: list[Advisor]) -> None:
        self.advisors = advisors

    def review(
        self,
        idea: TradeIdea,
        context: MarketContext,
        proposal: Mapping[str, object] | None = None,
    ) -> Advice:
        answers = [advisor.review(idea, context, proposal) for advisor in self.advisors]
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


def build_review_payload(
    idea: TradeIdea,
    context: MarketContext,
    proposal: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return the exact secret-free trade candidate sent to an AI reviewer."""
    timeframes: dict[str, object] = {}
    for timeframe, series in context.series.items():
        bars: list[dict[str, object]] = []
        for index, row in series.df.tail(3).iterrows():
            timestamp = index.to_pydatetime() if hasattr(index, "to_pydatetime") else index
            bars.append(
                {
                    "time": timestamp.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "tick_volume": int(row.get("tick_volume", 0)),
                }
            )
        timeframes[timeframe.value] = {
            "closed_bars": bars,
            "sample_size": len(series),
            "last_closed_bar": series.last_bar_time.isoformat(),
        }
    tick = None
    if context.tick is not None:
        tick = {
            "time": context.tick.time.isoformat(),
            "bid": context.tick.bid,
            "ask": context.tick.ask,
            "spread": context.tick.spread,
        }
    return {
        "symbol": idea.symbol,
        "direction": idea.direction.name if idea.direction else None,
        "score": idea.score,
        "confidence": idea.confidence,
        "entry": idea.entry,
        "stop_loss": idea.stop_loss,
        "take_profit": idea.take_profit,
        "modules": [asdict(signal) for signal in idea.signals],
        "timeframes": timeframes,
        "decision_tick": tick,
        "executable_proposal": dict(proposal or {}),
        "rule": "Veto or approve only. Never change size, stop, target, risk, or hard filters.",
    }


def _parse_review(
    text: str,
    provider: str,
    model: str,
    minimum: float,
    request_id: str,
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
        return Advice(
            approve and confidence >= minimum,
            confidence,
            thesis,
            tuple(str(item) for item in risks),
            provider,
            model,
            request_id,
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


def _failed_advice(provider: str, model: str, exc: Exception) -> Advice:
    error = _safe_error(exc)
    return Advice(
        False,
        0.0,
        f"{provider} unavailable; trade vetoed",
        provider=provider,
        model=model,
        error=error,
    )


def _failed_reflection(provider: str, model: str, exc: Exception) -> Reflection:
    return Reflection(
        f"{provider} reflection unavailable",
        provider=provider,
        model=model,
        error=_safe_error(exc),
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
    """Remove unsupported raw-schema constraints; local parsing enforces them."""
    if isinstance(value, dict):
        return {
            key: _anthropic_schema(item)
            for key, item in value.items()
            if key not in {"minimum", "maximum", "minLength", "maxLength"}
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

WHAT TO ACTUALLY JUDGE. Read the closed bars yourself and check the proposal against them. Veto
when the bars CONTRADICT the claim — a long whose lower timeframes are selling into the entry, a
"trend" that is a range, a stop sitting where price has already traded through, a target with
structure in the way, a quote too old for the level, a rationale that only restates its own
indicator. Approve when the bars SUPPORT the direction and the stop and target are sensibly
placed for it.

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
This account can hold two positions at a time, so an approval spends a scarce slot, and an
approval you would not defend is worse than a veto because it costs money rather than an
opportunity. Approving nine of the last ten is as broken as vetoing seventy-four in a row: both
mean the review is not reading the evidence, only applying a mood. Some setups genuinely are bad,
and those still deserve a clear veto — a stop that sits inside recent noise, a target that needs
the market to break a level it has already failed at twice, a trend that only exists on one
timeframe while the others range. Say no to those plainly.

Hard filters, risk limits and sizing have already run and are not yours to reconsider. You may not
propose a different trade, or change volume, stop, target or risk. Never infer missing data."""

_REFLECTION_INSTRUCTIONS = """Review one closed trade as a process auditor. Separate outcome luck
from decision quality. Identify evidence-supported lessons and process flags only. Never recommend
raising risk, martingale, averaging down, grid trading, or changing production parameters. This
reflection is logged for research and cannot directly modify the trading system."""

_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "approve": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "thesis": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["approve", "confidence", "thesis", "risks"],
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

"""Bounded LLM second opinions over structured evidence, never chart screenshots."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Protocol

from analysis.confluence import TradeIdea
from config.schema import AIConfig
from core.types import MarketContext


@dataclass(frozen=True, slots=True)
class Advice:
    approved: bool
    confidence: float
    thesis: str
    risks: tuple[str, ...] = ()
    provider: str = "disabled"


class Advisor(Protocol):
    def review(self, idea: TradeIdea, context: MarketContext) -> Advice: ...


class DisabledAdvisor:
    def review(self, _idea: TradeIdea, _context: MarketContext) -> Advice:
        return Advice(True, 1.0, "AI advisory disabled", provider="disabled")


class OpenAIAdvisor:
    def __init__(self, config: AIConfig) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=config.timeout_seconds)
        self.model = config.openai_model
        self.minimum = config.minimum_confidence

    def review(self, idea: TradeIdea, context: MarketContext) -> Advice:
        response = self.client.responses.create(
            model=self.model,
            instructions=_INSTRUCTIONS,
            input=json.dumps(_payload(idea, context)),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "trade_review",
                    "strict": True,
                    "schema": _SCHEMA,
                }
            },
        )
        return _parse(response.output_text, "openai", self.minimum)


class AnthropicAdvisor:
    def __init__(self, config: AIConfig) -> None:
        import anthropic

        if not config.anthropic_model:
            raise RuntimeError("ai.anthropic_model must be configured")
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = config.anthropic_model
        self.minimum = config.minimum_confidence
        self.timeout = config.timeout_seconds

    def review(self, idea: TradeIdea, context: MarketContext) -> Advice:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=_INSTRUCTIONS + " Return only JSON matching: " + json.dumps(_SCHEMA),
            messages=[{"role": "user", "content": json.dumps(_payload(idea, context))}],
            timeout=self.timeout,
        )
        text = "".join(block.text for block in message.content if hasattr(block, "text"))
        return _parse(text, "anthropic", self.minimum)


class ConsensusAdvisor:
    def __init__(self, advisors: list[Advisor]) -> None:
        self.advisors = advisors

    def review(self, idea: TradeIdea, context: MarketContext) -> Advice:
        answers = [advisor.review(idea, context) for advisor in self.advisors]
        approved = bool(answers) and all(answer.approved for answer in answers)
        confidence = min((answer.confidence for answer in answers), default=0.0)
        return Advice(
            approved,
            confidence,
            " | ".join(answer.thesis for answer in answers),
            tuple(risk for answer in answers for risk in answer.risks),
            "consensus",
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
        raise RuntimeError(
            f"AI enabled but provider dependency/credential is missing: {exc}"
        ) from exc


def _payload(idea: TradeIdea, context: MarketContext) -> dict[str, object]:
    return {
        "symbol": idea.symbol,
        "direction": idea.direction.name if idea.direction else None,
        "score": idea.score,
        "confidence": idea.confidence,
        "entry": idea.entry,
        "stop_loss": idea.stop_loss,
        "take_profit": idea.take_profit,
        "modules": [asdict(signal) for signal in idea.signals],
        "available_timeframes": [timeframe.value for timeframe in context.series],
        "rule": "May veto only. Never change size, stop, target, risk, or hard filters.",
    }


def _parse(text: str, provider: str, minimum: float) -> Advice:
    try:
        payload = json.loads(text)
        confidence = float(payload["confidence"])
        approved = bool(payload["approve"]) and confidence >= minimum
        return Advice(
            approved,
            confidence,
            str(payload["thesis"]),
            tuple(str(item) for item in payload["risks"]),
            provider,
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return Advice(False, 0.0, f"invalid {provider} response: {exc}", provider=provider)


_INSTRUCTIONS = """You are a conservative trade-review committee. Review only the supplied
closed-bar evidence. Veto if evidence conflicts, data is insufficient, the rationale is circular,
or regime risk is material. You may not propose a different trade or relax any rule."""

_SCHEMA = {
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

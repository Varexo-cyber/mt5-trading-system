"""Fail-closed LLM second opinions over bounded, structured market evidence."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import pandas as pd

from analysis.confluence import TradeIdea
from config.schema import AIConfig
from core.types import MarketContext, Timeframe

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


#: Closed bars sent per timeframe. Three was the original figure and it made
#: the review impossible: nobody can judge whether a target is reachable, or
#: whether price is running into a level, from three candles. The higher frames
#: need fewer bars to show structure; the lower ones are where the entry lives.
_BARS_SENT = {"MN1": 12, "W1": 20, "D1": 40, "H4": 60, "H1": 80, "M15": 80, "M5": 60, "M1": 60}
_BARS_SENT_DEFAULT = 40


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


def build_review_payload(
    idea: TradeIdea,
    context: MarketContext,
    proposal: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return the exact secret-free trade candidate sent to an AI reviewer.

    Carries enough of each chart to actually be read. The original three bars
    per timeframe left the reviewer nothing to work from but the engine's own
    summary, so every answer restated the module's reasoning back — it could
    not see whether price was running into a level, or whether the target was
    somewhere this market ever goes.
    """
    timeframes: dict[str, object] = {}
    for timeframe, series in context.series.items():
        frame = series.df
        count = _BARS_SENT.get(timeframe.value, _BARS_SENT_DEFAULT)
        recent = frame.tail(count)
        bars = [
            {
                "t": (index.to_pydatetime() if hasattr(index, "to_pydatetime") else index)
                .isoformat()
                .replace("+00:00", "Z"),
                "o": round(float(row["open"]), 6),
                "h": round(float(row["high"]), 6),
                "l": round(float(row["low"]), 6),
                "c": round(float(row["close"]), 6),
                "v": int(row.get("tick_volume", 0)),
            }
            for index, row in recent.iterrows()
        ]
        atr = _atr(frame)
        timeframes[timeframe.value] = {
            "closed_bars": bars,
            "bars_sent": len(bars),
            "history_available": len(series),
            "last_closed_bar": series.last_bar_time.isoformat(),
            "atr14": round(atr, 6),
            "range_high": round(float(recent["high"].max()), 6),
            "range_low": round(float(recent["low"].min()), 6),
        }

    tick = None
    if context.tick is not None:
        tick = {
            "time": context.tick.time.isoformat(),
            "bid": context.tick.bid,
            "ask": context.tick.ask,
            "spread": context.tick.spread,
        }

    # Is this target somewhere the market goes? Answered from its own history
    # rather than from the risk-reward ratio that produced it.
    target_check: dict[str, object] = {}
    signal = context.series.get(Timeframe.H1)
    if signal is not None and idea.entry:
        atr = _atr(signal.df)
        stop_distance = abs(idea.entry - idea.stop_loss) if idea.stop_loss else 0.0
        target_distance = abs(idea.take_profit - idea.entry) if idea.take_profit else 0.0
        target_check = {
            "timeframe": "H1",
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
            "history": _reachability(signal.df.tail(400), target_distance, bars_ahead=24),
            "note": (
                "history: of every 24-hour window in the last 400 H1 bars, the share that "
                "travelled at least the target distance. A low number means this target is "
                "rarely reached in a day on this instrument, whatever the risk-reward says."
            ),
        }

    return {
        "symbol": idea.symbol,
        "direction": idea.direction.name if idea.direction else None,
        "score": idea.score,
        "confidence": idea.confidence,
        "entry": idea.entry,
        "stop_loss": idea.stop_loss,
        "take_profit": idea.take_profit,
        "target_realism": target_check,
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

YOU HAVE THE CHARTS. Every timeframe from the weekly down to the one-minute arrives as closed
OHLC bars — dozens per frame, not a summary — with each frame's ATR and its high and low over
that window. Read them. The higher frames say whether there is a trend and where the structure
is; H1 and M15 say whether the entry sits at a sensible place in it; M5 and M1 say whether price
is moving toward the entry or away from it right now. An answer that could have been written from
the module scores alone means the bars went unread.

JUDGE THE STOP AND THE TARGET AS PLACEMENTS. `target_realism` gives you the stop and target in
ATR, the spread as a percentage of the stop, and — measured from this instrument's own recent
history — how often it has actually travelled the target distance within a day. The engine sets
the target at twice the stop by arithmetic; it never checks whether the market goes there. If
that percentage is low the target is decorative, and the trade is really a bet on the stop not
being hit. Veto a stop sitting inside ordinary noise, a stop or target on the wrong side of an
obvious level in the bars you were given, and a spread eating a large share of the stop.

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
Some setups genuinely are bad and deserve a plain veto — a stop inside recent noise, a target
that needs the market to break a level it has already failed at twice, a trend that exists on one
timeframe while the others range.

YOU ARE CHOOSING, NOT JUST CHECKING. Around two hundred markets are analysed each cycle and this
account can hold two positions. So the question is not "is this acceptable" but "is this among
the best few of two hundred". An approval spends a scarce slot and costs money; a veto costs only
an opportunity, and another candidate arrives within the minute. Grade accordingly.

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

"""A second pair of eyes on the whole account, with no authority over it.

Everything else that calls an AI in this repo answers a closed question about
one trade: approve or veto, hold or exit. That is the right shape for a gate —
it is auditable, it is cheap, and it cannot wander. It is the wrong shape for
the question the operator keeps actually asking, which is "look at all of it
and tell me what is wrong".

They have been asking a person that question every few hours, pasting
screenshots. This is that, running on the machine.

**It has no power and cannot be given any.** It returns prose. There is no
verdict enum, no `Reason`, no boolean anywhere in the response, and nothing
downstream reads it — the trading path does not import this module. That is
not a limitation to be lifted later; it is what makes an open-ended reasoner
safe to run against a live account. A gate must be predictable. An analyst
must be free to say something nobody anticipated, and those two properties
cannot live in the same component.

Model and cost. This runs on Opus because the operator asked for the most
capable reading available and because the frequency makes it affordable: a few
cents, a handful of times a day, against a per-trade reviewer running dozens of
times an hour. Running this per trade would be the wrong tool at the wrong
price.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from analyst.evidence import Evidence
from infra.logging import get_logger

log = get_logger(__name__)

#: Opus, because the request was explicitly for the most capable reading, and
#: because a handful of calls a day makes that affordable where a per-trade
#: gate would not.
DEFAULT_MODEL = "claude-opus-5"

#: Generous. This is asked to reason across dozens of trades and every setting,
#: a few times a day, and truncating the one answer that is supposed to be
#: thorough would defeat the point of running it at all.
MAX_TOKENS = 8000

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {
            "type": "string",
            "description": (
                "The single most important thing wrong right now, in one sentence "
                "an operator can act on. If nothing is wrong, say that plainly."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": (
                "How the evidence leads to the headline. Cite the actual numbers. "
                "Two or three paragraphs."
            ),
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "what": {"type": "string"},
                    "evidence": {
                        "type": "string",
                        "description": "The specific numbers from the payload that show it.",
                    },
                    "severity": {"type": "string", "enum": ["critical", "notable", "minor"]},
                    "suggested_change": {
                        "type": "string",
                        "description": (
                            "The setting or behaviour you would change, and to what. "
                            "Empty string when the honest answer is that you do not know."
                        ),
                    },
                },
                "required": ["what", "evidence", "severity", "suggested_change"],
                "additionalProperties": False,
            },
        },
        "what_is_working": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Things the evidence shows are working. Not filler — an operator "
                "who only ever hears what is broken starts changing what is not."
            ),
        },
        "what_i_cannot_tell": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Questions this evidence cannot answer, and what would be needed. "
                "The most valuable section when the sample is small."
            ),
        },
        "confidence": {
            "type": "number",
            "description": "0 to 1. Four trades is not a sample; say so with a low number.",
        },
    },
    "required": [
        "headline",
        "reasoning",
        "findings",
        "what_is_working",
        "what_i_cannot_tell",
        "confidence",
    ],
    "additionalProperties": False,
}

_INSTRUCTIONS = """You are reviewing a live algorithmic trading account for the person who
owns it. Real money, a small balance, and they are losing some of it.

WHAT YOU ARE AND ARE NOT. You are an analyst, not a gate. You cannot approve, block, size or
close anything, and nothing you write is executed. Say the useful thing rather than the safe
one, and if you think the whole approach is wrong, say that.

WHAT YOU ARE GIVEN. Closed trades with what each was worth at its best against what it
returned; every gate that refused a setup, with counts; the live read on open positions
including how old that read is; and the settings actually in force. The arithmetic that a
spreadsheet can settle has been done — your job is the part it cannot do.

HOW TO READ IT. Some specific traps, all of which have caught this account:

`kept_share_of_peak` separates a strategy that is wrong from one that is right and hands the
gain back. Those need opposite fixes: the first needs better entries, the second needs the
management layer to act.

`exits_chosen_by_the_system` against `exits_left_to_the_broker`. If nothing was ever closed by
a rule, the management layer is not running — a bug, not a tuning problem, and the loudest
finding available.

`stop_outs_worse_than_minus_one_r` should be zero by definition: a stop sits one R away. Any
number above zero is cost or execution eating the risk model, not a strategy failure.

The refusal histogram against the trade count. Two trades from three thousand decisions is
described by the 2,998, not the two. But a gate refusing a great deal is not automatically
wrong — read what it refuses and decide.

The age of a health reading. Anything more than a minute old on a loop that runs every second
means the loop is not running.

BE SPECIFIC AND BE HONEST ABOUT THE SAMPLE. "Improve the entries" helps nobody. "The three
losing trades all had stops under six pips while commission alone is 22% of the risk at that
width" is something they can act on. And with four trades you cannot distinguish a bad system
from an ordinary losing streak — when that is the situation, `what_i_cannot_tell` is the most
valuable section on the page and `confidence` belongs near the floor.

Do not invent numbers. Every figure you cite must be in the payload."""


@dataclass(frozen=True, slots=True)
class Finding:
    what: str
    evidence: str
    severity: str
    suggested_change: str


@dataclass(frozen=True, slots=True)
class Assessment:
    """What the analyst concluded. Prose and nothing else, by design."""

    headline: str
    reasoning: str
    findings: tuple[Finding, ...] = ()
    what_is_working: tuple[str, ...] = ()
    what_i_cannot_tell: tuple[str, ...] = ()
    confidence: float = 0.0
    model: str = ""
    generated_at: str = ""
    error: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        """For a terminal. The deck renders the same data its own way."""
        if self.error:
            return f"\n  Analyst unavailable: {self.error}\n"
        lines = [
            "",
            "=" * 78,
            f"  {self.headline}",
            "=" * 78,
            "",
            self.reasoning,
            "",
        ]
        if self.findings:
            lines.append("-" * 78)
            lines.append("  FINDINGS")
            lines.append("-" * 78)
            lines.append("")
            for finding in self.findings:
                lines.append(f"  [{finding.severity}] {finding.what}")
                lines.append(f"      evidence: {finding.evidence}")
                if finding.suggested_change:
                    lines.append(f"      change:   {finding.suggested_change}")
                lines.append("")
        if self.what_is_working:
            lines.append("  WORKING")
            for item in self.what_is_working:
                lines.append(f"    + {item}")
            lines.append("")
        if self.what_i_cannot_tell:
            lines.append("  CANNOT TELL FROM THIS EVIDENCE")
            for item in self.what_i_cannot_tell:
                lines.append(f"    ? {item}")
            lines.append("")
        lines.append(f"  confidence {self.confidence:.2f} · {self.model}")
        lines.append("")
        return "\n".join(lines)


def _failed(reason: str, model: str) -> Assessment:
    """No analysis is a missing opinion, never a signal.

    Returned rather than raised so a scheduled run cannot take the trading
    service down with it — and so the deck shows why instead of an empty panel.
    """
    log.warning("analyst call failed", extra={"event": "analyst_failed", "reason": reason})
    return Assessment(
        headline="",
        reasoning="",
        error=reason,
        model=model,
        generated_at=datetime.now(UTC).isoformat(),
    )


def analyse(
    evidence: Evidence, *, model: str = DEFAULT_MODEL, timeout: float = 180.0
) -> Assessment:
    """Ask for a reading of the whole account. Returns prose, never a decision."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return _failed("ANTHROPIC_API_KEY is not set", model)

    try:
        import anthropic
    except ImportError:
        return _failed("the anthropic package is not installed", model)

    client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=1)
    try:
        message = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": _INSTRUCTIONS,
                    # Byte-identical between runs, so a second call in the same
                    # hour reads it at a tenth of the price.
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(evidence.as_payload(), separators=(",", ":")),
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 - any transport failure is a missing opinion
        return _failed(f"{type(exc).__name__}: {exc}", model)

    text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return _failed("the response was not the requested JSON", model)

    usage = getattr(message, "usage", None)
    return Assessment(
        headline=str(parsed.get("headline", "")),
        reasoning=str(parsed.get("reasoning", "")),
        findings=tuple(
            Finding(
                what=str(item.get("what", "")),
                evidence=str(item.get("evidence", "")),
                severity=str(item.get("severity", "minor")),
                suggested_change=str(item.get("suggested_change", "")),
            )
            for item in parsed.get("findings", [])
            if isinstance(item, dict)
        ),
        what_is_working=tuple(str(item) for item in parsed.get("what_is_working", [])),
        what_i_cannot_tell=tuple(str(item) for item in parsed.get("what_i_cannot_tell", [])),
        confidence=float(parsed.get("confidence", 0.0)),
        model=model,
        generated_at=datetime.now(UTC).isoformat(),
        usage=(
            {
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            }
            if usage
            else {}
        ),
    )

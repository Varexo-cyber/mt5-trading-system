"""Verify the production AI review path with one tiny fictional candidate."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from advisory import build_advisor
from analysis.confluence import TradeIdea
from config.loader import load_settings
from core.types import Direction, MarketContext, Signal


def main() -> int:
    load_dotenv(ROOT / "config" / ".env", override=False)
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    if not settings.ai.enabled or settings.ai.provider != "anthropic":
        print("BLOCKED: Eightcap AI gate is not configured for Anthropic")
        return 2
    try:
        adviser = build_advisor(settings.ai)
        signal = Signal(
            "verification_only",
            70.0,
            0.8,
            "Fictional input used only to verify the production response contract.",
        )
        idea = TradeIdea(
            "VERIFY_ONLY",
            True,
            Direction.LONG,
            70.0,
            0.8,
            1.0,
            0.99,
            1.02,
            "Synthetic API verification; this symbol does not exist",
            (signal,),
        )
        context = MarketContext("VERIFY_ONLY", datetime.now(UTC), {})
        advice = adviser.review(
            idea,
            context,
            {
                "operation": "verification_only",
                "actual_risk_pct": 0.0,
                "executable": False,
            },
        )
        if advice.error:
            print(f"BLOCKED: production response failed validation ({advice.error})")
            return 3
        verdict = "APPROVE" if advice.approved else "VETO"
        request_id = advice.request_id or "not-reported"
        print(
            f"READY: provider={advice.provider} model={advice.model} "
            f"verdict={verdict} request_id={request_id}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - command reports safe diagnostics only
        status = getattr(exc, "status_code", None)
        suffix = f" http_status={status}" if isinstance(status, int) else ""
        print(f"BLOCKED: {type(exc).__name__}{suffix}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

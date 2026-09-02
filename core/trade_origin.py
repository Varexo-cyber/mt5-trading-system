"""Stable operator labels for the measured live strategy instances.

The broker comment is the only origin label visible in MT5 mobile.  Keeping
the mapping here also lets journal reports describe an already-open ticket in
the same words, even when that ticket predates the distinct broker comments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TradeOrigin:
    section: int
    strategy: str
    timeframe: str
    comment: str


# Specific aliases must precede their parent family.  Setup-family names carry
# the module alias plus the clock (for example ``order_block_fast_m1``).
_ORIGINS: tuple[tuple[str, TradeOrigin], ...] = (
    ("section_ten_gold_m1", TradeOrigin(10, "large_break_retest", "M1", "JARVIS-S10-AU-M1")),
    ("section_eight_trend_day_h1", TradeOrigin(8, "trend_day", "H1", "JARVIS-S8-TD-H1")),
    ("section_nine_vwap_m30", TradeOrigin(9, "session_vwap", "M30", "JARVIS-S9-VW-M30")),
    ("section_six_gold_m5", TradeOrigin(6, "adaptive_gold", "M5", "JARVIS-S6-AU-M5")),
    ("section_six_spx_h1", TradeOrigin(6, "adaptive_spx", "H1", "JARVIS-S6-SP-H1")),
    ("section_five_m5", TradeOrigin(5, "nonlinear_state", "M5", "JARVIS-S5-NL-M5")),
    (
        "failed_session_breakout",
        TradeOrigin(7, "failed_session_breakout", "M5", "JARVIS-S7-FSB-M5"),
    ),
    ("walkforward_index", TradeOrigin(4, "walkforward_index", "H1", "JARVIS-S4-WF-H1")),
    ("impulse_retest_m30", TradeOrigin(2, "impulse_retest", "M30", "JARVIS-S2-IR-M30")),
    ("impulse_retest", TradeOrigin(2, "impulse_retest", "M15", "JARVIS-S2-IR-M15")),
    ("order_block_fast", TradeOrigin(3, "order_block", "M1", "JARVIS-S3-OB-M1")),
    ("order_block_m15", TradeOrigin(3, "order_block", "M15", "JARVIS-S3-OB-M15")),
    ("order_block_h1", TradeOrigin(3, "order_block", "H1", "JARVIS-S3-OB-H1")),
    ("order_block", TradeOrigin(3, "order_block", "M30", "JARVIS-S3-OB-M30")),
)


def origin_for_setup_family(setup_family: str) -> TradeOrigin | None:
    """Return the shipped live instance represented by a setup-family name."""

    family = str(setup_family or "").lower()
    for marker, origin in _ORIGINS:
        if marker in family:
            return origin
    return None


def broker_comment(
    setup_family: str,
    *,
    is_addon: bool,
    experimental_live: bool,
) -> str:
    """Comment sent to MT5, bounded by its 31-character comment field."""

    if is_addon:
        return "jarvis-scalp"
    origin = origin_for_setup_family(setup_family)
    if origin is not None:
        return origin.comment
    return "jarvis-exp-live" if experimental_live else "jarvis"

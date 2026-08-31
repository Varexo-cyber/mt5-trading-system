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

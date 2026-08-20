"""Classify detectors by the independent market fact they observe."""

from __future__ import annotations

from core.types import Direction, Signal

_FAMILIES = {
    "trend_momentum": "trend",
    "drift_continuation": "trend",
    "ema_pullback_resume": "trend",
    "fast_ema_cross": "momentum",
    "impulse_break": "momentum",
    "m1_micro_breakout": "momentum",
    "market_structure": "structure",
    "level_reaction": "structure",
    "liquidity_sweep": "liquidity",
    "volatility_regime": "context",
    "market_regime": "context",
    "seasonality": "context",
    "mean_reversion": "mean_reversion",
    "volatility_squeeze": "volatility",
    "session_breakout": "session",
}


def family_for(module: str) -> str:
    return _FAMILIES.get(module, module)


def supporting_families(
    signals: tuple[Signal, ...], direction: Direction | None
) -> tuple[str, ...]:
    """Unique evidence families supporting ``direction``, stable for audit."""

    if direction is None:
        return ()
    sign = int(direction)
    families = {
        family_for(signal.module)
        for signal in signals
        if signal.score != 0.0 and (1 if signal.score > 0.0 else -1) == sign
    }
    return tuple(sorted(families))

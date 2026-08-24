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
    # Its own family, and that is the entire point of the module. Every
    # other reader here belongs to trend, momentum, structure or liquidity
    # — four ways of reading one price series. This one runs a hypothesis
    # test on whether the move is real and then fades it. Filing it under
    # any existing family would let it corroborate a reader it has nothing
    # in common with, which is the exact failure the families exist to stop.
    "drift_burst": "immediacy",
    # Also its own, and for the same reason: it is the only reader here
    # whose evidence is a RELATION between two instruments rather than a
    # property of one. Filing it beside a chart family would let it
    # corroborate exactly the reading it exists to check.
    "basket_divergence": "relative",
    # `momentum`, deliberately, and NOT a family of its own. It reads the
    # same fact the other momentum readers read — price moved, hard, just
    # now — and giving it its own label would let it "corroborate"
    # `impulse_break` on one observation seen twice, which is the exact
    # failure families exist to prevent.
    "candle_momentum": "momentum",
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

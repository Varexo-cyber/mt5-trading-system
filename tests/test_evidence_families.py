from analysis.evidence_families import family_for, supporting_families
from core.types import Direction, Signal


def test_related_ema_detectors_count_as_one_family() -> None:
    signals = (
        Signal("trend_momentum", 60, 0.8),
        Signal("drift_continuation", 55, 0.7),
        Signal("liquidity_sweep", 45, 0.7),
    )
    assert supporting_families(signals, Direction.LONG) == ("liquidity", "trend")


def test_opposite_direction_does_not_confirm() -> None:
    signals = (Signal("market_structure", -60, 0.8), Signal("impulse_break", 55, 0.8))
    assert supporting_families(signals, Direction.LONG) == ("momentum",)
    assert family_for("unmapped_detector") == "unmapped_detector"

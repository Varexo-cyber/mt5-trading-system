"""A bounded second brain for ordering already-valid trade setups.

The detector engine remains the source of setups.  This module answers a
different question: among the setups that exist now, which combination of
conditions has actually earned money on this account?  It deliberately has no
API that can approve, reject, resize or move a price level.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from brain.store import SelectionEvidence


@dataclass(frozen=True, slots=True)
class SelectionVerdict:
    """Bounded ranking adjustment and the evidence that produced it."""

    modifier: float = 0.0
    reasons: tuple[str, ...] = ()
    matched_dimensions: tuple[str, ...] = ()

    def summary(self) -> str:
        if not self.reasons:
            return "selection brain is neutral: no sufficiently repeated matching outcomes"
        return (
            f"selection brain {self.modifier:+.2f} from "
            f"{', '.join(self.matched_dimensions)}; " + "; ".join(self.reasons)
        )


def score_band(score: float) -> str:
    """The same ten-point band used by the Postgres evidence query."""
    return str(int(score // 10) * 10)


def combine_selection_evidence(
    evidence: Sequence[SelectionEvidence],
    *,
    asset_class: str,
    setup_family: str,
    horizon: str,
    direction: str,
    regime: str,
    session: str,
    score: float,
    detectors: Sequence[str],
    weights: Mapping[str, float],
    strength: float,
    cap: float,
) -> SelectionVerdict:
    """Blend matching facets without pretending correlated votes are independent.

    At most one voice per ordinary dimension is used.  Detector evidence is
    averaged into one voice even when several modules fired on the setup; this
    prevents a three-detector confluence from receiving three times the learned
    authority of a one-detector setup.
    """
    wanted = {
        "setup_horizon": f"{setup_family}|{horizon}",
        "setup_family": setup_family,
        "horizon": horizon,
        "asset_class": asset_class,
        "regime": regime,
        "session": session,
        "score_band": score_band(score),
        "direction": "*",
    }
    by_key = {item.key: item for item in evidence}
    voices: list[tuple[str, float, float, tuple[SelectionEvidence, ...]]] = []
    for dimension, value in wanted.items():
        item = by_key.get((dimension, value, direction))
        weight = float(weights.get(dimension, 0.0))
        if item is not None and weight > 0.0:
            voices.append((dimension, item.modifier, weight, (item,)))

    detector_rows = tuple(
        row
        for name in sorted(set(detectors))
        if (row := by_key.get(("detector", name, direction))) is not None
    )
    detector_weight = float(weights.get("detector", 0.0))
    if detector_rows and detector_weight > 0.0:
        detector_voice = sum(row.modifier for row in detector_rows) / len(detector_rows)
        voices.append(("detector", detector_voice, detector_weight, detector_rows))

    total_weight = sum(weight for _, _, weight, _ in voices)
    if total_weight <= 0.0:
        return SelectionVerdict()
    blended = sum(value * weight for _, value, weight, _ in voices) / total_weight
    modifier = max(-cap, min(cap, blended * strength))

    ranked = sorted(voices, key=lambda voice: abs(voice[1] * voice[2]), reverse=True)
    reasons: list[str] = []
    for dimension, value, _, rows in ranked[:4]:
        if dimension == "detector":
            names = ",".join(row.value for row in rows)
            sample = min(row.trades for row in rows)
            reasons.append(f"detectors {names} ({sample}+ trades) contribute {value:+.2f}")
        else:
            row = rows[0]
            reasons.append(
                f"{dimension}={row.value} ({row.trades} trades) contributes {value:+.2f}"
            )
    return SelectionVerdict(
        modifier=round(modifier, 3),
        reasons=tuple(reasons),
        matched_dimensions=tuple(dimension for dimension, _, _, _ in voices),
    )

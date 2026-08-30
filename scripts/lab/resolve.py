"""First-touch resolution for a whole batch of signals at once.

The single-signal loop in the first study took twenty minutes for one sweep of
eight instruments at M15. This grid is sixteen instruments at six timeframes
with millions of M1 bars, which is roughly three hundred times the work. So
the walk forward runs across all signals simultaneously: one numpy pass per
step of the horizon rather than one Python loop per trade.

THE RULES, each one from a mistake already made on this account:

* Barrier-resolved only. A trade that touched neither barrier inside the
  horizon is EXCLUDED, not counted as a loss and not closed at the clock.
  Mixing clock exits into a first-touch model manufactured a +16.7% edge on a
  random walk once already.
* Same-bar ambiguity counts as a LOSS. When one bar spans both barriers the
  order is unknowable from OHLC.
* A LIMIT fill is checked from its own bar. The fill happens partway through,
  so the rest of that bar can still take the stop, and skipping it scores a
  trade that was already dead as if it had never started. That single omission
  read +0.487R where the truth was +0.347R.
* The baseline is arithmetic: for a driftless walk, first-touch of a stop of 1
  before a target of k is 1/(1+k). Never 50%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNRESOLVED = -1


@dataclass(frozen=True, slots=True)
class Batch:
    """Signals from one instrument, as parallel arrays."""

    index: np.ndarray  # bar the trade starts on
    direction: np.ndarray  # +1 long, -1 short
    entry: np.ndarray  # fill price
    unit: np.ndarray  # one R, in price
    same_bar: bool  # True for limit fills, False for close entries

    def __len__(self) -> int:
        return int(self.index.size)

    def take(self, keep: np.ndarray) -> Batch:
        return Batch(
            self.index[keep],
            self.direction[keep],
            self.entry[keep],
            self.unit[keep],
            self.same_bar,
        )


def resolve(high: np.ndarray, low: np.ndarray, batch: Batch, k: float, horizon: int):
    """Return an array of 1 (target first), 0 (stop first), -1 (neither).

    One vector pass per step of the horizon. `settled` freezes a trade the
    moment it resolves so a later bar cannot overwrite the answer.
    """
    n = high.size
    count = len(batch)
    if count == 0:
        return np.empty(0, dtype=np.int8)

    stop = batch.entry - batch.direction * batch.unit
    target = batch.entry + batch.direction * batch.unit * k
    long = batch.direction > 0

    outcome = np.full(count, UNRESOLVED, dtype=np.int8)
    settled = np.zeros(count, dtype=bool)

    for step in range(0 if batch.same_bar else 1, horizon + 1):
        here = batch.index + step
        inside = here < n
        live = inside & ~settled
        if not live.any():
            break
        # Gather only where the index is valid; clip keeps the fancy-index
        # legal for the rest, and those lanes are masked off anyway.
        at = np.clip(here, 0, n - 1)
        bar_high = high[at]
        bar_low = low[at]

        hit_stop = np.where(long, bar_low <= stop, bar_high >= stop) & live
        hit_target = np.where(long, bar_high >= target, bar_low <= target) & live

        # Both in one bar: the order is unknowable, so it is a loss.
        both = hit_stop & hit_target
        only_stop = hit_stop & ~hit_target
        only_target = hit_target & ~hit_stop

        outcome[both | only_stop] = 0
        outcome[only_target] = 1
        settled |= both | only_stop | only_target

    return outcome


def summarise(outcome: np.ndarray) -> tuple[int, int, int]:
    """(wins, resolved, unresolved)."""
    resolved = int((outcome >= 0).sum())
    return int((outcome == 1).sum()), resolved, int((outcome == UNRESOLVED).sum())


def sigmas(wins: int, resolved: int, baseline: float) -> float:
    if resolved == 0:
        return 0.0
    spread = (baseline * (1.0 - baseline) / resolved) ** 0.5
    return (wins / resolved - baseline) / spread if spread > 0 else 0.0


def expectancy(wins: int, resolved: int, k: float) -> float:
    if resolved == 0:
        return 0.0
    hit = wins / resolved
    return hit * k - (1.0 - hit)


def chance(k: float) -> float:
    return 1.0 / (1.0 + k)

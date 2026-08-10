"""Stop re-analysing a market whose history the broker does not have.

The scanner was spending a full multi-timeframe pass per cycle on symbols it
then refused for a reason that cannot change between cycles:

    SPCX W1: 8 closed bars available, 50 required
    HSBC H4: 80 bars missing inside trading weeks (5.3%, limit 5.0%)

Neither of those is a fact about this minute. The first is the length of the
broker's history for that symbol; the second is a hole in its feed. A minute
later the answer is identical, and on a one-vCPU VPS with an 800-symbol
catalogue that work is repeated for every one of them, every cycle, forever.

So the answer is remembered, with a backoff, and the ladder is not fetched
again until the hold expires.

WHAT THIS IS NOT, AND THE DISTINCTION IS THE WHOLE SAFETY ARGUMENT. This does
not relax a gate, widen a universe or approve anything. A held symbol was
already producing "no trade" and still produces "no trade" — the only change is
that it costs no bars to say so. The set of tradeable symbols can only shrink
here, never grow, so no spread, session, liveliness or risk rule can be reached
differently because of it. Those gates run exactly as before on the symbols
that are analysed.

Only two failures are structural enough to hold: not enough history, and gaps
inside trading weeks. A stale quote is not one of them — that resolves the
moment the venue reopens, and holding it would keep a market out of the scan
for hours after it came back.

THE HOLD IS NOT REALLY A TIMER. It is a bet that nothing has changed, and the
first bar the venue prints settles that bet: a symbol whose exchange opens
again tomorrow is released by its own new bar, not by the clock. The cheap scan
already reports `latest_bar` for every candidate, so this costs nothing.

The clock is only the backstop for when no bar can be read, and it is
deliberately short. Going from a fetch every cycle to one an hour removes 98.3%
of the waste; stretching that to a day removes 99.93%. The extra
seven-hundredths of a percentage point is not worth being blind to a market for
a day, so the ceiling is hours rather than days.

Held in memory rather than on disk, deliberately. A restart re-learns the whole
set within one cycle, which costs one expensive pass; a stale file that
outlives a broker fixing its history costs a symbol nobody notices is missing.
Re-learning is the cheaper mistake.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from infra.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Hold:
    """One symbol being skipped, and what would bring it back."""

    symbol: str
    until: datetime
    reason: str
    failures: int
    #: Newest bar the cheap scan had seen when this symbol was held. A later
    #: one means the venue has traded since, so the window is not the window
    #: that failed and the answer is worth asking again.
    latest_bar: datetime | None = None

    def minutes_left(self, now: datetime) -> float:
        return max(0.0, (self.until - now).total_seconds() / 60.0)

    def superseded_by(self, latest_bar: datetime | None) -> bool:
        if latest_bar is None or self.latest_bar is None:
            return False
        return latest_bar > self.latest_bar

    def summary(self, now: datetime) -> str:
        return (
            f"{self.symbol}: broker history unusable ({self.failures}x); "
            f"re-checked on its next new bar, or in {self.minutes_left(now):.0f} min. "
            f"{self.reason}"
        )


class DataQuarantine:
    """Remember which symbols the broker cannot supply usable bars for.

    Backs off geometrically. A symbol with one bad fetch is worth another look
    within the hour; a symbol that has failed four cycles running is offering
    eight closed weekly bars where fifty are needed, and asking again every
    hour for the rest of the year buys nothing.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        initial_minutes: float = 30.0,
        backoff_multiple: float = 4.0,
        max_minutes: float = 240.0,
    ) -> None:
        self.enabled = enabled
        self.initial_minutes = initial_minutes
        self.backoff_multiple = backoff_multiple
        self.max_minutes = max_minutes
        self._holds: dict[str, Hold] = {}

    def hold_for(
        self, symbol: str, now: datetime, latest_bar: datetime | None = None
    ) -> Hold | None:
        """The live hold on this symbol, or None if it should be analysed.

        A new bar releases it immediately, ahead of the clock. That is the
        answer to "what if the exchange opens again tomorrow": the hold is not
        really a timer, it is a bet that nothing has changed, and the first bar
        the venue prints settles that bet. The clock is only the backstop for
        when the scan could not report a bar at all.
        """
        if not self.enabled:
            return None
        hold = self._holds.get(symbol)
        if hold is None:
            return None
        if hold.superseded_by(latest_bar):
            return None
        if now >= hold.until:
            # Expired. The failure count is kept: a symbol that has failed four
            # times and is about to fail a fifth should go straight back to the
            # long hold rather than restart at the initial window.
            return None
        return hold

    def record_failure(
        self,
        symbol: str,
        reason: str,
        now: datetime,
        latest_bar: datetime | None = None,
    ) -> Hold:
        """Hold this symbol, longer each time it fails again."""
        previous = self._holds.get(symbol)
        failures = (previous.failures if previous else 0) + 1
        minutes = min(
            self.max_minutes,
            self.initial_minutes * (self.backoff_multiple ** (failures - 1)),
        )
        hold = Hold(
            symbol=symbol,
            until=now + timedelta(minutes=minutes),
            # Kept short: this ends up in a journal row and on the deck, and the
            # full pandas message is three lines of noise there.
            reason=reason.strip().split("\n")[0][:200],
            failures=failures,
            latest_bar=latest_bar,
        )
        self._holds[symbol] = hold
        log.info(
            "symbol held out of deep analysis",
            extra={
                "event": "data_quarantine",
                "symbol": symbol,
                "failures": failures,
                "minutes": round(minutes, 1),
                "latest_bar": latest_bar.isoformat() if latest_bar else None,
                "reason": hold.reason,
            },
        )
        return hold

    def clear(self, symbol: str) -> None:
        """A symbol that analysed cleanly is not held, and owes nothing.

        The failure count goes with it. A broker that backfills history has
        genuinely fixed the problem, and carrying the old count would put the
        symbol straight back on a day-long hold at its next unrelated hiccup.
        """
        if self._holds.pop(symbol, None) is not None:
            log.info(
                "symbol released back into deep analysis",
                extra={"event": "data_quarantine_cleared", "symbol": symbol},
            )

    def held(self, now: datetime) -> tuple[Hold, ...]:
        """Every live hold, soonest to return first. For the deck and the log."""
        live = [hold for hold in self._holds.values() if now < hold.until]
        return tuple(sorted(live, key=lambda hold: hold.until))

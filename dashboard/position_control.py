"""Operator control over positions that already exist — and nothing more.

`DashboardService` is read-only on purpose: the dashboard must not become a
second way to open risk that skips the scanner, the filters, the sizer and the
AI gate. That rule is worth keeping, and this module does not weaken it. There
is no `open` here and there never should be.

What it does add is control over positions the engine already opened. The
operator was in the position of watching a live trade on their own money with
no way to touch it except the hard STOP, which flattens everything. "Close this
one" and "move this stop up" are ordinary, legitimate acts of management, and
the absence of them was pushing the operator toward the blunt instrument.

The invariant that survives: **risk per position may not grow.** Closing, part-
closing and tightening are unconditional — they can only reduce exposure.
Widening a stop is the one operation that increases it, and it is refused when
the result would exceed the configured ceiling. That ceiling is the same number
the position sizer used to open the trade, so the dashboard cannot be used to
retroactively take a trade the risk layer would never have sized.

Targets are unrestricted in both directions. A target does not bound loss, so
moving one cannot breach a risk limit — it only chooses when to stop being
right.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.schema import Settings
from core.broker import Broker
from core.types import Direction, Position


@dataclass(frozen=True, slots=True)
class ControlOutcome:
    ok: bool
    message: str


@dataclass(frozen=True, slots=True)
class RiskPreview:
    """What a proposed stop would mean, in money, before it is sent."""

    valid: bool
    risk_money: float
    risk_pct: float
    permitted: bool
    detail: str


class PositionControl:
    """Close and adjust existing positions, under a risk ceiling."""

    def __init__(self, broker: Broker, settings: Settings) -> None:
        self.broker = broker
        self.settings = settings

    # ---------------------------------------------------------------- checks

    def preview_stop(self, position: Position, sl: float, equity: float) -> RiskPreview:
        """Price a proposed stop against the account before anything is sent.

        Shown next to the input rather than only on failure. An operator typing
        a number into a live position should see "this is 3.80 EUR, 3.8% of the
        account" while typing it, not discover it afterwards.
        """
        sign = int(position.direction)
        if sl <= 0:
            return RiskPreview(False, 0.0, 0.0, False, "A position must keep a stop loss.")
        if (position.price_open - sl) * sign <= 0:
            side = "below" if position.direction is Direction.LONG else "above"
            return RiskPreview(
                False, 0.0, 0.0, False, f"A {position.direction.name} stop must sit {side} entry."
            )
        spec = self.broker.spec(position.symbol)
        distance = abs(position.price_open - sl)
        risk_money = spec.money_per_lot(distance) * position.volume
        risk_pct = (risk_money / equity * 100.0) if equity > 0 else 0.0
        ceiling = self.settings.effective_max_risk_pct()
        widening = abs(position.price_open - sl) > abs(position.price_open - position.sl) + 1e-12
        if widening and risk_pct > ceiling + 1e-9:
            return RiskPreview(
                True,
                risk_money,
                risk_pct,
                False,
                f"Refused: this widens the stop to {risk_pct:.2f}% of equity, above the "
                f"{ceiling:.2f}% ceiling this account trades under. Tightening is always "
                f"allowed; widening past the limit is not.",
            )
        verb = "widens" if widening else "tightens"
        return RiskPreview(
            True,
            risk_money,
            risk_pct,
            True,
            f"{verb.capitalize()} risk to {risk_money:.2f} ({risk_pct:.2f}% of equity).",
        )

    # ---------------------------------------------------------------- action

    def modify(
        self,
        position: Position,
        *,
        sl: float,
        tp: float,
        equity: float,
    ) -> ControlOutcome:
        """Set a new stop and target on a live position."""
        preview = self.preview_stop(position, sl, equity)
        if not preview.valid or not preview.permitted:
            return ControlOutcome(False, preview.detail)
        spec = self.broker.spec(position.symbol)
        invalid = self._target_invalid(position, tp)
        if invalid:
            return ControlOutcome(False, invalid)
        result = self.broker.modify_stops(
            position,
            sl=spec.normalize_price(sl),
            tp=spec.normalize_price(tp) if tp > 0 else 0.0,
        )
        if not result.ok:
            return ControlOutcome(False, f"Broker refused: {result.retcode_name} {result.comment}")
        return ControlOutcome(
            True,
            f"#{position.ticket} updated. {preview.detail}",
        )

    def close(self, position: Position, volume: float | None = None) -> ControlOutcome:
        """Close all of a position, or a part of it."""
        spec = self.broker.spec(position.symbol)
        if volume is not None:
            rounded = spec.round_volume_down(volume)
            if rounded < spec.volume_min:
                return ControlOutcome(
                    False,
                    f"{volume:g} lots rounds below the {spec.volume_min:g} minimum.",
                )
            # The exact remainder, deliberately not rounded. Rounding it down
            # turns an unclosable 0.005-lot stub into a reassuring 0.0 and the
            # check passes on a position the broker will then refuse to leave.
            remaining = position.volume - rounded
            if 1e-9 < remaining < spec.volume_min - 1e-9:
                return ControlOutcome(
                    False,
                    f"That would leave {remaining:g} lots, below the {spec.volume_min:g} "
                    f"minimum. Close the whole position instead.",
                )
            volume = rounded
        result = self.broker.close_position(position, volume)
        if not result.ok:
            return ControlOutcome(False, f"Broker refused: {result.retcode_name} {result.comment}")
        closed = self.broker.closed_position(position.ticket)
        booked = (
            f" Realised {closed.pnl_money:+.2f}."
            if closed is not None
            else " Awaiting deal history for the final figure."
        )
        what = f"{result.filled_volume:g} lots of" if volume is not None else "all of"
        return ControlOutcome(True, f"Closed {what} #{position.ticket}.{booked}")

    def close_all(self, positions: list[Position]) -> list[ControlOutcome]:
        """Flatten the book. Distinct from the kill switch, which also halts."""
        return [self.close(position) for position in positions]

    @staticmethod
    def _target_invalid(position: Position, tp: float) -> str:
        if tp <= 0:
            # Zero is MT5's "no target", which is a legitimate choice: it means
            # the trade is managed to an exit rather than to a price.
            return ""
        sign = int(position.direction)
        if (tp - position.price_open) * sign <= 0:
            side = "above" if position.direction is Direction.LONG else "below"
            return f"A {position.direction.name} target must sit {side} entry, or be 0 for none."
        return ""

"""Position sizing.

The whole module exists to answer one question honestly:

    Given this equity, this risk budget and this structural stop, how many
    lots — and if the answer is "fewer than the broker will accept", say so
    instead of trading anyway.

That last clause is the entire point. The common failure is to compute 0.004
lots, notice the broker's minimum is 0.01, and send 0.01 "because it is close
enough". It is not close enough: it is 2.5x the intended risk, and it is the
mechanism by which small accounts die while their owner believes they are
risking 1%.

Nothing here reads the config directly — `Settings` is passed in — and nothing
here talks to the terminal. It is pure arithmetic over an `InstrumentSpec`,
which is what makes it exhaustively testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.schema import Settings
from core.instrument import InstrumentSpec
from core.types import Direction
from infra.logging import get_logger
from risk.reasons import Reason, RiskDecision

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SizingResult:
    """Outcome of a sizing attempt, approved or not.

    Every field is journalled. On a rejection the numbers explain *how far off*
    the trade was, which is what makes "is my account too small" answerable
    from data instead of intuition.
    """

    decision: RiskDecision
    symbol: str
    direction: Direction
    volume: float
    entry: float
    sl: float
    tp: float

    #: Risk the sizer was aiming for, before lot rounding.
    intended_risk_money: float
    intended_risk_pct: float
    #: Risk actually taken at the rounded volume. Always <= intended, because
    #: volume rounds down.
    actual_risk_money: float
    actual_risk_pct: float

    sl_distance_price: float
    sl_distance_pips: float
    reward_risk: float
    #: Lots the maths asked for, before rounding to the broker's step.
    raw_volume: float

    @property
    def approved(self) -> bool:
        return self.decision.approved

    @property
    def reason(self) -> Reason:
        return self.decision.reason

    def journal_row(self) -> dict[str, object]:
        """Flat dict for the journal. Kept explicit so the schema is visible."""
        return {
            "approved": self.approved,
            "reason": str(self.reason),
            "detail": self.decision.detail,
            "symbol": self.symbol,
            "direction": self.direction.name,
            "volume": self.volume,
            "raw_volume": self.raw_volume,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "intended_risk_money": self.intended_risk_money,
            "intended_risk_pct": self.intended_risk_pct,
            "actual_risk_money": self.actual_risk_money,
            "actual_risk_pct": self.actual_risk_pct,
            "sl_distance_pips": self.sl_distance_pips,
            "reward_risk": self.reward_risk,
        }


class PositionSizer:
    """Turns a structural setup into a lot size, or into a documented refusal."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def size(
        self,
        *,
        spec: InstrumentSpec,
        equity: float,
        direction: Direction,
        entry: float,
        sl: float,
        tp: float = 0.0,
        risk_multiplier: float = 1.0,
    ) -> SizingResult:
        """Compute the lot size for one setup.

        Args:
            risk_multiplier: anti-martingale scaling from `RiskManager`. Must be
                <= 1.0 — increasing risk after a loss is a forbidden practice
                and is rejected here rather than trusted to the caller.
        """
        if risk_multiplier > 1.0:
            raise ValueError(
                f"risk_multiplier {risk_multiplier} > 1.0 would increase risk; "
                "scaling up after losses is forbidden"
            )

        sl_distance = abs(entry - sl)
        sl_pips = spec.price_to_pips(sl_distance)
        reward_risk = self._reward_risk(direction, entry, sl, tp)

        # What one lot costs if this trade is wrong: the price move to the
        # stop, plus the round-trip commission that is charged either way.
        # Computed here rather than at the division below so the recorded
        # `actual_risk_money` means the same thing — that number becomes
        # `risk_money` in the journal and is the denominator of every R the
        # account will ever report.
        commission_per_lot = self.settings.risk.commission_per_lot(spec.asset_class.value)
        cost_per_lot = spec.money_per_lot(sl_distance) + commission_per_lot

        def result(decision: RiskDecision, volume: float = 0.0, raw: float = 0.0) -> SizingResult:
            return SizingResult(
                decision=decision,
                symbol=spec.symbol,
                direction=direction,
                volume=volume,
                entry=entry,
                sl=sl,
                tp=tp,
                intended_risk_money=intended_money,
                intended_risk_pct=intended_pct,
                actual_risk_money=cost_per_lot * volume,
                actual_risk_pct=(100.0 * cost_per_lot * volume / equity if equity > 0 else 0.0),
                sl_distance_price=sl_distance,
                sl_distance_pips=sl_pips,
                reward_risk=reward_risk,
                raw_volume=raw,
            )

        intended_pct = self.settings.effective_risk_pct() * risk_multiplier
        intended_money = equity * intended_pct / 100.0

        # -- 1. the stop has to be a real stop ----------------------------
        invalid = self._validate_stop(direction, entry, sl, sl_distance)
        if invalid is not None:
            return result(invalid)

        if not spec.is_tradable:
            return result(
                RiskDecision.block(
                    Reason.SYMBOL_NOT_TRADABLE,
                    f"{spec.symbol}: broker reports trade_mode={spec.trade_mode}",
                )
            )

        # -- 2. broker and mode constraints on the stop distance ----------
        if spec.violates_stop_level(entry, sl):
            return result(
                RiskDecision.block(
                    Reason.SL_TOO_TIGHT_FOR_BROKER,
                    f"stop is {sl_distance / spec.point:.0f} points from entry, inside the "
                    f"broker's {spec.stops_level}-point stop level "
                    f"({spec.price_to_pips(spec.min_stop_distance_price):.1f} pips)",
                )
            )

        # The mode's pip ceiling is an FX-shaped rule and is applied only to
        # FX. A "pip" on gold is one point ($0.01), so a 30-pip ceiling there
        # would mean a 30-cent stop — absurd, and it would mask the real
        # constraint. For non-FX instruments the money-based undercapitalised
        # check below is the binding guard, and it is exact rather than a proxy.
        max_sl_pips = self.settings.active_limits.max_sl_pips
        if spec.is_forex and sl_pips > max_sl_pips:
            return result(
                RiskDecision.block(
                    Reason.SL_TOO_WIDE_FOR_ACCOUNT,
                    f"structural stop is {sl_pips:.1f} pips, above the "
                    f"{self.settings.mode.value} ceiling of {max_sl_pips:.0f} pips",
                )
            )

        # Is the stop wide enough that the trade, and not the cost of taking
        # it, decides the outcome?
        #
        # A live AUDNZD long had a 5-pip stop, which the broker accepted and
        # every gate passed. It returned -1.48R on a -1.00R plan: commission
        # was 0.22R and the fill came 1.7 pips *through* the stop for another
        # 0.34R. Over half the risk was cost. At that ratio the strategy is not
        # what is being tested.
        #
        # Placed after the broker's own stop level and before reward:risk,
        # because it is the same kind of statement — a fact about this
        # instrument that no quality of setup can argue with.
        cost_share = self._cost_share(spec, sl_distance, commission_per_lot)
        limit = self.settings.risk.max_cost_share_of_risk
        if limit > 0 and cost_share > limit:
            return result(
                RiskDecision.block(
                    Reason.SL_TOO_TIGHT_FOR_COSTS,
                    f"commission and slippage would be {cost_share:.0%} of the risk on a "
                    f"{sl_pips:.1f} pip stop, above the {limit:.0%} limit; a full stop-out "
                    f"would cost about {1 + cost_share:.2f}R rather than 1.00R",
                )
            )

        # -- 3. reward:risk on structural levels --------------------------
        minimum_rr = self.settings.risk.min_risk_reward
        if tp and reward_risk < minimum_rr:
            return result(
                RiskDecision.block(
                    Reason.RR_BELOW_MINIMUM,
                    f"reward:risk is 1:{reward_risk:.2f}, below the required "
                    f"1:{minimum_rr:.2f} ({sl_pips:.1f} pip stop vs "
                    f"{spec.price_to_pips(abs(tp - entry)):.1f} pip target)",
                )
            )

        # -- 4. the arithmetic --------------------------------------------
        money_per_lot = spec.money_per_lot(sl_distance)
        if money_per_lot <= 0:  # pragma: no cover - guarded in InstrumentSpec
            return result(
                RiskDecision.block(
                    Reason.INVALID_STOP, f"{spec.symbol}: zero money-per-lot at this stop distance"
                )
            )

        # A live AUDNZD stop-out cost EUR 1.93 against a modelled 1R of EUR
        # 1.53, and EUR 0.33 of that gap was commission, confirmed against the
        # deal in the terminal. Every threshold in this system is written in R,
        # so an R missing a fifth of the real loss arms the give-back late,
        # makes the profit lock secure less than it claims, and flatters every
        # expectancy figure the account will ever produce.
        #
        # It also prices tight stops honestly for the first time. Commission is
        # fixed per lot, so on a 2-pip stop it is a third of the risk and on a
        # 20-pip stop it is a rounding error — which makes scalp-width stops
        # unattractive by arithmetic instead of by a threshold someone guessed.
        #
        # Both terms scale linearly with volume, so solving for it stays a
        # single division rather than an iteration.
        raw_volume = intended_money / cost_per_lot
        volume = spec.round_volume_down(min(raw_volume, spec.volume_max))

        # -- 5. can this account express the trade at all? ----------------
        if volume < spec.volume_min:
            shortfall_pct = spec.min_risk_pct(sl_distance, equity)
            affordable_pips = spec.max_sl_pips_for_risk(intended_money)
            return result(
                RiskDecision.block(
                    Reason.UNDERCAPITALIZED,
                    f"{intended_pct:.2f}% of {equity:.2f} is {intended_money:.2f}, which buys "
                    f"{raw_volume:.4f} lots — below the {spec.volume_min:g} minimum. The "
                    f"smallest tradable position would risk {shortfall_pct:.2f}%. At this "
                    f"risk level the widest affordable stop is {affordable_pips:.1f} pips, "
                    f"and this setup needs {sl_pips:.1f}.",
                ),
                raw=raw_volume,
            )

        # -- 6. final ceiling check ---------------------------------------
        # Rounding down can only reduce risk, so this should never fire. It is
        # here because "should never fire" is exactly the assumption that turns
        # into an incident, and the cost of the check is nothing.
        actual_money = cost_per_lot * volume
        actual_pct = 100.0 * actual_money / equity if equity > 0 else float("inf")
        ceiling = self.settings.effective_max_risk_pct()
        if actual_pct > ceiling + 1e-9:
            return result(
                RiskDecision.block(
                    Reason.RISK_EXCEEDS_CAP,
                    f"{volume:g} lots would risk {actual_pct:.2f}%, above the "
                    f"{ceiling:.2f}% ceiling for {self.settings.mode.value}",
                ),
                volume=0.0,
                raw=raw_volume,
            )

        capped = raw_volume > spec.volume_max
        detail = (
            f"{volume:g} lots risks {actual_money:.2f} ({actual_pct:.2f}%) over a "
            f"{sl_pips:.1f} pip stop, R:R 1:{reward_risk:.2f}"
        )
        if capped:
            detail += f" — capped at the broker's {spec.volume_max:g} lot maximum"

        log.info(
            "position sized",
            extra={
                "event": "sizing",
                "symbol": spec.symbol,
                "direction": direction.name,
                "volume": volume,
                "raw_volume": round(raw_volume, 6),
                "intended_risk_money": round(intended_money, 2),
                "actual_risk_money": round(actual_money, 2),
                "actual_risk_pct": round(actual_pct, 4),
                "sl_pips": round(sl_pips, 2),
                "reward_risk": round(reward_risk, 3),
            },
        )
        return result(RiskDecision.allow(detail), volume=volume, raw=raw_volume)

    # -- helpers -----------------------------------------------------------

    def _cost_share(
        self, spec: InstrumentSpec, sl_distance: float, commission_per_lot: float
    ) -> float:
        """Commission plus expected slippage, as a fraction of the price risk.

        Per lot on both terms, so the volume cancels and the answer depends
        only on the instrument and how wide the stop is — which is the whole
        point. The same commission is a fifth of a 2-pip stop and a rounding
        error on a 40-pip one.

        Slippage is counted in price rather than as a ratio because that is how
        it was measured: a stop at 1.19722 filled at 1.19705. Multiplying it by
        `money_per_lot` puts it in the same units as everything else.
        """
        price_risk = spec.money_per_lot(sl_distance)
        if price_risk <= 0:
            return 1.0
        slip_pips = self.settings.risk.stop_slippage_pips.get(spec.asset_class.value, 0.0)
        slip_cost = spec.money_per_lot(spec.pips_to_price(slip_pips)) if slip_pips > 0 else 0.0
        return (commission_per_lot + slip_cost) / price_risk

    @staticmethod
    def _validate_stop(
        direction: Direction, entry: float, sl: float, sl_distance: float
    ) -> RiskDecision | None:
        """Reject a stop that is missing, at entry, or on the wrong side."""
        if sl <= 0.0:
            return RiskDecision.block(
                Reason.INVALID_STOP, "no stop loss supplied; trading without a stop is forbidden"
            )
        if sl_distance <= 0.0:
            return RiskDecision.block(Reason.INVALID_STOP, "stop loss equals the entry price")
        if direction is Direction.LONG and sl >= entry:
            return RiskDecision.block(
                Reason.INVALID_STOP, f"long stop {sl} is at or above entry {entry}"
            )
        if direction is Direction.SHORT and sl <= entry:
            return RiskDecision.block(
                Reason.INVALID_STOP, f"short stop {sl} is at or below entry {entry}"
            )
        return None

    @staticmethod
    def _reward_risk(direction: Direction, entry: float, sl: float, tp: float) -> float:
        """Reward-to-risk from structural levels, 0.0 when no target is set.

        A target on the wrong side of entry yields 0.0 rather than a negative
        ratio, so the R:R gate rejects it like any other unacceptable setup.
        """
        risk = abs(entry - sl)
        if risk <= 0 or not tp:
            return 0.0
        reward = (tp - entry) * int(direction)
        return max(reward, 0.0) / risk

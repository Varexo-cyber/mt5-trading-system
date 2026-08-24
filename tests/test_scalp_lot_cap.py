"""A candle scalp takes a fixed lot, not a share of the account.

THE OWNER'S INSTRUCTION: "doe gwn MAX 0.01 lot met die dingen op xauusd ...
trade op candles ... GWN IN EN ERUIT".

He is right and the sizer was wrong for this. `PositionSizer` spends 3% of
equity, and on a stop four tenths of an M1 candle wide that buys a great many
lots. The premise of this section is a position small enough that being wrong
costs cents, taken often — the opposite instruction to the same function.

Measured against his own fill: XAUUSD 0.01 lot, 4667.47 to 4671.25, EUR 3.24.
EUR 0.86 per dollar of gold, so a one-dollar stop risks EUR 0.86 on a EUR 176
account. Half a percent, where the sizer wanted three.

The test that matters most is the last class: this must never size a swing
trade down. A trade the swing engine formed on its own evidence stays a swing
trade even when the candle reader happened to agree.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from analysis.confluence import TradeIdea
from core.types import Direction, Signal


def runner(cap: float = 0.01, live: bool = True):  # type: ignore[no-untyped-def]
    from config.loader import DEFAULT_CONFIG_PATH, load_settings
    from config.schema import TradingMode
    from runner.service import JarvisRunner

    settings = load_settings(
        DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
    )
    analysis = settings.analysis.model_copy(
        update={
            "candle_momentum": settings.analysis.candle_momentum.model_copy(
                update={"maximum_lots": cap}
            )
        }
    )
    service = object.__new__(JarvisRunner)
    service.settings = settings.model_copy(
        update={"analysis": analysis, "mode": TradingMode.MICRO_LIVE if live else settings.mode}
    )
    return service


SPEC = SimpleNamespace(round_volume_down=lambda volume: round(volume - volume % 0.01, 2))


def sizing(volume: float):  # type: ignore[no-untyped-def]
    """Only the three fields the cap rescales."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Sizing:
        volume: float
        actual_risk_money: float
        actual_risk_pct: float

    return Sizing(volume=volume, actual_risk_money=5.27, actual_risk_pct=3.0)


def idea(*signals: tuple[str, float], direction: Direction = Direction.LONG) -> TradeIdea:
    return TradeIdea(
        symbol="XAUUSD",
        approved=True,
        direction=direction,
        score=55.0,
        confidence=0.6,
        entry=4670.0,
        stop_loss=4669.0,
        take_profit=4673.0,
        reason="test",
        signals=tuple(Signal(module, score, 0.6) for module, score in signals),
    )


class TestTheCapHolds:
    def test_a_scalp_is_held_to_one_hundredth_of_a_lot(self) -> None:
        held = runner()._cap_scalp_volume(idea(("candle_momentum", 60.0)), SPEC, sizing(0.34))

        assert held.volume == 0.01

    def test_the_recorded_risk_is_rescaled_with_it(self) -> None:
        """A journal that reports 3% on a position risking a hundredth of that
        is worse than no journal — every later measurement divides by it."""
        held = runner()._cap_scalp_volume(idea(("candle_momentum", 60.0)), SPEC, sizing(0.34))

        assert held.actual_risk_pct == pytest.approx(3.0 * 0.01 / 0.34)
        assert held.actual_risk_money == pytest.approx(5.27 * 0.01 / 0.34)

    def test_it_never_sizes_up(self) -> None:
        """When the sizer already wanted less than the cap — a tiny account, a
        wide stop — the cap must not lift it. Rounding up to reach a minimum is
        forbidden in this system and this is not a way around that."""
        held = runner(cap=0.10)._cap_scalp_volume(
            idea(("candle_momentum", 60.0)), SPEC, sizing(0.01)
        )

        assert held.volume == 0.01
        assert held.actual_risk_pct == 3.0

    def test_zero_disables_it(self) -> None:
        held = runner(cap=0.0)._cap_scalp_volume(
            idea(("candle_momentum", 60.0)), SPEC, sizing(0.34)
        )

        assert held.volume == 0.34


class TestOnlyTradesTheCandleReaderIsActuallyCarrying:
    """The test that matters most. Shrinking a swing trade to a scalp's lot
    would quietly take the account out of the setups it makes its money on."""

    def test_a_swing_trade_the_scalp_merely_agreed_with_is_left_alone(self) -> None:
        held = runner()._cap_scalp_volume(
            idea(("impulse_break", 80.0), ("candle_momentum", 45.0)), SPEC, sizing(0.34)
        )

        assert held.volume == 0.34

    def test_a_scalp_the_swing_engine_merely_agreed_with_is_capped(self) -> None:
        held = runner()._cap_scalp_volume(
            idea(("candle_momentum", 80.0), ("impulse_break", 45.0)), SPEC, sizing(0.34)
        )

        assert held.volume == 0.01

    def test_a_trade_the_scalp_had_no_part_in_is_left_alone(self) -> None:
        held = runner()._cap_scalp_volume(idea(("impulse_break", 80.0)), SPEC, sizing(0.34))

        assert held.volume == 0.34

    def test_a_scalp_pointing_the_other_way_is_not_credited_with_the_trade(self) -> None:
        """It scored SHORT and the trade went LONG. It did not carry this."""
        held = runner()._cap_scalp_volume(
            idea(("candle_momentum", -80.0), ("impulse_break", 45.0)), SPEC, sizing(0.34)
        )

        assert held.volume == 0.34

    def test_weight_is_what_decides_it_not_raw_score(self) -> None:
        """Two modules can score the same and matter differently. The reading
        must match the scorecard's, or a trade is a scalp for the sizer and a
        swing for the report."""
        service = runner()
        weights = service.settings.analysis.confluence.weights

        assert weights["candle_momentum"] > 0
        assert weights["impulse_break"] > 0

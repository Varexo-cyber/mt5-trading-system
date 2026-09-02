"""The broker minimum never grants permission to round risk upward."""

from __future__ import annotations

import pytest

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from config.schema import TradingMode


def _live_settings(**risk_changes):
    settings = load_settings(
        DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
    )
    settings = settings.model_copy(
        update={"system": settings.system.model_copy(update={"mode": TradingMode.MICRO_LIVE})}
    )
    if risk_changes:
        settings = settings.model_copy(
            update={"risk": settings.risk.model_copy(update=risk_changes)}
        )
    return settings


class TestMinimumLotCannotOverrideTheStake:
    def test_it_is_disabled_on_the_live_account(self) -> None:
        assert _live_settings().risk.allow_minimum_lot_above_target is False

    def test_the_base_config_leaves_it_off(self) -> None:
        """The hard rule is identical in the base and account overlay."""
        assert load_settings(env_overrides=False).risk.allow_minimum_lot_above_target is False

    def test_yaml_cannot_turn_the_unsafe_override_back_on(self) -> None:
        from pydantic import ValidationError

        from config.schema import RiskConfig

        with pytest.raises(ValidationError):
            RiskConfig(allow_minimum_lot_above_target=True)


class TestOnlySectionsTwoAndThreeTradeRealMoney:
    def test_the_live_list_is_the_two_measured_sections(self) -> None:
        """Two, then one on 30 August, and two again on 31 August.

        I removed impulse_retest on THIRTY days -- 48 trades, 39.6%, -1.44
        sigma -- and the 180-day measurement on the exit the account actually
        takes said the opposite:

                           trades   win     R/trade   EUR       days green
            impulse_retest    358  81.3%    +0.058   +126.73   89/116
            order_block      1283  76.8%    +0.026   +125.93   83/140

        Twice the edge per trade from a quarter of the trades. The sample I
        removed it on was seven times smaller and I let it weigh more than it
        should have.

        NEITHER IS PROVEN. order_block reads +1.33 sigma with 78% of six
        months coming from August alone, and impulse_retest's sigma was never
        printed at all because the report skips the verdict for a section that
        is not on this list. Both are live because the owner decided so on the
        numbers above, not because either cleared a bar.

        THIRD SECTION ADDED 31 AUGUST: `order_block_fast`, the same detector
        as `order_block` on M1 instead of M30, as its own instance with its own
        name, weight and breaker.

        It is the weakest evidence on the account and
        `docs/hypotheses/order_block_fast.md` says so in full: 105 trades over
        14 days, five markets, the adjacent M5 clock negative over 308 trades,
        and the run that produced the number did not charge the trades their
        spread. Live at the owner's explicit instruction with those four facts
        on the screen -- "risico's moeten genomen worden om te testen" -- under
        the strictest breaker here (30 trades, 50% loss share, 7-streak).
        """
        confluence = _live_settings().analysis.confluence

        assert set(confluence.live_enabled_modules) == {
            "failed_session_breakout",
            "section_five_m5",
            "section_six_gold_m5",
            "section_eight_trend_day_h1",
            "section_nine_vwap_m30",
            "section_ten_gold_m1",
        }
        for name in (
            "impulse_retest",
            "impulse_retest_m30",
            "order_block_m15",
            "order_block",
            "order_block_fast",
            "order_block_h1",
        ):
            assert confluence.weights.get(name, 0.0) > 0.0

    def test_the_unmeasured_sections_are_off(self) -> None:
        """Section 1 (market_structure, trend_momentum, m1_micro_breakout),
        section 5 (basket_divergence) and section 6 (candle_momentum). Two of
        those were never measured at all and one measured -0.365R over 163
        live trades."""
        live = set(_live_settings().analysis.confluence.live_enabled_modules)

        for module in (
            "market_structure",
            "trend_momentum",
            "m1_micro_breakout",
            "basket_divergence",
            "candle_momentum",
        ):
            assert module not in live, f"{module} may not trade real money"

    def test_they_keep_their_weights_so_the_backtest_still_grades_them(self) -> None:
        """Switched off is not deleted. A module with no weight cannot be
        measured, and that is how three M1 detectors went unjudged for months."""
        weights = _live_settings().analysis.confluence.weights

        assert weights.get("trend_momentum", 0.0) > 0.0
        assert weights.get("market_structure", 0.0) > 0.0


class TestTheCeilingIsTenAndAllThreeKnobsAgree:
    """Raised from 8% to 10% on 30 August at the owner's request.

    THE CONFIG WARNS ABOUT THIS EXACT CHANGE IN ITS OWN COMMENT: there are
    THREE ceilings and they only work together. `risk.max_risk_per_trade_pct`,
    `risk.conviction_risk.ceiling_pct` and
    `modes.micro_live.max_risk_per_trade_pct` each clamp the stake, so raising
    two of the three is a silent no-op -- in the words already written there,
    "de config zegt 9%, de rekening handelt 6%".

    So the test is not "is the number 10". It is "does the number the sizer
    actually enforces come out at 10", which is the only version that would
    have caught the half-done edit.
    """

    def test_the_enforced_ceiling_is_ten(self) -> None:
        settings = _live_settings()

        assert settings.effective_max_risk_pct() == pytest.approx(10.0)

    def test_all_three_knobs_were_moved(self) -> None:
        settings = _live_settings()

        assert settings.risk.max_risk_per_trade_pct == pytest.approx(10.0)
        assert settings.risk.conviction_risk.ceiling_pct == pytest.approx(10.0)

    def test_the_ordinary_stake_is_untouched(self) -> None:
        """A ceiling, not a stake. An ordinary approval is still 2%; only
        conviction scaling and the minimum-lot rounding may reach past it."""
        settings = _live_settings()

        assert settings.effective_risk_pct() == pytest.approx(2.0)
        assert settings.risk.conviction_risk.floor_pct == pytest.approx(2.0)

    def test_what_it_means_in_euros_on_this_account(self) -> None:
        """The number worth reading before agreeing to it. The thirty-day
        measurement put 75 of 255 trades above the 2% target because the broker
        minimum forced them there, topping out at 7.89%. That group now has
        room up to 10%, and there is no daily loss limit under it."""
        settings = _live_settings()
        equity = 215.34

        assert equity * settings.effective_risk_pct() / 100 == pytest.approx(4.31, abs=0.01)
        assert equity * settings.effective_max_risk_pct() / 100 == pytest.approx(21.53, abs=0.01)
        # Two such trades at once is a fifth of the account on two stops.
        assert settings.effective_max_positions(equity) * 10.0 >= 20.0

    def test_the_base_config_is_not_dragged_along(self) -> None:
        """This account's decision at this equity, not everyone's default."""
        assert load_settings(env_overrides=False).risk.max_risk_per_trade_pct < 10.0

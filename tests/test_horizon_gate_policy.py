"""Quick trades are judged on quick evidence without turning safety gates off."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from analysis.confluence import TradeIdea
from analysis.target_reach import ReachVerdict
from config.loader import load_settings
from core.types import Direction, MarketContext, Series, Signal, Timeframe
from runner.service import JarvisRunner


def runner() -> JarvisRunner:
    instance = JarvisRunner.__new__(JarvisRunner)
    instance.settings = load_settings(env_overrides=False)  # type: ignore[assignment]
    return instance


def idea(*, horizon: str, family: str, signals: tuple[Signal, ...] = ()) -> TradeIdea:
    return TradeIdea(
        symbol="EURUSD.i",
        approved=True,
        direction=Direction.LONG,
        score=70.0,
        confidence=0.7,
        entry=1.10,
        stop_loss=1.099,
        take_profit=1.102,
        reason="test",
        signals=signals,
        setup_family=family,
        horizon=horizon,
        planning_timeframe="M5" if horizon == "quick" else "H1",
        expected_horizon_minutes=30 if horizon == "quick" else 1440,
    )


def context() -> MarketContext:
    now = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    series: dict[Timeframe, Series] = {}
    for timeframe, frequency in ((Timeframe.M1, "1min"), (Timeframe.M5, "5min")):
        index = pd.date_range(end=now, periods=80, freq=frequency)
        close = 1.10 + np.arange(80) * 0.00001
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.0001,
                "low": close - 0.0001,
                "close": close,
                "tick_volume": 100,
                "spread": 10,
                "real_volume": 0,
            },
            index=index,
        )
        series[timeframe] = Series("EURUSD.i", timeframe, frame, now)
    return MarketContext("EURUSD.i", now, series, None)


def test_quick_only_softens_plausible_base_rate_failures() -> None:
    service = runner()
    quick = idea(horizon="quick", family="m1_micro_breakout_m1")
    swing = idea(horizon="swing", family="trend_momentum_swing")

    plausible = ReachVerdict(200, 20.0, 27.0, 33.3)
    fantasy = ReachVerdict(200, 10.0, 27.0, 33.3)

    assert service._reach_failure_is_advisory(quick, plausible)
    assert not service._reach_failure_is_advisory(quick, fantasy)
    assert not service._reach_failure_is_advisory(swing, plausible)


def test_large_direction_disadvantage_remains_hard() -> None:
    service = runner()
    quick = idea(horizon="quick", family="m1_micro_breakout_m1")

    assert service._direction_failure_is_advisory(quick, ReachVerdict(200, 30.0, 42.0, 33.3))
    assert not service._direction_failure_is_advisory(quick, ReachVerdict(200, 20.0, 45.0, 33.3))


def test_quick_relaxation_does_not_relax_adverse_reversal_limit() -> None:
    service = runner()
    quick = idea(horizon="quick", family="m1_micro_breakout_m1")
    swing = idea(horizon="swing", family="trend_momentum_swing")
    quick_config = service._entry_quality_config_for(quick)
    swing_config = service._entry_quality_config_for(swing)

    assert quick_config.max_favourable_extension_atr["forex"] > (
        swing_config.max_favourable_extension_atr["forex"]
    )
    assert quick_config.max_last_bar_adverse_atr == swing_config.max_last_bar_adverse_atr


def test_new_m1_event_gets_its_own_review_identity() -> None:
    service = runner()
    signal = Signal(
        "m1_micro_breakout",
        75.0,
        0.8,
        details={"timeframe": "M1"},
    )
    quick = idea(
        horizon="quick",
        family="m1_micro_breakout_m1",
        signals=(signal,),
    )

    key = service._review_key(quick, context())

    assert key is not None
    assert key[4] == "M1"


def test_broad_old_veto_does_not_silence_a_new_quick_event() -> None:
    service = runner()
    quick = idea(horizon="quick", family="m1_micro_breakout_m1")
    swing = idea(horizon="swing", family="trend_momentum_swing")

    assert not service._broad_veto_memory_applies(quick)
    assert service._broad_veto_memory_applies(swing)


def test_the_replay_can_judge_in_the_account_s_own_live_mode() -> None:
    """The offline tool measured a different engine than the account runs.

    `live_enabled_modules` is only consulted when the mode is live, and the
    replay hardcoded BACKTEST. So every offline figure counted every weighted
    detector while the account counted a subset — which made the tool built to
    answer "why did nothing trade" answer it about the wrong engine.
    """
    from backtesting.replay import HistoricalContextReplay
    from core.types import TradingMode

    engine = object()

    assert HistoricalContextReplay(engine).mode is TradingMode.BACKTEST  # type: ignore[arg-type]
    assert (
        HistoricalContextReplay(engine, mode=TradingMode.MICRO_LIVE).mode  # type: ignore[arg-type]
        is TradingMode.MICRO_LIVE
    )


def test_the_live_allowlist_follows_the_measured_record() -> None:
    """The allowlist follows the measured record, and on 26 August the record
    was read twice on one day.

    `modules.cmd --days 240` graded all nine detectors for the first time,
    over 2,192 proposals with the broker's own spreads. Every one came back
    negative, and the first response was to clear the allowlist. That lasted
    half an hour and it was wrong, for a reason printed in the same table:

        all eight: 54-57% win, avg winner +0.68R, avg loser -1.04R

    Eight readers that share no logic do not have eight private problems.
    They had one, and it was not theirs: `minimum_r_multiple` held the target
    at 0.72R against a 1.00R stop, a ratio needing 60.5% against the 54-57%
    they deliver. Switching off detectors for a planning floor is treating the
    symptom, so the floor moved to 0.35 and the detectors came back.

    `trend_momentum` stays off, and that is knowingly inconsistent: its own
    kill finding (-0.365R over 163 trades on twenty markets, t = -4.01) was
    also measured under the broken floor and is therefore just as suspect. It
    has never been live, and re-enabling a module on a theory is not the same
    risk as restoring one that was already running. It is first in line if the
    next run confirms the fix.

    NONE OF THIS IS PROOF. The eight figures were measured WITH the broken
    floor; that they turn positive without it is arithmetic on the give-back
    buckets, not a replay. One `modules.cmd` run settles it and this list is
    one edit either way.
    """
    from pathlib import Path

    from config.loader import load_settings

    root = Path(__file__).resolve().parents[1]
    confluence = load_settings(
        overlay=root / "config" / "eightcap.yaml", env_overrides=False
    ).analysis.confluence

    # Back on, once the planning floor that shaped their losses was fixed.
    restored = (
        "session_breakout",
        "ema_pullback_resume",
        "liquidity_sweep",
        "fast_ema_cross",
        "drift_continuation",
        "market_structure",
        "impulse_break",
    )
    for module in restored:
        assert module in confluence.live_enabled_modules, module
    # Never had a record worth restoring.
    assert "seasonality" not in confluence.live_enabled_modules
    # Still off, deliberately, and explained in the docstring.
    assert "trend_momentum" not in confluence.live_enabled_modules
    # The fix the restoration depends on. If either drifts back, the reason
    # these are on has gone with it.
    assert confluence.minimum_r_multiple <= 0.40
    assert confluence.max_spread_share_of_stop <= 0.10
    # A detector with no live figure of its own keeps the single global floor.
    assert (
        confluence.lone_floor_for("m1_micro_breakout") == confluence.lone_module_minimum_confidence
    )

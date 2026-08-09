"""Reading an open trade the way a person reads it.

The mechanical rules look at one number, R, which says nothing about why. This
is the layer that answers "is this still working" — and the layer with the most
room to do harm, because an over-eager exit destroys edge quietly. Two
properties carry the safety argument and both are asserted here: it may only
ever reduce risk, and it needs two independent signals before it may close
anything.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from analysis.position_health import (
    MIN_AGE_MINUTES,
    HealthSignal,
    adverse_run,
    assess_position,
    momentum_turned,
    spread_blowout,
    structure_broken,
)

LONG, SHORT = 1, -1


def frame(closes: list[float], *, spread_hl: float = 0.2) -> pd.DataFrame:
    """Bars from a list of closes, with a small symmetric high/low around each."""
    start = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + spread_hl for c in closes],
            "low": [c - spread_hl for c in closes],
            "close": closes,
        },
        index=pd.DatetimeIndex([start + timedelta(minutes=i) for i in range(len(closes))]),
    )


def flat(n: int = 40, value: float = 100.0) -> pd.DataFrame:
    return frame([value] * n)


# ------------------------------------------------------------- structure ---


def test_a_long_that_loses_its_swing_low_is_flagged() -> None:
    """The most informative thing that can happen to an open trade: the shape
    that made it worth taking is gone."""
    closes = [100, 101, 102, 99, 100, 101, 102, 103, 104, 103, 102, 101, 96.0]
    signal = structure_broken(frame([float(c) for c in closes]), LONG)
    assert signal is not None
    assert signal.name == "structure_broken"
    assert signal.severity > 0


def test_a_long_holding_its_swing_low_is_not_flagged() -> None:
    closes = [100, 101, 102, 99, 100, 101, 102, 103, 104, 103, 102, 101, 100.5]
    assert structure_broken(frame([float(c) for c in closes]), LONG) is None


def test_the_short_side_is_mirrored() -> None:
    closes = [100, 99, 98, 101, 100, 99, 98, 97, 96, 97, 98, 99, 104.0]
    assert structure_broken(frame(closes), SHORT) is not None
    assert structure_broken(frame(closes), LONG) is None


def test_a_flat_series_has_no_structure_to_break() -> None:
    """Every bar identical means no pivot is a strict extreme, so there is
    nothing to be through. Returning a signal here would fire on dead markets."""
    assert structure_broken(flat(), LONG) is None


def test_too_few_bars_is_silent_rather_than_wrong() -> None:
    assert structure_broken(frame([100.0, 101.0, 99.0]), LONG) is None


# -------------------------------------------------------------- momentum ---


def test_a_steady_drift_against_a_long_is_flagged() -> None:
    signal = momentum_turned(frame([100.0 - i * 0.3 for i in range(20)]), LONG)
    assert signal is not None
    assert signal.severity > 0


def test_a_drift_in_our_favour_is_not() -> None:
    assert momentum_turned(frame([100.0 + i * 0.3 for i in range(20)]), LONG) is None


def test_noise_around_flat_is_not_a_turn() -> None:
    """A slope inside the threshold is the market breathing, and reacting to it
    would close every trade that did not go straight up."""
    closes = [100.0 + (0.05 if i % 2 else -0.05) for i in range(20)]
    assert momentum_turned(frame(closes), LONG) is None


def test_one_spike_does_not_decide_the_slope() -> None:
    """Least squares rather than last-minus-first.

    Both series end the same distance below where they started, so a
    last-minus-first reading would rate them identically. A single print at the
    edge of the window is not the same event as twelve bars of steady pressure,
    and the slope is what tells them apart.
    """
    drift = momentum_turned(frame([100.0 - i * 0.5 for i in range(12)]), LONG)
    spike = momentum_turned(frame([100.0] * 11 + [94.5]), LONG)
    assert drift is not None
    assert spike is None or spike.severity < drift.severity


def test_severity_still_moves_once_the_reader_has_fired() -> None:
    """The first version scaled from the threshold and reached full severity at
    barely twice it, so on any trending market this reader sat pinned at 1.0.
    A reader that always says the same thing is not evidence.
    """
    mild = momentum_turned(frame([100.0 - i * 0.05 for i in range(12)]), LONG)
    hard = momentum_turned(frame([100.0 - i * 0.40 for i in range(12)]), LONG)
    assert mild is not None and hard is not None
    assert mild.severity < hard.severity


# ------------------------------------------------------------ adverse run ---


def test_four_of_five_bars_against_is_a_run() -> None:
    signal = adverse_run(frame([100.0, 99.5, 99.0, 98.5, 99.0, 98.0]), LONG)
    assert signal is not None
    assert "of the last 5" in signal.detail


def test_a_mixed_stretch_is_not_a_run() -> None:
    assert adverse_run(frame([100.0, 99.5, 100.0, 99.5, 100.0, 99.5]), LONG) is None


def test_a_full_run_is_more_severe_than_a_bare_one() -> None:
    bare = adverse_run(frame([100.0, 99.5, 99.0, 98.5, 99.0, 98.0]), LONG)
    full = adverse_run(frame([100.0, 99.5, 99.0, 98.5, 98.0, 97.5]), LONG)
    assert bare is not None and full is not None
    assert full.severity > bare.severity


# ----------------------------------------------------------------- spread ---


def test_a_spread_eating_the_trade_is_flagged() -> None:
    """What liquidity withdrawal looks like from inside the terminal."""
    signal = spread_blowout(spread=0.6, risk=2.0)
    assert signal is not None
    assert "30%" in signal.detail


def test_a_normal_spread_is_not() -> None:
    assert spread_blowout(spread=0.02, risk=2.0) is None


def test_no_risk_means_no_reading() -> None:
    assert spread_blowout(spread=0.6, risk=0.0) is None


# ---------------------------------------------------------------- verdict ---


def deteriorating_bars() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Structure gone and momentum turned — two independent signals."""
    fast = frame([100.0 - i * 0.4 for i in range(20)])
    structure = frame([100, 101, 102, 99, 100, 101, 102, 103, 104, 103, 102, 101, 96.0])
    return fast, structure


def test_a_young_trade_is_never_judged() -> None:
    """Entry noise — the first tick against you, the spread crossing — would
    otherwise read as a momentum turn and close good trades instantly."""
    fast, structure = deteriorating_bars()
    health = assess_position(
        sign=LONG, r_now=0.0, age_minutes=MIN_AGE_MINUTES - 0.1, fast=fast, structure=structure
    )
    assert health.action == "hold"
    assert health.verdict == "healthy"


def test_a_quiet_market_reads_as_healthy() -> None:
    health = assess_position(sign=LONG, r_now=0.3, age_minutes=30, fast=flat(), structure=flat())
    assert health.verdict == "healthy"
    assert health.action == "hold"
    assert health.signals == ()


def test_two_signals_and_a_loss_is_an_exit() -> None:
    fast, structure = deteriorating_bars()
    health = assess_position(sign=LONG, r_now=-0.3, age_minutes=30, fast=fast, structure=structure)
    assert len(health.signals) >= 2
    assert health.action == "exit"


def test_two_signals_and_a_profit_is_banked_not_cut() -> None:
    """Same reading, opposite bookkeeping. Kept apart because the journal
    reason is the only record of why, and "banked as it turned" and "cut before
    it got worse" are different lessons."""
    fast, structure = deteriorating_bars()
    health = assess_position(sign=LONG, r_now=0.8, age_minutes=30, fast=fast, structure=structure)
    assert health.action == "secure"


def test_one_family_can_never_close_a_trade() -> None:
    """The safeguard the whole design rests on. A lone family fires on an
    ordinary pullback, and closing on that bleeds the account by a thousand
    cuts — a worse failure than holding a loser slightly too long.
    """
    fast, _ = deteriorating_bars()
    health = assess_position(sign=LONG, r_now=-0.5, age_minutes=30, fast=fast, structure=flat())
    assert health.action not in ("exit", "secure")


def test_the_slope_and_the_run_are_not_two_opinions() -> None:
    """A steady drift trips both readers. If those counted as corroboration,
    one observation seen twice would be enough to close a trade — the exact
    failure the two-signal rule exists to prevent.
    """
    fast, _ = deteriorating_bars()
    health = assess_position(sign=LONG, r_now=-0.5, age_minutes=30, fast=fast, structure=flat())
    assert {s.name for s in health.signals} == {"momentum_turned", "adverse_run"}
    assert {s.family for s in health.signals} == {"drift"}
    assert health.action not in ("exit", "secure")


def test_a_lone_family_may_still_tighten() -> None:
    """Tightening costs nothing if the reading is wrong, so a single strong
    warning is allowed to buy protection it cannot be punished for."""
    _, structure = deteriorating_bars()
    health = assess_position(sign=LONG, r_now=0.4, age_minutes=30, fast=flat(), structure=structure)
    assert {s.family for s in health.signals} == {"structure"}
    assert health.action == "tighten"


def test_a_warning_below_the_tighten_level_does_nothing() -> None:
    _, structure = deteriorating_bars()
    health = assess_position(
        sign=LONG, r_now=0.05, age_minutes=30, fast=flat(), structure=structure
    )
    assert health.action == "hold"


def test_the_engine_can_only_ever_reduce_risk() -> None:
    """A property of the type, not of the caller being careful: there is no
    verdict that adds size, widens a stop or reverses, so no sequence of
    readings can produce a trade the risk layer never approved."""
    fast, structure = deteriorating_bars()
    for r in (-2.0, -0.5, 0.0, 0.3, 1.0, 5.0):
        for sign in (LONG, SHORT):
            health = assess_position(
                sign=sign,
                r_now=r,
                age_minutes=30,
                fast=fast,
                structure=structure,
                spread=0.5,
                risk=2.0,
            )
            assert health.action in ("hold", "tighten", "secure", "exit")


def test_missing_bars_are_survivable() -> None:
    """A dropped copy_rates must produce no opinion, not an exception — the
    guard runs this every second and a stale symbol is routine."""
    health = assess_position(sign=LONG, r_now=0.5, age_minutes=30, fast=None, structure=None)
    assert health.action == "hold"
    assert health.verdict == "healthy"


def test_the_summary_is_prompt_sized() -> None:
    """This goes into the supervisor's payload; it has to be small and flat."""
    fast, structure = deteriorating_bars()
    health = assess_position(sign=LONG, r_now=0.0, age_minutes=30, fast=fast, structure=structure)
    summary = health.summary()
    assert set(summary) == {"verdict", "severity", "action", "reason", "signals"}
    assert all(set(item) == {"name", "severity", "detail"} for item in summary["signals"])
    # `reason` was already computed and was the one field `summary` dropped.
    # On an ordinary reading it condenses the signals; on `unmanaged` it is the
    # whole answer, because there are no signals and the verdict only says a
    # reading was never taken.
    assert "broken" in summary["reason"]
    assert "structure_broken" in summary["reason"]


def test_severity_is_bounded() -> None:
    """Weights are tunable; the scale the thresholds are read against is not."""
    fast, structure = deteriorating_bars()
    health = assess_position(
        sign=LONG,
        r_now=0.0,
        age_minutes=30,
        fast=fast,
        structure=structure,
        spread=5.0,
        risk=2.0,
    )
    assert 0.0 <= health.severity <= 1.0


def test_signals_carry_their_own_reason() -> None:
    fast, structure = deteriorating_bars()
    health = assess_position(sign=LONG, r_now=0.0, age_minutes=30, fast=fast, structure=structure)
    assert all(isinstance(s, HealthSignal) and s.detail for s in health.signals)
    assert health.reason


def _names(health) -> set[str]:
    return {signal.name for signal in health.signals}


class TestDriftReadersWaitForTheirOwnBars:
    """A drift reader must not judge a trade on bars from before it existed.

    `MIN_AGE_MINUTES` alone did not do this. At two minutes old, ten of the
    slope's twelve bars predate the entry — the same stretch of chart the
    playbook read when it decided to enter. Entry says the move is continuing
    and health says it turned, from one set of bars, and health wins because it
    is asked again every second.

    Live evidence, 6 August: three trades cut at 2:14, 2:19 and 2:38 against a
    two-minute floor, for -0.16R, -0.71R and -0.48R. Not one reached its stop.
    """

    def test_the_slope_says_nothing_on_a_trade_minutes_old(self) -> None:
        fast, structure = deteriorating_bars()
        health = assess_position(
            sign=LONG, r_now=-0.3, age_minutes=2.5, fast=fast, structure=structure
        )
        assert _names(health) == {"structure_broken"}

    def test_and_so_the_trade_is_tightened_rather_than_closed(self) -> None:
        """The corroboration rule does the rest: one family left standing is
        not evidence, so `exit` becomes `tighten` and the trade keeps its
        chance instead of being cut at two and a half minutes."""
        fast, structure = deteriorating_bars()
        health = assess_position(
            sign=LONG, r_now=-0.3, age_minutes=2.5, fast=fast, structure=structure
        )
        assert health.action != "exit"

    def test_the_run_reader_speaks_first(self) -> None:
        """Six bars against twelve. It needs less history, so it earns its say
        sooner — one number for both would have been wrong for one of them."""
        fast, structure = deteriorating_bars()
        early = assess_position(
            sign=LONG, r_now=-0.3, age_minutes=7.0, fast=fast, structure=structure
        )
        assert "adverse_run" in _names(early)
        assert "momentum_turned" not in _names(early)

    def test_an_older_trade_is_judged_in_full(self) -> None:
        """The gate delays the readers; it must not disable them."""
        fast, structure = deteriorating_bars()
        health = assess_position(
            sign=LONG, r_now=-0.3, age_minutes=30.0, fast=fast, structure=structure
        )
        assert "momentum_turned" in _names(health)
        assert health.action == "exit"

    def test_a_slower_fast_frame_waits_proportionally_longer(self) -> None:
        """The gate counts bars, not minutes. On a five-minute fast frame the
        same twelve bars are an hour, and the reader must wait that hour."""
        fast, structure = deteriorating_bars()
        health = assess_position(
            sign=LONG,
            r_now=-0.3,
            age_minutes=30.0,
            fast=fast,
            structure=structure,
            fast_bar_minutes=5.0,
        )
        assert "momentum_turned" not in _names(health)

    def test_structure_is_not_held_back(self) -> None:
        """Deliberately exempt. The swing holding a trade up legitimately
        formed before the trade, and closing through it is a real break
        whenever it happens."""
        fast, structure = deteriorating_bars()
        health = assess_position(
            sign=LONG, r_now=-0.3, age_minutes=2.5, fast=fast, structure=structure
        )
        assert "structure_broken" in _names(health)

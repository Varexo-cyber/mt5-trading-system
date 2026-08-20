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


def test_a_young_trade_is_spared_the_drift_readers() -> None:
    """Entry noise — the first tick against you, the spread crossing — would
    otherwise read as a momentum turn and close good trades instantly.

    Scoped to the drift family, which is the only one that has this problem: a
    slope or a run computed across the entry tick. A broken structure is broken
    whenever it breaks, and the age floor used to silence that too.
    """
    fast, structure = deteriorating_bars()
    health = assess_position(
        sign=LONG, r_now=0.0, age_minutes=MIN_AGE_MINUTES - 0.1, fast=fast, structure=structure
    )

    assert "momentum_turned" not in _names(health)
    assert "adverse_run" not in _names(health)


def test_but_a_broken_structure_is_heard_immediately() -> None:
    """ "Even inside two minutes, if it is going the wrong way it has to be able
    to get out" — and the clock was the wrong thing to gate that on."""
    fast, structure = deteriorating_bars()
    health = assess_position(
        sign=LONG, r_now=-0.6, age_minutes=MIN_AGE_MINUTES - 0.1, fast=fast, structure=structure
    )

    assert "structure_broken" in _names(health)
    assert health.action == "exit"


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
    # Inside the trajectory reader's arming level on purpose. At -0.5R this was
    # never testing what it claimed: being that far offside arms a SECOND
    # family, so a two-family reading was passing a one-family assertion, and
    # it only passed because the trajectory weight was too small to carry it
    # over the line. Raising that weight is what exposed it.
    health = assess_position(sign=LONG, r_now=-0.20, age_minutes=30, fast=fast, structure=flat())
    assert {signal.family for signal in health.signals} == {"drift"}
    assert health.action not in ("exit", "secure")


def test_the_slope_and_the_run_are_not_two_opinions() -> None:
    """A steady drift trips both readers. If those counted as corroboration,
    one observation seen twice would be enough to close a trade — the exact
    failure the two-signal rule exists to prevent.
    """
    fast, _ = deteriorating_bars()
    # Inside the trajectory reader's own arming level on purpose. This test is
    # about the two DRIFT readers counting once, and a trade materially offside
    # would add a second, genuinely independent family and change the subject.
    health = assess_position(sign=LONG, r_now=-0.1, age_minutes=30, fast=fast, structure=flat())
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


class TestDriftReadersAreWeightedByWhatTheyHaveSeen:
    """A reader always speaks. What it says is worth what it has seen.

    These used to be hard gates: below twelve bars the slope said nothing at
    all, below six the run said nothing at all. The reason was sound — at two
    minutes old, ten of the slope's twelve bars predate the entry, the same
    stretch of chart the playbook read when it decided to enter, and health gets
    asked again every second while entry is asked once. Three trades on 6 August
    were cut at 2:14, 2:19 and 2:38 for exactly that, none of which reached its
    stop.

    But silence is the wrong way to say "I have only seen part of this". It
    blinded the whole drift family for the first twelve minutes of every trade
    on an M1 frame, which is most of the life of a fast one: the live UK100 long
    was over in fifteen minutes and got its first actionable reading at
    fourteen, already -0.83R deep.

    Partial evidence, weighted as partial. The 6 August failure stays out of
    reach by arithmetic rather than by silence — a two-minute-old reading is far
    too weak to corroborate into an exit whatever it sees.
    """

    def test_the_slope_speaks_early_but_quietly(self) -> None:
        fast, structure = deteriorating_bars()
        early = assess_position(
            sign=LONG, r_now=-0.3, age_minutes=3.0, fast=fast, structure=structure
        )
        mature = assess_position(
            sign=LONG, r_now=-0.3, age_minutes=30.0, fast=fast, structure=structure
        )

        def strength(health, name):  # type: ignore[no-untyped-def]
            return next((s.severity for s in health.signals if s.name == name), 0.0)

        assert strength(early, "momentum_turned") > 0.0, "it was silenced, not weighted"
        assert strength(early, "momentum_turned") < strength(mature, "momentum_turned")

    def test_and_it_still_cannot_close_a_trade_minutes_old(self) -> None:
        """The guarantee the old gate bought, kept by arithmetic instead. This
        is the 6 August failure and it must stay out of reach."""
        fast, structure = deteriorating_bars()
        health = assess_position(
            sign=LONG, r_now=-0.3, age_minutes=2.5, fast=fast, structure=structure
        )

        assert health.action != "exit"

    def test_the_run_reader_carries_more_at_the_same_age(self) -> None:
        """Six bars against twelve. It needs less history, so at any given age
        more of its window is about this trade — one number for both would have
        been wrong for one of them."""
        fast, structure = deteriorating_bars()
        health = assess_position(
            sign=LONG, r_now=-0.3, age_minutes=4.0, fast=fast, structure=structure
        )
        strength = {s.name: s.severity for s in health.signals}

        assert strength.get("adverse_run", 0.0) > strength.get("momentum_turned", 0.0)

    def test_an_older_trade_is_judged_in_full(self) -> None:
        """The weighting discounts a reader; it must not permanently damp it."""
        fast, structure = deteriorating_bars()
        health = assess_position(
            sign=LONG, r_now=-0.3, age_minutes=30.0, fast=fast, structure=structure
        )

        assert "momentum_turned" in _names(health)
        assert health.action == "exit"

    def test_a_slower_fast_frame_is_discounted_proportionally(self) -> None:
        """The share counts bars, not minutes. Thirty minutes of a five-minute
        frame is six bars, so a twelve-bar reader is half heard and a six-bar
        reader is heard in full."""
        fast, structure = deteriorating_bars()
        health = assess_position(
            sign=LONG,
            r_now=-0.3,
            age_minutes=30.0,
            fast=fast,
            structure=structure,
            fast_bar_minutes=5.0,
        )
        strength = {s.name: s.severity for s in health.signals}

        assert strength.get("adverse_run", 0.0) > strength.get("momentum_turned", 0.0)
        assert "weighted to 50%" in next(
            s.detail for s in health.signals if s.name == "momentum_turned"
        )

    def test_a_reading_thinner_than_a_tenth_is_dropped(self) -> None:
        """Arithmetic on noise, costing a signal slot and a log line."""
        from analysis.position_health import HealthSignal, _weighted

        signal = HealthSignal("momentum_turned", 0.9, "detail", "drift")

        assert _weighted(signal, bars_since_entry=1.0, bars_needed=12) is None
        assert _weighted(signal, bars_since_entry=6.0, bars_needed=12) is not None

    def test_structure_is_not_held_back(self) -> None:
        """Deliberately exempt. The swing holding a trade up legitimately
        formed before the trade, and closing through it is a real break
        whenever it happens."""
        fast, structure = deteriorating_bars()
        health = assess_position(
            sign=LONG, r_now=-0.3, age_minutes=2.5, fast=fast, structure=structure
        )
        assert "structure_broken" in _names(health)


class TestBeingDownIsNeverAReasonToClose:
    """The one reader that skips the chart, and the one it must never overrule.

    Every other reader needs bars formed SINCE the entry — twelve for the
    momentum slope, six for the adverse run, and both are the same family so
    they count once. `corroborated` needs two families, so on an M1 fast frame
    two-family agreement is arithmetically impossible before minute twelve. A
    live UK100 long was held fifteen minutes and the first actionable reading
    arrived at fourteen, by which point it was -0.83R of a -0.99R worst case.
    The layer was not wrong; it was not allowed to speak.

    But "this trade is down" must not become a reason to close it BY ITSELF —
    that is the ordinary condition of a trade that has not finished. The
    safeguard is the corroboration rule, not the weight: a lone family's exit
    is demoted to a stop tightening whatever it scores.

    The weight was 0.30 and is 0.50, changed by the owner after watching trades
    run from a manageable loss to the full stop with every reader watching and
    none able to speak. What that buys is a second family reaching an exit
    where it previously only tightened; what it does not touch is a trade in
    profit, because this reader does not exist above -0.35R.
    """

    def test_it_cannot_close_a_trade_alone(self) -> None:
        """Stated as behaviour, not as a number. The old form asserted the
        weight was under `DETERIORATING_AT`, which conflated "cannot close"
        with "cannot say anything" — and it is the corroboration rule, not the
        weight, that stops a lone family closing a trade.
        """
        from analysis.position_health import BROKEN_AT, HealthWeights

        assert HealthWeights().trajectory < BROKEN_AT

        health = assess_position(
            sign=LONG, r_now=-1.50, age_minutes=30, fast=flat(), structure=flat()
        )

        assert {signal.family for signal in health.signals} == {"trajectory"}
        assert health.action not in ("exit", "secure")

    def test_a_second_family_now_reaches_an_exit_and_not_only_a_tighten(self) -> None:
        """The change the owner asked for, written as arithmetic.

        `structure` + `trajectory` still closes a trade at any age: the swing
        that justified the position has broken AND the market has already taken
        real money. Neither is a clock reading and neither waits for a bar to
        form, so waiting adds nothing except the loss.

        `drift` and `liquidity` now join it. Before, being 0.60R down while
        momentum ran against the trade scored 0.65 — `deteriorating`, which
        tightens a stop and lets the trade continue to the full stop. That was
        the case being lost, repeatedly.

        `drift` + `liquidity` remains short of an exit, and should: neither of
        them says the money is gone.
        """
        from analysis.position_health import BROKEN_AT, HealthWeights

        weights = HealthWeights()

        assert weights.trajectory + weights.structure >= BROKEN_AT
        assert weights.trajectory + weights.drift >= BROKEN_AT
        assert weights.trajectory + weights.liquidity >= BROKEN_AT
        assert weights.drift + weights.liquidity < BROKEN_AT

    def test_a_trade_inside_its_own_noise_says_nothing(self) -> None:
        from analysis.position_health import adverse_excursion

        assert adverse_excursion(-0.20) is None
        assert adverse_excursion(0.50) is None

    def test_a_materially_offside_trade_speaks(self) -> None:
        from analysis.position_health import adverse_excursion

        signal = adverse_excursion(-0.60)

        assert signal is not None
        assert signal.family == "trajectory"

    def test_a_modest_offside_still_only_tightens(self) -> None:
        """The lower rung survives. A trade just past the noise level with the
        tape drifting is not a closed trade — it gets a smaller stop, which
        costs nothing when the reading is wrong."""
        fast, _ = deteriorating_bars()
        health = assess_position(
            sign=LONG, r_now=-0.40, age_minutes=30, fast=fast, structure=flat()
        )

        assert {signal.family for signal in health.signals} == {"drift", "trajectory"}
        assert health.action == "tighten"

    def test_materially_offside_with_the_tape_against_it_now_closes(self) -> None:
        """The case the owner kept watching run to the full stop. Down 0.60R
        with momentum turned scored 0.58 and tightened a stop; it scores 0.85
        and gets out. On this account that is -1.93 EUR instead of -3.22."""
        fast, _ = deteriorating_bars()
        health = assess_position(
            sign=LONG, r_now=-0.60, age_minutes=30, fast=fast, structure=flat()
        )

        assert {signal.family for signal in health.signals} == {"drift", "trajectory"}
        assert health.verdict == "broken"
        assert health.action == "exit"

    def test_it_never_reaches_for_a_trade_that_is_winning(self) -> None:
        """The reader does not exist above -0.35R, so no amount of weight on it
        can shorten a winner. Worth pinning next to the change that raised it."""
        fast, _ = deteriorating_bars()
        health = assess_position(sign=LONG, r_now=0.30, age_minutes=30, fast=fast, structure=flat())

        assert {signal.family for signal in health.signals} == {"drift"}
        assert health.action not in ("exit", "secure")


def _tighten_fixture():  # type: ignore[no-untyped-def]
    """A live long with room between price and its stop, and a manager to run.

    Built from the position-guard fixtures so the broker and journal stubs are
    the ones the rest of the suite already trusts.
    """
    from core.types import Direction, Position, Tick
    from tests.test_position_guard import NOW, BrokerStub, JournalStub, manager_for

    position = Position(
        ticket=4242,
        symbol="EURUSD",
        direction=Direction.LONG,
        volume=0.05,
        price_open=1.1000,
        sl=1.0980,
        tp=1.1060,
        profit=0.0,
        swap=0.0,
        opened_at=NOW,
    )
    tick = Tick(symbol="EURUSD", time=NOW, bid=1.1012, ask=1.1013)
    return manager_for(BrokerStub(), JournalStub()), position, tick


class TestSecuringFastIsTheOwnersChoiceAndTheRatchetIsWhatMakesItWork:
    """The tightener is left alone; the rule that WALKS THE STOP UP is fixed.

    Securing early is the owner's stated priority and his record supports it:
    85% win rate, +EUR 41 over 48 hours, and on a EUR 180 account a string of
    full-R losses is existential. What went wrong was never that the stop moved
    early — it was that nothing moved it again afterwards. One position was
    tightened at +0.12R to a stop at +0.06R, ran to +0.59R, and came back to
    close at +0.06R.

    `_profit_lock` is the rule that cannot do that: it measures from the PEAK,
    so it ratchets and never retreats, it runs every management pass, and it
    lives at the broker where it survives a VPS restart. It was simply armed
    above where these trades live — at 0.7R, on a record where four of
    thirty-nine ever saw 0.6R. It rescued exactly one trade.
    """

    @staticmethod
    def _live():  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH,
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml",
            env_overrides=False,
        ).trade_management

    def test_the_ratchet_arms_where_the_trades_actually_peak(self) -> None:
        """39 closed trades, replayed at each threshold: 0.70R rescued 1 for
        +0.18R, 0.40R rescued 5 for +0.97R, 0.20R rescued 19 for +1.74R."""
        config = self._live()

        assert config.profit_lock_from_r <= 0.25
        assert config.profit_lock_fraction >= 0.6

    def test_it_arms_below_break_even_because_that_is_the_whole_point(self) -> None:
        """Break-even waits for 0.6R and almost nothing reaches it. A rule that
        only protects trades which never happen protects nothing."""
        config = self._live()

        assert config.profit_lock_from_r < config.break_even_at_r

    def test_the_health_tightener_still_secures_early(self) -> None:
        """Not gated on R. Securing fast is the point, and the ratchet above is
        what stops a fast secure from becoming a frozen one."""
        from analysis.position_health import PositionHealth

        manager, position, tick = _tighten_fixture()
        health = PositionHealth(
            verdict="deteriorating",
            severity=0.45,
            action="tighten",
            reason="momentum_turned",
        )
        risk = abs(position.price_open - position.sl)

        assert manager._act_on_health(position, health, 0.12, risk, tick) is not None


class TestTheRatchetCannotManufactureALossOnATightSpread:
    """EURAUD LONG, 20 August. Seven tightenings inside one minute.

        worst it reached   -0.60R      the market never went lower
        its real stop      -1.00R      never came near it
        what it returned   -0.66R      the tightened stop

    Severity climbed 0.46 -> 0.68 and never reached the 0.75 an exit needs, so
    nothing ever decided to leave. The stop was simply walked up into the price
    until it was taken, at a level worse than the position's own deepest point.

    The spread floor added for FRA40 did nothing here and could not: FRA40 had
    32 pips of spread, so two of them was 64 pips of protection; EURAUD had
    0.10 pips, so two was 0.2. A floor has to be measured in something that
    means the same on both, and the trade's own risk is the only such unit.

    Replayed on the real numbers — entry 1.64035, stop 1.63918, 11.7 pips —
    the spread floor alone walks the stop to 1.63967 and the trade's deepest
    point of 1.63965 takes it out. With 0.15R the walk stops at 1.63954 after
    two moves and the position survives.
    """

    ENTRY, STOP, SPREAD = 1.64035, 1.63918, 0.00001

    def _walk(self, *, spreads: float, r_floor: float) -> float:
        """The tightening rule's own arithmetic, run to a standstill."""
        risk = self.ENTRY - self.STOP
        floor = max(self.SPREAD * spreads, risk * r_floor)
        stop = self.STOP
        for r_now in (-0.53, -0.54, -0.55, -0.56, -0.57, -0.57, -0.58):
            price = self.ENTRY + risk * r_now
            locked = min(price + (stop - price) * 0.5, price - floor)
            if locked <= stop:  # the `improves` test declines it
                break
            stop = locked
        return stop

    def test_the_spread_floor_alone_walks_the_stop_into_the_price(self) -> None:
        deepest = self.ENTRY + (self.ENTRY - self.STOP) * -0.60

        assert self._walk(spreads=2.0, r_floor=0.0) > deepest  # taken out

    def test_a_floor_in_r_leaves_the_position_alive(self) -> None:
        deepest = self.ENTRY + (self.ENTRY - self.STOP) * -0.60

        assert self._walk(spreads=2.0, r_floor=0.15) < deepest  # survives

    def test_the_larger_of_the_two_floors_applies(self) -> None:
        """Neither floor replaces the other; each covers what the other cannot.

        The R floor is the bigger number on both cases measured so far —
        EURAUD at 11.7 pips and FRA40 at 874 — because a stop is usually many
        spreads wide. Where the spread floor takes over is the case this
        account trades a lot of: a tight stop on a market whose round trip is a
        large share of it. At a 7-pip stop and a 0.8-pip spread, two spreads is
        1.6 pips against the R floor's 1.05 — and a stop inside the spread is
        not a stop whatever share of R it happens to be.
        """
        tight_stop, its_spread = 0.00070, 0.00008

        assert its_spread * 2.0 > tight_stop * 0.15
        assert 0.00117 * 0.15 > 0.00001 * 2.0  # EURAUD: the R floor wins
        assert 0.874 * 0.15 > 0.032 * 2.0  # FRA40: the R floor wins there too

    def test_the_live_overlay_carries_both(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        config = load_settings(
            DEFAULT_CONFIG_PATH,
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml",
            env_overrides=False,
        ).trade_management

        assert config.min_stop_room_spreads > 0
        assert config.min_stop_room_r > 0

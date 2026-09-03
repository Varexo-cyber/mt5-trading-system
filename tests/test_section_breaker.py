"""A new section that is going badly stops itself.

Sections two, five and six went live on the owner's authorisation with ZERO
measured trades between them. Every number behind them is reasoning, and this
is what survives that reasoning being wrong.

The test that matters most is the last class: that a tripped section actually
refuses a trade. A breaker that computes the right verdict and does not act on
it is worse than none, because the report says protected while the account is
not.
"""

from __future__ import annotations

import sqlite3

import pytest

from config.schema import SectionBreakerConfig
from risk.section_breaker import assess, tripped_modules


def journal(outcomes: list[float], module: str = "candle_momentum") -> sqlite3.Connection:
    """A journal whose newest trade is the LAST element of `outcomes`."""
    db = sqlite3.connect(":memory:")
    db.executescript(
        "CREATE TABLE trades (id INTEGER PRIMARY KEY, cycle_pk INTEGER, direction TEXT,"
        " pnl_r REAL, closed_at TEXT);"
        "CREATE TABLE module_scores (id INTEGER PRIMARY KEY, cycle_pk INTEGER,"
        " module TEXT, score REAL, weight REAL);"
    )
    for i, pnl in enumerate(outcomes, start=1):
        db.execute(
            "INSERT INTO trades (id, cycle_pk, direction, pnl_r, closed_at)" " VALUES (?,?,?,?,?)",
            (i, i, "LONG", pnl, f"2026-08-24T{i // 60:02d}:{i % 60:02d}:00"),
        )
        db.execute(
            "INSERT INTO module_scores (cycle_pk, module, score, weight) VALUES (?,?,?,?)",
            (i, module, 60.0, 0.6),
        )
    db.commit()
    return db


STRICT = SectionBreakerConfig(
    window=10, minimum_trades=10, maximum_loss_share=0.75, losing_streak=6
)


class TestItWillNotJudgeTooSoon:
    """Switching a section off on noise is the same mistake as switching one
    on for it."""

    def test_a_handful_of_losses_is_not_a_verdict(self) -> None:
        verdict = assess(journal([-1.0] * 4), "candle_momentum", STRICT)

        assert not verdict.tripped
        assert "of the 10 trades needed" in verdict.reason

    def test_an_empty_journal_concludes_nothing(self) -> None:
        verdict = assess(journal([]), "candle_momentum", STRICT)

        assert not verdict.tripped

    def test_a_journal_without_module_scores_concludes_nothing(self) -> None:
        """Silence is not a clean record and it is not a bad one. An older
        database must not be read as either."""
        db = sqlite3.connect(":memory:")
        db.executescript("CREATE TABLE trades (id INTEGER PRIMARY KEY, pnl_r REAL);")

        verdict = assess(db, "candle_momentum", STRICT)

        assert not verdict.tripped
        assert "no journal evidence" in verdict.reason


class TestTheTwoRules:
    def test_eight_losses_in_ten_stops_it(self) -> None:
        """The owner's second case, in as many words: "1 winnende en 8
        verloren"."""
        # Eight of ten, with the streak deliberately kept under six so this
        # isolates the SHARE rule. The streak firing first would be correct
        # behaviour and would test the other rule twice.
        outcomes = [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0]
        verdict = assess(journal(outcomes), "candle_momentum", STRICT)

        assert verdict.tripped
        assert verdict.streak < STRICT.losing_streak
        assert "losses in the last 10 trades" in verdict.reason

    def test_a_losing_streak_stops_it_before_the_window_fills(self) -> None:
        """THE URGENT RULE. A share over a window catches a section that is
        quietly wrong; a run catches one that is wrong RIGHT NOW. On a section
        taking several trades a day that is the difference between stopping
        today and stopping next week."""
        outcomes = [1.0] * 4 + [-1.0] * 6  # 6 losses, but only 60% of the window
        verdict = assess(journal(outcomes), "candle_momentum", STRICT)

        assert verdict.tripped
        assert verdict.streak == 6
        assert "in a row" in verdict.reason

    def test_the_streak_reads_from_the_most_recent_trade(self) -> None:
        """Six old losses followed by a win is a section that recovered, not
        one on a streak. Reading the streak from the wrong end would stop a
        section for something it already fixed."""
        outcomes = [-1.0] * 6 + [1.0]
        verdict = assess(journal(outcomes), "candle_momentum", STRICT)

        assert verdict.streak == 0

    def test_a_healthy_section_keeps_running(self) -> None:
        outcomes = [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0]
        verdict = assess(journal(outcomes), "candle_momentum", STRICT)

        assert not verdict.tripped
        assert "7 of 10 won" in verdict.reason

    def test_break_even_counts_as_a_loss(self) -> None:
        """A trade that returned exactly nothing paid a spread to do it."""
        verdict = assess(journal([0.0] * 10), "candle_momentum", STRICT)

        assert verdict.tripped

    def test_a_disabled_breaker_never_trips(self) -> None:
        config = SectionBreakerConfig(enabled=False, window=10, minimum_trades=10, losing_streak=2)

        assert not assess(journal([-1.0] * 10), "candle_momentum", config).tripped

    def test_negative_expectancy_stops_a_high_win_rate_section(self) -> None:
        """Tiny protected wins must not hide fewer full-stop losses."""
        config = SectionBreakerConfig(
            window=10,
            minimum_trades=10,
            maximum_loss_share=0.75,
            losing_streak=9,
            minimum_average_r=0.0,
        )
        outcomes = [0.05] * 7 + [-0.25, -0.25, -0.25]

        verdict = assess(journal(outcomes), "candle_momentum", config)

        assert verdict.tripped
        assert verdict.losses == 3
        assert "average" in verdict.reason


class TestAttribution:
    """A trade belongs to a module when that module carried weight and scored
    in the direction actually taken — the same reading the scorecard's detector
    table uses, so a section cannot look healthy in one place and stopped in
    another."""

    def test_another_modules_losses_do_not_count(self) -> None:
        db = journal([-1.0] * 10, module="impulse_break")

        assert not assess(db, "candle_momentum", STRICT).tripped

    def test_a_module_that_pointed_the_other_way_is_not_blamed(self) -> None:
        db = journal([-1.0] * 10)
        db.execute("UPDATE module_scores SET score = -60.0")  # short call, long trade
        db.commit()

        assert not assess(db, "candle_momentum", STRICT).tripped

    def test_a_module_carrying_no_weight_is_not_blamed(self) -> None:
        db = journal([-1.0] * 10)
        db.execute("UPDATE module_scores SET weight = 0.0")
        db.commit()

        assert not assess(db, "candle_momentum", STRICT).tripped


class TestTheShippedSettings:
    def _breakers(self):  # type: ignore[no-untyped-def]
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        ).risk.section_breakers

    def test_every_live_section_that_has_no_record_carries_a_breaker(self) -> None:
        """The nine original readers have months of record and are judged as a
        book. These three have nothing, so each carries its own stop — and a
        new section reaching live without one is the failure this asserts
        against."""
        breakers = self._breakers()

        for section in ("drift_burst", "basket_divergence", "candle_momentum"):
            assert section in breakers, section

    def test_the_scalp_is_the_strict_one(self) -> None:
        """It takes several trades a day, so ten losses in a row is half a day
        of paying for the same fault. Six, and eight of the last ten."""
        scalp = self._breakers()["candle_momentum"]

        assert scalp.losing_streak == 6
        assert scalp.window == 10
        assert scalp.maximum_loss_share == 0.75

    def test_the_slower_sections_are_given_room(self) -> None:
        """Section two fades panic, which is by nature a bet that is often
        wrong before it is right. Judging it on ten trades would switch off a
        working strategy for having a bad morning."""
        for section in ("drift_burst", "basket_divergence"):
            config = self._breakers()[section]
            assert config.minimum_trades >= 25, section
            assert config.losing_streak >= 10, section

    def test_a_breaker_that_can_never_judge_is_refused(self) -> None:
        """It would sit permanently disarmed while reading as armed, which is
        the worst state of the three."""
        with pytest.raises(ValueError, match="cannot exceed window"):
            SectionBreakerConfig(window=10, minimum_trades=25)


class TestATrippedSectionActuallyRefusesTheTrade:
    """The test that matters most. A breaker that computes the right verdict
    and does not act on it is worse than none, because the report reads
    protected while the account is not."""

    def _runner(self, outcomes: list[float]):  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from runner.service import JarvisRunner

        runner = object.__new__(JarvisRunner)
        runner.settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        runner.journal = SimpleNamespace(conn=journal(outcomes))  # type: ignore[assignment]
        runner._breaker_cache = {}
        runner._breaker_cache_cycle = -1
        runner._cycle_serial = 1
        return runner

    def _idea(self, module: str = "candle_momentum"):  # type: ignore[no-untyped-def]
        from analysis.confluence import TradeIdea
        from core.types import Direction, Signal

        return TradeIdea(
            symbol="XAUUSD",
            approved=True,
            direction=Direction.LONG,
            score=55.0,
            confidence=0.6,
            entry=2400.0,
            stop_loss=2398.0,
            take_profit=2404.0,
            reason="test",
            signals=(Signal(module, 60.0, 0.6),),
        )

    def test_a_stopped_section_blocks_its_own_setup(self) -> None:
        runner = self._runner([-1.0] * 10)

        blocked = runner._tripped_section(self._idea())

        assert blocked is not None
        assert "stopped itself" in blocked
        assert "Re-arm" in blocked

    def test_a_healthy_section_is_left_alone(self) -> None:
        runner = self._runner([1.0] * 10)

        assert runner._tripped_section(self._idea()) is None

    def test_one_stopped_section_does_not_stop_the_others(self) -> None:
        """The nine original readers must keep trading while a new section is
        switched off. A breaker that halted the account would be a far larger
        intervention than the one authorised."""
        runner = self._runner([-1.0] * 10)

        assert runner._tripped_section(self._idea("impulse_break")) is None

    def test_the_verdict_is_recomputed_each_cycle(self) -> None:
        """Cached per cycle, not for the life of the process. Without the
        serial bump the first verdict of the run would be reused forever and a
        section that tripped mid-session would keep trading until a restart."""
        runner = self._runner([1.0] * 10)
        assert runner._tripped_section(self._idea()) is None

        runner.journal.conn = journal([-1.0] * 10)
        assert runner._tripped_section(self._idea()) is None  # still cached
        runner._cycle_serial += 1

        assert runner._tripped_section(self._idea()) is not None


class TestTrippedModules:
    def test_it_names_only_the_sections_that_stopped(self) -> None:
        db = journal([-1.0] * 10)
        breakers = {
            "candle_momentum": STRICT,
            "drift_burst": SectionBreakerConfig(window=30, minimum_trades=25),
        }

        stopped = tripped_modules(db, breakers)

        assert set(stopped) == {"candle_momentum"}

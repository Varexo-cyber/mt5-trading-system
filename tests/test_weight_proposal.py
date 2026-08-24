"""Learning from the losses, without learning from the noise.

THE OWNER'S QUESTION: "kan je niet leren van die loss op DIE range en
transition en die fouten op losses voorkomen voor next time en de winsten
aannemen".

The right question, and the honest answer is that the promotion half of it
already existed — `ConfigControl` shadow-tests a candidate config, measures
the paired lift with a confidence bound, and refuses anything that is not a
bounded weight change. Nothing ever wrote it a candidate. Weights moved when
a human read a bucket after a bad afternoon and decided something.

That is exactly how a 44-trade bucket got a regime switched off, and the next
day that regime was the best performer on the card. So the tests that matter
most here are the ones about what this REFUSES to conclude. Anything can
notice a losing module; the value is entirely in not acting on nine trades.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from learning.weight_proposal import MAX_STEP, measure, proposals, propose


def journal(outcomes: dict[str, list[float]]) -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.executescript(
        "CREATE TABLE trades (id INTEGER PRIMARY KEY, cycle_pk INTEGER, direction TEXT,"
        " pnl_r REAL, closed_at TEXT);"
        "CREATE TABLE module_scores (id INTEGER PRIMARY KEY, cycle_pk INTEGER,"
        " module TEXT, score REAL, weight REAL);"
    )
    # STRICTLY INCREASING TIMESTAMPS. `measure` orders by `closed_at` to read
    # the record newest-first, so a fixture whose dates wrap around scrambles
    # the two halves and the holdout tests silently measure nothing.
    start = datetime(2026, 1, 1, tzinfo=UTC)
    cycle = 0
    for module, values in outcomes.items():
        for pnl in values:
            cycle += 1
            db.execute(
                "INSERT INTO trades (id, cycle_pk, direction, pnl_r, closed_at)"
                " VALUES (?,?,?,?,?)",
                (
                    cycle,
                    cycle,
                    "LONG",
                    pnl,
                    (start + timedelta(minutes=cycle)).isoformat(),
                ),
            )
            db.execute(
                "INSERT INTO module_scores (cycle_pk, module, score, weight) VALUES (?,?,?,?)",
                (cycle, module, 60.0, 0.7),
            )
    db.commit()
    return db


def steady(value: float, n: int, jitter: float = 0.02) -> list[float]:
    """A tight record: the mean is well determined, so the interval is narrow."""
    return [value + (jitter if i % 2 else -jitter) for i in range(n)]


def noisy(mean: float, n: int, spread: float = 1.2) -> list[float]:
    """The same mean carried by wild outcomes — the interval swallows zero."""
    return [mean + (spread if i % 2 else -spread) for i in range(n)]


class TestWhatItRefusesToConclude:
    """More important than what it proposes."""

    def test_a_thin_record_moves_nothing(self) -> None:
        """Nine losing trades is an anecdote. This is the guard that was
        missing when a regime was switched off on one bad bucket."""
        db = journal({"drift_continuation": steady(-0.4, 9)})

        proposal = propose(measure(db, "drift_continuation"), 0.7)

        assert not proposal.changed
        assert "of the 30 trades needed" in proposal.why

    def test_a_losing_mean_inside_the_noise_moves_nothing(self) -> None:
        """It lost money. It did not lose money DETECTABLY, and those are
        different findings."""
        db = journal({"drift_continuation": noisy(-0.05, 60)})
        evidence = measure(db, "drift_continuation")

        assert evidence.mean_r < 0
        assert not evidence.decided
        assert not propose(evidence, 0.7).changed

    def test_a_module_already_switched_off_stays_off(self) -> None:
        """`ConfigControl` refuses to reinstate a zero-weight module and this
        must not try. Turning one back on has no evidence behind it by
        definition — there are no trades to measure."""
        db = journal({"mean_reversion": steady(0.4, 80)})

        proposal = propose(measure(db, "mean_reversion"), 0.0)

        assert not proposal.changed
        assert "needs a decision, not a measurement" in proposal.why

    def test_an_empty_journal_concludes_nothing(self) -> None:
        assert not propose(measure(journal({}), "impulse_break"), 0.7).changed

    def test_a_journal_without_module_scores_concludes_nothing(self) -> None:
        db = sqlite3.connect(":memory:")
        db.executescript("CREATE TABLE trades (id INTEGER PRIMARY KEY, pnl_r REAL);")

        evidence = measure(db, "impulse_break")

        assert evidence.trades == 0
        assert not propose(evidence, 0.7).changed


class TestWhatItDoesConclude:
    def test_a_detector_that_loses_clearly_is_walked_down(self) -> None:
        db = journal({"drift_continuation": steady(-0.35, 60)})

        proposal = propose(measure(db, "drift_continuation"), 0.7)

        assert proposal.changed
        assert proposal.proposed < 0.7
        assert "cost money" in proposal.why

    def test_a_detector_that_earns_clearly_is_walked_up(self) -> None:
        db = journal({"ema_pullback_resume": steady(0.30, 60)})

        proposal = propose(measure(db, "ema_pullback_resume"), 0.8)

        assert proposal.proposed > 0.8
        assert "earned more" in proposal.why

    def test_no_step_exceeds_the_cap(self) -> None:
        """`ConfigControl` rejects a shift beyond 15%. A proposal that gets
        refused for being too eager is a proposal that teaches nothing, so
        this stays well inside it."""
        db = journal({"drift_continuation": steady(-5.0, 60)})

        proposal = propose(measure(db, "drift_continuation"), 0.7)

        assert abs(proposal.proposed / 0.7 - 1.0) <= MAX_STEP + 1e-9

    def test_a_weight_never_passes_one(self) -> None:
        db = journal({"market_structure": steady(2.0, 60)})

        assert propose(measure(db, "market_structure"), 1.0).proposed <= 1.0

    def test_a_wide_record_moves_less_than_a_tight_one_at_the_same_mean(self) -> None:
        """The step scales with the part of the evidence that is beyond doubt,
        not with the headline. Two detectors reporting the same average should
        not be treated the same when one of them is far less certain."""
        tight = propose(measure(journal({"a": steady(-0.30, 200)}), "a"), 0.7)
        loose = propose(measure(journal({"a": noisy(-0.30, 200, spread=0.9)}), "a"), 0.7)

        assert tight.proposed < loose.proposed <= 0.7

    def test_a_weight_worn_down_to_nothing_is_proposed_as_zero(self) -> None:
        """Below the floor it is off in all but name, and a weight of 0.03
        pretending to be a vote is worse than an honest zero."""
        db = journal({"seasonality": steady(-0.40, 60)})

        assert propose(measure(db, "seasonality"), 0.05).proposed == 0.0


class TestTheReport:
    def test_every_weighted_module_is_listed_changed_or_not(self) -> None:
        """A report showing only the movers reads as though the rest were
        examined and found fine, when most were simply too thin to judge."""
        db = journal({"drift_continuation": steady(-0.35, 60), "impulse_break": steady(0.2, 5)})
        weights = {"drift_continuation": 0.7, "impulse_break": 0.6, "seasonality": 0.25}

        report = {item.module: item for item in proposals(db, weights)}

        assert set(report) == set(weights)
        assert report["drift_continuation"].changed
        assert not report["impulse_break"].changed
        assert not report["seasonality"].changed

    def test_the_reason_says_which_bar_was_missed(self) -> None:
        db = journal({"impulse_break": steady(0.2, 5)})

        assert "5 of the 30 trades" in propose(measure(db, "impulse_break"), 0.6).why


class TestAttributionMatchesEveryOtherReport:
    """A detector cannot earn its keep in one report and lose it in another."""

    def test_a_module_that_pointed_the_other_way_is_not_credited(self) -> None:
        db = journal({"impulse_break": steady(-0.4, 60)})
        db.execute("UPDATE module_scores SET score = -60.0")  # short call, long trade
        db.commit()

        assert measure(db, "impulse_break").trades == 0

    def test_a_module_carrying_no_weight_is_not_credited(self) -> None:
        db = journal({"impulse_break": steady(-0.4, 60)})
        db.execute("UPDATE module_scores SET weight = 0.0")
        db.commit()

        assert measure(db, "impulse_break").trades == 0

    def test_an_open_trade_is_not_counted(self) -> None:
        db = journal({"impulse_break": steady(-0.4, 60)})
        db.execute("UPDATE trades SET closed_at = NULL WHERE id % 2 = 0")
        db.commit()

        assert measure(db, "impulse_break").trades == 30


class TestTheHoldout:
    """ "pas dan aan tot het winstgevend wordt" — the honest version.

    Tuned against the whole record, "until it is profitable" is guaranteed to
    succeed and guaranteed to mean nothing. Split the record and only act when
    both halves agree, and the tuning has to survive data it was not fitted
    to. It is not a proper walk-forward; it is the cheapest test that catches
    a proposal working only on the stretch that produced it.
    """

    def test_a_module_that_turned_mid_record_moves_nothing(self) -> None:
        """Forty good trades then forty bad ones average out to something
        significant and mean nothing. This is the case the interval alone
        cannot see."""
        db = journal({"a": steady(-0.40, 40) + steady(0.20, 40)})
        evidence = measure(db, "a")

        assert evidence.significant
        assert not evidence.holds_out_of_sample
        assert not propose(evidence, 0.7).changed

    def test_the_reason_names_both_halves(self) -> None:
        db = journal({"a": steady(-0.40, 40) + steady(0.20, 40)})

        why = propose(measure(db, "a"), 0.7).why

        assert "the two halves of the record disagree" in why

    def test_a_module_that_lost_in_both_halves_is_still_acted_on(self) -> None:
        """The holdout must not become a way of never concluding anything. A
        detector that lost in both stretches has a stable, measured problem."""
        db = journal({"a": steady(-0.30, 40) + steady(-0.40, 40)})
        evidence = measure(db, "a")

        assert evidence.holds_out_of_sample
        assert propose(evidence, 0.7).proposed < 0.7

    def test_it_reads_the_halves_the_right_way_round(self) -> None:
        """`measure` returns newest-first. Reading the split backwards would
        report a recovering detector as a decaying one and vice versa, which
        is worse than not splitting at all."""
        db = journal({"a": steady(-0.40, 40) + steady(0.20, 40)})  # older, then newer
        evidence = measure(db, "a")

        assert evidence.early_mean_r < 0
        assert evidence.late_mean_r > 0

    def test_a_record_too_short_to_split_is_caught_by_the_sample_bar_first(self) -> None:
        db = journal({"a": steady(-0.4, 3)})

        assert "of the 30 trades needed" in propose(measure(db, "a"), 0.7).why

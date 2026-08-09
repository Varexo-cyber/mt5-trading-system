"""The analyst: a reading of the whole account, with no authority over it.

Everything else that calls an AI here answers a closed question about one
trade. That is the right shape for a gate — auditable, cheap, unable to
wander. It is the wrong shape for "look at all of it and tell me what is
wrong", which is the question the operator has been asking a person every few
hours by pasting screenshots.

The tests that matter most are the ones asserting it cannot do anything. A
gate has to be predictable and an analyst has to be free to say something
nobody anticipated; keeping those in separate components is what makes the
second one safe to point at a live account.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from analyst.evidence import Evidence, TradeFact, dominant_refusal, gather, read_open_positions
from analyst.review import Assessment, Finding
from config.loader import load_settings

NOW = datetime(2026, 8, 6, 14, 0, tzinfo=UTC)


def fact(**overrides) -> TradeFact:  # type: ignore[no-untyped-def]
    base = {
        "symbol": "EURGBP.i",
        "direction": "LONG",
        "stop_pips": 11.9,
        "peak_r": 1.30,
        "pnl_r": 1.12,
        "pnl_money": 1.62,
        "exit_reason": "PEAK_STALL",
        "held_minutes": 77.0,
    }
    return TradeFact(**{**base, **overrides})


class TestKeptShareOfPeak:
    """The number that separates a wrong strategy from one that hands it back."""

    def test_a_trade_that_held_its_gain(self) -> None:
        assert fact().kept == pytest.approx(0.86, abs=0.01)

    def test_the_live_usdchf_shape(self) -> None:
        """Peaked at 0.92R, returned 0.13R: 14% survived."""
        assert fact(peak_r=0.92, pnl_r=0.13).kept == pytest.approx(0.14, abs=0.01)

    def test_a_trade_that_never_went_green_has_no_share(self) -> None:
        """Zero peak is not zero percent kept — it is a different question,
        and dividing by it would answer neither."""
        assert fact(peak_r=0.0, pnl_r=-1.26).kept is None

    def test_a_missing_peak_is_unknown_rather_than_zero(self) -> None:
        assert fact(peak_r=None).kept is None


class TestSummary:
    def test_it_counts_who_chose_the_exits(self) -> None:
        evidence = Evidence(
            window_hours=24.0,
            generated_at=NOW,
            trades=[
                fact(exit_reason="PEAK_STALL"),
                fact(exit_reason="BROKER_SL", peak_r=0.0, pnl_r=-1.26, pnl_money=-1.93),
                fact(exit_reason="SL", peak_r=0.92, pnl_r=0.13, pnl_money=0.22),
            ],
        )
        summary = evidence.summary()
        assert summary["exits_chosen_by_the_system"] == 1
        assert summary["exits_left_to_the_broker"] == 2

    def test_it_separates_a_stop_out_that_cost_more_than_its_risk(self) -> None:
        """Below -1.00R is cost or execution, not strategy. Different fix."""
        evidence = Evidence(
            window_hours=24.0,
            generated_at=NOW,
            trades=[fact(pnl_r=-1.26), fact(pnl_r=-0.95), fact(pnl_r=1.12)],
        )
        assert evidence.summary()["stop_outs_worse_than_minus_one_r"] == 1

    def test_it_totals_the_money_and_the_r(self) -> None:
        evidence = Evidence(
            window_hours=24.0,
            generated_at=NOW,
            trades=[fact(pnl_r=1.12, pnl_money=1.62), fact(pnl_r=-1.26, pnl_money=-1.93)],
        )
        summary = evidence.summary()
        assert summary["net_r"] == pytest.approx(-0.14)
        assert summary["net_money"] == pytest.approx(-0.31)
        assert (summary["wins"], summary["losses"]) == (1, 1)

    def test_an_empty_window_summarises_cleanly(self) -> None:
        summary = Evidence(window_hours=24.0, generated_at=NOW).summary()
        assert summary["closed_trades"] == 0
        assert summary["median_kept_share_of_peak"] is None


class TestGather:
    def test_a_missing_journal_still_produces_evidence(self, tmp_path: Path) -> None:
        """A half-complete picture is worth reading. An analyst that cannot run
        because one table is empty is an analyst nobody keeps."""
        settings = load_settings(env_overrides=False)
        evidence = gather(tmp_path / "nope.db", settings, now=NOW)

        assert evidence.trades == []
        assert evidence.rules["risk_per_trade_pct"] > 0

    def test_the_rules_carry_the_settings_that_decide_behaviour(self, tmp_path: Path) -> None:
        """Hand-picked rather than the whole tree: a full dump buries the six
        numbers that matter under thousands of untouched defaults."""
        settings = load_settings(env_overrides=False)
        rules = gather(tmp_path / "nope.db", settings, now=NOW).rules

        for key in (
            "max_cost_share_of_risk",
            "min_guard_seconds",
            "profit_lock_from_r",
            "require_entry_confirmation",
            "max_positions_per_currency",
        ):
            assert key in rules, key

    def test_it_reads_trades_and_refusals(self, tmp_path: Path) -> None:
        path = tmp_path / "trading.db"
        db = sqlite3.connect(path)
        db.executescript("""
            CREATE TABLE trades (symbol TEXT, direction TEXT, sl_distance_pips REAL,
                mfe_r REAL, pnl_r REAL, pnl_money REAL, exit_reason TEXT,
                opened_at TEXT, closed_at TEXT);
            CREATE TABLE analysis_cycles (ts TEXT, reason TEXT);
            """)
        db.execute(
            "INSERT INTO trades VALUES ('GBPJPY.i','SHORT',10.6,0.0,-1.16,-1.93,'BROKER_SL',?,?)",
            ((NOW - timedelta(hours=2)).isoformat(), (NOW - timedelta(hours=1)).isoformat()),
        )
        for reason in ("NO_SIGNAL", "NO_SIGNAL", "AWAITING_CONFIRMATION"):
            db.execute(
                "INSERT INTO analysis_cycles VALUES (?, ?)",
                ((NOW - timedelta(minutes=30)).isoformat(), reason),
            )
        db.commit()
        db.close()

        settings = load_settings(env_overrides=False)
        evidence = gather(path, settings, now=NOW)

        assert len(evidence.trades) == 1
        assert evidence.trades[0].symbol == "GBPJPY.i"
        assert evidence.refusals == {"NO_SIGNAL": 2, "AWAITING_CONFIRMATION": 1}

    def test_trades_outside_the_window_are_left_out(self, tmp_path: Path) -> None:
        path = tmp_path / "trading.db"
        db = sqlite3.connect(path)
        db.executescript("""
            CREATE TABLE trades (symbol TEXT, direction TEXT, sl_distance_pips REAL,
                mfe_r REAL, pnl_r REAL, pnl_money REAL, exit_reason TEXT,
                opened_at TEXT, closed_at TEXT);
            CREATE TABLE analysis_cycles (ts TEXT, reason TEXT);
            """)
        db.execute(
            "INSERT INTO trades VALUES ('OLD.i','LONG',10.0,1.0,1.0,1.0,'SL',?,?)",
            ((NOW - timedelta(days=9)).isoformat(), (NOW - timedelta(days=8)).isoformat()),
        )
        db.commit()
        db.close()

        settings = load_settings(env_overrides=False)
        assert gather(path, settings, window_hours=24.0, now=NOW).trades == []


class TestHealthAge:
    def test_the_age_of_a_reading_travels_with_it(self, tmp_path: Path) -> None:
        """Nine minutes old on a one-second loop is the finding, and an analyst
        that cannot see the timestamp reports the stale verdict as current."""
        path = tmp_path / "position_health.json"
        written = (datetime.now(UTC) - timedelta(minutes=9)).isoformat()
        path.write_text(
            json.dumps(
                {"recorded_at": written, "positions": [{"ticket": 1, "verdict": "healthy"}]}
            ),
            encoding="utf-8",
        )
        entries = read_open_positions(path)
        assert entries[0]["reading_age_seconds"] == pytest.approx(540, abs=5)

    def test_a_missing_file_is_an_empty_list(self, tmp_path: Path) -> None:
        assert read_open_positions(tmp_path / "nope.json") == []

    def test_a_corrupt_file_is_an_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "position_health.json"
        path.write_text("{ not json", encoding="utf-8")
        assert read_open_positions(path) == []


class TestItHasNoAuthority:
    """The property that makes an open-ended reasoner safe on a live account.

    A gate has to be predictable. An analyst has to be free to say something
    nobody anticipated. Those cannot be the same component, and the separation
    is structural rather than a convention someone might forget.
    """

    def test_the_trading_path_does_not_import_the_analyst(self) -> None:
        for module in ("runner/service.py", "main.py", "jarvis.py", "execution/manager.py"):
            source = (Path(__file__).resolve().parent.parent / module).read_text(encoding="utf-8")
            assert "analyst" not in source, module

    def test_an_assessment_carries_no_verdict_of_any_kind(self) -> None:
        """Prose, and nothing a caller could mistake for permission."""
        fields = set(Assessment.__dataclass_fields__)
        for forbidden in ("approved", "verdict", "reason", "action", "direction", "volume"):
            assert forbidden not in fields, forbidden

    def test_a_finding_suggests_and_never_instructs(self) -> None:
        finding = Finding(
            what="stops too tight",
            evidence="three trades under 6 pips",
            severity="critical",
            suggested_change="raise the floor",
        )
        assert isinstance(finding.suggested_change, str)
        assert not hasattr(finding, "apply")


class TestFailureIsAMissingOpinion:
    def test_no_api_key_returns_an_error_rather_than_raising(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """A scheduled run must never be able to take the service down."""
        from analyst.review import analyse

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assessment = analyse(Evidence(window_hours=24.0, generated_at=NOW))

        assert not assessment.ok
        assert "ANTHROPIC_API_KEY" in assessment.error
        assert assessment.headline == ""

    def test_an_error_renders_as_a_sentence_not_a_traceback(self) -> None:
        assessment = Assessment(headline="", reasoning="", error="timeout")
        assert "Analyst unavailable: timeout" in assessment.render()


def test_the_dominant_refusal_ignores_success() -> None:
    """OK is not a reason anything was blocked."""
    assert dominant_refusal({"OK": 50, "NO_SIGNAL": 12, "AI_VETO": 3}) == ("NO_SIGNAL", 12)
    assert dominant_refusal({"OK": 5}) is None
    assert dominant_refusal({}) is None

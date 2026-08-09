from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pandas as pd
import pytest

from advisory.ledger import AIReviewLedger, read_recent_reviews, read_trade_reflections
from advisory.providers import DisabledAdvisor, build_advisor, build_review_payload
from analysis.confluence import TradeIdea
from config.loader import load_settings
from config.schema import AIConfig
from core.clock import SimulatedClock
from core.types import AccountSnapshot, Direction, MarketContext, Series, Signal, Tick, Timeframe
from journal.database import Journal
from reporting.daily_report import DailyReportGenerator
from reporting.execution_report import ExecutionReportGenerator
from reporting.weekly_report import WeeklyReportGenerator


def test_disabled_advisor_never_invents_a_trade() -> None:
    settings = load_settings()
    advisor = build_advisor(settings.ai)
    idea = TradeIdea("EURUSD", False, None, 0, 0, 0, 0, 0, "none", ())
    context = MarketContext("EURUSD", datetime(2026, 1, 1, tzinfo=UTC), {})

    advice = advisor.review(idea, context)

    assert isinstance(advisor, DisabledAdvisor)
    assert advice.approved
    assert advice.provider == "disabled"


def test_anthropic_review_is_structured_compact_and_fail_closed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    class Messages:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            # Adaptive thinking is on by default from Sonnet 5 onward, so a real
            # response leads with a thinking block. Only the text block is the
            # schema-constrained answer.
            thinking = SimpleNamespace(type="thinking", thinking="{not the answer}")
            block = SimpleNamespace(
                type="text",
                text=(
                    '{"approve":true,"confidence":0.8,"thesis":"coherent","risks":["event risk"]}'
                ),
            )
            return SimpleNamespace(
                content=[thinking, block], stop_reason="end_turn", _request_id="req-test"
            )

    class Client:
        def __init__(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.messages = Messages()

    monkeypatch.setitem(__import__("sys").modules, "anthropic", SimpleNamespace(Anthropic=Client))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unit-test-secret")
    adviser = build_advisor(
        AIConfig(enabled=True, provider="anthropic", anthropic_model="claude-test")
    )
    index = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.2, 1.3, 1.4, 1.5],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.1, 1.2, 1.3, 1.4],
            "tick_volume": [10, 11, 12, 13],
        },
        index=index,
    )
    now = datetime(2026, 1, 1, 5, tzinfo=UTC)
    context = MarketContext(
        "EURUSD",
        now,
        {Timeframe.H1: Series("EURUSD", Timeframe.H1, frame, now)},
        Tick("EURUSD", now, 1.3999, 1.4),
    )
    signal = Signal("market_structure", 70, 0.8, "break")
    idea = TradeIdea("EURUSD", True, Direction.LONG, 70, 0.8, 1.4, 1.3, 1.6, "two agree", (signal,))

    advice = adviser.review(idea, context, {"actual_risk_pct": 1.0})
    safe_payload = build_review_payload(idea, context, {"actual_risk_pct": 1.0})

    assert advice.approved
    assert advice.request_id == "req-test"
    request_text = str(captured["messages"])
    assert "unit-test-secret" not in request_text
    assert "actual_risk_pct" in request_text
    assert safe_payload["symbol"] == "EURUSD"
    assert safe_payload["executable_proposal"] == {"actual_risk_pct": 1.0}
    # Enough chart to read. Three bars per timeframe left the reviewer nothing
    # but the engine's own summary to restate, which is what every early answer
    # did — it could not see a level being run into or judge whether the target
    # was somewhere this market goes.
    h1 = safe_payload["timeframes"]["H1"]
    assert len(h1["rows"]) == len(frame)
    assert h1["atr14"] > 0
    assert "range_high" in h1
    # Rows, not objects: the same numbers at half the tokens, and bars are 91%
    # of the request. A row is [open, high, low, close, tick_volume] and
    # `columns` says so — a bare array with no header is a puzzle, not data.
    assert h1["columns"] == "open,high,low,close,tick_volume"
    assert all(len(row) == 5 for row in h1["rows"])
    assert h1["rows"][-1][3] == pytest.approx(float(frame["close"].iloc[-1]), abs=1e-6)
    # Per-bar timestamps are gone; the window is still pinned at both ends,
    # and the interval makes every bar in between arithmetic.
    assert h1["bar_interval"] == "H1"
    assert h1["oldest_bar_opened"] and h1["last_closed_bar"]
    output_config = captured["output_config"]
    assert isinstance(output_config, dict)
    assert output_config["format"]["type"] == "json_schema"
    assert "minimum" not in str(output_config)
    # The thinking block must never reach the JSON parser.
    assert advice.thesis == "coherent"


def test_anthropic_market_scout_nominates_without_approving_or_sizing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    class Messages:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            text = (
                '{"action":"LONG","symbol":"EURUSD.i","confidence":0.78,'
                '"thesis":"strongest aligned trend","counter_thesis":"near resistance",'
                '"invalidation_price":1.09,"target_price":1.12,'
                '"patterns":["trend"],"risks":["event"],"wait_for":""}'
            )
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)],
                stop_reason="end_turn",
                _request_id="scout-1",
            )

    class Client:
        def __init__(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.messages = Messages()

    monkeypatch.setitem(__import__("sys").modules, "anthropic", SimpleNamespace(Anthropic=Client))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unit-test-secret")
    adviser = build_advisor(
        AIConfig(enabled=True, provider="anthropic", anthropic_model="claude-test")
    )

    decision = adviser.scout({"world": {"risk_tone": "mixed"}, "markets": [{"symbol": "EURUSD.i"}]})

    assert decision.directional
    assert decision.symbol == "EURUSD.i"
    assert decision.action == "LONG"
    assert decision.request_id == "scout-1"
    assert "volume" not in str(captured["messages"])
    assert "risk_per_trade" not in str(captured["messages"])


def test_anthropic_request_omits_parameters_the_current_models_reject() -> None:
    """Regression: `temperature=0` returned HTTP 400 and read as a permanent veto.

    Sonnet 5 and the Opus 4.7+ family reject `temperature`, `top_p`, `top_k` and
    `thinking.budget_tokens`. Because the adviser is fail-closed, such a request
    does not crash — it vetoes every candidate forever while looking healthy, so
    the shape of the request is asserted rather than left to a live call.
    """
    captured: dict[str, object] = {}

    class Messages:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            verdict = '{"approve":false,"confidence":0.1,"thesis":"no","risks":[]}'
            block = SimpleNamespace(type="text", text=verdict)
            return SimpleNamespace(content=[block], stop_reason="end_turn", _request_id="r")

    class Client:
        def __init__(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.messages = Messages()

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(__import__("sys").modules, "anthropic", SimpleNamespace(Anthropic=Client))
        patch.setenv("ANTHROPIC_API_KEY", "unit-test-secret")
        adviser = build_advisor(
            AIConfig(enabled=True, provider="anthropic", anthropic_model="claude-test")
        )
        idea = TradeIdea("EURUSD", False, None, 0, 0, 0, 0, 0, "none", ())
        adviser.review(idea, MarketContext("EURUSD", datetime(2026, 1, 1, tzinfo=UTC), {}))
        adviser.reflect({"symbol": "EURUSD", "r_multiple": -1.0})

    for rejected in ("temperature", "top_p", "top_k"):
        assert rejected not in captured
    assert "budget_tokens" not in str(captured.get("thinking", ""))
    # Thinking tokens are charged against max_tokens. A budget sized for the
    # verdict alone truncates into stop_reason="max_tokens" — another silent veto.
    assert int(captured["max_tokens"]) >= 2_000


def test_a_rejected_request_reports_why_instead_of_a_bare_status(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A fail-closed veto with no reason is indistinguishable from a considered no."""

    class Rejected(Exception):
        status_code = 400
        body: ClassVar[dict[str, object]] = {
            "error": {"message": "temperature: Extra inputs are not permitted"}
        }

    class Messages:
        def create(self, **_kwargs):  # type: ignore[no-untyped-def]
            raise Rejected

    class Client:
        def __init__(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.messages = Messages()

    monkeypatch.setitem(__import__("sys").modules, "anthropic", SimpleNamespace(Anthropic=Client))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unit-test-secret")
    adviser = build_advisor(
        AIConfig(enabled=True, provider="anthropic", anthropic_model="claude-test")
    )
    idea = TradeIdea("EURUSD", False, None, 0, 0, 0, 0, 0, "none", ())

    advice = adviser.review(idea, MarketContext("EURUSD", datetime(2026, 1, 1, tzinfo=UTC), {}))

    assert not advice.approved
    assert advice.error.startswith("Rejected:http_400:")
    assert "temperature" in advice.error


def test_an_authentication_failure_reports_only_its_status(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """401/403/429/5xx bodies can carry organisation detail; the status is enough."""

    class Unauthorised(Exception):
        status_code = 401
        body: ClassVar[dict[str, object]] = {
            "error": {"message": "organisation acme-corp key sk-live-tail is revoked"}
        }

    class Messages:
        def create(self, **_kwargs):  # type: ignore[no-untyped-def]
            raise Unauthorised

    class Client:
        def __init__(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.messages = Messages()

    monkeypatch.setitem(__import__("sys").modules, "anthropic", SimpleNamespace(Anthropic=Client))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unit-test-secret")
    adviser = build_advisor(
        AIConfig(enabled=True, provider="anthropic", anthropic_model="claude-test")
    )
    idea = TradeIdea("EURUSD", False, None, 0, 0, 0, 0, 0, "none", ())

    advice = adviser.review(idea, MarketContext("EURUSD", datetime(2026, 1, 1, tzinfo=UTC), {}))

    assert advice.error == "Unauthorised:http_401"
    assert "acme-corp" not in advice.error


def test_a_truncated_response_names_the_stop_reason(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class Messages:
        def create(self, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(content=[], stop_reason="max_tokens", _request_id="r")

    class Client:
        def __init__(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.messages = Messages()

    monkeypatch.setitem(__import__("sys").modules, "anthropic", SimpleNamespace(Anthropic=Client))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unit-test-secret")
    adviser = build_advisor(
        AIConfig(enabled=True, provider="anthropic", anthropic_model="claude-test")
    )
    idea = TradeIdea("EURUSD", False, None, 0, 0, 0, 0, 0, "none", ())

    advice = adviser.review(idea, MarketContext("EURUSD", datetime(2026, 1, 1, tzinfo=UTC), {}))

    assert not advice.approved
    assert advice.error == "incomplete_response:max_tokens"


def test_anthropic_timeout_vetoes_without_leaking_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class Messages:
        def create(self, **_kwargs):  # type: ignore[no-untyped-def]
            raise TimeoutError("request headers could contain a secret")

    class Client:
        def __init__(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.messages = Messages()

    monkeypatch.setitem(__import__("sys").modules, "anthropic", SimpleNamespace(Anthropic=Client))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unit-test-secret")
    adviser = build_advisor(
        AIConfig(enabled=True, provider="anthropic", anthropic_model="claude-test")
    )
    idea = TradeIdea("EURUSD", False, None, 0, 0, 0, 0, 0, "none", ())
    context = MarketContext("EURUSD", datetime(2026, 1, 1, tzinfo=UTC), {})

    advice = adviser.review(idea, context)

    assert not advice.approved
    assert advice.error == "TimeoutError"
    assert "secret" not in advice.thesis


def test_ai_review_ledger_round_trips_safe_events(tmp_path: Path) -> None:
    path = tmp_path / "ai.jsonl"
    AIReviewLedger(path).append("pretrade_review", {"symbol": "EURUSD", "approved": False})

    rows = read_recent_reviews(path)

    assert rows[-1]["event"] == "pretrade_review"
    assert rows[-1]["symbol"] == "EURUSD"


def test_ai_review_ledger_uses_the_injected_clock(tmp_path: Path) -> None:
    path = tmp_path / "ai.jsonl"
    moment = datetime(2026, 8, 9, 12, 34, tzinfo=UTC)
    AIReviewLedger(path, clock=SimulatedClock(moment)).append("test", {})

    assert read_recent_reviews(path)[0]["timestamp"] == moment.isoformat()


def test_ai_review_ledger_also_mirrors_events_into_the_journal(tmp_path: Path) -> None:
    path = tmp_path / "ai.jsonl"
    journal = Journal(tmp_path / "journal.db", SimulatedClock(datetime(2026, 1, 1, tzinfo=UTC)))
    journal.open()

    AIReviewLedger(path, database=journal).append(
        "pretrade_request",
        {
            "cycle_id": "cycle-1",
            "symbol": "EURUSD",
            "direction": "LONG",
            "provider": "anthropic",
            "model": "claude-test",
            "request": {"score": 72},
        },
    )

    row = journal.query("SELECT * FROM ai_events")[0]
    assert row["event"] == "pretrade_request"
    assert row["cycle_id"] == "cycle-1"
    assert row["symbol"] == "EURUSD"
    assert json.loads(row["payload_json"])["request"]["score"] == 72
    journal.close()


def test_ai_review_reader_skips_a_partial_live_line(tmp_path: Path) -> None:
    path = tmp_path / "ai.jsonl"
    path.write_text(
        '{"event":"pretrade_request","cycle_id":"ok"}\n{"event":',
        encoding="utf-8",
    )

    rows = read_recent_reviews(path)

    assert rows == [{"event": "pretrade_request", "cycle_id": "ok"}]


def test_reflection_reader_excludes_failed_and_unrelated_events(tmp_path: Path) -> None:
    path = tmp_path / "ai.jsonl"
    ledger = AIReviewLedger(path)
    ledger.append("pretrade_response", {"decision": {"approved": True}})
    ledger.append(
        "posttrade_reflection",
        {"outcome": {"trade_id": 1}, "reflection": {"lessons": ["useful"], "error": ""}},
    )
    ledger.append(
        "posttrade_reflection",
        {"outcome": {"trade_id": 2}, "reflection": {"lessons": [], "error": "timeout"}},
    )

    rows = read_trade_reflections(path)

    assert [row["outcome"]["trade_id"] for row in rows] == [1]  # type: ignore[index]


def test_ai_backfill_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "ai.jsonl"
    AIReviewLedger(path).append("pretrade_response", {"cycle_id": "one"})
    journal = Journal(tmp_path / "journal.db", SimulatedClock(datetime(2026, 1, 1, tzinfo=UTC)))
    journal.open()
    ledger = AIReviewLedger(path, database=journal)

    assert ledger.backfill_database() == 1
    assert ledger.backfill_database() == 1
    assert journal.scalar("SELECT COUNT(*) FROM ai_events") == 1
    journal.close()


def test_daily_report_writes_markdown_and_pdf(tmp_path: Path) -> None:
    now = datetime(2026, 1, 5, 12, tzinfo=UTC)
    journal = Journal(tmp_path / "journal.db", SimulatedClock(now)).open()
    account = AccountSnapshot(1, "PAPER", "EUR", 100, 100, 0, 100, 0, 500, True, now)
    generator = DailyReportGenerator(journal, tmp_path / "reports")

    markdown, pdf = generator.generate(account, now)

    assert "Balance: 100.00 EUR" in markdown.read_text(encoding="utf-8")
    assert pdf.read_bytes().startswith(b"%PDF")
    journal.close()


def test_weekly_and_execution_reports_state_insufficient_sample(tmp_path: Path) -> None:
    now = datetime(2026, 1, 5, 12, tzinfo=UTC)
    settings = load_settings(env_overrides=False)
    journal = Journal(tmp_path / "journal.db", SimulatedClock(now)).open()
    account = AccountSnapshot(1, "PAPER", "EUR", 100, 100, 0, 100, 0, 500, True, now)

    weekly, weekly_pdf = WeeklyReportGenerator(journal, tmp_path / "reports", settings).generate(
        account, now
    )
    execution = ExecutionReportGenerator(
        journal, tmp_path / "reports" / "EXECUTION_REPORT.md"
    ).generate(now)

    assert "More trades needed before inference: 100" in weekly.read_text(encoding="utf-8")
    assert weekly_pdf.read_bytes().startswith(b"%PDF")
    assert "No order attempts" in execution.read_text(encoding="utf-8")
    journal.close()


def test_the_reviewer_is_told_whether_the_target_is_ever_reached() -> None:
    """A target is not realistic because the arithmetic says so.

    The engine places it at twice the stop with no reference to whether the
    market ever travels that far. Measured against the instrument's own recent
    history, the reviewer can see that a target needing six days of one-way
    movement is not a target, whatever the risk-reward ratio reads.
    """
    index = pd.date_range("2026-01-01", periods=300, freq="h", tz="UTC")
    step = pd.Series(range(300), index=index) * 0.0001
    frame = pd.DataFrame(
        {
            "open": 1.0 + step,
            "high": 1.0006 + step,
            "low": 0.9994 + step,
            "close": 1.0 + step,
            "tick_volume": 100,
        },
        index=index,
    )
    now = index[-1].to_pydatetime() + timedelta(hours=1)
    context = MarketContext(
        "EURUSD",
        now,
        {Timeframe.H1: Series("EURUSD", Timeframe.H1, frame, now)},
        Tick("EURUSD", now, 1.0299, 1.0300),
    )
    near = TradeIdea("EURUSD", True, Direction.LONG, 70, 0.8, 1.03, 1.0288, 1.0324, "x", ())
    far = TradeIdea("EURUSD", True, Direction.LONG, 70, 0.8, 1.03, 1.0288, 1.30, "x", ())

    reachable = build_review_payload(near, context, None)["target_realism"]
    unreachable = build_review_payload(far, context, None)["target_realism"]

    assert reachable["target_in_atr"] < unreachable["target_in_atr"]
    assert (
        reachable["history"]["moved_up_that_far_pct"]
        > unreachable["history"]["moved_up_that_far_pct"]
    )
    assert unreachable["history"]["moved_up_that_far_pct"] == 0.0

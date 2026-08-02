from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from advisory.ledger import AIReviewLedger, read_recent_reviews
from advisory.providers import DisabledAdvisor, build_advisor
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
            block = SimpleNamespace(
                text=(
                    '{"approve":true,"confidence":0.8,"thesis":"coherent","risks":["event risk"]}'
                )
            )
            return SimpleNamespace(content=[block], stop_reason="end_turn", _request_id="req-test")

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

    assert advice.approved
    assert advice.request_id == "req-test"
    request_text = str(captured["messages"])
    assert "unit-test-secret" not in request_text
    assert "actual_risk_pct" in request_text
    assert request_text.count("tick_volume") == 3
    output_config = captured["output_config"]
    assert isinstance(output_config, dict)
    assert output_config["format"]["type"] == "json_schema"
    assert "minimum" not in str(output_config)


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

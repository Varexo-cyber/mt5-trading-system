from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from advisory.providers import DisabledAdvisor, build_advisor
from analysis.confluence import TradeIdea
from config.loader import load_settings
from core.clock import SimulatedClock
from core.types import AccountSnapshot, MarketContext
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

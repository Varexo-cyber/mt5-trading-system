"""Read-only dashboard service and PDF report smoke tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from config.loader import load_settings
from config.schema import MT5Config
from core.mt5_codes import TIMEFRAME_VALUES
from core.mt5_connector import MT5Connector
from core.types import Timeframe
from dashboard.service import (
    DashboardService,
    catalogue_asset_class,
    load_market_intelligence,
    load_paper_snapshot,
    stop_confirmation_matches,
)
from reporting.pdf_report import build_pdf_report
from tests.fakes.fake_mt5 import FakeMT5


def test_every_mt5_timeframe_is_selectable() -> None:
    assert {timeframe.value for timeframe in Timeframe} == set(TIMEFRAME_VALUES)


def test_clear_stop_confirmation_ignores_case_and_extra_spaces() -> None:
    assert stop_confirmation_matches("CLEAR STOP")
    assert stop_confirmation_matches("clear stop")
    assert stop_confirmation_matches("  Clear    Stop  ")
    assert not stop_confirmation_matches("stop")


#: A Wednesday, 14:00 UTC. Pinned, and the pin is the point.
#:
#: This test used to seed the fake with `datetime.now(UTC) + 3h` and then
#: assert the newest bar was within two hours of the real clock. That holds
#: from Monday to Friday and fails every Saturday and Sunday: `_bar_times`
#: correctly skips weekends, so on a Saturday morning the newest bar it can
#: produce is Friday evening — twelve hours away, and growing until Monday.
#: A suite that goes red on its own at the weekend teaches everyone to ignore
#: a red suite, which is the expensive part.
MIDWEEK = datetime(2026, 8, 5, 14, 0, tzinfo=UTC)


def test_dashboard_reads_catalogue_bars_and_builds_pdf() -> None:
    connector = MT5Connector(MT5Config(), mt5_module=FakeMT5(now=MIDWEEK + timedelta(hours=3)))
    service = DashboardService(connector, load_settings(env_overrides=False))
    try:
        account = service.connect()
        symbols = service.symbols()
        spec = service.spec("EURUSD")
        tick = service.tick("EURUSD")
        frame = service.bars("EURUSD", Timeframe.H1, count=50)

        assert any(item.name == "EURUSD" for item in symbols)
        assert catalogue_asset_class("Cryptos\\High Cap\\BTCUSD").value == "crypto"
        assert len(frame) == 50
        # Still the real assertion — the connector must serve bars that reach
        # up to the present, not stale history — measured against the clock the
        # fake was actually given.
        newest = frame.index[-1].to_pydatetime()
        assert abs((newest - (MIDWEEK + timedelta(hours=3))).total_seconds()) < 7200

        pdf = build_pdf_report(account, [], "EURUSD", spec, tick, {"H1": frame})
        assert pdf.startswith(b"%PDF")
        assert len(pdf) > 10_000
    finally:
        service.close()


def test_dashboard_reads_persistent_paper_positions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "paper.json"
    path.write_text('{"balance":99.5,"currency":"EUR","positions":[]}', encoding="utf-8")

    snapshot = load_paper_snapshot(path)

    assert snapshot is not None
    assert snapshot.equity == 99.5
    assert snapshot.currency == "EUR"


def test_dashboard_reads_market_brain_snapshot(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "market_intelligence.json"
    path.write_text('{"world":{"risk_tone":"mixed"}}', encoding="utf-8")

    snapshot = load_market_intelligence(path)

    assert snapshot is not None
    assert snapshot["world"] == {"risk_tone": "mixed"}


class TestTheDeckShowsWhoMaySpendMoney:
    """`render_live_sections` answers the question the owner asked out loud --
    "welke staan er nu allemaal live, ik snap het niet meer" -- after four
    promotions and three removals in two days.

    THESE DRIVE THE PANEL, they do not read its source. `dashboard/app.py`
    connects to MT5 at import time, so the panel's own block is compiled and
    executed against a fake Streamlit. A panel asserted by substring is a panel
    nobody has ever run, and the first time it ran it produced `USDJPY.i.i` --
    a symbol that exists nowhere, from appending the broker suffix twice.
    """

    @staticmethod
    def _run(settings):  # type: ignore[no-untyped-def]
        """Execute the panel with a recording stand-in for Streamlit."""
        import sys
        import types
        from pathlib import Path

        import pandas as pd

        calls: list[tuple[str, object]] = []

        class _Ctx:
            def __enter__(self):  # type: ignore[no-untyped-def]
                return self

            def __exit__(self, *args):  # type: ignore[no-untyped-def]
                return False

        fake = types.ModuleType("streamlit")
        for name in ("subheader", "warning", "error", "caption", "dataframe", "divider"):
            fake.__dict__[name] = (
                lambda key: (lambda *a, **k: calls.append((key, a[0] if a else None)))
            )(name)
        fake.expander = lambda *a, **k: _Ctx()

        root = Path(__file__).resolve().parents[1]
        source = (root / "dashboard" / "app.py").read_text()
        block = source[
            source.index("def _section_markets(") : source.index("def render_account_header(")
        ]
        namespace: dict = {"st": fake, "pd": pd, "ROOT": root}
        previous = sys.modules.get("streamlit")
        sys.modules["streamlit"] = fake
        try:
            exec(compile(block, "panel", "exec"), namespace)
            namespace["render_live_sections"](settings)
        finally:
            if previous is None:
                sys.modules.pop("streamlit", None)
            else:
                sys.modules["streamlit"] = previous
        return calls, namespace

    @staticmethod
    def _live_settings():  # type: ignore[no-untyped-def]
        from pathlib import Path

        return load_settings(
            overlay=Path(__file__).resolve().parents[1] / "config" / "eightcap.yaml",
            env_overrides=False,
        )

    def test_it_names_every_section_that_may_spend_money(self) -> None:
        settings = self._live_settings()
        calls, _ns = self._run(settings)
        tables = [payload for kind, payload in calls if kind == "dataframe"]
        assert tables, "the panel drew no table at all"
        shown = set(tables[0]["sectie"])
        assert shown == set(settings.analysis.confluence.live_enabled_modules)

    def test_the_count_in_the_heading_matches_the_config(self) -> None:
        settings = self._live_settings()
        calls, _ns = self._run(settings)
        heading = next(payload for kind, payload in calls if kind == "subheader")
        assert str(len(settings.analysis.confluence.live_enabled_modules)) in str(heading)

    def test_no_market_carries_the_broker_suffix_twice(self) -> None:
        """The bug the first run produced. `section_nine_vwap_m30` stores
        `USDJPY.i` -- already the broker's spelling -- and resolving it again
        made `USDJPY.i.i`, which is not a symbol on any account."""
        settings = self._live_settings()
        suffix = settings.instruments.symbol_suffix
        calls, _ns = self._run(settings)
        for kind, payload in calls:
            if kind != "dataframe" or "markt" not in getattr(payload, "columns", []):
                continue
            for market in payload["markt"]:
                assert suffix == "" or not str(market).endswith(suffix + suffix), market

    def test_a_permitted_section_with_no_weight_is_called_a_fault(self) -> None:
        """The pair that is permitted and inert. `ConfluenceEngine` tests
        `if weight > 0`, so this section is allowed to trade and counted by
        nothing -- zero trades forever, and from every other surface on the
        deck it looks exactly like a quiet market. It has been wrong on this
        account twice, so the panel has to say it rather than show a 0.0."""
        settings = self._live_settings()
        confluence = settings.analysis.confluence
        victim = confluence.live_enabled_modules[0]
        weights = dict(confluence.weights)
        weights[victim] = 0.0
        broken = settings.model_copy(
            update={
                "analysis": settings.analysis.model_copy(
                    update={"confluence": confluence.model_copy(update={"weights": weights})}
                )
            }
        )
        calls, _ns = self._run(broken)
        errors = [str(payload) for kind, payload in calls if kind == "error"]
        assert any(victim in text and "gewicht 0" in text for text in errors), errors

    def test_an_empty_allowlist_says_so_instead_of_drawing_nothing(self) -> None:
        """An empty table reads as a broken panel; the words read as the truth."""
        settings = self._live_settings()
        confluence = settings.analysis.confluence
        empty = settings.model_copy(
            update={
                "analysis": settings.analysis.model_copy(
                    update={
                        "confluence": confluence.model_copy(update={"live_enabled_modules": ()})
                    }
                )
            }
        )
        calls, _ns = self._run(empty)
        warnings = [str(payload) for kind, payload in calls if kind == "warning"]
        assert any("GEEN ENKELE" in text for text in warnings), warnings

    def test_the_shadow_sections_are_listed_separately(self) -> None:
        settings = self._live_settings()
        live = set(settings.analysis.confluence.live_enabled_modules)
        expected = {
            name
            for name in vars(settings.analysis)
            if name.startswith("section_")
            and getattr(getattr(settings.analysis, name), "enabled", False)
            and name not in live
        }
        calls, _ns = self._run(settings)
        tables = [payload for kind, payload in calls if kind == "dataframe"]
        if not expected:
            assert len(tables) == 1
            return
        assert len(tables) == 2, "the shadow table is missing"
        assert set(tables[1]["sectie"]) == expected

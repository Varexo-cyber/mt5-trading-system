"""Startup guard arithmetic and the kill switch."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from config.schema import MT5Config
from core.errors import StartupGuardError
from core.mt5_connector import MT5Connector
from core.startup import enforce, run_startup_guard
from infra.killswitch import KillSwitch
from tests.fakes.fake_mt5 import FakeMT5


@pytest.fixture
def raw() -> dict[str, Any]:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def settings_for(tmp_path: Path, raw: dict[str, Any], **system: Any) -> Any:
    data = copy.deepcopy(raw)
    data["system"].update(system)
    # Keep the fake broker's two symbols so the guard has something to assess.
    data["instruments"]["whitelist"]["micro_live"] = ["EURUSD", "USDJPY"]
    data["instruments"]["whitelist"]["scaling"] = ["EURUSD", "USDJPY"]
    data["instruments"]["whitelist"]["backtest"] = ["EURUSD", "USDJPY"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_settings(path, env_overrides=False)


def connected(equity: float, *, is_demo: bool = True) -> MT5Connector:
    fake = FakeMT5(equity=equity, balance=equity, currency="EUR", is_demo=is_demo)
    connector = MT5Connector(MT5Config(), mt5_module=fake)
    connector.connect()
    return connector


class TestFeasibilityArithmetic:
    def test_hundred_euro_account_is_reported_as_barely_expressible(
        self, tmp_path: Path, raw: dict[str, Any]
    ) -> None:
        settings = settings_for(tmp_path, raw, mode="micro_live")
        connector = connected(100.0)
        report = run_startup_guard(settings, connector, connector.account())

        eurusd = next(s for s in report.symbols if s.symbol == "EURUSD")
        # 1% of 100 = EUR 1; at 0.01 lots EURUSD costs ~0.10 per pip.
        assert eurusd.max_affordable_sl_pips == pytest.approx(10.0, abs=0.5)
        # The mode's 30-pip ceiling would cost 3% — above the 2% cap.
        assert eurusd.risk_pct_at_max_sl == pytest.approx(3.0, abs=0.1)
        assert any("skipped" in w for w in report.warnings)

    def test_larger_account_has_room(self, tmp_path: Path, raw: dict[str, Any]) -> None:
        settings = settings_for(tmp_path, raw, mode="scaling")
        connector = connected(5_000.0)
        report = run_startup_guard(settings, connector, connector.account())

        eurusd = next(s for s in report.symbols if s.symbol == "EURUSD")
        assert eurusd.max_affordable_sl_pips == pytest.approx(500.0, abs=5.0)
        assert eurusd.is_expressible
        assert report.ok

    def test_report_renders_without_crashing(self, tmp_path: Path, raw: dict[str, Any]) -> None:
        settings = settings_for(tmp_path, raw, mode="micro_live")
        connector = connected(150.0)
        text = run_startup_guard(settings, connector, connector.account()).render()
        assert "STARTUP CHECK" in text
        assert "EURUSD" in text
        assert "Risk/trade" in text


class TestGuardBlocks:
    def test_equity_below_the_mode_floor_blocks(self, tmp_path: Path, raw: dict[str, Any]) -> None:
        settings = settings_for(tmp_path, raw, mode="micro_live")
        connector = connected(20.0)  # micro_live floor is 50
        report = run_startup_guard(settings, connector, connector.account())
        assert not report.ok
        assert any("below the micro_live minimum" in e for e in report.errors)

    def test_equity_above_the_mode_ceiling_blocks(
        self, tmp_path: Path, raw: dict[str, Any]
    ) -> None:
        settings = settings_for(tmp_path, raw, mode="micro_live")
        connector = connected(5_000.0)  # micro_live ceiling is 500
        report = run_startup_guard(settings, connector, connector.account())
        assert not report.ok
        assert any("exceeds the micro_live ceiling" in e for e in report.errors)

    def test_paper_mode_on_a_live_account_blocks(self, tmp_path: Path, raw: dict[str, Any]) -> None:
        """A backtest run one code path away from a real order is unacceptable."""
        settings = settings_for(tmp_path, raw, mode="backtest")
        connector = connected(1_000.0, is_demo=False)
        report = run_startup_guard(settings, connector, connector.account())
        assert not report.ok
        assert any("LIVE" in e for e in report.errors)

    def test_micro_live_on_demo_warns_but_does_not_block(
        self, tmp_path: Path, raw: dict[str, Any]
    ) -> None:
        settings = settings_for(tmp_path, raw, mode="micro_live")
        connector = connected(150.0, is_demo=True)
        report = run_startup_guard(settings, connector, connector.account())
        assert report.ok
        assert any("DEMO" in w for w in report.warnings)

    def test_a_symbol_the_account_cannot_afford_yet_warns_but_does_not_block(
        self, tmp_path: Path, raw: dict[str, Any]
    ) -> None:
        """A EUR 100 whitelist naming XAUUSD is a small account, not a broken config.

        The equity floor already forbids the trade. Refusing to *start* on top of
        that would mean hand-editing the whitelist every time equity crosses a
        threshold, which is exactly what the floors exist to avoid.
        """
        data = copy.deepcopy(raw)
        data["system"]["mode"] = "micro_live"
        data["instruments"]["whitelist"]["micro_live"] = ["EURUSD", "XAUUSD"]
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        settings = load_settings(path, env_overrides=False)

        connector = connected(100.0)
        report = run_startup_guard(settings, connector, connector.account())

        assert report.ok
        assert any("XAUUSD" in w and "as the account grows" in w for w in report.warnings)
        assert not any("XAUUSD" in e for e in report.errors)

    def test_an_equity_floor_that_rules_out_everything_still_blocks(
        self, tmp_path: Path, raw: dict[str, Any]
    ) -> None:
        """Downgrading the per-symbol check must not lose the case that matters."""
        data = copy.deepcopy(raw)
        data["system"]["mode"] = "micro_live"
        data["instruments"]["whitelist"]["micro_live"] = ["XAUUSD"]
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        settings = load_settings(path, env_overrides=False)

        connector = connected(100.0)
        report = run_startup_guard(settings, connector, connector.account())

        assert not report.ok
        assert any("no tradable symbol survived" in e for e in report.errors)

    def test_missing_symbol_blocks_startup(self, tmp_path: Path, raw: dict[str, Any]) -> None:
        data = copy.deepcopy(raw)
        data["system"]["mode"] = "backtest"
        data["instruments"]["whitelist"]["backtest"] = ["EURUSD", "GBPUSD"]  # fake has no GBPUSD
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        settings = load_settings(path, env_overrides=False)

        connector = connected(1_000.0)
        report = run_startup_guard(settings, connector, connector.account())
        assert not report.ok
        assert any("GBPUSD" in e for e in report.errors)

    def test_enforce_raises_on_a_failing_report(
        self, tmp_path: Path, raw: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings = settings_for(tmp_path, raw, mode="micro_live")
        connector = connected(20.0)
        report = run_startup_guard(settings, connector, connector.account())
        with pytest.raises(StartupGuardError):
            enforce(report, require_confirmation=False)
        assert "STARTUP CHECK" in capsys.readouterr().out


class TestKillSwitch:
    def test_absent_file_means_running(self, tmp_path: Path) -> None:
        assert not KillSwitch.in_dir(tmp_path).is_engaged()

    def test_present_file_means_halt(self, tmp_path: Path) -> None:
        (tmp_path / "STOP").write_text("manual halt\n", encoding="utf-8")
        switch = KillSwitch.in_dir(tmp_path)
        assert switch.is_engaged()
        assert switch.reason() == "manual halt"

    def test_engage_and_clear_roundtrip(self, tmp_path: Path) -> None:
        switch = KillSwitch.in_dir(tmp_path)
        switch.engage("drawdown circuit breaker")
        assert switch.is_engaged()
        assert "drawdown circuit breaker" in switch.reason()

        switch.clear()
        assert not switch.is_engaged()

    def test_an_empty_stop_file_still_halts(self, tmp_path: Path) -> None:
        """`touch STOP` from a phone must work, note or no note."""
        (tmp_path / "STOP").touch()
        switch = KillSwitch.in_dir(tmp_path)
        assert switch.is_engaged()
        assert switch.reason() == ""

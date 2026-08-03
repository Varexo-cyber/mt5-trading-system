"""Configuration loading and the hard rules encoded in the schema."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from config.loader import DEFAULT_CONFIG_PATH, load_credentials, load_settings
from config.schema import Settings
from core.errors import ConfigError
from core.types import TradingMode


@pytest.fixture
def raw() -> dict[str, Any]:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestShippedConfig:
    def test_default_config_is_valid(self) -> None:
        settings = load_settings(env_overrides=False)
        assert isinstance(settings, Settings)

    def test_default_mode_is_not_live(self) -> None:
        # Cloning the repo and running main.py must never place a real order.
        settings = load_settings(env_overrides=False)
        assert not settings.mode.is_live

    def test_micro_live_whitelist_is_majors_only(self, raw: dict[str, Any]) -> None:
        assert set(raw["instruments"]["whitelist"]["micro_live"]) == {
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "AUDUSD",
        }

    def test_gold_requires_more_equity(self, raw: dict[str, Any]) -> None:
        assert raw["instruments"]["min_equity_for_symbol"]["XAUUSD"] >= 500


class TestHardRules:
    """Rules that must be impossible to disable by editing YAML."""

    @pytest.mark.parametrize(
        "practice",
        [
            "martingale",
            "grid",
            "averaging_down",
            "recovery_lot_increase",
            "trade_without_stop_loss",
        ],
    )
    def test_forbidden_practices_cannot_be_enabled(
        self, raw: dict[str, Any], tmp_path: Path, practice: str
    ) -> None:
        data = copy.deepcopy(raw)
        data["risk"]["forbidden"][practice] = True
        with pytest.raises(ConfigError):
            load_settings(write(tmp_path, data), env_overrides=False)

    def test_news_filter_cannot_be_disabled(self, raw: dict[str, Any], tmp_path: Path) -> None:
        data = copy.deepcopy(raw)
        data["filters"]["news"]["enabled"] = False
        with pytest.raises(ConfigError):
            load_settings(write(tmp_path, data), env_overrides=False)

    def test_news_filter_cannot_fail_open(self, raw: dict[str, Any], tmp_path: Path) -> None:
        data = copy.deepcopy(raw)
        data["filters"]["news"]["fail_closed"] = False
        with pytest.raises(ConfigError):
            load_settings(write(tmp_path, data), env_overrides=False)

    def test_news_window_cannot_be_narrowed(self, raw: dict[str, Any], tmp_path: Path) -> None:
        data = copy.deepcopy(raw)
        data["filters"]["news"]["high_impact"]["minutes_before"] = 30
        with pytest.raises(ConfigError, match="60/30"):
            load_settings(write(tmp_path, data), env_overrides=False)

    def test_news_window_may_be_widened(self, raw: dict[str, Any], tmp_path: Path) -> None:
        data = copy.deepcopy(raw)
        data["filters"]["news"]["high_impact"]["minutes_before"] = 90
        settings = load_settings(write(tmp_path, data), env_overrides=False)
        assert settings.filters.news.high_impact.minutes_before == 90

    def test_single_calendar_provider_is_rejected(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        data = copy.deepcopy(raw)
        data["filters"]["news"]["providers"] = ["jblanked"]
        with pytest.raises(ConfigError, match="two calendar providers"):
            load_settings(write(tmp_path, data), env_overrides=False)


class TestRiskCoherence:
    def test_risk_above_its_own_ceiling_is_rejected(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        data = copy.deepcopy(raw)
        data["risk"]["risk_per_trade_pct"] = 5.0
        data["risk"]["max_risk_per_trade_pct"] = 2.0
        with pytest.raises(ConfigError, match="exceeds"):
            load_settings(write(tmp_path, data), env_overrides=False)

    def test_mode_cannot_loosen_the_global_ceiling(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        data = copy.deepcopy(raw)
        data["modes"]["micro_live"]["max_risk_per_trade_pct"] = 10.0
        with pytest.raises(ConfigError, match="global ceiling"):
            load_settings(write(tmp_path, data), env_overrides=False)

    def test_mode_cannot_exceed_global_trade_count(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        data = copy.deepcopy(raw)
        data["modes"]["micro_live"]["max_trades_per_day"] = 20
        with pytest.raises(ConfigError, match="max_trades_per_day"):
            load_settings(write(tmp_path, data), env_overrides=False)

    def test_daily_stop_must_sit_below_weekly(self, raw: dict[str, Any], tmp_path: Path) -> None:
        data = copy.deepcopy(raw)
        data["modes"]["micro_live"]["daily_loss_limit_pct"] = 9.0
        with pytest.raises(ConfigError, match="weekly"):
            load_settings(write(tmp_path, data), env_overrides=False)

    def test_circuit_breaker_must_sit_above_weekly(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        data = copy.deepcopy(raw)
        data["risk"]["max_drawdown_circuit_breaker_pct"] = 5.0
        with pytest.raises(ConfigError, match="circuit breaker"):
            load_settings(write(tmp_path, data), env_overrides=False)

    def test_kelly_is_capped_at_a_half(self, raw: dict[str, Any], tmp_path: Path) -> None:
        data = copy.deepcopy(raw)
        data["risk"]["kelly_fraction"] = 1.0
        with pytest.raises(ConfigError):
            load_settings(write(tmp_path, data), env_overrides=False)


class TestModeResolution:
    def test_micro_live_limits_apply(self, raw: dict[str, Any], tmp_path: Path) -> None:
        data = copy.deepcopy(raw)
        data["system"]["mode"] = "micro_live"
        settings = load_settings(write(tmp_path, data), env_overrides=False)

        assert settings.mode is TradingMode.MICRO_LIVE
        assert settings.effective_max_risk_pct() == 2.0
        assert settings.effective_risk_pct() == 1.0  # configured 1%, mode allows up to 2%
        assert settings.effective_max_trades_per_day() == 6
        assert settings.effective_daily_loss_limit_pct() == 4.0
        assert settings.active_limits.max_journal_detail is True

    def test_scaling_tightens_risk(self, raw: dict[str, Any], tmp_path: Path) -> None:
        data = copy.deepcopy(raw)
        data["system"]["mode"] = "scaling"
        settings = load_settings(write(tmp_path, data), env_overrides=False)
        assert settings.effective_max_risk_pct() == 1.0

    def test_missing_mode_entry_is_rejected(self, raw: dict[str, Any], tmp_path: Path) -> None:
        data = copy.deepcopy(raw)
        data["system"]["mode"] = "micro_live"
        del data["modes"]["micro_live"]
        with pytest.raises(ConfigError, match="no entry for the active mode"):
            load_settings(write(tmp_path, data), env_overrides=False)

    def test_symbol_suffix_is_applied_to_the_whitelist(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        data = copy.deepcopy(raw)
        data["system"]["mode"] = "micro_live"
        data["instruments"]["symbol_suffix"] = ".pro"
        settings = load_settings(write(tmp_path, data), env_overrides=False)
        assert "EURUSD.pro" in settings.active_whitelist
        assert settings.symbol_allowed_at_equity("EURUSD.pro", 100.0)[0]

    def test_symbol_override_bypasses_global_suffix(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        data = copy.deepcopy(raw)
        data["system"]["mode"] = "backtest"
        data["instruments"]["symbol_suffix"] = ".i"
        data["instruments"]["symbol_overrides"] = {"XAUUSD": "XAUUSD"}
        settings = load_settings(write(tmp_path, data), env_overrides=False)

        assert "EURUSD.i" in settings.active_whitelist
        assert "XAUUSD" in settings.active_whitelist
        assert "XAUUSD.i" not in settings.active_whitelist
        assert settings.symbol_allowed_at_equity("XAUUSD", 1000.0) == (True, "OK")

    def test_equity_gate_blocks_gold_on_a_small_account(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        data = copy.deepcopy(raw)
        data["system"]["mode"] = "scaling"
        settings = load_settings(write(tmp_path, data), env_overrides=False)

        allowed, reason = settings.symbol_allowed_at_equity("XAUUSD", equity=100.0)
        assert not allowed
        assert reason == "SYMBOL_BLOCKED_EQUITY_BELOW_500"

        allowed, reason = settings.symbol_allowed_at_equity("XAUUSD", equity=1000.0)
        assert allowed and reason == "OK"

    def test_non_whitelisted_symbol_is_refused(self, raw: dict[str, Any], tmp_path: Path) -> None:
        data = copy.deepcopy(raw)
        data["system"]["mode"] = "micro_live"
        settings = load_settings(write(tmp_path, data), env_overrides=False)
        allowed, reason = settings.symbol_allowed_at_equity("XAUUSD", equity=10_000.0)
        assert not allowed
        assert reason == "SYMBOL_NOT_WHITELISTED_FOR_MICRO_LIVE"


class TestLoaderMechanics:
    def test_unknown_key_is_an_error(self, raw: dict[str, Any], tmp_path: Path) -> None:
        data = copy.deepcopy(raw)
        data["risk"]["risk_per_trade_pc"] = 1.0  # typo
        with pytest.raises(ConfigError, match=r"Extra inputs|extra"):
            load_settings(write(tmp_path, data), env_overrides=False)

    def test_missing_file_is_reported_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_settings(tmp_path / "nope.yaml", env_overrides=False)

    def test_malformed_yaml_is_reported_clearly(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("system: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError, match="could not parse"):
            load_settings(path, env_overrides=False)

    def test_env_override_is_typed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TS_SYSTEM__LOOP_INTERVAL_SECONDS", "12.5")
        settings = load_settings()
        assert settings.system.loop_interval_seconds == 12.5

    def test_overlay_replaces_lists_wholesale(self, raw: dict[str, Any], tmp_path: Path) -> None:
        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(
            yaml.safe_dump({"instruments": {"whitelist": {"backtest": ["EURUSD"]}}}),
            encoding="utf-8",
        )
        data = copy.deepcopy(raw)
        settings = load_settings(write(tmp_path, data), overlay=overlay, env_overrides=False)
        assert settings.instruments.whitelist["backtest"] == ("EURUSD",)

    def test_settings_are_immutable(self) -> None:
        settings = load_settings(env_overrides=False)
        with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError
            settings.risk.risk_per_trade_pct = 99.0  # type: ignore[misc]


class TestCredentials:
    def test_missing_credentials_are_optional_for_backtests(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"):
            monkeypatch.delenv(name, raising=False)
        assert load_credentials(tmp_path / ".env", required=False) is None

    def test_missing_credentials_are_fatal_when_required(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"):
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(ConfigError, match="MT5_LOGIN"):
            load_credentials(tmp_path / ".env", required=True)

    def test_password_never_appears_in_repr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MT5_LOGIN", "42")
        monkeypatch.setenv("MT5_PASSWORD", "hunter2-should-not-leak")
        monkeypatch.setenv("MT5_SERVER", "Broker-Demo")
        creds = load_credentials(tmp_path / ".env")
        assert creds is not None
        assert "hunter2" not in repr(creds)
        assert "hunter2" not in f"{creds}"
        assert creds.password == "hunter2-should-not-leak"


class TestUniverseMode:
    """`affordable` widens what is considered, never what may be risked."""

    def test_whitelist_mode_blocks_everything_else(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        settings = load_settings(write(tmp_path, raw), env_overrides=False)
        assert settings.instruments.universe_mode == "whitelist"
        allowed, reason = settings.symbol_allowed_at_equity("AAPL", 10_000.0)
        assert not allowed
        assert reason.startswith("SYMBOL_NOT_WHITELISTED")

    def test_affordable_mode_lets_the_sizer_decide(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        data = copy.deepcopy(raw)
        data["instruments"]["universe_mode"] = "affordable"
        settings = load_settings(write(tmp_path, data), env_overrides=False)

        allowed, reason = settings.symbol_allowed_at_equity("AAPL", 10_000.0)
        assert allowed and reason == "OK"

    def test_equity_floors_still_apply_in_affordable_mode(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        """Widening the universe must not disarm the per-symbol equity gate."""
        data = copy.deepcopy(raw)
        data["instruments"]["universe_mode"] = "affordable"
        settings = load_settings(write(tmp_path, data), env_overrides=False)

        allowed, reason = settings.symbol_allowed_at_equity("XAUUSD", 100.0)
        assert not allowed
        assert reason == "SYMBOL_BLOCKED_EQUITY_BELOW_500"

    def test_blocklist_wins_in_both_modes(self, raw: dict[str, Any], tmp_path: Path) -> None:
        for mode in ("whitelist", "affordable"):
            data = copy.deepcopy(raw)
            data["instruments"]["universe_mode"] = mode
            data["instruments"]["blocklist"] = ["EURUSD"]
            settings = load_settings(write(tmp_path, data), env_overrides=False)
            allowed, reason = settings.symbol_allowed_at_equity("EURUSD", 10_000.0)
            assert not allowed, mode
            assert reason == "SYMBOL_BLOCKLISTED"

    def test_blocklist_matches_through_a_broker_suffix(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        data = copy.deepcopy(raw)
        data["instruments"]["universe_mode"] = "affordable"
        data["instruments"]["symbol_suffix"] = ".i"
        data["instruments"]["blocklist"] = ["GBPJPY"]
        settings = load_settings(write(tmp_path, data), env_overrides=False)
        assert settings.symbol_allowed_at_equity("GBPJPY.i", 10_000.0)[1] == "SYMBOL_BLOCKLISTED"


class TestTradeFrequency:
    def test_the_daily_loss_limit_binds_before_the_trade_count(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        """The count is a backstop against a bug, not the throttle on trading.

        At 1% risk a 4% daily stop halts after four full losers, well before
        the six-trade ceiling. If the count were the binding constraint the
        system would be refusing good setups for bookkeeping reasons.
        """
        data = copy.deepcopy(raw)
        data["system"]["mode"] = "micro_live"
        settings = load_settings(write(tmp_path, data), env_overrides=False)

        losers_to_daily_stop = (
            settings.effective_daily_loss_limit_pct() / settings.effective_risk_pct()
        )
        assert losers_to_daily_stop < settings.effective_max_trades_per_day()

    def test_shipped_eightcap_overlay_is_valid(self) -> None:
        """The live overlay must load; a duplicate YAML key silently drops half."""
        settings = load_settings(
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml", env_overrides=False
        )
        assert settings.instruments.symbol_suffix == ".i"
        assert settings.instruments.universe_mode == "affordable"
        assert settings.instruments.symbol_overrides["XAUUSD"] == "XAUUSD"
        assert settings.ai.anthropic_model == "claude-sonnet-5"
        # Large enough to cover everything the cheap scan lets through on a full
        # broker catalogue, rather than a top-N slice of it.
        assert settings.scanner.deep_candidates >= 200
        # One module plus Claude. Two modules cannot agree in practice: the two
        # heaviest look for opposite market states.
        assert settings.analysis.confluence.minimum_directional_modules == 1
        # Gold, silver and the indices are analysed like everything else. The
        # position sizer decides what EUR 100 can express; a hand-written floor
        # refused them before anyone measured the actual setup.
        assert settings.instruments.min_equity_for_symbol == {}
        for symbol in ("XAUUSD", "US30", "BTCUSD"):
            allowed, reason = settings.symbol_allowed_at_equity(symbol, 100.0)
            assert allowed, f"{symbol} blocked: {reason}"
        # The full ladder, weekly down to one minute.
        assert set(settings.data.timeframes) >= {"W1", "D1", "H4", "H1", "M15", "M5", "M1"}


def test_an_empty_mapping_in_an_overlay_clears_the_inherited_one(tmp_path: Path) -> None:
    """Regression: `min_equity_for_symbol: {}` silently did nothing.

    A recursive merge visits no keys inside an empty mapping, so an overlay that
    plainly says "there are no equity floors" left every inherited floor
    standing and gold stayed blocked. Nobody writes `{}` to mean "no change" —
    they omit the key — so it can only mean "clear this".
    """
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert base["instruments"]["min_equity_for_symbol"], "fixture assumes the base has floors"

    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump({"instruments": {"min_equity_for_symbol": {}}}), encoding="utf-8"
    )
    settings = load_settings(overlay=overlay, env_overrides=False)

    assert settings.instruments.min_equity_for_symbol == {}


def test_a_populated_mapping_in_an_overlay_still_merges(tmp_path: Path) -> None:
    """Clearing on empty must not turn every partial override into a wipe."""
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump({"instruments": {"min_equity_for_symbol": {"XAUUSD": 250}}}),
        encoding="utf-8",
    )
    settings = load_settings(overlay=overlay, env_overrides=False)

    floors = settings.instruments.min_equity_for_symbol
    assert floors["XAUUSD"] == 250.0
    assert "US30" in floors, "keys not mentioned in the overlay must survive"

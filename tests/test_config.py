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

    def test_eightcap_overlay_scans_markets_that_differ_from_each_other(self) -> None:
        """Four families that are genuinely four bets, not one bet four times.

        The overlay ran for a night with no narrowing at all: 83,183 decisions,
        zero trades. Half the work went to single-name shares whose history the
        broker cannot supply, and two thirds of everything that did produce a
        signal died on CORRELATED_EXPOSURE — correctly, because UK shares track
        their index above 0.7 on H1.

        `stock` is therefore out. That is a narrowing and not a loosening: no
        gate moved, only the noise the gates were tripping over.

        `stock` is back, and the reasoning above is now enforced by MEASUREMENT
        rather than by the list. Every failure it describes has a gate of its
        own: ENR gapped 2.61R through its stop and `max_gap_atr` refuses an
        instrument that jumps; QIA and ANTO quoted two cents on a EUR 36 price
        and `min_bar_range_in_spreads` refuses a market whose bars do not cover
        their own round trip; the CORRELATED_EXPOSURE deaths are the correlation
        filter doing its job.

        That is strictly better than the class ban. The real defect of ENR was
        not that it is a share, it is that it opened past its stop — and a
        forex pair that started doing that would now be refused on exactly the
        same evidence, where the list would have waved it through.
        """
        overlay = DEFAULT_CONFIG_PATH.parent / "eightcap.yaml"
        settings = load_settings(overlay=overlay, env_overrides=False)
        assert settings.instruments.asset_classes == (
            "forex",
            "metal",
            "index",
            "commodity",
            "stock",
            "crypto",
        )
        assert settings.instruments.symbols_only == ()
        # A cycle has to be able to FINISH. With shares back in the catalogue
        # an uncapped pass is 800+ symbols, and five minutes after launch the
        # log carried only "slow MT5 calls" and "symbol held out of deep
        # analysis" — no cycle line at all. Scanning was happening; nothing was
        # ever completed, and an M1 entry three minutes old does not exist.
        #
        # 240 a cycle with the liquid core recurring is the design that
        # `instruments.asset_classes` already described and that was never
        # switched on: forex, crypto and the priority symbols get fresh data
        # every pass, the rest of the catalogue rotates behind them.
        assert settings.scanner.batch_size == 240
        assert settings.scanner.priority_every_cycle
        assert settings.scanner.deep_candidates >= 847
        assert settings.instruments.ignored_symbols == ("XAUUSD",)
        assert settings.instruments.is_ignored("XAUUSD")
        assert settings.scanner.priority_asset_classes == ("forex", "crypto")
        assert {"EURUSD", "BTCUSD", "XAUUSD"} <= set(settings.scanner.priority_symbols)
        assert settings.scanner.priority_spread_weight > 0
        assert not settings.ai.market_scout.enabled
        assert settings.analysis.playbooks.enabled
        # Switched on at the owner's request on 18 August and switched back off
        # the same day, on a measurement rather than an argument.
        # `backtest_playbooks.py --days 180 --min-conviction 75` over four
        # majors: momentum_scalp -172.15R over 307 trades, range_fade -156.32R
        # over 898, range_break -76.28R over 214. Against a coin flip taking the
        # same moments, stops and targets, not one of them won.
        # `range_fade` is back on at the owner's instruction after he read the
        # backtest himself: 898 trades, -156.32R, and +0.082R against a coin
        # flip taking the same moments and stops, which is inside chance.
        assert settings.analysis.playbooks.live_execution_enabled
        assert settings.analysis.playbooks.range_fade
        # The two that were WORSE than the coin stay off. Nothing was asked for
        # them and there is nothing to say for them.
        assert not settings.analysis.playbooks.momentum_scalp
        assert not settings.analysis.playbooks.range_break
        # And the route stays the strictest one in the system, which is what
        # keeps this a setup type rather than a shortcut.
        assert settings.analysis.playbooks.min_conviction > (
            settings.analysis.confluence.score_threshold
        )
        assert settings.analysis.playbooks.require_method_agreement

    def test_eightcap_overlay_temporarily_disables_the_rio_bridge(self) -> None:
        overlay = DEFAULT_CONFIG_PATH.parent / "eightcap.yaml"
        settings = load_settings(overlay=overlay, env_overrides=False)

        assert not settings.external_signals.enabled
        assert not settings.external_signals.gold_follow_enabled

    def test_micro_live_whitelist_is_majors_only(self, raw: dict[str, Any]) -> None:
        assert set(raw["instruments"]["whitelist"]["micro_live"]) == {
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "AUDUSD",
        }

    def test_gold_requires_more_equity(self, raw: dict[str, Any]) -> None:
        assert raw["instruments"]["min_equity_for_symbol"]["XAUUSD"] >= 500

    def test_ai_management_cooldown_cannot_exceed_routine_cadence(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        data = copy.deepcopy(raw)
        data["trade_management"]["supervision_interval_minutes"] = 5.0
        data["trade_management"]["supervision_min_interval_minutes"] = 6.0
        with pytest.raises(ConfigError):
            load_settings(write(tmp_path, data), env_overrides=False)

    def test_analysis_parameter_relationships_are_validated(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        data = copy.deepcopy(raw)
        data["analysis"]["trend_momentum"]["fast_ema"] = 100
        data["analysis"]["trend_momentum"]["slow_ema"] = 50

        with pytest.raises(ConfigError, match="fast EMA"):
            load_settings(write(tmp_path, data), env_overrides=False)


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

    def test_entry_quality_cannot_fail_open(self, raw: dict[str, Any], tmp_path: Path) -> None:
        data = copy.deepcopy(raw)
        data["analysis"]["entry_quality"]["fail_closed"] = False
        with pytest.raises(ConfigError):
            load_settings(write(tmp_path, data), env_overrides=False)

    def test_entry_quality_requires_every_asset_class_limit(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        data = copy.deepcopy(raw)
        del data["analysis"]["entry_quality"]["max_single_bar_body_atr"]["stock"]
        with pytest.raises(ConfigError, match="max_single_bar_body_atr"):
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

    def test_ignored_symbol_is_stronger_than_the_affordable_universe(
        self, raw: dict[str, Any], tmp_path: Path
    ) -> None:
        data = copy.deepcopy(raw)
        data["instruments"]["universe_mode"] = "affordable"
        data["instruments"]["symbol_suffix"] = ".i"
        data["instruments"]["symbol_overrides"] = {"XAUUSD": "XAUUSD"}
        data["instruments"]["ignored_symbols"] = ["XAUUSD"]
        settings = load_settings(write(tmp_path, data), env_overrides=False)

        assert settings.instruments.is_ignored("XAUUSD")
        assert not settings.instruments.is_ignored("EURUSD.i")
        assert settings.symbol_allowed_at_equity("XAUUSD", 10_000.0) == (
            False,
            "SYMBOL_IGNORED",
        )


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
        # Every broker asset family except single shares. `stock` came off on
        # this account's own evidence: ENR gapped 2.61R through its stop while
        # its exchange was shut, QIA and ANTO quoted two cents on a EUR 36 price
        # so the spread decided the outcome rather than the setup, and three
        # share positions once held three of four slots with their exchange
        # closed and no way to modify them. Indices stay — same exchange, a
        # fraction of the spread, and a book that quotes nearly around the clock.
        assert settings.instruments.asset_classes == (
            "forex",
            "metal",
            "index",
            "commodity",
            "stock",
            "crypto",
        )
        assert settings.instruments.symbols_only == ()
        assert settings.instruments.symbol_overrides["XAUUSD"] == "XAUUSD"
        assert settings.ai.provider == "local_history"
        assert settings.ai.anthropic_model == ""
        assert not settings.ai.market_scout.enabled
        # Large enough to cover everything the cheap scan lets through on a full
        # broker catalogue, rather than a top-N slice of it.
        assert settings.scanner.deep_candidates >= 200
        # One module plus Claude. Two modules cannot agree in practice: the two
        # heaviest look for opposite market states.
        assert settings.analysis.confluence.minimum_directional_modules == 1
        # Hand-written equity floors remain absent. Gold is deliberately a
        # complete Jarvis opt-out; silver and indices still use exact sizing.
        assert settings.instruments.min_equity_for_symbol == {}
        assert settings.risk.release_slots_when_unmanageable
        assert not settings.trade_management.bank_enabled
        assert settings.symbol_allowed_at_equity("XAUUSD", 100.0) == (
            False,
            "SYMBOL_IGNORED",
        )
        for symbol in ("US30", "BTCUSD"):
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


def test_asset_classes_narrow_the_catalogue_without_changing_judgement(tmp_path: Path) -> None:
    """Choosing which markets to look at is a horizon decision, not a quality one.

    The stop is 1.5 ATR and the target twice that, so in market hours every
    class reaches its target in about the same time. A London share is open 42
    hours a week against FX's 120, so the same setup takes roughly six calendar
    days instead of two — and with two position slots that is half the account's
    capacity held for a week.
    """
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump({"instruments": {"asset_classes": ["forex", "metal"]}}), encoding="utf-8"
    )
    settings = load_settings(overlay=overlay, env_overrides=False)

    assert settings.instruments.asset_classes == ("forex", "metal")


def test_an_unknown_asset_class_is_rejected_at_load(tmp_path: Path) -> None:
    """A typo must not silently scan nothing at all."""
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump({"instruments": {"asset_classes": ["forex", "stonks"]}}), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="stonks"):
        load_settings(overlay=overlay, env_overrides=False)


class TestACycleCanFinish:
    """Five minutes after launch there was no cycle line in the log at all.

    Only `slow MT5 calls` and `symbol held out of deep analysis`. Scanning was
    happening; nothing was ever completed. With shares back in the catalogue an
    uncapped pass is 800+ symbols across seven timeframes on one vCPU, and an
    M1 or M5 entry three minutes old is not an entry any more.

    This is also where "more opportunities" actually comes from, rather than from
    a looser gate: a cycle that finishes produces more usable setups per minute
    than one that stalls halfway, however many symbols it touches.
    """

    def test_the_batch_is_small_enough_to_complete(self) -> None:
        settings = load_settings(
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml", env_overrides=False
        )

        assert settings.scanner.batch_size is not None
        assert settings.scanner.batch_size <= 300

    def test_the_liquid_core_is_seen_every_cycle(self) -> None:
        """Rotation alone would let EURUSD go three cycles without fresh data."""
        settings = load_settings(
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml", env_overrides=False
        )

        assert settings.scanner.priority_every_cycle
        assert "forex" in settings.scanner.priority_asset_classes

    def test_the_whole_catalogue_is_still_reachable(self) -> None:
        """Batching narrows a cycle, not the universe. Nothing is excluded — it
        rotates through, which is the difference between this and a blocklist."""
        settings = load_settings(
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml", env_overrides=False
        )

        assert settings.instruments.symbols_only == ()
        assert settings.scanner.deep_candidates >= 847

    def test_the_priority_recurrence_needs_a_batch_to_mean_anything(self) -> None:
        """Documented in `scanner/universe.py`: with no batch the whole universe
        is inspected anyway, so the recurring tier is a no-op. Pinned so the two
        settings are never split apart by a later edit."""
        settings = load_settings(
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml", env_overrides=False
        )

        assert not (settings.scanner.priority_every_cycle and settings.scanner.batch_size is None)


class TestABuyIsNotAllowedAtTheTopOfTheRange:
    """ "Buy high" — the operator's words, on a chart that made it obvious.

    GBPUSD LONG at 19:01 on 17 August. The last twelve M5 bars ran 1.35559 to
    1.35646 and the entry was 1.35639: ninety-two percent of the way to the top,
    the peak of a spike that fell straight back to 1.35609. UK100 LONG the same
    afternoon, in at 10732.25 and out at 10723.55 for -6.10.

    `directional_extreme_location` had been raised to 0.95 for a reason that
    still holds — a breakout SITS near the top of its range, and 0.88 refused
    every impulse the engine would ever see. But 0.95 is not a gate, it is a
    formality: everything except literally the highest tick passes. The middle
    ground between "refuse every impulse" and "refuse nothing" had not been
    tried.
    """

    def test_the_overlay_refuses_the_top_of_the_range(self) -> None:
        settings = load_settings(
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml", env_overrides=False
        )

        assert settings.analysis.entry_quality.directional_extreme_location <= 0.85

    def test_the_live_gbpusd_entry_would_now_be_refused(self) -> None:
        settings = load_settings(
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml", env_overrides=False
        )
        low, high, entry = 1.35559, 1.35646, 1.35639
        location = (entry - low) / (high - low)

        assert location > settings.analysis.entry_quality.directional_extreme_location

    def test_but_buying_strength_is_still_allowed(self) -> None:
        """A breakout entering at seventy percent of its range is not a chase,
        and refusing it is what 0.88 got wrong in the first place."""
        settings = load_settings(
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml", env_overrides=False
        )

        assert settings.analysis.entry_quality.directional_extreme_location > 0.70


class TestNoBarIsFetchedThatNothingReads:
    """The cycle was spending its time on data no reader ever looks at.

    From a live cycle line: 432 of 843 symbols scanned in 118 seconds with a
    `slow MT5 calls` warning in front of it, and one setup out of 132
    analysable symbols. M5 loses its cache every five minutes and a symbol
    comes round again roughly every seven at 240 per cycle, so 2,000 M5 bars
    went over the wire per symbol per rotation — half a million bars a cycle
    for a consumption of four hundred.

    Four hundred is the deepest window that exists anywhere in this codebase:
    `_reachable_target`, `measure_target_reach`, the playbooks, the review
    payload and the liveliness gap counter all sit on `tail(400)`. Everything
    else is smaller. This pins that relationship, because the failure mode is
    silent — nobody notices a slow cycle the way they notice a wrong number.
    """

    #: The deepest `tail(...)` in the repository. Raise this only alongside the
    #: reader that needs it, and raise the bar counts with it.
    DEEPEST_WINDOW = 400

    def _bars(self):  # type: ignore[no-untyped-def]
        settings = load_settings(
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml", env_overrides=False
        )
        return settings.data.bars

    #: M1 and W1 run on their own documented budgets rather than this rule. M1
    #: is read 30 bars deep by the adviser and 15 by the scalp, and it is the
    #: one series that refetches every single cycle. W1 cannot reach 400 plus
    #: room on most share CFDs at all — COFFEE offered 81 weekly bars — which
    #: is why it has a `min_bars_by_timeframe` floor of 50 and is only ever
    #: background bias.
    OWN_BUDGET = ("M1", "W1")

    def test_every_timeframe_carries_the_deepest_reader_plus_room(self) -> None:
        """Enough for the deepest window with headroom for a weekend gap or a
        half day, on the frames the 400-bar readers actually run on."""
        for timeframe, count in self._bars().items():
            if timeframe in self.OWN_BUDGET:
                continue
            assert count >= self.DEEPEST_WINDOW + 100, timeframe

    #: H4 is exempt from the ceiling below, and the reason is the whole point of
    #: this class getting a second look.
    #:
    #: "Nothing reads more than 400 bars" was true of the ANALYSIS and false of
    #: the system. `DataManager._missing_bars` reads the window too, and it is
    #: not looking for a value — it is deciding whether a gap recurs often
    #: enough to be the instrument's schedule rather than a hole in the feed.
    #: That judgement needs a long window, and shortening H4 to 600 flipped it:
    #: every exchange-traded symbol came back with "30 bars missing (5.0% of the
    #: window, limit 5.0%)", because 30 is exactly where the 5% limit falls at
    #: 600 bars and is 2.0% at 1500.
    #:
    #: It quarantined 45% of the catalogue — 11,430 of 25,373 decisions in six
    #: live hours, against 0.9% before — and it cost nothing to fix, because H4
    #: refetches once every four hours. The cycle time was never there.
    JUDGED_ON_THE_WHOLE_WINDOW = ("H4",)

    def test_and_no_timeframe_carries_far_more_than_that(self) -> None:
        """The regression this exists to catch. A bar fetched and never read is
        pure cycle time, and cycle time is the only thing that decides whether
        a five-minute setup is still there when the analysis reaches it."""
        for timeframe, count in self._bars().items():
            if timeframe in self.JUDGED_ON_THE_WHOLE_WINDOW:
                continue
            assert count <= 2 * self.DEEPEST_WINDOW, timeframe

    def test_the_frames_the_gap_check_judges_keep_a_long_window(self) -> None:
        """The other half, so the saving is never reclaimed from the frame that
        cannot afford it. A short window does not make the gap check stricter in
        any meaningful sense — it makes the same number of missing bars a larger
        share of a smaller denominator."""
        bars = self._bars()

        for timeframe in self.JUDGED_ON_THE_WHOLE_WINDOW:
            assert bars[timeframe] >= 1200, timeframe

    def test_the_floor_the_data_manager_enforces_is_still_cleared(self) -> None:
        settings = load_settings(
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml", env_overrides=False
        )

        for timeframe, count in settings.data.bars.items():
            assert count >= settings.data.minimum_bars_for(timeframe), timeframe


class TestTheDetectorsAreTunedByWhatTheyEarned:
    """Four days of live money per detector, applied to the two knobs that
    decide how many of their setups reach the gate.

    The knobs were set on judgement before there was evidence and never
    revisited once it arrived, which left the three best earners carrying the
    three lowest weights and `session_breakout` behind a floor above its own
    confidence ceiling — able to agree with a setup and never to make one.
    """

    def _confluence(self):  # type: ignore[no-untyped-def]
        settings = load_settings(
            overlay=DEFAULT_CONFIG_PATH.parent / "eightcap.yaml", env_overrides=False
        )
        return settings.analysis.confluence

    def test_the_one_detector_measured_losing_money_no_longer_votes_live(self) -> None:
        """`seasonality` at -1.01 EUR a trade, the only net-negative of the
        nine. Its weight stays so the backtest keeps following it."""
        confluence = self._confluence()

        assert "seasonality" not in confluence.live_enabled_modules
        assert confluence.weights["seasonality"] > 0
        assert confluence.effective_weights(TradingMode.MICRO_LIVE)["seasonality"] == 0.0

    def test_the_proven_detectors_have_a_lower_bar_to_stand_alone(self) -> None:
        """This is where extra setups come from, and it is aimed rather than
        global: dropping the floor for everyone took setups from 71 to 153 in
        one step, which is how 20 August happened."""
        confluence = self._confluence()

        assert confluence.lone_floor_for("ema_pullback_resume") == 0.55
        assert confluence.lone_floor_for("impulse_break") == 0.55
        # Untouched detectors keep the global floor, so this cannot leak.
        assert confluence.lone_floor_for("drift_continuation") == 0.65
        assert confluence.lone_floor_for("m1_micro_breakout") == 0.65

    def test_the_hk50_disaster_that_created_this_floor_is_still_refused(self) -> None:
        """`impulse_break` alone at exactly 0.45 for -0.56R. Loosening its
        floor to 0.55 must not buy that trade back."""
        assert self._confluence().lone_floor_for("impulse_break") > 0.45

    def test_the_detector_that_contradicts_itself_gets_half_a_step(self) -> None:
        """`fast_ema_cross` is +1.30 EUR a trade live and the worst module in
        the offline table, and it alone accounts for 473 of the refusals a full
        step would release. Two sources disagreeing plus the largest volume
        effect is where you move half way and measure."""
        confluence = self._confluence()

        assert 0.55 < confluence.lone_floor_for("fast_ema_cross") < 0.65

    def test_session_breakout_can_now_reach_a_setup_of_its_own(self) -> None:
        """It sat at 1.00 against a confidence ceiling of 0.80 — deliberately
        unable to decide anything until it had a number. It has one: +1.20 EUR
        a trade. A gap and not a door, so still under its ceiling."""
        floor = self._confluence().lone_floor_for("session_breakout")

        assert 0.55 < floor < 0.80

    def test_the_best_earners_no_longer_carry_the_lowest_weights(self) -> None:
        """Weight only moves setups where two or more detectors agree — for a
        lone module it cancels out of numerator and denominator alike — so this
        is the other half of the same question."""
        weights = self._confluence().weights

        assert weights["ema_pullback_resume"] > weights["drift_continuation"]
        assert weights["fast_ema_cross"] > weights["seasonality"]
        assert weights["session_breakout"] > weights["fast_ema_cross"]

    def test_every_live_detector_still_carries_weight(self) -> None:
        """A module on the allowlist with no weight is voting on nothing, which
        is the exact defect the effective-weight fix exists to expose."""
        confluence = self._confluence()
        live = confluence.effective_weights(TradingMode.MICRO_LIVE)

        for module in confluence.live_enabled_modules:
            assert live.get(module, 0.0) > 0.0, module

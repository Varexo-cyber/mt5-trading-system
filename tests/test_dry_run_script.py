"""The dry-run script has to survive contact with the real APIs.

It was shipped twice and crashed twice on the owner's machine before fetching a
single bar, and both times for the same kind of reason: a call written from
memory instead of from the code it calls.

    MT5Connector(credentials, ...)      -> reads credentials.terminal_path
                                           which does not exist. It takes
                                           settings.mt5 FIRST.
    connector.disconnect()              -> the method is shutdown()
    load_settings(env_overrides=True)   -> no overlay, so live_enabled_modules
                                           is empty and it exits saying there
                                           is nothing to run, on an account
                                           that is correctly configured

None of that is catchable by running the script here: MT5 does not exist in
this environment, so the first two lines that touch it are also the first lines
that would fail. These tests check the call SHAPES against the real objects
instead, which is the part that was actually wrong.
"""

from __future__ import annotations

import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "scripts" / "dry_run_sections.py").read_text()


class TestItCallsTheRealApi:
    def test_the_connector_gets_the_settings_block_first(self) -> None:
        """`MT5Connector(config, credentials, ...)`. Passing only credentials
        made it read `credentials.terminal_path`, and the run died there."""
        from core.mt5_connector import MT5Connector

        parameters = list(inspect.signature(MT5Connector.__init__).parameters)

        assert parameters[1] == "config"
        assert parameters[2] == "credentials"
        assert "MT5Connector(\n        settings.mt5,\n        credentials," in SOURCE

    def test_it_tears_the_connection_down_by_its_real_name(self) -> None:
        from core.mt5_connector import MT5Connector

        assert hasattr(MT5Connector, "shutdown")
        assert not hasattr(MT5Connector, "disconnect")
        assert "connector.shutdown()" in SOURCE
        assert "connector.disconnect()" not in SOURCE

    def test_it_reads_the_instrument_spec_by_its_real_name(self) -> None:
        from core.mt5_connector import MT5Connector

        assert hasattr(MT5Connector, "spec")
        assert "connector.spec(symbol)" in SOURCE

    def test_the_sizer_result_fields_exist(self) -> None:
        """`actual_risk_money` and `actual_risk_pct` are what the report
        prints. A typo here would produce a run that works and reports
        nothing."""
        from risk.position_sizer import SizingResult

        fields = set(SizingResult.__dataclass_fields__)

        assert {"actual_risk_money", "actual_risk_pct", "volume", "decision"} <= fields

    def test_the_risk_decision_is_read_by_its_real_field(self) -> None:
        from risk.reasons import RiskDecision

        assert "approved" in RiskDecision.__dataclass_fields__
        assert "sized.decision.approved" in SOURCE


class TestItLoadsTheAccountItIsMeantToMeasure:
    def test_it_loads_the_overlay(self) -> None:
        """Permission to trade real money lives in the Eightcap overlay, not in
        the base config. Loading without it gives an empty
        `live_enabled_modules` and the script exits saying there is nothing to
        run -- on an account that is perfectly well configured."""
        assert 'overlay=ROOT / "config" / "eightcap.yaml"' in SOURCE

    def test_the_overlay_actually_has_live_modules(self) -> None:
        from config.loader import load_settings

        settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)

        assert settings.analysis.confluence.live_enabled_modules

    def test_it_checks_for_live_modules_before_connecting(self) -> None:
        """Failing after `connect()` produced two stacked tracebacks, with the
        teardown error masking the real one. Fail before touching MT5."""
        connect = SOURCE.index("connector.connect()")
        check = SOURCE.index("no live modules in this configuration")

        assert check < connect


class TestItMeasuresTheUniverseThatTrades:
    def test_it_uses_the_scanner_own_classifier(self) -> None:
        """`active_whitelist` is four names; the live scan walks the broker's
        whole catalogue filtered by asset class. The first run of this reported
        "4 symbols" on an account that scans a couple of hundred, so it was
        measuring a universe the account does not trade.

        And the filter is the scanner's own `_path_class`, not a substring
        match on the folder name -- an approximation would quietly disagree
        with the live filter, which is the same defect one level down."""
        assert "UniverseScanner._path_class" in SOURCE
        assert "settings.active_whitelist" not in SOURCE

    def test_the_classifier_is_reachable_as_used(self) -> None:
        from scanner.universe import UniverseScanner

        assert UniverseScanner._path_class("forex\\majors").value == "forex"

    def test_history_is_fetched_per_timeframe(self) -> None:
        """Warmup is counted in BARS. One 27-day window gives M15 about 2,600
        bars and H4 about 160 -- under the guard -- so every symbol was skipped
        "for want of history" and the run reported zero decisions."""
        assert "def _fetch_from(tf: Timeframe)" in SOURCE
        assert "(WARMUP + 20) * tf.duration" in SOURCE

    def test_only_the_clocks_in_use_must_be_deep_enough(self) -> None:
        """Requiring the warmup of every fetched timeframe threw symbols away
        over a frame no pass was going to read."""
        assert "for tf in used if tf in frames" in SOURCE


class TestTheSweepIsHonest:
    def test_it_resolves_on_a_finer_timeframe_than_it_decides_on(self) -> None:
        """An M30 trade resolved on M30 bars cannot tell which barrier the bar
        touched first, and assuming the good one is how a backtest lies."""
        assert "clock.duration <= finest.duration" in SOURCE

    def test_each_section_is_swept_on_its_own(self) -> None:
        """Two sections on one clock merge into a single confluence idea, and
        the result would then say nothing about either."""
        assert "m.name == name" in SOURCE

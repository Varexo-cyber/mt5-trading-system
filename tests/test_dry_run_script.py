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
import re
from datetime import UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "scripts" / "dry_run_sections.py").read_text()


def cmd_argv(launcher: str, **values: str) -> list[str]:
    """The argv a .cmd file actually hands the script, given its variables.

    Two cmd behaviours matter and both have bitten this script. Arguments are
    split on whitespace AND on commas, and an EMPTY variable expands to nothing
    at all rather than to an empty argument -- so `%SCOPE%` unset removes the
    word entirely instead of passing "".

    Every launcher test goes through here, so adding a variable to a .cmd file
    without teaching the tests about it fails loudly instead of leaving a
    `%SCOPE%` literal to be silently accepted as a filename.
    """
    line = next(ln for ln in launcher.splitlines() if "scripts.dry_run_sections" in ln)
    argv: list[str] = []
    for token in line.split("scripts.dry_run_sections", 1)[1].split():
        if token.startswith("%") and token.endswith("%"):
            assert token in values, f"the launcher uses {token} and this test does not set it"
            token = values[token]
        argv.extend(piece for piece in token.split(",") if piece)
    return argv


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


class TestTheLauncherSurvivesCmd:
    """`dryrun.cmd 7 M15,M30 --limit 40` died on the owner's machine because
    CMD SPLITS ARGUMENTS ON COMMAS. The shell handed the script "M15" and
    "M30" as separate words, `--limit` collected "M30", and argparse rejected
    the rest. No validation inside the script could have caught it: the damage
    is done before it is called.

    Two fixes, and both are needed. The launcher no longer takes timeframes at
    all -- only two numbers -- and the parser accepts the split form anyway,
    because someone will type a comma sooner or later.
    """

    LAUNCHER = (ROOT / "dryrun.cmd").read_text()

    def test_the_launcher_takes_no_comma_arguments(self) -> None:
        set_lines = [line for line in self.LAUNCHER.splitlines() if line.strip().startswith("set ")]
        for line in set_lines:
            assert "," not in line, f"a comma in an argument will be split by cmd: {line}"

    def test_the_launcher_writes_the_timeframes_itself(self) -> None:
        """So the list never passes through the shell."""
        assert "--sweep M5 M15 M30 H1 H4" in self.LAUNCHER

    def test_the_launchers_own_command_line_parses(self) -> None:
        """THE TEST THAT SHOULD HAVE EXISTED FIRST. It reads the flags out of
        `dryrun.cmd` and feeds them to the real parser.

        `--limit` was added by a string edit that silently matched nothing, so
        the flag was missing from the parser while the launcher was already
        sending it and the code reading `args.limit` was already there. Two
        runs died on that, and nothing was checking that the two files agree.
        """
        from scripts.dry_run_sections import build_parser

        argv = cmd_argv(self.LAUNCHER, **{"%DAYS%": "7", "%LIMIT%": "40", "%SCOPE%": ""})

        parsed = build_parser().parse_args(argv)

        assert parsed.days == 7
        assert parsed.limit == 40
        assert parsed.core is False
        assert parsed.sweep == ["M5", "M15", "M30", "H1", "H4"]

    def test_every_flag_the_code_reads_is_a_flag_the_parser_defines(self) -> None:
        """`args.limit` existed in the body before `--limit` existed in the
        parser. An AttributeError at runtime, on the owner's machine, after
        the history fetch."""
        from scripts.dry_run_sections import build_parser

        defined = {action.dest for action in build_parser()._actions if action.dest != "help"}
        used = set(re.findall(r"args\.([a-z_0-9]+)", SOURCE))

        assert used <= defined, f"read but never defined: {sorted(used - defined)}"

    def test_the_parser_accepts_the_split_form(self) -> None:
        assert 'nargs="*"' in SOURCE

    def test_the_parser_also_accepts_a_comma_list(self) -> None:
        """Both arrive in practice, so both have to work."""
        assert 'str(chunk).split(",")' in SOURCE


class TestTheSweepIsHonest:
    def test_it_resolves_on_a_finer_timeframe_than_it_decides_on(self) -> None:
        """An M30 trade resolved on M30 bars cannot tell which barrier the bar
        touched first, and assuming the good one is how a backtest lies."""
        assert "clock.duration <= finest.duration" in SOURCE

    def test_each_section_is_swept_on_its_own(self) -> None:
        """Two sections on one clock merge into a single confluence idea, and
        the result would then say nothing about either."""
        assert "m.name == name" in SOURCE


class TestItMeasuresWhatTheAccountWouldActuallyDo:
    """The 30 August run reported -1,120 EUR on a 215 EUR account, and that
    number was not the live configuration. Three things separated them, and
    every one of them was in this script rather than in the strategy."""

    def test_it_refuses_a_second_position_on_a_busy_symbol(self) -> None:
        """`Reason.POSITION_ALREADY_OPEN` live; nothing at all here. Without it
        the loop takes a fresh trade on EVERY bar the setup stays valid, and
        the over-count is BIASED: a retest that works leaves the level in a bar
        and yields one entry, a retest that fails sits on it and yields five.
        Duplicates are drawn from the losers."""
        assert "busy_until" in SOURCE
        assert "a position is open on this symbol; live would refuse" in SOURCE

    def test_an_unresolved_trade_keeps_holding_the_symbol(self) -> None:
        """Freeing the symbol on a trade that never resolved would let the same
        signal re-enter while the position is still open."""
        assert "busy_until = exit_at if exit_at is not None else end + clock.duration" in SOURCE

    def test_the_resolver_reports_when_the_trade_left(self) -> None:
        """The slot cap is a rule about how many trades are open AT ONCE, which
        cannot be answered from entry times.

        Asserted by CALLING it. This test first checked for the literal
        `return -1.0, stamp`, and the string went away the moment the resolver
        grew its managed column -- a red test over a rename, on a behaviour
        that had not changed. The same brittleness that made
        `test_the_override_actually_reaches_the_target` pass over a dead
        config, pointing the other way."""
        from dataclasses import dataclass
        from datetime import UTC, datetime, timedelta

        import pandas as pd

        from core.types import Direction
        from scripts.dry_run_sections import _resolve

        @dataclass
        class Idea:
            direction: Direction
            entry: float
            stop_loss: float
            take_profit: float

        start = datetime(2026, 8, 24, 9, 0, tzinfo=UTC)
        index = pd.DatetimeIndex([start, start + timedelta(minutes=1)])
        frame = pd.DataFrame(
            {
                "open": [100.0, 99.5],
                "high": [100.1, 100.0],
                "low": [99.9, 98.5],
                "close": [100.0, 99.0],
            },
            index=index,
        )

        r, exit_at, _managed = _resolve(
            frame, start, Idea(Direction.LONG, 100.0, 99.0, 101.0), horizon_bars=2
        )

        assert r == -1.0
        assert exit_at == index[1]

    def test_the_slot_cap_is_applied_in_time_order(self) -> None:
        from datetime import datetime, timedelta

        from scripts.dry_run_sections import Decision, _under_the_slot_cap

        base = datetime(2026, 8, 24, tzinfo=UTC)
        # Four signals inside one hour; the first two hold their slots for a day.
        trades = [
            Decision(
                when=base + timedelta(minutes=15 * i),
                symbol=f"S{i}",
                module="m",
                outcome="TRADE",
                exit_at=base + timedelta(days=1),
            )
            for i in range(4)
        ]

        taken = _under_the_slot_cap(trades, slots=2)

        assert [d.symbol for d in taken] == ["S0", "S1"]

    def test_a_freed_slot_is_reused(self) -> None:
        """The cap must not be a cap on trades per window."""
        from datetime import datetime, timedelta

        from scripts.dry_run_sections import Decision, _under_the_slot_cap

        base = datetime(2026, 8, 24, tzinfo=UTC)
        trades = [
            Decision(base, "A", "m", "TRADE", exit_at=base + timedelta(minutes=30)),
            Decision(
                base + timedelta(hours=1), "B", "m", "TRADE", exit_at=base + timedelta(days=1)
            ),
        ]

        assert len(_under_the_slot_cap(trades, slots=1)) == 2

    def test_the_live_pair_gets_its_own_block(self) -> None:
        """The two shipped rows were on the same screen as eight combinations
        that will never run together, unlabelled, and the total of all ten was
        printed as the answer. A report that must be disentangled to be read
        will be misread."""
        assert "def _live_config_report" in SOURCE
        assert "_live_config_report(results, settings, equity, args.days)" in SOURCE

    def test_the_live_block_uses_each_sections_configured_clock(self) -> None:
        """Not the sweep's clocks. Picking the best row out of the sweep and
        calling it the live result is the same lie one level up."""
        assert 'clock = getattr(section, "timeframe", None)' in SOURCE

    def test_the_live_block_applies_the_accounts_own_cap(self) -> None:
        assert "settings.effective_max_positions(equity)" in SOURCE
        assert "_under_the_slot_cap(everything, slots)" in SOURCE


class TestTheLiveOnlyLauncher:
    """A month over the whole catalogue is five times cheaper without the
    sweep, and the sweep answers a question the owner is no longer asking."""

    LAUNCHER = (ROOT / "dryrun-live.cmd").read_text()

    def test_its_command_line_parses(self) -> None:
        from scripts.dry_run_sections import build_parser

        # Both forms the launcher can emit: the default core run, and the
        # "all" run where %SCOPE% expands to nothing.
        core = build_parser().parse_args(
            cmd_argv(self.LAUNCHER, **{"%DAYS%": "30", "%SCOPE%": "--core"})
        )

        assert core.days == 30
        assert core.live_only is True
        assert core.core is True
        assert core.sweep == []

        every = build_parser().parse_args(cmd_argv(self.LAUNCHER, **{"%DAYS%": "7", "%SCOPE%": ""}))

        assert every.days == 7
        assert every.live_only is True
        assert every.core is False

    def test_it_takes_no_comma_arguments(self) -> None:
        for line in self.LAUNCHER.splitlines():
            if line.strip().startswith("set "):
                assert "," not in line, line

    def test_live_only_beats_a_sweep_that_is_also_present(self) -> None:
        """Otherwise the flag is advisory and the run costs what it always did."""
        assert "if args.live_only:\n            args.sweep = []" in SOURCE


class TestTheCoreUniverse:
    """232 symbols x 5 clocks x M1 history is a run that does not get run.

    And the cut is not arbitrary. Both sections were chosen, tuned and
    holdout-tested on eleven FX majors and gold; every other market in the
    catalogue is an extrapolation, so most of that time was spent on markets
    that cannot confirm or refute the finding.
    """

    def test_it_is_the_set_the_research_used(self) -> None:
        from scripts.dry_run_sections import CORE_UNIVERSE

        majors = {
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCHF",
            "USDCAD",
            "AUDUSD",
            "NZDUSD",
            "EURGBP",
            "EURJPY",
            "GBPJPY",
            "EURCHF",
        }

        assert majors <= set(CORE_UNIVERSE)
        assert "XAUUSD" in CORE_UNIVERSE, "gold is shipped with its own stop and must be measured"
        assert len(CORE_UNIVERSE) <= 20, "the point of this list is that it is short"

    def test_it_matches_a_broker_that_decorates_its_symbol_names(self) -> None:
        """Suffixes for account type, a dot, a trailing m for micro. Matching
        the literal string would return an EMPTY list on such a broker, and an
        empty list reads as "no setups" rather than "no symbols" -- which is
        this run's signature failure in another costume."""
        from types import SimpleNamespace

        from scripts.dry_run_sections import _core_universe

        catalogue = [
            SimpleNamespace(name=name, path=f"forex\\{name}")
            for name in ("EURUSD.r", "GBPUSD.r", "XAUUSD.r", "US30.r", "EURUSDX", "NOTAPAIR")
        ]
        connector = SimpleNamespace(symbols=lambda: catalogue)
        settings = SimpleNamespace(instruments=SimpleNamespace(is_ignored=lambda _n: False))

        found = _core_universe(connector, settings)

        assert "EURUSD.r" in found
        assert "XAUUSD.r" in found
        assert "NOTAPAIR" not in found

    def test_the_exact_base_beats_a_longer_lookalike(self) -> None:
        """EURUSD must not match EURUSDX when EURUSD itself is on the books."""
        from types import SimpleNamespace

        from scripts.dry_run_sections import _core_universe

        catalogue = [
            SimpleNamespace(name=name, path="forex\\majors")
            for name in ("EURUSDX", "EURUSD", "EURUSD.pro")
        ]
        connector = SimpleNamespace(symbols=lambda: catalogue)
        settings = SimpleNamespace(instruments=SimpleNamespace(is_ignored=lambda _n: False))

        assert "EURUSD" in _core_universe(connector, settings)

    def test_it_honours_the_ignore_list(self) -> None:
        from types import SimpleNamespace

        from scripts.dry_run_sections import _core_universe

        catalogue = [SimpleNamespace(name="EURUSD", path="forex\\majors")]
        connector = SimpleNamespace(symbols=lambda: catalogue)
        settings = SimpleNamespace(instruments=SimpleNamespace(is_ignored=lambda n: n == "EURUSD"))

        assert _core_universe(connector, settings) == []

    def test_both_launchers_offer_it_and_both_command_lines_parse(self) -> None:
        """The flag has to survive cmd, which is where every previous version
        of this died."""
        from scripts.dry_run_sections import build_parser

        fast = build_parser().parse_args(
            cmd_argv(
                (ROOT / "dryrun-live.cmd").read_text(),
                **{"%DAYS%": "7", "%SCOPE%": "--core"},
            )
        )

        assert fast.core is True and fast.live_only is True and fast.days == 7

        sweep = build_parser().parse_args(
            cmd_argv(
                (ROOT / "dryrun.cmd").read_text(),
                **{"%DAYS%": "7", "%LIMIT%": "0", "%SCOPE%": "--core"},
            )
        )

        assert sweep.core is True and sweep.sweep == ["M5", "M15", "M30", "H1", "H4"]

    def test_an_empty_scope_variable_still_parses(self) -> None:
        """cmd expands an unset variable to nothing, so the flagless form is
        the OTHER command line each launcher can emit."""
        from scripts.dry_run_sections import build_parser

        parsed = build_parser().parse_args(
            ["--days", "7", "--limit", "0", "--sweep", "M5", "--csv", "x.csv"]
        )

        assert parsed.core is False


class TestTheReportRunsEndToEnd:
    """`_report` was split in half by an edit that inserted a new function into
    the middle of its body. The tail reattached to the wrong def, referenced
    `trades` and `closed` from a scope that no longer had them, and the file
    still imported -- because none of the report functions were ever CALLED by
    a test. Every assertion about them was a substring search over the source.

    So these call them, with a fabricated set of decisions, and assert on what
    lands on stdout.
    """

    def _decisions(self):
        from datetime import datetime, timedelta

        from scripts.dry_run_sections import Decision

        base = datetime(2026, 8, 24, 9, 0, tzinfo=UTC)
        rows = []
        # A market that trades.
        for i in range(3):
            rows.append(
                Decision(
                    base + timedelta(hours=i),
                    "NDX100",
                    "order_block",
                    "TRADE",
                    direction="LONG",
                    risk_money=8.0,
                    risk_pct=4.0,
                    result_r=1.0 if i else -1.0,
                    pnl_money=8.0 if i else -8.0,
                    exit_at=base + timedelta(hours=i, minutes=30),
                    pass_key=("order_block", "M30"),
                    managed_r=0.1 if i else -1.0,
                    managed_money=0.8 if i else -8.0,
                )
            )
        # A market whose detector fired and was refused every time.
        rows += [
            Decision(
                base,
                "EURUSD.i",
                "impulse_retest",
                "REFUSED_CONFLUENCE",
                note="score 38.8 below threshold",
            )
            for _ in range(4)
        ]
        # A market where nothing ever fired.
        rows += [
            Decision(
                base, "GBPUSD.i", "-", "REFUSED_CONFLUENCE", note="no weighted directional evidence"
            )
            for _ in range(4)
        ]
        return rows

    def test_the_whole_report_prints_without_raising(self, capsys) -> None:
        from scripts.dry_run_sections import _report

        _report(self._decisions(), equity=215.34, days=7, skipped=0)
        out = capsys.readouterr().out

        assert "TRADES" in out
        assert "BY DAY" in out

    def test_it_separates_a_silent_detector_from_a_refused_one(self, capsys) -> None:
        """The distinction the 30 August run could not make. Eleven FX majors
        took zero trades and the report could not say whether no setup existed
        or whether a gate ate all of them."""
        from scripts.dry_run_sections import _report

        _report(self._decisions(), equity=215.34, days=7, skipped=0)
        out = capsys.readouterr().out

        assert "PER MARKET" in out
        assert "ALL refused by a gate" in out
        assert "never fired at all" in out
        assert "2 of 3 markets took no trade at all" in out

    def test_it_totals_what_the_refusal_actually_said(self, capsys) -> None:
        """`REFUSED_CONFLUENCE 98.4%` names a bucket. The engine writes a
        sentence, this script already stored it, and the report discarded it."""
        from scripts.dry_run_sections import _report

        _report(self._decisions(), equity=215.34, days=7, skipped=0)
        out = capsys.readouterr().out

        assert "score 38.8 below threshold" in out
        assert "no weighted directional evidence" in out

    def test_the_break_even_counters_agree_with_the_total(self, capsys) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from scripts.dry_run_sections import _break_even_verdict

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        trades = [d for d in self._decisions() if d.outcome == "TRADE"]

        _break_even_verdict(trades, settings)
        out = capsys.readouterr().out

        # Two winners cut from +1.00R to +0.10R.
        assert "cut short              2" in out or "cut short           2" in out
        assert "-1.80 R" in out

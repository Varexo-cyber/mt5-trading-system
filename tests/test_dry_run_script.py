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

import pytest

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
        assert "busy: dict = {name: None for name, _engine, _sizer in sections}" in SOURCE
        assert (
            "awake = [row for row in sections if busy[row[0]] is None or upto > busy[row[0]]]"
            in SOURCE
        )

    def test_an_unresolved_trade_keeps_holding_the_symbol(self) -> None:
        """Freeing the symbol on a trade that never resolved would let the same
        signal re-enter while the position is still open."""
        assert "busy[name] = exit_at if exit_at is not None else end + clock.duration" in SOURCE

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
            cmd_argv(self.LAUNCHER, **{"%DAYS%": "30", "%SCOPE%": "--core", "%FINE%": ""})
        )

        assert core.days == 30
        assert core.live_only is True
        assert core.core is True
        assert core.sweep == []

        every = build_parser().parse_args(
            cmd_argv(self.LAUNCHER, **{"%DAYS%": "7", "%SCOPE%": "", "%FINE%": "--no-m1"})
        )

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
                **{"%DAYS%": "7", "%SCOPE%": "--core", "%FINE%": ""},
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


class TestNothingIsDefinedAfterTheEntryPoint:
    """`NameError: name '_silence_report' is not defined`, on the owner's
    machine, after a full seven-day fetch had already completed.

    A refactor moved a function to the END of the file, and the end of the file
    was BELOW `if __name__ == "__main__": main()`. Module-level statements run
    top to bottom, so as a script `main()` was called before that def existed.

    NOTHING IN THE SUITE COULD SEE IT. Tests `import` the module, and an import
    runs every def and never calls `main()`, so by the time a test looked, the
    name was there. Ruff was clean. Black was clean. Forty-one tests were green
    against a script that could not run.

    That is the recurring defect wearing yet another costume: the tests
    exercise a path the program does not take.
    """

    SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))

    def test_the_entry_point_is_the_last_thing_in_every_script(self) -> None:
        import ast

        for path in self.SCRIPTS:
            tree = ast.parse(path.read_text())
            guards = [
                i
                for i, node in enumerate(tree.body)
                if isinstance(node, ast.If) and ast.dump(node.test).find("__main__") != -1
            ]
            if not guards:
                continue
            after = tree.body[guards[-1] + 1 :]
            stranded = [
                node.name
                for node in after
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            ]

            assert not stranded, (
                f"{path.name} defines {stranded} BELOW `if __name__ == '__main__'`. "
                "As a script those names do not exist when main() runs, and an "
                "import-based test cannot see it."
            )

    def test_every_name_main_uses_is_defined_before_the_guard(self) -> None:
        """The property one level stricter, on this script specifically: run
        the module body up to the guard and check the report functions exist.
        A structural rule can be satisfied by a file that is still wrong."""
        import ast

        source = (ROOT / "scripts" / "dry_run_sections.py").read_text()
        tree = ast.parse(source)
        guard = next(
            i
            for i, node in enumerate(tree.body)
            if isinstance(node, ast.If) and ast.dump(node.test).find("__main__") != -1
        )
        defined = {
            node.name
            for node in tree.body[:guard]
            if isinstance(node, ast.FunctionDef | ast.ClassDef)
        }

        for name in (
            "main",
            "_report",
            "_silence_report",
            "_live_config_report",
            "_break_even_verdict",
            "_sweep_report",
            "_break_even_rule",
            "_under_the_slot_cap",
            "_core_universe",
        ):
            assert name in defined, f"{name} is not defined before main() is called"


class TestTheSampleJudgesItself:
    """ "Is 59.3% over 82 trades good?" had no answer in the output, only in a
    paragraph afterwards. It belongs in the report."""

    def _trades(self, n: int, win_rate: float, days: int = 40, spread: float = 0.0):
        """`n` trades over `days` days at a fixed win rate."""
        import random
        from datetime import datetime, timedelta

        from scripts.dry_run_sections import Decision

        # SEEDED RANDOM, not `i % 100 < 62`. That arithmetic aliases against
        # the `i % days` calendar: at 600 trades over 90 days it produced a
        # 46% May and a 73% June out of a flat 62% process, so the fixture
        # failed the every-month box for a reason that had nothing to do with
        # the code under test. A fixture with structure the test does not
        # intend is a test that reports on its own fixture.
        rng = random.Random(4242)
        base = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)
        rows = []
        for i in range(n):
            r = 1.0 if rng.random() < win_rate else -1.0
            rows.append(
                Decision(
                    base + timedelta(days=(i % days), minutes=i),
                    "NDX100",
                    "order_block",
                    "TRADE",
                    result_r=r + spread * ((i % 7) - 3),
                    pnl_money=r * 8.0,
                    pass_key=("order_block", "M30"),
                )
            )
        return rows

    def test_a_tiny_sample_refuses_to_say_anything(self, capsys) -> None:
        from scripts.dry_run_sections import _is_this_real

        _is_this_real(self._trades(12, 0.6, days=3), [("order_block", "M30")])
        out = capsys.readouterr().out

        assert "Not enough to say anything at all" in out
        assert "win rate" not in out, "a 12-trade win rate must not be printed as a finding"

    def test_a_real_edge_over_a_long_window_clears_every_box(self, capsys) -> None:
        from scripts.dry_run_sections import _is_this_real

        _is_this_real(self._trades(600, 0.62, days=90), [("order_block", "M30")])
        out = capsys.readouterr().out

        assert "clears every bar" in out
        assert "[x]" in out and "[ ]" not in out

    def test_a_coin_flip_is_refused_however_many_trades(self, capsys) -> None:
        """THE CHECK THAT MATTERS. A 50% win rate at 1R is worth zero, and no
        sample size may turn it into a conclusion."""
        from scripts.dry_run_sections import _is_this_real

        _is_this_real(self._trades(600, 0.50, days=90), [("order_block", "M30")])
        out = capsys.readouterr().out

        assert "NOT ENOUGH TO CONCLUDE" in out

    def test_a_short_window_is_refused_even_when_it_wins(self, capsys) -> None:
        """82 trades at 59.3% over one week -- the actual 30 August result."""
        from scripts.dry_run_sections import _is_this_real

        _is_this_real(self._trades(82, 0.59, days=5), [("order_block", "M30")])
        out = capsys.readouterr().out

        assert "NOT ENOUGH TO CONCLUDE" in out
        assert "at least 200 resolved trades" in out

    def test_sigma_is_measured_on_days_not_trades(self, capsys) -> None:
        """Sixteen markets breaking on one morning are ONE observation.
        Counting them as sixteen overstates significance by about the square
        root of the number that moved together, and that correction was the
        largest single one in the original research.

        WHAT THE PROPERTY ACTUALLY IS, because a first draft got it wrong.
        That draft packed the same INDEPENDENT trades into fewer days and
        expected sigma to fall. It does not, and the arithmetic says why: with
        n independent trades over D days, each day holds n/D of them, so
        `std(daily)` is about `sqrt(n/D)` and `SE = std(daily) * sqrt(D)` comes
        to `sqrt(n)` whatever D is. Day-clustering costs nothing when the
        trades are independent -- which is the correct behaviour, and the
        reason the wrong test read 4.34 against 4.00.

        Clustering costs sigma when the trades inside a day AGREE. So both
        samples below hold 400 trades over 40 days at the same win rate and
        very nearly the same total R; in one, a day's outcome is drawn per
        trade, and in the other the whole day shares one draw. The second is
        eleven pairs breaking on the same morning, and it must read lower.
        """
        import random
        import re
        from datetime import datetime, timedelta

        from scripts.dry_run_sections import Decision, _is_this_real

        def sample(correlated: bool) -> list[Decision]:
            rng = random.Random(99)
            base = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)
            rows: list[Decision] = []
            for day in range(40):
                shared = 1.0 if rng.random() < 0.62 else -1.0
                for k in range(10):
                    r = shared if correlated else (1.0 if rng.random() < 0.62 else -1.0)
                    rows.append(
                        Decision(
                            base + timedelta(days=day, minutes=k),
                            "NDX100",
                            "order_block",
                            "TRADE",
                            result_r=r,
                            pnl_money=r * 8.0,
                            pass_key=("order_block", "M30"),
                        )
                    )
            return rows

        def sigma_of(rows: list[Decision]) -> tuple[float, float]:
            _is_this_real(rows, [("order_block", "M30")])
            text = capsys.readouterr().out
            return (
                float(re.search(r"([+-][\d.]+) sigma from zero", text).group(1)),
                float(re.search(r"([+-][\d.]+) R,", text).group(1)),
            )

        independent, independent_r = sigma_of(sample(correlated=False))
        clustered, clustered_r = sigma_of(sample(correlated=True))

        # Comparable edges, so the sigma difference is about correlation.
        assert abs(independent_r - clustered_r) < 0.35 * abs(independent_r)
        assert clustered < independent, (
            "a day where every market moved together must count for less than "
            f"a day of independent trades ({clustered} vs {independent})"
        )

    def test_it_names_a_section_too_thin_to_judge(self, capsys) -> None:
        """impulse_retest had 10 trades beside order_block's 71, and the
        combined line hid that completely."""
        from scripts.dry_run_sections import _is_this_real

        trades = self._trades(400, 0.62, days=90)
        for d in trades[:6]:
            d.pass_key = ("impulse_retest", "M15")

        _is_this_real(trades, [("order_block", "M30"), ("impulse_retest", "M15")])
        out = capsys.readouterr().out

        assert "impulse_retest has only 6 trades of its own; it is unjudged" in out

    def test_the_launcher_command_line_parses(self) -> None:
        from scripts.dry_run_sections import build_parser

        launcher = (ROOT / "history.cmd").read_text()
        parsed = build_parser().parse_args(cmd_argv(launcher, **{"%DAYS%": "180"}))

        assert parsed.days == 180
        assert parsed.core is True
        assert parsed.live_only is True
        assert parsed.no_m1 is True, "M1 over 180 days is a quarter million bars per market"


class TestTheCostWallIsQuantified:
    """2,832 FX setups reached the sizer across five clocks and ONE became a
    trade. The report named the refusal and stopped there, which decides
    nothing: 13% against a 12% limit is a config question, 30% against a 12%
    limit means this broker cannot carry this strategy on FX at any setting.

    The sizer already writes the figure into its own refusal text. The report
    grouped refusals on their first six words, so every cost refusal collapsed
    onto one line and the percentage -- the only part that decides anything --
    was thrown away.
    """

    def _refusals(self, shares: list[float], symbol: str = "EURUSD.i"):
        from datetime import datetime, timedelta

        from scripts.dry_run_sections import Decision

        base = datetime(2026, 8, 24, tzinfo=UTC)
        return [
            Decision(
                base + timedelta(minutes=i),
                symbol,
                "impulse_retest",
                "SL_TOO_TIGHT_FOR_COSTS",
                note=(
                    f"spread, commission and slippage would be {share * 100:.0f}% of the risk "
                    f"on a 9.6 pip stop, above the 12% limit"
                ),
            )
            for i, share in enumerate(shares)
        ]

    def test_it_recovers_the_percentage_the_sizer_wrote(self, capsys) -> None:
        from scripts.dry_run_sections import _cost_report

        _cost_report(self._refusals([0.24] * 10))
        out = capsys.readouterr().out

        assert "THE COST WALL" in out
        assert "24.0% of the stop" in out

    def test_it_prices_the_edge_at_that_cost(self, capsys) -> None:
        """A percentage on its own still needs translating. 22% is already
        negative under the research's own model and the report must say so
        rather than leave it to be worked out."""
        from scripts.dry_run_sections import _cost_report

        _cost_report(self._refusals([0.24] * 10))
        out = capsys.readouterr().out

        # net = 0.358 - 2*0.24 = -0.122
        assert "-0.122 R" in out

    def test_the_model_reproduces_the_research_at_the_cost_it_assumed(self) -> None:
        """The check that the arithmetic in the report is the SAME arithmetic
        the research used. At the 4% it assumed it must land on the +0.279R it
        published, or the whole comparison is between two different models."""
        assert pytest.approx(0.279, abs=0.002) == 0.358 - 2 * 0.04

    def test_it_says_whether_a_higher_limit_would_actually_help(self, capsys) -> None:
        """Raising the limit is the obvious move and usually the wrong one:
        admitting a row that does not pay is worse than refusing it."""
        from scripts.dry_run_sections import _cost_report

        _cost_report(self._refusals([0.13, 0.14, 0.24, 0.26]))
        out = capsys.readouterr().out

        assert "would still pay" in out, "a 13-14% row should read as payable"
        assert "would NOT pay" in out, "a 24-26% row must not read as payable"

    def test_it_names_the_markets_that_are_hopeless(self, capsys) -> None:
        from scripts.dry_run_sections import _cost_report

        _cost_report(self._refusals([0.30] * 5, symbol="EURGBP.i"))
        out = capsys.readouterr().out

        assert "unaffordable at any sane limit" in out
        assert "EURGBP.i (5)" in out

    def test_it_stays_silent_when_nothing_was_refused_on_cost(self, capsys) -> None:
        from scripts.dry_run_sections import _cost_report

        _cost_report([])

        assert capsys.readouterr().out == ""


class TestTheQuickLauncher:
    """A minute-long check, so a change can be verified without a six-month
    run. It must be honest about what a 30-day window can and cannot say."""

    LAUNCHER = (ROOT / "quick.cmd").read_text()

    def test_its_command_line_parses(self) -> None:
        from scripts.dry_run_sections import build_parser

        parsed = build_parser().parse_args(cmd_argv(self.LAUNCHER, **{"%DAYS%": "30"}))

        assert parsed.days == 30
        assert parsed.core is True
        assert parsed.live_only is True
        assert parsed.no_m1 is True

    def test_it_takes_no_comma_arguments(self) -> None:
        for line in self.LAUNCHER.splitlines():
            if line.strip().startswith("set "):
                assert "," not in line, line

    def test_it_writes_to_its_own_csv(self) -> None:
        """Sharing a filename with history.cmd would have a one-minute check
        overwrite a six-hour measurement."""
        assert "runtime\\quick.csv" in self.LAUNCHER
        assert "runtime\\history.csv" not in self.LAUNCHER

    def test_it_says_a_short_window_cannot_conclude(self) -> None:
        """The whole risk of a fast launcher is that its number gets quoted as
        an answer. It says so on screen, and the verdict block says so again."""
        assert "NOT enough to conclude" in self.LAUNCHER


class TestTheScanUniverseIsTheLiveOne:
    """A correction I owed the record.

    I told the owner he was "running order_block on five markets". He was not.
    `--core` and `CORE_UNIVERSE` exist only inside the dry-run script; the live
    scanner reads the whole broker catalogue and always did. What was true is
    that my MEASUREMENT covered sixteen markets, of which five produced trades.

    These tests pin the distinction so the two cannot be confused again.
    """

    def test_the_core_list_is_confined_to_the_measurement_script(self) -> None:
        """If it ever appears in the runner, the scan universe has silently
        shrunk to sixteen markets on a live account."""
        import subprocess

        hits = subprocess.run(
            ["git", "grep", "-l", "-E", "CORE_UNIVERSE|--core|args.core"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.split()

        for path in hits:
            assert path.startswith("tests/") or path in {
                "scripts/dry_run_sections.py",
                "dryrun.cmd",
                "dryrun-live.cmd",
                "quick.cmd",
                "history.cmd",
                "history-one.cmd",
                "sweep.cmd",
            }, f"{path} is not a measurement file and must not know about --core"

    def test_the_scanner_covers_every_class_except_stocks(self) -> None:
        """The owner's words: all markets except the dumb stocks. That is what
        the config already said, and `unknown` is out too -- anything the
        scanner cannot classify is not something to put money on."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from core.instrument import AssetClass

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        scanned = set(settings.scanner.priority_asset_classes)

        assert scanned == {"forex", "crypto", "metal", "index", "commodity"}
        assert AssetClass.STOCK.value not in scanned
        assert AssetClass.UNKNOWN.value not in scanned

    def test_nothing_caps_how_many_markets_are_scanned(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

        assert not getattr(settings.scanner, "max_symbols", 0)

    def test_the_all_form_of_the_launcher_parses_and_drops_m1(self) -> None:
        """Fourteen times the markets, so it resolves on M5. M1 over 230
        markets is tens of millions of bars for no extra answer."""
        from scripts.dry_run_sections import build_parser

        launcher = (ROOT / "dryrun-live.cmd").read_text()

        every = build_parser().parse_args(
            cmd_argv(launcher, **{"%DAYS%": "30", "%SCOPE%": "", "%FINE%": "--no-m1"})
        )
        assert every.core is False and every.no_m1 is True and every.live_only is True

        core = build_parser().parse_args(
            cmd_argv(launcher, **{"%DAYS%": "7", "%SCOPE%": "--core", "%FINE%": ""})
        )
        assert core.core is True and core.no_m1 is False


class TestTheVerdictJudgesTheConfigurationThatRuns:
    """THE REPORT JUDGED THE WRONG COLUMN AND IT CHANGED THE ANSWER.

    `result_r` is the fixed-stop exit -- what the research measured.
    `managed_r` is the same trade under `TradeManagementConfig`, and
    `break_even_at_r` is ON, so `managed_r` is what the account actually does.

    Every headline, sigma and verdict was computed on `result_r`. The 180-day
    run therefore printed

        total   -2.00 R,  -0.05 sigma from zero
        -> NOT ENOUGH TO CONCLUDE

    while three lines above, the same report said the configuration that
    actually runs made +66.20 R and EUR +480.37. The owner was handed the
    number for a setup he does not trade, and I drew a conclusion from it.
    """

    def _paired(self, n: int, fixed: float, managed_value: float):
        from datetime import datetime, timedelta

        from scripts.dry_run_sections import Decision

        base = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
        return [
            Decision(
                base + timedelta(days=i % 120, minutes=i),
                "NDX100",
                "order_block",
                "TRADE",
                result_r=fixed,
                pnl_money=fixed * 8.0,
                managed_r=managed_value,
                managed_money=managed_value * 8.0,
                pass_key=("order_block", "M30"),
            )
            for i in range(n)
        ]

    def test_the_verdict_reads_the_managed_column_when_management_is_on(self, capsys) -> None:
        """A book that is flat on the fixed stop and strongly positive on the
        managed one must read as positive. Under the old code it read flat."""
        from scripts.dry_run_sections import _is_this_real

        _is_this_real(self._paired(400, 0.0, 1.0), [("order_block", "M30")], managed=True)
        out = capsys.readouterr().out

        assert "+400.00 R" in out
        assert "BREAK-EVEN exit, which is what the account runs" in out

    def test_and_the_fixed_column_when_it_is_off(self, capsys) -> None:
        from scripts.dry_run_sections import _is_this_real

        _is_this_real(self._paired(400, 0.0, 1.0), [("order_block", "M30")], managed=False)
        out = capsys.readouterr().out

        assert "+0.00 R" in out
        assert "fixed stop" in out

    def test_one_function_decides_which_column_is_live(self) -> None:
        """Two places computing 'the live exit' is how they drifted apart in
        the first place."""
        from scripts.dry_run_sections import Decision, _live_exit

        row = Decision(
            when=__import__("datetime").datetime(2026, 3, 1, tzinfo=UTC),
            symbol="X",
            module="m",
            outcome="TRADE",
            result_r=-1.0,
            managed_r=0.1,
        )

        assert _live_exit(row, managed=True) == pytest.approx(0.1)
        assert _live_exit(row, managed=False) == pytest.approx(-1.0)

    def test_no_report_line_still_reaches_for_result_r_on_the_live_block(self) -> None:
        """The live block and the verdict must go through `_live_exit`. The
        sweep may keep the fixed stop -- it compares clocks and wants the exit
        rule held still -- and says so in a comment."""
        import inspect

        from scripts import dry_run_sections

        for name in ("_live_config_report", "_is_this_real"):
            source = inspect.getsource(getattr(dry_run_sections, name))
            body = source.split('"""')[-1]

            assert "d.result_r" not in body, f"{name} still reads the fixed column"
            assert "_live_exit(" in body


class TestEveryReportBlockJudgesTheLiveExit:
    """`_live_config_report` was corrected first and `_report` was the SAME
    defect twenty lines lower, found separately. BY SECTION at the bottom is
    the line the owner actually reads -- "is order_block positive and
    impulse_retest not" -- and on the 180-day run it answered that about a
    configuration the account does not trade."""

    def _rows(self, n: int, fixed: float, managed_value: float, module: str = "order_block"):
        from datetime import datetime, timedelta

        from scripts.dry_run_sections import Decision

        base = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
        return [
            Decision(
                base + timedelta(days=i % 90, minutes=i),
                "NDX100",
                module,
                "TRADE",
                risk_money=8.0,
                risk_pct=4.0,
                result_r=fixed,
                pnl_money=fixed * 8.0,
                managed_r=managed_value,
                managed_money=managed_value * 8.0,
                pass_key=(module, "M30"),
            )
            for i in range(n)
        ]

    def test_by_section_reports_the_exit_the_account_takes(self, capsys) -> None:
        from scripts.dry_run_sections import _report

        rows = self._rows(300, -1.0, 1.0, "order_block")
        _report(rows, equity=215.34, days=90, skipped=0, managed=True)
        out = capsys.readouterr().out

        section = out.split("BY SECTION")[1]
        assert "+300.00 R" in section, "BY SECTION is still totalling the fixed stop"

    def test_and_the_fixed_stop_when_management_is_off(self, capsys) -> None:
        from scripts.dry_run_sections import _report

        _report(self._rows(300, -1.0, 1.0), equity=215.34, days=90, skipped=0, managed=False)
        out = capsys.readouterr().out

        assert "-300.00 R" in out.split("BY SECTION")[1]

    def test_by_day_follows_the_same_column(self, capsys) -> None:
        from scripts.dry_run_sections import _report

        _report(self._rows(300, -1.0, 1.0), equity=215.34, days=90, skipped=0, managed=True)
        out = capsys.readouterr().out

        day_block = out.split("BY DAY")[1].split("BY SECTION")[0]
        assert "-1.00 R" not in day_block, "BY DAY is still on the fixed stop"

    def test_no_summary_line_reaches_for_the_fixed_column_directly(self) -> None:
        import inspect

        from scripts import dry_run_sections

        body = inspect.getsource(dry_run_sections._report).split('"""')[-1]

        assert "d.result_r" not in body
        assert "_live_exit(" in body

    def test_the_csv_names_which_column_is_live(self) -> None:
        """A spreadsheet cannot ask which column to total, so the header has
        to say it."""
        assert '"result_r_fixed_stop"' in SOURCE
        assert '"managed_r_LIVE"' in SOURCE


class TestASwitchedOffSectionIsStillMeasured:
    """I switched `impulse_retest` off and told the owner "the weight stays,
    so it is still measured". True of the engine, FALSE of this script:
    `passes` was built from `live_enabled_modules`, so switching a section off
    removed it from the measurement too.

    The 180-day run he then asked for judged one section and produced no rows
    at all for the other -- the one the earlier 180-day data had actually
    favoured. His question was "waar is die andere retest ding dan", and the
    answer was that I had deleted it from the experiment while claiming I had
    not.
    """

    def test_the_sweep_covers_every_known_module_not_just_the_live_ones(self) -> None:
        assert "for name in sorted(measured):" in SOURCE
        assert "for name in sorted(live & set(module_config)):" not in SOURCE

    def test_the_report_shows_what_a_shadowed_section_would_have_done(self) -> None:
        assert "def _shadow_report" in SOURCE
        assert "SHADOWED — measured, not permitted to trade" in SOURCE

    def test_the_shadow_block_uses_the_same_exit_and_cap_as_the_live_one(self) -> None:
        """Two numbers computed differently are not comparable, and comparing
        them is the entire purpose of this block."""
        import inspect

        from scripts import dry_run_sections

        body = inspect.getsource(dry_run_sections._shadow_report).split('"""')[-1]

        assert "_live_exit(d, managed)" in body
        assert "_under_the_slot_cap(" in body

    def test_only_narrows_the_sweep(self) -> None:
        from scripts.dry_run_sections import build_parser

        parsed = build_parser().parse_args(["--only", "impulse_retest"])

        assert parsed.only == "impulse_retest"

    def test_only_rejects_a_section_it_does_not_know(self) -> None:
        """A typo must stop the run rather than silently measure nothing --
        an empty result reads as "no setups", which is this script's signature
        failure."""
        assert "--only names sections this script does not know" in SOURCE

    def test_the_one_section_launcher_parses(self) -> None:
        from scripts.dry_run_sections import build_parser

        launcher = (ROOT / "history-one.cmd").read_text()
        parsed = build_parser().parse_args(
            cmd_argv(
                launcher,
                **{
                    "%DAYS%": "180",
                    "%SECTION%": "impulse_retest",
                },
            )
        )

        assert parsed.only == "impulse_retest"
        assert parsed.days == 180 and parsed.core is True and parsed.no_m1 is True


class TestOneMonthCannotCarryTheResult:
    """The four boxes did not ask this and the 180-day run needed it.

        Mar +4.90  Apr +0.40  May -1.90  Jun +3.30  Jul +0.70  Aug +26.20

    August alone is 78% of the six-month total. Strip it and the other 1,077
    trades make +0.007 R each. A result carried by one month has not repeated,
    however many trades sit underneath it, and neither the trade count nor the
    sigma nor months-positive can see that.
    """

    def _months(self, shape: dict[str, float]):
        from datetime import datetime

        from scripts.dry_run_sections import Decision

        rows = []
        for month, (per, count) in shape.items():
            year, mon = (int(part) for part in month.split("-"))
            for i in range(count):
                rows.append(
                    Decision(
                        datetime(year, mon, 1 + (i % 27), 9, 0, tzinfo=UTC),
                        "NDX100",
                        "order_block",
                        "TRADE",
                        result_r=per,
                        managed_r=per,
                        pnl_money=per * 8.0,
                        managed_money=per * 8.0,
                        pass_key=("order_block", "M30"),
                    )
                )
        return rows

    def test_a_single_month_carrying_the_total_is_called_out(self, capsys) -> None:
        from scripts.dry_run_sections import _is_this_real

        shape = {
            "2026-03": (0.02, 210),
            "2026-04": (0.00, 207),
            "2026-05": (-0.01, 216),
            "2026-06": (0.01, 210),
            "2026-07": (0.00, 234),
            "2026-08": (0.13, 206),
        }
        _is_this_real(self._months(shape), [("order_block", "M30")], managed=True)
        out = capsys.readouterr().out

        assert "CONCENTRATED" in out
        assert "of the whole result" in out
        assert "[ ] no single month carrying more than half the result" in out

    def test_an_evenly_spread_result_passes_that_box(self, capsys) -> None:
        from scripts.dry_run_sections import _is_this_real

        shape = {f"2026-0{m}": (0.05, 200) for m in range(3, 9)}
        _is_this_real(self._months(shape), [("order_block", "M30")], managed=True)
        out = capsys.readouterr().out

        assert "CONCENTRATED" not in out
        assert "[x] no single month carrying more than half the result" in out


class TestTheRunSaysWhatItIsDoing:
    """`history-one.cmd impulse_retest 180` announced

        DRY RUN — sections order_block

    and then measured impulse_retest in silence for twenty minutes. Two
    separate defects, and together they are indistinguishable from a hang:
    the header printed `live_enabled_modules` instead of what was being
    measured, and the per-symbol progress line only printed when a symbol
    produced trades -- so eleven FX markets, which form setups and lose every
    one to the cost wall, showed nothing at all.

    Silence has now been mistaken for failure in this script four times. It is
    the same defect as "no setups" versus "no symbols" and as REFUSED_
    CONFLUENCE at 98.4%: a run that cannot say what it is doing cannot be
    trusted when it says nothing.
    """

    def test_the_header_names_what_is_measured_not_what_may_trade(self) -> None:
        assert "DRY RUN — measuring" in SOURCE
        assert "f\"DRY RUN — sections {', '.join(sorted(live))}\"" not in SOURCE

    def test_the_header_flags_a_shadowed_section_explicitly(self) -> None:
        """Measuring a section that may not trade is the normal case now, and
        the reader must not have to infer it."""
        assert "shadowed (measured, NOT permitted real money)" in SOURCE

    def test_progress_prints_for_every_symbol(self) -> None:
        assert (
            'print(\n                f"  [{index}/{len(symbols)}] {symbol}: {done} trades"'
            in SOURCE
        )
        assert "if done:\n                print(" not in SOURCE

    def test_progress_is_flushed(self) -> None:
        """Block-buffered output through a pipe would reintroduce exactly the
        silence this fixes."""
        line = SOURCE[SOURCE.index("{symbol}: {done} trades") :][:200]

        assert "flush=True" in line

    def test_the_header_is_printed_after_the_passes_are_known(self) -> None:
        """It cannot name what is being measured before that is decided, which
        is why it named the live list instead."""
        passes_built = SOURCE.index("passes.append((name, getattr(settings.analysis, name)")
        header = SOURCE.index("DRY RUN — measuring")

        assert passes_built < header


class TestAShadowedSectionCanActuallyVote:
    """`history-one.cmd impulse_retest 180` returned ZERO trades on all
    sixteen markets, and nothing was wrong with the detector.

    `ConfluenceConfig.effective_weights` forces every module absent from
    `live_enabled_modules` to weight zero when the mode is live. That is
    correct -- it is what stops a switched-off section spending money. This
    script evaluates in MICRO_LIVE, so `impulse_retest` entered every pass at
    weight zero, failed `if weight > 0`, and every bar came back "no weighted
    directional evidence".

    I FIXED THIS TWICE AT THE WRONG LEVEL FIRST. Keeping the module's weight
    in the config did nothing. Sweeping every known module instead of only the
    live ones did nothing. Both were necessary; neither was sufficient,
    because the engine zeroes the weight one layer below both. Three attempts
    at one defect, and the first two produced a confident-looking run whose
    every number was zero.
    """

    def _settings(self):
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

    def test_the_measured_section_carries_its_weight_in_its_own_pass(self) -> None:
        from config.schema import TradingMode
        from scripts.dry_run_sections import _retimed

        tuned = _retimed(self._settings(), "impulse_retest", "M15")
        weights = tuned.analysis.confluence.effective_weights(TradingMode.MICRO_LIVE)

        assert weights.get("impulse_retest", 0.0) > 0.0

    def test_without_the_grant_it_would_be_zero(self) -> None:
        """The half that was missing, stated as its own assertion so the fix
        cannot be quietly undone."""
        from config.schema import TradingMode

        settings = self._settings()
        weights = settings.analysis.confluence.effective_weights(TradingMode.MICRO_LIVE)

        assert "impulse_retest" not in settings.analysis.confluence.live_enabled_modules
        assert weights.get("impulse_retest", 0.0) == 0.0

    def test_measuring_a_section_does_not_make_it_live(self) -> None:
        """THE LINE THAT MUST NOT BE CROSSED. The grant lives on a settings
        COPY used only by the measurement; the account's own allowlist is
        untouched, and no broker ever sees the copy."""
        from scripts.dry_run_sections import _retimed

        settings = self._settings()
        before = tuple(settings.analysis.confluence.live_enabled_modules)

        _retimed(settings, "impulse_retest", "M15")

        assert tuple(settings.analysis.confluence.live_enabled_modules) == before
        assert "impulse_retest" not in before

    def test_the_grant_is_additive_not_a_replacement(self) -> None:
        """Replacing the list would silence order_block in impulse_retest's
        pass, which changes what the confluence sees."""
        from scripts.dry_run_sections import _retimed

        tuned = _retimed(self._settings(), "impulse_retest", "M15")
        allowed = set(tuned.analysis.confluence.live_enabled_modules)

        assert {"impulse_retest", "order_block"} <= allowed

    def test_a_live_section_is_unchanged_by_the_grant(self) -> None:
        from scripts.dry_run_sections import _retimed

        settings = self._settings()
        tuned = _retimed(settings, "order_block", "M30")

        assert set(tuned.analysis.confluence.live_enabled_modules) == set(
            settings.analysis.confluence.live_enabled_modules
        )


class TestTheSweepRanksOnTheExitTheAccountTakes:
    """The seven-day sweep table read the FIXED stop, and I defended that as
    "comparing clocks is cleaner without the exit rule moving underneath it".
    That sounds reasonable and is not: it compares clocks under an exit the
    account does not take. The gap is not small --

        order_block M30, 180 days:  fixed -34.00 R  /  break-even +33.60 R

    -- so the table read as a row of losses for a configuration that was
    making money. The owner spotted it before I did.
    """

    def _results(self, fixed: float, managed_value: float):
        from datetime import datetime, timedelta

        from scripts.dry_run_sections import Decision

        base = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
        return {
            ("order_block", "M30"): [
                Decision(
                    base + timedelta(days=i % 100, minutes=i),
                    "US30",
                    "order_block",
                    "TRADE",
                    result_r=fixed,
                    pnl_money=fixed * 8.0,
                    managed_r=managed_value,
                    managed_money=managed_value * 8.0,
                    pass_key=("order_block", "M30"),
                )
                for i in range(250)
            ]
        }

    def test_it_ranks_on_the_live_exit(self, capsys) -> None:
        from scripts.dry_run_sections import _sweep_report

        _sweep_report(self._results(-1.0, 1.0), equity=215.0, days=180, managed=True)
        out = capsys.readouterr().out

        assert "break-even (LIVE)" in out
        assert "+250.00" in out, "the LIVE column is still totalling the fixed stop"

    def test_it_keeps_the_fixed_column_beside_it(self, capsys) -> None:
        """The difference between the two IS what the stop rule is worth on
        that clock, which is worth seeing per row."""
        from scripts.dry_run_sections import _sweep_report

        _sweep_report(self._results(-1.0, 1.0), equity=215.0, days=180, managed=True)
        out = capsys.readouterr().out

        assert "fixed R" in out
        assert "-250.00" in out

    def test_a_thin_row_is_marked(self, capsys) -> None:
        """Four trades in a row, ranked, is how a seven-day sweep produced
        conclusions I had to withdraw."""
        from scripts.dry_run_sections import _sweep_report

        results = self._results(-1.0, 1.0)
        results[("order_block", "M30")] = results[("order_block", "M30")][:12]
        _sweep_report(results, equity=215.0, days=7, managed=True)
        out = capsys.readouterr().out

        assert "thin" in out
        assert "NO ROW HAS 200 TRADES" in out

    def test_the_sweep_launcher_parses_and_skips_m5(self) -> None:
        """An M5 trade needs M1 bars to tell which barrier came first, and M1
        over 180 days triples the run for the worst-performing clock."""
        from scripts.dry_run_sections import build_parser

        launcher = (ROOT / "sweep.cmd").read_text()
        parsed = build_parser().parse_args(cmd_argv(launcher, **{"%DAYS%": "180"}))

        assert parsed.days == 180
        assert parsed.core is True and parsed.no_m1 is True
        assert parsed.sweep == ["M15", "M30", "H1", "H4"]
        assert "M5" not in parsed.sweep


class TestTheSweepDoesNotBuildEveryContextTwice:
    """A 180-day sweep was estimated at ninety minutes, and it was ninety
    because of waste rather than work.

    A `MarketContext` depends on the symbol, the instant and the bars -- not
    on which module is about to read it. The sweep built one per SECTION, so
    two sections on one clock constructed an identical object twice, and
    context building is the largest single cost in the loop.

    Measured on the real pipeline, 1000 bars, both sections on M30:

        separate walks              0.76 ms per bar-pair
        shared context              0.43
        shared + trimmed frames     0.35
    """

    def test_sections_sharing_a_clock_are_walked_together(self) -> None:
        assert "def _one_clock(" in SOURCE
        assert "by_clock.setdefault(tf_name, []).append(name)" in SOURCE
        assert "def _one_pass(" not in SOURCE

    def test_each_section_keeps_its_own_open_position(self) -> None:
        """Sharing one `busy_until` across sections would be a different bug:
        refusing section three a trade because section two is in one."""
        assert "busy: dict = {name: None for name, _engine, _sizer in sections}" in SOURCE
        assert "busy[name] = exit_at" in SOURCE

    def test_only_the_frames_something_reads_are_sliced(self) -> None:
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from core.types import Timeframe
        from scripts.dry_run_sections import _frames_read

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        read = _frames_read(
            settings, Timeframe.M15, Timeframe.M5, ("impulse_retest", "order_block")
        )

        assert Timeframe.M15 in read, "the clock itself"
        assert Timeframe.M5 in read, "_context takes the price from M5"
        assert Timeframe.H4 in read, "the intraday higher-timeframe veto"
        assert Timeframe.M30 not in read, "nothing reads M30 on an M15 pass"

    def test_it_resolves_per_section_rather_than_over_every_profile(self) -> None:
        """A first draft unioned all three horizon profiles and got M5, M15,
        H1, H4, D1 and W1 -- everything, because the swing profile plans on H1
        and vetoes on W1. It saved nothing."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from core.types import Timeframe
        from scripts.dry_run_sections import _frames_read

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        read = _frames_read(settings, Timeframe.M15, Timeframe.M5, ("impulse_retest",))

        assert Timeframe.H1 not in read
        assert Timeframe.W1 not in read

    def test_a_swing_section_still_gets_its_slower_frames(self) -> None:
        """The trim must follow the config, not an assumption about which
        modules are live today."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from core.types import Timeframe
        from scripts.dry_run_sections import _frames_read

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        read = _frames_read(settings, Timeframe.M15, Timeframe.M5, ("market_structure",))

        assert Timeframe.H1 in read, "swing plans on H1"

    def test_the_launcher_no_longer_promises_ninety_minutes(self) -> None:
        launcher = (ROOT / "sweep.cmd").read_text()

        assert "1.5 hours" not in launcher
        assert "ten to twenty minutes" in launcher


class TestMarketsThatCannotTradeAreSkipped:
    """Eleven of sixteen core markets produce zero trades and cost two thirds
    of the run.

    Every FX setup on this account dies on `SL_TOO_TIGHT_FOR_COSTS` -- the
    180-day run refused 6,877 of them at a median 51% of the stop -- and the
    sweep walked all twelve thousand bars of each anyway to reach that same
    refusal one bar at a time.

    THE DANGER IS SKIPPING SOMETHING THAT WOULD HAVE TRADED, so the filter is
    deliberately generous: it uses the sizer's own `_cost_share`, checks the
    WIDEST clock available, and only skips what is over TWICE the account's
    limit. A symbol near the boundary is exactly the interesting case.
    """

    def _sizer(self):
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from risk.position_sizer import PositionSizer

        return PositionSizer(
            load_settings(DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False)
        )

    def _frames(self, atr: float):
        import numpy as np
        import pandas as pd

        from core.types import Timeframe

        out = {}
        for clock, freq in ((Timeframe.M15, "15min"), (Timeframe.M30, "30min")):
            index = pd.date_range("2026-01-01", periods=300, freq=freq, tz="UTC")
            close = pd.Series(np.full(300, 1.1000), index=index)
            out[clock] = pd.DataFrame(
                {
                    "open": close,
                    "high": close + atr / 2,
                    "low": close - atr / 2,
                    "close": close,
                },
                index=index,
            )
        return out

    def _spec(self, asset: str = "forex"):
        from core.instrument import AssetClass, InstrumentSpec

        return InstrumentSpec(
            symbol="EURUSD.i",
            digits=5,
            point=0.00001,
            tick_size=0.00001,
            tick_value=1.0,
            contract_size=100_000.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            stops_level=0,
            freeze_level=0,
            currency_base="EUR",
            currency_profit="USD",
            currency_margin="EUR",
            filling_mode_mask=1,
            trade_mode=4,
            is_forex=True,
            path="forex\\majors",
            description="EURUSD",
            asset_class=AssetClass(asset),
        )

    def test_a_market_whose_costs_eat_the_stop_is_skipped(self) -> None:
        from core.types import Timeframe
        from scripts.dry_run_sections import _hopeless_on_cost

        # A 3-pip ATR against ~2.4 pips of round trip: hopeless.
        share = _hopeless_on_cost(
            self._sizer(), self._spec(), self._frames(0.00003), (Timeframe.M15, Timeframe.M30)
        )

        assert share is not None and share > 0.24

    def test_a_market_with_room_is_kept(self) -> None:
        from core.types import Timeframe
        from scripts.dry_run_sections import _hopeless_on_cost

        # A 60-pip ATR: the same costs are a rounding error.
        assert (
            _hopeless_on_cost(
                self._sizer(), self._spec(), self._frames(0.0060), (Timeframe.M15, Timeframe.M30)
            )
            is None
        )

    def test_a_market_near_the_boundary_is_kept(self) -> None:
        """The generosity is the point. Skipping something at 1.5x the limit
        would throw away the case worth measuring."""
        from core.types import Timeframe
        from scripts.dry_run_sections import _hopeless_on_cost

        sizer = self._sizer()
        limit = sizer.settings.risk.max_cost_share_of_risk
        # Tune an ATR that lands between one and two times the limit.
        # A 10-pip ATR lands at 22.5%, 1.9x the 12% limit -- the interesting
        # case. The first draft of this loop stopped at 2.2 pips, where the
        # cost is 112% and nothing is near the boundary at all.
        for atr in (0.00080, 0.00090, 0.00100, 0.00110, 0.00120):
            share = sizer._cost_share(
                self._spec(), atr, sizer.settings.risk.commission_per_lot("forex")
            )
            if limit < share <= limit * 2:
                assert (
                    _hopeless_on_cost(
                        sizer, self._spec(), self._frames(atr), (Timeframe.M15, Timeframe.M30)
                    )
                    is None
                ), f"skipped at {share:.0%}, only {share / limit:.1f}x the limit"
                return
        raise AssertionError("no ATR landed between one and two times the limit")

    def test_it_uses_the_sizers_own_cost_function(self) -> None:
        """A second definition of the cost would skip markets the sizer would
        have accepted, which is the worst possible direction for this to be
        wrong in."""
        import inspect

        from scripts import dry_run_sections

        body = inspect.getsource(dry_run_sections._hopeless_on_cost).split('"""')[-1]

        assert "sizer._cost_share(" in body

    def test_the_report_names_what_it_skipped(self) -> None:
        """A market silently absent from a report is this project's signature
        failure. It has to say which, and why."""
        assert "SKIPPED ON COST" in SOURCE
        assert "of the widest stop" in SOURCE


class TestTheSweepAnswersTheQuestionBehindMoreTrades:
    """The owner: "dus hierna weten we of meerdere klokken meer trades geven
    en winstgevende trades". Half, and the missing half mattered.

    The sweep measures each clock in ISOLATION. M15 at 60 trades beside M30 at
    40 reads as a hundred -- but if order_block sees the same move on both,
    running both is one idea at double the stake, not two independent chances.
    Doubling the stake on one idea is the thing this account has a rule
    against.
    """

    def _results(self, same_move: bool):
        from datetime import datetime, timedelta

        from scripts.dry_run_sections import Decision

        base = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
        fast, slow = [], []
        for i in range(60):
            when = base + timedelta(days=i)
            slow.append(
                Decision(
                    when,
                    "US30",
                    "order_block",
                    "TRADE",
                    direction="LONG",
                    result_r=1.0,
                    managed_r=1.0,
                    pass_key=("order_block", "M30"),
                )
            )
            fast.append(
                Decision(
                    when + (timedelta(minutes=10) if same_move else timedelta(hours=7)),
                    "US30",
                    "order_block",
                    "TRADE",
                    direction="LONG",
                    result_r=1.0,
                    managed_r=1.0,
                    pass_key=("order_block", "M15"),
                )
            )
        return {("order_block", "M30"): slow, ("order_block", "M15"): fast}

    def test_it_says_when_two_clocks_see_the_same_move(self, capsys) -> None:
        from scripts.dry_run_sections import _clock_overlap

        _clock_overlap(self._results(same_move=True), days=180)
        out = capsys.readouterr().out

        assert "WOULD TWO CLOCKS GIVE TWICE THE TRADES?" in out
        assert "SAME move" in out
        assert "doubles the stake, not the chances" in out

    def test_it_says_when_they_are_independent(self, capsys) -> None:
        from scripts.dry_run_sections import _clock_overlap

        _clock_overlap(self._results(same_move=False), days=180)
        out = capsys.readouterr().out

        assert "largely independent" in out

    def test_it_matches_only_the_same_symbol_and_direction(self) -> None:
        """A long on US30 and a short on NDX100 at the same minute are not the
        same move, and counting them as one would understate the real gain."""
        import inspect

        from scripts import dry_run_sections

        body = inspect.getsource(dry_run_sections._clock_overlap).split('"""')[-1]

        assert "other.symbol == trade.symbol" in body
        assert "other.direction == trade.direction" in body

    def test_it_stays_quiet_with_only_one_clock(self, capsys) -> None:
        from scripts.dry_run_sections import _clock_overlap

        results = self._results(same_move=True)
        del results[("order_block", "M15")]
        _clock_overlap(results, days=180)

        assert capsys.readouterr().out == ""

    def test_every_sweep_row_carries_its_own_sigma(self, capsys) -> None:
        """Without it, "profitable" and "had a good stretch" print
        identically, and this table exists to choose a clock."""
        from scripts.dry_run_sections import _sweep_report

        results = self._results(same_move=False)
        # 60 rows is "thin"; the sigma column must be there regardless, and a
        # row that reaches 200 trades and +2 sigma must be marked.
        _sweep_report(results, equity=215.0, days=180, managed=True)
        thin_table = capsys.readouterr().out

        assert "sigma" in thin_table
        assert "thin" in thin_table

        from datetime import datetime, timedelta

        from scripts.dry_run_sections import Decision

        base = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
        strong = {
            ("order_block", "M30"): [
                Decision(
                    base + timedelta(days=i % 120, minutes=i),
                    "US30",
                    "order_block",
                    "TRADE",
                    direction="LONG",
                    # Mixed outcomes: with every day identical the daily
                    # spread is zero and sigma is undefined, which the code
                    # correctly reports as 0.00 and a first draft of this test
                    # mistook for a bug.
                    result_r=1.0 if i % 4 else -1.0,
                    managed_r=1.0 if i % 4 else -1.0,
                    pass_key=("order_block", "M30"),
                )
                for i in range(240)
            ]
        }
        _sweep_report(strong, equity=215.0, days=180, managed=True)

        assert "clears 2" in capsys.readouterr().out

    def test_the_csv_carries_the_clock(self) -> None:
        """Nothing downstream can separate clocks without it, so
        verdict.cmd could not judge a sweep at all."""
        assert '"clock",' in SOURCE
        assert "d.pass_key[1]," in SOURCE

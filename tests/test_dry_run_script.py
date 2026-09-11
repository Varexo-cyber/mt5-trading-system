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
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "scripts" / "dry_run_sections.py").read_text(encoding="utf-8")


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
    tail = line.split("scripts.dry_run_sections", 1)[1]

    # EXPAND FIRST, TOKENISE SECOND -- the order cmd itself uses, and the order
    # matters. Tokenising first hides a quoted value that arrives from inside a
    # variable: `%SECTIES%` holding `--only "a,b"` is one token before
    # expansion and three arguments after it, and a helper that decided on the
    # unexpanded token would never see the quotes at all.
    for marker in sorted(set(re.findall(r"%[A-Z_][A-Z0-9_]*%", tail)), reverse=True):
        assert marker in values, f"the launcher uses {marker} and this test does not set it"
        tail = tail.replace(marker, values[marker])

    argv: list[str] = []
    # A quoted run is one argument; everything else is a run of non-space.
    for piece in re.findall(r'"[^"]*"|\S+', tail):
        if piece.startswith('"') and piece.endswith('"') and len(piece) > 1:
            # NOT SPLIT, on whitespace or on commas. That is what the quotes
            # are for: a launcher quotes `--only a,b` precisely so cmd cannot
            # turn it into two arguments and stop argparse mid-run.
            argv.append(piece[1:-1])
            continue
        # THE SPLIT HAPPENS AFTER EXPANSION, which is the cmd behaviour that
        # matters for the unquoted case. `%CLOCKS%` holding "M1 M5" becomes two
        # arguments, not one argument containing a space -- and a helper that
        # produced the latter would let a launcher pass "M1 M5" as a single
        # timeframe name and report it as parsed.
        argv.extend(part for word in piece.split() for part in word.split(",") if part)
    return argv


class TestItCallsTheRealApi:
    def test_the_connector_gets_the_settings_block_first(self) -> None:
        """`MT5Connector(config, credentials, ...)`. Passing only credentials
        made it read `credentials.terminal_path`, and the run died there."""
        from core.mt5_connector import MT5Connector

        parameters = list(inspect.signature(MT5Connector.__init__).parameters)

        assert parameters[1] == "config"
        assert parameters[2] == "credentials"
        # WHITESPACE-INSENSITIVE. This matched an exact indentation and broke
        # the moment the construction moved inside an `else:` for the offline
        # cache -- failing for a reformat rather than for the defect it names.
        assert "MT5Connector( settings.mt5, credentials," in " ".join(SOURCE.split())

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

    def test_the_overlay_keeps_quarantined_modules_measurable(self) -> None:
        """A quarantined module keeps its raw weight so a shadow pass can still
        measure it; what stops it spending money is its absence from the
        allowlist, not a zeroed weight.

        This asserts that PROPERTY and not a frozen list of live names -- an
        earlier version pinned the four modules that happened to be live, so
        promoting a fifth failed here with nothing actually wrong.
        """
        from config.loader import load_settings

        settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
        live = set(settings.analysis.confluence.live_enabled_modules)

        assert live, "the overlay is what grants permission to trade; it cannot be empty"
        for quarantined in ("impulse_retest", "order_block"):
            assert quarantined not in live
            assert settings.analysis.confluence.weights.get(quarantined, 0.0) > 0.0

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
    def test_a_bar_holding_both_barriers_is_a_loss(self) -> None:
        """THIS IS WHAT MAKES SAME-CLOCK RESOLUTION SAFE TO ALLOW.

        Rewritten 31 August. It used to assert the string
        `clock.duration <= finest.duration` was somewhere in the source, which
        is the defect this project keeps producing in its own tests: it can
        only see that a line exists, never what it does. The guard has since
        changed to `<` so M1 -- which has nothing finer beneath it -- can be
        swept at all, and a string test would have failed for the change while
        proving nothing about the property it was named after.

        The property is that ambiguity resolves AGAINST the trade. One bar,
        long, low through the stop and high through the target: the order
        inside it is unknowable and it must book -1R.
        """
        from datetime import datetime, timedelta

        import pandas as pd

        from core.types import Direction
        from scripts.dry_run_sections import _resolve

        base = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
        frame = pd.DataFrame(
            {"high": [1.1030], "low": [1.0970], "open": [1.10], "close": [1.10]},
            index=pd.DatetimeIndex([base + timedelta(minutes=1)]),
        )

        class _Idea:
            direction = Direction.LONG
            entry = 1.10
            stop_loss = 1.099
            take_profit = 1.102

        fixed, exit_at, _managed, _managed_at = _resolve(frame, base, _Idea(), horizon_bars=96)

        assert fixed == -1.0, "the bar touched both, so it may not be scored as a win"
        assert exit_at is not None

    def test_a_clock_finer_than_its_resolution_frame_is_refused_out_loud(self) -> None:
        """`--no-m1 --sweep M1` cannot be measured at all: M5 bars cannot walk
        out an M1 trade. The old code dropped such a clock with a bare
        `continue`, and an absent row reads exactly like a row of zeros. It is
        a complaint now, raised before the fetch rather than after it."""
        from core.types import Timeframe
        from scripts.dry_run_sections import _unresolvable_clocks

        assert _unresolvable_clocks(("M15", "M30", "H1"), Timeframe.M5) == ""
        assert _unresolvable_clocks(("M1", "M5"), Timeframe.M1) == ""

        complaint = _unresolvable_clocks(("M1", "M5", "M15"), Timeframe.M5)

        assert complaint.startswith("M1 cannot be resolved on M5 bars")
        assert "--no-m1" in complaint, "the message has to name the flag that fixes it"

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
        assert "busy: dict = {row[0]: None for row in sections}" in SOURCE
        assert (
            "awake = [row for row in sections if busy[row[0]] is None or upto > busy[row[0]]]"
            in SOURCE
        )

    def test_an_unresolved_trade_keeps_holding_the_symbol(self) -> None:
        """Freeing the symbol on a trade that never resolved would let the same
        signal re-enter while the position is still open."""
        source = " ".join(SOURCE.split())

        assert "busy[name] = freed if freed is not None else end + clock.duration" in source

    def test_the_symbol_is_freed_at_the_exit_the_account_takes(self) -> None:
        """THE BUG THIS REPLACED. `busy` was set from `exit_at`, the FIXED
        stop, on every run -- including runs judged on the break-even column.
        A trade that scratched after four minutes kept its symbol occupied
        until the fixed stop resolved hours later, and every setup in between
        was dropped without a trace.

        On an M1 section that is most of the trades, so the harness reported
        far fewer than the strategy would have taken.
        """
        source = " ".join(SOURCE.split())

        # `resolved_manage`, not `section_manage`: the rule is resolved per
        # trade now, and it comes back None when the H1 ATR is unknown -- live
        # refuses to move a stop in that case too. Reading the CONFIGURED rule
        # here would free the symbol on a managed exit that never happened.
        assert (
            "freed = managed_at if (resolved_manage is not None or jarvis_replay) else exit_at"
            in source
        )

    def test_the_resolver_reports_both_exit_times(self) -> None:
        """Measured, not grepped. Break-even scratches this trade on the first
        bar; the fixed stop only resolves on the last. Those are different
        instants and the resolver has to hand back both."""
        from datetime import datetime, timedelta

        import pandas as pd

        from core.types import Direction
        from scripts.dry_run_sections import _resolve

        base = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
        # Bar 1 runs up 1R (arms break-even) then back through entry.
        # Bars 2-3 drift, bar 4 takes out the original stop.
        index = pd.DatetimeIndex([base + timedelta(minutes=i) for i in range(1, 5)])
        frame = pd.DataFrame(
            {
                "high": [1.1020, 1.1005, 1.1005, 1.1005],
                "low": [1.0999, 1.0995, 1.0992, 1.0985],
                "open": [1.10] * 4,
                "close": [1.10] * 4,
            },
            index=index,
        )

        class _Idea:
            direction = Direction.LONG
            entry = 1.1000
            stop_loss = 1.0990
            take_profit = 1.1030

        # OFFSET IN PRICE. Risk here is 0.0010, so a 0.10R protective step is
        # 0.00010 of price. Passing 0.10 meant a stop a tenth of a euro above
        # entry on a 1.10 quote -- a hundred times the whole stop -- and the
        # arming guard correctly refused it.
        fixed, exit_at, managed, managed_at = _resolve(
            frame, base, _Idea(), horizon_bars=96, manage=(0.25, 0.00010)
        )

        assert fixed == -1.0, "the original stop was taken on the last bar"
        assert managed is not None and managed > 0, "break-even saved it above entry"
        assert managed_at is not None and exit_at is not None
        assert managed_at < exit_at, "the managed exit came first and frees the symbol first"

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

        r, exit_at, _managed, _managed_at = _resolve(
            frame, start, Idea(Direction.LONG, 100.0, 99.0, 101.0), horizon_bars=2
        )

        assert r == -1.0
        assert exit_at == index[1]

    def test_fixed_trade_is_closed_at_its_configured_pre_pause_time(self) -> None:
        """S10 has no break-even manager, but live closes it before gold's
        daily pause. The replay must do the same instead of holding one quiet
        trade for the rest of a 180-day run."""
        from datetime import datetime, timedelta

        import pandas as pd

        from core.types import Direction
        from scripts.dry_run_sections import _resolve

        start = datetime(2026, 8, 24, 19, 0, tzinfo=UTC)
        index = pd.DatetimeIndex([start + timedelta(minutes=30 * i) for i in range(1, 5)])
        frame = pd.DataFrame(
            {
                "open": [100.0, 100.1, 100.2, 100.3],
                "high": [100.2, 100.3, 100.4, 100.5],
                "low": [99.8, 99.9, 100.0, 100.1],
                "close": [100.1, 100.2, 100.3, 100.4],
            },
            index=index,
        )

        class _Idea:
            direction = Direction.LONG
            entry = 100.0
            stop_loss = 99.0
            take_profit = 102.0

        fixed, exit_at, _managed, _managed_at = _resolve(
            frame,
            start,
            _Idea(),
            horizon_bars=2,
            force_close_at=start.replace(hour=20, minute=50),
        )

        assert fixed == pytest.approx(0.4)
        assert exit_at == index[-1]

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

        taken, _why = _under_the_slot_cap(trades, slots=2)

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

        assert len(_under_the_slot_cap(trades, slots=1)[0]) == 2

    def test_two_sections_cannot_open_the_same_symbol_at_once(self) -> None:
        """The account permits one position per symbol, not one per module."""
        from datetime import datetime, timedelta

        from scripts.dry_run_sections import Decision, _under_the_slot_cap

        base = datetime(2026, 8, 24, tzinfo=UTC)
        trades = [
            Decision(base, "US30", "order_block_fast", "TRADE", exit_at=base + timedelta(hours=1)),
            Decision(
                base + timedelta(minutes=5),
                "US30",
                "order_block_m15",
                "TRADE",
                exit_at=base + timedelta(hours=2),
            ),
            Decision(
                base + timedelta(minutes=10),
                "NDX100",
                "order_block_fast",
                "TRADE",
                exit_at=base + timedelta(hours=1),
            ),
            Decision(
                base + timedelta(hours=1, minutes=1),
                "US30",
                "order_block_m15",
                "TRADE",
                exit_at=base + timedelta(hours=2),
            ),
        ]

        taken, _why = _under_the_slot_cap(trades, slots=4)

        assert [(row.symbol, row.module) for row in taken] == [
            ("US30", "order_block_fast"),
            ("NDX100", "order_block_fast"),
            ("US30", "order_block_m15"),
        ]

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
        """DRIVEN, NOT MATCHED. This asserted the literal call text
        `_under_the_slot_cap(everything, slots)`, so adding an argument to
        that call broke it while the behaviour was unchanged -- and a
        substring is not a behaviour. `test_the_override_actually_reaches_
        the_target` in test_family_target_is_the_exit is the standing example:
        it checked for a string that appeared, in a loop whose answer was
        discarded four hundred lines later."""
        from datetime import UTC, datetime, timedelta

        from scripts.dry_run_sections import Decision, _under_the_slot_cap

        base = datetime(2026, 3, 2, tzinfo=UTC)

        def held(minute: int, symbol: str, module: str, hours: int = 4) -> Decision:
            row = Decision(base + timedelta(minutes=minute), symbol, module, "TRADE")
            row.exit_at = base + timedelta(minutes=minute, hours=hours)
            row.direction = "LONG"
            return row

        # Two slots, three markets wanting one at the same moment.
        wanted = [held(0, "XAUUSD", "a"), held(1, "SPX500", "a"), held(2, "US30", "a")]
        assert len(_under_the_slot_cap(wanted, 2)[0]) == 2
        assert len(_under_the_slot_cap(wanted, 0)[0]) == 3, "0 means no cap at all"

        # One market, twice, while the first is still open.
        twice = [held(0, "XAUUSD", "a"), held(30, "XAUUSD", "a")]
        assert len(_under_the_slot_cap(twice, 4)[0]) == 1

        # And after it closes, the second is taken.
        later = [held(0, "XAUUSD", "a", hours=1), held(120, "XAUUSD", "a")]
        assert len(_under_the_slot_cap(later, 4)[0]) == 2

    def test_two_sections_share_a_symbol_only_when_the_account_allows_it(self) -> None:
        """The replay has to hold the same book the account holds.

        `sections_may_share_a_symbol` lets a SECOND section join a symbol
        another section already has. A replay that keeps refusing per symbol
        measures a different account from the one that trades, which is the
        one thing this function exists to prevent.
        """
        from datetime import UTC, datetime, timedelta

        from scripts.dry_run_sections import Decision, _under_the_slot_cap

        base = datetime(2026, 3, 2, tzinfo=UTC)

        def row(minute: int, module: str, direction: str = "LONG") -> Decision:
            made = Decision(base + timedelta(minutes=minute), "XAUUSD", module, "TRADE")
            made.exit_at = base + timedelta(minutes=minute, hours=4)
            made.direction = direction
            return made

        pair = [row(0, "section_six_gold_m5"), row(30, "section_ten_gold_m1")]

        assert len(_under_the_slot_cap(pair, 4)[0]) == 1
        assert len(_under_the_slot_cap(pair, 4, share_between_sections=True)[0]) == 2

        # The same section twice is pyramiding, and it stays refused.
        same = [row(0, "section_ten_gold_m1"), row(30, "section_ten_gold_m1")]
        assert len(_under_the_slot_cap(same, 4, share_between_sections=True)[0]) == 1

        # Opposite directions is flat exposure bought with two spreads.
        against = [row(0, "section_six_gold_m5"), row(30, "section_ten_gold_m1", "SHORT")]
        assert len(_under_the_slot_cap(against, 4, share_between_sections=True)[0]) == 1
        assert (
            len(
                _under_the_slot_cap(
                    against, 4, share_between_sections=True, refuse_opposite=False
                )[0]
            )
            == 2
        )


class TestTheSectionsFiveToTenLauncher:
    """The operator's short run measures only S5-S10 on their own clocks.

    S8-S10 remain shadowed; including them in a read-only replay must never be
    confused with adding them to the account's real-money allowlist.
    """

    LAUNCHER = (ROOT / "dryrun-live.cmd").read_text()

    def test_its_command_line_parses(self) -> None:
        from scripts.dry_run_sections import build_parser

        # Both forms the launcher can emit: the default core run, and the
        # "all" run where %SCOPE% expands to nothing.
        core = build_parser().parse_args(
            cmd_argv(self.LAUNCHER, **{"%DAYS%": "30", "%SCOPE%": "--core", "%FINE%": ""})
        )

        assert core.days == 30
        assert core.sections_five_to_ten is True
        assert core.live_only is True
        assert core.core is True
        assert core.sweep == []

        every = build_parser().parse_args(
            cmd_argv(self.LAUNCHER, **{"%DAYS%": "7", "%SCOPE%": "", "%FINE%": "--no-m1"})
        )

        assert every.days == 7
        assert every.sections_five_to_ten is True
        assert every.live_only is True
        assert every.core is False

    def test_the_preset_names_exactly_the_six_requested_sections(self) -> None:
        wanted = {
            "failed_session_breakout",
            "section_six_gold_m5",
            "section_eight_trend_day_h1",
            "section_ten_gold_m1",
        }

        for name in wanted:
            assert f'"{name}"' in SOURCE
        assert "if args.sections_five_to_ten:" in SOURCE

    def test_it_takes_no_comma_arguments(self) -> None:
        for line in self.LAUNCHER.splitlines():
            if line.strip().startswith("set "):
                assert "," not in line, line

    def test_live_only_beats_a_sweep_that_is_also_present(self) -> None:
        """Otherwise the flag is advisory and the run costs what it always did."""
        assert "if args.live_only:\n            args.sweep = []" in SOURCE

    def test_live_only_replays_only_the_real_money_allowlist(self) -> None:
        """The old flag disabled the clock sweep but still replayed every
        shadow module. That made `dryrun-live.cmd` slow and printed impulse and
        order-block totals even though neither was active."""
        live_filter = SOURCE.index("measured = measured & live")
        passes = SOURCE.index("passes: list[tuple[str, str]]")

        assert live_filter < passes
        assert "missing = live - measured" in SOURCE


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
        shrunk to sixteen markets on a live account.

        PINNED AS A PROPERTY, NOT AS A LIST OF FILENAMES. This used to hold a
        hand-kept set of launchers, so every new measurement launcher failed
        it and the fix was to add a name -- which trains you to add names, and
        the day a LIVE launcher needed adding it would have been added too.
        What actually matters is what the file starts: a measurement launcher
        drives the replay script, a live launcher starts the runner, and only
        the second one may never narrow the universe.
        """
        import subprocess

        hits = subprocess.run(
            ["git", "grep", "-l", "-E", "CORE_UNIVERSE|--core|args.core"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.split()
        assert hits, "the flag has vanished entirely; this test is guarding nothing"

        # What makes a file a measurement: it drives the replay, and it does
        # not start the live runner.
        measures = re.compile(r"dry_run_sections|replay_requested_markets|fetch_history")
        # `jarvis.py` and `main.py` are how a live launcher starts the runner.
        # Deliberately NOT `--live`: `--live-only` is a replay flag meaning
        # "measure only the sections that may spend money", and matching it
        # would fail exactly the launchers this test wants to allow.
        runs_live = re.compile(r"\bmain\.py|\bjarvis\.py|runner\.service")

        for path in hits:
            if path.startswith("tests/") or path == "scripts/dry_run_sections.py":
                continue
            body = (ROOT / path).read_text(encoding="utf-8", errors="replace")
            assert measures.search(body), (
                f"{path} knows about --core but drives no measurement script"
            )
            assert not runs_live.search(body), (
                f"{path} narrows the universe AND starts the live runner"
            )

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

    def test_btc_replay_sections_are_known(self) -> None:
        for section in (
            "section_fifteen_btc_m1",
            "section_sixteen_btc_m5",
            "section_seventeen_btc_m15",
        ):
            assert f'"{section}": "{section}"' in SOURCE

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

        tuned = _retimed(self._settings(), "market_structure", "M15")
        weights = tuned.analysis.confluence.effective_weights(TradingMode.MICRO_LIVE)

        assert weights.get("market_structure", 0.0) > 0.0

    def test_without_the_grant_it_would_be_zero(self) -> None:
        """The half that was missing, stated as its own assertion so the fix
        cannot be quietly undone.

        Uses `market_structure` rather than `impulse_retest`: section two went
        back on the live list on 31 August, so it is no longer an example of a
        shadowed module. `market_structure` carries a weight of 1.0 and has
        never been on the allowlist, which is exactly the shape this fix is
        for."""
        from config.schema import TradingMode

        settings = self._settings()
        confluence = settings.analysis.confluence
        weights = confluence.effective_weights(TradingMode.MICRO_LIVE)

        assert "market_structure" not in confluence.live_enabled_modules
        assert confluence.weights.get("market_structure", 0.0) > 0.0
        assert weights.get("market_structure", 0.0) == 0.0

    def test_measuring_a_section_does_not_make_it_live(self) -> None:
        """THE LINE THAT MUST NOT BE CROSSED. The grant lives on a settings
        COPY used only by the measurement; the account's own allowlist is
        untouched, and no broker ever sees the copy."""
        from scripts.dry_run_sections import _retimed

        settings = self._settings()
        before = tuple(settings.analysis.confluence.live_enabled_modules)

        tuned = _retimed(settings, "market_structure", "M15")

        assert tuple(settings.analysis.confluence.live_enabled_modules) == before
        assert "market_structure" not in before
        assert "market_structure" in tuned.analysis.confluence.live_enabled_modules

    def test_the_grant_is_additive_not_a_replacement(self) -> None:
        """Replacing the list would silence order_block in impulse_retest's
        pass, which changes what the confluence sees."""
        from scripts.dry_run_sections import _retimed

        settings = self._settings()
        before = set(settings.analysis.confluence.live_enabled_modules)

        tuned = _retimed(settings, "market_structure", "M15")
        allowed = set(tuned.analysis.confluence.live_enabled_modules)

        assert allowed == before | {"market_structure"}

    def test_a_shadowed_section_is_granted_for_its_measurement_pass(self) -> None:
        from scripts.dry_run_sections import _retimed

        settings = self._settings()
        before = set(settings.analysis.confluence.live_enabled_modules)

        tuned = _retimed(settings, "order_block", "M30")

        assert "order_block" not in before
        assert set(settings.analysis.confluence.live_enabled_modules) == before
        assert set(tuned.analysis.confluence.live_enabled_modules) == before | {"order_block"}


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

    def test_the_sweep_launcher_defaults_to_the_four_wide_clocks(self) -> None:
        """No clocks named: the same four as before, and M1 history off.

        M1 bars are the expensive fetch and nothing in M15..H4 is resolved on
        anything finer than M5, so `--no-m1` is right here and only here."""
        from scripts.dry_run_sections import build_parser

        launcher = (ROOT / "sweep.cmd").read_text()
        parsed = build_parser().parse_args(
            cmd_argv(
                launcher,
                **{
                    "%DAYS%": "180",
                    "%CLOCKS%": "M15 M30 H1 H4",
                    "%FINE%": "--no-m1",
                    "%CSVFILE%": "runtime\\sweep-180d-M15-M30-H1-H4.csv",
                },
            )
        )

        assert parsed.days == 180
        assert parsed.core is True and parsed.no_m1 is True
        assert parsed.sweep == ["M15", "M30", "H1", "H4"]

    def test_the_sweep_launcher_takes_m1_and_m5_and_turns_the_fetch_back_on(self) -> None:
        """`sweep.cmd 14 M1 M5`.

        THE CLOCKS ARE SEPARATE ARGUMENTS. `--sweep` is nargs="*", so a
        `%CLOCKS%` that arrived as one argument "M1 M5" would parse without
        complaint and then be looked up as a timeframe named "M1 M5". The
        helper expands the way cmd does, so this test can tell.

        And `%FINE%` MUST BE EMPTY HERE. `--no-m1` with an M1 clock asks for a
        clock finer than the frame it would be resolved on, which is not a
        slow run or a coarse one -- it is a row that silently does not exist.
        The launcher's `for` loop clears the flag; the script raises if it
        somehow arrives anyway.
        """
        from scripts.dry_run_sections import build_parser

        launcher = (ROOT / "sweep.cmd").read_text()
        parsed = build_parser().parse_args(
            cmd_argv(
                launcher,
                **{
                    "%DAYS%": "14",
                    "%CLOCKS%": "M1 M5",
                    "%FINE%": "",
                    "%CSVFILE%": "runtime\\sweep-14d-M1-M5.csv",
                },
            )
        )

        assert parsed.days == 14
        assert parsed.sweep == ["M1", "M5"]
        assert parsed.no_m1 is False

    def test_the_launcher_clears_no_m1_for_exactly_the_two_fine_clocks(self) -> None:
        """The `for` loop in the .cmd is the thing being read here. A launcher
        that kept `--no-m1` while asking for M5 would produce a sweep with the
        M5 row missing and no message, which is the failure this file has
        produced six times under different names."""
        launcher = (ROOT / "sweep.cmd").read_text().upper()

        assert 'IF /I "%%C"=="M1" SET FINE=' in launcher
        assert 'IF /I "%%C"=="M5" SET FINE=' in launcher

    def test_m1_and_m5_are_both_sweepable_now(self) -> None:
        from core.types import Timeframe
        from scripts.dry_run_sections import NEEDED, SWEEPABLE

        assert Timeframe.M1 in SWEEPABLE
        assert Timeframe.M5 in SWEEPABLE
        # Everything sweepable must also be fetched, or the clock is asked for
        # and then dropped for want of the frame it needs.
        assert set(SWEEPABLE) <= set(NEEDED)

    def test_the_m1_row_carries_its_own_caveat(self, capsys) -> None:
        """M1 is the one clock resolved on its own bars, so its number is not
        comparable to the rest of the table. Printing it beside them without
        saying so invites exactly the comparison it cannot support."""
        from scripts.dry_run_sections import _sweep_report

        results = self._results(-1.0, 1.0)
        results[("order_block", "M1")] = results.pop(("order_block", "M30"))
        _sweep_report(results, equity=215.0, days=14, managed=True)
        out = capsys.readouterr().out

        assert "M1 IS RESOLVED ON ITS OWN BARS" in out
        assert "counted as a LOSS" in out

    def test_no_caveat_when_no_m1_row_was_measured(self, capsys) -> None:
        from scripts.dry_run_sections import _sweep_report

        _sweep_report(self._results(-1.0, 1.0), equity=215.0, days=14, managed=True)

        assert "M1 IS RESOLVED ON ITS OWN BARS" not in capsys.readouterr().out


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
        assert "busy: dict = {row[0]: None for row in sections}" in SOURCE
        assert "busy[name] = freed" in " ".join(SOURCE.split())

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
        assert Timeframe.M5 in read, "the confluence engine reads M5"
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

    def test_the_launcher_quotes_a_measured_time_not_a_guessed_one(self) -> None:
        """Three estimates were given for this run -- ninety minutes, then
        ten, then six -- and the truth was forty-five. The launcher stopped
        promising a single number and now quotes the measurement it has (100
        days, four clocks, 16 markets, 45 minutes) plus the thing that scales:
        bars. A reader can then work out M1 for themselves and get the right
        answer instead of a reassuring one."""
        launcher = (ROOT / "sweep.cmd").read_text()

        assert "1.5 hours" not in launcher
        assert "ten to twenty minutes" not in launcher
        assert "45 minutes" in launcher
        assert "bars per market per day" in launcher
        assert "six hours" in launcher, "the M1 cost has to be stated before it is paid"


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


class TestTheWhyLauncher:
    """ "68 analysed, 0 opened" invites exactly one question and answers none
    of it. A night with no trades because nothing set up and a night with no
    trades because a gate is broken look identical from the outside and need
    opposite responses.

    `scripts/why_no_trades.py` has answered this from the journal all along
    and had no launcher, so it was never reached from the machine that has the
    journal on it."""

    LAUNCHER = (ROOT / "waarom.cmd").read_text()

    def test_its_command_line_parses(self) -> None:
        import importlib

        module = importlib.import_module("scripts.why_no_trades")
        parser = module.build_parser() if hasattr(module, "build_parser") else None
        if parser is None:  # the script builds its parser inside main()
            import re as _re

            source = (ROOT / "scripts" / "why_no_trades.py").read_text()
            flags = set(_re.findall(r'add_argument\("(--[a-z-]+)"', source))
            assert {"--hours", "--symbol"} <= flags
            return
        parsed = parser.parse_args(["--hours", "24"])
        assert parsed.hours == 24

    def test_it_takes_no_comma_arguments(self) -> None:
        for line in self.LAUNCHER.splitlines():
            if line.strip().startswith("set "):
                assert "," not in line, line

    def test_it_states_what_the_measurement_expects(self) -> None:
        """Without the expected rate on screen, "0 trades" cannot be judged.
        Seven a day means zero overnight is a finding; 0.1 a day would mean it
        is nothing."""
        assert "about 7 trades a day" in self.LAUNCHER
        assert "ZERO over a full session is not" in self.LAUNCHER


class TestTheRunSaysWhatItDidNotMeasure:
    """414 setups formed live in 24 hours and 0 trades were taken.

    Eleven of those died at something this script models. The other 403 died
    at eight gates it has never applied -- they live in `runner/service.py`
    and `filters/`, and nothing here reaches them.

    So "+36.80 R over 100 days" and "0 trades overnight" were never in
    conflict. They are two different systems, and the report gave no way to
    know that from the screen it printed.
    """

    def test_every_gate_named_is_one_the_sizer_cannot_raise(self) -> None:
        """A gate this script DOES apply must not be listed as missing --
        that would be the same lie in the other direction."""
        from risk.reasons import Reason
        from scripts.dry_run_sections import NOT_MODELLED

        source = inspect.getsource(__import__("risk.position_sizer", fromlist=["x"]))
        for name, _count, _why in NOT_MODELLED:
            assert hasattr(Reason, name), f"{name} is not a real refusal reason"
            assert f"Reason.{name}" not in source, f"the sizer can raise {name}; it IS modelled"

    def test_every_gate_named_is_one_the_runner_really_raises(self) -> None:
        """The list is only worth printing if it is true. Each reason has to
        appear in the live path this script skips."""
        from pathlib import Path

        from scripts.dry_run_sections import NOT_MODELLED

        live_path = (Path(ROOT) / "runner" / "service.py").read_text()
        filters = "\n".join(path.read_text() for path in (Path(ROOT) / "filters").glob("*.py"))
        for name, _count, _why in NOT_MODELLED:
            assert f"Reason.{name}" in live_path + filters, f"nothing raises {name}"

    def test_it_prints_the_warning_with_the_arithmetic(self, capsys) -> None:
        from scripts.dry_run_sections import _gates_this_run_does_not_apply

        _gates_this_run_does_not_apply()
        out = capsys.readouterr().out

        assert "WHAT THIS RUN DID NOT MODEL" in out
        assert "AWAITING_CONFIRMATION" in out
        assert "414 setups" in out
        assert "403 died at the gates above" in out
        assert "not a forecast of the account" in out

    def test_the_counts_add_up_to_the_number_quoted(self) -> None:
        """403 + 11 = 414. A funnel that does not close is a funnel with a
        stage missing, and this file has shipped one of those before."""
        from scripts.dry_run_sections import NOT_MODELLED

        assert sum(count for _n, count, _w in NOT_MODELLED) == 403


class TestATakenTradePaysItsOwnSpread:
    """The largest silent error this file has carried.

    `_resolve` walks raw highs and lows against raw entry, stop and target, so
    a winner returned the full reward and a loser exactly -1.00R. The cost
    model existed the whole time -- `_hopeless_on_cost` skipped markets on it,
    `_cost_report` printed it, the sizer refused 982 setups on it in one run --
    and none of it touched the money of a trade that was TAKEN. The survivors
    were paid as though trading were free.

    It is not a uniform haircut. `cost_share` divides the round trip by the
    STOP DISTANCE, and the stop is one ATR of the clock, so the same spread is
    about 1% of an H4 stop and 12% of an M1 one. The error grew as the clock
    shrank -- and the cell that then looked best, order_block on M1 at +34.00R
    over 105 trades, is exactly where it was largest.
    """

    def test_the_charge_is_subtracted_from_both_columns(self) -> None:
        """Both, not one. The report reads `managed_r` and the sweep prints
        `result_r` beside it; charging only one would make the gap between
        them read as the value of the stop rule."""
        from scripts import dry_run_sections

        source = " ".join(inspect.getsource(dry_run_sections).split())

        assert "cost = sizer.cost_share(spec, abs(idea.entry - idea.stop_loss), spread_price)" in (
            source
        )
        assert "r = None if r is None else r - cost" in source
        assert "managed_r = None if managed_r is None else managed_r - cost" in source

    def test_it_uses_the_sizer_s_own_definition(self) -> None:
        """One definition of a cost. A second one here would eventually
        disagree with the one that refuses trades, and the disagreement would
        be silent."""
        from risk.position_sizer import PositionSizer
        from scripts import dry_run_sections

        assert hasattr(PositionSizer, "cost_share")
        assert "PositionSizer._cost_share" not in inspect.getsource(
            dry_run_sections
        ), "call the public helper, do not reach past it"

    def test_the_gross_number_stays_recoverable(self) -> None:
        """Every figure produced before 31 August was gross. The difference
        has to stay visible rather than being quietly absorbed."""
        from scripts import dry_run_sections
        from scripts.dry_run_sections import Decision

        assert "cost_r" in Decision.__dataclass_fields__
        assert "cost_r_charged" in inspect.getsource(dry_run_sections)

    def test_charging_it_once_matches_what_the_sizer_says_a_loss_costs(self) -> None:
        """The sizer refuses with "a stop-out would cost about 1+cost_share R
        rather than 1.00R". So the charge is ONE cost_share, not two --
        `cost_share` is already the round trip."""
        sizer_source = " ".join(
            inspect.getsource(__import__("risk.position_sizer", fromlist=["x"])).split()
        )

        assert "1 + cost_share:.2f}R rather than 1.00R" in sizer_source
        # -1R gross becomes -(1 + cost) net, which is r - cost.
        assert pytest.approx(-1.0 - 0.12) == -1.0 - 0.12

    def test_the_sweep_csv_is_named_after_the_run(self) -> None:
        """It was always runtime\\sweep.csv, so a 14-day M1+M5 run destroyed
        the 100-day M15..H4 one and four clocks of trades were simply gone."""
        launcher = (ROOT / "sweep.cmd").read_text()
        invocation = next(
            line for line in launcher.splitlines() if "scripts.dry_run_sections" in line
        )

        assert "runtime\\sweep.csv" not in invocation
        assert "sweep-%DAYS%d-%TAG%.csv" in launcher


class TestALiveSectionOnM1DoesNotVanishFromItsOwnReport:
    """`--no-m1` and a live section configured on M1 are not a contradiction.

    `--no-m1 --sweep M1` is: the user asked for two incompatible things on one
    command line, and the run stops and says so.

    `--no-m1 --live-only` when a LIVE section sits on M1 is different. That
    clock came out of config/eightcap.yaml, not off the command line. Dying on
    it breaks the launcher that answers "what would the account have done" the
    moment a section moves to M1, and dropping the row silently is the missing
    row this file has shipped six times. The flag loses, loudly.
    """

    def test_an_explicit_sweep_still_refuses(self) -> None:
        from core.types import Timeframe
        from scripts.dry_run_sections import _unresolvable_clocks

        assert _unresolvable_clocks(("M1",), Timeframe.M5).startswith("M1 cannot be resolved")

    def test_the_script_prefers_fetching_over_dropping(self) -> None:
        from scripts import dry_run_sections

        source = " ".join(inspect.getsource(dry_run_sections).split())

        assert "if needs_m1 and args.sweep: raise SystemExit(needs_m1)" in source
        assert "args.no_m1 = False" in source
        assert "--no-m1 ignored" in source

    def test_fetch_these_is_decided_after_that(self) -> None:
        """Order matters and it is the whole fix: `fetch_these` used to be
        built from `args.no_m1` a hundred lines before `passes` existed, so
        flipping the flag afterwards would have changed nothing."""
        from scripts import dry_run_sections

        source = inspect.getsource(dry_run_sections)

        assert source.index("args.no_m1 = False") < source.index("required_frames =")

    def test_the_m1_section_is_measurable_at_all(self) -> None:
        """`module_config` is the list of sections this script knows. A live
        section absent from it cannot appear in the live-configuration report,
        which is the same silence in a new place."""
        from scripts import dry_run_sections

        source = inspect.getsource(dry_run_sections)

        assert '"order_block_fast": "order_block_fast"' in source


class TestTheLiveDryRunMeasuresOnlyWhatRuns:
    """ "Ik wilde alleen degene die runned meten, niet die shadow."

    `--live-only` already intersected with `live_enabled_modules`.
    `--sections-five-to-ten` did not: it carried its own hardcoded set, written
    when the book had six entries. Section five and section nine came off the
    allowlist on 2 September (-1.09 R over 170 trades, -0.02 R over 6) and the
    hardcoded copy still named them, so `dryrun-live.cmd` -- which passes both
    flags -- would have gone on replaying two sections that cannot trade.

    Two lists that must agree are two lists that will disagree. This one is
    derived now.
    """

    def _settings(self):
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

    def test_the_book_is_intersected_with_the_live_allowlist(self) -> None:
        from scripts import dry_run_sections

        source = " ".join(inspect.getsource(dry_run_sections).split())

        assert "measured = measured & book & live" in source
        assert "off the live allowlist" in source, "a benched section is named, not dropped"

    def test_a_benched_section_is_bounded_the_same_way_a_live_one_is(self) -> None:
        """THE PROPERTY, NOT WHICH TWO SECTIONS ARE BENCHED THIS WEEK.

        This asserted that `section_five_ndx100_m5` and `section_nine_vwap_m30`
        were both off the allowlist. That is a DECISION -- it was taken on
        2 September and reversed for section five on 6 September, on 546 trades
        and +EUR 181.75 over 180 days -- and the test failed while nothing was
        wrong. Same defect as the frozen allowlist this class already documents
        one method below.

        What has to hold whichever way the owner decides: a section either has
        real-money permission AND everything permission requires (a weight, a
        breaker, a broker label), or it has none of it. The half-promoted state
        is the dangerous one.
        """
        from core.trade_origin import origin_for_setup_family
        from core.types import TradingMode

        settings = self._settings()
        confluence = settings.analysis.confluence
        live = set(confluence.live_enabled_modules)
        effective = confluence.effective_weights(TradingMode.MICRO_LIVE)
        assert live, "the account is live with no sections at all"
        for name in sorted(live):
            assert effective.get(name, 0.0) > 0.0, f"{name} is live and votes on nothing"
            assert name in settings.risk.section_breakers, f"{name} is live with no breaker"
            assert origin_for_setup_family(name) is not None, f"{name} is live with no MT5 label"

    def test_section_six_is_live_and_bounded_by_its_breaker(self) -> None:
        """S6 came off on 3 September and went back on the same day.

        The measurement that took it off is not disputed and is not softened
        here: 885 trades over 180 days, 24.5% win, -71.65 R. The recent +43.90R
        was one regime.

        It is back because two things changed under it -- a causal 12-bar M5
        confirmation, and position management finally reaching it at all (it
        sat on `fixed_exit_comments`, so the manager returned early and no
        break-even ever ran) -- and because the owner asked to forward-test
        that combination. Its section breaker is what bounds it, and THAT is
        what this asserts. An earlier version pinned the whole allowlist as a
        frozen set of four, so promoting a fifth section failed here with
        nothing wrong -- a test that pins a decision instead of a property.
        """
        settings = self._settings()
        live = set(settings.analysis.confluence.live_enabled_modules)

        from core.trade_origin import origin_for_setup_family

        assert "section_six_gold_m5" in live
        assert "section_six_gold_m5" in settings.risk.section_breakers

        origin = origin_for_setup_family("section_six_gold_m5")
        assert origin is not None
        assert origin.comment not in settings.trade_management.fixed_exit_comments, (
            "section six is measured WITH break-even management; a fixed-exit entry "
            "makes the manager return early and no break-even ever runs"
        )

    def test_every_live_section_still_has_a_breaker(self) -> None:
        settings = self._settings()

        for module in settings.analysis.confluence.live_enabled_modules:
            assert module in settings.risk.section_breakers, module

    def test_every_live_section_is_measurable_by_this_script(self) -> None:
        """`dryrun-live.cmd 180` DIED ON THIS, and the guard that killed it was
        right: section eleven went on the live allowlist and was never added to
        this script's table, so the run that was supposed to judge it before
        Jarvis starts could not run at all.

        The guard exists and did its job. This test is what stops the same
        promotion from reaching the VPS in that state again -- a live section
        this script cannot see is a live section the "what would the account
        have done" report answers about in silence.
        """
        import re

        live = set(self._settings().analysis.confluence.live_enabled_modules)
        table = re.search(r"module_config = \{(.+?)\n        \}", SOURCE, re.S)
        assert table is not None, "module_config is no longer a literal dict"
        known = set(re.findall(r'"([a-z0-9_]+)":', table.group(1)))
        book = re.search(r"book = \{(.+?)\n            \}", SOURCE, re.S)
        assert book is not None, "the --sections-five-to-ten book is no longer a literal set"
        booked = set(re.findall(r'"([a-z0-9_]+)"', book.group(1)))

        assert not live - known, sorted(live - known)
        assert not live - booked, sorted(live - booked)

    def test_a_live_section_walks_the_markets_it_can_trade(self) -> None:
        """The quieter half of the same failure, and it hit TWO live sections.

        `--core` is sixteen markets. Section ten's `allowed_symbols` is five
        metals, section eleven's is four crosses, and only XAUUSD of those nine
        is in core -- so this run would have replayed section ten on one of its
        five markets and section eleven on NONE of its four, then reported the
        result as the book. Section eleven's zero row would have read as "the
        strategy found nothing".
        """
        assert "symbols = symbols + absent" in SOURCE
        assert "markets the universe missed" in SOURCE

    def test_every_section_declares_its_markets_in_a_way_the_widening_reads(self) -> None:
        """A PROPERTY, NOT THE LINE THAT IMPLEMENTS IT.

        The previous version of this asserted the literal source string
        `getattr(getattr(settings.analysis, name), "allowed_symbols", ())`.
        That pinned a DECISION rather than a property, and it did the damage
        such a test always does: sections five to ten and the BTC three carry
        `allowed_symbols`, but the XAUJPY sections carry a single `symbol`, so
        the pinned line returned an empty tuple for all three of them. The
        widening the test was guarding did not happen for the sections it was
        written to guard, and the test passed the whole time.

        So this asks the question the widening asks -- "which markets does this
        section need" -- of every section the dry run can measure, and fails
        when one of them answers with nothing.
        """
        import re

        settings = self._settings()
        table = re.search(r"module_config = \{(.+?)\n        \}", SOURCE, re.S)
        assert table is not None
        for name in sorted(set(re.findall(r'"([a-z0-9_]+)":', table.group(1)))):
            config = getattr(settings.analysis, name, None)
            if config is None:
                continue
            allowed = tuple(getattr(config, "allowed_symbols", ()) or ())
            one = getattr(config, "symbol", "")
            declared = allowed or ((one,) if one else ())
            if not declared and not hasattr(config, "allowed_symbols"):
                # A section with neither field trades the whole universe on
                # purpose; there is nothing to widen for it.
                continue
            assert declared, (
                f"{name} declares no market the dry run can add to the walk, so a "
                f"--core run replays it on nothing and reports the empty result as a zero"
            )

    def test_the_launcher_passes_both_flags(self) -> None:
        launcher = (ROOT / "dryrun-live.cmd").read_text()

        assert "--live-only" in launcher
        assert "--sections-five-to-ten" in launcher


class TestASectionThatTookNothingStillHasARow:
    """ "Waar is sectie 10?"

    Section ten was on the live allowlist, ran, took no trades, and therefore
    had no rows to group -- so it vanished from BY SECTION, the one table the
    owner actually reads. An absent row and a zero row look identical and mean
    opposite things: one is a section that found nothing, the other is a
    section that is not wired in.

    That is the seventh time this project has shipped that same confusion.
    """

    def _decisions(self):
        from datetime import datetime, timedelta

        from scripts.dry_run_sections import Decision

        base = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
        rows = [
            Decision(
                base + timedelta(minutes=i),
                "XAUUSD",
                "section_six_gold_m5",
                "TRADE",
                result_r=1.0,
                pnl_money=8.0,
                managed_r=1.0,
                managed_money=8.0,
                pass_key=("section_six_gold_m5", "M5"),
            )
            for i in range(5)
        ]
        # Section ten formed setups and the sizer refused every one.
        rows += [
            Decision(
                base + timedelta(minutes=i),
                "XAUUSD",
                "section_ten_gold_m1",
                "SL_TOO_TIGHT_FOR_COSTS",
                pass_key=("section_ten_gold_m1", "M1"),
            )
            for i in range(40)
        ]
        return rows

    def test_a_section_with_no_trades_is_named_with_its_reason(self, capsys) -> None:
        from scripts.dry_run_sections import _report

        _report(
            self._decisions(),
            equity=216.0,
            days=30,
            skipped=0,
            managed=True,
            sections=("section_six_gold_m5", "section_ten_gold_m1"),
        )
        out = capsys.readouterr().out

        assert "section_ten_gold_m1" in out, "a live section may not vanish from BY SECTION"
        assert "0 trades" in out
        assert "SL_TOO_TIGHT_FOR_COSTS" in out, "say WHERE it died, not just that it did"

    def test_a_section_with_no_decisions_at_all_reads_differently(self, capsys) -> None:
        """Took nothing and never ran need opposite responses."""
        from scripts.dry_run_sections import _report

        _report(
            self._decisions(),
            equity=216.0,
            days=30,
            skipped=0,
            managed=True,
            sections=("section_six_gold_m5", "section_ten_gold_m1", "never_built"),
        )
        out = capsys.readouterr().out

        assert "NO DECISIONS AT ALL" in out
        assert "never_built" in out

    def test_refusals_carry_the_section_that_was_refused(self) -> None:
        """Without this there are no rows anywhere bearing a silent section's
        name, and the table above cannot be written at all."""
        from scripts import dry_run_sections

        source = " ".join(inspect.getsource(dry_run_sections).split())

        assert (
            source.count("pass_key=(name, clock.value)") >= 3
        ), "trade rows, confluence refusals and sizer refusals all need it"

    def test_the_run_passes_its_measured_sections_in(self) -> None:
        from scripts import dry_run_sections

        source = " ".join(inspect.getsource(dry_run_sections).split())

        assert "sections=tuple(sorted({name for name, _tf in passes}))" in source


class TestTheExitsAreNotModelledEither:
    """What the account replay applies after entry, and what it still cannot.

    THIS CLASS USED TO ASSERT THE OPPOSITE OF THE TRUTH. It required
    `partial_close_at_r` to appear in the "not modelled" list -- and
    `--jarvis-replay` hands `_resolve` the live `TradeManagementConfig`, which
    walks partials, the ATR trail, the profit lock, give-back, peak-stall and
    the time exit bar by bar. Six rules the replay applies were being disowned
    on screen, and the test was holding that mislabel in place.

    That is the repository's most repeated defect running backwards: instead of
    claiming a check it does not perform, the report denied six it does. It
    made the measurement read as far cruder than it is.
    """

    def test_every_named_rule_is_real_in_both_lists(self) -> None:
        """A list of rules that do not exist would be worse than no list."""
        from config.loader import DEFAULT_CONFIG_PATH, load_settings
        from scripts.dry_run_sections import (
            EXITS_MODELLED_UNDER_FULL_MANAGEMENT,
            EXITS_NOT_MODELLED,
        )

        settings = load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )
        management = settings.trade_management

        for name, _what in (*EXITS_NOT_MODELLED, *EXITS_MODELLED_UNDER_FULL_MANAGEMENT):
            field = name.split()[0].rstrip("*")
            assert hasattr(management, field), f"{field} is not a real setting"

    def test_no_rule_is_claimed_as_both_applied_and_missing(self) -> None:
        from scripts.dry_run_sections import (
            EXITS_MODELLED_UNDER_FULL_MANAGEMENT,
            EXITS_NOT_MODELLED,
        )

        applied = {name.split()[0] for name, _ in EXITS_MODELLED_UNDER_FULL_MANAGEMENT}
        missing = {name.split()[0] for name, _ in EXITS_NOT_MODELLED}
        assert not (applied & missing)

    def test_the_rules_resolve_actually_reads_are_on_the_applied_side(self) -> None:
        """`_resolve`'s full-management branch reads these fields by name. If a
        field it reads is listed as NOT modelled, the contract is lying about
        the number printed under it."""
        from scripts.dry_run_sections import EXITS_NOT_MODELLED

        body = SOURCE.split("def _resolve(", 1)[1].split("\ndef ", 1)[0]
        for name, _what in EXITS_NOT_MODELLED:
            field = name.split()[0]
            assert f"full_management.{field}" not in body, (
                f"{field} is listed as not modelled and _resolve reads it"
            )

    def test_break_even_is_modelled_and_is_not_in_the_missing_list(self) -> None:
        from scripts.dry_run_sections import EXITS_NOT_MODELLED

        named = " ".join(name for name, _ in EXITS_NOT_MODELLED)
        assert "break_even_at_r" not in named
        assert "partial_close_at_r" not in named, (
            "the replay walks the partial; listing it as missing understates the run"
        )

    def test_the_report_prints_both_sides(self, capsys) -> None:
        """Both lists, and on the right side of the sentence. Printing only the
        missing three would understate the run; printing only the applied six
        would overstate it."""
        from scripts.dry_run_sections import (
            EXITS_MODELLED_UNDER_FULL_MANAGEMENT,
            EXITS_NOT_MODELLED,
            _gates_this_run_does_not_apply,
        )

        _gates_this_run_does_not_apply()
        out = capsys.readouterr().out.casefold()

        assert "and the exits" in out
        assert "fixed-exit" in out
        applied, missing = out.split("still cannot apply", 1)
        for name, _what in EXITS_MODELLED_UNDER_FULL_MANAGEMENT:
            field = name.split()[0].casefold()
            assert field in applied, f"{field} is applied and is not printed as applied"
        for name, _what in EXITS_NOT_MODELLED:
            field = name.split()[0].casefold()
            assert field in missing, f"{field} is missing and is not printed as missing"
        # The within-bar ordering is the honest remaining caveat and has to stay.
        assert "unknowable" in out


class TestTheBreakEvenGridComparesExitsAndNotEntries:
    """`--manage-grid` answers one narrow question and must not overstate it.

    The owner asked whether a LOOSER break-even -- one that only arms after
    the trade has already run a long way -- would beat section ten's fixed
    stop and target. That cannot be settled by switching break-even on and
    comparing two runs: an earlier exit frees the symbol, a freed symbol
    takes the next setup, and the two runs then hold different trades. The
    comparison silently becomes "these entries against those entries".

    So every level is resolved on the SAME entry, and these tests pin that.
    """

    def _row(self, when, fixed, grid_values):
        from scripts.dry_run_sections import MANAGE_GRID, Decision

        return Decision(
            when,
            "XAUUSD",
            "section_ten_gold_m1",
            "TRADE",
            result_r=fixed,
            grid_r=tuple(
                (variant.label, value)
                for variant, value in zip(MANAGE_GRID, grid_values, strict=True)
            ),
        )

    def _rows(self, count=40):
        from datetime import UTC, datetime, timedelta

        from scripts.dry_run_sections import MANAGE_GRID

        base = datetime(2026, 3, 1, tzinfo=UTC)
        made = []
        for index in range(count):
            fixed = 1.5 if index % 3 == 0 else -1.0
            # Break-even scratches every loser and half the winners.
            managed = 0.0 if fixed < 0 or index % 6 == 0 else fixed
            made.append(
                self._row(base + timedelta(days=index), fixed, [managed] * len(MANAGE_GRID))
            )
        return made

    def test_every_level_is_reported_beside_the_fixed_baseline(self, capsys) -> None:
        from scripts.dry_run_sections import MANAGE_GRID, _manage_grid_report

        _manage_grid_report(self._rows())
        out = capsys.readouterr().out

        assert "fixed SL/TP" in out, "the baseline the levels are judged against is missing"
        for variant in MANAGE_GRID:
            assert variant.label in out, f"{variant.label} was measured and not printed"

    def test_both_halves_of_the_period_are_shown(self, capsys) -> None:
        """Picking the best of seven columns on one sample is how this project
        has produced most of its disappointments."""
        from scripts.dry_run_sections import _manage_grid_report

        _manage_grid_report(self._rows())
        out = capsys.readouterr().out

        assert "early R" in out and "late R" in out
        assert "early /" in out or "early" in out

    def test_a_run_without_the_flag_prints_nothing(self, capsys) -> None:
        """The grid is opt-in, and a section with no grid rows is not a section
        whose grid was all zeros."""
        from datetime import UTC, datetime

        from scripts.dry_run_sections import Decision, _manage_grid_report

        plain = [
            Decision(
                datetime(2026, 3, 1, tzinfo=UTC),
                "XAUUSD",
                "section_ten_gold_m1",
                "TRADE",
                result_r=1.0,
            )
        ]
        _manage_grid_report(plain)
        assert capsys.readouterr().out == ""

    def test_each_section_gets_its_own_table(self, capsys) -> None:
        """Whether break-even pays is a property of the section, and averaging
        section six and section ten together answers neither."""
        from datetime import UTC, datetime, timedelta

        from scripts.dry_run_sections import MANAGE_GRID

        rows = self._rows()
        base = datetime(2026, 3, 1, tzinfo=UTC)
        for index in range(20):
            row = self._row(base + timedelta(days=index), -1.0, [0.0] * len(MANAGE_GRID))
            row.module = "section_six_gold_m5"
            rows.append(row)

        from scripts.dry_run_sections import _manage_grid_report

        _manage_grid_report(rows)
        out = capsys.readouterr().out

        assert "section_ten_gold_m1" in out and "section_six_gold_m5" in out
        # Two tables, so two baselines.
        assert out.count("fixed SL/TP") == 2

    def test_the_triggers_are_in_r_and_not_in_pips(self) -> None:
        """ "Fifty pips toward the target" is a different rule on every
        instrument and on every day. On this account's gold M1 stop it is
        roughly ten times the risk -- past the target, so it would never fire
        and the column would read as "break-even does not help"."""
        from scripts.dry_run_sections import MANAGE_GRID

        assert MANAGE_GRID
        moves = [v for v in MANAGE_GRID if v.kind == "break-even"]
        assert moves, "the grid compares no break-even level at all"
        for variant in moves:
            assert 0.0 < variant.trigger_r <= 2.0, variant.label
            # The lock is where the stop GOES; past the trigger it would sit
            # beyond the price that armed it.
            assert 0.0 <= variant.lock_r < variant.trigger_r, variant.label
            assert variant.lock_atr >= 0.0, variant.label
            assert not (variant.lock_r and variant.lock_atr), variant.label


class TestSectionTenRunsOnlyWhereItCanTrade:
    """A run that walks markets the section refuses on the first bar.

    The first section-ten replay spent 252 seconds on EURUSD and 505 on
    GBPUSD before reaching a single metal, with an hour still to go, and
    section ten cannot open a trade in either -- `allowed_symbols` refuses
    them immediately. Sixteen markets to measure six is an hour of walking
    bars to watch a symbol filter say no.
    """

    def test_the_universe_shrinks_to_the_sections_own_symbols(self) -> None:
        from config.loader import load_settings

        settings = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        )
        allowed = set(settings.analysis.section_ten_gold_m1.allowed_symbols)
        universe = ["EURUSD.i", "GBPUSD.i", "XAUUSD", "SPX500", "XAUEUR", "US30"]

        kept = [name for name in universe if name in allowed]

        assert "XAUUSD" in kept
        assert "EURUSD.i" not in kept and "SPX500" not in kept

    def test_section_ten_is_back_to_gold_alone(self) -> None:
        """The widening of 3 September was reverted on 4 September, measured.

        Full 180-day replay, per market:

            XAUUSD  569 trades  +75.91 R  +0.133 each, positive in BOTH halves
            XAUJPY  422 trades  +15.57 R  +0.037   all of it in the late half
            XAUEUR  459 trades   +5.09 R  +0.011   all of it in the late half
            XAUGBP  466 trades  -16.76 R  -0.036
            XAUAUD  550 trades  -40.12 R  -0.073

        The four crosses together lost 36.22 R over 1,897 trades, and the two
        that were positive were positive only in the second half -- the same
        regime-concentration shape that took section six off live.

        On top of that the position cap charged section six 244 trades and
        30.55 R, because it also trades XAUUSD and it is one position per
        symbol. The widening cost 66.77 R in total.

        XAUUSD on its own reads +0.133 R a trade here against +0.114 in the
        original engine replay. Two independent measurements of the same
        number. The section was not too narrow.
        """
        from config.loader import load_settings

        settings = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        )
        allowed = settings.analysis.section_ten_gold_m1.allowed_symbols

        assert "XAUUSD" in allowed
        # The crosses came back on 4 September, but only with 16:00 and 17:00
        # UTC shut. That -36.22 R was the average over ALL hours; per market
        # and per hour, 16:00 is negative in both halves in four crosses out
        # of four and positive in both halves on gold. With those two hours
        # closed the same 180 days read +53.03 R.
        #
        # So a cross on the live list without its own blocked hours is the
        # measurement being ignored, and that is what this checks.
        config = settings.analysis.section_ten_gold_m1
        for symbol in allowed:
            if symbol == "XAUUSD":
                continue
            assert config.hour_is_blocked(symbol, 16), (
                f"{symbol} trades at 16:00 UTC. Four crosses out of four lose "
                f"money in that hour in both halves of the period."
            )


class TestSectionTenCanExpressAnOpenTradingDay:
    """Equal blocked hours mean no blocked window, and that has to be sayable.

    The validator demanded `blocked_start < blocked_end`, so switching the
    07:00-13:00 block off had no expressible value: the nearest thing was a
    one-hour block at some quiet hour, which is a different setting that
    merely looks like "off". A question you cannot express is a question you
    cannot answer, and this one is open -- that window was chosen by cutting
    the worst six hours out of the same 180 days the section was calibrated
    on, which finds a bad block in any sequence.
    """

    def test_equal_hours_are_accepted_and_block_nothing(self) -> None:
        from config.schema import SectionTenGoldM1Config

        config = SectionTenGoldM1Config(
            entry_start_hour_utc=3,
            entry_end_hour_utc=19,
            blocked_start_hour_utc=3,
            blocked_end_hour_utc=3,
        )

        blocked = range(config.blocked_start_hour_utc, config.blocked_end_hour_utc)
        assert len(blocked) == 0
        assert not any(
            config.blocked_start_hour_utc <= hour < config.blocked_end_hour_utc
            for hour in range(24)
        )

    def test_a_block_outside_the_entry_window_is_still_refused(self) -> None:
        """Relaxing one bound must not relax the ordering itself."""
        import pytest as _pytest
        from pydantic import ValidationError

        from config.schema import SectionTenGoldM1Config

        with _pytest.raises(ValidationError):
            SectionTenGoldM1Config(
                entry_start_hour_utc=3,
                entry_end_hour_utc=19,
                blocked_start_hour_utc=1,
                blocked_end_hour_utc=5,
            )
        with _pytest.raises(ValidationError):
            SectionTenGoldM1Config(
                entry_start_hour_utc=3,
                entry_end_hour_utc=19,
                blocked_start_hour_utc=13,
                blocked_end_hour_utc=7,
            )

    def test_the_live_config_blocks_the_hours_the_replay_condemned(self) -> None:
        """The window went open on 3 September and shut again the same day.

        I argued the block was probably fitted noise -- the worst six hours of
        any sequence are bad by construction. That is a fair general objection
        and it was simply wrong here. `sectie10.cmd 180 goud`, 989 trades with
        the hours open:

            07:00-13:00 together   422 trades   -65.96 R   (-0.156 each)
            every other hour       567 trades   +73.03 R   (+0.129 each)

        Six of the six blocked hours negative, and 13:00 -- the first hour
        outside the block -- the best in the day at +0.523. A different
        measurement from the one that chose the window, landing on the same
        boundary.

        So this pins the hours, and it pins them with the number. Opening them
        again means beating that measurement, not repeating the argument.
        """
        from config.loader import load_settings

        config = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.section_ten_gold_m1

        assert config.blocked_start_hour_utc == 7
        assert config.blocked_end_hour_utc == 13


class TestSectionTenOnlyWalksTheSectionsOwnMarkets:
    """`--section-ten-only` replaces the universe; it does not filter it.

    The first version intersected section ten's `allowed_symbols` with the
    run's universe, and `--core` is the sixteen markets the research was done
    on: eleven FX majors, gold, four indices. Five of section ten's six
    metals are not in it, so the intersection was {XAUUSD}. The run printed
    "walking 1 markets", measured the one market that did not need measuring,
    and called itself a test of the widening to six.
    """

    def test_the_sections_own_list_survives_a_universe_that_lacks_it(self) -> None:
        allowed = ("XAUUSD", "XAUEUR", "XAUGBP", "XAUAUD", "XAUJPY", "XAGUSD")
        core = ["EURUSD.i", "GBPUSD.i", "XAUUSD", "SPX500", "US30"]

        # What the code does now: the section's list becomes the universe.
        walked = list(allowed)

        assert len(walked) == 6
        # And what it used to do, kept as the counterexample.
        intersected = [name for name in core if name in allowed]
        assert intersected == ["XAUUSD"], "the old behaviour was not a one-market run"
        assert len(walked) > len(intersected)

    def test_the_live_config_would_walk_the_metals_that_can_pay(self) -> None:
        """Whatever section ten may trade, gold is in it and silver is not.

        XAGUSD is out on cost: 830 setups and 0 trades over 180 days, over 20%
        of the stop. The crosses came off on result and came back with 16:00
        and 17:00 UTC shut, which is where their loss was.
        """
        from config.loader import load_settings

        allowed = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        ).analysis.section_ten_gold_m1.allowed_symbols

        assert "XAGUSD" not in allowed, "silver takes 0 trades; it only costs bar-walking time"
        assert len(allowed) >= 1
        assert "XAUUSD" in allowed


class TestTheLauncherEchoesSurviveCmd:
    """`<` and `>` are redirections in cmd, inside an echo line as well.

    An arrow in a help line -- "~ 1 uur  <- begin hier" -- made cmd try to
    read input from a file called `-`, so the launcher printed "The system
    cannot find the file specified" in the middle of its own instructions.
    Harmless to the run and alarming to read, which is the worst combination
    for a message whose whole job is to tell you what is happening.
    """

    def test_no_launcher_echo_carries_an_unescaped_redirect(self) -> None:
        import re
        from pathlib import Path

        offenders: list[str] = []
        for path in sorted(Path().glob("*.cmd")):
            for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                stripped = line.strip()
                if not stripped.lower().startswith("echo"):
                    continue
                # `echo %~1| findstr ... >nul` is deliberate plumbing: the
                # echo feeds a pipe and the redirect belongs to what is on the
                # other side of it. Only lines whose whole job is to print
                # text can be broken by an arrow in the text.
                if "|" in stripped:
                    continue
                # A redirect not preceded by a caret.
                if re.search(r"(?<!\^)[<>]", stripped):
                    offenders.append(f"{path.name}:{number}: {stripped}")
        assert not offenders, "unescaped < or > in an echo line:\n" + "\n".join(offenders)

    def test_no_launcher_puts_a_comment_inside_a_continued_command(self) -> None:
        """A `rem` between two `^` lines becomes PART of the command.

        cmd joins a line ending in `^` to the one after it, whatever that line
        is. So a comment written to explain a flag ends up passed as arguments,
        and the run dies with something that has nothing to do with the
        explanation. Caught while adding exactly such a comment to
        `sectie11.cmd`; the same shape as every other defect in this file --
        a thing that is correct in isolation and lands where it cannot work.
        """
        from pathlib import Path

        offenders: list[str] = []
        for path in sorted(Path().glob("*.cmd")):
            lines = path.read_text(errors="replace").splitlines()
            for number, line in enumerate(lines[:-1], 1):
                if not line.rstrip().endswith("^"):
                    continue
                nxt = lines[number].strip()
                if nxt.lower().startswith("rem ") or nxt.lower() == "rem" or not nxt:
                    offenders.append(f"{path.name}:{number + 1}: {nxt or '(blank line)'}")
        assert not offenders, "comment or blank line inside a ^ continuation:\n" + "\n".join(
            offenders
        )


class TestRawBtcShadowIsExplicitlyContained:
    def test_launcher_requests_the_raw_shadow_lane(self) -> None:
        from pathlib import Path

        launcher = Path("sectie151617.cmd").read_text(errors="replace")
        assert "--btc-research-parity" in launcher
        assert "section_fifteen_btc_m1,section_sixteen_btc_m5,section_seventeen_btc_m15" in launcher

    def test_flag_exists_and_is_off_by_default(self) -> None:
        from scripts.dry_run_sections import build_parser

        assert build_parser().parse_args([]).btc_research_parity is False
        assert build_parser().parse_args(["--btc-research-parity"]).btc_research_parity is True
        assert build_parser().parse_args(["--raw-btc-shadow"]).btc_research_parity is True

    def test_second_launcher_requests_the_jarvis_account_lane(self) -> None:
        from pathlib import Path

        from scripts.dry_run_sections import build_parser

        launcher = Path("sectie151617-jarvis.cmd").read_text(errors="replace")
        assert "--btc-jarvis-replay" in launcher
        assert "runtime\\secties15-17-jarvis.csv" in launcher
        parsed = build_parser().parse_args(["--btc-jarvis-replay"])
        assert parsed.btc_jarvis_replay is True
        assert parsed.btc_research_parity is False

    def test_raw_lane_is_hard_limited_to_the_three_btc_sections(self) -> None:
        from scripts.dry_run_sections import RAW_BTC_SHADOW_SECTIONS

        assert {
            "section_fifteen_btc_m1",
            "section_sixteen_btc_m5",
            "section_seventeen_btc_m15",
        } == RAW_BTC_SHADOW_SECTIONS


class TestTheHeartbeatFiresOnEveryWindowLength:
    """The progress line must not have a window length baked into it.

    The first version beat every 20,000 bars and only above 40,000, which
    silently assumed a 180-day run. A 30-day M1 window is about 29,000 bars --
    gold trades 23 hours on weekdays only -- so it fell under the floor and
    printed nothing for the whole walk. The threshold reproduced exactly the
    silence it was added to remove, and it did so on the SHORTER run, which is
    the one somebody reaches for because they do not want to wait.
    """

    def _beat(self, total_bars: int) -> int:
        return max(total_bars // 8, 2_000)

    def test_a_thirty_day_m1_window_still_reports(self) -> None:
        # ~21 trading days x 1,380 M1 bars.
        total = 29_000
        beat = self._beat(total)
        assert beat > 0
        assert total // beat >= 4, "a 30-day walk would print fewer than four times"

    def test_a_hundred_and_eighty_day_window_does_not_flood(self) -> None:
        total = 235_000
        beat = self._beat(total)
        assert 4 <= total // beat <= 12, "a 180-day walk should print a handful of times"

    def test_a_short_window_does_not_print_on_every_bar(self) -> None:
        """An H1 clock over 180 days is about 3,000 bars. One line is fine;
        three thousand is worse than silence."""
        total = 3_000
        assert total // self._beat(total) <= 1


class TestEveryLauncherSetsWhatItReads:
    """A MERGE ATE `sectie11.cmd`'S ARGUMENT BLOCK AND GIT REPORTED NO CONFLICT.

    Two sessions rewrote the same launcher. The hunks did not overlap, so git
    spliced them together: the banner and the python call survived, the
    `set DAGEN=180` / `:lees` block did not. The run printed its whole banner
    and then died on

        argument --days: expected one argument

    because an undefined `%DAGEN%` in cmd is SILENT -- it expands to nothing
    and vanishes out of the command line rather than raising. That is the
    house defect in yet another form: a value that is correct where it is
    written and absent where it is read.

    So: every `%VAR%` a launcher expands must be set, or inherited, before the
    line that reads it.
    """

    #: Set by cmd itself, or by the enclosing shell, and never by the script.
    _AMBIENT: ClassVar[set[str]] = {
        "TEMP",
        "TMP",
        "PATH",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "COMPUTERNAME",
        "USERNAME",
        "WINDIR",
        "SYSTEMROOT",
        "PROGRAMFILES",
        "CD",
        "DATE",
        "TIME",
        "RANDOM",
        "ERRORLEVEL",
    }

    def test_no_launcher_reads_a_variable_it_never_sets(self) -> None:
        import re
        from pathlib import Path

        offenders: list[str] = []
        for path in sorted(Path().glob("*.cmd")):
            lines = path.read_text(errors="replace").splitlines()
            assigned: set[str] = set()
            for number, line in enumerate(lines, 1):
                # `set NAME=`, `set /a NAME=`, and `for %%x in (...) do set NAME=`
                for match in re.finditer(
                    r"\bset\s+(?:/a\s+)?\"?([A-Za-z_][A-Za-z0-9_]*)\s*=", line
                ):
                    assigned.add(match.group(1).upper())
                # A `rem` line explains the code, it does not run it -- and the
                # comment recording THIS bug names %DAGEN% before it is set.
                if line.strip().lower().startswith("rem"):
                    continue
                # `%%` is an escaped literal percent inside a for loop or a
                # format string, not a variable. `%%Y%%m%%d` is a strftime
                # pattern and reading it as %Y% is how a checker cries wolf.
                scanned = line.replace("%%", "\x00")
                for name in re.findall(r"%([A-Za-z_][A-Za-z0-9_]*)%", scanned):
                    upper = name.upper()
                    if upper in self._AMBIENT or upper in assigned:
                        continue
                    offenders.append(f"{path.name}:{number}: %{name}% is read before any set")
        assert not offenders, "a launcher expands a variable nothing sets:\n" + "\n".join(offenders)

    def test_the_section_eleven_launcher_still_parses_its_arguments(self) -> None:
        """The exact block the merge removed, asserted by name so a future
        merge that drops it again fails here instead of on the VPS."""
        from pathlib import Path

        launcher = Path("sectie11.cmd").read_text()

        assert "set DAGEN=180" in launcher
        assert ":lees" in launcher and "goto lees" in launcher
        assert "--days %DAGEN%" in launcher
        # `--days 90` is not a number, so the loop skips it and reads the 90.
        assert 'findstr /r "^[0-9][0-9]*$"' in launcher


class TestATimeoutDoesNotKillTheSectionForTheRestOfTheRun:
    """SIXTEEN TRADES IN THIRTY DAYS, ONE IN NINETY. SAME CONFIG.

    Section eleven on M1 took 16 trades over a 30-day replay and 1 over a
    90-day replay of the same instrument with the same settings. A longer
    window contains the shorter one, so more days cannot mean fewer trades --
    the cause was in the harness.

    `_resolve` stops following a trade after `horizon_bars`. A trade that
    reached neither its stop nor its target by then returned no exit time, and
    the caller read that as "still open" and wrote

        busy[name] = end + clock.duration

    which is the end of the WHOLE RUN. One timeout on day two took the section
    out for the other eighty-eight days, silently, and the low trade count read
    as "the mechanism rarely fires".

    A timeout is the harness losing track of a trade, not a position held
    forever. The section is freed at the end of the horizon, and the count is
    printed per clock.
    """

    def test_the_freeing_uses_the_horizon_and_not_the_end_of_the_run(self) -> None:
        import inspect

        from scripts import dry_run_sections

        source = " ".join(inspect.getsource(dry_run_sections).split())

        assert "_first, _last = _horizon_window(resolve_frame.index, upto, horizon)" in source
        assert "freed = resolve_frame.index[_last - 1]" in source
        assert "timed_out[name] = timed_out.get(name, 0) + 1" in source

    def test_the_horizon_window_has_one_definition(self) -> None:
        """The caller needs the same boundary `_resolve` uses. Computing it a
        second time beside it is how the two drift, which is the defect this
        repository produces more reliably than any other."""
        import inspect

        from scripts.dry_run_sections import _horizon_window, _resolve

        assert "_horizon_window(index, start, horizon_bars)" in inspect.getsource(_resolve)
        assert _horizon_window.__doc__

    def test_the_window_stops_at_the_horizon_and_at_the_data(self) -> None:
        import pandas as pd

        from scripts.dry_run_sections import _horizon_window

        index = pd.date_range("2026-01-01", periods=100, freq="1min", tz="UTC")

        first, last = _horizon_window(index, index[10], 20)
        assert (first, last) == (10, 30)

        # Never past the end of the data.
        first, last = _horizon_window(index, index[90], 50)
        assert (first, last) == (90, 100)

        # A start after the last bar leaves nothing to walk.
        first, last = _horizon_window(index, index[-1] + pd.Timedelta("1min"), 20)
        assert first >= last

    def test_a_timed_out_trade_frees_the_section_within_the_window(self) -> None:
        """The property, end to end on the resolver itself: a trade whose bars
        never touch either barrier returns no exit, and the horizon boundary
        the caller then uses sits INSIDE the data rather than past it."""
        import numpy as np
        import pandas as pd

        from core.types import Direction
        from scripts.dry_run_sections import _horizon_window, _resolve

        index = pd.date_range("2026-01-01", periods=400, freq="1min", tz="UTC")
        flat = pd.DataFrame(
            {
                "open": np.full(400, 100.0),
                "high": np.full(400, 100.1),
                "low": np.full(400, 99.9),
                "close": np.full(400, 100.0),
            },
            index=index,
        )

        class _Wide:
            direction = Direction.LONG
            entry = 100.0
            stop_loss = 90.0
            take_profit = 110.0

        fixed, exit_at, _managed, managed_at = _resolve(flat, index[0], _Wide(), horizon_bars=60)

        assert fixed is None and exit_at is None and managed_at is None
        _first, last = _horizon_window(index, index[0], 60)
        assert index[last - 1] < index[-1], "the section would be freed past the data"


class TestTheAccountReplayClaimsOnlyWhatItApplies:
    """`--jarvis-replay` says four gates ran. Four gates have to have run.

    THE DEFECT THIS GUARDS IS THE ONE THIS FILE IS MOSTLY MADE OF: a claim in
    a report that outlives the code behind it. The run prints "APPLIED" over a
    list of gate names and then prints the REMAINING gates by subtracting that
    list from `NOT_MODELLED`. If a name in the applied set is not in the table,
    the subtraction silently removes nothing and the report says a gate was
    applied while also listing it as missing; if a gate is implemented but not
    named, the report understates itself. Both are wrong in the direction that
    gets read wrong.
    """

    def test_every_gate_claimed_as_applied_exists_in_the_table(self) -> None:
        from scripts.dry_run_sections import JARVIS_REPLAY_APPLIES, NOT_MODELLED

        named = {name for name, _count, _why in NOT_MODELLED}
        missing = JARVIS_REPLAY_APPLIES - named
        assert not missing, (
            f"the replay claims to apply {sorted(missing)}, which is not in NOT_MODELLED, "
            f"so subtracting it removes nothing and the same gate is printed as both "
            f"applied and missing"
        )

    def test_every_gate_the_code_can_return_is_claimed(self) -> None:
        """The other direction: a reason `_historical_jarvis_gate` can emit
        and the report does not claim is a gate whose work is invisible."""
        import re

        body = SOURCE[
            SOURCE.index("def _historical_jarvis_gate(") : SOURCE.index("class _RawShadowEngine")
        ]
        from filters.base import Reason
        from scripts.dry_run_sections import JARVIS_REPLAY_APPLIES

        emitted = set(re.findall(r'"([A-Z_]{4,})"', body))
        # `lively.reason.name` is returned dynamically rather than as a
        # literal, so the liveliness reasons are added from the enum itself
        # instead of being read out of the source.
        assert "MARKET_TOO_QUIET" in {reason.name for reason in Reason}
        emitted.add("MARKET_TOO_QUIET")
        unclaimed = emitted - JARVIS_REPLAY_APPLIES
        assert not unclaimed, (
            f"the gate can refuse with {sorted(unclaimed)} and the contract does not "
            f"say so, so those refusals appear in the run with nothing explaining them"
        )

    def test_the_launcher_asks_for_the_replay_and_keeps_m1(self) -> None:
        launcher = (ROOT / "hoeveel.cmd").read_text()

        assert "--jarvis-replay" in launcher
        # THE VOLUME GATE READS M1 AND ONLY M1. A launcher that passes
        # --no-m1 would produce a run claiming a gate that never fires.
        assert "--no-m1" not in launcher

    def test_no_m1_and_the_replay_are_refused_together(self) -> None:
        assert "--jarvis-replay needs M1 bars" in SOURCE


class TestTheEuroAnswerIsActuallyComputed:
    """`_what_the_account_would_be_worth` runs once, at the end of a run that
    takes an hour. That is precisely the code that ships broken: nothing calls
    it until the owner has waited sixty minutes for it.

    So these call it directly, on decisions with known results, and assert on
    what lands on stdout.
    """

    def _settings(self):
        from pathlib import Path

        from config.loader import load_settings

        return load_settings(
            overlay=Path(__file__).resolve().parents[1] / "config" / "eightcap.yaml",
            env_overrides=False,
        )

    def _trades(self, results):
        from datetime import datetime, timedelta

        from scripts.dry_run_sections import Decision

        base = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
        return [
            Decision(
                base + timedelta(hours=i),
                "XAUUSD",
                "section_six_gold_m5",
                "TRADE",
                direction="LONG",
                risk_money=4.0,
                result_r=r,
                pnl_money=r * 4.0,
                exit_at=base + timedelta(hours=i, minutes=20),
                pass_key=("section_six_gold_m5", "M5"),
                managed_r=r,
                managed_money=r * 4.0,
            )
            for i, r in enumerate(results)
        ]

    def test_the_flat_total_is_the_sum_of_the_trades(self, capsys) -> None:
        from scripts.dry_run_sections import _what_the_account_would_be_worth

        _what_the_account_would_be_worth(
            self._trades([1.0, -1.0, 1.5, -1.0]), self._settings(), 200.0, 180
        )
        out = capsys.readouterr().out

        # 4.0 - 4.0 + 6.0 - 4.0 = +2.00
        assert "+2.00" in out
        assert "IF THIS HAD BEEN RUNNING FOR 180 DAYS" in out

    def test_fixed_daily_stop_uses_closed_pnl_before_later_entries(self) -> None:
        from scripts.dry_run_sections import _under_daily_money_stop

        rows = self._trades([-1.0] * 12)
        accepted, skipped = _under_daily_money_stop(rows, 20.0, managed=True)

        assert len(accepted) == 5
        assert skipped == 7

    def _lotted(self, results, *, per_lot: float, minimum: float = 0.01, step: float = 0.01):
        """Trades that carry the lot economics a real re-size needs."""
        rows = self._trades(results)
        for row in rows:
            row.risk_per_lot = per_lot
            row.volume_min = minimum
            row.volume_step = step
        return rows

    def _live_mode(self):  # type: ignore[no-untyped-def]
        from core.types import TradingMode

        settings = self._settings()
        return settings.model_copy(
            update={"system": settings.system.model_copy(update={"mode": TradingMode.MICRO_LIVE})}
        )

    def test_the_stake_moves_only_in_whole_lot_steps(self) -> None:
        """THE THING A RESCALE CANNOT SHOW, and the reason this was rewritten.

        The old compounded line multiplied each result by
        `balance_then / balance_start`, which grows the stake smoothly through
        lot sizes no broker accepts. Really the stake is indivisible: 2% of
        EUR 252 is EUR 5.04 and one minimum lot of gold risks EUR 3.85, so the
        sizer buys 0.0131 lots and rounds DOWN to 0.01. At EUR 358 it buys
        0.0186 and still rounds to 0.01.

        So on this account the compounded and flat numbers are ARITHMETICALLY
        IDENTICAL until the balance roughly triples, and a report that shows
        them differing is showing a curve the account cannot have.
        """
        from scripts.dry_run_sections import _compound

        settings = self._live_mode()
        # Twenty winners at EUR 3.85 is +EUR 77, so the balance ends near
        # EUR 329 -- still short of the roughly EUR 385 a second lot step
        # needs. Forty winners WOULD cross it, and the stake would correctly
        # double; that case is the next test.
        rows = self._lotted([1.0] * 20, per_lot=385.0)
        run = _compound(rows, settings, 252.18)

        assert run.balance < run.next_step_balance
        assert run.first_risk == run.last_risk == run.biggest_risk
        assert run.next_step_balance > 0.0
        assert run.taken == len(rows)

    def test_the_stake_steps_up_the_moment_the_balance_can_carry_it(self) -> None:
        """The same instrument, enough winners to cross the threshold. Without
        this the test above is satisfied by a walk that never re-sizes."""
        from scripts.dry_run_sections import _compound

        settings = self._live_mode()
        run = _compound(self._lotted([1.0] * 40, per_lot=385.0), settings, 252.18)

        assert run.last_risk == pytest.approx(2 * run.first_risk)

    def test_the_stake_does_grow_once_a_step_is_affordable(self) -> None:
        """Without this the test above passes on a walk that never re-sizes
        anything, which would be the old bug with a new name."""
        from scripts.dry_run_sections import _compound

        settings = self._live_mode()
        # A cheap instrument: 0.01 lot risks EUR 0.50, so 2% of EUR 252 already
        # buys ten steps and every win buys more.
        rows = self._lotted([1.0] * 60, per_lot=50.0)
        run = _compound(rows, settings, 252.18)

        assert run.last_risk > run.first_risk, (run.first_risk, run.last_risk)
        assert run.next_step_balance == 0.0, "the stake grew, so there is no step to wait for"

    def test_a_trade_the_account_cannot_afford_does_not_happen(self) -> None:
        """Live that trade is UNDERCAPITALIZED and no order is sent. A rescale
        takes it anyway, at a stake the account could not have posted -- the
        single largest lie in the old number."""
        from scripts.dry_run_sections import _compound

        settings = self._live_mode()
        # 0.01 lot risks EUR 100, and 2% of EUR 252 is EUR 5.04.
        rows = self._lotted([1.0] * 12, per_lot=10_000.0)
        run = _compound(rows, settings, 252.18)

        assert run.taken == 0
        assert run.skipped_too_small == len(rows)
        assert run.balance == pytest.approx(252.18)

    def test_bounded_minimum_lot_override_takes_the_trade(self) -> None:
        from scripts.dry_run_sections import _compound

        settings = self._live_mode()
        # Target stake is about EUR 6.84; the minimum lot risks EUR 10.
        # That is above 2%, but below both 10% and the EUR 20 daily stop.
        run = _compound(self._lotted([1.0], per_lot=1_000.0), settings, 341.95)

        assert run.taken == 1
        assert run.minimum_lot_overrides == 1
        assert run.skipped_too_small == 0

    def test_daily_money_stop_pauses_new_entries_until_next_day(self) -> None:
        from scripts.dry_run_sections import _compound

        settings = self._live_mode()
        rows = self._lotted([-1.0] * 12, per_lot=385.0)
        run = _compound(rows, settings, 252.18)

        assert run.skipped_daily_loss > 0
        assert run.taken + run.skipped_daily_loss == len(rows)

    def test_a_wiped_account_stops_trading(self) -> None:
        """Booking results on a negative balance is how a replay produces a
        recovery that could not have happened."""
        from scripts.dry_run_sections import _compound

        settings = self._live_mode()
        rows = self._lotted([-1.0] * 400, per_lot=385.0)
        run = _compound(rows, settings, 252.18)

        assert run.taken < len(rows), "it kept trading past the point of no money"
        assert run.balance <= 252.18

    def test_the_report_says_the_stake_never_moved(self, capsys) -> None:
        from scripts.dry_run_sections import _what_the_account_would_be_worth

        _what_the_account_would_be_worth(
            self._lotted([1.0, -1.0, 1.0, 1.0], per_lot=385.0), self._live_mode(), 252.18, 180
        )
        out = capsys.readouterr().out

        assert "THE STAKE NEVER MOVED" in out
        assert "COMPOUNDED" in out and "FLAT STAKE" in out

    def test_compounding_beats_a_flat_stake_on_a_rising_curve(self, capsys) -> None:
        """Not a tautology: it is the property that makes the second number
        worth printing at all, and a sign error in the rescale inverts it."""
        from scripts.dry_run_sections import _what_the_account_would_be_worth

        _what_the_account_would_be_worth(self._trades([1.0] * 12), self._settings(), 200.0, 180)
        out = capsys.readouterr().out
        numbers = [
            float(line.split("EUR")[1].split()[0].replace("+", ""))
            for line in out.splitlines()
            if "result " in line and "EUR" in line
        ]
        assert len(numbers) == 2
        assert numbers[1] > numbers[0]

    def test_a_run_with_no_trades_says_so_instead_of_printing_nothing(self, capsys) -> None:
        """An absent block reads as a zero block. That confusion is the single
        most-repeated defect in this file's history."""
        from scripts.dry_run_sections import _what_the_account_would_be_worth

        _what_the_account_would_be_worth([], self._settings(), 200.0, 180)
        out = capsys.readouterr().out

        assert "zero observations" in out

    def test_a_shadow_section_is_named_as_hypothetical(self, capsys) -> None:
        """THE SHADOW SECTION IS FOUND, NOT TYPED.

        This named `section_eleven_xaujpy_legs_m5` and passed until the owner
        promoted that section on 6 September -- at which point the test failed
        while the behaviour it guards was perfectly intact. A test that pins
        WHICH section is shadowed breaks every time permission changes; the
        property is that a section off the allowlist is labelled hypothetical,
        whichever one that is.
        """
        from scripts.dry_run_sections import _what_the_account_would_be_worth

        settings = self._settings()
        live = set(settings.analysis.confluence.live_enabled_modules)
        shadow = next(
            name
            for name in sorted(vars(settings.analysis))
            if name.startswith("section_") and name not in live
        )
        rows = self._trades([1.0, -1.0])
        for row in rows:
            row.pass_key = (shadow, "M5")
        _what_the_account_would_be_worth(rows, settings, 200.0, 180)
        out = capsys.readouterr().out

        assert "NOT on the real-money allowlist" in out
        assert shadow in out

    def test_the_contract_prints_and_names_the_position_cap(self, capsys) -> None:
        from scripts.dry_run_sections import _jarvis_replay_contract

        settings = self._settings()
        _jarvis_replay_contract(settings, 215.0, offered=40, allowed=25)
        out = capsys.readouterr().out

        assert "15 of 40 entries were refused by that book" in out
        assert str(settings.effective_max_positions(215.0)) in out
        assert "news blackout" in out


class TestTheReplayHandsTheSectionItsLegs:
    """`_context` grew a second instrument. If it hands the module nothing, or
    hands it a bar the cross has not reached yet, section eleven is either
    permanently silent or permanently looking ahead -- and both are invisible
    from the outcome column.
    """

    def _frames(self):
        from datetime import datetime, timedelta

        import numpy as np
        import pandas as pd

        from analysis.section_eleven_legs import MIN_LEG_BARS

        count = MIN_LEG_BARS + 400
        start = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
        index = pd.DatetimeIndex([start + timedelta(minutes=5 * i) for i in range(count)])
        steps = np.arange(count, dtype=float)
        values = 2000.0 + np.sin(steps / 9.0) * 5.0
        return pd.DataFrame(
            {
                "open": values,
                "high": values + 1.0,
                "low": values - 1.0,
                "close": values,
                "tick_volume": np.full(count, 100.0),
            },
            index=index,
        )

    def test_the_legs_reach_the_context_under_the_agreed_key(self) -> None:
        from analysis.section_eleven_legs import LEGS_META_KEY, LegBars
        from core.types import Timeframe
        from scripts.dry_run_sections import _context

        frame = self._frames()
        upto = frame.index[-1].to_pydatetime() + Timeframe.M5.duration
        ctx = _context(
            "XAUJPY",
            {Timeframe.M5: frame},
            upto,
            0.5,
            legs={
                "XAUUSD": (Timeframe.M5, frame),
                "USDJPY": (Timeframe.M5, frame),
            },
        )
        assert ctx is not None
        payload = ctx.meta[LEGS_META_KEY]
        assert set(payload) == {"XAUUSD", "USDJPY"}
        assert all(isinstance(leg, LegBars) for leg in payload.values())

    def test_a_leg_is_cut_at_the_same_instant_as_the_cross(self) -> None:
        """A leg one bar ahead is a look-ahead of exactly the size of the lag
        this section trades, and it flatters the result."""
        from analysis.section_eleven_legs import LEGS_META_KEY
        from core.types import Timeframe
        from scripts.dry_run_sections import _context

        frame = self._frames()
        # Mid-history, so there are unseen bars on both sides.
        upto = frame.index[450].to_pydatetime() + Timeframe.M5.duration
        ctx = _context(
            "XAUJPY",
            {Timeframe.M5: frame},
            upto,
            0.5,
            legs={"XAUUSD": (Timeframe.M5, frame), "USDJPY": (Timeframe.M5, frame)},
        )
        assert ctx is not None
        leg = ctx.meta[LEGS_META_KEY]["XAUUSD"]
        cross_last = ctx.series[Timeframe.M5].df.index[-1]
        assert leg.frame.index[-1] == cross_last
        assert leg.frame.index[-1] + Timeframe.M5.duration <= upto

    def test_a_leg_with_too_little_history_attaches_nothing(self) -> None:
        """Not a truncated payload: NOTHING. Half the legs is not half a read."""
        from analysis.section_eleven_legs import LEGS_META_KEY
        from core.types import Timeframe
        from scripts.dry_run_sections import _context

        frame = self._frames()
        upto = frame.index[-1].to_pydatetime() + Timeframe.M5.duration
        ctx = _context(
            "XAUJPY",
            {Timeframe.M5: frame},
            upto,
            0.5,
            legs={
                "XAUUSD": (Timeframe.M5, frame),
                "USDJPY": (Timeframe.M5, frame.iloc[:20]),
            },
        )
        assert ctx is not None
        assert LEGS_META_KEY not in ctx.meta


class TestEachClaimedGateIsActuallyReachable:
    """Four gates are claimed. Four gates have to be able to fire, AND a clean
    setup has to be able to pass all four.

    THIS IS THE DEFECT THIS REPOSITORY PRODUCES MORE RELIABLY THAN ANY OTHER: a
    check that exists, is correct, is documented, is tested against its own
    inputs -- and is never reached by the path the code takes. The previous
    tests around this asserted that the gate function contains the right
    strings. That is not the same claim.

    A gate that never fires makes the replay optimistic and silent about it; a
    gate that always fires makes it empty and equally silent. Both directions
    are asserted here.
    """

    def _settings(self):
        from pathlib import Path

        from config.loader import load_settings

        return load_settings(
            overlay=Path(__file__).resolve().parents[1] / "config" / "eightcap.yaml",
            env_overrides=False,
        )

    def _tradeable_market(self):
        """Bars a real instrument could have printed.

        The jump-to-range ratio matters: the liveliness filter refuses a market
        that opens past its own stops, so a random walk with tiny bar ranges is
        refused for a reason that has nothing to do with the gate under test.
        """
        from datetime import datetime, timedelta

        import numpy as np
        import pandas as pd

        n = 400
        rng = np.random.default_rng(3)
        step = rng.normal(0, 0.6, n)
        close = 2000.0 + np.cumsum(step)
        opens = np.concatenate(([close[0]], close[:-1]))
        spans = np.abs(step) + rng.uniform(1.5, 3.0, n)
        frame = pd.DataFrame(
            {
                "open": opens,
                "high": np.maximum(opens, close) + spans / 2,
                "low": np.minimum(opens, close) - spans / 2,
                "close": close,
                "tick_volume": np.full(n, 100.0),
            },
            index=pd.DatetimeIndex(
                [datetime(2026, 1, 5, tzinfo=UTC) + timedelta(minutes=5 * i) for i in range(n)]
            ),
        )
        minute = frame.copy()
        minute.index = pd.DatetimeIndex(
            [datetime(2026, 1, 5, tzinfo=UTC) + timedelta(minutes=i) for i in range(n)]
        )
        return frame, minute

    def _context(self, frame, minute, *, spread: float, last_volume: float = 100.0):
        from core.types import MarketContext, Series, Tick, Timeframe

        minute = minute.copy()
        minute.loc[minute.index[-1], "tick_volume"] = last_volume
        now = frame.index[-1].to_pydatetime() + Timeframe.M5.duration
        price = float(frame["close"].iloc[-1])
        return MarketContext(
            "XAUJPY",
            now,
            {
                Timeframe.M5: Series("XAUJPY", Timeframe.M5, frame, now),
                Timeframe.M1: Series("XAUJPY", Timeframe.M1, minute, now),
            },
            Tick("XAUJPY", now, price - spread / 2, price + spread / 2),
        )

    def _idea(self, frame, *, width: float, ratio: float):
        from types import SimpleNamespace

        from core.types import Direction

        entry = float(frame["close"].iloc[-1])
        return SimpleNamespace(
            direction=Direction.LONG,
            entry=entry,
            stop_loss=entry - width,
            take_profit=entry + width * ratio,
            # A REAL `TradeIdea` CARRIES THIS and the gate reads it to pick a
            # per-family spread limit. A fixture without it passed until the
            # per-family limit arrived, then raised AttributeError -- on the
            # gate path, which is the path that decides every trade.
            setup_family="section_eleven_xaujpy_legs_m5",
        )

    def _spec(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            asset_class=SimpleNamespace(value="metal"),
            point=0.01,
            volume_min=0.01,
            digits=2,
        )

    def _gate(self, ctx, idea):
        from core.types import Timeframe
        from scripts.dry_run_sections import REACH_HORIZON, _historical_jarvis_gate

        return _historical_jarvis_gate(
            ctx, idea, self._spec(), self._settings(), Timeframe.M5, horizon=REACH_HORIZON
        )

    def test_a_clean_setup_passes_every_gate(self) -> None:
        """Without this the three below pass on a run that takes no trades at
        all, which is the same silence one level up."""
        frame, minute = self._tradeable_market()
        ctx = self._context(frame, minute, spread=0.1)
        assert self._gate(ctx, self._idea(frame, width=6.0, ratio=1.5)) is None

    def test_a_wide_spread_against_a_narrow_stop_is_refused(self) -> None:
        frame, minute = self._tradeable_market()
        ctx = self._context(frame, minute, spread=1.0)
        blocked = self._gate(ctx, self._idea(frame, width=6.0, ratio=1.5))
        assert blocked is not None and blocked[0] == "SPREAD_EATS_THE_STOP"

    def test_a_volume_spike_is_refused(self) -> None:
        frame, minute = self._tradeable_market()
        ctx = self._context(frame, minute, spread=0.1, last_volume=900.0)
        blocked = self._gate(ctx, self._idea(frame, width=6.0, ratio=1.5))
        assert blocked is not None and blocked[0] == "VOLUME_SPIKE"

    def test_a_target_this_market_does_not_reach_is_refused(self) -> None:
        frame, minute = self._tradeable_market()
        ctx = self._context(frame, minute, spread=0.1)
        blocked = self._gate(ctx, self._idea(frame, width=6.0, ratio=4.0))
        assert blocked is not None and blocked[0] == "TARGET_RARELY_REACHED"

    def test_a_market_that_gaps_past_its_own_stops_is_refused(self) -> None:
        """MARKET_TOO_QUIET is the enum name; what it actually refuses here is
        a market whose bar-to-bar jump swamps its own range."""
        from datetime import datetime, timedelta

        import numpy as np
        import pandas as pd

        n = 400
        rng = np.random.default_rng(11)
        close = 2000.0 + np.cumsum(rng.normal(0, 1.5, n))
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.05,
                "low": close - 0.05,
                "close": close,
                "tick_volume": np.full(n, 100.0),
            },
            index=pd.DatetimeIndex(
                [datetime(2026, 1, 5, tzinfo=UTC) + timedelta(minutes=5 * i) for i in range(n)]
            ),
        )
        minute = frame.copy()
        minute.index = pd.DatetimeIndex(
            [datetime(2026, 1, 5, tzinfo=UTC) + timedelta(minutes=i) for i in range(n)]
        )
        ctx = self._context(frame, minute, spread=0.01)
        blocked = self._gate(ctx, self._idea(frame, width=6.0, ratio=1.5))
        assert blocked is not None and blocked[0] == "MARKET_TOO_QUIET"


class TestTheCounterfactualSurvivesEveryClock:
    """`RAW_BTC_HORIZONS[clock]` and an `entry_spread_price` that only exists
    on the BTC path.

    Both arrived with the counterfactual, which was written for three BTC
    sections on M1/M5/M15 and is now run for the whole book. Section eight
    runs H1 and section nine M30, so the first gated setup on either would
    have raised -- a KeyError on one line, a NameError on the next -- after
    however many hours the run had already spent. Neither is reachable from a
    BTC-only test, which is exactly why they got through.
    """

    def test_every_configured_section_clock_has_a_counterfactual_horizon(self) -> None:
        from pathlib import Path

        from config.loader import load_settings
        from core.types import Timeframe
        from scripts.dry_run_sections import RAW_BTC_HORIZONS, REACH_HORIZON

        settings = load_settings(
            overlay=Path(__file__).resolve().parents[1] / "config" / "eightcap.yaml",
            env_overrides=False,
        )
        import re

        table = re.search(r"module_config = \{(.+?)\n        \}", SOURCE, re.S)
        assert table is not None
        for name in sorted(set(re.findall(r'"([a-z0-9_]+)":', table.group(1)))):
            config = getattr(settings.analysis, name, None)
            clock = getattr(config, "timeframe", None)
            if clock is None:
                continue
            horizon = RAW_BTC_HORIZONS.get(Timeframe.parse(clock), REACH_HORIZON)
            assert horizon > 0, f"{name} runs {clock} and has no counterfactual horizon"

    def test_the_gate_path_never_reads_a_btc_only_local(self) -> None:
        """`entry_spread_price` is assigned inside `if raw_shadow:` and read
        by the gate block below it. The gate block must not touch it."""
        block = SOURCE[
            SOURCE.index("            if jarvis_replay:") : SOURCE.index(
                "            if raw_shadow:\n                volume = float(spec.volume_min)"
            )
        ]
        offenders = [
            line.strip()
            for line in block.splitlines()
            if "entry_spread_price" in line and "raw_shadow" not in line and "#" not in line
        ]
        assert not offenders, (
            "the gate block reads entry_spread_price outside the raw-shadow guard, "
            f"so a non-BTC section raises NameError on its first refusal:\n{offenders}"
        )


class TestTheWalkCanBeNarrowedToWhatTheSectionsTrade:
    """Six of eleven markets in the 90-day book run took no trade at all --
    GBPUSD, USDCHF, EURJPY, GBPJPY, US30, GER40 -- and not one section on the
    book is ALLOWED to trade any of them. Three hundred thousand bars each,
    judged seven times per bar, to prove a thing the config already knew.
    """

    def _settings(self):
        from pathlib import Path

        from config.loader import load_settings

        return load_settings(
            overlay=Path(__file__).resolve().parents[1] / "config" / "eightcap.yaml",
            env_overrides=False,
        )

    def test_it_finds_a_market_for_every_live_section(self) -> None:
        """THE PROPERTY, and the reason it is not a list in a launcher.

        A section declares its market in one of three places -- `allowed_symbols`
        on the config, `symbol` on the config, or `symbol` on the module class.
        Reading only the first is the bug this file already shipped once: it
        returned an empty tuple for every XAUJPY section.
        """
        from scripts.dry_run_sections import _markets_each_section_needs

        settings = self._settings()
        live = settings.analysis.confluence.live_enabled_modules
        for name in live:
            markets, unbounded = _markets_each_section_needs(settings, (name,))
            assert not unbounded, f"{name} declares no market, so the walk cannot be narrowed"
            assert markets, f"{name} resolved to no market at all"

    def test_narrowing_never_drops_a_market_a_section_needs(self) -> None:
        """The union has to contain every section's own markets. A narrow walk
        that leaves one out produces a zero row, and a zero row reads as
        'the strategy found nothing'."""
        from scripts.dry_run_sections import _markets_each_section_needs

        settings = self._settings()
        live = tuple(settings.analysis.confluence.live_enabled_modules)
        together, _unbounded = _markets_each_section_needs(settings, live)
        for name in live:
            alone, _ = _markets_each_section_needs(settings, (name,))
            missing = [market for market in alone if market not in together]
            assert not missing, f"{name} needs {missing}, absent from the narrowed universe"

    def test_a_section_that_trades_anything_refuses_the_narrowing(self) -> None:
        """`impulse_retest` has no market of its own. Narrowing around it would
        silently stop measuring it -- the same silence, in a new place -- so it
        is reported instead of ignored."""
        from scripts.dry_run_sections import _markets_each_section_needs

        settings = self._settings()
        _markets, unbounded = _markets_each_section_needs(settings, ("impulse_retest",))
        assert unbounded == ["impulse_retest"]

    def test_the_narrowing_is_refused_rather_than_applied_when_unbounded(self) -> None:
        assert "--section-markets ignored" in SOURCE

    def test_the_launcher_asks_for_it(self) -> None:
        launcher = (ROOT / "hoeveel.cmd").read_text()

        assert "--section-markets" in launcher
        # AND ONLY THE SECTIONS THAT SPEND MONEY, by default. Without this the
        # run also measures the shadow sections, and each of those drags in its
        # own market and its own clock -- USDJPY.i and M30 for section nine,
        # XAUJPY for eleven. Useful when you want to know what a benched
        # section WOULD do; pure waiting when you want to know what the account
        # does. `schaduw` is the word that puts them back.
        assert "--live-only" in launcher
        assert "schaduw" in launcher


class TestASilentSectionSaysWhichSilenceItIs:
    """Section eleven produced 204,575 decisions and zero trades, and every
    one of them read "no weighted directional evidence" -- the ENGINE's words
    for "the module sent nothing". Six different causes, one sentence, and no
    way to tell a broken build from a rare mechanism.
    """

    def test_the_sections_own_reason_reaches_the_row(self) -> None:
        assert "own = next(" in SOURCE
        assert "sig.module == name and not sig.score" in SOURCE

    def test_the_legs_section_gives_a_different_reason_per_cause(self) -> None:
        """Distinct strings, asserted by DRIVING the module rather than by
        reading its source. A reason that exists and never reaches a caller is
        this repository's most-repeated defect."""
        from datetime import datetime, timedelta

        import numpy as np
        import pandas as pd

        from analysis.section_eleven_legs import (
            LEGS_META_KEY,
            MIN_LEG_BARS,
            SectionElevenLegs,
            leg_bars,
        )
        from config.schema import SectionElevenLegsConfig
        from core.types import MarketContext, Series, Tick, Timeframe

        count = MIN_LEG_BARS + 60
        index = pd.DatetimeIndex(
            [datetime(2026, 2, 2, 8, tzinfo=UTC) + timedelta(minutes=5 * i) for i in range(count)]
        )
        steps = np.arange(count, dtype=float)
        gold = 2000.0 + np.sin(steps / 7.0) * 6.0
        yen = 150.0 + np.cos(steps / 11.0) * 0.5

        def frame(values, wobble):
            return pd.DataFrame(
                {
                    "open": values,
                    "high": values + wobble,
                    "low": values - wobble,
                    "close": values,
                    "tick_volume": np.full(count, 100.0),
                },
                index=index,
            )

        base, quote = frame(gold, 0.4), frame(yen, 0.02)
        cross = frame(gold * yen + 500.0, 20.0)
        now = index[-1].to_pydatetime() + timedelta(minutes=5)

        def read(meta):
            ctx = MarketContext(
                "XAUJPY",
                now,
                {Timeframe.M5: Series("XAUJPY", Timeframe.M5, cross, now)},
                Tick("XAUJPY", now, 1.0, 1.1),
            )
            if meta is not None:
                ctx.meta[LEGS_META_KEY] = meta
            section = SectionElevenLegs(
                "section_eleven_xaujpy_legs_m5",
                SectionElevenLegsConfig(enabled=True, gap_atr=0.50, minimum_gap_atr=0.0),
            )
            signal = section.analyze(ctx)
            assert signal.score == 0.0
            return signal.reasoning

        fresh = {
            "XAUUSD": leg_bars("XAUUSD", base, now),
            "USDJPY": leg_bars("USDJPY", quote, now),
        }
        stale = dict(fresh)
        old = stale["XAUUSD"]
        stale["XAUUSD"] = type(old)(old.symbol, old.frame, age_seconds=99_999.0)

        reasons = {
            "none": read(None),
            "missing": read({"XAUUSD": fresh["XAUUSD"]}),
            "stale": read(stale),
            "quiet": read(fresh),
        }
        assert len(set(reasons.values())) == 4, reasons
        assert "no legs attached at all" in reasons["none"]
        assert "USDJPY" in reasons["missing"]
        assert "old" in reasons["stale"]
        # The legs are present and current; the gap simply is not there.
        assert "gap" in reasons["quiet"]


class TestSymbolsAreSpelledTheWayTheBrokerSpellsThem:
    """`--symbols US30` on a broker that lists `US30.i`.

    Every other way of choosing a universe reads names FROM the broker, so
    they arrive suffixed. `--symbols` is the one path from the command line
    straight to the fetch, and an unresolved name there does not raise: it
    prints "no history" and yields a report with no rows, which is
    indistinguishable from a strategy that found nothing. That is the same
    suffix mismatch that has silently disabled four things in this project,
    so these tests pin the BEHAVIOUR (a typed name reaches the catalogue's
    spelling) rather than any particular list of markets.
    """

    @staticmethod
    def _settings():
        # THE EIGHTCAP OVERLAY, because that is where `symbol_suffix: ".i"`
        # lives. Against the default config the suffix is empty, the resolve
        # is a no-op, and the one test that matters here SKIPPED -- which is
        # exactly the shape of proof this bug keeps hiding behind.
        from config.loader import load_settings

        return load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)

    @staticmethod
    def _connector(*names):
        class _Cat:
            def symbols(self):
                return [type("S", (), {"name": n})() for n in names]

        return _Cat()

    def test_a_plain_name_reaches_the_suffixed_market(self):
        from scripts.dry_run_sections import _as_this_broker_spells_them

        settings = self._settings()
        suffix = settings.instruments.symbol_suffix
        assert suffix, "the live overlay must carry a broker suffix for this to test anything"
        assert _as_this_broker_spells_them(
            ["US30"], self._connector(f"US30{suffix}"), settings
        ) == [f"US30{suffix}"]

    def test_an_already_correct_name_is_left_exactly_alone(self):
        # The double-suffix half of the bug: `broker_symbol` on an already
        # suffixed name produces `US30.i.i`, which the Control Deck shipped.
        from scripts.dry_run_sections import _as_this_broker_spells_them

        settings = self._settings()
        suffix = settings.instruments.symbol_suffix
        listed = f"US30{suffix}"
        assert _as_this_broker_spells_them(
            [listed], self._connector(listed), settings
        ) == [listed]

    def test_a_name_this_broker_does_not_list_is_returned_as_typed(self):
        # So the fetch loop's per-symbol failure names what the operator
        # wrote, not a suffix this code invented for them.
        from scripts.dry_run_sections import _as_this_broker_spells_them

        assert _as_this_broker_spells_them(
            ["NOTAMARKET"], self._connector("EURUSD.i"), self._settings()
        ) == ["NOTAMARKET"]

    def test_no_catalogue_means_no_rewriting(self):
        from scripts.dry_run_sections import _as_this_broker_spells_them

        class _Broken:
            def symbols(self):
                raise RuntimeError("terminal not connected")

        settings = self._settings()
        assert _as_this_broker_spells_them(["US30"], _Broken(), settings) == ["US30"]
        assert _as_this_broker_spells_them(["US30"], self._connector(), settings) == ["US30"]

    def test_the_symbols_branch_actually_calls_it(self):
        # The defect class this repository keeps producing is a correct helper
        # that nothing on the live path calls. Pin the wiring, not the helper.
        branch = SOURCE.split("if args.symbols:", 1)[1].split("elif args.core:", 1)[0]
        assert "_as_this_broker_spells_them" in branch

    def test_the_requested_market_launchers_name_markets_this_resolves(self):
        # `replay_requested_markets` hardcodes plain names. That is fine ONLY
        # because the branch above resolves them; if that ever stops being
        # true this test is what says so.
        from scripts.replay_requested_markets import commands

        named = {
            argv[argv.index("--symbols") + 1]
            for kind in ("us30", "s5")
            for argv in commands(kind, 30)
            if "--symbols" in argv
        }
        assert named, "the launchers pass no --symbols at all any more"
        settings = self._settings()
        for name in named:
            assert settings.instruments.canonical_symbol(name) == name, (
                f"{name} is already a broker spelling; the launcher should pass the canonical one"
            )


class TestAFixedExitReplayReallyRemovesEveryManager:
    """`--fixed-exits`: entry stop and target, and nothing else touches it.

    The owner asked for this after seeing a gold M1 trade sit at +EUR 1.60 and
    give it all back. The question underneath is whether a mechanism's ENTRY
    earns anything, or whether the number comes from a rule laid over the top
    afterwards -- and that question is only answered if the flag removes ALL
    of them. A flag that removed three of four and printed "no management"
    would answer a question nobody asked, convincingly.

    So these pin the four independent ways management can survive: the
    account-wide break-even rule, a section's own shadow override, the full
    replay manager, and the evening flatten.
    """

    @staticmethod
    def _args(*extra):
        from scripts.dry_run_sections import build_parser

        return build_parser().parse_args(
            ["--days", "30", "--only", "section_us30_impulse_m1", "--jarvis-replay", *extra]
        )

    def test_the_account_wide_break_even_rule_is_gone(self):
        from config.loader import load_settings
        from scripts.dry_run_sections import _break_even_rule

        settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
        assert _break_even_rule(settings) is not None, (
            "this config has no break-even rule at all, so this test proves nothing"
        )
        stripped = settings.model_copy(
            update={
                "trade_management": settings.trade_management.model_copy(
                    update={"break_even_at_r": 0.0}
                )
            }
        )
        assert _break_even_rule(stripped) is None

    def test_main_actually_strips_it_rather_than_only_defining_the_flag(self):
        # The wiring, not the helper. A correct rule that no live path reaches
        # is the defect this repository produces most often.
        block = SOURCE.split("if args.fixed_exits:", 2)[2]
        assert "break_even_at_r" in block.split("settings = settings.model_copy(", 2)[1]

    def test_a_sections_own_shadow_override_cannot_put_it_back(self):
        # Three sections carry `shadow_break_even_at_r`. Zeroing the account
        # rule does not touch them, so the loop has to refuse it explicitly.
        guard = SOURCE.split("shadow_trigger is not None", 1)[1].split("\n", 1)[0]
        assert "args.fixed_exits" in guard

    def test_the_full_replay_manager_is_switched_off(self):
        # `_one_clock` reads one field to decide this; the flag must set it.
        # A WINDOW, NOT A SPLIT ON ")". The expression contains its own
        # brackets, so splitting on the first one cut it in half and the test
        # failed against code that was correct.
        row = SOURCE.split("args.btc_jarvis_replay or args.jarvis_replay,", 1)[1][:800]
        assert 'args.s5_exit or ("fixed" if args.fixed_exits else "")' in row
        assert 'full_replay_management = jarvis_replay and not comparison_exit' in SOURCE

    def test_the_evening_flatten_is_a_time_exit_and_goes_too(self):
        guard = SOURCE.split("if comment.casefold() in flattened", 1)[1].split(":", 1)[0]
        assert "not args.fixed_exits" in guard

    def test_it_is_refused_where_it_would_be_a_no_op(self):
        # Without --jarvis-replay there is no management to remove, so the run
        # would print a comparison it never made.
        from scripts.dry_run_sections import build_parser, main

        args = build_parser().parse_args(
            ["--days", "30", "--only", "section_us30_impulse_m1", "--fixed-exits"]
        )
        assert args.fixed_exits and not args.jarvis_replay
        with pytest.raises(SystemExit):
            main(["--days", "30", "--only", "section_us30_impulse_m1", "--fixed-exits"])

    def test_it_refuses_to_share_the_field_with_the_s5_comparison(self):
        from scripts.dry_run_sections import main

        with pytest.raises(SystemExit):
            main(
                [
                    "--days", "30", "--only", "section_five_ndx100_m5",
                    "--jarvis-replay", "--s5-exit", "fixed", "--fixed-exits",
                ]
            )

    def test_neither_guard_reaches_the_broker_or_the_config(self):
        # A refusal has to happen before login, or the operator waits for a
        # connection to be told the arguments were wrong.
        from unittest.mock import patch

        from scripts.dry_run_sections import main

        with patch("scripts.dry_run_sections.load_settings") as load:
            with pytest.raises(SystemExit):
                main(["--days", "30", "--only", "section_us30_impulse_m1", "--fixed-exits"])
            load.assert_not_called()

    def test_the_launcher_offers_it_and_writes_its_own_file(self):
        launcher = (ROOT / "us30.cmd").read_text(encoding="utf-8")
        assert "--fixed-exits" in launcher
        # TWO STANDS, TWO FILES. Sharing runtime\us30.csv would let the second
        # run overwrite the first, and cmd expands an unset %CSVTAG% to
        # nothing at all rather than failing.
        assert "set CSVTAG=" in launcher
        assert 'if /i "%~1"=="vast" set CSVTAG=-vast' in launcher
        argv = cmd_argv(
            launcher,
            **{
                "%DAGEN%": "180",
                "%SECTIES%": "section_us30_impulse_m1",
                "%EXITS%": "--fixed-exits",
                "%CSVTAG%": "-vast",
            },
        )
        parsed = build_parser_for_launcher(argv)
        assert parsed.fixed_exits and parsed.jarvis_replay
        assert parsed.csv.endswith("us30-vast.csv")

    def test_the_default_stand_is_still_the_configured_one(self):
        launcher = (ROOT / "us30.cmd").read_text(encoding="utf-8")
        argv = cmd_argv(
            launcher,
            **{
                "%DAGEN%": "180",
                "%SECTIES%": "section_us30_impulse_m1",
                "%EXITS%": "",
                "%CSVTAG%": "",
            },
        )
        parsed = build_parser_for_launcher(argv)
        assert not parsed.fixed_exits
        assert parsed.csv.endswith("us30.csv")


def build_parser_for_launcher(argv):
    from scripts.dry_run_sections import build_parser

    return build_parser().parse_args(argv)


class TestTheExitGridComparesEveryWayOfManagingATrade:
    """`beheer.cmd` / `--exit-grid`: which exit rule is best, per section.

    The owner asked to measure "ELKE MOGELIJKHEID" -- every break-even
    trigger, every stop placement, trailing, partials -- and to be told which
    one wins. The measuring is the easy half. The half that decides whether
    this costs him money is the verdict: comparing thirty rules and keeping
    the best is thirty chances to be fooled, and a section here holds about
    150 trades. So these tests pin the guard as hard as the arithmetic.
    """

    @staticmethod
    def _rows(count, variants, edge_for=None, edge=0.0, seed=11):
        import random
        from datetime import UTC, datetime, timedelta

        from scripts.dry_run_sections import Decision

        rng = random.Random(seed)
        base = datetime(2026, 3, 1, tzinfo=UTC)
        made = []
        for index in range(count):
            fixed = rng.choice([-1.0, -1.0, 2.0])
            grid = []
            for variant in variants:
                if variant.kind == "fixed":
                    grid.append((variant.label, fixed))
                elif edge_for and variant.label == edge_for:
                    grid.append((variant.label, fixed + edge + rng.gauss(0, 0.05)))
                else:
                    grid.append((variant.label, fixed + rng.gauss(0, 0.25)))
            made.append(
                Decision(
                    base + timedelta(hours=index),
                    "XAUUSD.i",
                    "section_ten_gold_m1",
                    "TRADE",
                    result_r=fixed,
                    grid_r=tuple(grid),
                )
            )
        return made

    # -- what is compared ---------------------------------------------------

    def test_the_triggers_the_owner_named_are_all_there(self):
        from scripts.dry_run_sections import BREAK_EVEN_TRIGGERS

        for asked in (0.10, 0.15, 0.20, 0.25, 0.35, 0.50):
            assert any(abs(t - asked) < 1e-9 for t in BREAK_EVEN_TRIGGERS), asked

    def test_the_sweep_runs_past_where_a_section_last_won(self):
        """Section ten lost on every trigger up to 0.75R and won on all three
        variants of 1.00R -- the last row of the table. A result sitting on the
        edge of a sweep is usually the sweep running out rather than the
        answer, so the sweep has to reach past it or the reader cannot tell
        those two apart."""
        from scripts.dry_run_sections import BREAK_EVEN_TRIGGERS

        assert max(BREAK_EVEN_TRIGGERS) > 1.00
        beyond = [t for t in BREAK_EVEN_TRIGGERS if t > 1.00]
        assert len(beyond) >= 2, "one cell past the edge cannot show a direction"

    def test_stops_are_placed_in_r_and_in_atr(self):
        # The live rule is an ATR offset, and an ATR offset is NOT a fixed
        # fraction of the stop -- 0.10 ATR is about 0.44R on section six's M5
        # stop and nearer 0.10R on an H1 one. Measuring only R would answer a
        # question this account does not ask.
        from scripts.dry_run_sections import exit_grid

        moves = [v for v in exit_grid(wide=True) if v.kind == "break-even"]
        assert any(v.lock_atr > 0 for v in moves), "no ATR-based stop placement is compared"
        assert any(v.lock_r > 0 for v in moves), "no R-based stop placement is compared"
        assert any(v.lock_r == 0 and v.lock_atr == 0 for v in moves), "no stop-at-entry row"

    def test_trailing_and_partials_are_compared_too(self):
        from scripts.dry_run_sections import exit_grid

        labels = [v.label for v in exit_grid(wide=True) if v.kind == "mechanism"]
        assert any(label.startswith("trail") for label in labels)
        assert any(label.startswith("part") for label in labels)
        assert any(label.startswith("lock") for label in labels)

    def test_wide_is_a_superset_of_narrow(self):
        from scripts.dry_run_sections import exit_grid

        narrow = {v.label for v in exit_grid(wide=False)}
        assert narrow < {v.label for v in exit_grid(wide=True)}

    def test_no_rule_puts_its_stop_on_the_price_that_arms_it(self):
        # "Move to +0.1R once the trade is +0.1R" is an instant exit at the
        # trigger, not a break-even rule, and it would have sat in the table
        # as a plausible row that always scratches.
        from scripts.dry_run_sections import exit_grid

        for variant in exit_grid(wide=True):
            if variant.kind == "break-even":
                assert variant.lock_r < variant.trigger_r, variant.label

    def test_every_label_is_unique(self):
        # The report reads results back BY LABEL. Two rules sharing one would
        # silently report the first one's numbers for both.
        from scripts.dry_run_sections import exit_grid

        for wide in (False, True):
            labels = [v.label for v in exit_grid(wide=wide)]
            assert len(labels) == len(set(labels))

    # -- isolation ----------------------------------------------------------

    def test_a_mechanism_row_switches_on_only_what_it_names(self):
        from config.loader import load_settings
        from scripts.dry_run_sections import _variant_management, exit_grid

        settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
        base = settings.trade_management
        trail = next(v for v in exit_grid(wide=True) if v.label.startswith("trail 2.0A"))
        built = _variant_management(base, trail)

        assert built.trailing_mode == "atr" and built.trailing_atr_multiple == 2.0
        # Everything else is a threshold the trade can never reach.
        assert built.break_even_at_r >= 99.0
        assert built.profit_lock_from_r >= 99.0
        assert built.peak_stall_arm_r >= 99.0
        assert built.giveback_arm_r >= 99.0
        assert built.capital_protection_at_equity_pct == 0.0

    def test_the_built_config_is_validated_and_not_just_copied(self):
        # `model_copy(update=...)` writes past pydantic, so an out-of-bounds
        # value would be accepted here and only misbehave later inside the bar
        # walk as a threshold that is quietly never reached.
        from config.loader import load_settings
        from scripts.dry_run_sections import ExitVariant, _variant_management

        settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
        from pydantic import ValidationError

        bad = ExitVariant(label="impossible", manage_fields=(("partial_close_fraction", 5.0),))
        with pytest.raises(ValidationError):
            _variant_management(settings.trade_management, bad)

    def test_trailing_rows_do_not_also_take_a_partial(self):
        # The two share `partial_close_at_r` as their arm, so without this the
        # trailing rows would quietly be measuring trail-plus-partial.
        from scripts.dry_run_sections import exit_grid

        for variant in exit_grid(wide=True):
            if variant.label.startswith("trail"):
                assert not variant.partial, variant.label
            if variant.label.startswith("part"):
                assert variant.partial, variant.label

    # -- the verdict, which is the part that protects the account -----------

    def test_pure_noise_produces_no_winner(self, capsys):
        """150 trades, 29 rules, and nothing but noise between them.

        SEED 2 IS CHOSEN, NOT ARBITRARY. On it the best of the 29 reaches a
        paired t of 2.17 and leads in BOTH halves -- so it clears the naive
        two-sided 5% bar of 1.96 and every hurdle except the one raised for
        having tried 29 rules. That makes this test fail the moment the
        correction is weakened, which a seed with a quieter winner would not:
        it would pass for the wrong reason and prove nothing.
        """
        from scripts.dry_run_sections import _manage_grid_report, exit_grid

        variants = exit_grid(wide=False)
        _manage_grid_report(self._rows(150, variants, seed=2), variants)
        out = capsys.readouterr().out
        assert "keep the exit as configured" in out
        assert "<-- best" not in out
        assert "beats the fixed exit" not in out
        # The refusal has to say it was best-of-many that disqualified it, or
        # the reader will go back to the table and pick the biggest number.
        assert "best-of-" in out

    def test_a_real_edge_is_found(self, capsys):
        from scripts.dry_run_sections import _manage_grid_report, exit_grid

        variants = exit_grid(wide=False)
        rows = self._rows(150, variants, edge_for="BE@0.35R", edge=0.30)
        _manage_grid_report(rows, variants)
        out = capsys.readouterr().out
        assert "VERDICT: BE@0.35R beats the fixed exit" in out
        assert "Not live yet" in out

    def test_the_bar_rises_with_the_number_of_rules_tried(self):
        # This is the whole guard. A single pre-registered rule needs t>1.96;
        # the best of thirty needs far more, or best-of-N noise ships.
        from scripts.dry_run_sections import _bonferroni_t

        assert _bonferroni_t(1) == pytest.approx(1.96, abs=0.01)
        assert _bonferroni_t(29) > 3.0
        assert _bonferroni_t(63) > _bonferroni_t(29)

    def test_a_rule_that_wins_in_only_one_half_is_refused(self, capsys):
        from datetime import UTC, datetime, timedelta

        from scripts.dry_run_sections import Decision, _manage_grid_report, exit_grid

        variants = exit_grid(wide=False)
        target = "BE@0.25R"
        base = datetime(2026, 3, 1, tzinfo=UTC)
        rows = []
        for index in range(150):
            fixed = -1.0
            # Enormous edge, but only in the early 60% of the period.
            lift = 5.0 if index < 90 else -0.2
            grid = []
            for v in variants:
                if v.kind == "fixed" or v.label != target:
                    grid.append((v.label, fixed))
                else:
                    grid.append((v.label, fixed + lift))
            rows.append(
                Decision(
                    base + timedelta(hours=index),
                    "XAUUSD.i",
                    "section_ten_gold_m1",
                    "TRADE",
                    result_r=fixed,
                    grid_r=tuple(grid),
                )
            )
        _manage_grid_report(rows, variants)
        out = capsys.readouterr().out
        assert "not in both halves" in out
        assert "beats the fixed exit" not in out

    def test_the_paired_t_is_paired(self):
        # Unpaired, these two sets of trades have vast variance and no
        # detectable difference; paired, the constant +0.20 is obvious. That
        # difference is the entire reason the grid resolves identical entries.
        from scripts.dry_run_sections import _paired_t

        swings = [(-1.0 if i % 3 else 4.0) for i in range(60)]
        # A constant improvement has no spread, so t is enormous -- not
        # literally infinite, because summing floats leaves a variance around
        # 1e-33 rather than a true zero.
        assert _paired_t([0.20] * 60) > 1e6
        assert _paired_t([s + 0.20 - s for s in swings]) > 1e6
        # The same trades under both rules: no difference, and no evidence of
        # one either.
        assert _paired_t([0.0] * 60) == 0.0
        # Too few observations is arithmetic, not evidence.
        assert _paired_t([1.0, 1.0, 1.0]) == 0.0
        # Sign is carried: a rule that consistently loses reads negative.
        assert _paired_t([-0.20] * 60) < -1e6

    def test_the_table_reads_results_by_label_not_by_position(self, capsys):
        # Reading `grid_r[i]` against the grid's i-th entry means adding or
        # reordering a rule silently relabels every column.
        from scripts.dry_run_sections import _manage_grid_report, exit_grid

        variants = exit_grid(wide=False)
        rows = self._rows(60, variants, edge_for="BE@0.35R", edge=0.30)
        shuffled = []
        for row in rows:
            row.grid_r = tuple(reversed(row.grid_r))
            shuffled.append(row)
        _manage_grid_report(shuffled, variants)
        out = capsys.readouterr().out
        # Same answer despite the stored order being reversed.
        assert "BE@0.35R" in out and "keep the exit as configured" not in out

    def test_a_measured_rule_missing_from_the_table_is_announced(self, capsys):
        from scripts.dry_run_sections import _manage_grid_report, exit_grid

        variants = exit_grid(wide=False)
        rows = self._rows(40, variants)
        for row in rows:
            row.grid_r = (*row.grid_r, ("BE@9.99R invented", 0.5))
        _manage_grid_report(rows, variants)
        out = capsys.readouterr().out
        assert "missing from this table" in out
        assert "BE@9.99R invented" in out

    # -- wiring -------------------------------------------------------------

    def test_the_flag_builds_the_grid_and_the_walk_gets_it(self):
        branch = SOURCE.split("exit_variants = (", 1)[1][:400]
        assert 'args.exit_grid == "alles"' in branch
        assert "args.manage_grid or args.exit_grid" in branch
        assert "manage_grid=exit_variants," in SOURCE

    def test_the_launcher_asks_for_the_grid_and_the_live_book(self):
        launcher = (ROOT / "beheer.cmd").read_text(encoding="utf-8")
        argv = cmd_argv(
            launcher,
            **{"%DAGEN%": "180", "%MARKTEN%": "--section-markets",
               "%BOEK%": "--live-only", "%GRID%": "kern", "%SECTIES%": "",
               "%CSVTAG%": ""},
        )
        parsed = build_parser_for_launcher(argv)
        assert parsed.exit_grid == "kern"
        assert parsed.jarvis_replay and parsed.live_only and parsed.section_markets
        assert parsed.days == 180

    def test_the_launcher_writes_a_separate_file_for_the_wide_grid(self):
        launcher = (ROOT / "beheer.cmd").read_text(encoding="utf-8")
        # The BEHAVIOUR, not the source line: the wide grid has to land on its
        # own filename. Asserting the literal `set CSVTAG=-alles` broke the
        # moment the tag was split in two to stop `goud alles` colliding --
        # which is a fix, not a regression, and a test should not have to be
        # edited to allow one.
        argv = cmd_argv(
            launcher,
            **{"%DAGEN%": "180", "%MARKTEN%": "--section-markets",
               "%BOEK%": "--live-only", "%GRID%": "alles", "%SECTIES%": "",
               "%CSVTAG%": "-alles"},
        )
        parsed = build_parser_for_launcher(argv)
        assert parsed.exit_grid == "alles"
        assert parsed.csv.endswith("beheer-alles.csv")

    def test_the_biggest_total_is_named_when_it_is_not_the_pick(self, capsys):
        """The owner asked which rule pays the MOST, and the verdict ranks on
        consistency instead. Both belong on screen: hiding the fattest column
        answers a different question than the one asked, and ranking on it
        would ship the rule that got lucky on the biggest trades."""
        from datetime import UTC, datetime, timedelta

        from scripts.dry_run_sections import Decision, _manage_grid_report, exit_grid

        variants = exit_grid(wide=False)
        steady, spiky = "BE@0.25R", "BE@0.50R"
        base = datetime(2026, 3, 1, tzinfo=UTC)
        rows = []
        for index in range(150):
            fixed = -1.0
            values = {}
            # A small, relentless improvement.
            values[steady] = fixed + 0.30
            # One enormous windfall, nothing the rest of the time.
            values[spiky] = fixed + (200.0 if index == 40 else 0.0)
            grid = [
                (v.label, values.get(v.label, fixed) if v.kind != "fixed" else fixed)
                for v in variants
            ]
            rows.append(
                Decision(
                    base + timedelta(hours=index),
                    "XAUUSD.i",
                    "section_ten_gold_m1",
                    "TRADE",
                    result_r=fixed,
                    grid_r=tuple(grid),
                )
            )
        _manage_grid_report(rows, variants)
        out = capsys.readouterr().out
        assert f"Biggest total: {spiky}" in out
        assert "not the pick" in out
        assert f"VERDICT: {steady} beats the fixed exit" in out

    def test_an_atr_lock_and_an_r_lock_really_differ_in_the_walk(self):
        """`stop_offset` returns a PRICE, and the two kinds of lock have to
        land the stop in different places or the ATR half of this grid is an
        elaborate no-op printing a second copy of the R column."""
        from datetime import UTC
        from types import SimpleNamespace

        import pandas as pd

        from core.types import Direction
        from scripts.dry_run_sections import ExitVariant, _resolve

        index = pd.date_range("2026-06-01", periods=6, freq="min", tz=UTC)
        # Runs to +0.6R, falls back through entry, and stops out.
        frame = pd.DataFrame(
            {
                "high": [104, 106, 106, 101, 100, 100],
                "low": [99, 103, 100, 99, 89, 89],
                "close": [103, 105, 101, 100, 90, 90],
            },
            index=index,
        )
        idea = SimpleNamespace(direction=Direction.LONG, entry=100, stop_loss=90, take_profit=120)
        risk_price, atr = 10.0, 20.0

        def managed_r(variant):
            _f, _fa, managed, _ma = _resolve(
                frame,
                index[0],
                idea,
                6,
                manage=(variant.trigger_r, variant.stop_offset(risk_price, atr)),
            )
            return managed

        at_entry = ExitVariant(label="e", trigger_r=0.25)
        in_r = ExitVariant(label="r", trigger_r=0.25, lock_r=0.1)
        in_atr = ExitVariant(label="a", trigger_r=0.25, lock_atr=0.10)

        assert in_r.stop_offset(risk_price, atr) == pytest.approx(1.0)
        assert in_atr.stop_offset(risk_price, atr) == pytest.approx(2.0)
        # Unmanaged this trade loses its full R; each lock keeps more, and the
        # ATR lock keeps more than the R lock because on this clock it is the
        # wider offset.
        fixed, _fa, _m, _ma = _resolve(frame, index[0], idea, 6)
        assert fixed == pytest.approx(-1.0)
        assert managed_r(at_entry) == pytest.approx(0.0)
        assert managed_r(in_r) == pytest.approx(0.1)
        assert managed_r(in_atr) == pytest.approx(0.2)

    def test_the_launcher_can_measure_one_or_two_sections(self):
        """`beheer.cmd 180 goud` -- section six and ten only.

        Both trade gold and nothing else, so narrowing to them walks ONE
        market instead of five. That is the difference between a run the owner
        starts and waits for and one he abandons.
        """
        launcher = (ROOT / "beheer.cmd").read_text(encoding="utf-8")
        cases = {
            "goud": {"section_six_gold_m5", "section_ten_gold_m1"},
            "zes": {"section_six_gold_m5"},
            "tien": {"section_ten_gold_m1"},
            "vijf": {"section_five_ndx100_m5"},
        }
        from config.loader import load_settings

        settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
        for word, wanted in cases.items():
            assert f'if /i "%~1"=="{word}" set SECTIES=--only ' in launcher, word
            named = launcher.split(f'"%~1"=="{word}" set SECTIES=--only ', 1)[1].split("\n", 1)[0]
            named = named.strip()
            if "," in named:
                assert named.startswith('"') and named.endswith('"'), (
                    f"{word} passes a comma list unquoted; cmd may split it into two arguments"
                )
            assert set(named.strip('"').split(",")) == wanted, word
            # `beheer` is also the research tool for a benched section. It may
            # name one that cannot spend live money, but never one disabled so
            # completely that the run would only manufacture empty output.
            assert all(getattr(settings.analysis, name).enabled for name in wanted)

    def test_two_choices_cannot_collide_on_one_filename(self):
        """`beheer.cmd 180 goud alles` is offered in the help text, and with a
        single CSVTAG the second word overwrote the first -- so a two-section
        run over 78 rules landed on the same filename as an all-section run
        over 78 rules, and the later one silently replaced the earlier."""
        launcher = (ROOT / "beheer.cmd").read_text(encoding="utf-8")
        assert "set CSVTAG=%SECTAG%%GRIDTAG%" in launcher
        # No branch may write the combined variable directly, or the split is
        # undone by whichever line runs last.
        for line in launcher.splitlines():
            if line.strip().startswith("if /i") and "CSVTAG=" in line:
                raise AssertionError(f"a branch still overwrites CSVTAG: {line}")

        argv = cmd_argv(
            launcher,
            **{
                "%DAGEN%": "180",
                "%MARKTEN%": "--section-markets",
                "%BOEK%": "--live-only",
                "%GRID%": "alles",
                "%SECTIES%": '--only "section_six_gold_m5,section_ten_gold_m1"',
                "%CSVTAG%": "-goud-alles",
            },
        )
        parsed = build_parser_for_launcher(argv)
        assert parsed.exit_grid == "alles"
        assert set(parsed.only.split(",")) == {"section_six_gold_m5", "section_ten_gold_m1"}
        assert parsed.csv.endswith("beheer-goud-alles.csv")

    def test_the_default_still_measures_everything_live(self):
        launcher = (ROOT / "beheer.cmd").read_text(encoding="utf-8")
        argv = cmd_argv(
            launcher,
            **{
                "%DAGEN%": "180",
                "%MARKTEN%": "--section-markets",
                "%BOEK%": "--live-only",
                "%GRID%": "kern",
                "%SECTIES%": "",
                "%CSVTAG%": "",
            },
        )
        parsed = build_parser_for_launcher(argv)
        assert parsed.only == "", "the default must not narrow to a section"
        assert parsed.live_only
        assert parsed.csv.endswith("beheer.csv")


class TestTheDoorstepCloseIsMeasuredAndNotOnlyLive:
    """The replay has to resolve trades the way the account now exits them.

    `PositionManager._close_at_the_doorstep` takes a winner that has arrived
    within a hair of its target. If the replay kept resolving on the untouched
    target, `hoeveel.cmd` would report the OLD exit and the owner would be
    measuring a system he is not running. That gap -- a rule that is live, is
    correct, and is absent from the measurement -- is the single most repeated
    defect in this repository.
    """

    @staticmethod
    def _frame():
        from datetime import UTC

        import pandas as pd

        index = pd.date_range("2026-06-01", periods=5, freq="min", tz=UTC)
        # Runs to 109.5, one tick short of the 110.0 target, then collapses
        # through the stop. Without the doorstep close this is a full loss.
        return pd.DataFrame(
            {
                "high": [104.0, 109.5, 109.5, 101.0, 99.0],
                "low": [99.5, 103.0, 100.0, 97.0, 97.0],
                "close": [103.0, 109.0, 101.0, 98.0, 97.0],
            },
            index=index,
        )

    @staticmethod
    def _idea():
        from types import SimpleNamespace

        from core.types import Direction

        return SimpleNamespace(
            direction=Direction.LONG, entry=100.0, stop_loss=98.0, take_profit=110.0
        )

    def test_a_trade_that_stalls_on_the_doorstep_is_a_winner_not_a_loser(self):
        from scripts.dry_run_sections import _resolve

        frame, idea = self._frame(), self._idea()
        index = frame.index

        # Untouched target: the high of 109.5 never reaches 110.0, so this runs
        # on to the stop and books a full -1R.
        plain, _at, _m, _ma = _resolve(frame, index[0], idea, 5)
        assert plain == pytest.approx(-1.0)

        # 1.5 spreads of 0.4 is 0.6, capped at 10% of the 10.0 reward = 1.0.
        # The effective target is 109.4, which the second bar reaches.
        near, near_at, _m2, _ma2 = _resolve(
            frame, index[0], idea, 5, near_target_tolerance=0.6, near_target_max_share=0.10
        )
        assert near is not None and near > 0
        assert near_at == index[1]

    def test_the_r_credited_is_the_r_of_the_price_actually_taken(self):
        """Crediting the full reward for an exit taken short of the target
        would pay the replay for money the account never receives -- and it
        would do it on every winner, which is where an optimistic rounding
        does the most damage."""
        from scripts.dry_run_sections import _resolve

        frame, idea = self._frame(), self._idea()
        near, _at, _m, _ma = _resolve(
            frame, frame.index[0], idea, 5, near_target_tolerance=0.6, near_target_max_share=0.10
        )
        # Entry 100, stop 98 so 1R = 2.0. Target 110 is +5R; taken at 109.4
        # that is +4.7R, and it must be the smaller number.
        assert near == pytest.approx(4.7)
        assert near < (110.0 - 100.0) / 2.0

    def test_the_share_cap_binds_in_the_replay_too(self):
        from scripts.dry_run_sections import _resolve

        frame, idea = self._frame(), self._idea()
        # A huge spread tolerance, capped at 1% of the move = 0.1, so the
        # effective target is 109.9 and the 109.5 high does not reach it.
        result, _at, _m, _ma = _resolve(
            frame, frame.index[0], idea, 5, near_target_tolerance=9.0, near_target_max_share=0.01
        )
        assert result == pytest.approx(-1.0)

    def test_zero_tolerance_changes_nothing(self):
        from scripts.dry_run_sections import _resolve

        frame, idea = self._frame(), self._idea()
        a, a_at, _m, _ma = _resolve(frame, frame.index[0], idea, 5)
        b, b_at, _m2, _ma2 = _resolve(
            frame, frame.index[0], idea, 5, near_target_tolerance=0.0, near_target_max_share=0.10
        )
        assert (a, a_at) == (b, b_at)

    def test_a_short_gets_the_target_moved_the_other_way(self):
        from datetime import UTC
        from types import SimpleNamespace

        import pandas as pd

        from core.types import Direction
        from scripts.dry_run_sections import _resolve

        index = pd.date_range("2026-06-01", periods=4, freq="min", tz=UTC)
        frame = pd.DataFrame(
            {"high": [100.5, 97.0, 103.0, 103.0], "low": [96.0, 90.5, 99.0, 99.0],
             "close": [97.0, 91.0, 102.0, 103.0]},
            index=index,
        )
        idea = SimpleNamespace(
            direction=Direction.SHORT, entry=100.0, stop_loss=102.0, take_profit=90.0
        )
        plain, _at, _m, _ma = _resolve(frame, index[0], idea, 4)
        assert plain == pytest.approx(-1.0)
        near, _at2, _m2, _ma2 = _resolve(
            frame, index[0], idea, 4, near_target_tolerance=0.6, near_target_max_share=0.10
        )
        assert near == pytest.approx(4.7)

    def test_the_walk_hands_the_tolerance_to_both_the_trade_and_the_grid(self):
        # The wiring, not the helper. An exit grid resolved against a target
        # the account no longer uses would judge every rule -- including the
        # fixed baseline they are all compared to -- against the wrong price.
        body = SOURCE.split("doorstep_spread = entry_spread_price", 1)[1]
        assert body.count("near_target_tolerance=near_target") >= 2
        assert body.count("near_target_max_share=near_target_share") >= 2

    def test_the_live_config_switches_it_on_in_spreads(self):
        from config.loader import load_settings

        settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
        tm = settings.trade_management
        assert tm.close_near_target_spreads > 0.0, "the owner asked for this and it is off"
        assert 0.0 < tm.close_near_target_max_share <= 0.5, (
            "without the share cap a wide spread fires this halfway to the target"
        )


class TestSectionTenIsNoLongerRefusedByTheReachGate:
    """`TARGET_RARELY_REACHED` refused 312 section-ten setups over 180 days.

    84.3% of them would have won, together +38.86 R -- against the +24.23 R
    the section actually earned in the same window. The gate cost more than the
    section made, and section six, which was already on the advisory list, took
    no refusals from it at all.
    """

    @staticmethod
    def _settings():
        from config.loader import load_settings

        return load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)

    def test_section_ten_is_advisory(self):
        families = self._settings().analysis.confluence.target_reach_advisory_families
        assert any("section_ten_gold_m1" in family for family in families)

    def test_the_gate_still_runs_and_is_only_advisory(self):
        # ADVISORY, NOT DELETED. The gate keeps measuring and keeps appearing
        # in the journal; what lapses is its right to refuse the trade. A
        # removed gate cannot be judged later, and this one has not been.
        assert "TARGET_RARELY_REACHED" in SOURCE
        assert "target_reach_advisory_families" in SOURCE

    def test_both_the_replay_and_the_runner_read_the_same_list(self):
        """Two lists would mean the measurement and the account disagree about
        which sections the gate may stop -- and the replay is what the decision
        to change this was made on."""
        runner = (ROOT / "runner" / "service.py").read_text(encoding="utf-8")
        assert "target_reach_advisory_families" in runner
        assert "target_reach_advisory_families" in SOURCE

    def test_the_two_remaining_live_sections_use_their_measured_advisory_policy(self):
        # The allowlist was intentionally reduced to the two measured gold
        # sections. Both already had a measured advisory reach policy; this is
        # not a blanket change to the shadow modules.
        families = set(self._settings().analysis.confluence.target_reach_advisory_families)
        live = set(self._settings().analysis.confluence.live_enabled_modules)
        assert live == {"section_six_gold_m5", "section_ten_gold_m1"}
        assert live <= families


class TestTheSharedBookCountsPositionsAndNamesItsRefusals:
    """Two defects that between them sent an afternoon after the wrong number.

    The owner read "429 signals dropped: every slot was already busy" and
    "section six missed 296 trades worth +38.28 R", raised the account cap from
    four slots to ten, re-measured, and the number did not move. It could not
    have: the cap was never what refused them.
    """

    @staticmethod
    def _rows(spec):
        from datetime import UTC, datetime, timedelta

        from scripts.dry_run_sections import Decision

        base = datetime(2026, 3, 1, tzinfo=UTC)
        made = []
        for minute, module, symbol, hold, direction in spec:
            row = Decision(
                base + timedelta(minutes=minute), symbol, module, "TRADE", direction=direction
            )
            row.exit_at = base + timedelta(minutes=minute + hold)
            made.append(row)
        return made

    def test_the_cap_counts_open_positions_and_not_busy_markets(self):
        """Three sections sharing gold are THREE positions, not one market.

        Keyed by symbol, a cap of two allowed three simultaneous positions --
        more risk than the account sanctions, reported as if it were less.
        """
        from scripts.dry_run_sections import _under_the_slot_cap

        rows = self._rows(
            [
                (0, "a", "XAUUSD.i", 600, "LONG"),
                (1, "b", "XAUUSD.i", 600, "LONG"),
                (2, "c", "XAUUSD.i", 600, "LONG"),
            ]
        )
        taken, refused = _under_the_slot_cap(rows, 2, share_between_sections=True)
        assert len(taken) == 2, "a cap of two allowed a third simultaneous position"
        assert refused[id(rows[2])] == "ACCOUNT_POSITION_LIMIT"

    def test_a_cap_above_the_market_count_could_never_bind(self):
        """The run walks four markets. Keyed by symbol the count could not pass
        four, so a cap of ten was unreachable and raising it was a no-op --
        while the report went on blaming it for every refusal."""
        from scripts.dry_run_sections import _under_the_slot_cap

        rows = self._rows([(m, f"s{m}", "XAUUSD.i", 600, "LONG") for m in range(12)])
        taken, refused = _under_the_slot_cap(rows, 10, share_between_sections=True)
        assert len(taken) == 10
        assert {refused[id(r)] for r in rows[10:]} == {"ACCOUNT_POSITION_LIMIT"}

    def test_a_sections_own_position_is_named_as_such(self):
        """THE ONE THAT MATTERS. Section six's 296 missed trades are its own
        open gold position in the way, not a full account. The two have
        different fixes and only one of them is the slot count."""
        from scripts.dry_run_sections import _under_the_slot_cap

        rows = self._rows(
            [
                (0, "section_six_gold_m5", "XAUUSD.i", 60, "LONG"),
                (5, "section_six_gold_m5", "XAUUSD.i", 60, "LONG"),
            ]
        )
        taken, refused = _under_the_slot_cap(rows, 100, share_between_sections=True)
        assert len(taken) == 1
        assert refused[id(rows[1])] == "SYMBOL_ALREADY_HELD", (
            "its own position blocked it and the report must not call that a full account"
        )

    def test_another_sections_hold_is_named_differently_again(self):
        from scripts.dry_run_sections import _under_the_slot_cap

        rows = self._rows(
            [
                (0, "section_six_gold_m5", "XAUUSD.i", 60, "LONG"),
                (5, "section_ten_gold_m1", "XAUUSD.i", 60, "LONG"),
            ]
        )
        taken, refused = _under_the_slot_cap(rows, 100, share_between_sections=False)
        assert len(taken) == 1
        assert refused[id(rows[1])] == "SYMBOL_HELD_BY_ANOTHER_SECTION"

    def test_raising_the_slot_count_cannot_free_a_symbol_rule(self):
        """The experiment the owner actually ran, as a test: more slots do not
        buy a section a second position in a symbol it already holds."""
        from scripts.dry_run_sections import _under_the_slot_cap

        rows = self._rows(
            [(m, "section_six_gold_m5", "XAUUSD.i", 60, "LONG") for m in range(0, 300, 5)]
        )
        counts = {
            slots: len(_under_the_slot_cap(rows, slots, share_between_sections=True)[0])
            for slots in (4, 10, 100)
        }
        assert len(set(counts.values())) == 1, (
            f"the slot count changed the answer: {counts} -- then the symbol rule is not what binds"
        )

    def test_every_refusal_carries_a_reason_the_report_can_print(self):
        from scripts.dry_run_sections import _under_the_slot_cap

        rows = self._rows(
            [
                (0, "a", "XAUUSD.i", 600, "LONG"),
                (1, "a", "XAUUSD.i", 600, "LONG"),
                (2, "b", "NDX100.i", 600, "LONG"),
                (3, "c", "SPX500.i", 600, "LONG"),
            ]
        )
        taken, refused = _under_the_slot_cap(rows, 2, share_between_sections=True)
        for row in rows:
            if row not in taken:
                assert id(row) in refused, "a refusal with no name reads as an oversight"
        # AND THE CALLER HAS A SENTENCE FOR EACH ONE. Read from the `why` map
        # itself rather than from a window of characters after the call: the
        # window version broke the moment an unrelated print was added above
        # it, which is a test that fails for the wrong reason.
        table = SOURCE.split("        why = {", 1)[1].split("\n        }", 1)[0]
        for name in ("SYMBOL_ALREADY_HELD", "SYMBOL_HELD_BY_ANOTHER_SECTION",
                     "ACCOUNT_POSITION_LIMIT"):
            assert name in table, f"{name} can be produced and the report cannot name it"
        # Every reason the walk can emit must be in that map, or the caller
        # raises a KeyError on a real run instead of printing a refusal.
        emitted = set(re.findall(r'refused\[id\(trade\)\] = \(?\s*"([A-Z_]+)"', SOURCE))
        for name in emitted:
            assert name in table, f"{name} is emitted and has no sentence"

    def test_the_contract_no_longer_calls_every_refusal_a_missing_slot(self):
        assert "arrived with no slot free" not in SOURCE, (
            "that sentence blamed the slot count for refusals it never made"
        )

    def test_stacking_is_measurable_and_off_by_default(self):
        """The owner's question: is section six leaving money on the table by
        refusing a setup while it already holds gold?

        The honest answer is a number, not an argument -- and the number has to
        be read with the drawdown, because stacking the same model on the same
        market at the same time is not a second bet, it is the first one at a
        larger size.
        """
        from scripts.dry_run_sections import _under_the_slot_cap, build_parser

        assert build_parser().parse_args(["--days", "30"]).legs_per_symbol == 1, (
            "the account holds one position per section per symbol; that is the default"
        )
        rows = self._rows(
            [(m, "section_six_gold_m5", "XAUUSD.i", 60, "LONG") for m in range(0, 300, 5)]
        )
        counts = {
            legs: len(
                _under_the_slot_cap(rows, 100, share_between_sections=True, legs_per_symbol=legs)[0]
            )
            for legs in (1, 2, 3)
        }
        assert counts[2] > counts[1] and counts[3] > counts[2], counts

    def test_stacking_never_escapes_the_account_cap(self):
        """More legs may not buy more positions than the book allows -- that
        would measure an account nobody could have held."""
        from scripts.dry_run_sections import _under_the_slot_cap

        rows = self._rows([(m, "s", "XAUUSD.i", 600, "LONG") for m in range(6)])
        taken, refused = _under_the_slot_cap(
            rows, 2, share_between_sections=True, legs_per_symbol=99
        )
        assert len(taken) == 2
        assert set(refused.values()) == {"ACCOUNT_POSITION_LIMIT"}

    def test_stacking_does_not_let_a_section_join_against_itself(self):
        """A second leg is the same idea again, not the opposite one. Allowing
        a stacked short under a stacked long would be flat exposure bought with
        two spreads."""
        from scripts.dry_run_sections import _under_the_slot_cap

        rows = self._rows(
            [
                (0, "s", "XAUUSD.i", 600, "LONG"),
                (1, "s", "XAUUSD.i", 600, "SHORT"),
            ]
        )
        taken, _why = _under_the_slot_cap(
            rows, 10, share_between_sections=True, legs_per_symbol=3, refuse_opposite=True
        )
        assert len(taken) == 1

    def test_the_launcher_offers_it_and_says_what_to_read(self):
        launcher = (ROOT / "hoeveel.cmd").read_text(encoding="utf-8")
        assert "--legs-per-symbol 2" in launcher
        assert "stapel3" in launcher
        # It must warn, in the launcher itself, that more R is expected and
        # means nothing alone. The number gets screenshotted; the caveat does
        # not travel with it unless it is on the same screen.
        assert "terugval" in launcher.casefold()
        argv = cmd_argv(
            launcher,
            **{
                "%DAGEN%": "180",
                "%MARKTEN%": "--section-markets",
                "%BOEK%": "--live-only",
                "%STAPEL%": "--legs-per-symbol 2",
                "%CSVTAG%": "-stapel2",
            },
        )
        parsed = build_parser_for_launcher(argv)
        assert parsed.legs_per_symbol == 2
        assert parsed.csv.endswith("hoeveel-stapel2.csv")

    def test_the_plain_launcher_still_measures_the_account(self):
        launcher = (ROOT / "hoeveel.cmd").read_text(encoding="utf-8")
        argv = cmd_argv(
            launcher,
            **{
                "%DAGEN%": "180",
                "%MARKTEN%": "--section-markets",
                "%BOEK%": "--live-only",
                "%STAPEL%": "",
                "%CSVTAG%": "",
            },
        )
        parsed = build_parser_for_launcher(argv)
        assert parsed.legs_per_symbol == 1
        assert parsed.csv.endswith("hoeveel.csv")

    def test_the_live_block_names_the_rule_that_refused(self, capsys):
        """The 180-day run printed "432 signals dropped: every slot was already
        busy" while the refusal table on the same screen said
        SYMBOL_ALREADY_HELD 432 and ACCOUNT_POSITION_LIMIT zero.

        Two lines contradicting each other, and the owner acted on the wrong
        one: he raised the account cap from four slots to ten, re-measured, and
        nothing moved. A refusal has to be named by the rule that made it.
        """
        from config.loader import load_settings
        from scripts.dry_run_sections import _live_config_report

        settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
        rows = self._rows(
            [
                (0, "section_six_gold_m5", "XAUUSD.i", 600, "LONG"),
                (5, "section_six_gold_m5", "XAUUSD.i", 600, "LONG"),
            ]
        )
        for row in rows:
            row.result_r = 1.0
            row.pass_key = (row.module, "M5")
        # BOTH still read TRADE, which is the state this block actually sees:
        # it runs before the caller stamps the refusals. A fixture that
        # pre-stamped them would have passed against code that prints nothing.
        results = {("section_six_gold_m5", "M5"): rows}
        _live_config_report(results, settings, 233.02, 180)
        out = capsys.readouterr().out

        assert "every slot was already busy" not in out
        assert "already held that market" in out
        assert "raising it cannot buy a trade" in out, (
            "when the cap refused nothing, the report has to say so or it will be blamed again"
        )


def test_trend_grid_reports_wins_cut_losses_cut_and_net(capsys) -> None:
    from datetime import datetime

    from scripts.dry_run_sections import Decision, _trend_grid_report

    def row(result: float, m5: int, m15: int) -> Decision:
        return Decision(
            datetime(2026, 9, 1, tzinfo=UTC),
            "NDX100",
            "section_five_ndx100_m5",
            "TRADE",
            direction="LONG",
            result_r=result,
            pass_key=("section_five_ndx100_m5", "M5"),
            trend_m5=m5,
            trend_m15=m15,
        )

    _trend_grid_report([row(1.0, 1, 1), row(-1.0, -1, -1)], managed=False)
    output = capsys.readouterr().out

    assert "M5 aligned" in output
    assert "+0.00" in output  # baseline
    assert "+1.00" in output  # loss removed / net saved / filtered result


def test_trendcheck_is_a_fast_entry_grid_not_an_exit_grid() -> None:
    from scripts.dry_run_sections import build_parser

    launcher = (ROOT / "trendcheck.cmd").read_text(encoding="utf-8")
    assert "--trend-grid" in launcher
    assert "--exit-grid" not in launcher
    parsed = build_parser().parse_args(["--trend-grid"])
    assert parsed.trend_grid
    assert "required_frames.update({Timeframe.M5, Timeframe.M15})" in SOURCE


def test_foutcheck_is_shadow_only_and_defaults_to_180_days() -> None:
    from scripts.dry_run_sections import build_parser

    launcher = (ROOT / "foutcheck.cmd").read_text(encoding="utf-8")
    assert "set DAGEN=180" in launcher
    assert "--fault-exit-grid" in launcher
    assert "--jarvis-replay" in launcher
    assert "section_five_ndx100_m5,section_six_gold_m5" in launcher
    assert "--manage-grid" not in launcher
    assert build_parser().parse_args(["--fault-exit-grid"]).fault_exit_grid


def test_fault_exit_can_act_immediately_using_pre_entry_context() -> None:
    from types import SimpleNamespace

    import pandas as pd

    from core.types import Direction
    from scripts.dry_run_sections import _fault_exit_grid

    index = pd.date_range("2026-01-01", periods=190, freq="1min", tz="UTC")
    prices = [100.0 + minute * 0.002 for minute in range(160)]
    prices += [100.32 - minute * 0.22 for minute in range(30)]
    frame = pd.DataFrame(
        {
            "open": prices,
            "high": [price + 0.03 for price in prices],
            "low": [price - 0.03 for price in prices],
            "close": prices,
        },
        index=index,
    )
    idea = SimpleNamespace(
        entry=100.32,
        stop_loss=90.32,
        direction=Direction.LONG,
    )
    measured = dict(
        _fault_exit_grid(frame, index[159], idea, -1.0, index[-1], 0.0)
    )
    assert measured["LOSS@-0.15R"] > -1.0
    assert measured["LOSS@-0.25R"] > -1.0

"""The readers the engine actually votes with.

WHY THIS FILE EXISTS. Section six went live and every cycle logged

    AttributeError: 'MomentumScalp' object has no attribute 'analyze'
    candidate analysis failed; continuing with the rest of the batch

on every candidate. `MomentumScalp` was the name of a playbook as well as of a
detector; `from analysis.playbooks import MomentumScalp` came later in
`runner/service.py` and won, so the engine was handed a playbook object. The
damage was not confined to section six: the exception fired inside the
comprehension that scores ALL modules, so every candidate of every cycle failed
and the account traded nothing at all.

Nine hundred tests passed while that was true, because every one of them built
its modules by hand and no test ever built the list the runner builds. So these
tests do exactly that and nothing else — they construct the real list from the
shipped config and ask the questions the crash answered the hard way.
"""

from __future__ import annotations

from typing import ClassVar

from config.loader import DEFAULT_CONFIG_PATH, load_settings
from runner.service import build_analysis_modules


def settings():  # type: ignore[no-untyped-def]
    return load_settings(DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False)


def modules():  # type: ignore[no-untyped-def]
    return build_analysis_modules(settings())


class TestTheEngineCanActuallyCallThem:
    def test_every_reader_answers_to_analyze(self) -> None:
        """The bug, in one assertion."""
        for module in modules():
            assert callable(getattr(module, "analyze", None)), type(module).__name__

    def test_every_reader_has_a_name(self) -> None:
        for module in modules():
            name = getattr(module, "name", None)
            assert isinstance(name, str) and name, type(module).__name__


class TestNoTwoReadersShareAName:
    def test_names_are_unique(self) -> None:
        """A duplicate name is not cosmetic. `module_scores`, `weights`,
        `live_enabled_modules`, the evidence families and the section breakers
        are all keyed by it, so two readers sharing one means two things being
        credited, weighted and stopped as though they were one."""
        names = [module.name for module in modules()]

        assert len(names) == len(set(names)), sorted(names)

    def test_the_candle_reader_does_not_borrow_the_playbooks_name(self) -> None:
        """`momentum_scalp` belongs to a playbook with a measured record of
        -0.561R over 307 trades, switched off on purpose and documented as
        such in the same config file. A new reader living under that name
        would inherit its history in every report that groups by module."""
        names = {module.name for module in modules()}

        assert "candle_momentum" in names
        assert "momentum_scalp" not in names


class TestTheConfigAndTheEngineAgree:
    """Every failure this session has had the same shape: a number is computed
    and never reaches the thing that needs it. A weight for a module the engine
    does not build is that shape exactly — the config says the reader carries
    0.6 of a vote and no such reader ever votes."""

    def test_every_weighted_module_exists(self) -> None:
        names = {module.name for module in modules()}
        weights = settings().analysis.confluence.weights

        assert set(weights) <= names, sorted(set(weights) - names)

    def test_every_live_enabled_module_exists(self) -> None:
        names = {module.name for module in modules()}
        live = set(settings().analysis.confluence.live_enabled_modules)

        assert live <= names, sorted(live - names)

    def test_every_watched_section_exists(self) -> None:
        """A breaker on a module that is not built would read as armed forever
        while protecting nothing."""
        names = {module.name for module in modules()}
        watched = set(settings().risk.section_breakers)

        assert watched <= names, sorted(watched - names)


def test_the_module_backtest_grades_the_modules_the_account_runs() -> None:
    """A report headed "which detector actually makes money" has to be about
    the detectors that are running.

    `scripts/backtest_modules.py` named its modules by hand. The account built
    eighteen and the script built fifteen, so three were graded by nothing at
    all: `drift_burst`, `basket_divergence`, and `candle_momentum` -- section
    six, the newest thing on the account and the one most in need of an answer.
    Nothing was wrong with the report; the three simply were not in it, and a
    missing row looks exactly like a module with no trades.

    `build_analysis_modules` was extracted so this file could stop keeping a
    second copy, and its docstring says so. This test is what makes that true
    rather than intended.
    """
    from scripts.backtest_modules import build_engine

    settings = load_settings(
        DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
    )

    graded = [module.name for module in build_engine(settings).modules]
    running = [module.name for module in build_analysis_modules(settings)]

    assert graded == running


class TestALiveSectionAnswersToTheNameItsBrokerUses:
    """The suffix bug, asked of every section instead of fixed per module.

    `ctx.symbol` carries the BROKER's name. Eightcap suffixes almost
    everything -- `BTCUSD.i`, `USDJPY.i`, `SPX500.i` -- while the configs say
    `BTCUSD`, `USDJPY`, `SPX500`. A section that compares the two raw is
    silent on every bar, forever, and NOTHING says why: no error, no refusal
    row, no log line, just a detector that never fires.

    IT HAS BEEN FIXED THREE TIMES -- `SectionXauJpy`, `SectionElevenLegs`, and
    `GoldCrossDiscovery` on the day its three sections were promoted to real
    money. Every time it survived review because the DRY RUN walks plain
    symbol names: the replay that granted permission compared `BTCUSD` to
    `BTCUSD` and passed, while live could not work. A green backtest is not
    evidence against this bug; it is the reason the bug is invisible.

    Gold is why it went unnoticed for so long. `symbol_overrides` maps XAUUSD
    to itself, so sections six and ten -- the two that have actually been
    paying -- compare `XAUUSD` to `XAUUSD` and are fine.
    """

    #: Live sections that filter on a raw config name and do NOT resolve the
    #: broker's spelling. Listed rather than silently skipped, because an
    #: unlisted exemption is the same silence this whole class is about.
    #:
    #: Both are SPX500, both predate this test, and neither is changed here:
    #: whether Eightcap lists `SPX500` or `SPX500.i` is a question for the
    #: terminal, not for a test, and guessing wrong would switch OFF a section
    #: that is currently trading. Recorded so somebody checks it against a
    #: live Market Watch, and so a NEW section cannot join them by accident.
    UNRESOLVED: ClassVar[set[str]] = {
        "failed_session_breakout",
        "section_eight_trend_day_h1",
    }

    def _live_settings(self):
        from pathlib import Path

        from config.loader import load_settings

        return load_settings(
            overlay=Path(__file__).resolve().parents[1] / "config" / "eightcap.yaml",
            env_overrides=False,
        )

    def _single_market_sections(self, settings):
        from runner.service import build_analysis_modules

        for module in build_analysis_modules(settings):
            config = getattr(settings.analysis, module.name, None)
            if config is None:
                continue
            declared = tuple(getattr(config, "allowed_symbols", ()) or ()) or (
                (getattr(config, "symbol", ""),) if getattr(config, "symbol", "") else ()
            )
            # A section trading a basket, or nothing in particular, is not what
            # this is about.
            if len(declared) == 1:
                yield module, declared[0]

    def test_every_resolved_section_resolves_to_the_right_name(self) -> None:
        settings = self._live_settings()
        checked = 0
        for module, canonical in self._single_market_sections(settings):
            known = getattr(module, "broker_symbol", None)
            if known is None:
                continue
            expected = settings.instruments.broker_symbol(canonical)
            assert known == expected, (
                f"{module.name} answers to {known!r} but this broker prints "
                f"{expected!r}; live it would be silent on every bar"
            )
            checked += 1
        assert checked, "no section resolves a broker symbol; that cannot be right"

    def test_no_live_section_filters_on_a_raw_name_without_being_listed(self) -> None:
        """The one that catches the NEXT occurrence.

        A live single-market section either resolves its broker name or is in
        `UNRESOLVED` with a reason written above it. There is no third state,
        and in particular there is no silent one.
        """
        settings = self._live_settings()
        live = set(settings.analysis.confluence.live_enabled_modules)
        offenders = [
            module.name
            for module, canonical in self._single_market_sections(settings)
            if module.name in live and getattr(module, "broker_symbol", None) is None
            # ONLY WHERE THE TWO SPELLINGS ACTUALLY DIFFER. Section ten trades
            # XAUUSD, which `symbol_overrides` maps to itself, so its raw
            # comparison is correct on this broker and demanding a resolver
            # would be noise. That is also exactly why the bug hid for months:
            # the two sections that were paying happened to be the two whose
            # names match.
            and settings.instruments.broker_symbol(canonical) != canonical
            and module.name not in self.UNRESOLVED
        ]
        assert not offenders, (
            f"{sorted(offenders)} are LIVE on real money, trade one market each, and "
            f"compare the broker's symbol against the config's spelling raw. On this "
            f"broker that is a section that never fires. Resolve it with "
            f"`settings.instruments.broker_symbol(...)` where the module is built."
        )

    def test_the_unresolved_list_does_not_outlive_its_entries(self) -> None:
        """An exemption for a section that has since been fixed, or removed,
        is a stale allowance that quietly re-opens the hole for its name."""
        settings = self._live_settings()
        built = {module.name: module for module, _c in self._single_market_sections(settings)}
        for name in self.UNRESOLVED:
            assert name in built, f"{name} is exempted here and no longer exists"
            assert getattr(built[name], "broker_symbol", None) is None, (
                f"{name} now resolves its broker symbol -- take it out of UNRESOLVED "
                f"so the exemption cannot cover a future regression"
            )

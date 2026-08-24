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

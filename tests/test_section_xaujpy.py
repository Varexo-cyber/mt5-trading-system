"""Sections eleven, twelve and thirteen: one searched mechanism on XAUJPY.

The section this replaced was a fitted model whose untouched holdout came back
negative in four markets out of four. What is tested here is mostly the
machinery that stops the same thing happening quietly again: a section with no
mechanism is SILENT rather than neutral, a typo in the mechanism name is
refused at load rather than becoming a section that never fires, and the
mechanism the search measures is by construction the one the section runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from analysis.mechanisms import FAMILIES, WARMUP
from analysis.section_xaujpy import MECHANISMS, SectionXauJpy
from config.schema import SectionXauJpyConfig
from core.types import MarketContext, Timeframe


def _bars(count: int = WARMUP + 60, minutes: int = 5, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-05 00:00", periods=count, freq=f"{minutes}min", tz="UTC")
    step = rng.normal(0.0, 300.0, size=count).cumsum() + 724_000.0
    frame = pd.DataFrame(
        {
            "open": step,
            "high": step + np.abs(rng.normal(0.0, 200.0, size=count)),
            "low": step - np.abs(rng.normal(0.0, 200.0, size=count)),
            "close": step + rng.normal(0.0, 120.0, size=count),
            "volume": rng.integers(50, 500, size=count),
        },
        index=index,
    )
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return frame


class _Series:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df


def _context(frame: pd.DataFrame, clock: Timeframe, symbol: str = "XAUJPY") -> MarketContext:
    context = MarketContext.__new__(MarketContext)
    object.__setattr__(context, "symbol", symbol)
    object.__setattr__(context, "series", {clock: _Series(frame)})
    return context


class TestTheRegistryIsShared:
    """THE WHOLE REASON `analysis/mechanisms.py` EXISTS.

    These functions used to live inside the search script. A mechanism a search
    measures and a live section re-implements is two implementations of one
    definition, and nothing in the measured numbers says which one the account
    runs. That is the defect this repository has shipped more often than any
    other, so it is asserted rather than remembered.
    """

    def test_the_section_reads_the_same_registry_the_search_does(self) -> None:
        assert MECHANISMS is FAMILIES["all"]

    def test_the_search_script_imports_rather_than_redefines(self) -> None:
        import inspect

        import scripts.search_section_four as search
        import scripts.search_xaujpy as jpy

        assert "from analysis.mechanisms import" in inspect.getsource(search)
        assert "from analysis.mechanisms import" in inspect.getsource(jpy)

    def test_every_mechanism_a_section_may_name_is_one_the_search_tries(self) -> None:
        for name in MECHANISMS:
            SectionXauJpyConfig(mechanism=name)


class TestASectionWithNoMechanismIsSilent:
    """Absent, not neutral, and the difference is the whole point.

    A zero read means "I looked and found nothing". No read at all means
    nobody has searched this clock yet. The old section eleven produced the
    first while meaning the second for an entire day.
    """

    def test_an_empty_mechanism_emits_no_read(self) -> None:
        section = SectionXauJpy("section_twelve_xaujpy_m5", SectionXauJpyConfig(enabled=True))
        signal = section.analyze(_context(_bars(), Timeframe.M5))

        assert signal.score == 0.0
        assert "no read" in signal.reasoning

    def test_a_disabled_section_emits_no_read(self) -> None:
        config = SectionXauJpyConfig(enabled=False, mechanism="stretch_fade")
        section = SectionXauJpy("section_twelve_xaujpy_m5", config)

        assert section.analyze(_context(_bars(), Timeframe.M5)).score == 0.0


class TestATypoIsRefusedAtLoad:
    """A mechanism nothing implements is a section that never fires, and
    silence here is indistinguishable from a mechanism that found nothing."""

    def test_an_unknown_mechanism_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="not a known mechanism"):
            SectionXauJpyConfig(mechanism="stretch_fayde")

    def test_an_hour_outside_the_clock_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="not a UTC hour"):
            SectionXauJpyConfig(allowed_hours=(24,))

    def test_an_allow_list_wholly_blocked_is_refused(self) -> None:
        """Two settings that each read as reasonable and together switch the
        section off, which is exactly the shape that goes unnoticed."""
        with pytest.raises(ValidationError, match="can never enter"):
            SectionXauJpyConfig(allowed_hours=(8, 9), blocked_hours=(7, 8, 9, 10))


class TestTheHoursBind:
    def _firing(self) -> tuple[SectionXauJpy, pd.DataFrame]:
        """A config and a frame whose last bar actually produces a direction."""
        frame = _bars()
        for name in sorted(MECHANISMS):
            config = SectionXauJpyConfig(enabled=True, mechanism=name)
            section = SectionXauJpy("section_twelve_xaujpy_m5", config)
            if int(section.readings(frame)[-1]) != 0:
                return section, frame
        pytest.skip("no mechanism fired on this frame")

    def test_a_blocked_hour_is_refused_even_when_the_mechanism_fires(self) -> None:
        section, frame = self._firing()
        assert section.analyze(_context(frame, Timeframe.M5)).score != 0.0

        hour = int(frame.index[-1].hour)
        blocked = SectionXauJpy(
            section.name,
            section.config.model_copy(update={"blocked_hours": (hour,)}),
        )
        assert blocked.analyze(_context(frame, Timeframe.M5)).score == 0.0

    def test_an_hour_outside_the_allow_list_is_refused(self) -> None:
        section, frame = self._firing()
        hour = int(frame.index[-1].hour)
        elsewhere = SectionXauJpy(
            section.name,
            section.config.model_copy(update={"allowed_hours": ((hour + 5) % 24,)}),
        )

        assert elsewhere.analyze(_context(frame, Timeframe.M5)).score == 0.0

    def test_the_wrong_symbol_is_refused(self) -> None:
        section, frame = self._firing()

        assert section.analyze(_context(frame, Timeframe.M5, symbol="XAUUSD")).score == 0.0


class TestTheSignalCarriesAStop:
    """`VERBODEN: trades zonder stoploss.` A signal with no invalidation price
    is a trade the sizer cannot bound."""

    def test_every_firing_signal_names_an_invalidation_price(self) -> None:
        frame = _bars()
        for name in sorted(MECHANISMS):
            config = SectionXauJpyConfig(enabled=True, mechanism=name, stop_atr=1.0)
            section = SectionXauJpy("section_twelve_xaujpy_m5", config)
            signal = section.analyze(_context(frame, Timeframe.M5))
            if signal.score == 0.0:
                continue
            assert signal.invalidation_price is not None, name
            close = float(frame["close"].iloc[-1])
            if signal.score > 0:
                assert signal.invalidation_price < close, name
            else:
                assert signal.invalidation_price > close, name

    def test_long_only_refuses_the_short_half(self) -> None:
        frame = _bars()
        for name in sorted(MECHANISMS):
            config = SectionXauJpyConfig(enabled=True, mechanism=name, long_only=True)
            section = SectionXauJpy("section_twelve_xaujpy_m5", config)
            if int(section.readings(frame)[-1]) >= 0:
                continue
            assert section.analyze(_context(frame, Timeframe.M5)).score == 0.0
            return
        pytest.skip("no mechanism went short on this frame")


class TestTheLiveWiringIsComplete:
    """Every previous section reached the live allowlist missing one registry
    entry, and the missing one was always silent. These are the registries."""

    def _settings(self):
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

    def test_all_three_sections_are_built(self) -> None:
        from runner.service import XAUJPY_SECTIONS, build_analysis_modules

        built = {module.name for module in build_analysis_modules(self._settings())}

        assert set(XAUJPY_SECTIONS) <= built

    def test_all_three_have_a_breaker_and_an_intraday_clock(self) -> None:
        from runner.service import XAUJPY_SECTIONS

        settings = self._settings()
        for name in XAUJPY_SECTIONS:
            assert name in settings.risk.section_breakers, name
            assert name in settings.analysis.confluence.intraday_modules, name

    def test_all_three_carry_their_own_broker_label(self) -> None:
        from core.trade_origin import origin_for_setup_family, section_of_comment
        from runner.service import XAUJPY_SECTIONS

        seen = set()
        for name in XAUJPY_SECTIONS:
            origin = origin_for_setup_family(name)
            assert origin is not None, name
            assert len(origin.comment) <= 31, origin.comment
            assert section_of_comment(origin.comment) == origin.comment, name
            seen.add(origin.comment)
        assert len(seen) == len(XAUJPY_SECTIONS), "two sections share one label"

    def test_the_dry_run_can_measure_all_three(self) -> None:
        """A live section the replay cannot see is a live section the "what
        would the account have done" report answers about in silence. Section
        eleven shipped in exactly that state and `dryrun-live.cmd` died on it."""
        import inspect

        from runner.service import XAUJPY_SECTIONS
        from scripts import dry_run_sections

        source = inspect.getsource(dry_run_sections)
        for name in XAUJPY_SECTIONS:
            assert source.count(f'"{name}"') >= 2, name

    def test_none_of_the_three_is_live_without_a_mechanism(self) -> None:
        """The guard, and the reason it exists: a live section with no
        mechanism trades nothing and reports a zero row that reads as a
        strategy finding nothing."""
        from runner.service import unsearched_live_sections

        assert unsearched_live_sections(self._settings()) == []


class TestTheSearchPaysForWhatItSearches:
    def test_selecting_a_session_raises_the_bar(self) -> None:
        from scripts.search_xaujpy import SESSIONS, _cell_count

        plain = _cell_count(28, 3, 2, hours=False)
        with_hours = _cell_count(28, 3, 2, hours=True)

        assert with_hours == plain * len(SESSIONS)

    def test_the_sessions_cover_every_hour_exactly_once(self) -> None:
        """A gap means trades in that hour vanish from the session table, and
        an overlap means they are counted twice. Both are silent."""
        from scripts.search_xaujpy import SESSIONS

        hours = [hour for _name, block in SESSIONS for hour in block]

        assert sorted(hours) == list(range(24))

    def test_a_survivor_must_clear_the_holdout_too(self) -> None:
        """The old section eleven cleared every other test. This is the one it
        failed, so it is the one that must be in the code rather than in a
        docstring."""
        import inspect

        from scripts import search_xaujpy

        source = " ".join(inspect.getsource(search_xaujpy).split())

        assert "c.sigma >= bar and c.beats_its_coin and c.holdout_r > 0 and c.total_r > 0" in source


class TestASectionCanActuallyClearTheGateItMustPass:
    """32,407 SETUPS, ZERO TRADES, AND NOT ONE OF THEM COULD EVER HAVE PASSED.

    The first 180-day replay of sections eleven, twelve and thirteen formed
    32,407 setups on XAUJPY and took nothing. Every single refusal read:

        confluence score 33.0 below threshold

    A lone module scores exactly `|score| x confidence`. The module emitted a
    hardcoded 60.0 at 0.55 confidence, so 33.0, against a `score_threshold` of
    35.0. Two points short, by construction, on every bar that will ever exist.
    Every other section carries its score as a CONFIG field around 70; this one
    had a number written into the module and nothing compared the two.

    An hour of the owner's time went into a run whose answer was arithmetic.
    These tests are the arithmetic, run every time.
    """

    def _settings(self):
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

    def test_every_section_with_a_score_can_clear_the_score_threshold(self) -> None:
        """The general property, over every section this config carries.

        Written for all of them rather than for the three that broke, because
        the next section to be added will have the same chance of landing two
        points under the bar.
        """
        settings = self._settings()
        threshold = settings.analysis.confluence.score_threshold

        checked = 0
        for name in type(settings.analysis).model_fields:
            section = getattr(settings.analysis, name, None)
            score = getattr(section, "score", None)
            confidence = getattr(section, "confidence", None)
            if score is None or confidence is None or not isinstance(score, int | float):
                continue
            checked += 1
            assert score * confidence >= threshold, (
                f"{name} emits {score} x {confidence} = {score * confidence:.1f}, under the "
                f"{threshold} score threshold. It can form setups and never take a trade."
            )
        assert checked >= 5, "this test has gone blind; no section exposes score + confidence"

    def test_every_section_clears_its_own_lone_module_floor(self) -> None:
        """The second gate on the same path. A section trading alone -- which
        is exactly what a `--only` replay does -- has to clear its lone floor
        as well, and a floor set above the confidence the module sends is the
        same silent refusal one gate along."""
        settings = self._settings()
        confluence = settings.analysis.confluence

        for name in type(settings.analysis).model_fields:
            section = getattr(settings.analysis, name, None)
            confidence = getattr(section, "confidence", None)
            if confidence is None or not isinstance(confidence, int | float):
                continue
            floor = confluence.lone_floor_for(name)
            assert confidence >= floor, (
                f"{name} sends {confidence} confidence against a lone floor of {floor}, "
                f"so every signal it makes alone is refused."
            )

    def test_the_three_xaujpy_sections_take_their_score_from_config(self) -> None:
        """Not a number in the module. That is how the 60.0 got there and how
        nothing noticed it disagreed with every other section in the book."""
        import inspect

        from analysis import section_xaujpy

        source = inspect.getsource(section_xaujpy)

        assert "score = cfg.score if direction is Direction.LONG else -cfg.score" in source
        assert "60.0" not in source, "a hardcoded score is back"


class TestTheBrokersOwnSymbolNameIsUsed:
    """A SUFFIX KILLED A RESEARCH RUN AND WOULD HAVE SILENCED A LIVE SECTION.

    Eightcap puts `.i` on its FX pairs and not on its metals. The legs research
    asked MT5 for `USDJPY` and died three symbols in, while XAUUSD and XAUJPY
    had just worked -- and `instruments.broker_symbol` had encoded that rule
    since long before either script existed.

    The live section had the same hole one layer up: `ctx.symbol` carries the
    BROKER's name (the core universe prints `USDJPY.i`) and the section compared
    it against the plain `XAUJPY` in its own config. This broker happens to list
    a plain XAUJPY too, so it worked by luck; on a broker that lists only
    `XAUJPY.i` the section would have been silent forever with nothing saying
    why.
    """

    def _settings(self):
        from config.loader import DEFAULT_CONFIG_PATH, load_settings

        return load_settings(
            DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
        )

    def test_the_builder_resolves_the_symbol_through_the_config(self) -> None:
        from runner.service import XAUJPY_SECTIONS, build_analysis_modules

        settings = self._settings()
        built = {m.name: m for m in build_analysis_modules(settings) if m.name in XAUJPY_SECTIONS}

        assert built
        for name, module in built.items():
            expected = settings.instruments.broker_symbol(module.config.symbol)
            assert module.broker_symbol == expected, name

    def test_the_section_answers_to_the_suffixed_name(self) -> None:
        frame = _bars()
        config = SectionXauJpyConfig(enabled=True, mechanism="stretch_fade", symbol="XAUJPY")
        section = SectionXauJpy("section_twelve_xaujpy_m5", config, broker_symbol="XAUJPY.i")

        # Whichever the scanner hands it, both are this instrument.
        for name in ("XAUJPY.i", "XAUJPY"):
            section.analyze(_context(frame, Timeframe.M5, symbol=name))

        assert section.analyze(_context(frame, Timeframe.M5, symbol="XAUUSD")).score == 0.0
        assert section.analyze(_context(frame, Timeframe.M5, symbol="XAUEUR.i")).score == 0.0

    def test_it_still_answers_when_no_broker_name_is_given(self) -> None:
        """Every test and script that builds one directly passes two arguments,
        so the third has to default to the config's own symbol rather than to
        None -- a None here would refuse every bar."""
        config = SectionXauJpyConfig(enabled=True, mechanism="stretch_fade")
        section = SectionXauJpy("section_twelve_xaujpy_m5", config)

        assert section.broker_symbol == config.symbol

    def test_the_legs_research_asks_the_config_for_the_name(self) -> None:
        import inspect

        from scripts import search_xaujpy_legs

        source = inspect.getsource(search_xaujpy_legs)

        assert "settings.instruments.broker_symbol(canonical)" in source
        assert "_broker_name(connector, settings, args.yen)" in source

"""Section eleven: the same mechanism as section six, fitted per market.

None of this asserts that any model is any good -- no model exists until the
trainer writes one, and it only writes one for a market that clears a
walk-forward fit, a random control, a Bonferroni bar and an untouched holdout.
What is pinned here is the machinery around that decision, because every way
this can go wrong is silent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.section_eleven_metals import (
    FEATURE_VERSION,
    PROJECTION,
    WARMUP,
    MetalModel,
    SectionElevenMetals,
    feature_frame,
    feature_row,
    load_models,
    write_model,
)
from config.schema import SectionElevenMetalsConfig
from core.types import MarketContext, Series, Timeframe


def _bars(rows: int = 400, seed: int = 3, start: str = "2026-01-05") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=rows, freq="5min", tz="UTC")
    close = 3300.0 + np.cumsum(rng.normal(0.0, 0.9, rows))
    return pd.DataFrame(
        {
            "open": np.r_[close[0], close[:-1]],
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
            "volume": rng.integers(50, 500, rows).astype(float),
        },
        index=index,
    )


class TestTheTrainerAndTheLivePathReadOneThing:
    """THE FAILURE THIS WHOLE MODULE IS SHAPED AROUND.

    A model trained on one definition of a feature and run on another is not a
    weak model, it is a different model, and nothing in its measured numbers
    says so. Section six computes its thirteen features for the LAST bar only;
    a trainer needs them for every bar, and the obvious move is a fast
    vectorised copy -- two implementations of one definition, which is the
    defect this project keeps producing.

    So there is one function and the live path takes its last row.
    """

    def test_the_live_row_is_the_last_row_of_the_training_frame(self) -> None:
        frame = _bars()
        every = feature_frame(frame)
        live = feature_row(frame)

        assert live is not None
        assert np.array_equal(live, every[-1])

    def test_the_live_row_is_the_same_on_a_frame_that_grew(self) -> None:
        """A section reads a rolling window, so the same bar must produce the
        same features whether it is the 300th row or the 400th. A feature that
        depends on how much history happens to be loaded is a feature that
        changes between the fit and the run."""
        frame = _bars(rows=400)
        short = feature_row(frame.iloc[:300])
        long = feature_frame(frame)[299]

        assert short is not None
        assert np.allclose(short, long, atol=1e-9, equal_nan=True)

    def test_a_short_frame_has_no_opinion(self) -> None:
        assert feature_row(_bars(rows=WARMUP - 1)) is None

    def test_thirteen_features_in_a_fixed_order(self) -> None:
        """The projection matrix has thirteen inputs. Adding a feature without
        bumping FEATURE_VERSION would feed every stored model a vector it never
        learned, and the numbers would keep coming."""
        assert feature_frame(_bars()).shape[1] == PROJECTION.shape[0] == 13


class TestAModelFileCarriesItsOwnProvenance:
    def _model(self, symbol: str = "XAUEUR", version: int = FEATURE_VERSION) -> MetalModel:
        rng = np.random.default_rng(11)
        return MetalModel(
            symbol=symbol,
            centre=tuple(rng.normal(size=13)),
            scale=tuple(np.abs(rng.normal(size=13)) + 0.5),
            beta=tuple(rng.normal(size=PROJECTION.shape[1] + 1) * 0.01),
            feature_version=version,
            trained_from="2025-01-01",
            trained_through="2026-06-30",
            holdout_trades=310,
            holdout_r=12.5,
            holdout_sigma=2.4,
            threshold=0.2,
        )

    def test_a_written_model_reads_back_identically(self, tmp_path) -> None:
        written = self._model()
        write_model(written, tmp_path)

        back = load_models(tmp_path)["XAUEUR"]

        assert back.symbol == "XAUEUR"
        assert back.threshold == pytest.approx(0.2)
        assert back.holdout_trades == 310
        assert back.holdout_sigma == pytest.approx(2.4)
        assert np.allclose(back.centre, written.centre, atol=1e-9)
        assert np.allclose(back.beta, written.beta, atol=1e-9)

    def test_a_model_from_another_feature_version_is_refused(self, tmp_path) -> None:
        """Not loaded-and-warned. Running a model against inputs that are no
        longer the inputs it learned fails silently: the readings keep coming
        and they are simply wrong."""
        write_model(self._model(version=FEATURE_VERSION + 1), tmp_path)

        with pytest.raises(ValueError, match="feature version"):
            load_models(tmp_path)

    def test_an_empty_directory_is_no_models_and_not_an_error(self, tmp_path) -> None:
        assert load_models(tmp_path / "nothing-here") == {}

    def test_the_holdout_number_travels_with_the_model(self, tmp_path) -> None:
        """The figure that justified shipping it lives in the file, not in
        somebody's memory of a terminal window."""
        write_model(self._model(), tmp_path)
        raw = (tmp_path / "XAUEUR.json").read_text()

        assert "holdout_r" in raw and "trained_through" in raw


class TestTheSectionIsSilentWhereItWasNeverFitted:
    def _context(self, symbol: str, frame: pd.DataFrame) -> MarketContext:
        now = frame.index[-1].to_pydatetime()
        return MarketContext(symbol, now, {Timeframe.M5: Series(symbol, Timeframe.M5, frame, now)})

    def test_a_market_with_no_model_takes_no_trade(self) -> None:
        """ABSENT, NOT NEUTRAL, and the difference is the whole point. A
        section that quietly trades an unfitted market is section six pointed
        at XAUEUR."""
        config = SectionElevenMetalsConfig(enabled=True, allowed_symbols=("XAUEUR",))
        section = SectionElevenMetals(config, models={})

        signal = section.analyze(self._context("XAUEUR", _bars()))

        assert signal.score == 0.0
        assert signal.confidence == 0.0

    def test_a_market_outside_allowed_symbols_takes_no_trade(self) -> None:
        config = SectionElevenMetalsConfig(enabled=True, allowed_symbols=("XAUEUR",))
        rng = np.random.default_rng(5)
        model = MetalModel(
            symbol="XAUGBP",
            centre=tuple(np.zeros(13)),
            scale=tuple(np.ones(13)),
            beta=tuple(rng.normal(size=PROJECTION.shape[1] + 1)),
        )
        section = SectionElevenMetals(config, models={"XAUGBP": model})

        assert section.analyze(self._context("XAUGBP", _bars())).score == 0.0

    def test_a_blocked_hour_takes_no_trade(self) -> None:
        """Same shape and same reason as section ten's per-symbol hours: the
        crosses lose money at London's close and gold does not, so one window
        cannot serve both."""
        frame = _bars(start="2026-01-05 16:00")
        config = SectionElevenMetalsConfig(
            enabled=True,
            allowed_symbols=("XAUEUR",),
            blocked_hours_by_symbol={"XAUEUR": (frame.index[-1].hour,)},
        )
        # A model that would otherwise shout: a large intercept clears any
        # threshold, so only the hour can be keeping it quiet.
        model = MetalModel(
            symbol="XAUEUR",
            centre=tuple(np.zeros(13)),
            scale=tuple(np.ones(13)),
            beta=tuple(np.r_[5.0, np.zeros(PROJECTION.shape[1])]),
        )
        section = SectionElevenMetals(config, models={"XAUEUR": model})

        assert section.analyze(self._context("XAUEUR", frame)).score == 0.0

    def test_a_loud_model_on_an_open_hour_does_speak(self) -> None:
        """The mirror of the tests above: if nothing fires here, they prove
        nothing."""
        frame = _bars(start="2026-01-05 09:00")
        config = SectionElevenMetalsConfig(enabled=True, allowed_symbols=("XAUEUR",))
        model = MetalModel(
            symbol="XAUEUR",
            centre=tuple(np.zeros(13)),
            scale=tuple(np.ones(13)),
            beta=tuple(np.r_[5.0, np.zeros(PROJECTION.shape[1])]),
        )
        section = SectionElevenMetals(config, models={"XAUEUR": model})

        signal = section.analyze(self._context("XAUEUR", frame))

        assert signal.score > 0.0
        assert signal.invalidation_price is not None

    def test_a_live_section_eleven_is_fully_wired(self) -> None:
        """It went live on 4 September at the owner's instruction, on models
        that did NOT clear the trainer's bar -- best cell +2.56 sigma against
        2.96. That is his call, taken with the number in view.

        What is not his call is a section that is half-wired. Every one of
        these was missing when it was first added to the allowlist, and each
        one is a way for a live section to behave as a different strategy than
        the one measured:

          * no target ratio -> the engine searches for a target instead of
            trading the 1.5 R the models were fitted at
          * not in `intraday_modules` -> falls through to "swing", which hands
            an M5 model H1 planning authority, a 24-bar horizon and a D1/W1
            veto at one conflict
          * not exempt from the entry-timing gate -> the gate refuses an entry
            where price ran against the idea over the last three M5 bars,
            which is the mechanism of a model that reads exactly that bar
          * no section breaker -> nothing switches it off

        If section eleven is on the live list, it is wired. If it is off, this
        test says nothing and that is fine.
        """
        from config.loader import load_settings

        settings = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        )
        confluence = settings.analysis.confluence
        if "section_eleven_metals" not in confluence.live_enabled_modules:
            pytest.skip("section eleven is not live")

        assert settings.analysis.section_eleven_metals.enabled is True
        assert confluence.target_r_multiple_by_family["section_eleven_metals"] == 1.5
        assert "section_eleven_metals" in confluence.intraday_modules
        assert "section_eleven_metals" in confluence.entry_timing_exempt_families
        assert confluence.weights.get("section_eleven_metals", 0.0) > 0
        breaker = settings.risk.section_breakers.get("section_eleven_metals")
        assert breaker is not None and breaker.enabled

    def test_a_live_section_eleven_without_models_is_reported_not_silent(self) -> None:
        """A live section with no model files is not dangerous. It is SILENT.

        The config says live, the replay shows zero trades, and the empty
        result reads as "the strategy found nothing" rather than "nothing was
        ever fitted". This repository has produced that confusion in half a
        dozen other forms and it is the reason most of its comments exist.

        `unfitted_live_sections` names the markets, and the startup guard turns
        that into a refusal to start. Deliberately NOT raised from
        `build_analysis_modules`: that function is called by a dozen tests
        about other things, and a refusal there fails all of them for a reason
        none of them is about.
        """
        from config.loader import load_settings
        from runner.service import unfitted_live_sections

        settings = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        )
        config = settings.analysis.section_eleven_metals
        if "section_eleven_metals" not in settings.analysis.confluence.live_enabled_modules:
            pytest.skip("section eleven is not live")

        models = load_models(config.model_dir)
        expected = [s for s in config.allowed_symbols if s not in models]

        assert unfitted_live_sections(settings) == expected

    def test_the_startup_guard_refuses_a_live_section_with_nothing_fitted(self) -> None:
        """And it says which command writes the models, because a refusal that
        does not tell you what to do next gets worked around."""
        import inspect

        from core import startup

        source = inspect.getsource(startup.run_startup_guard)

        assert "unfitted_live_sections" in source
        assert "train11" in source

    def test_a_section_that_is_not_live_needs_no_models(self, tmp_path) -> None:
        """The refusal is about the LIVE list, not about the section existing.
        A section being measured in the shadow with no models is a section
        that takes no trades, which is a fine thing for a shadow to do."""
        from config.loader import load_settings
        from runner.service import build_analysis_modules

        settings = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        )
        confluence = settings.analysis.confluence
        without = tuple(m for m in confluence.live_enabled_modules if m != "section_eleven_metals")
        shadowed = settings.model_copy(
            update={
                "analysis": settings.analysis.model_copy(
                    update={
                        "confluence": confluence.model_copy(
                            update={"live_enabled_modules": without}
                        ),
                        "section_eleven_metals": (
                            settings.analysis.section_eleven_metals.model_copy(
                                update={"model_dir": str(tmp_path)}
                            )
                        ),
                    }
                )
            }
        )

        assert build_analysis_modules(shadowed)

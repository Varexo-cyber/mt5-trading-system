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

    def test_it_is_off_and_unfitted_in_the_shipped_config(self) -> None:
        from config.loader import load_settings

        settings = load_settings(
            "config/config.yaml", overlay="config/eightcap.yaml", env_overrides=False
        )
        config = settings.analysis.section_eleven_metals

        assert config.enabled is False, (
            "section eleven is enabled. It trades nothing without model files, but "
            "enabling it before a replay agrees is how section six went live."
        )
        assert "section_eleven_metals" not in settings.analysis.confluence.live_enabled_modules


class TestTheTrainerRefusesRatherThanShips:
    def test_the_bar_rises_with_the_number_of_cells_tried(self) -> None:
        from scripts.train_section_eleven import bonferroni_sigma

        assert bonferroni_sigma(1) == pytest.approx(1.96, abs=0.02)
        assert bonferroni_sigma(16) > 2.9
        assert bonferroni_sigma(64) > bonferroni_sigma(16)

    def test_sigma_is_clustered_by_day_and_not_by_trade(self) -> None:
        """Five metals breaking on one morning are one observation. Counting
        each trade independently overstates significance by roughly the square
        root of however many moved together."""
        from scripts.train_section_eleven import Trades, stats

        spread = Trades([1.0, -1.0] * 50, [f"d{i}" for i in range(100)])
        clumped = Trades([1.0, -1.0] * 50, ["d0", "d1"] * 50)

        assert stats(spread)[3] == stats(clumped)[3] == 100
        # Same trades, same total, fewer independent days -> not more certain.
        assert abs(stats(clumped)[2]) <= abs(stats(spread)[2]) + 1e-9

    def test_a_trade_that_reaches_no_barrier_is_discarded_not_scratched(self) -> None:
        """It has not answered the question, and calling it a scratch is an
        answer it did not give."""
        from scripts.train_section_eleven import resolve

        index = pd.date_range("2026-02-02", periods=60, freq="5min", tz="UTC")
        flat = np.full(60, 3300.0)
        frame = pd.DataFrame(
            {"open": flat, "high": flat + 1.0, "low": flat - 1.0, "close": flat}, index=index
        )

        taken = resolve(
            frame, np.ones(60, dtype=int), stop_atr=50.0, ratio=1.0, cost_r=0.0, horizon=5
        )

        assert len(taken) == 0

    def test_the_resolver_holds_one_position_at_a_time(self) -> None:
        from scripts.train_section_eleven import resolve

        frame = _bars(rows=300, seed=8)
        always = np.ones(len(frame), dtype=int)

        taken = resolve(frame, always, stop_atr=1.0, ratio=1.5, cost_r=0.0, horizon=24)

        assert 0 < len(taken) < len(frame)

    def test_the_target_is_measured_in_atr_not_in_price(self) -> None:
        """A model that predicts a number in one unit while the risk is
        expressed in another can be perfectly calibrated and still size
        everything wrong."""
        from scripts.train_section_eleven import forward_target

        frame = _bars(rows=200, seed=12)
        target = forward_target(frame, horizon=10)

        assert len(target) == len(frame)
        assert np.isnan(target[-1]), "the last bars have no future to measure"
        finite = target[np.isfinite(target)]
        assert len(finite) > 100
        # ATR-normalised, so a gold-priced series does not produce values in
        # the thousands.
        assert np.abs(finite).max() < 50.0


class TestTheFollowUpPaysForTheGridItAlreadySearched:
    """The 4 September run spent sixteen cells and cleared nothing.

    XAUJPY was the one market whose sigma ROSE with the threshold -- +0.82,
    +0.86, +1.67, +2.56 across 0.10 to 0.30 -- which is the shape a real
    signal makes when you select harder, and the shape noise does not reliably
    make. That is a prediction worth testing at higher thresholds.

    It is only a test if the earlier sixteen are still on the bill. Searching
    a grid twice and paying for it once is how a search launders itself into
    a discovery, and this repo has the receipts.
    """

    def test_declared_earlier_cells_raise_the_bar(self) -> None:
        from scripts.train_section_eleven import bonferroni_sigma

        fresh = bonferroni_sigma(10)
        with_history = bonferroni_sigma(10 + 16)

        assert with_history > fresh

    def test_the_launcher_declares_the_sixteen_it_already_spent(self) -> None:
        from pathlib import Path

        text = Path("train11.cmd").read_text(errors="replace")

        assert "--cells-already-tried 16" in text, (
            "the follow-up run does not declare the sixteen cells the first run "
            "spent, so its Bonferroni bar is too low"
        )
        assert "--thresholds 0.30,0.40,0.50,0.60,0.80" in text

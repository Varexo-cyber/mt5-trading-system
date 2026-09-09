from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from analysis.section_six_adaptive import SectionSixGoldM5, SectionSixSpxH1
from config.loader import load_settings
from config.schema import SectionSixModelConfig
from core.types import MarketContext, Series, Tick, Timeframe

NOW = datetime(2026, 8, 31, 20, 0, tzinfo=UTC)


def _context(symbol: str, timeframe: Timeframe) -> MarketContext:
    index = pd.date_range(end=NOW, periods=120, freq=timeframe.duration)
    drift = np.linspace(0.0, 8.0, len(index))
    wave = np.sin(np.arange(len(index)) / 3.0)
    close = 1000.0 + drift + wave
    frame = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "tick_volume": np.linspace(100.0, 200.0, len(index)),
            "spread": 10,
        },
        index=index,
    )
    return MarketContext(
        symbol,
        NOW,
        {timeframe: Series(symbol, timeframe, frame, NOW)},
        Tick(symbol, NOW, bid=float(close[-1] - 0.1), ask=float(close[-1] + 0.1)),
    )


def test_gold_model_only_reads_xauusd_m5() -> None:
    config = SectionSixModelConfig(
        enabled=True, timeframe="M5", polarity=-1, threshold=0.0001, stop_atr=1.0
    )
    module = SectionSixGoldM5(config)

    signal = module.analyze(_context("XAUUSD", Timeframe.M5))
    wrong_market = module.analyze(_context("SPX500", Timeframe.M5))

    assert signal.score != 0.0
    assert signal.invalidation_price is not None
    assert wrong_market.score == 0.0


def test_failed_spx_variant_stays_disabled_by_default() -> None:
    module = SectionSixSpxH1(SectionSixModelConfig(timeframe="H1"))

    assert module.analyze(_context("SPX500", Timeframe.H1)).score == 0.0


def test_gold_route_does_not_emit_outside_its_measured_session() -> None:
    config = SectionSixModelConfig(
        enabled=True,
        timeframe="M5",
        polarity=-1,
        threshold=0.0001,
        stop_atr=0.8,
        session_start_hour_utc=0,
        session_end_hour_utc=1,
    )

    signal = SectionSixGoldM5(config).analyze(_context("XAUUSD", Timeframe.M5))

    assert signal.score == 0.0
    assert "outside measured" in signal.reasoning


def test_both_closed_bar_horizons_must_confirm(monkeypatch) -> None:
    """A local bounce is not enough while the four-hour leg still falls."""
    from analysis import section_six_adaptive

    ctx = _context("XAUUSD", Timeframe.M5)
    frame = ctx.series[Timeframe.M5].df.copy()
    frame.loc[:, "close"] = 100.0
    frame.iloc[-1, frame.columns.get_loc("close")] = 101.0
    frame.iloc[-49, frame.columns.get_loc("close")] = 102.0
    series = Series("XAUUSD", Timeframe.M5, frame, NOW)
    ctx = MarketContext(ctx.symbol, ctx.now, {Timeframe.M5: series}, ctx.tick)
    monkeypatch.setattr(section_six_adaptive, "model_reading", lambda *_args: (-1.0, 1.0))
    config = SectionSixModelConfig(
        enabled=True,
        timeframe="M5",
        polarity=-1,
        threshold=0.15,
        long_only=True,
        confirmation_bars=12,
        secondary_confirmation_bars=48,
    )

    signal = SectionSixGoldM5(config).analyze(ctx)

    assert signal.score == 0.0
    assert "48 closed M5 bars do not confirm" in signal.reasoning


def test_live_overlay_keeps_the_measured_gold_exit_and_rejects_spx() -> None:
    settings = load_settings(overlay="config/eightcap.yaml", env_overrides=False)

    assert settings.analysis.section_six_gold_m5.enabled is True
    assert settings.analysis.section_six_spx_h1.enabled is False
    assert settings.analysis.section_six_gold_m5.threshold == 0.15
    assert settings.analysis.section_six_gold_m5.stop_atr == 0.8
    assert settings.analysis.section_six_gold_m5.long_only is True
    assert settings.analysis.section_six_gold_m5.session_start_hour_utc == 20
    assert settings.analysis.section_six_gold_m5.session_end_hour_utc == 2
    assert settings.analysis.confluence.target_r_multiple_by_family["section_six_gold_m5"] == 3.0
    assert "section_six_gold_m5" in settings.analysis.confluence.strategy_owned_entry_families
    assert "section_six_gold_m5" in settings.analysis.confluence.target_reach_advisory_families
    assert "section_fifteen_btc_m1" in settings.analysis.confluence.target_reach_advisory_families
    assert settings.analysis.confluence.max_spread_share_of_stop_by_family == {
        "section_sixteen_btc_m5": 0.12,
        "section_seventeen_btc_m15": 0.10,
    }
    # OFF THE FIXED-EXIT LIST ON 3 SEPTEMBER, and that is the whole change the
    # owner asked for. While the label sat there the manager returned early and
    # this route received NO position management -- not live and not in the dry
    # run -- so "dry run with position management" could not be done at all.
    # On the same 180-day entries, break-even alone moved the outcome from
    # -53.23R to +36.42R, and that figure still stood on the broken offset.
    assert "JARVIS-S6-AU-M5" not in settings.trade_management.fixed_exit_comments
    assert settings.analysis.section_six_gold_m5.confirmation_bars == 12
    assert settings.analysis.section_six_gold_m5.secondary_confirmation_bars == 48
    assert "JARVIS-S6-AU-M5" in settings.trade_management.break_even_only_comments


class TestEachSilenceNamesItself:
    """Three faults printed one sentence and it cost three days.

    Section six ran 257,352 times over a weekend, scored zero every time, and
    the only reason available said "needs 80 closed M5 bars" -- while the live
    scan fetches 200. Every hour after that went into ruling out a cause the
    message had already named wrongly: the clock was there, the bars were
    there, the model clears the threshold on ~30% of realistic gold windows.

    A silence that cannot say which silence it is turns a five-minute answer
    into a weekend.
    """

    @staticmethod
    def _context(symbol: str, frame, timeframe=None, hour: int = 22):
        from datetime import UTC, datetime
        from types import SimpleNamespace

        from core.types import Timeframe

        clock = timeframe or Timeframe.M5
        series = None if frame is None else SimpleNamespace(df=frame)
        # 22:00 UTC by default: INSIDE section six's measured 20:00-02:00 gold
        # session. Outside it the section refuses before reaching the model,
        # and a fixture that sat at noon would have proved the model unreachable
        # rather than the message wrong.
        return SimpleNamespace(
            symbol=symbol,
            series={} if series is None else {clock: series},
            now=datetime(2026, 9, 8, hour, 30, tzinfo=UTC),
            tick=None,
            meta={},
        )

    @staticmethod
    def _frame(rows: int, flat_volume: bool = False):
        from datetime import UTC, datetime

        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(5)
        index = pd.date_range(datetime(2026, 9, 8, tzinfo=UTC), periods=rows, freq="5min")
        close = 4400 + np.cumsum(rng.normal(0, 1.5, rows))
        return pd.DataFrame(
            {
                "open": close + rng.normal(0, 0.5, rows),
                "high": close + np.abs(rng.normal(0, 0.8, rows)),
                "low": close - np.abs(rng.normal(0, 0.8, rows)),
                "close": close,
                "tick_volume": np.zeros(rows) if flat_volume else rng.integers(
                    200, 900, rows
                ).astype(float),
            },
            index=index,
        )

    def _section(self):
        from pathlib import Path

        from analysis.section_six_adaptive import SectionSixGoldM5
        from config.loader import load_settings

        settings = load_settings(
            overlay=Path("config/eightcap.yaml"), env_overrides=False
        )
        return SectionSixGoldM5(settings.analysis.section_six_gold_m5)

    def test_a_missing_clock_says_the_clock_is_missing(self) -> None:
        signal = self._section().analyze(self._context("XAUUSD", None))
        assert "not in this context" in signal.reasoning
        assert "80" not in signal.reasoning, "this is not a bar-count problem"

    def test_a_short_frame_says_how_short(self) -> None:
        signal = self._section().analyze(self._context("XAUUSD", self._frame(40)))
        assert "40 closed M5 bars and needs 80" in signal.reasoning

    def test_a_non_finite_input_says_so_and_not_bars(self) -> None:
        """200 bars and still nothing. The old message blamed the bar count,
        which is the one thing that was demonstrably fine."""
        signal = self._section().analyze(
            self._context("XAUUSD", self._frame(200, flat_volume=True))
        )
        assert "not finite" in signal.reasoning
        assert "needs 80" not in signal.reasoning

    def test_the_three_sentences_are_all_different(self) -> None:
        section = self._section()
        said = {
            section.analyze(self._context("XAUUSD", None)).reasoning,
            section.analyze(self._context("XAUUSD", self._frame(40))).reasoning,
            section.analyze(
                self._context("XAUUSD", self._frame(200, flat_volume=True))
            ).reasoning,
            section.analyze(self._context("EURUSD.i", self._frame(200))).reasoning,
        }
        assert len(said) == 4, f"two faults still read the same: {said}"

    def test_a_healthy_frame_still_reaches_the_model(self) -> None:
        # The split must not have broken the working path.
        signal = self._section().analyze(self._context("XAUUSD", self._frame(200)))
        assert "not in this context" not in signal.reasoning
        assert "needs 80" not in signal.reasoning
        assert "not finite" not in signal.reasoning

    def test_the_session_window_is_a_fourth_distinct_sentence(self) -> None:
        """Section six trades gold for six hours a night, long only. That is a
        design decision and it must not read like a fault -- nor a fault like
        it. At noon the section is correctly silent and says so in its own
        words."""
        signal = self._section().analyze(
            self._context("XAUUSD", self._frame(200), hour=12)
        )
        assert "outside measured" in signal.reasoning
        assert "20:00-02:00" in signal.reasoning

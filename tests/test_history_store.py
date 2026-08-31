"""Bars kept on disk, so a measurement never pays for the same fetch twice.

180 days over sixteen markets and six clocks is roughly four million bars.
That download should happen once. Everything here is about the store handing
back exactly what went in -- a cache that quietly drops or rounds a column is
worse than no cache, because the number it changes is the one somebody trades
on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtesting.history_store import HistoryStore
from core.instrument import AssetClass, InstrumentSpec
from core.types import Timeframe


def frame(bars: int = 500, start: datetime | None = None) -> pd.DataFrame:
    begin = start or datetime(2026, 3, 2, tzinfo=UTC)
    index = pd.date_range(begin, periods=bars, freq="15min", tz=UTC, name="time")
    close = 1.1000 + np.arange(bars) * 0.00001
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.00012,
            "low": close - 0.00012,
            "close": close,
            "tick_volume": np.arange(bars, dtype="int64") + 1,
            "spread": np.full(bars, 14, dtype="int64"),
            "real_volume": np.zeros(bars, dtype="int64"),
        },
        index=index,
    )


def spec(symbol: str = "EURUSD.i") -> InstrumentSpec:
    return InstrumentSpec(
        symbol=symbol,
        digits=5,
        point=0.00001,
        tick_size=0.00001,
        tick_value=1.0,
        contract_size=100_000.0,
        volume_min=0.01,
        volume_max=200.0,
        volume_step=0.01,
        stops_level=0,
        freeze_level=0,
        currency_base="EUR",
        currency_profit="USD",
        currency_margin="EUR",
        filling_mode_mask=1,
        trade_mode=4,
        is_forex=True,
        path="Forex\\Majors\\EURUSD.i",
        description="Euro vs US Dollar",
        asset_class=AssetClass.FOREX,
    )


class TestWhatGoesInComesBack:
    def test_every_column_survives_the_round_trip(self, tmp_path: Path) -> None:
        """INCLUDING `spread`. The cost model reads it, and a lossy round trip
        would zero it and make every stored market look free to trade."""
        store = HistoryStore(tmp_path)
        original = frame()

        store.write("EURUSD.i", Timeframe.M15, original)
        back = store.frame("EURUSD.i", Timeframe.M15)

        assert list(back.columns) == list(original.columns)
        # The index is NORMALISED to nanoseconds. pandas 2 hands out
        # microsecond-unit indexes from `date_range` and nanosecond ones from
        # MT5, and a store that preserved whichever it was handed would return
        # frames that compare unequal for a reason nobody would find. The
        # instants are identical, which is what every caller compares.
        pd.testing.assert_frame_equal(back, original, check_freq=False, check_index_type=False)
        assert (back.index == original.index).all()
        assert back.index.dtype == "datetime64[ns, UTC]"

    def test_prices_are_not_rounded(self, tmp_path: Path) -> None:
        """A pip on EURUSD is the fourth decimal and a point the fifth. Storing
        at float32 would lose the fifth and quietly change every stop."""
        store = HistoryStore(tmp_path)
        original = frame()
        store.write("EURUSD.i", Timeframe.M15, original)

        back = store.frame("EURUSD.i", Timeframe.M15)

        assert back["close"].dtype == original["close"].dtype
        assert back["close"].iloc[-1] == original["close"].iloc[-1]

    def test_the_index_keeps_its_timezone(self, tmp_path: Path) -> None:
        """Everything downstream slices on UTC timestamps. A naive index would
        compare unequal against every window bound and return nothing."""
        store = HistoryStore(tmp_path)
        store.write("EURUSD.i", Timeframe.M15, frame())

        back = store.frame("EURUSD.i", Timeframe.M15)

        assert back.index.tz is not None
        assert str(back.index.tz) in ("UTC", "utc")

    def test_a_window_slices_inclusively_at_both_ends(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path)
        original = frame(bars=100)
        store.write("EURUSD.i", Timeframe.M15, original)
        begin = original.index[10]
        end = original.index[20]

        back = store.frame("EURUSD.i", Timeframe.M15, begin, end)

        assert back.index[0] == begin
        assert back.index[-1] == end
        assert len(back) == 11


class TestItDoesNotFailQuietly:
    """The recurring defect in this project is a silence that reads as an
    answer. A cache is a place that shape of bug loves to live."""

    def test_a_missing_frame_raises_rather_than_returning_empty(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path)

        with pytest.raises(FileNotFoundError, match="not in the store"):
            store.frame("EURUSD.i", Timeframe.M15)

    def test_the_error_says_how_to_fix_it(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path)

        with pytest.raises(FileNotFoundError, match="fetch_history"):
            store.frame("GBPUSD.i", Timeframe.H1)

    def test_an_empty_frame_is_refused_rather_than_stored(self, tmp_path: Path) -> None:
        """Storing nothing and reporting success would make the next run report
        "no setups" for a market that was simply never downloaded."""
        store = HistoryStore(tmp_path)

        with pytest.raises(ValueError, match="empty"):
            store.write("EURUSD.i", Timeframe.M15, frame().iloc[0:0])

    def test_a_missing_spec_raises(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path)

        with pytest.raises(KeyError):
            store.spec("EURUSD.i")


class TestTheSpecTravelsWithTheBars:
    """Sizing needs `volume_min`, `tick_value` and `point`. A store of bars
    alone would still require a live terminal for every run, which is the
    thing being removed."""

    def test_it_round_trips_exactly(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path)
        original = spec()

        store.write_spec(original)

        assert store.spec("EURUSD.i") == original

    def test_the_asset_class_comes_back_as_an_enum(self, tmp_path: Path) -> None:
        """It is written to JSON as a string. Coming back as a bare string
        would break every `spec.asset_class.value` downstream."""
        store = HistoryStore(tmp_path)
        store.write_spec(spec())

        assert store.spec("EURUSD.i").asset_class is AssetClass.FOREX

    def test_writing_a_second_spec_keeps_the_first(self, tmp_path: Path) -> None:
        """The manifest is rewritten per symbol during a long fetch. Losing the
        earlier entries would empty the store one market at a time."""
        store = HistoryStore(tmp_path)
        store.write_spec(spec("EURUSD.i"))
        store.write_spec(spec("GBPUSD.i"))

        assert store.symbols() == ["EURUSD.i", "GBPUSD.i"]

    def test_a_symbol_with_a_dot_gets_a_usable_directory(self, tmp_path: Path) -> None:
        """`EURUSD.i` and `US30.cash` contain characters that are legal on one
        platform and not another."""
        store = HistoryStore(tmp_path)
        store.write("US30.cash", Timeframe.H1, frame(bars=50))

        assert store.has("US30.cash", Timeframe.H1)
        assert store.frame("US30.cash", Timeframe.H1).shape[0] == 50


class TestResumingAnInterruptedDownload:
    """Four million bars will be interrupted. Restarting must cost only what
    was missing."""

    def test_what_is_stored_is_reported_as_stored(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path)
        store.write("EURUSD.i", Timeframe.M15, frame())

        assert store.has("EURUSD.i", Timeframe.M15)
        assert not store.has("EURUSD.i", Timeframe.H4)
        assert store.timeframes("EURUSD.i") == ["M15"]

    def test_a_corrupt_manifest_does_not_take_the_bars_with_it(self, tmp_path: Path) -> None:
        """An interrupted write used to be able to leave unreadable JSON. The
        bars are separate files and must survive it."""
        store = HistoryStore(tmp_path)
        store.write_spec(spec())
        store.write("EURUSD.i", Timeframe.M15, frame())
        store.manifest_path.write_text("{ this is not json")

        assert store.symbols() == []
        assert store.frame("EURUSD.i", Timeframe.M15).shape[0] == 500

    def test_the_window_is_recorded_so_a_reader_knows_what_it_has(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path)
        store.write_spec(spec())
        end = datetime(2026, 8, 31, tzinfo=UTC)

        store.note_window(180, end)

        days, stamp = store.window()
        assert days == 180
        assert stamp.startswith("2026-08-31")
        assert store.symbols() == ["EURUSD.i"], "noting the window keeps the specs"


class TestTheSizeIsWorthKnowingBeforeDownloading:
    def test_it_reports_bars_and_bytes(self, tmp_path: Path) -> None:
        store = HistoryStore(tmp_path)
        store.write_spec(spec())
        store.write("EURUSD.i", Timeframe.M15, frame(bars=1000))
        store.write("EURUSD.i", Timeframe.H1, frame(bars=250))

        rows = {(symbol, tf): bars for symbol, tf, bars, _size in store.summary()}

        assert rows[("EURUSD.i", "M15")] == 1000
        assert rows[("EURUSD.i", "H1")] == 250
        assert store.size_bytes() > 0

    def test_compression_actually_happens(self, tmp_path: Path) -> None:
        """Uncompressed, 5000 bars of seven float64 columns is ~280 KB. If the
        store ever silently stopped compressing, 180 days of M1 across sixteen
        markets would be gigabytes instead of tens of megabytes."""
        store = HistoryStore(tmp_path)
        store.write("EURUSD.i", Timeframe.M1, frame(bars=5000))

        raw = 5000 * 7 * 8
        assert store.size_bytes() < raw


class TestMeasuringFromTheStoreInsteadOfTheTerminal:
    def test_the_flag_exists_and_is_a_path(self) -> None:
        from scripts.dry_run_sections import build_parser

        parsed = build_parser().parse_args(["--cache", "data/history", "--equity", "216"])

        assert parsed.cache == "data/history"

    def test_an_empty_store_says_how_to_fill_it_rather_than_measuring_nothing(
        self, tmp_path: Path
    ) -> None:
        """A cache pointed at the wrong folder would otherwise produce a run
        with zero symbols, which reads exactly like a market with no setups."""
        from scripts.dry_run_sections import main

        with pytest.raises(SystemExit, match="fetch_history"):
            main(["--cache", str(tmp_path), "--equity", "216", "--days", "7"])

    def test_the_stored_market_answers_what_the_run_asks_of_a_broker(self, tmp_path: Path) -> None:
        from scripts.dry_run_sections import _StoredMarket

        store = HistoryStore(tmp_path)
        store.write_spec(spec())
        market = _StoredMarket(store, equity=216.0)

        assert market.spec("EURUSD.i").volume_min == pytest.approx(0.01)
        assert market.account().equity == pytest.approx(216.0)
        assert [item.name for item in market.symbols()] == ["EURUSD.i"]
        assert market.shutdown() is None

    def test_it_refuses_to_invent_an_equity(self, tmp_path: Path) -> None:
        """There is no terminal to ask. Defaulting to a number would size every
        trade in the measurement against a balance the account does not have."""
        from scripts.dry_run_sections import _StoredMarket

        store = HistoryStore(tmp_path)
        store.write_spec(spec())

        with pytest.raises(SystemExit, match="--equity"):
            _StoredMarket(store, equity=0.0).account()


class TestTheFetcher:
    def test_it_stores_every_clock_the_sweep_can_use(self) -> None:
        from scripts.dry_run_sections import SWEEPABLE
        from scripts.fetch_history import ALL_CLOCKS

        assert set(ALL_CLOCKS) == {tf.value for tf in SWEEPABLE}

    def test_the_defaults_are_the_long_window_and_the_core_markets(self) -> None:
        from scripts.fetch_history import build_parser

        parsed = build_parser().parse_args([])

        assert parsed.days == 180
        assert parsed.symbols == ""
        assert list(parsed.timeframes) == ["M1", "M5", "M15", "M30", "H1", "H4"]

    def test_clocks_accept_both_a_comma_list_and_separate_words(self) -> None:
        """cmd splits on commas, so both forms arrive and both must work. The
        comma form silently losing the rest of the line has cost two runs on
        this project already."""
        from scripts.fetch_history import _wanted_clocks

        assert _wanted_clocks(["M15,M30", "H1"]) == [
            Timeframe.M15,
            Timeframe.M30,
            Timeframe.H1,
        ]
        assert _wanted_clocks(["m15", "h4"]) == [Timeframe.M15, Timeframe.H4]


class TestTheLaunchers:
    """Both take the clocks as arguments, and `snel.cmd` has to clear
    `--no-m1` for exactly the two clocks that need M1 bars -- the same trap
    `sweep.cmd` fell into, in a new file."""

    def test_the_fetcher_launcher_parses(self) -> None:
        from scripts.fetch_history import build_parser
        from tests.test_dry_run_script import cmd_argv

        launcher = (Path(__file__).resolve().parent.parent / "ophalen.cmd").read_text()
        line = next(ln for ln in launcher.splitlines() if "scripts.fetch_history" in ln)
        argv: list[str] = []
        for token in line.split("scripts.fetch_history", 1)[1].split():
            if token.startswith("%") and token.endswith("%"):
                token = {"%DAYS%": "180", "%CLOCKS%": "M1 M5 M15 M30 H1 H4"}[token]
            argv.extend(piece for word in token.split() for piece in word.split(",") if piece)
        parsed = build_parser().parse_args(argv)

        assert parsed.days == 180
        assert list(parsed.timeframes) == ["M1", "M5", "M15", "M30", "H1", "H4"]
        assert parsed.out.replace("\\", "/") == "data/history"
        assert cmd_argv is not None  # the shared helper stays the reference

    def test_the_offline_sweep_launcher_reads_the_store(self) -> None:
        from scripts.dry_run_sections import build_parser
        from tests.test_dry_run_script import cmd_argv

        launcher = (Path(__file__).resolve().parent.parent / "snel.cmd").read_text()
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

        assert parsed.cache.replace("\\", "/") == "data/history"
        assert parsed.sweep == ["M15", "M30", "H1", "H4"]
        assert parsed.equity > 0, "there is no terminal to ask for the balance"

    def test_the_offline_launcher_clears_no_m1_for_the_two_fine_clocks(self) -> None:
        launcher = (Path(__file__).resolve().parent.parent / "snel.cmd").read_text().upper()

        assert 'IF /I "%%C"=="M1" SET FINE=' in launcher
        assert 'IF /I "%%C"=="M5" SET FINE=' in launcher

    def test_the_offline_launcher_refuses_an_empty_store(self) -> None:
        """Pointing it at a folder that was never filled would otherwise
        produce a run with no symbols, which reads like a quiet market."""
        launcher = (Path(__file__).resolve().parent.parent / "snel.cmd").read_text()

        assert "data\\history\\manifest.json" in launcher
        assert "Run ophalen.cmd first" in launcher

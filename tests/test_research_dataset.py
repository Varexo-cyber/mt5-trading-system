from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pandas as pd

from backtesting.research_dataset import ResearchDataset
from core.instrument import AssetClass, InstrumentSpec
from core.types import Timeframe
from scripts.capture_research_data import resolve_symbols


def frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=3, freq="1min", tz=UTC)
    return pd.DataFrame(
        {
            "open": [1.1, 1.2, 1.3],
            "high": [1.2, 1.3, 1.4],
            "low": [1.0, 1.1, 1.2],
            "close": [1.15, 1.25, 1.35],
            "tick_volume": [10, 11, 12],
            "spread": [8, 9, 10],
            "real_volume": [0, 0, 0],
        },
        index=index,
    )


def spec() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="EURUSD.i",
        digits=5,
        point=0.00001,
        tick_size=0.00001,
        tick_value=1.0,
        contract_size=100_000,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        stops_level=0,
        freeze_level=0,
        currency_base="EUR",
        currency_profit="USD",
        currency_margin="EUR",
        filling_mode_mask=3,
        trade_mode=4,
        is_forex=True,
        path="Forex\\Majors",
        description="Euro vs US Dollar",
        asset_class=AssetClass.FOREX,
    )


def test_one_file_round_trips_bars_specs_metadata_and_coverage(tmp_path: Path) -> None:
    path = tmp_path / "research.sqlite3"
    bars = frame()
    with ResearchDataset(path) as dataset:
        dataset.set_metadata("research_starting_equity", 203.0)
        dataset.put_instrument("EURUSD", spec())
        dataset.put_frame(
            "EURUSD.i",
            Timeframe.M1,
            bars,
            requested_from=bars.index[0],
            requested_to=bars.index[-1],
            captured_at="2026-01-01T00:03:00+00:00",
        )

    with ResearchDataset(path, read_only=True) as dataset:
        restored = dataset.load_frame("EURUSD", "M1")
        assert dataset.metadata("research_starting_equity") == 203.0
        assert len(dataset.coverage()) == 1
        assert dataset.symbols() == ["EURUSD.i"]
        assert dataset.spec("EURUSD").volume_min == 0.01
        assert dataset.frame("EURUSD", Timeframe.M1, bars.index[1]).index[0] == bars.index[1]
        stored_spec = dataset.connection.execute(
            "SELECT spec_json FROM instruments WHERE canonical_symbol='EURUSD'"
        ).fetchone()[0]
        assert '"filling_mode_mask":3' in stored_spec
        assert '"trade_mode":4' in stored_spec
        assert list(restored["spread"]) == [8, 9, 10]
        assert list(restored["close"]) == [1.15, 1.25, 1.35]
        assert str(restored.index.tz) == "UTC"


def test_symbol_resolution_prefers_a_decorated_exact_market() -> None:
    catalogue = ["EURUSDX", "EURUSD.i", "GBPUSD.i", "US30", "US30Cash"]
    assert resolve_symbols(catalogue, ("EURUSD", "GBPUSD", "US30")) == {
        "EURUSD": "EURUSD.i",
        "GBPUSD": "GBPUSD.i",
        "US30": "US30",
    }

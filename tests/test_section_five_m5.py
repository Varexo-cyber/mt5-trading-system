from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import section_five_m5 as live
from scripts.search_multimarket_section import Model, _predict
from scripts.search_walkforward_section import _features


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=140, freq="5min", tz="UTC")
    close = 20_000.0 + np.cumsum(np.sin(np.arange(140) / 5.0) + 0.15)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 2.0,
            "low": np.minimum(open_, close) - 2.0,
            "close": close,
            "volume": 1000.0 + np.arange(140),
            "spread": 20.0,
        },
        index=index,
    )


def test_live_reader_matches_the_frozen_research_transform() -> None:
    frame = _frame()
    x, _y, _atr = _features(frame, 6)
    model = Model(live._CENTRE, live._SCALE, live._PROJECTION, live._OFFSET, live._BETA)

    actual = live.model_reading(frame)

    assert actual is not None
    assert actual[0] == pytest.approx(_predict(x, model)[-1])

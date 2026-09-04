from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import walkforward_index as live
from config.loader import DEFAULT_CONFIG_PATH, load_settings
from scripts.search_walkforward_section import _features, _predict


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=140, freq="h", tz="UTC")
    close = 5000.0 + np.cumsum(np.sin(np.arange(140) / 7.0) + 0.2)
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


def test_live_reader_is_identical_to_the_frozen_research_model() -> None:
    frame = _frame()
    x, _y, _atr = _features(frame, 12)
    expected = _predict(x, (live._BETA, live._CENTRE, live._SCALE))[-1]

    actual = live.model_reading(frame)

    assert actual is not None
    assert actual[0] == pytest.approx(expected)


def test_only_the_measured_new_sections_may_trade_real_money() -> None:
    settings = load_settings(
        DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
    )

    # THE POINT OF THIS TEST IS `walkforward_index`, not the whole allowlist.
    # It asserted the live list as a frozen tuple, so every legitimate change
    # to it broke a test about a different module -- which is how a suite stops
    # being read. What has to hold here is that the unmeasured section is not
    # on the list.
    live = settings.analysis.confluence.live_enabled_modules
    assert "walkforward_index" not in live
    assert live, "the account is live with no sections at all"
    assert settings.analysis.walkforward_index.enabled is False
    assert settings.analysis.walkforward_index.allowed_symbols == ("SPX500",)
    assert settings.analysis.confluence.target_r_multiple_by_family["walkforward_index"] == 1.5
    assert "walkforward_index" in settings.risk.section_breakers
    assert "failed_session_breakout" in settings.risk.section_breakers
    assert "section_five_m5" in settings.risk.section_breakers
    assert "section_six_gold_m5" in settings.risk.section_breakers
    assert "section_eight_trend_day_h1" in settings.risk.section_breakers
    assert "section_nine_vwap_m30" in settings.risk.section_breakers
    assert "section_ten_gold_m1" in settings.risk.section_breakers

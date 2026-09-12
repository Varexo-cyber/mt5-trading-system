from datetime import UTC, datetime

import pandas as pd

from scripts.gold_intraday_close_momentum import trades


def test_close_momentum_uses_new_york_dst_and_previous_close(monkeypatch):
    index = pd.date_range("2025-03-07 18:30", "2025-03-10 17:30", freq="5min", tz=UTC)
    frame = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=index)
    # March 10 is EDT: 13:00 NY == 17:00 UTC and 13:30 == 17:30 UTC.
    frame.loc[pd.Timestamp(datetime(2025, 3, 7, 18, 30, tzinfo=UTC)), "open"] = 100.0
    frame.loc[pd.Timestamp(datetime(2025, 3, 10, 17, 0, tzinfo=UTC)), "open"] = 102.0
    frame.loc[pd.Timestamp(datetime(2025, 3, 10, 17, 30, tzinfo=UTC)), "open"] = 103.0
    monkeypatch.setattr(
        "scripts.gold_intraday_close_momentum._atr",
        lambda data: pd.Series(1.0, index=data.index),
    )
    result = trades(frame, stop_atr=1.0, cost_r=0.1)
    row = result[result["day"].astype(str) == "2025-03-10"].iloc[0]
    assert row["direction"] == "LONG"
    assert row["gross_r"] == 1.0
    assert row["net_r"] == 0.9


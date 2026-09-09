import pandas as pd
import pytest

from config.schema import HistoryClosure, DataConfig
from core.data_manager import DataManager, _missing_bars
from core.errors import DataIntegrityError
from core.types import Timeframe


def sample(start, missing):
    index = pd.date_range('2026-09-07', periods=850, freq='5min', tz='UTC')
    first = index.get_loc(pd.Timestamp(start))
    return pd.DataFrame(index=index.delete(slice(first, first + missing)))


@pytest.mark.parametrize('start,count,covered', [
    ('2026-09-07T16:50:00Z', 73, 50),
    ('2026-09-07T18:20:00Z', 55, 32),
])
def test_only_published_slots_are_deducted(start, count, covered):
    frame = sample(start, count)
    closure = HistoryClosure(start=start, end='2026-09-07T21:00:00Z')
    assert _missing_bars(frame, Timeframe.M5) == count
    assert _missing_bars(frame, Timeframe.M5, closures=(closure,)) == count - covered
    # Overlapping configuration must not double-discount.
    assert _missing_bars(frame, Timeframe.M5, closures=(closure, closure)) == count - covered


def test_wrong_date_does_not_hide_outage():
    frame = sample('2026-09-07T16:50:00Z', 73)
    closure = HistoryClosure(start='2026-09-06T16:50:00Z', end='2026-09-06T21:00:00Z')
    assert _missing_bars(frame, Timeframe.M5, closures=(closure,)) == 73


def test_exact_symbol_and_real_outage_still_rejected():
    closure = HistoryClosure(start='2026-09-07T16:50:00Z', end='2026-09-07T21:00:00Z')
    manager = DataManager(None, DataConfig(max_gap_fraction=.05, history_closures={'NDX100': (closure,)}), None)
    frame = sample('2026-09-07T16:50:00Z', 73).iloc[-600:]
    manager._check_gaps('NDX100', Timeframe.M5, frame)
    with pytest.raises(DataIntegrityError, match='Gap boundaries UTC'):
        manager._check_gaps('BTCUSD', Timeframe.M5, frame)
    outage = sample('2026-09-07T16:50:00Z', 140)
    with pytest.raises(DataIntegrityError):
        manager._check_gaps('NDX100', Timeframe.M5, outage)


@pytest.mark.parametrize('start,end', [
    ('2026-09-07T16:50:00', '2026-09-07T21:00:00'),
    ('2026-09-07T21:00:00Z', '2026-09-07T16:50:00Z'),
])
def test_invalid_closures_rejected(start, end):
    with pytest.raises(ValueError):
        HistoryClosure(start=start, end=end)

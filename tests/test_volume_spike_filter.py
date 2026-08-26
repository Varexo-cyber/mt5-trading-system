"""An event is not momentum, and section one had no way of telling them apart.

Section six has refused a minute carrying many times its normal activity since
it was written -- "that is an event, not momentum, and the spread after it is
the risk". Section one had no such rule anywhere. Seven of its eight live
detectors read price shape and never look at volume, and the one that does
(`m1_micro_breakout`) has a volume FLOOR and no ceiling: its confidence RISES
with volume, without limit, so a release printing ten times normal activity
produces its strongest reading.

What stood in for it was the economic calendar, and a calendar only knows what
is scheduled. An unscheduled headline, a central banker off script, a stop
cascade: none are in it, all print this candle.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from config.schema import VolumeSpikeFilterConfig
from filters.volume_spike_filter import VolumeSpikeFilter
from risk.reasons import Reason


class _Data:
    """Serves one M1 frame, or refuses to."""

    def __init__(self, volumes: list[float] | None, *, raises: bool = False) -> None:
        self.volumes = volumes
        self.raises = raises

    def get_series(self, symbol, timeframe):  # type: ignore[no-untyped-def]
        if self.raises:
            raise RuntimeError("no connection")
        if self.volumes is None:
            return None
        frame = pd.DataFrame({"tick_volume": self.volumes})
        return SimpleNamespace(df=frame)


def _check(volumes, **kwargs):  # type: ignore[no-untyped-def]
    config = VolumeSpikeFilterConfig(**kwargs)
    filt = VolumeSpikeFilter(config, _Data(volumes))  # type: ignore[arg-type]
    return filt.check(SimpleNamespace(symbol="XAUUSD"))  # type: ignore[arg-type]


def _normal(n: int = 40, level: float = 100.0) -> list[float]:
    return [level] * n


def test_a_release_candle_is_refused() -> None:
    """The live shape: an ordinary market, then one minute at ten times."""
    verdict = _check([*_normal(), 1000.0])

    assert not verdict.passed
    assert verdict.reason is Reason.VOLUME_SPIKE
    assert "10.0x" in verdict.detail


def test_an_ordinary_busy_minute_still_trades() -> None:
    """Busy is not an event. The gate must not become a way of refusing every
    minute that is livelier than the last."""
    verdict = _check([*_normal(), 250.0])

    assert verdict.passed


def test_the_boundary_is_the_same_figure_section_six_uses() -> None:
    """Both refuse at 3.0x. Two rules about the same candle that disagree is
    how this project has repeatedly hurt itself."""
    from config.loader import DEFAULT_CONFIG_PATH, load_settings

    settings = load_settings(
        DEFAULT_CONFIG_PATH, overlay="config/eightcap.yaml", env_overrides=False
    )

    assert (
        settings.filters.volume_spike.extreme_multiple
        == settings.analysis.candle_momentum.extreme_volume_multiple
    )
    assert not _check([*_normal(), 300.0]).passed
    assert _check([*_normal(), 299.0]).passed


def test_the_baseline_is_a_median_so_one_earlier_spike_cannot_hide_the_next() -> None:
    """A mean would be dragged up by the previous release and quietly let the
    following one through."""
    with_spike = [*_normal(38), 4000.0, 100.0]

    verdict = _check([*with_spike, 400.0])

    assert not verdict.passed, "an earlier spike raised the baseline and hid this one"


def test_missing_volume_history_does_not_refuse() -> None:
    """The one gate here that fails OPEN, and deliberately. Thin volume is the
    ordinary condition of some instruments and is not evidence of an event;
    refusing them would remove markets for a reason unrelated to risk. The
    calendar and the spread filter above still fail closed."""
    assert _check(None).passed
    assert _check([1.0, 2.0]).passed


def test_a_broken_data_layer_does_not_refuse_either() -> None:
    config = VolumeSpikeFilterConfig()
    filt = VolumeSpikeFilter(config, _Data(None, raises=True))  # type: ignore[arg-type]

    assert filt.check(SimpleNamespace(symbol="XAUUSD")).passed  # type: ignore[arg-type]


def test_disabled_means_disabled() -> None:
    assert _check([*_normal(), 1000.0], enabled=False).passed

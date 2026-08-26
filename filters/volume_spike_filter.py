"""An M1 candle carrying many times its normal activity is an event.

Section six has refused these since it was written, in its own words:

    "this minute carried 8.4x its normal activity -- that is an event, not
     momentum, and the spread after it is the risk"

Section one had no such rule anywhere. Of its eight live detectors, seven read
price shape and never look at volume at all, and the one that does --
`m1_micro_breakout` -- has a volume FLOOR and no ceiling: more volume raises
its confidence, without limit. A release printing ten times normal activity
gets its highest reading.

What protected section one was the economic calendar, and the calendar only
knows what is scheduled. An unscheduled headline, a central banker off script,
a stop cascade: none of those appear in it, and all of them print exactly this
candle. The strongest-looking bar such a system will ever see is the one it
should never trade.

A filter rather than a module, so it applies to whatever the account is
running rather than to the detectors that happen to consult volume. It is a
statement about the market at this instant, not an opinion about a setup.

FAIL-OPEN, DELIBERATELY, AND ONLY HERE. The rest of this system fails closed:
no calendar means no trade. This one does not, because "no volume data" is the
ordinary condition of an instrument the broker reports thinly, and refusing
those outright would silently remove markets for a reason that has nothing to
do with risk. The calendar and the spread filter above still fail closed.
"""

from __future__ import annotations

import pandas as pd

from config.schema import VolumeSpikeFilterConfig
from core.data_manager import DataManager
from core.types import Timeframe
from filters.base import FilterContext, FilterVerdict
from infra.logging import get_logger
from risk.reasons import Reason

log = get_logger(__name__)


class VolumeSpikeFilter:
    """Refuse a market whose last closed minute is carrying an event."""

    name = "volume_spike"

    def __init__(self, config: VolumeSpikeFilterConfig, data: DataManager) -> None:
        self.config = config
        self.data = data

    def check(self, ctx: FilterContext) -> FilterVerdict:
        config = self.config
        if not config.enabled:
            return FilterVerdict.allow(self.name, "volume spike filter disabled")

        ratio = self._ratio(ctx.symbol)
        if ratio is None:
            # See the module docstring: thin or missing volume is not evidence
            # of an event, and this is the one gate here that may not refuse on
            # ignorance.
            return FilterVerdict.allow(self.name, "no usable volume history")

        if ratio >= config.extreme_multiple:
            return FilterVerdict.block(
                self.name,
                Reason.VOLUME_SPIKE,
                f"the last closed minute carried {ratio:.1f}x its normal activity, at or "
                f"above the {config.extreme_multiple:.1f}x that marks an event rather than "
                f"momentum; the spread after one of these is the risk",
                volume_ratio=round(ratio, 2),
            )
        return FilterVerdict.allow(
            self.name,
            f"last minute at {ratio:.1f}x normal activity",
            volume_ratio=round(ratio, 2),
        )

    def _ratio(self, symbol: str) -> float | None:
        """The last closed M1 bar against the median of the ones before it.

        Median rather than mean, for the same reason `candle_momentum` uses
        one: a single earlier spike inside the lookback would raise a mean
        enough to hide the next one.
        """
        try:
            series = self.data.get_series(symbol, Timeframe.M1)
        except Exception:  # noqa: BLE001 - see the fail-open note in the docstring
            return None
        if series is None:
            return None
        frame: pd.DataFrame = series.df
        lookback = self.config.lookback_bars
        if "tick_volume" not in frame.columns or len(frame) < lookback + 2:
            return None
        window = frame["tick_volume"].iloc[-(lookback + 1) : -1]
        baseline = float(window.median())
        if baseline <= 0:
            return None
        return float(frame["tick_volume"].iloc[-1]) / baseline

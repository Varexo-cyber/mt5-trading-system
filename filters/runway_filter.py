"""Time-to-work gate: refuse an entry that cannot finish before we close it.

Every other time-based gate in this system asks "are we inside a bad window".
None of them asks the question that actually loses money, which is "is there
any point starting now". Those are not the same question, and the gap between
them is exactly one minute wide: at 20:14 UTC the session filter is happy, the
spread filter is happy, the setup is real — and at 20:15 the manager flattens
the position because that is what the wind-down is for. The trade paid the
spread twice and was never given a chance to be right or wrong.

That is not a rare edge. It is the last half hour of every single trading day,
and on a small account with tight stops it is a steady, invisible leak.

So this filter asks the missing question and nothing else. It does not know
the setup, the score, or the target — like every filter here it decides on
conditions alone. It knows the clock and the instrument's asset class, and it
refuses when there is less runway left than a trade plausibly needs.

The runner runs a second, sharper version of the same idea once the setup
*is* known (`_target_is_reachable_in_time`), which asks whether this specific
target is reachable in the time left at the market's current speed. This one
is the blunt floor underneath it: cheap, absolute, and impossible to argue
with, which is what a filter is for.
"""

from __future__ import annotations

from config.schema import RunwayFilterConfig
from filters.base import Filter, FilterContext, FilterVerdict
from filters.session_filter import SessionFilter
from infra.logging import get_logger
from risk.reasons import Reason

log = get_logger(__name__)


class RunwayFilter(Filter):
    """Blocks entries taken too close to our own forced exit."""

    name = "runway"

    def __init__(self, config: RunwayFilterConfig, session: SessionFilter) -> None:
        self.config = config
        #: Shared with the session filter on purpose. The deadline this gate
        #: protects *is* the session filter's wind-down; reading it from a
        #: second copy of the config would let the two drift apart, and the
        #: failure would be silent — entries allowed right up to a flatten the
        #: manager still performs on the original schedule.
        self.session = session

    def check(self, ctx: FilterContext) -> FilterVerdict:
        asset_class = ctx.spec.asset_class.value
        if not self.config.enabled:
            return FilterVerdict.allow(self.name, "runway filter disabled")

        runway = self.session.minutes_of_runway(ctx.now, asset_class)
        if runway is None:
            return FilterVerdict.allow(
                self.name,
                f"{asset_class} has no forced exit today; runway is unbounded",
                runway_minutes=None,
                asset_class=asset_class,
            )

        required = self.config.minutes_for(asset_class)
        if runway < required:
            return FilterVerdict.block(
                self.name,
                Reason.INSUFFICIENT_RUNWAY,
                f"{runway:.0f} min left before the {asset_class} wind-down, and a trade "
                f"needs {required:.0f}; entering now buys a round-trip in spread and a "
                f"forced exit before the setup has resolved either way",
                runway_minutes=round(runway, 1),
                runway_required_minutes=required,
                asset_class=asset_class,
            )

        return FilterVerdict.allow(
            self.name,
            f"{runway:.0f} min of runway before the {asset_class} wind-down",
            runway_minutes=round(runway, 1),
            runway_required_minutes=required,
            asset_class=asset_class,
        )

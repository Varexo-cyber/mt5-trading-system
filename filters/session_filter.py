"""Session and time-of-day gate.

Blocks entries when the book is thin: outside the sessions we trade, across the
daily rollover, and at the two weekend edges.

All times are UTC. Broker server time is deliberately not used here — brokers
run on GMT+2/+3 with their own DST rules, so a session defined in server time
silently shifts by an hour twice a year, in opposite directions from the actual
London and New York opens it is supposed to track.

Session windows are allowed to wrap midnight (Asia is 22:00-08:00 on some
definitions), which is why membership is tested rather than compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

from config.schema import SessionFilterConfig
from filters.base import Filter, FilterContext, FilterVerdict
from infra.logging import get_logger
from risk.reasons import Reason

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Window:
    """A daily time window in UTC, possibly wrapping midnight."""

    name: str
    start: time
    end: time

    @property
    def wraps_midnight(self) -> bool:
        return self.end <= self.start

    def contains(self, moment: datetime) -> bool:
        current = moment.timetz().replace(tzinfo=None)
        if self.wraps_midnight:
            return current >= self.start or current < self.end
        return self.start <= current < self.end

    def describe(self) -> str:
        return f"{self.name} {self.start:%H:%M}-{self.end:%H:%M}"


class SessionFilter(Filter):
    """Permits entries only during the sessions we actually want to trade."""

    name = "session"

    def __init__(self, config: SessionFilterConfig) -> None:
        self.config = config
        self.sessions = {
            name: Window(name, _parse(start), _parse(end))
            for name, (start, end) in config.sessions.items()
        }
        unknown = set(config.tradable_sessions) - set(self.sessions)
        if unknown:
            raise ValueError(
                f"tradable_sessions names no such session: {sorted(unknown)}; "
                f"defined: {sorted(self.sessions)}"
            )
        self.rollover = Window(
            "rollover", _parse(config.rollover_block[0]), _parse(config.rollover_block[1])
        )
        # Runs from the wind-down time to the end of the rollover, so it strictly
        # contains the rollover block. Entries must stop at the same moment the
        # flatten starts, or the loop would close a position and immediately
        # re-open it into a widening spread, paying that cost twice.
        self.evening_flat = (
            Window("evening-flat", _parse(config.evening_flat_from), self.rollover.end)
            if config.evening_flat_from
            else None
        )
        # Per-asset-class overrides, each strictly earlier than the FX one (the
        # config validator refuses anything later). An index does not follow the
        # FX rollover: its cash session closes at 20:00 UTC and the CFD quote
        # widens from that moment, so holding to 20:15 spends the last quarter
        # of an hour in the widest spread of the day.
        self.evening_flat_by_class = {
            name: Window(f"evening-flat-{name}", _parse(when), self.rollover.end)
            for name, when in config.evening_flat_by_class.items()
        }

        self.continuous_maintenance = Window(
            "continuous-maintenance",
            _parse(config.continuous_maintenance_block[0]),
            _parse(config.continuous_maintenance_block[1]),
        )
        self.friday_cutoff = (
            _parse(config.block_friday_after) if config.block_friday_after else None
        )
        self.sunday_open = (
            _parse(config.block_sunday_before) if config.block_sunday_before else None
        )

    # -- queries -----------------------------------------------------------

    def evening_flat_window(self, asset_class: str) -> Window | None:
        """The wind-down that applies to this asset class."""
        if asset_class in self.config.continuous_asset_classes:
            return None
        if asset_class in self.evening_flat_by_class:
            return self.evening_flat_by_class[asset_class]
        if asset_class in self.config.evening_flat_asset_classes:
            return self.evening_flat
        return None

    def minutes_of_runway(self, moment: datetime, asset_class: str) -> float | None:
        """Minutes left before *we* force this instrument flat.

        The deadline is whichever comes first: this asset class's evening
        wind-down, or the Friday cut-off on a Friday. `None` means no deadline
        applies — a continuous market, or a profile with the wind-down off.

        This is the number the session gate never had. `check` answers "are we
        inside a blocked window", which is a different question, and it leaves
        the minute before the window wide open: an entry at 20:14 clears every
        gate and is flattened by us at 20:15, having paid the spread twice to
        find out nothing.

        A deadline already behind us returns 0.0 rather than tomorrow's, which
        keeps the failure direction safe. `check` blocks those moments anyway,
        so the only way to see 0.0 is a caller asking out of band — and for
        that caller, "no runway" is the honest answer.
        """
        if asset_class in self.config.continuous_asset_classes:
            return None

        deadlines: list[time] = []
        window = self.evening_flat_window(asset_class)
        if window is not None:
            deadlines.append(window.start)
        if self.friday_cutoff is not None and moment.weekday() == 4:
            deadlines.append(self.friday_cutoff)
        if not deadlines:
            return None

        current = moment.timetz().replace(tzinfo=None)
        return min(_minutes_until(current, deadline) for deadline in deadlines)

    def should_be_flat(self, moment: datetime, asset_class: str) -> bool:
        """Is this a moment we have already decided not to be in the market for?

        The same deadline set `minutes_of_runway` counts down to, asked as a
        yes or no. Both have to be one definition, and they were not.

        The hole: `block_friday_after` refuses entries from 19:00 on a Friday
        because of the weekend, and the runway correctly reports zero minutes
        left from then. But the flatten only knew about the generic evening
        window at 20:15, so for seventy-five minutes the system refused to open
        anything — on the grounds that there was no time left to open it in —
        while leaving whatever was already on to sit there. A losing position
        in particular just waited, in exactly the thin Friday book the gate
        exists to avoid.

        Continuous markets are exempt here as everywhere: crypto has no FX
        rollover and no weekend.
        """
        if asset_class in self.config.continuous_asset_classes:
            return False
        window = self.evening_flat_window(asset_class)
        if window is not None and window.contains(moment):
            return True
        if self.friday_cutoff is None or moment.weekday() != 4:
            return False
        return moment.timetz().replace(tzinfo=None) >= self.friday_cutoff

    def active_sessions(self, moment: datetime) -> tuple[str, ...]:
        """Sessions currently open. Overlaps are real and reported as such."""
        return tuple(name for name, window in self.sessions.items() if window.contains(moment))

    def session_label(self, moment: datetime) -> str:
        """One string for the journal, e.g. `london+newyork` or `none`.

        The overlap is its own regime — it is where the day's range usually
        gets made — so it is recorded as a distinct label rather than collapsed
        into whichever session happens to be listed first.
        """
        active = self.active_sessions(moment)
        return "+".join(active) if active else "none"

    def is_weekend(self, moment: datetime) -> bool:
        """True when the FX market is closed outright."""
        weekday = moment.weekday()  # Monday = 0
        if weekday == 5:
            return True
        clock_time = moment.timetz().replace(tzinfo=None)
        if weekday == 4 and self.friday_cutoff and clock_time >= time(21, 0):
            return True
        return bool(weekday == 6 and clock_time < time(22, 0))

    # -- gate --------------------------------------------------------------

    def check(self, ctx: FilterContext) -> FilterVerdict:
        now = ctx.now
        label = self.session_label(now)
        asset_class = ctx.spec.asset_class.value

        if not self.config.enabled:
            return FilterVerdict.allow(
                self.name,
                "session filter disabled",
                session=label,
                asset_class=asset_class,
            )

        if asset_class in self.config.continuous_asset_classes:
            if self.continuous_maintenance.contains(now):
                return FilterVerdict.block(
                    self.name,
                    Reason.ROLLOVER_WINDOW,
                    f"{asset_class} maintenance buffer "
                    f"({self.continuous_maintenance.describe()} UTC)",
                    session="continuous-maintenance",
                    asset_class=asset_class,
                )
            return FilterVerdict.allow(
                self.name,
                f"{asset_class} uses its continuous-market profile; quote freshness and "
                "spread are checked separately",
                session="continuous",
                session_overlap=False,
                asset_class=asset_class,
            )

        if asset_class in self.config.broker_hours_asset_classes:
            weekday = now.weekday()
            clock_time = now.timetz().replace(tzinfo=None)
            if weekday == 5 or (weekday == 6 and clock_time < time(22, 0)):
                return FilterVerdict.block(
                    self.name,
                    Reason.MARKET_CLOSED,
                    f"{asset_class} market is closed ({now:%A %H:%M} UTC)",
                    session="broker-closed",
                    asset_class=asset_class,
                )
            if self.rollover.contains(now):
                return FilterVerdict.block(
                    self.name,
                    Reason.ROLLOVER_WINDOW,
                    f"{asset_class} broker-maintenance buffer ({self.rollover.describe()} UTC)",
                    session="broker-maintenance",
                    asset_class=asset_class,
                )
            wind_down = self.evening_flat_window(asset_class)
            if wind_down is not None and wind_down.contains(now):
                return FilterVerdict.block(
                    self.name,
                    Reason.EVENING_WIND_DOWN,
                    f"{asset_class} evening wind-down ({wind_down.describe()} UTC)",
                    session=wind_down.name,
                    asset_class=asset_class,
                )
            return FilterVerdict.allow(
                self.name,
                f"{asset_class} uses broker-hours profile; quote freshness and spread "
                "prove whether its own venue is open",
                session="broker-hours",
                session_overlap=False,
                asset_class=asset_class,
            )

        weekday = now.weekday()
        clock_time = now.timetz().replace(tzinfo=None)

        # Hard closed. Not a preference — there is no market.
        if weekday == 5 or (weekday == 6 and clock_time < time(22, 0)):
            return FilterVerdict.block(
                self.name,
                Reason.MARKET_CLOSED,
                f"FX market is closed ({now:%A %H:%M} UTC)",
                session=label,
                asset_class=asset_class,
            )

        # Evening wind-down. Strictly wider than the rollover block, and checked
        # first so the message names the real reason: not "we are inside the
        # rollover" but "we are going flat for the evening".
        flat_window = self.evening_flat_window(asset_class)
        if flat_window is not None and flat_window.contains(now):
            return FilterVerdict.block(
                self.name,
                Reason.EVENING_WIND_DOWN,
                f"evening wind-down ({flat_window.describe()} UTC); the book thins "
                f"and spreads widen from here, and open positions are being flattened",
                session=label,
                asset_class=asset_class,
            )

        # Rollover: the widest spreads of the day, on the thinnest book.
        if self.rollover.contains(now):
            return FilterVerdict.block(
                self.name,
                Reason.ROLLOVER_WINDOW,
                f"inside the daily rollover window ({self.rollover.describe()} UTC), "
                f"where spreads widen and liquidity disappears",
                session=label,
                asset_class=asset_class,
            )

        # Weekend edges: gap risk on Friday, thin reopen on Sunday.
        if self.friday_cutoff is not None and weekday == 4 and clock_time >= self.friday_cutoff:
            return FilterVerdict.block(
                self.name,
                Reason.WEEKEND_EDGE,
                f"Friday after {self.friday_cutoff:%H:%M} UTC; a position held over the "
                f"weekend carries gap risk a stop cannot bound",
                session=label,
                asset_class=asset_class,
            )
        if self.sunday_open is not None and weekday == 6 and clock_time < self.sunday_open:
            return FilterVerdict.block(
                self.name,
                Reason.WEEKEND_EDGE,
                f"Sunday before {self.sunday_open:%H:%M} UTC; the reopen book is too thin",
                session=label,
                asset_class=asset_class,
            )

        active = self.active_sessions(now)
        tradable = [name for name in active if name in self.config.tradable_sessions]
        if not tradable:
            return FilterVerdict.block(
                self.name,
                Reason.OUTSIDE_TRADABLE_SESSION,
                f"{now:%H:%M} UTC is in {label or 'no session'}; this mode trades "
                f"{', '.join(self.config.tradable_sessions)}",
                session=label,
                asset_class=asset_class,
            )

        overlap = len(tradable) > 1
        return FilterVerdict.allow(
            self.name,
            f"{label}{' (overlap)' if overlap else ''}",
            session=label,
            session_overlap=overlap,
            asset_class=asset_class,
        )


def _minutes_until(current: time, deadline: time) -> float:
    """Minutes from `current` to `deadline` on the same day, floored at zero.

    Deliberately does not roll over to tomorrow. Every deadline this is used
    for is an end-of-day one, so a deadline that reads as "past" means the day
    is over — answering with 1400 minutes of runway would be the one wrong
    answer that lets a trade through.
    """
    minutes = (deadline.hour - current.hour) * 60 + (deadline.minute - current.minute)
    minutes -= current.second / 60.0
    return max(0.0, minutes)


def _parse(text: str) -> time:
    """Parse HH:MM into a naive `time` compared against UTC clock time."""
    try:
        hour, minute = (int(part) for part in text.split(":", 1))
        return time(hour, minute)
    except ValueError as exc:
        raise ValueError(f"expected HH:MM, got {text!r}") from exc

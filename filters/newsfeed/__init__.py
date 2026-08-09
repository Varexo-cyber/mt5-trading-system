"""Unscheduled news, as a risk layer rather than a signal.

`filters.calendar` knows what is coming and when. This knows what is happening
now — the central bank that moved between meetings, the headline nobody had on
a calendar — and it is used to keep the system's hands still while it happens,
never to pick a direction. `filters.newsfeed.items.NewsPressure` carries the
argument for that in full.
"""

from filters.newsfeed.items import Headline, NewsPressure
from filters.newsfeed.providers import (
    DEFAULT_FEEDS,
    FeedUnavailableError,
    HeadlineProvider,
    RssHeadlineProvider,
    build_providers,
    parse_feed,
)
from filters.newsfeed.service import HeadlineService
from filters.newsfeed.tagging import currencies_in, is_systemic

__all__ = [
    "DEFAULT_FEEDS",
    "FeedUnavailableError",
    "Headline",
    "HeadlineProvider",
    "HeadlineService",
    "NewsPressure",
    "RssHeadlineProvider",
    "build_providers",
    "currencies_in",
    "is_systemic",
    "parse_feed",
]

"""Check every headline feed actually answers, and show what it parsed.

Nothing in `filters.newsfeed` has been confirmed against a live feed. The
environment it was written in refuses outbound connections to all of these
hosts, so the URLs are candidates and the parser is untested against real
dialect. That is the reason `filters.headlines.enabled` ships as false.

Run this on the machine that will use it. It reports, per feed: whether it
answered, how many items parsed, how many did not, and which currencies the
tagger found. Then it shows the same pressure reading the filter would take,
per currency, so the thresholds can be set against real traffic instead of
against a guess.

    python scripts/verify_newsfeed.py
    python scripts/verify_newsfeed.py --window 20 --show 15

Reads nothing but the feeds. No MT5, no journal, no orders, no Claude API.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.loader import load_settings
from core.clock import LiveClock
from filters.newsfeed.providers import (
    DEFAULT_FEEDS,
    FeedUnavailableError,
    RssHeadlineProvider,
)
from filters.newsfeed.service import HeadlineService
from filters.newsfeed.tagging import CURRENCY_TERMS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=float, default=0.0, help="recent window, minutes")
    parser.add_argument("--show", type=int, default=10, help="headlines to print per feed")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)

    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    config = settings.filters.headlines
    feeds = config.feeds or dict(DEFAULT_FEEDS)
    window = args.window or config.window_minutes

    print()
    print("  HEADLINE FEEDS")
    print("  " + "-" * 72)

    now = datetime.now(UTC)
    working = []
    for name, url in feeds.items():
        provider = RssHeadlineProvider(name, url, timeout_seconds=args.timeout)
        try:
            items = provider.fetch(now)
        except FeedUnavailableError as exc:
            print(f"  {name:<16} FAILED   {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - the report is the point
            print(f"  {name:<16} FAILED   {type(exc).__name__}: {exc}")
            continue

        tagged = [item for item in items if item.currencies or item.systemic]
        newest = max((i.published for i in items), default=None)
        stale = f"{(now - newest).total_seconds() / 60:.0f} min old" if newest else "no dates"
        print(
            f"  {name:<16} ok       {len(items):>4} items, {len(tagged):>4} tagged, newest {stale}"
        )
        working.append((provider, items))

    if not working:
        print()
        print("  Not one feed answered. Either this machine has no outbound HTTPS,")
        print("  or every URL has moved. Do not enable the filter on this result —")
        print("  a layer that returns nothing reports a quiet market.")
        print()
        return 1

    for provider, items in working:
        print()
        print(f"  {provider.name} — newest {args.show}")
        print("  " + "-" * 72)
        for item in sorted(items, key=lambda i: i.published, reverse=True)[: args.show]:
            tags = "/".join(sorted(item.currencies)) or "-"
            flag = " SYSTEMIC" if item.systemic else ""
            print(f"    {item.published:%H:%M} {tags:<12}{flag:<10} {item.title[:76]}")

    # The reading the filter would actually take, so the thresholds can be set
    # against this machine's real traffic rather than against the defaults.
    service = HeadlineService(
        [provider for provider, _ in working],
        LiveClock(),
        refresh_interval_seconds=config.refresh_interval_seconds,
        window_minutes=window,
        baseline_hours=config.baseline_hours,
        max_age_minutes=config.max_age_minutes,
    )
    service.refresh(force=True)

    print()
    print(f"  WHAT THE FILTER WOULD SEE — {window:.0f} min window, {service.count} held")
    print("  " + "-" * 72)
    print(f"  {'currency':<12}{'recent':>8}{'normal':>9}{'multiple':>11}   verdict")
    print("  " + "-" * 72)
    for code in sorted(CURRENCY_TERMS):
        pressure = service.pressure(code, frozenset({code}))
        blocked = (
            pressure.recent >= config.min_headlines and pressure.multiple >= config.spike_multiple
        ) or (pressure.systemic and config.block_on_systemic)
        print(
            f"  {code:<12}{pressure.recent:>8}{pressure.baseline:>9.1f}"
            f"{pressure.multiple:>10.1f}x   {'WOULD BLOCK' if blocked else 'clear'}"
        )
    print("  " + "-" * 72)
    print(
        f"  Thresholds: at least {config.min_headlines} headlines AND at least "
        f"{config.spike_multiple:.0f}x normal."
    )
    print()
    print("  If a currency reads WOULD BLOCK on an ordinary afternoon, the")
    print("  thresholds are too tight for these feeds. Raise them in")
    print("  config/eightcap.yaml before enabling the filter.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

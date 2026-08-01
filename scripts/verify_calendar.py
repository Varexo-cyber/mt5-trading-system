"""Verify the calendar providers against live feeds. Run this before Phase 8.

The remote parsers in `filters/calendar/providers.py` were written against each
feed's documented shape but could not be exercised against live responses in
the build environment, which blocks outbound HTTPS to third-party hosts. This
script closes that gap:

    python scripts/verify_calendar.py                  # fetch, parse, report
    python scripts/verify_calendar.py --raw            # also dump raw payloads
    python scripts/verify_calendar.py --archive        # append to the archive
    python scripts/verify_calendar.py --symbol EURUSD  # show its blackouts

What to check in the output:

* Both providers reachable, and neither reporting zero high-impact events for
  a normal week — that is the signature of a parser that silently lost the
  impact field.
* The two providers roughly agreeing on the high-impact count. A large gap
  means one of them is being read wrong.
* The blackout windows for a known release (this week's NFP or CPI) starting
  and ending where you expect.

`--archive` appends fetched events to `data/calendar/archive.json`. Run it
weekly from a scheduled task: the free feeds only publish the current and next
week, so the archive is the only way to get a calendar the backtester can use
over a multi-year window.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import PACKAGE_ROOT, load_settings
from core.clock import LiveClock
from filters.calendar.events import deduplicate
from filters.calendar.providers import (
    CalendarUnavailableError,
    FileCalendarProvider,
    build_providers,
)
from filters.calendar.service import CalendarService
from filters.news_filter import NewsFilter
from infra.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="days ahead to fetch")
    parser.add_argument("--raw", action="store_true", help="dump raw payloads to disk")
    parser.add_argument("--archive", action="store_true", help="append to the event archive")
    parser.add_argument("--symbol", help="show blackout windows for this instrument")
    args = parser.parse_args(argv)

    setup_logging(level="INFO", log_dir=PACKAGE_ROOT / "logs", filename="verify_calendar.jsonl")
    settings = load_settings(env_overrides=False)
    news = settings.filters.news

    calendar_dir = PACKAGE_ROOT / Path(news.cache_path).parent
    calendar_dir.mkdir(parents=True, exist_ok=True)
    providers = build_providers(news.providers, calendar_dir=calendar_dir)

    clock = LiveClock()
    now = clock.now()
    start, end = now - timedelta(days=1), now + timedelta(days=args.days)

    print(f"\nwindow: {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} UTC\n")

    everything = []
    failures = 0
    for provider in providers:
        print(f"  {provider.name}")
        try:
            events = provider.fetch(start, end)
        except CalendarUnavailableError as exc:
            failures += 1
            print(f"    FAILED: {exc}\n")
            continue

        high = [e for e in events if e.impact >= 3]
        everything.extend(events)
        print(f"    ok: {len(events)} events, {len(high)} high impact")
        if not high:
            print(
                "    WARNING: zero high-impact events in a full week is the signature "
                "of a parser that lost the impact field. Investigate before trusting this."
            )
        for event in high[:8]:
            print(f"      {event.describe()}")
        print()

        if args.raw:
            dump = calendar_dir / f"raw_{provider.name}_{now:%Y%m%d_%H%M}.json"
            dump.write_text(
                json.dumps(
                    [
                        {
                            "when": e.when.isoformat(),
                            "currency": e.currency,
                            "title": e.title,
                            "impact": e.impact.name,
                        }
                        for e in events
                    ],
                    indent=1,
                ),
                encoding="utf-8",
            )
            print(f"    parsed events written to {dump}\n")

    if failures == len(providers):
        print("EVERY provider failed. Live, this means the system would refuse to trade.")
        return 1

    merged = deduplicate(everything)
    print(f"  merged: {len(merged)} unique events ({len(everything)} before de-duplication)")

    # Cross-check: a release one provider calls high-impact and the other does
    # not is exactly the disagreement worth seeing before going live.
    by_source: dict[str, set[tuple[str, str, str]]] = {}
    for event in everything:
        if event.impact >= 3:
            by_source.setdefault(event.source, set()).add(event.key)
    if len(by_source) > 1:
        names = sorted(by_source)
        only_first = by_source[names[0]] - by_source[names[1]]
        only_second = by_source[names[1]] - by_source[names[0]]
        print(
            f"  high-impact disagreement: {len(only_first)} only in {names[0]}, "
            f"{len(only_second)} only in {names[1]}"
        )
        for key in sorted(only_first | only_second)[:10]:
            print(f"      {key[0]} {key[1]} {key[2]}")

    if args.archive:
        archive = calendar_dir / "archive.json"
        existing = []
        if archive.exists():
            existing = FileCalendarProvider(archive).fetch(
                datetime(1970, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC)
            )
        combined = deduplicate([*existing, *merged])
        FileCalendarProvider.write(archive, combined)
        added = len(combined) - len(existing)
        print(f"\n  archive: {len(combined)} events ({added} new) -> {archive}")

    if args.symbol:
        _show_blackouts(settings, providers, clock, calendar_dir, args.symbol)

    return 0


def _show_blackouts(settings, providers, clock, calendar_dir, symbol: str) -> None:  # type: ignore[no-untyped-def]
    """Print the windows the filter would enforce for one instrument."""
    news_config = settings.filters.news
    service = CalendarService(
        providers,
        clock,
        cache_path=calendar_dir / Path(news_config.cache_path).name,
        refresh_interval_minutes=news_config.refresh_interval_minutes,
        max_age_minutes=news_config.max_calendar_age_minutes,
    )
    filter_ = NewsFilter(news_config, service, clock)

    base, profit = symbol[:3].upper(), symbol[3:6].upper()
    print(f"\n  blackout windows for {symbol} ({base}/{profit}):")
    now = clock.now()
    for blackout in filter_.blackouts_for(base, profit):
        if blackout.end < now:
            continue
        marker = "NOW" if blackout.contains(now) else "   "
        kind = "extreme" if blackout.extreme else "high   "
        print(
            f"    {marker} {kind} {blackout.start:%a %d %H:%M} -> {blackout.end:%H:%M} UTC"
            f"   {blackout.event.currency} {blackout.event.title}"
        )


if __name__ == "__main__":
    raise SystemExit(main())

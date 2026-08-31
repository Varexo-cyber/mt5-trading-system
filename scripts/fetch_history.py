"""Download the bars once and keep them, so no measurement fetches twice.

    python scripts/fetch_history.py                 180 days, core 16, all six clocks
    python scripts/fetch_history.py --days 365
    python scripts/fetch_history.py --timeframes M15 M30 H1 H4
    python scripts/fetch_history.py --force         refetch what is already there

WHAT IT COSTS AND WHAT IT SAVES. 180 days over sixteen markets:

    M1    ~186,000 bars a market      2,970,000 total
    M5     ~37,000                      594,000
    M15    ~12,400                      198,000
    M30     ~6,200                       99,000
    H1      ~3,100                       50,000
    H4        ~780                       12,000
                                     ---------
                                     ~3,900,000 bars, roughly 60-150 MB on disk

That download happens ONCE. Every `sweep.cmd` and `dryrun.cmd` afterwards can
read it off the disk with `--cache`, with MT5 closed, from any machine holding
the folder.

RESUMABLE, because a fetch of this size will be interrupted. Each
(symbol, timeframe) is written as it arrives and skipped on the next run, so
stopping this halfway and starting it again costs only what was missing. The
manifest is written whole and moved into place, so an interruption cannot leave
a half-written index that makes every stored spec disappear.

THE SPECS COME TOO. Sizing needs `volume_min`, `tick_value` and `point`, and
those live in `mt5.symbol_info()`. A store of bars alone would still require a
live terminal for every run, which is the thing being removed.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.history_store import HistoryStore
from backtesting.replay import fetch_mt5_history
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from core.types import Timeframe
from scripts.dry_run_sections import WARMUP, _core_universe

#: Every clock the sweep can put a section on. Fetched together because the
#: expensive part is being connected and walking the catalogue, not the last
#: timeframe in the list.
ALL_CLOCKS: tuple[str, ...] = ("M1", "M5", "M15", "M30", "H1", "H4")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=list(ALL_CLOCKS),
        metavar="TF",
        help="clocks to store, space or comma separated (default: all six)",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="comma list; default is the sixteen core markets",
    )
    parser.add_argument("--out", default="data/history", help="where to keep the store")
    parser.add_argument(
        "--force",
        action="store_true",
        help="refetch and overwrite what is already stored",
    )
    return parser


def _wanted_clocks(values: list[str]) -> list[Timeframe]:
    names = [
        piece.strip().upper()
        for chunk in values
        for piece in str(chunk).split(",")
        if piece.strip()
    ]
    return [Timeframe.parse(name) for name in names]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=True)
    store = HistoryStore(ROOT / args.out if not Path(args.out).is_absolute() else args.out)
    clocks = _wanted_clocks(args.timeframes)

    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)

    connector = MT5Connector(
        settings.mt5,
        load_credentials(),
        terminal_path=terminal_path_from_env(),
    )
    connector.connect()
    try:
        if args.symbols:
            symbols = [name.strip() for name in args.symbols.split(",") if name.strip()]
        else:
            symbols = _core_universe(connector, settings)

        print(f"\n{'=' * 78}")
        print(f"STORING HISTORY — {args.days} days to {end:%Y-%m-%d %H:%M} UTC")
        print(f"{'=' * 78}")
        print(f"  {len(symbols)} markets x {len(clocks)} clocks into {store.root}")
        print(f"  {', '.join(tf.value for tf in clocks)}")
        if not args.force:
            print("  already-stored files are skipped; --force refetches them")
        print()

        began = time.perf_counter()
        written = skipped = failed = 0
        bars_total = 0

        for index, symbol in enumerate(symbols, 1):
            try:
                store.write_spec(connector.spec(symbol))
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the run
                print(f"  [{index}/{len(symbols)}] {symbol}: no spec ({exc})")
                failed += 1
                continue

            parts: list[str] = []
            for timeframe in clocks:
                if store.has(symbol, timeframe) and not args.force:
                    skipped += 1
                    parts.append(f"{timeframe.value} kept")
                    continue
                try:
                    # The same padding the measurement uses, so a stored frame
                    # is deep enough for the warm-up the sweep needs and the
                    # first day of the window is not silently unusable.
                    padded = start - max(
                        (WARMUP + 20) * timeframe.duration * 1.6, timedelta(days=3)
                    )
                    frame = fetch_mt5_history(connector, symbol, timeframe, padded, end)
                    bars = store.write(symbol, timeframe, frame)
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    parts.append(f"{timeframe.value} FAILED ({str(exc)[:40]})")
                    continue
                written += 1
                bars_total += bars
                parts.append(f"{timeframe.value} {bars:,}")

            print(f"  [{index}/{len(symbols)}] {symbol:<12} {'  '.join(parts)}", flush=True)

        store.note_window(args.days, end)
    finally:
        connector.shutdown()

    took = time.perf_counter() - began
    megabytes = store.size_bytes() / 1_048_576
    print(f"\n{'=' * 78}")
    print(f"  {written} frames written, {skipped} already present, {failed} failed")
    print(f"  {bars_total:,} new bars, {megabytes:.1f} MB on disk at {store.root}")
    print(f"  took {took / 60:.1f} min")
    if failed:
        print("\n  FAILURES ARE NOT FATAL and the store is usable without them, but a")
        print("  clock that is missing for a market is a row that will silently not")
        print("  appear in the next sweep. Re-run to fill the gaps; what succeeded")
        print("  is kept.")
    print("\n  Now measure without touching MT5:")
    print(f"    dryrun.cmd --cache {args.out}")
    print(f"    sweep.cmd 180 --cache {args.out}")
    print(f"{'=' * 78}\n")
    return 1 if failed and not written else 0


if __name__ == "__main__":
    raise SystemExit(main())

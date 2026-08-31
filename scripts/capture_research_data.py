"""Capture a reusable multi-market, multi-timeframe MT5 research database."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtesting.replay import fetch_mt5_history
from backtesting.research_dataset import BAR_COLUMNS, ResearchDataset
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from core.types import Timeframe

DEFAULT_SYMBOLS = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
    "EURGBP",
    "EURJPY",
    "XAUUSD",
    "GBPJPY",
    "EURCHF",
    "US30",
    "NDX100",
    "SPX500",
    "GER40",
)
DEFAULT_TIMEFRAMES = (
    Timeframe.M1,
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.M30,
    Timeframe.H1,
    Timeframe.H4,
)


def _base(name: str) -> str:
    return "".join(character for character in name.upper() if character.isalnum())


def resolve_symbols(catalogue: list[str], wanted: tuple[str, ...]) -> dict[str, str]:
    """Resolve broker decorations without confusing EURUSD with EURUSDX."""

    resolved: dict[str, str] = {}
    for canonical in wanted:
        key = _base(canonical)

        def rank(
            name: str, *, expected_key: str = key, expected_name: str = canonical
        ) -> tuple[int, int]:
            normal = _base(name)
            if normal == expected_key:
                return (0, len(name))
            tail = name.upper().replace(expected_name.upper(), "", 1)
            separated = bool(tail) and not tail[0].isalnum()
            return (1 if separated else 2, len(name))

        matches = sorted((name for name in catalogue if _base(name).startswith(key)), key=rank)
        if not matches:
            raise SystemExit(
                f"broker has no symbol matching {canonical}; nothing was guessed or skipped"
            )
        resolved[canonical] = matches[0]
    if len(set(resolved.values())) != len(resolved):
        raise SystemExit(f"symbol resolution is not one-to-one: {resolved}")
    return resolved


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--days", type=int, default=180)
    command.add_argument("--equity", type=float, default=203.0)
    command.add_argument("--warmup-bars", type=int, default=400)
    command.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runtime" / "research" / "market-history.sqlite3",
    )
    command.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    command.add_argument("--timeframes", default="M1,M5,M15,M30,H1,H4")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.days <= 0 or args.warmup_bars < 0 or args.equity <= 0:
        raise SystemExit("days and equity must be positive; warmup-bars cannot be negative")
    wanted = tuple(item.strip().upper() for item in args.symbols.split(",") if item.strip())
    timeframes = tuple(Timeframe.parse(item) for item in args.timeframes.split(",") if item.strip())
    if not wanted or not timeframes:
        raise SystemExit("at least one symbol and timeframe are required")

    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=True)
    credentials = load_credentials(required=True)
    connector = MT5Connector(
        settings.mt5,
        credentials,
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    account = connector.connect()
    captured = datetime.now(UTC)
    evaluation_start = captured - timedelta(days=args.days)
    failures: list[str] = []
    try:
        catalogue = [item.name for item in connector.symbols()]
        symbols = resolve_symbols(catalogue, wanted)
        print("Resolved broker symbols:")
        for canonical, broker in symbols.items():
            print(f"  {canonical:<8} -> {broker}")
        print(f"\nOutput: {args.output}")
        print(f"Evaluation window: {evaluation_start.isoformat()} -> {captured.isoformat()}")
        print(f"Research equity: EUR {args.equity:.2f}\n")

        with ResearchDataset(args.output) as dataset:
            dataset.set_metadata("captured_at", captured.isoformat())
            dataset.set_metadata("evaluation_start", evaluation_start.isoformat())
            dataset.set_metadata("evaluation_end", captured.isoformat())
            dataset.set_metadata("evaluation_days", args.days)
            dataset.set_metadata("warmup_bars", args.warmup_bars)
            dataset.set_metadata("research_starting_equity", args.equity)
            dataset.set_metadata(
                "account",
                {
                    "server": account.server,
                    "currency": account.currency,
                    "leverage": account.leverage,
                    "captured_balance": account.balance,
                    "captured_equity": account.equity,
                },
            )
            dataset.set_metadata("settings", settings.model_dump(mode="json"))
            dataset.set_metadata("symbols", symbols)
            dataset.set_metadata("timeframes", [timeframe.value for timeframe in timeframes])

            total = len(symbols) * len(timeframes)
            done = 0
            for canonical, symbol in symbols.items():
                try:
                    dataset.put_instrument(canonical, connector.spec(symbol))
                except Exception as exc:  # noqa: BLE001 - finish the resumable capture
                    failures.append(f"{canonical}/{symbol} specification: {exc}")
                    print(f"ERROR {failures[-1]}")
                    continue
                for timeframe in timeframes:
                    done += 1
                    warmup = timeframe.duration * args.warmup_bars * 1.6
                    requested_start = evaluation_start - max(warmup, timedelta(days=3))
                    print(
                        f"[{done:>2}/{total}] {symbol:<10} {timeframe.value:<3} fetching ... ",
                        end="",
                        flush=True,
                    )
                    try:
                        frame = fetch_mt5_history(
                            connector,
                            symbol,
                            timeframe,
                            requested_start,
                            captured,
                            max_bars_per_chunk=20_000,
                        )
                        # The forming candle is not research evidence.
                        frame = frame.loc[
                            (frame.index >= pd.Timestamp(requested_start))
                            & (frame.index + timeframe.duration <= pd.Timestamp(captured)),
                            list(BAR_COLUMNS),
                        ]
                        _check_history(frame, requested_start)
                        dataset.put_frame(
                            symbol,
                            timeframe,
                            frame,
                            requested_from=pd.Timestamp(requested_start),
                            requested_to=pd.Timestamp(captured),
                            captured_at=captured.isoformat(),
                        )
                        print(
                            f"{len(frame):,} bars  "
                            f"{frame.index[0].date()} -> {frame.index[-1].date()}"
                        )
                    except Exception as exc:  # noqa: BLE001 - report every missing slice
                        failure = f"{canonical}/{symbol} {timeframe.value}: {exc}"
                        failures.append(failure)
                        print(f"ERROR {failure}")
            dataset.set_metadata("complete", not failures)
            dataset.set_metadata("failures", failures)
            dataset.connection.execute("PRAGMA optimize")

        size_mb = args.output.stat().st_size / (1024 * 1024) if args.output.exists() else 0.0
        print(f"\nDatabase: {args.output} ({size_mb:.1f} MB)")
        if failures:
            print(
                f"INCOMPLETE: {len(failures)} slice(s) failed. "
                "Re-run the same command to resume."
            )
            for failure in failures:
                print(f"  - {failure}")
            return 1
        print("COMPLETE: all requested markets, clocks, costs and contract specs are stored.")
        return 0
    finally:
        connector.shutdown()


def _check_history(frame: pd.DataFrame, requested_start: datetime) -> None:
    if frame.empty:
        raise ValueError("no closed bars returned")
    # A week allows a weekend plus a holiday. Anything later means MT5's
    # Max bars setting truncated the requested history.
    latest_acceptable_start = pd.Timestamp(requested_start + timedelta(days=7))
    if frame.index[0] > latest_acceptable_start:
        raise ValueError(
            f"history begins {frame.index[0].isoformat()}, requested "
            f"{requested_start.isoformat()}; increase MT5 'Max bars in chart'"
        )


if __name__ == "__main__":
    raise SystemExit(main())

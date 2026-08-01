"""Fetch MT5 history and run look-ahead-safe multi-timeframe validation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import (
    ConfluenceEngine,
    LevelReaction,
    LiquiditySweep,
    MarketStructure,
    TrendMomentum,
    VolatilityRegime,
)
from backtesting.replay import (
    REPLAY_TIMEFRAMES,
    HistoricalContextReplay,
    archive_frame,
    build_segment_evidence,
    fetch_mt5_history,
    write_evidence_report,
)
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from core.types import Timeframe


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--symbol", action="append", required=True)
    result.add_argument("--start", required=True, help="UTC date, e.g. 2023-01-01")
    result.add_argument("--end", required=True, help="exclusive UTC date")
    result.add_argument("--configurations-tested", type=int, required=True)
    result.add_argument("--stride", type=int, default=1, help="evaluate every Nth H1 close")
    result.add_argument("--unlock-holdout", action="store_true")
    return result


def utc_date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def engine(
    settings,  # type: ignore[no-untyped-def]
    *,
    external_lookback: int = 3,
    bos_buffer_atr: float = 0.05,
):  # type: ignore[no-untyped-def]
    structure = settings.analysis.market_structure.model_copy(
        update={
            "external_swing_lookback": external_lookback,
            "bos_close_buffer_atr": bos_buffer_atr,
        }
    )
    confluence = settings.analysis.confluence.model_copy(
        update={
            "score_threshold": 1.0,
            "minimum_confidence": structure.minimum_confidence,
            "minimum_directional_modules": 1,
            "minimum_agreement_ratio": 0.5,
            "weights": {
                name: 1.0 if name == "market_structure" else 0.0
                for name in settings.analysis.confluence.weights
            },
        }
    )
    return ConfluenceEngine(
        [
            MarketStructure(structure),
            TrendMomentum(),
            LiquiditySweep(),
            LevelReaction(),
            VolatilityRegime(),
        ],
        confluence,
    )


def main() -> int:
    args = parser().parse_args()
    if args.configurations_tested < 1:
        raise SystemExit("--configurations-tested must be at least 1")
    hypothesis = ROOT / "docs" / "hypotheses" / "market_structure.md"
    if not hypothesis.exists():
        raise SystemExit("pre-registration missing: docs/hypotheses/market_structure.md")
    start, end = utc_date(args.start), utc_date(args.end)
    if end <= start:
        raise SystemExit("--end must be after --start")

    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=False),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    evidence = []
    sweep_rows: list[dict[str, object]] = []
    nominal_validation: dict[str, tuple[int, float]] = {}
    access_path = ROOT / "runtime" / "validation" / "HOLDOUT_ACCESSES.json"
    accesses = json.loads(access_path.read_text(encoding="utf-8")) if access_path.exists() else {}
    try:
        connector.connect()
        for symbol in args.symbol:
            connector.tick(symbol)  # calibrate broker-wall-clock timestamps first
            spec = connector.spec(symbol)
            warmup = start - timedelta(days=500)
            frames = {}
            for timeframe in REPLAY_TIMEFRAMES:
                frame = fetch_mt5_history(
                    connector,
                    symbol,
                    timeframe,
                    warmup,
                    end + timeframe.duration,
                )
                frames[timeframe] = frame
                archive_frame(
                    frame,
                    ROOT / "data" / "history" / symbol / f"{timeframe.value}.csv",
                )

            decisions = frames[Timeframe.H1]
            eligible = decisions[
                (decisions.index + Timeframe.H1.duration >= start)
                & (decisions.index + Timeframe.H1.duration < end)
            ]
            if len(eligible) < 10:
                raise RuntimeError(f"{symbol}: fewer than 10 H1 decisions in requested range")
            train_at = (
                eligible.index[int(len(eligible) * 0.60)] + Timeframe.H1.duration
            ).to_pydatetime()
            validation_at = (
                eligible.index[int(len(eligible) * 0.80)] + Timeframe.H1.duration
            ).to_pydatetime()
            holdout_key = f"market_structure|{symbol}|{start.isoformat()}|{end.isoformat()}"
            if args.unlock_holdout and holdout_key in accesses:
                raise RuntimeError(
                    f"holdout already accessed at {accesses[holdout_key]}; it is no longer OOS"
                )
            execution = frames[Timeframe.M5]
            nominal_orders = []
            for lookback in (2, 3, 4):
                for buffer in (0.0, 0.05, 0.10):
                    variant = HistoricalContextReplay(
                        engine(
                            settings,
                            external_lookback=lookback,
                            bos_buffer_atr=buffer,
                        ),
                        decision_stride_bars=args.stride,
                    )
                    variant_orders = variant.orders(
                        symbol,
                        frames,
                        point=spec.point,
                        start=start,
                        end=validation_at,
                    )
                    validation_frame = execution[
                        (execution.index >= train_at) & (execution.index < validation_at)
                    ]
                    variant_evidence, _ = build_segment_evidence(
                        f"{symbol}/validation",
                        validation_frame,
                        variant_orders,
                        start=train_at,
                        end=validation_at,
                        configurations_tested=max(args.configurations_tested, 9),
                    )
                    sweep_rows.append(
                        {
                            "symbol": symbol,
                            "external_swing_lookback": lookback,
                            "bos_close_buffer_atr": buffer,
                            "trades": variant_evidence.trades,
                            "expectancy_r": variant_evidence.expectancy_r,
                            "deflated_sharpe_probability": (
                                variant_evidence.deflated_sharpe_probability
                            ),
                        }
                    )
                    if lookback == 3 and buffer == 0.05:
                        nominal_orders = variant_orders
                        nominal_validation[symbol] = (
                            variant_evidence.trades,
                            variant_evidence.expectancy_r,
                        )
            if args.unlock_holdout:
                nominal_orders = HistoricalContextReplay(
                    engine(settings), decision_stride_bars=args.stride
                ).orders(symbol, frames, point=spec.point, start=start, end=end)
            segments = [
                ("train", start, train_at),
                ("validation", train_at, validation_at),
            ]
            if args.unlock_holdout:
                segments.append(("holdout", validation_at, end))
            for name, segment_start, segment_end in segments:
                segment_frame = execution[
                    (execution.index >= segment_start) & (execution.index < segment_end)
                ]
                segment, _ = build_segment_evidence(
                    f"{symbol}/{name}",
                    segment_frame,
                    nominal_orders,
                    start=segment_start,
                    end=segment_end,
                    configurations_tested=max(args.configurations_tested, 9),
                )
                evidence.append(segment)
            if args.unlock_holdout:
                accesses[holdout_key] = datetime.now(UTC).isoformat()
    finally:
        connector.shutdown()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "runtime" / "validation" / f"strategy-{stamp}.json"
    development = ("EURUSD", "GBPUSD", "USDJPY")
    holdbacks = ("AUDUSD", "USDCAD")

    def matching(prefix: str) -> list[dict[str, object]]:
        return [row for row in sweep_rows if str(row["symbol"]).upper().startswith(prefix)]

    development_plateau = all(
        len(matching(prefix)) == 9
        and sum(
            int(row["trades"]) >= 100 and float(row["expectancy_r"]) > 0 for row in matching(prefix)
        )
        >= 6
        for prefix in development
    )
    holdback_passed = all(
        any(
            symbol.upper().startswith(prefix) and trades >= 100 and expectancy > 0
            for symbol, (trades, expectancy) in nominal_validation.items()
        )
        for prefix in holdbacks
    )
    write_evidence_report(
        output,
        metadata={
            "hypothesis": str(hypothesis.relative_to(ROOT)),
            "symbols": args.symbol,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "configurations_tested": max(args.configurations_tested, 9),
            "decision_stride_h1_bars": args.stride,
            "holdout_unlocked": args.unlock_holdout,
            "validated_modules": ["market_structure"],
            "parameter_sweep": sweep_rows,
            "parameter_stability_passed": development_plateau,
            "independent_holdback_passed": holdback_passed,
            "warning": "Results below 100 trades per module are inconclusive.",
        },
        segments=evidence,
    )
    if args.unlock_holdout:
        access_path.parent.mkdir(parents=True, exist_ok=True)
        access_path.write_text(json.dumps(accesses, indent=2), encoding="utf-8")
    print(output)
    for segment in evidence:
        print(
            f"{segment.name}: trades={segment.trades}, expectancy={segment.expectancy_r:.3f}R, "
            f"maxDD={segment.max_drawdown_r:.2f}R, DSR={segment.deflated_sharpe_probability:.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

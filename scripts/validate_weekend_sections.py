"""Out-of-sample evidence gate for Jarvis sections 1, 2 and 6.

This is deliberately a promotion tool, not an optimiser.  It runs the current
registered settings, charges spread/commission/slippage, uses each detector's
native decision clock, and keeps the final 20 percent hidden unless explicitly
unlocked.  A large setup count is reported, but cannot compensate for negative
expectancy.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analysis import BASKET_META_KEY, ConfluenceEngine, PeerMove
from backtesting.engine import (
    BacktestOrder,
    PessimisticBacktester,
    deflated_sharpe_probability,
)
from backtesting.replay import REPLAY_TIMEFRAMES, HistoricalContextReplay, fetch_mt5_history
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from core.types import Timeframe, TradingMode
from runner.service import build_analysis_modules
from scripts.backtest_section_six import proposals as section_six_proposals

DEFAULT_SYMBOLS = (
    "EURUSD.i",
    "GBPUSD.i",
    "USDJPY.i",
    "AUDUSD.i",
    "ETHUSD",
    "BTCUSD",
    "US30",
    "NDX100",
    "FRA40",
    "GER40",
)
CHECKPOINT_VERSION = 3


@dataclass(frozen=True, slots=True)
class StrategySpec:
    section: str
    module: str
    decision_timeframe: Timeframe
    execution_timeframe: Timeframe


STRATEGIES = (
    StrategySpec("1", "market_structure", Timeframe.H1, Timeframe.M5),
    StrategySpec("1", "trend_momentum", Timeframe.H1, Timeframe.M5),
    StrategySpec("1", "m1_micro_breakout", Timeframe.M1, Timeframe.M1),
    StrategySpec("1", "basket_divergence", Timeframe.M1, Timeframe.M1),
    # Candle momentum is a section-1 confirmer and the section-6 trigger.  Its
    # independent read is measured here; its separate lane is measured below.
    StrategySpec("1", "candle_momentum", Timeframe.M1, Timeframe.M1),
    StrategySpec("2", "vwap_reversion", Timeframe.M5, Timeframe.M5),
)
STRATEGY_NAMES = (*(spec.module for spec in STRATEGIES), "own_lane")


@dataclass(frozen=True, slots=True)
class Verdict:
    section: str
    strategy: str
    segment: str
    setups: int
    trades: int
    total_r: float
    expectancy_r: float
    win_rate: float
    profit_factor: float | None
    max_drawdown_r: float
    dsr: float


@dataclass(frozen=True, slots=True)
class Diagnostic:
    section: str
    strategy: str
    symbol: str
    segment: str
    decisions: int
    firings: int
    approved: int
    trades: int
    total_r: float
    top_refusal: str


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--days", type=int, default=180)
    result.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    result.add_argument("--stride", type=int, default=1)
    result.add_argument(
        "--strategy",
        action="append",
        choices=STRATEGY_NAMES,
        help="test only this strategy; repeat the option to select more than one",
    )
    result.add_argument(
        "--fresh",
        action="store_true",
        help="ignore and replace a matching unfinished checkpoint",
    )
    result.add_argument(
        "--unlock-holdout",
        action="store_true",
        help="include the untouched final 20%%; use only after reviewing validation",
    )
    return result


def isolated_engine(settings, module: str) -> ConfluenceEngine:  # type: ignore[no-untyped-def]
    """Run one directional opinion through the production plan builder.

    Regime readers and structural plan construction still run.  Only the vote
    is isolated, so a profitable passenger cannot make a losing detector look
    good and vice versa.
    """

    original = settings.analysis.confluence
    names = set(original.weights) | {module}
    confluence = original.model_copy(
        update={
            "score_threshold": 1.0,
            "minimum_confidence": 0.0,
            "minimum_directional_modules": 1,
            "minimum_agreement_ratio": 0.5,
            "weights": {name: 1.0 if name == module else 0.0 for name in names},
        }
    )
    # The old validator set seventeen weights to zero but still executed all
    # eighteen readers on every M1 close.  A zero-weight opinion cannot affect
    # the isolated verdict, yet it consumed almost the whole runtime.  Keep
    # only the selected detector and the two regime readers the engine itself
    # consults as safety/context inputs.
    required = {module, "market_regime", "volatility_regime"}
    modules = [reader for reader in build_analysis_modules(settings) if reader.name in required]
    return ConfluenceEngine(modules, confluence)


def basket_enricher(
    symbol: str,
    frames_by_symbol: dict[str, dict[Timeframe, pd.DataFrame]],
    asset_classes: dict[str, str],
    *,
    move_bars: int,
):  # type: ignore[no-untyped-def]
    """Attach simultaneous, closed peer bars instead of scan-order leftovers."""

    peer_close_times = {
        other: frames[Timeframe.M1].index + Timeframe.M1.duration
        for other, frames in frames_by_symbol.items()
        if Timeframe.M1 in frames
    }

    def enrich(context) -> None:  # type: ignore[no-untyped-def]
        if asset_classes.get(symbol) != "index":
            return
        peers: list[PeerMove] = []
        for other, frames in frames_by_symbol.items():
            if other == symbol or asset_classes.get(other) != "index":
                continue
            minute = frames.get(Timeframe.M1)
            if minute is None or minute.empty:
                continue
            cut = int(peer_close_times[other].searchsorted(pd.Timestamp(context.now), side="right"))
            if cut < move_bars + 1:
                continue
            sample = minute.iloc[cut - (move_bars + 1) : cut]
            first, last = float(sample["close"].iloc[0]), float(sample["close"].iloc[-1])
            if first <= 0 or last <= 0:
                continue
            stamp = sample.index[-1].to_pydatetime()
            peers.append(
                PeerMove(
                    symbol=other,
                    move_bp=(last / first - 1.0) * 10_000.0,
                    age_seconds=max(0.0, (context.now - stamp).total_seconds()),
                )
            )
        if peers:
            context.meta[BASKET_META_KEY] = peers

    return enrich


def summarise(
    section: str,
    strategy: str,
    segment: str,
    setups: int,
    returns: list[float],
) -> Verdict:
    values = np.asarray(returns, dtype=float)
    if not len(values):
        return Verdict(section, strategy, segment, setups, 0, 0.0, 0.0, 0.0, None, 0.0, 0.0)
    wins, losses = values[values > 0], values[values < 0]
    curve = np.cumsum(values)
    drawdown = np.maximum.accumulate(np.insert(curve, 0, 0.0))[1:] - curve
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) else None
    return Verdict(
        section,
        strategy,
        segment,
        setups,
        len(values),
        float(values.sum()),
        float(values.mean()),
        float((values > 0).mean()),
        profit_factor,
        float(drawdown.max(initial=0.0)),
        deflated_sharpe_probability(values.tolist(), configurations_tested=1),
    )


def passed(row: Verdict) -> bool:
    """Predeclared promotion floor; volume never cancels a negative edge."""

    profit_factor = float("inf") if row.profit_factor is None else row.profit_factor
    return (
        row.trades >= 100 and row.expectancy_r >= 0.05 and profit_factor >= 1.10 and row.dsr >= 0.80
    )


def render(rows: list[Verdict], *, holdout: bool) -> str:
    lines = [
        "WEEKEND VALIDATION — sections 1, 2 and 6",
        "",
        "Costs included. Natural clocks: H1/M5/M1. Final 20% "
        + ("UNLOCKED." if holdout else "LOCKED."),
        "Promotion floor: >=100 trades, >=+0.05R/trade, PF>=1.10, DSR>=80%",
        "",
        f"{'sec':<5}{'strategy':<24}{'segment':<12}{'setups':>9}{'trades':>9}"
        f"{'exp':>10}{'PF':>8}{'DD':>9}{'DSR':>8}{'gate':>10}",
        "-" * 104,
    ]
    for row in rows:
        pf = (
            "inf"
            if row.profit_factor is None and row.trades
            else f"{(row.profit_factor if row.profit_factor is not None else 0):.2f}"
        )
        gate = "PASS" if passed(row) else ("TOO THIN" if row.trades < 100 else "FAIL")
        lines.append(
            f"{row.section:<5}{row.strategy:<24}{row.segment:<12}{row.setups:>9}{row.trades:>9}"
            f"{row.expectancy_r:>+9.3f}R{pf:>8}{row.max_drawdown_r:>8.2f}R"
            f"{row.dsr:>7.0%}{gate:>10}"
        )
    lines.extend(
        [
            "",
            "LIVE VERDICT",
            "A strategy is a live candidate only when TRAIN and VALIDATION both PASS.",
            "Holdout is opened once, after choices are frozen; it is not tuning data.",
        ]
    )
    strategies = sorted({(row.section, row.strategy) for row in rows})
    for section, strategy in strategies:
        relevant = [row for row in rows if row.section == section and row.strategy == strategy]
        development = [row for row in relevant if row.segment in {"train", "validation"}]
        candidate = len(development) == 2 and all(passed(row) for row in development)
        if holdout:
            final = next((row for row in relevant if row.segment == "holdout"), None)
            candidate = candidate and final is not None and passed(final)
        label = "LIVE CANDIDATE" if candidate else "KEEP OFF"
        lines.append(f"  section {section} / {strategy}: {label}")
    return "\n".join(lines) + "\n"


def render_diagnostics(rows: list[Diagnostic]) -> str:
    """Separate detector opportunities from engine-approved setups by market."""
    if not rows:
        return ""
    lines = [
        "",
        "BY MARKET — firing is a raw detector opinion; approved is an executable setup",
        f"{'strategy':<20}{'symbol':<12}{'segment':<12}{'fired':>8}{'approved':>10}"
        f"{'trades':>8}{'total':>10}  top refusal",
        "-" * 112,
    ]
    for row in rows:
        lines.append(
            f"{row.strategy:<20}{row.symbol:<12}{row.segment:<12}{row.firings:>8}"
            f"{row.approved:>10}{row.trades:>8}{row.total_r:>+9.2f}R  "
            f"{row.top_refusal[:45]}"
        )
    return "\n".join(lines) + "\n"


def save_checkpoint(
    path: Path,
    *,
    signature: dict[str, object],
    start: datetime,
    end: datetime,
    rows: list[Verdict],
    diagnostics: list[Diagnostic] | None = None,
) -> None:
    """Persist every completed strategy so an interruption resumes safely."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "signature": signature,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "rows": [asdict(row) for row in rows],
                "diagnostics": [asdict(row) for row in diagnostics or []],
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.days < 30:
        raise SystemExit("Use at least 30 days; 180 is preferred.")
    if args.stride < 1:
        raise SystemExit("--stride must be positive")
    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml")
    selected_strategies = set(args.strategy or STRATEGY_NAMES)
    output_dir = ROOT / "runtime" / "validation"
    checkpoint = output_dir / "weekend-sections-checkpoint.json"
    signature = {
        "version": CHECKPOINT_VERSION,
        "days": args.days,
        "stride": args.stride,
        "holdout": args.unlock_holdout,
        "symbols": list(args.symbols),
        "strategies": sorted(selected_strategies),
        "config": hashlib.sha256(settings.model_dump_json().encode()).hexdigest(),
    }
    rows: list[Verdict] = []
    diagnostics: list[Diagnostic] = []
    saved: dict[str, object] | None = None
    if checkpoint.exists() and not args.fresh:
        with contextlib.suppress(ValueError, OSError, json.JSONDecodeError):
            candidate = json.loads(checkpoint.read_text(encoding="utf-8"))
            if candidate.get("signature") == signature:
                saved = candidate
    if saved is not None:
        start = datetime.fromisoformat(str(saved["start"]))
        end = datetime.fromisoformat(str(saved["end"]))
        rows = [Verdict(**row) for row in saved.get("rows", [])]
        diagnostics = [Diagnostic(**row) for row in saved.get("diagnostics", [])]
        completed = sorted({f"{row.section}/{row.strategy}" for row in rows})
        print(f"resuming checkpoint; already complete: {', '.join(completed)}", flush=True)
    else:
        end = datetime.now(UTC)
        start = end - timedelta(days=args.days)
    train_end = start + (end - start) * 0.60
    validation_end = start + (end - start) * 0.80
    segments = [("train", start, train_end), ("validation", train_end, validation_end)]
    if args.unlock_holdout:
        segments.append(("holdout", validation_end, end))
    fetch_end = end if args.unlock_holdout else validation_end

    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=False),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    frames_by_symbol: dict[str, dict[Timeframe, pd.DataFrame]] = {}
    specs = {}
    asset_classes: dict[str, str] = {}
    try:
        connector.connect()
        for symbol in args.symbols:
            print(f"fetching {symbol} ...", flush=True)
            try:
                connector.tick(symbol)
                spec = connector.spec(symbol)
                frames = {}
                for timeframe in (*REPLAY_TIMEFRAMES, Timeframe.M1):
                    warmup = start - timeframe.duration * 400
                    frames[timeframe] = fetch_mt5_history(
                        connector, symbol, timeframe, warmup, fetch_end + timeframe.duration
                    )
                frames_by_symbol[symbol] = frames
                specs[symbol] = spec
                asset_classes[symbol] = spec.asset_class.value
            except Exception as exc:  # noqa: BLE001 - report every unavailable market
                print(f"  skipped {symbol}: {type(exc).__name__}: {exc}")

        if not frames_by_symbol:
            raise RuntimeError("No requested symbol returned complete history")

        for strategy in (item for item in STRATEGIES if item.module in selected_strategies):
            if any(
                row.section == strategy.section and row.strategy == strategy.module for row in rows
            ):
                print(
                    f"checkpoint: skipping section {strategy.section} / {strategy.module}",
                    flush=True,
                )
                continue
            print(f"replaying section {strategy.section} / {strategy.module} ...", flush=True)
            per_segment_returns = {name: [] for name, _, _ in segments}
            per_segment_setups = {name: 0 for name, _, _ in segments}
            for symbol, frames in frames_by_symbol.items():
                print(f"  {strategy.module}: {symbol}", flush=True)
                enricher = None
                if strategy.module == "basket_divergence":
                    enricher = basket_enricher(
                        symbol,
                        frames_by_symbol,
                        asset_classes,
                        move_bars=settings.analysis.basket_divergence.move_bars,
                    )
                replay = HistoricalContextReplay(
                    isolated_engine(settings, strategy.module),
                    decision_stride_bars=args.stride,
                    decision_timeframe=strategy.decision_timeframe,
                    execution_timeframe=strategy.execution_timeframe,
                    mode=TradingMode.BACKTEST,
                    context_enricher=enricher,
                )
                orders: list[BacktestOrder] = []
                decisions: Counter[str] = Counter()
                firings: Counter[str] = Counter()
                approvals: Counter[str] = Counter()
                refusals: dict[str, Counter[str]] = {name: Counter() for name, _, _ in segments}
                for decided_at, spread, idea in replay.ideas(
                    symbol,
                    frames,
                    point=specs[symbol].point,
                    start=start,
                    end=fetch_end,
                ):
                    segment = next(
                        (name for name, left, right in segments if left <= decided_at < right),
                        None,
                    )
                    if segment is None:
                        continue
                    decisions[segment] += 1
                    target = next(
                        (signal for signal in idea.signals if signal.module == strategy.module),
                        None,
                    )
                    if target is not None and target.score != 0 and target.confidence > 0:
                        firings[segment] += 1
                    order = replay.order_from_idea(symbol, decided_at, spread, idea)
                    if order is None:
                        refusals[segment][idea.reason] += 1
                    else:
                        approvals[segment] += 1
                        orders.append(order)
                execution = frames[strategy.execution_timeframe]
                for name, left, right in segments:
                    selected = [order for order in orders if left <= order.decided_at < right]
                    # A train trade may not borrow its exit from validation,
                    # and validation may not peek into holdout.  Clip the bars
                    # as well as the order timestamps; anything still open is
                    # conservatively marked to market at the split boundary.
                    segment_execution = execution[
                        (execution.index >= left) & (execution.index < right)
                    ]
                    result = PessimisticBacktester().run_non_overlapping(
                        segment_execution, selected
                    )
                    per_segment_setups[name] += len(selected)
                    per_segment_returns[name].extend(trade.net_r for trade in result.trades)
                    top_refusal = refusals[name].most_common(1)[0][0] if refusals[name] else "-"
                    diagnostics.append(
                        Diagnostic(
                            strategy.section,
                            strategy.module,
                            symbol,
                            name,
                            decisions[name],
                            firings[name],
                            approvals[name],
                            len(result.trades),
                            sum(trade.net_r for trade in result.trades),
                            top_refusal,
                        )
                    )
            for name, _, _ in segments:
                rows.append(
                    summarise(
                        strategy.section,
                        strategy.module,
                        name,
                        per_segment_setups[name],
                        per_segment_returns[name],
                    )
                )
            save_checkpoint(
                checkpoint,
                signature=signature,
                start=start,
                end=end,
                rows=rows,
                diagnostics=diagnostics,
            )

        lane_selected = "own_lane" in selected_strategies
        lane_complete = any(row.section == "6" and row.strategy == "own_lane" for row in rows)
        if lane_complete:
            print("checkpoint: skipping section 6 / own_lane", flush=True)
        elif lane_selected:
            print("replaying section 6 / own_lane ...", flush=True)
        lane_returns = {name: [] for name, _, _ in segments}
        lane_setups = {name: 0 for name, _, _ in segments}
        for symbol, frames in (
            frames_by_symbol.items() if lane_selected and not lane_complete else ()
        ):
            if asset_classes[symbol] not in {"index", "metal", "crypto"}:
                continue
            orders: list[BacktestOrder] = section_six_proposals(
                symbol,
                {tf: frames[tf] for tf in (Timeframe.M1, Timeframe.M5, Timeframe.M15)},
                settings,
                point=specs[symbol].point,
                start=start,
                end=fetch_end,
                stride=args.stride,
            )
            for name, left, right in segments:
                selected = [order for order in orders if left <= order.decided_at < right]
                minute = frames[Timeframe.M1]
                segment_execution = minute[(minute.index >= left) & (minute.index < right)]
                result = PessimisticBacktester().run_non_overlapping(segment_execution, selected)
                lane_setups[name] += len(selected)
                lane_returns[name].extend(trade.net_r for trade in result.trades)
        if lane_selected and not lane_complete:
            for name, _, _ in segments:
                rows.append(summarise("6", "own_lane", name, lane_setups[name], lane_returns[name]))
            save_checkpoint(
                checkpoint,
                signature=signature,
                start=start,
                end=end,
                rows=rows,
            )
    finally:
        with contextlib.suppress(Exception):
            connector.shutdown()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / f"weekend-sections-{stamp}.txt"
    payload = output_dir / f"weekend-sections-{stamp}.json"
    rendered = render(rows, holdout=args.unlock_holdout) + render_diagnostics(diagnostics)
    report.write_text(rendered, encoding="utf-8")
    payload.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "start": start.isoformat(),
                "data_end": fetch_end.isoformat(),
                "holdout_unlocked": args.unlock_holdout,
                "symbols": list(frames_by_symbol),
                "rows": [asdict(row) for row in rows],
                "diagnostics": [asdict(row) for row in diagnostics],
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print("\n" + rendered)
    print(f"Report: {report}")
    print(f"Data:   {payload}")
    checkpoint.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

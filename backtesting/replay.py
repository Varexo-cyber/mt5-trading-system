"""Look-ahead-safe multi-timeframe context replay and evidence reporting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path

import pandas as pd

from analysis.confluence import ConfluenceEngine, TradeIdea
from backtesting.engine import (
    BacktestOrder,
    BacktestResult,
    PessimisticBacktester,
    deflated_sharpe_probability,
)
from core.broker import MarketDataProvider
from core.types import MarketContext, Series, Tick, Timeframe, TradingMode

REPLAY_TIMEFRAMES = (
    Timeframe.D1,
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M15,
    Timeframe.M5,
)


@dataclass(frozen=True, slots=True)
class SegmentEvidence:
    name: str
    start: str
    end: str
    decisions: int
    trades: int
    total_r: float
    expectancy_r: float
    win_rate: float
    profit_factor: float | None
    max_drawdown_r: float
    sharpe: float
    deflated_sharpe_probability: float
    module_trade_counts: dict[str, int]
    returns_by_module: dict[str, tuple[float, ...]]


class HistoricalContextReplay:
    """Recreate exactly what modules could know at each closed H1 bar."""

    def __init__(
        self,
        engine: ConfluenceEngine,
        *,
        history_bars: int = 300,
        decision_stride_bars: int = 1,
    ) -> None:
        if history_bars < 120:
            raise ValueError("history_bars must be at least 120")
        if decision_stride_bars < 1:
            raise ValueError("decision_stride_bars must be positive")
        self.engine = engine
        self.history_bars = history_bars
        self.decision_stride_bars = decision_stride_bars

    def ideas(
        self,
        symbol: str,
        frames: dict[Timeframe, pd.DataFrame],
        *,
        point: float,
        start: datetime,
        end: datetime,
    ) -> Iterator[tuple[datetime, float, TradeIdea]]:
        """Every verdict the engine reached, refusals included.

        `orders` throws the refusals away, and the refusals are where 97.5% of
        the live decisions go. Nothing could measure a change to an entry rule
        except the live account, one confounded hour at a time; the whole point
        of this is that a rule can be judged over months of history before a
        cent is put behind it.

        Refusals carry the score they reached, so "no module fired" and "the
        modules fired and the threshold was out of reach" stay distinguishable
        — which is exactly the question asked after a day with no trades.

        The look-ahead safety lives here and only here. A timeframe's bar is
        visible only once its own close time has passed, so a decision at 10:05
        sees the M15 that closed at 10:00 and not the one still forming. Get
        that wrong and every result downstream is a fiction.
        """
        missing = set(REPLAY_TIMEFRAMES) - set(frames)
        if missing:
            raise ValueError(f"replay missing timeframes: {sorted(tf.value for tf in missing)}")
        decisions = frames[Timeframe.H1]
        closed_at = decisions.index + Timeframe.H1.duration
        eligible = decisions[(closed_at >= start) & (closed_at < end)]
        # Close times per timeframe, computed once instead of per decision.
        #
        # The slice below used to be `frame[frame.index + duration <= decided_at]`,
        # which builds a boolean mask over the WHOLE frame and copies the
        # matching rows — every decision, every timeframe. Over 90 days the M5
        # frame is around 26,000 rows, so a five-symbol run was doing hundreds
        # of millions of row comparisons to find a cut point in a sorted index,
        # and the tool built to make measurement cheap took long enough that
        # nobody would run it twice.
        #
        # The index is sorted and never changes, so the cut point is a binary
        # search and the window is a view.
        close_times = {tf: frames[tf].index + tf.duration for tf in REPLAY_TIMEFRAMES}
        for sequence, opened_at in enumerate(eligible.index):
            if sequence % self.decision_stride_bars:
                continue
            decided_at = (opened_at + Timeframe.H1.duration).to_pydatetime()
            moment = pd.Timestamp(decided_at)
            series: dict[Timeframe, Series] = {}
            complete = True
            for timeframe in REPLAY_TIMEFRAMES:
                # `side="right"` counts the bars whose close is at or before the
                # decision, which is exactly what `<= decided_at` selected.
                cut = int(close_times[timeframe].searchsorted(moment, side="right"))
                if cut < 120:
                    complete = False
                    break
                available = frames[timeframe].iloc[max(0, cut - self.history_bars) : cut]
                if len(available) < 120:
                    complete = False
                    break
                series[timeframe] = Series(symbol, timeframe, available, decided_at)
            if not complete:
                continue
            executable = series[Timeframe.M5].df.iloc[-1]
            mid = float(executable["close"])
            spread_points = max(float(executable.get("spread", 0.0)), 0.0)
            spread = spread_points * point
            tick = Tick(
                symbol,
                decided_at,
                bid=mid - spread / 2,
                ask=mid + spread / 2,
            )
            yield decided_at, spread, self.engine.evaluate(
                MarketContext(symbol, decided_at, series, tick), TradingMode.BACKTEST
            )

    def orders(
        self,
        symbol: str,
        frames: dict[Timeframe, pd.DataFrame],
        *,
        point: float,
        start: datetime,
        end: datetime,
    ) -> list[BacktestOrder]:
        """The approved half of `ideas`, as executable orders."""
        orders: list[BacktestOrder] = []
        for decided_at, spread, idea in self.ideas(
            symbol, frames, point=point, start=start, end=end
        ):
            if not idea.approved or idea.direction is None:
                continue
            active = tuple(
                signal.module
                for signal in idea.signals
                if signal.score * int(idea.direction) > 0
                and signal.confidence > 0
                and self.engine.config.weights.get(signal.module, 0.0) > 0
                and signal.confidence >= self.engine.config.minimum_confidence
            )
            orders.append(
                BacktestOrder(
                    symbol=symbol,
                    decided_at=decided_at,
                    direction=idea.direction,
                    entry=idea.entry,
                    stop_loss=idea.stop_loss,
                    take_profit=idea.take_profit,
                    score=idea.score,
                    confidence=idea.confidence,
                    modules=active,
                    spread=spread,
                )
            )
        return orders


def frame_from_mt5(raw: object) -> pd.DataFrame:
    frame = pd.DataFrame(raw)
    if frame.empty or "time" not in frame:
        raise ValueError("MT5 returned no timestamped historical bars")
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    return frame.set_index("time").sort_index().loc[lambda value: ~value.index.duplicated()]


def fetch_mt5_history(
    market: MarketDataProvider,
    symbol: str,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
    *,
    max_bars_per_chunk: int = 40_000,
) -> pd.DataFrame:
    """Fetch long ranges in bounded chunks accepted by the MT5 terminal."""
    if end <= start:
        raise ValueError("history end must be after start")
    chunks: list[pd.DataFrame] = []
    cursor = start
    span = timeframe.duration * max_bars_per_chunk
    while cursor < end:
        chunk_end = min(cursor + span, end)
        raw = market.copy_rates_range(symbol, timeframe.mt5_value, cursor, chunk_end)
        frame = pd.DataFrame(raw)
        if not frame.empty:
            chunks.append(frame_from_mt5(raw))
        cursor = chunk_end
    if not chunks:
        raise ValueError(f"{symbol} {timeframe.value}: no historical bars returned")
    combined = pd.concat(chunks).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def archive_frame(frame: pd.DataFrame, path: Path) -> None:
    """Merge an MT5 slice into a durable, de-duplicated CSV archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = frame
    if path.exists():
        previous = pd.read_csv(path, index_col="time", parse_dates=["time"])
        previous.index = pd.DatetimeIndex(previous.index)
        combined = pd.concat([previous, frame])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_csv(path, index_label="time")


def frame_digest(frame: pd.DataFrame) -> str:
    """Stable identity of the exact bars used by an evidence report."""
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    columns = "|".join(str(column) for column in frame.columns).encode()
    return hashlib.sha256(columns + hashed).hexdigest()


def evidence_digest(payload: dict[str, object]) -> str:
    """Content address an evidence report, excluding its own digest field."""
    body = {key: value for key, value in payload.items() if key != "evidence_digest"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def implementation_digest(root: Path) -> str:
    """Identify the executable research implementation behind a report.

    Data and configuration hashes are insufficient when strategy or replay
    code changes. This covers production Python and schema files while
    excluding tests, runtime state and generated evidence.
    """
    locations = (
        "advisory",
        "analysis",
        "backtesting",
        "config",
        "core",
        "execution",
        "filters",
        "learning",
        "promotion",
        "risk",
        "runner",
    )
    files = sorted(
        path for location in locations for path in (root / location).rglob("*.py") if path.is_file()
    )
    validator = root / "scripts" / "validate_strategy.py"
    if validator.is_file():
        files.append(validator)
        files.sort()
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def build_segment_evidence(
    name: str,
    execution_frame: pd.DataFrame,
    orders: list[BacktestOrder],
    *,
    start: datetime,
    end: datetime,
    configurations_tested: int,
) -> tuple[SegmentEvidence, BacktestResult]:
    selected = [order for order in orders if start <= order.decided_at < end]
    result = PessimisticBacktester().run_non_overlapping(execution_frame, selected)
    returns = [trade.net_r for trade in result.trades]
    counts: dict[str, int] = {}
    module_returns: dict[str, list[float]] = {}
    for trade in result.trades:
        for module in trade.order.modules:
            counts[module] = counts.get(module, 0) + 1
            module_returns.setdefault(module, []).append(trade.net_r)
    evidence = SegmentEvidence(
        name=name,
        start=start.isoformat(),
        end=end.isoformat(),
        decisions=len(selected),
        trades=result.sample_size,
        total_r=result.total_r,
        expectancy_r=result.expectancy_r,
        win_rate=result.win_rate,
        profit_factor=result.profit_factor if isfinite(result.profit_factor) else None,
        max_drawdown_r=result.max_drawdown_r,
        sharpe=result.sharpe,
        deflated_sharpe_probability=deflated_sharpe_probability(returns, configurations_tested),
        module_trade_counts=counts,
        returns_by_module={key: tuple(values) for key, values in module_returns.items()},
    )
    return evidence, result


def write_evidence_report(
    path: Path,
    *,
    metadata: dict[str, object],
    segments: list[SegmentEvidence],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "metadata": metadata,
        "segments": [asdict(segment) for segment in segments],
    }
    payload["evidence_digest"] = evidence_digest(payload)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

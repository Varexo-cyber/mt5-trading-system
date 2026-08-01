"""Plot exactly what the market-structure module can see.

Use a CSV exported from MT5 with UTC time, open, high, low, and close columns.
Without --csv the script renders a deterministic synthetic BOS example. The
demo proves the annotations work; it is not evidence that the hypothesis has
an edge.
"""

from __future__ import annotations

import argparse
from datetime import UTC
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from analysis.market_structure import MarketStructure, StructureSnapshot
from config.loader import load_settings
from core.types import MarketContext, Series, Timeframe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="UTC OHLC CSV; omit for synthetic demo")
    parser.add_argument("--output", type=Path, required=True, help="PNG destination")
    parser.add_argument("--timeframe", default="H1", help="timeframe represented by the CSV")
    args = parser.parse_args(argv)

    timeframe = Timeframe.parse(args.timeframe)
    frame = load_frame(args.csv, timeframe)
    now = frame.index[-1].to_pydatetime() + timeframe.duration
    context = MarketContext(
        symbol="EURUSD",
        now=now,
        series={timeframe: Series("EURUSD", timeframe, frame, now)},
    )
    module = MarketStructure(load_settings(env_overrides=False).analysis.market_structure)
    snapshot = module.inspect(context, timeframe)
    if snapshot is None:
        parser.error("not enough rows for the configured ATR and swing lookbacks")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(frame, snapshot, args.output)
    print(f"market-structure chart written to {args.output}")
    return 0


def load_frame(path: Path | None, timeframe: Timeframe) -> pd.DataFrame:
    if path is None:
        return demo_frame(timeframe)
    frame = pd.read_csv(path)
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame.set_index("time").sort_index()


def demo_frame(timeframe: Timeframe) -> pd.DataFrame:
    values = [
        1.00,
        1.20,
        1.40,
        1.60,
        1.40,
        1.20,
        1.10,
        1.40,
        1.70,
        2.00,
        1.80,
        1.60,
        1.50,
        1.80,
        2.10,
        2.40,
        2.20,
        2.00,
        1.90,
        2.20,
        2.50,
        2.80,
        2.60,
        2.40,
        2.30,
        2.50,
        2.60,
        2.70,
        2.75,
        3.00,
    ]
    index = pd.date_range("2026-01-05", periods=len(values), freq=timeframe.duration, tz=UTC)
    return pd.DataFrame(
        {
            "open": [value - 0.01 for value in values],
            "high": [value + 0.05 for value in values],
            "low": [value - 0.05 for value in values],
            "close": values,
        },
        index=index,
    )


def plot(frame: pd.DataFrame, snapshot: StructureSnapshot, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 7))
    _candles(axis, frame)

    for swing in snapshot.internal_swings:
        marker = "^" if swing.kind == "low" else "v"
        axis.scatter(swing.index, swing.price, marker=marker, s=28, color="0.55", zorder=3)
    for swing in snapshot.external_swings:
        marker = "^" if swing.kind == "low" else "v"
        color = "tab:green" if swing.kind == "low" else "tab:red"
        axis.scatter(swing.index, swing.price, marker=marker, s=80, color=color, zorder=4)
        axis.annotate(
            f"{swing.kind} (known @{swing.confirmed_index})",
            (swing.index, swing.price),
            xytext=(0, 12 if swing.kind == "high" else -18),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    for level in snapshot.equal_levels:
        axis.axhline(level.price, color="tab:purple", linestyle=":", linewidth=1)

    if snapshot.event is not None:
        event = snapshot.event
        axis.axhline(event.level, color="tab:blue", linestyle="--", linewidth=1.5)
        axis.annotate(
            f"{event.kind} {event.direction}",
            (len(frame) - 1, float(frame["close"].iloc[-1])),
            xytext=(-80, 24),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "tab:blue"},
            color="tab:blue",
        )
    if snapshot.invalidation_price is not None:
        axis.axhline(
            snapshot.invalidation_price,
            color="tab:orange",
            linestyle="-.",
            linewidth=1.5,
            label="structural invalidation",
        )

    axis.set_title(
        f"{snapshot.timeframe.value} market structure — "
        f"external {snapshot.external_direction}, internal {snapshot.internal_direction}"
    )
    axis.set_xlabel("closed bars")
    axis.set_ylabel("price")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _candles(axis: Axes, frame: pd.DataFrame) -> None:
    for position, (_, row) in enumerate(frame.iterrows()):
        bullish = float(row["close"]) >= float(row["open"])
        color = "tab:green" if bullish else "tab:red"
        axis.vlines(position, float(row["low"]), float(row["high"]), color=color, linewidth=1)
        bottom = min(float(row["open"]), float(row["close"]))
        height = max(abs(float(row["close"]) - float(row["open"])), 0.002)
        axis.add_patch(
            Rectangle(
                (position - 0.3, bottom),
                0.6,
                height,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())

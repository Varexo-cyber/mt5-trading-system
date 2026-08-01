"""Generate a compact PDF account and chart report."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

if TYPE_CHECKING:
    import pandas as pd

    from core.instrument import InstrumentSpec
    from core.types import AccountSnapshot, Position, Tick


def build_pdf_report(
    account: AccountSnapshot,
    positions: list[Position],
    symbol: str,
    spec: InstrumentSpec,
    tick: Tick | None,
    frames: dict[str, pd.DataFrame],
) -> bytes:
    """Return a PDF report as bytes; nothing is written to disk."""
    output = BytesIO()
    with PdfPages(output) as pdf:
        _summary_page(pdf, account, positions, symbol, spec, tick)
        for timeframe, frame in frames.items():
            if not frame.empty:
                _chart_page(pdf, symbol, timeframe, frame)
    return output.getvalue()


def _summary_page(
    pdf: PdfPages,
    account: AccountSnapshot,
    positions: list[Position],
    symbol: str,
    spec: InstrumentSpec,
    tick: Tick | None,
) -> None:
    fig = Figure(figsize=(8.27, 11.69))
    fig.suptitle("MT5 Trading System — Operator Report", fontsize=18, fontweight="bold")
    lines = [
        f"Generated: {account.taken_at:%Y-%m-%d %H:%M:%S} UTC",
        f"Broker: {account.server} ({'DEMO' if account.is_demo else 'LIVE'})",
        f"Balance / equity: {account.balance:.2f} / {account.equity:.2f} {account.currency}",
        f"Free margin: {account.margin_free:.2f} {account.currency} | "
        f"leverage 1:{account.leverage}",
        f"Open positions: {len(positions)}",
        "",
        f"Selected: {symbol} — {spec.description or spec.asset_class.value}",
        f"Asset class: {spec.asset_class.value}",
    ]
    if tick is None:
        lines.append("Quote: unavailable/stale — entry must remain blocked")
    else:
        spread_bps = tick.spread / tick.mid * 10_000
        cost = spec.money_per_lot(tick.spread) * spec.volume_min
        tick_age = max(0.0, (account.taken_at - tick.time).total_seconds())
        lines.extend(
            [
                f"Bid / ask: {tick.bid:g} / {tick.ask:g}",
                f"Spread: {spec.price_to_pips(tick.spread):.2f} pips / {spread_bps:.3f} bps",
                f"Spread cost at {spec.volume_min:g} lot: {cost:.4f} {account.currency}",
                f"Quote age: {tick_age:.1f} seconds",
            ]
        )
    fig.text(0.08, 0.88, "\n".join(lines), va="top", family="monospace", fontsize=11)

    if positions:
        columns = ["Symbol", "Side", "Lots", "Open", "SL", "TP", "P/L"]
        rows = [
            [
                p.symbol,
                p.direction.name,
                f"{p.volume:g}",
                f"{p.price_open:g}",
                f"{p.sl:g}",
                f"{p.tp:g}",
                f"{p.profit + p.swap:.2f}",
            ]
            for p in positions
        ]
        axis = fig.add_axes((0.06, 0.10, 0.88, 0.35))
        axis.axis("off")
        axis.table(cellText=rows, colLabels=columns, loc="upper center", cellLoc="center")
    pdf.savefig(fig, bbox_inches="tight")
    fig.clear()


def _chart_page(pdf: PdfPages, symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
    fig = Figure(figsize=(11.69, 8.27))
    axis = fig.subplots()
    axis.plot(frame.index, frame["close"], color="#2dd4bf", linewidth=1.2)
    axis.fill_between(frame.index, frame["low"], frame["high"], color="#2dd4bf", alpha=0.08)
    axis.set_title(f"{symbol} — {timeframe} closed bars")
    axis.set_ylabel("Price")
    axis.grid(alpha=0.2)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.autofmt_xdate()
    pdf.savefig(fig, bbox_inches="tight")
    fig.clear()

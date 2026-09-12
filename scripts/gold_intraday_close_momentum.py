"""Pre-geregistreerde XAUUSD-replicatie van JFE intraday close momentum."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtesting.replay import fetch_mt5_history
from config.loader import load_credentials, load_settings, terminal_path_from_env
from core.mt5_connector import MT5Connector
from core.types import Timeframe
from scripts.dumb_session_baseline import _atr, _drawdown

NEW_YORK = ZoneInfo("America/New_York")


def _at_or_after(frame: pd.DataFrame, stamp: pd.Timestamp) -> pd.Series | None:
    pos = frame.index.searchsorted(stamp, side="left")
    return None if pos >= len(frame) else frame.iloc[pos]


def trades(frame: pd.DataFrame, *, stop_atr: float, cost_r: float) -> pd.DataFrame:
    """One fixed 13:00--13:30 New York trade per complete business day."""

    atr = _atr(frame)
    first_day = frame.index[0].tz_convert(NEW_YORK).date() + timedelta(days=1)
    last_day = frame.index[-1].tz_convert(NEW_YORK).date()
    rows: list[dict[str, object]] = []
    for day in pd.date_range(first_day, last_day, freq="D"):
        local_day = day.date()
        if local_day.weekday() >= 5:
            continue
        previous = local_day - timedelta(days=1)
        while previous.weekday() >= 5:
            previous -= timedelta(days=1)
        previous_close = pd.Timestamp(
            datetime.combine(previous, time(13, 30), NEW_YORK)
        ).tz_convert(UTC)
        entry_at = pd.Timestamp(datetime.combine(local_day, time(13, 0), NEW_YORK)).tz_convert(UTC)
        exit_at = pd.Timestamp(datetime.combine(local_day, time(13, 30), NEW_YORK)).tz_convert(UTC)
        prev = _at_or_after(frame, previous_close)
        entry = _at_or_after(frame, entry_at)
        exit_bar = _at_or_after(frame, exit_at)
        if prev is None or entry is None or exit_bar is None:
            continue
        # Reject holidays/data gaps instead of silently using a later bar.
        prev_pos = frame.index.searchsorted(previous_close, side="left")
        entry_pos = frame.index.searchsorted(entry_at, side="left")
        exit_pos = frame.index.searchsorted(exit_at, side="left")
        if any(
            abs((frame.index[pos] - target).total_seconds()) > 300
            for pos, target in (
                (prev_pos, previous_close),
                (entry_pos, entry_at),
                (exit_pos, exit_at),
            )
        ):
            continue
        direction = float(np.sign(float(entry["open"]) - float(prev["open"])))
        risk = float(atr.iloc[entry_pos]) * stop_atr
        if direction == 0.0 or not np.isfinite(risk) or risk <= 0.0:
            continue
        gross = direction * (float(exit_bar["open"]) - float(entry["open"])) / risk
        always_long = (float(exit_bar["open"]) - float(entry["open"])) / risk - cost_r
        rows.append(
            {
                "day": local_day,
                "year": local_day.year,
                "direction": "LONG" if direction > 0 else "SHORT",
                "gross_r": gross,
                "net_r": gross - cost_r,
                "always_long_r": always_long,
            }
        )
    return pd.DataFrame(rows)


def _line(label: str, values: pd.Series) -> None:
    n = len(values)
    mean = float(values.mean()) if n else 0.0
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    t_value = mean / (sd / np.sqrt(n)) if sd else 0.0
    print(
        f"  {label:<18} n={n:>4}  win={(values > 0).mean():>6.1%}  "
        f"totaal={values.sum():>+8.2f}R  gem={mean:>+7.3f}R  "
        f"t={t_value:>5.2f}  DD={_drawdown(values):>7.2f}R"
    )


def run(args: argparse.Namespace) -> None:
    settings = load_settings(overlay=args.config, env_overrides=False)
    symbol = settings.instruments.broker_symbol("XAUUSD")
    end = args.end_date or datetime.now(UTC)
    start = args.start_date or end - timedelta(days=args.days)
    credentials = load_credentials(required=True)
    connector = MT5Connector(
        settings.mt5,
        credentials,
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    connector.connect()
    try:
        frame = fetch_mt5_history(connector, symbol, Timeframe.M5, start - timedelta(days=5), end)
    finally:
        connector.shutdown()
    frame = frame[(frame.index >= start - timedelta(days=5)) & (frame.index <= end)]
    result = trades(frame, stop_atr=args.stop_atr, cost_r=args.cost_r)
    print("\nGOLD INTRADAY CLOSE MOMENTUM -- SHADOW, NIETS LIVE")
    print("Vaste paperregel: 13:00 New York richting dagsignaal, exit 13:30.")
    print(f"Kosten {args.cost_r:.3f}R/trade; R = {args.stop_atr:.2f} x M5 ATR(14).")
    if result.empty:
        print("Geen complete handelsdagen gevonden.")
        return
    _line("ALLE JAREN", result["net_r"])
    _line("ALTIJD LONG", result["always_long_r"])
    for year, group in result.groupby("year"):
        _line(str(year), group["net_r"])
    verdict = (
        result["net_r"].mean() > 0
        and all(group["net_r"].mean() > 0 for _, group in result.groupby("year"))
        and len(result.groupby("year")) >= 2
    )
    print("\nOORDEEL:", "KANDIDAAT VOOR FORWARD SHADOW" if verdict else "VERWORPEN")
    print("Ook een kandidaat blijft uit live totdat een nieuwe forwardperiode slaagt.\n")


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--days", type=int, default=730)
    out.add_argument("--start-date", type=datetime.fromisoformat)
    out.add_argument("--end-date", type=datetime.fromisoformat)
    out.add_argument("--cost-r", type=float, default=0.066)
    out.add_argument("--stop-atr", type=float, default=0.8)
    out.add_argument("--config", default="config/eightcap.yaml")
    return out


def main() -> None:
    args = parser().parse_args()
    for field in ("start_date", "end_date"):
        value = getattr(args, field)
        if value is not None and value.tzinfo is None:
            setattr(args, field, value.replace(tzinfo=UTC))
    run(args)


if __name__ == "__main__":
    main()

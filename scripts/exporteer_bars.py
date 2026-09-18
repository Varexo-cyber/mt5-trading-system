"""Bars uit MT5 naar een CSV, zodat een meting niet aan een machine vastzit.

WAAROM DIT BESTAAT. MetaTrader5 is Windows-only. Elke studie in dit project
haalt zijn bars rechtstreeks uit de terminal, en daarmee kan alleen de eigenaar
op zijn eigen PC ooit iets meten. Elke vraag "draai dit even" wordt dan een
wachtmoment.

Dit script haalt de bars EEN KEER op en schrijft ze weg. Daarna kan iedereen
met dat bestand dezelfde meting draaien, op elke machine.

EN HET IS HERHAALBAAR, wat minstens zo belangrijk is. Twee runs rechtstreeks
tegen de terminal meten twee verschillende reeksen, want er is een bar
bijgekomen en het venster is verschoven. Twee runs tegen hetzelfde bestand
meten dezelfde bars. Dat is het verschil tussen "ik kreeg een ander getal" en
"de code is veranderd".

HET FORMAAT IS NIET NIEUW. `backtesting.replay.archive_frame` schreef deze CSV
al: index `time` in UTC, kolommen open/high/low/close/tick_volume. Dit script
zet er alleen een aanroep omheen die je kunt draaien.

HET BESTAND IS GEEN GEHEIM, maar het is wel groot: 180 dagen M1 is ongeveer
190.000 regels en een dikke 10 MB. Zip het voor je het verstuurt.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd


def samenvatting(frame: pd.DataFrame) -> str:
    """Wat er in het bestand zit, zodat je het kunt controleren voor je het stuurt.

    DE GATEN HOREN HIER TE STAAN. Een reeks M1-bars met een weekend erin is
    normaal; een reeks met een gat van drie dagen midden in de week betekent
    dat de terminal zijn historie niet compleet had, en dan meet je een markt
    die er niet was.
    """

    verschillen = frame.index.to_series().diff().dropna()
    gaten = verschillen[verschillen > pd.Timedelta(minutes=5)]
    lang = gaten[gaten > pd.Timedelta(hours=12)]
    return "\n".join([
        f"  bars        {len(frame):,}",
        f"  van         {frame.index[0]}",
        f"  tot         {frame.index[-1]}",
        f"  kolommen    {', '.join(frame.columns)}",
        f"  gaten >5m   {len(gaten)}  (weekenden en feestdagen horen erbij)",
        f"  gaten >12u  {len(lang)}",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--uit", default="runtime/xauusd_m1.csv")
    parser.add_argument("--config", default="config/eightcap.yaml")
    args = parser.parse_args()

    from backtesting.replay import archive_frame, fetch_mt5_history
    from config.loader import load_credentials, load_settings, terminal_path_from_env
    from core.mt5_connector import MT5Connector
    from core.types import Timeframe

    settings = load_settings(overlay=args.config, env_overrides=False)
    # VIA broker_symbol, want de overlay draagt de suffix. "XAUUSD" vragen aan
    # een broker die het "XAUUSD.i" noemt levert niets op, en dat faalt stil.
    symbool = settings.instruments.broker_symbol(args.symbol)
    eind = datetime.now(UTC)
    start = eind - timedelta(days=args.days)

    print(f"\n  {symbool}  M1  {args.days} dagen  ->  {args.uit}")
    connector = MT5Connector(
        settings.mt5,
        load_credentials(required=True),
        terminal_path=settings.mt5.terminal_path or terminal_path_from_env(),
    )
    connector.connect()
    try:
        frame = fetch_mt5_history(connector, symbool, Timeframe.M1, start, eind)
    finally:
        connector.shutdown()

    pad = Path(args.uit)
    archive_frame(frame, pad)
    print()
    print(samenvatting(frame))
    print(f"\n  geschreven: {pad.resolve()}  ({pad.stat().st_size / 1e6:.1f} MB)")
    print("\n  Draai hiermee:")
    print(f"    python -m scripts.section_twenty_pullback_ladder --csv {args.uit} --balans 400")
    print(f"    python -m scripts.section_twentyone_straddle --csv {args.uit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""What a stop-out ACTUALLY costs, from this account's own fills.

    python scripts/measure_slippage.py
    python scripts/measure_slippage.py --db journal/trading.db

WHY THIS DECIDES WHETHER FX CAN TRADE AT ALL.

`PositionSizer._cost_share` divides the round trip by the stop distance, and on
EURUSD the round trip is currently priced like this, per lot:

    commission   $ 5.50   (round trip, 2 x 2.75)
    slippage     $17.00   <- stop_slippage_pips.forex = 1.7
    spread       $ 2.00
                 -------
                 $24.50

Slippage is two thirds of it, and `stop_slippage_pips` is not a measurement.
Its own docstring names the single fill it came from: "a stop at 1.19722 filled
at 1.19705". One observation, possibly during news, applied to every trade on
every pair forever.

What it decides, against `max_cost_share_of_risk = 0.12`:

    clock   stop     at 1.7 pips   at 0.3 pips
    M15     3.5 pip      70%           30%
    M30     5   pip      49%           21%
    H1      8   pip      31%           13%
    H4      20  pip      12%            5%

So the difference between "FX never clears the wall" and "FX clears it on H1"
is one number that nobody has ever checked. This checks it.

HOW THE STOP-OUT IS IDENTIFIED. Geometrically, not by label: a closed trade
whose exit price is at or past its own stop, on the losing side. `exit_reason`
strings vary by close path and a filter built on them would quietly miss whole
categories.

HOW PIPS ARE DERIVED. From the trade's own row: `sl_distance_pips` is recorded
next to `entry_price` and `sl`, so one pip is `|entry - sl| / sl_distance_pips`
for that symbol. No pip table, no broker lookup, and it cannot disagree with
the sizer's own arithmetic because it comes from what the sizer wrote down.
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Symbol name -> asset class, matching the keys in `stop_slippage_pips`.
#:
#: By NAME rather than by broker path, because this runs against a journal and
#: not against a live terminal. A symbol that matches nothing is reported under
#: "unclassified" rather than being silently dropped into forex -- guessing
#: here would put index fills into the number that decides FX.
_METALS = ("XAU", "XAG", "XPT", "XPD", "GOLD", "SILVER")
_INDICES = (
    "US30",
    "US500",
    "NAS",
    "NDX",
    "SPX",
    "GER",
    "DAX",
    "UK100",
    "FRA",
    "EUSTX",
    "JPN",
    "AUS200",
    "HK50",
    "ASX",
    "STOXX",
    "NTH",
    "SUI",
    "ESP",
    "ITA",
)
_CRYPTO = ("BTC", "ETH", "LTC", "XRP", "SOL", "ADA", "DOGE")
_FX_CODES = (
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CHF",
    "AUD",
    "NZD",
    "CAD",
    "SEK",
    "NOK",
    "DKK",
    "PLN",
    "HUF",
    "CZK",
    "TRY",
    "ZAR",
    "MXN",
    "SGD",
    "HKD",
)


def asset_class(symbol: str) -> str:
    name = symbol.upper()
    for prefix in _CRYPTO:
        if name.startswith(prefix):
            return "crypto"
    for token in _INDICES:
        if token in name:
            return "index"
    for token in _METALS:
        if token in name:
            return "metal"
    # A six-character pair of two known currency codes, before any broker
    # suffix. `EURUSD.i`, `GBPJPY.r`, `USDCHF` all land here; `USDX` does not,
    # because "X" is not a currency and that symbol is a dollar index.
    head = "".join(ch for ch in name if ch.isalpha())[:6]
    if len(head) == 6 and head[:3] in _FX_CODES and head[3:] in _FX_CODES:
        return "forex"
    return "unclassified"


def stop_outs(conn: sqlite3.Connection) -> list[dict]:
    """Every closed trade that finished at or through its own stop.

    THE BAND ON THE FAVOURABLE SIDE IS THE SUBTLE PART. A stop can fill BETTER
    than its price -- price gaps, the broker fills at the next available quote,
    and on a quiet pair that sometimes lands in your favour. Those are real
    stop-outs and they belong in the average. Requiring `exit <= sl` exactly
    would drop every one of them, and dropping only the favourable tail biases
    the mean UPWARD -- which is the direction that keeps FX refused, i.e. it
    would quietly confirm the very number this script exists to check.

    So the band is a quarter of the stop distance on the favourable side, and
    unbounded on the adverse side. A fill half a pip inside a ten-pip stop is a
    stop-out; a manual close five pips inside it is not, and counting that as
    "-5 pips of slippage" would be far worse than dropping it.
    """
    rows = conn.execute(
        "SELECT symbol, direction, entry_price, sl, exit_price, sl_distance_pips, "
        "       exit_reason, pnl_r, opened_at "
        "FROM trades "
        "WHERE closed_at IS NOT NULL AND exit_price IS NOT NULL "
        "  AND sl_distance_pips > 0 AND sl > 0"
    ).fetchall()

    found: list[dict] = []
    for row in rows:
        pip = abs(float(row["entry_price"]) - float(row["sl"])) / float(row["sl_distance_pips"])
        if pip <= 0:
            continue
        stop, exit_price = float(row["sl"]), float(row["exit_price"])
        band = max(0.1 * pip, 0.25 * abs(float(row["entry_price"]) - stop))
        if str(row["direction"]).upper().startswith("L"):
            if exit_price > stop + band:
                continue
            adverse = stop - exit_price
        else:
            if exit_price < stop - band:
                continue
            adverse = exit_price - stop
        found.append(
            {
                "symbol": str(row["symbol"]),
                "asset_class": asset_class(str(row["symbol"])),
                "slippage_pips": adverse / pip,
                "sl_distance_pips": float(row["sl_distance_pips"]),
                "opened_at": str(row["opened_at"]),
            }
        )
    return found


def _percentile(values: list[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(share * (len(ordered) - 1)))
    return ordered[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="journal/trading.db", help="journal path")
    parser.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help="below this many stop-outs an asset class is reported but not judged",
    )
    args = parser.parse_args(argv)

    path = Path(args.db)
    if not path.exists():
        raise SystemExit(f"no such journal: {path}")

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        try:
            found = stop_outs(conn)
        except sqlite3.OperationalError as exc:
            raise SystemExit(f"cannot read {path}: {exc}") from exc

    from config.loader import load_settings

    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
    configured = dict(settings.risk.stop_slippage_pips)
    limit = settings.risk.max_cost_share_of_risk

    print(f"\n{'=' * 78}")
    print("WHAT A STOP-OUT ACTUALLY COST — measured from this account's fills")
    print(f"{'=' * 78}")
    if not found:
        print("\n  No stop-outs in the journal yet.")
        print("  Until there are, `stop_slippage_pips` stays an assumption and every")
        print("  FX refusal on cost rests on it. Nothing to conclude — come back after")
        print("  the account has taken some losses.")
        return 1

    by_class: dict[str, list[float]] = {}
    for row in found:
        by_class.setdefault(row["asset_class"], []).append(row["slippage_pips"])

    print(f"\n  {len(found)} stop-outs across {len(by_class)} asset classes\n")
    print(f"  {'class':<14}{'n':>5}{'median':>9}{'mean':>9}{'p90':>9}{'worst':>9}   configured")
    for name in sorted(by_class):
        pips = by_class[name]
        used = configured.get(name)
        shown = f"{used:.2f}" if used is not None else "-- (none set)"
        thin = "   thin" if len(pips) < args.min_trades else ""
        print(
            f"  {name:<14}{len(pips):>5}{statistics.median(pips):>9.2f}"
            f"{statistics.fmean(pips):>9.2f}{_percentile(pips, 0.9):>9.2f}"
            f"{max(pips):>9.2f}   {shown}{thin}"
        )

    print(
        "\n  NEGATIVE MEDIAN IS NORMAL AND IT IS NOT A BUG. A stop can fill BETTER\n"
        "  than its price -- price gaps through, the broker fills at the next\n"
        "  available quote, and on a quiet pair that is sometimes in your favour.\n"
        "  What matters for the cost model is the average, not the worst case."
    )

    # WHAT IT WOULD CHANGE, in the only terms that matter: which clocks open.
    print(f"\n{'=' * 78}")
    print(f"WHAT THIS DOES TO THE COST WALL   (limit {limit:.0%} of the stop)")
    print(f"{'=' * 78}")
    print("  Cost share = (commission + slippage + spread) / stop distance, per lot.")
    print("  Stop widths below are one ATR of each clock on a typical major.\n")

    for name in sorted(by_class):
        pips = by_class[name]
        if len(pips) < args.min_trades:
            continue
        measured = max(0.0, statistics.fmean(pips))
        assumed = configured.get(name, 0.0)
        commission = settings.risk.commission_per_lot(name)
        # Per-lot money for one pip, taken from the trades themselves is not
        # possible here (no spec), so the comparison is done in PIPS, which is
        # the unit both numbers already share. Commission is converted using
        # the account's own pip value for a major; stated, not hidden.
        pip_value = 10.0 if name == "forex" else 1.0
        commission_pips = commission / pip_value
        spread_pips = 0.2 if name == "forex" else 2.0
        print(
            f"  {name}   commission {commission_pips:.2f} pip"
            f"   spread ~{spread_pips:.2f} pip"
            f"   slippage {assumed:.2f} -> {measured:.2f} pip"
        )
        print(f"    {'stop':>8}{'now':>10}{'measured':>11}")
        for stop_pips in (3.5, 5.0, 8.0, 20.0):
            now = (commission_pips + assumed + spread_pips) / stop_pips
            then = (commission_pips + measured + spread_pips) / stop_pips
            mark = "  <- opens up" if now > limit >= then else ""
            print(f"    {stop_pips:>8.1f}{now:>9.0%}{then:>11.0%}{mark}")
        print()

    print("  A row that opens up is a clock this account could trade and currently")
    print("  refuses. It is NOT a promise that the strategy makes money there --")
    print("  the edge still has to clear what is left. Re-measure with sweep.cmd")
    print("  before changing anything live.")
    print(f"{'=' * 78}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

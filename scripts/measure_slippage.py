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

A MOVED STOP STILL COUNTS. Break-even management pulls the stop to entry on
most trades here, and an earlier version discarded every such row -- 222 of
349, leaving five. That was wrong. The question is how far past the stop the
fill landed, which is about the broker's execution and not about where the stop
happened to sit; a break-even stop is an ordinary stop at a different price.

HOW PIPS ARE DERIVED. From the symbol's name and its price: the fourth decimal
on FX, the second on JPY crosses, 0.1 on gold, 0.001 on silver, whole points on
indices and crypto. That matches `InstrumentSpec`, where a forex pip is ten
points and everything else is one, without needing a live terminal.

It deliberately does NOT come from the trade row. `|entry - sl| /
sl_distance_pips` looks exact -- both are written by the sizer -- and is wrong
whenever the stop has moved, because `sl` is the current stop and
`sl_distance_pips` is the one recorded at open. That version reported a forex
median of 8,563 pips.
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


def pip_size(symbol: str, price: float) -> float:
    """One pip for this symbol, from its name and its price magnitude.

    THE FIRST VERSION DERIVED THIS FROM THE TRADE ROW and it produced medians
    of eight thousand pips. `pip = |entry - sl| / sl_distance_pips` looks exact
    -- both numbers come from the sizer, so they cannot disagree -- and it is
    wrong for one reason: `trades.sl` is the CURRENT stop, not the one the
    trade opened with. Break-even management moves it to entry. `|entry - sl|`
    then collapses toward zero, the derived pip with it, and every slippage
    divided by that pip explodes.

    A moved stop is the normal case on this account, so the derivation was
    broken for most rows rather than a few.

    The convention below is the broker's own: FX quotes a pip at the fourth
    decimal, JPY crosses at the second, gold at 0.1, silver at 0.001, and
    indices and crypto quote whole points. It matches `InstrumentSpec`, where
    a forex pip is ten points and everything else is one -- but it needs no
    live terminal, which is the point of reading a journal.
    """
    name = symbol.upper()
    family = asset_class(symbol)
    if family == "metal":
        if "XAG" in name or "SILVER" in name:
            return 0.001
        return 0.1
    if family in ("index", "crypto"):
        return 1.0
    if family == "forex":
        return 0.01 if "JPY" in name else 0.0001
    # Unclassified: fall back on the magnitude of the price itself. A quote
    # near 1 is priced to four decimals; one in the thousands to whole points.
    if price >= 500:
        return 1.0
    if price >= 20:
        return 0.01
    return 0.0001


def stop_outs(conn: sqlite3.Connection) -> tuple[list[dict], dict[str, int]]:
    """Stop-outs with their slippage, and a count of what was discarded.

    Returns `(rows, discarded)`. The second value is not decoration: the first
    version of this script silently kept rows it could not read and reported a
    median of eight thousand pips as though it were a measurement.

    THE BAND ON THE FAVOURABLE SIDE. A stop can fill BETTER than its price --
    price gaps, the broker fills at the next available quote, and on a quiet
    pair that sometimes lands in your favour. Those are real stop-outs and they
    belong in the average. Requiring `exit <= sl` exactly would drop every one
    of them, and dropping only the favourable tail biases the mean UPWARD --
    the direction that keeps FX refused, i.e. it would quietly confirm the very
    number this script exists to check.

    So the band is a quarter of the stop distance on the favourable side and
    unbounded on the adverse one. A fill half a pip inside a ten-pip stop is a
    stop-out; a manual close five pips inside it is not.
    """
    rows = conn.execute(
        "SELECT symbol, direction, entry_price, sl, exit_price, sl_distance_pips, "
        "       exit_reason, pnl_r, opened_at "
        "FROM trades "
        "WHERE closed_at IS NOT NULL AND exit_price IS NOT NULL "
        "  AND sl_distance_pips > 0 AND sl > 0"
    ).fetchall()

    found: list[dict] = []
    discarded: dict[str, int] = {}

    def drop(why: str) -> None:
        discarded[why] = discarded.get(why, 0) + 1

    for row in rows:
        symbol = str(row["symbol"])
        entry, stop, exit_price = (
            float(row["entry_price"]),
            float(row["sl"]),
            float(row["exit_price"]),
        )
        pip = pip_size(symbol, entry)
        recorded = float(row["sl_distance_pips"])
        active = abs(entry - stop) / pip

        # A MOVED STOP IS STILL A STOP, and discarding those threw away 222 of
        # 349 rows on the first real run -- leaving five.
        #
        # The discard was inherited from the version that derived one pip from
        # `|entry - sl| / sl_distance_pips`. THAT needed the two to agree.
        # Nothing does any more: `pip_size` comes from the symbol. And the
        # question here is "how far past the stop did the fill land", which is
        # about the broker's execution, not about where the stop happened to
        # sit. A break-even stop is an ordinary stop at a different price.
        #
        # So only a stop at zero distance is unusable -- there is no order to
        # slip against.
        if active <= 0:
            drop("stop sits exactly at entry; nothing to measure against")
            continue

        band = max(0.1 * pip, 0.25 * abs(entry - stop))
        if str(row["direction"]).upper().startswith("L"):
            if exit_price > stop + band:
                drop("closed before reaching its stop")
                continue
            adverse = stop - exit_price
        else:
            if exit_price < stop - band:
                drop("closed before reaching its stop")
                continue
            adverse = exit_price - stop

        slippage = adverse / pip
        # A fill further from the stop than the ORIGINAL stop was wide is not a
        # slipped fill, it is a broken row -- a spec change, a rename, a symbol
        # this script classified wrongly. Measured against `recorded` rather
        # than against the active stop on purpose: a break-even stop sits a
        # tenth of an ATR from entry, and bounding slippage by THAT would throw
        # away every genuinely slipped fill on a managed trade.
        if abs(slippage) > recorded:
            drop("fill further out than the whole original stop; row not trusted")
            continue

        found.append(
            {
                "symbol": symbol,
                "asset_class": asset_class(symbol),
                "slippage_pips": slippage,
                "sl_distance_pips": recorded,
                "opened_at": str(row["opened_at"]),
            }
        )
    return found, discarded


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
            found, discarded = stop_outs(conn)
        except sqlite3.OperationalError as exc:
            raise SystemExit(f"cannot read {path}: {exc}") from exc

    from config.loader import load_settings

    settings = load_settings(overlay=ROOT / "config" / "eightcap.yaml", env_overrides=False)
    configured = dict(settings.risk.stop_slippage_pips)
    limit = settings.risk.max_cost_share_of_risk

    print(f"\n{'=' * 78}")
    print("WHAT A STOP-OUT ACTUALLY COST — measured from this account's fills")
    print(f"{'=' * 78}")
    # WHAT WAS THROWN AWAY, ALWAYS, BEFORE THE ANSWER. The first version of
    # this script kept rows it could not read and reported a median of eight
    # thousand pips as though it were a measurement. A discard count that only
    # shows on failure is the same defect one level up.
    if discarded:
        print("\n  rows read but not usable:")
        for why, count in sorted(discarded.items(), key=lambda row: -row[1]):
            print(f"    {count:>5}  {why}")

    if not found:
        print("\n  No usable stop-outs in the journal.")
        if discarded:
            print("  Everything found was discarded for the reasons above — the most")
            print("  likely one is that break-even moved the stop before it was hit,")
            print("  which makes 'how far past the original stop' unanswerable.")
        else:
            print("  Until there are, `stop_slippage_pips` stays an assumption and every")
            print("  FX refusal on cost rests on it.")
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

    judgeable = [name for name in sorted(by_class) if len(by_class[name]) >= args.min_trades]
    if not judgeable:
        print(
            f"  Nothing has {args.min_trades} stop-outs yet, so there is nothing to\n"
            f"  conclude and no table below. The counts above are real; they are just\n"
            f"  too few to move a live limit on. Come back after more trades close,\n"
            f"  or pass --min-trades to see it anyway."
        )
    for name in judgeable:
        pips = by_class[name]
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

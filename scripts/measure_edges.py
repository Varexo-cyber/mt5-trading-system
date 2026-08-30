"""First-touch measurement on ten years of real M15 bars.

RUN IT:  python scripts/measure_edges.py [detector ...]

The bars are not in the repository. Fetch them once into `data/m15/`:

    for s in XAUUSD EURUSD GBPUSD USDJPY AUDUSD USDCAD EURJPY GBPJPY; do
      curl -sSL -o "data/m15/$s.m15.csv" \
        "https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/$s/${s}m15.csv"
    done

That is HistData M15 bid bars, 2012-2022, ~230,000 bars per instrument. It is
not the Eightcap feed and carries no spread, so it answers "does this entry
have an edge" and not "does it survive this broker" -- the second question is
`max_spread_share_of_stop` and the live journal.

THE RULES OF THIS HARNESS, each one because of a mistake already made:

* Barrier-resolved only. A trade that neither touched its stop nor its target
  inside the horizon is EXCLUDED from the win rate, not counted as a loss and
  not closed at the clock. Mixing clock exits into a first-touch model is what
  manufactured a +16.7% "edge" on a random walk.
* Same-bar ambiguity counts as a LOSS. When one bar spans both barriers the
  order is unknowable at M15; assuming the good one is how backtests lie.
* The baseline is the arithmetic one. For a driftless walk the chance of
  touching a stop of 1 before a target of k is 1/(1+k), so that is what the
  hit rate is measured against -- not against 50%.
* Bonferroni. Testing six payoffs on four strategies is 24 comparisons, and
  at 24 comparisons a 2-sigma result is expected by chance more often than
  not.
* Entries are at the close of the signal bar. Every indicator is computed on
  bars strictly before it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

MKT = Path(__file__).resolve().parent.parent / "data" / "m15"
SYMBOLS = ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY", "GBPJPY")
#: Bars a trade is allowed before it is written off as unresolved. 96 M15 bars
#: is 24 hours, which is the swing horizon the live account plans for.
HORIZON = 96


def load(symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(MKT / f"{symbol}.m15.csv", parse_dates=["Date"])
    frame = frame.rename(columns=str.lower).set_index("date")
    # The feed stores prices scaled by 100; scale is irrelevant to an
    # ATR-normalised measurement but makes the printouts readable.
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column] / 100.0
    return frame.dropna()


def atr(frame: pd.DataFrame, period: int = 14) -> np.ndarray:
    previous = frame["close"].shift(1)
    spans = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return spans.rolling(period).mean().to_numpy()


@dataclass(frozen=True, slots=True)
class Signal:
    index: int
    direction: int
    entry: float
    unit: float  # one R, in price


def resolve(
    high: np.ndarray,
    low: np.ndarray,
    signal: Signal,
    k: float,
    *,
    same_bar: bool = False,
) -> int | None:
    """+1 target first, 0 stop first, None neither inside the horizon.

    `same_bar` is not a detail. A LIMIT fill happens partway through its bar,
    so the rest of that bar can take the stop -- and skipping it scores a
    trade that was already dead as if it had never started. With a stop of
    0.35 ATR under the fill that is not a rare case, it is the common one, and
    it inflates the hit rate of exactly the strategy this file is arguing for.
    Entries taken AT A BAR'S CLOSE have no such remainder and start at the
    next bar.
    """
    stop = signal.entry - signal.direction * signal.unit
    target = signal.entry + signal.direction * signal.unit * k
    last = min(signal.index + HORIZON, len(high) - 1)
    for i in range(signal.index + (0 if same_bar else 1), last + 1):
        if signal.direction > 0:
            hit_stop = low[i] <= stop
            hit_target = high[i] >= target
        else:
            hit_stop = high[i] >= stop
            hit_target = low[i] <= target
        if hit_stop and hit_target:
            return 0  # same bar: unknowable order, count the bad one
        if hit_stop:
            return 0
        if hit_target:
            return 1
    return None


# --------------------------------------------------------------------------
# The detectors. Each returns entries as (index, direction, entry, unit).
# --------------------------------------------------------------------------


def donchian_break(frame: pd.DataFrame, a: np.ndarray, period: int = 20) -> list[Signal]:
    """Close beyond the N-bar extreme of the bars BEFORE it.

    The off-by-one matters: a channel that includes the current bar can almost
    never be closed beyond, and the strategy silently produces nothing.
    """
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    upper = pd.Series(high).shift(1).rolling(period).max().to_numpy()
    lower = pd.Series(low).shift(1).rolling(period).min().to_numpy()
    out: list[Signal] = []
    for i in range(period + 15, len(close) - HORIZON - 1):
        unit = a[i]
        if not np.isfinite(unit) or unit <= 0:
            continue
        if close[i] > upper[i]:
            out.append(Signal(i, 1, close[i], unit))
        elif close[i] < lower[i]:
            out.append(Signal(i, -1, close[i], unit))
    return out


def donchian_retest(
    frame: pd.DataFrame, a: np.ndarray, period: int = 20, tolerance: float = 0.35
) -> list[Signal]:
    """The same break, entered when price comes BACK to the level it cleared.

    The level is the channel edge that was broken, frozen at the break. The
    entry is the first later bar that trades within `tolerance` ATR of it, and
    the trade is abandoned if price closes back through the level (the break
    failed) or if the retest has not arrived within the horizon.

    One R is measured from the ENTRY, not from the break, so this is a shorter
    stop on the same objective -- which is the whole argument for it.
    """
    high, low, close = (frame[c].to_numpy() for c in ("high", "low", "close"))
    upper = pd.Series(high).shift(1).rolling(period).max().to_numpy()
    lower = pd.Series(low).shift(1).rolling(period).min().to_numpy()
    out: list[Signal] = []
    i = period + 15
    end = len(close) - HORIZON - 1
    while i < end:
        direction = 0
        if close[i] > upper[i]:
            direction, level = 1, upper[i]
        elif close[i] < lower[i]:
            direction, level = -1, lower[i]
        if direction == 0:
            i += 1
            continue
        unit_at_break = a[i]
        if not np.isfinite(unit_at_break) or unit_at_break <= 0:
            i += 1
            continue
        # Walk forward for the retest.
        for j in range(i + 1, min(i + HORIZON, end)):
            if direction > 0:
                failed = close[j] < level - 0.5 * unit_at_break
                touched = low[j] <= level + tolerance * unit_at_break
            else:
                failed = close[j] > level + 0.5 * unit_at_break
                touched = high[j] >= level - tolerance * unit_at_break
            # TOUCHED BEFORE FAILED, and this ordering was wrong until
            # 30 August. The entry sits between the level and the stop, so
            # price cannot reach the stop without passing through the limit
            # order first. Testing the failure first DISCARDED every bar that
            # swept through both at once -- 6% to 16% of the sample, all of
            # them losses -- and that alone accounted for three quarters of
            # this strategy's apparent edge (+0.336R -> +0.063R).
            if touched:
                entry = level + direction * tolerance * unit_at_break
                # The stop sits half an ATR beyond the level it retested.
                unit = 0.5 * unit_at_break + tolerance * unit_at_break
                out.append(Signal(j, direction, entry, unit))
                i = j
                break
            if failed:
                break
        i += 1
    return out


def bollinger_exhaustion(
    frame: pd.DataFrame, a: np.ndarray, period: int = 20, deviation: float = 2.5
) -> list[Signal]:
    """Price pierces a band and RSI is at an extreme. Fade it."""
    close = frame["close"]
    middle = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    upper = (middle + deviation * sigma).to_numpy()
    lower = (middle - deviation * sigma).to_numpy()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 7, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 7, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).to_numpy()
    values = close.to_numpy()
    out: list[Signal] = []
    for i in range(period + 15, len(values) - HORIZON - 1):
        unit = a[i]
        if not np.isfinite(unit) or unit <= 0 or not np.isfinite(rsi[i]):
            continue
        if values[i] < lower[i] and rsi[i] < 15:
            out.append(Signal(i, 1, values[i], unit))
        elif values[i] > upper[i] and rsi[i] > 85:
            out.append(Signal(i, -1, values[i], unit))
    return out


def trend_momentum(frame: pd.DataFrame, a: np.ndarray) -> list[Signal]:
    """EMA20 over EMA50, both rising, entered on a shallow pullback."""
    close = frame["close"]
    fast = close.ewm(span=20, adjust=False).mean().to_numpy()
    slow = close.ewm(span=50, adjust=False).mean().to_numpy()
    values = close.to_numpy()
    out: list[Signal] = []
    for i in range(60, len(values) - HORIZON - 1):
        unit = a[i]
        if not np.isfinite(unit) or unit <= 0:
            continue
        rising = fast[i] > slow[i] and fast[i] > fast[i - 5]
        falling = fast[i] < slow[i] and fast[i] < fast[i - 5]
        if rising and values[i] <= fast[i]:
            out.append(Signal(i, 1, values[i], unit))
        elif falling and values[i] >= fast[i]:
            out.append(Signal(i, -1, values[i], unit))
    return out


DETECTORS = {
    "donchian_break": donchian_break,
    "donchian_retest": donchian_retest,
    "bollinger_exhaustion": bollinger_exhaustion,
    "trend_momentum": trend_momentum,
}
RATIOS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)


def sigmas(wins: int, n: int, baseline: float) -> float:
    if n == 0:
        return 0.0
    spread = (baseline * (1 - baseline) / n) ** 0.5
    return (wins / n - baseline) / spread if spread > 0 else 0.0


def main() -> None:
    names = sys.argv[1:] or list(DETECTORS)
    comparisons = len(names) * len(RATIOS)
    bar = NormalDist().inv_cdf(1 - 0.05 / (2 * comparisons))
    print(f"horizon {HORIZON} M15 bars = {HORIZON / 4:.0f}h   symbols {len(SYMBOLS)}")
    print(f"Bonferroni over {comparisons} comparisons: needs {bar:.2f} sigma\n")

    frames = {s: load(s) for s in SYMBOLS}
    atrs = {s: atr(f) for s, f in frames.items()}

    for name in names:
        detector = DETECTORS[name]
        found: dict[str, tuple[pd.DataFrame, list[Signal]]] = {}
        total = 0
        for symbol in SYMBOLS:
            found[symbol] = (frames[symbol], detector(frames[symbol], atrs[symbol]))
            total += len(found[symbol][1])
        print(f"=== {name}   {total} signals over {len(SYMBOLS)} symbols")
        print(
            f"{'R:R':>5} {'resolved':>9} {'unres%':>7} {'hit':>7} {'chance':>7} "
            f"{'edge':>7} {'sigma':>7} {'E(c=0)':>8} {'E(c=10%)':>9}"
        )
        for k in RATIOS:
            wins = resolved = unresolved = 0
            for symbol in SYMBOLS:
                frame, signals = found[symbol]
                high = frame["high"].to_numpy()
                low = frame["low"].to_numpy()
                for signal in signals:
                    outcome = resolve(high, low, signal, k, same_bar=(name == "donchian_retest"))
                    if outcome is None:
                        unresolved += 1
                        continue
                    resolved += 1
                    wins += outcome
            if resolved == 0:
                continue
            hit = wins / resolved
            chance = 1.0 / (1.0 + k)
            edge = hit - chance
            s = sigmas(wins, resolved, chance)
            expectancy = hit * k - (1 - hit)
            with_cost = expectancy - 0.10
            flag = "  <--" if abs(s) >= bar else ""
            print(
                f"{k:>5.1f} {resolved:>9} {100 * unresolved / (resolved + unresolved):>6.1f}% "
                f"{100 * hit:>6.1f}% {100 * chance:>6.1f}% {100 * edge:>+6.1f} {s:>+7.2f} "
                f"{expectancy:>+8.3f} {with_cost:>+9.3f}{flag}"
            )
        print()


if __name__ == "__main__":
    main()

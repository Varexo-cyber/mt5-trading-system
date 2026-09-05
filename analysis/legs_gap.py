"""XAUJPY against its own two legs: the arithmetic, in one place.

    XAUJPY = XAUUSD x USDJPY

Two legs, quoted in two different books. When gold moves and the yen does not,
the cross has to follow, and whoever quotes the cross does that with a lag. The
counterparty is that market maker, and the trade pays when the quote catches
up. That is a sentence you can say out loud, which is more than any of the
twenty-eight generic mechanisms in `analysis/mechanisms.py` could manage on a
gold cross.

WHY THIS FILE EXISTS RATHER THAN THE FUNCTIONS LIVING IN THE SEARCH SCRIPT.
`scripts/search_xaujpy_legs.py` measured this and `analysis/section_eleven_legs.py`
trades it. When those are two implementations of "the same" arithmetic, the
thing that was measured is not the thing that runs, and every disagreement
between them is invisible -- the search reports a number for a rule nothing
executes. `analysis/mechanisms.py` exists for exactly this reason and this is
the same move for the one mechanism that needs three instruments instead of
one.

WHAT IS DELIBERATELY NOT TRADED. The relationship carries a constant offset --
contract size, a broker markup, a financing component -- and none of it is
tradeable. The reading is therefore the gap's DEVIATION FROM ITS OWN RECENT
NORMAL, not the raw difference. A structural offset cancels; only the lag
survives.

THE FAILURE THIS FILE IS MOSTLY MADE OF. Gold pauses daily around 21:00-22:00
UTC while the yen leg trades on. The implied cross then walks away from a
FROZEN cross quote, the gap explodes, and none of it is tradeable because the
thing you would trade is not being priced. The first run of the search put two
thirds of its profit in exactly that window. `_alive` is the guard, and
removing it is how this mechanism turns back into a screenshot of one leg held
against a live one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.mechanisms import _atr

#: Bars the gap is compared against to remove the structural offset. Long
#: enough that a real lag is an outlier against it, short enough that a slow
#: drift in the offset does not become a permanent signal.
NORMAL_BARS = 96

#: Below this median absolute gap the cross is being COMPUTED from its legs
#: rather than quoted independently, and there is no lag to trade at all.
#: Expressed in ATR of the cross, so it is scale-free.
#:
#: This is the question the search answers before it resolves a single trade:
#: many brokers synthesise a cross from its legs, and a search that finds an
#: edge inside a gap that cannot exist has found its own rounding error.
DEAD_GAP_ATR = 0.02


def alive(frame: pd.DataFrame) -> np.ndarray:
    """Bars this instrument actually traded on.

    A BAR WITH NO RANGE IS A QUOTE THAT DID NOT MOVE, and on a gold cross that
    is not a quiet market, it is a CLOSED one. A range of zero is the signature
    and it needs no calendar: it works on every instrument, on every clock, and
    on whatever hours this broker happens to pause.
    """

    return (frame["high"].to_numpy() > frame["low"].to_numpy()) & np.isfinite(
        frame["close"].to_numpy()
    )


def implied_cross(gold: pd.DataFrame, yen: pd.DataFrame) -> pd.Series:
    """What XAUJPY has to be, from the two books that actually price it.

    Closes only, and on the shared timestamps only. An inner join is the whole
    alignment: MT5 puts every symbol on the same bar grid, so a timestamp
    present in one and missing in the other is a bar one of them did not trade,
    and inventing it would invent the gap this is looking for.
    """

    joined = gold.join(yen, how="inner", lsuffix="_au", rsuffix="_jp")
    # EITHER LEG STANDING STILL IS ENOUGH TO INVENT A GAP.
    gold_alive = alive(joined.rename(columns={c: c.replace("_au", "") for c in joined.columns}))
    yen_alive = joined["high_jp"].to_numpy() > joined["low_jp"].to_numpy()
    product = joined["close_au"] * joined["close_jp"]
    return product.where(pd.Series(gold_alive & yen_alive, index=joined.index))


def gap_reading(
    cross: pd.DataFrame, implied: pd.Series
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """`(the shared bars, gap in ATR, raw gap in price)`.

    RETURNS THE FRAME IT ALIGNED, and that is not tidiness. Handing back only
    the readings leaves the caller to rebuild the shared index with a second
    intersection, and two alignments of one thing is how a reading ends up one
    bar out of step with the bar it labels -- silently, and in the direction
    that flatters, because a gap read against the NEXT bar's price is a
    look-ahead. One alignment, returned.
    """

    aligned = cross.join(implied.rename("implied"), how="inner")
    frame = aligned[["open", "high", "low", "close"]]
    raw = (aligned["close"] - aligned["implied"]).to_numpy()
    # A frozen cross quote makes the gap, it does not reveal one.
    raw = np.where(alive(frame), raw, np.nan)
    normal = pd.Series(raw).rolling(NORMAL_BARS, min_periods=NORMAL_BARS // 2).mean().to_numpy()
    unit = _atr(frame)
    with np.errstate(invalid="ignore", divide="ignore"):
        reading = (raw - normal) / np.where(unit > 0, unit, np.nan)
    return frame, reading, raw


def signals_from_gap(reading: np.ndarray, threshold: float) -> np.ndarray:
    """Rich cross is sold, cheap cross is bought.

    A cross above its legs is a quote that has not come down yet; the trade is
    that it does. Direction comes from the GAP and not from the market, which
    is why this can be right about the trade while being wrong about where gold
    goes -- the same property that makes `basket_divergence` worth having.
    """

    out = np.zeros(len(reading), dtype=int)
    with np.errstate(invalid="ignore"):
        out[reading >= threshold] = -1
        out[reading <= -threshold] = 1
    return out

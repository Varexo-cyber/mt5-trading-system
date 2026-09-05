# Section 11 — the XAUJPY quote against its own two legs, on M5

> The pre-registration for this is `docs/hypotheses/xaujpy_searched_mechanism.md`
> plus the bar written into `scripts/search_xaujpy_legs.py` before it was run:
> sixteen cells, Bonferroni 2,96 sigma, a rate-matched random control and an
> untouched holdout. **This document records that the search did not clear that
> bar, and what was built anyway.**

## The mechanism

    XAUJPY = XAUUSD × USDJPY

Gold moves, the yen does not, and the cross has to follow. Whoever quotes the
cross does that with a lag. The trade is that the quote catches up.

This matters because it is the **first mechanism on this account with a
counterparty you can name**. The twenty-eight in `analysis/mechanisms.py` are
generic one-to-four-bar patterns written for index CFDs; pointed at a gold
cross the best of them was "four closes the same way, then trade against it",
which fires on 10,3 % of all bars and lost 0,18 R a trade. Nobody loses money
because four one-minute candles went up. Somebody does lose money quoting a
stale cross.

The reading is the gap's **deviation from its own rolling normal**, in ATR of
the cross. Contract size, a broker markup and a financing component all put a
constant between the cross and the product of its legs and none of it is
tradeable; subtracting the 96-bar normal removes exactly that and leaves the
lag.

## What the search answered first, before any backtest

Many brokers **compute** a cross from its legs. If Eightcap did, the gap would
be zero by construction — no lag, no counterparty, nothing to trade — and any
edge found inside it would be the rounding error of a multiplication.

    M5    139 979 bars   median |gap| 0,015 ATR   90th 0,053   max 5,818
    M15    46 877 bars   median |gap| 0,010 ATR   90th 0,037   max 2,007

Above the 0,005 floor, and **not by much**. `minimum_gap_atr` is a live check
rather than a line in this document because that margin can close.

## The result, and it missed

`benen.cmd 720` — 720 days of Eightcap bars, sixteen cells, bar 2,96 sigma:

| clock | gap | R:R | trades | /day | total R | per trade | sigma | holdout |
|---|---|---|---|---|---|---|---|---|
| M5 | 0,50 | 1,5 | 152 | 0,3 | +54,80 | **+0,361** | **+2,93** | +1,47 |
| M5 | 0,25 | 1,5 | 438 | 0,8 | +69,99 | +0,160 | +2,63 | +18,39 |
| M5 | 0,25 | 1,0 | 492 | 0,8 | +51,74 | +0,105 | +2,36 | +11,89 |

All twelve surviving cells were positive. **None cleared.** The best missed by
0,03 sigma, and the bar does not move afterwards — a bar you adjust once you
have seen the number is not a bar.

So section 11 is `enabled: true` for replay and journal and is **not** on
`live_enabled_modules`. What is configured is "the strongest thing the search
found", not "a proven edge".

## Three things that have to be said next to that number

**The trade rate is 0,3 a day.** Three to five was asked for. This is a
sixteenth of it, and it is not fixable with a threshold: gap 0,25 traded three
times as often for less than half the edge a trade.

**The legs are read on closed bars of the same clock.** A lag that opens and
closes inside one M5 bar is invisible to the measurement and would be invisible
to the live module too. If the real lag is faster than five minutes then
+0,361 is a fraction of what is there, and reading the legs on M1 or on ticks
is the next experiment rather than a tweak to this one.

**A frozen leg invents a gap.** Gold pauses daily around 21:00–22:00 UTC while
the yen leg keeps trading; the implied cross then walks away from a quote that
is not moving, and the difference is an artefact. The first run of the search
put two thirds of its profit in exactly that window. `alive()` in
`analysis/legs_gap.py` excludes it by measuring a zero range rather than by
consulting an hour table, and adding the guard cost about half a sigma — which
is the size of the thing it removed.

## What would move this to real money

Not a re-run of the same search, and not a wider grid — a wider grid raises the
bar it has to clear. Two things would:

1. **Forward data.** 152 trades over 720 days is thin. Trades taken after the
   search date are not part of what was searched, and enough of them turn +2,93
   into a number with no selection in it at all.
2. **A faster read of the legs.** If the lag is sub-M5, an M1 or tick version
   measures a different and larger thing. That is a new pre-registration, not
   an amendment to this one.

## Where it lives

| what | where |
|---|---|
| the arithmetic, shared with the search | `analysis/legs_gap.py` |
| the live module | `analysis/section_eleven_legs.py` |
| the config and its numbers | `SectionElevenLegsConfig`, `config/eightcap.yaml` |
| the legs handed to it live | `JarvisRunner._attach_xaujpy_legs` |
| the search | `scripts/search_xaujpy_legs.py`, `benen.cmd` |
| the replay | `sectie11.cmd`, and `hoeveel.cmd` for the whole account |

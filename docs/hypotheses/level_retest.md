# Hypothesis: level_retest

## Claim

**A range break is worth trading at the level it cleared, and worth nothing at
the break itself.**

Not "a breakout works" and not "a breakout fails". The same break, entered in
two places, is two different trades with opposite signs — and the account has
been taking the losing one for two months.

## Economic rationale

The edge a broken level carries is not prediction, it is a queue. Resting
orders sit at the edge of a range: stops from the players who were fading it,
and entries from the players who wanted in on a break but not at any price.
When the range gives way the stops go first and price runs. What is left at the
old edge is the second group — real buyers with a defined invalidation one tick
below.

Entering at the break means buying from that first group at the worst price in
the range, with no one underneath. Entering at the retest means standing where
the remaining bids are, with a stop just beyond them. The reason the second
works is not that the market is more likely to go up; it is that being wrong is
cheap and identifiable, and the same objective sits four times further away.

That also explains why the tolerance is so unforgiving. The queue is *at the
level*. Half an ATR above it there is nothing to stand on, and the trade is
just a slightly less extended version of the one that loses.

## Predictions, stated in advance

1. Buying the break is negative, not merely zero. If the only problem were
   cost, an uncosted measurement would read ~0.00R.
2. Buying its retest is positive on the same population.
3. The edge is monotone in distance from the level, and turns negative some
   way out.
4. The break's expectancy is flat in R:R — a fixed first-touch edge times
   (k+1) is a constant — so raising the payoff cannot rescue it.

## Test design

HistData M15 bid bars, 2012–2022, eight instruments (XAUUSD, EURUSD, GBPUSD,
USDJPY, AUDUSD, USDCAD, EURJPY, GBPJPY). 1.84 million bars, ~230,000 each.

`scripts/measure_edges.py` and `scripts/measure_retest_sweep.py`. Rules, each
one from a mistake made earlier on this account:

- **Barrier-resolved only.** A trade that touched neither barrier inside 96
  bars (24h) is excluded, not counted as a loss and not closed at the clock.
  Mixing clock exits into a first-touch model is what manufactured a +16.7%
  edge on a random walk in `backtest_section_six.py`.
- **The stop is checked inside the fill bar.** A limit fill happens partway
  through its bar, so the rest of that bar can take the stop. The first run of
  this study skipped it and reported E +0.487R where the honest number is
  +0.347R.
- **Same-bar ambiguity counts as a loss.** When one bar spans both barriers the
  order is unknowable at M15.
- **The baseline is arithmetic**, 1/(1+k), not 50%.
- **Bonferroni** over every cell swept.
- **Parameters chosen on 2012–2017.** The 2018–2021 holdout was read once, for
  the winning cell, after it was fixed.

## Result

| detector | signals | E | sigma |
|---|---|---|---|
| buy the channel break | 190,505 | **−0.067R** | −29 |
| buy its retest (best cell, train) | 49,700 | **+0.340R** | +54 |
| buy its retest (best cell, holdout) | 30,507 | **+0.347R** | +43 |
| buy its retest (shipped cell, train) | 55,582 | **+0.134R** | +22 |

All four predictions held. The break's expectancy was −0.067R at every ratio
from 1:1 to 5:1 — flat, as predicted. The holdout reproduced its training half
to within 0.007R across four unseen years.

Distance from the level, the parameter the whole thing turns on:

| entered within | edge over chance | sigma |
|---|---|---|
| 0.15 ATR | +11.3 points | +54 |
| 0.35 ATR | +5.0 points | +20 |
| 0.60 ATR | **−3.4 points** | −20 |

Monotone across three channel lengths and 27 configurations, and it changes
sign. This is why `lifecycle_retest_level_atr` moved from 0.35 to 0.15.

### Measured on the same data and rejected

| candidate | verdict |
|---|---|
| Bollinger 2.5σ + RSI(7) < 15 / > 85 | 16,391 signals, +0.5σ. No edge at any ratio 1:1–5:1. |
| Far-break continuation (close > 2 ATR past level) | train +0.103R, **holdout −0.012R**. Mined. |
| trend_momentum (EMA20/50 alignment) | 176,341 signals, E +0.02R. Real (+8σ) and worthless — dies at any cost above 1% of R. |
| Fade a break closing on its own extreme | holdout +0.086R at +6.4σ; net of modelled cost +0.006R. Real, too thin to trade. |

The Bollinger/RSI row is the gold scalp the owner was quoted. It has no edge.

## Correction, 30 August: the result above was my own bug

The measurement that produced +0.340R/+0.347R checked the setup's FAILURE
before its FILL:

    if failed: break        <- first
    if touched: enter

The entry sits between the level and the stop, so price cannot reach the stop
without passing through the limit order. Every bar that swept through both at
once was therefore discarded instead of entered -- and those are the worst
losses the strategy has. Six to sixteen percent of the sample, all losers,
silently removed.

With the two checks in the right order, on the same bars:

| stop | before | after |
|---|---|---|
| 0.50 ATR | +0.336R | **+0.063R** |
| 0.90 ATR | +0.133R | **+0.036R** |

And net of a spread, across every asset class and timeframe measured:

| asset | tf | R | gross | cost | net | sigma |
|---|---|---|---|---|---|---|
| index | M5 | 0.90 | +0.098 | 0.178 | **−0.079** | +3.5 |
| index | H1 | 0.90 | +0.090 | 0.178 | **−0.088** | +3.0 |
| fx | H1 | 0.90 | +0.084 | 0.089 | **−0.004** | +2.9 |
| fx | M15 | 0.90 | +0.034 | 0.089 | **−0.054** | +1.2 |
| metal | H1 | 0.90 | +0.084 | 0.222 | **−0.138** | +2.5 |

Gross positive everywhere. Net negative in all fourteen rows measured. The
best of them is zero.

**What survives the correction:** the retest is still 0.10–0.20R better than
buying the same break (−0.067R), and that comparison is untouched by the bug —
a break entry has no limit order and so no fill-ordering question. Choosing the
level over the extreme remains right. Claiming it pays for itself does not.

## Verdict

**NOT live.** Pulled from `live_enabled_modules` on 30 August, the same day it
was added, once the harness bug was found.

What is *not* established, and it is the whole residual risk: this is HistData
M15 **bid** bars on forex and gold. It contains no Eightcap feed, no indices,
no spread, no commission and no slippage. Cost is exactly what has killed every
other detector here — the account's own four-market table shows an edge of ~+11
points over chance in all four markets and cost deciding which ones survived.

The shipped stop is 0.90 ATR rather than the 0.50 that measured best, because
`ConfluenceConfig.min_stop_atr` floors the stop at 0.8 and a floor that
silently widens it would leave the module measuring one trade and sending
another. That costs expectancy (+0.134R against +0.340R) and buys the guarantee
that what was measured is what gets sent.

**If it is ever re-enabled, the pre-registered kill condition is:** 40 live trades at 65% losers. Expected
is 55% losers at a 2R target, so 65% over forty is about four sigma the wrong
way — that is a measurement which does not hold on this feed, not bad luck.

## Related

- `analysis/setup_lifecycle.py` — the same rule applied to the other breakout
  families, which have their own levels and now retest them.
- `docs/hypotheses/entry_quality.md` — reads distance from the recent extreme,
  which is what let an entry 2.5 ATR above a broken level look timely.
- `docs/hypotheses/market_structure.md` — publishes the swing it broke as its
  first `key_level`, which is where the lifecycle's level now comes from.

# Hypothesis: impulse_retest

## Claim

**A break that closes a full ATR clear of its level is worth buying — at the
level, not at the break.**

Three clauses, each measured on its own: the break must be *decisive*, the
entry must be *at the level*, and the target must be *near*.

## Economic rationale

The edge is a queue, not a forecast.

Resting orders sit at the edge of a range: stops from the people fading it, and
entries from the people who want the break but not at any price. When the range
gives way, the stops go first and price runs. What is left behind at the old
edge is the second group — real bids with a defined invalidation one tick below
them.

Buying the break means buying from the first group, at the worst price in the
range, with nothing underneath. Buying the retest means standing where the
remaining bids are. You are not more likely to be right; being wrong is simply
cheap and identifiable.

**Why the impulse filter matters more than anything else here.** A break that
closes a full ATR beyond the level consumed every resting offer on the way
through — that is what a one-ATR close *is*. A break that pokes through
consumed almost nothing, and there is no queue behind it to stand in. The same
retest without this filter nets roughly zero.

**Why the target is near, not far.** The queue is a local fact. It supports
price for as long as it takes to be filled, which is not long. A far target
asks the trade to keep working after its reason has expired.

## Predictions, stated in advance

1. Buying the break itself is negative, not merely zero.
2. The same break, bought at its level, is positive.
3. The edge is monotone in impulse size and in closeness to the level.
4. Expectancy peaks at a near target and decays toward higher payoffs.
5. It fails on gold — not for lack of edge, but for cost.

## Test design

HistData, 2010–2022. Sixteen instruments: four equity indices at native M1
(SPX500, DAX40, Nikkei225, EuroStoxx50) and twelve FX/metal at M15–H4. 1.84
million bars on the FX side, ~3.3 million per index.

Ninety-four detectors — candlestick shapes, oscillators, trend, volatility,
prior-session levels, round numbers, sixteen clock detectors, volume, and
fifteen filters on the retest itself — over six timeframes and four payoffs.

`scripts/measure_edges.py`. The rules, each from a mistake made on this
account:

- **Barrier-resolved only.** Unresolved trades are excluded, not closed at the
  clock. Clock exits in a first-touch model once manufactured a +16.7% edge on
  a random walk.
- **Same-bar ambiguity counts as a loss.**
- **Limit fills resolve from their own bar.** The rest of the fill bar can take
  the stop. Skipping it read +0.487R where the truth was +0.347R.
- **The fill is checked before the failure.** The entry sits between the level
  and the stop, so price cannot reach the stop without passing through the
  limit. Testing the failure first discarded every bar that swept through both
  — 6–16% of the sample, all losses — and that alone was three quarters of an
  earlier version's apparent edge.
- **Sigma clustered by day.** Eleven pairs break on the same morning; counting
  that as eleven observations overstates significance by about √11.
- **Bonferroni over the entire grid**, not per family.
- **Parameters on 2010–2016; the 2017–2022 holdout must agree on its own.**
- **A coin-flip control.** It did not read zero — see below.

### The control, and why every number here is smaller than it looks

Random entries, same instruments, same resolver:

| R:R | E | sigma |
|---|---|---|
| 1.0 | −0.002 | −0.7 |
| 2.0 | +0.049 | +11.2 |
| 3.0 | +0.073 | +13.8 |

Thirteen sigma of edge from entering at random. A bar registers a barrier when
its extreme crosses it, and that overshoot is proportionally larger on the
nearer barrier, so the stop is effectively further away than nominal and the
bias grows with the payoff. **Every expectancy below has this subtracted**, and
it is a second reason to trade at 1:1: that is the point on the curve where the
harness itself is honest.

## Result

Shipped configuration — 20-bar channel, impulse ≥ 1.0 ATR, fill within 0.15
ATR, stop 0.85 ATR beyond the level (R = 1.00 ATR), target 1R, FX majors:

| tf | trades | hit | gross | control | spread | **net** | train / test |
|---|---|---|---|---|---|---|---|
| M15 | 18,828 | 67.9% | +0.357 | −0.002 | 0.080 | **+0.279** | +0.272 / +0.285 |
| H1 | 5,235 | 67.9% | +0.359 | −0.002 | 0.080 | **+0.281** | +0.276 / +0.284 |

By year, net R per trade — eleven years, eleven positive:

| | | | | | |
|---|---|---|---|---|---|
| 2012 +0.394 | 2013 +0.254 | 2014 +0.244 | 2015 +0.199 | 2016 +0.245 | 2017 +0.239 |
| 2018 +0.194 | 2019 +0.181 | 2020 +0.249 | 2021 +0.292 | 2022 +0.386 | |

Prediction 4, the target curve at a one-ATR stop:

| target | net | target | net |
|---|---|---|---|
| 0.75R | +0.245 | 1.5R | +0.176 |
| **1.00R** | **+0.279** | 2.0R | +0.081 |
| 1.25R | +0.235 | 3.0R | +0.016 |

Prediction 1, on the same population: buying the break is **−0.067R** over
190,505 signals at −29 sigma, flat across every ratio from 1:1 to 5:1.

Prediction 5, gold: gross **+0.324R**, spread 0.40R, net **−0.099R**. It works
and cannot be paid for — the same arithmetic behind this account's 595 real
gold trades at −0.203R.

All five predictions held.

### What else was measured, and failed

Of ninety-four detectors, everything below either failed its holdout or was
eaten by cost: pin bars, engulfing, outside bars, three-bar reversals,
momentum runs both ways, RSI 7/14 at three thresholds, RSI divergence,
z-score, Bollinger touches, EMA stretch, MACD, EMA cross, turtle breaks, NR7,
ATR expansion both ways, squeeze release, opening range, first-hour
projections, prior-day breaks and fades, round numbers, sixteen hour-of-day
detectors, weekday effects, session-close reversals, volume spikes both ways,
gap fill, false breaks, failed auctions, inside bars, range fades, midpoint
reversion, streak reversals, and opening drives.

Two are worth naming because they were close: `retest_slow` (net +0.064R) and
`exhaustion_fade` on M1 indices (+0.041R). Both real, both too thin.

`gap_fill` appeared to be the biggest winner in the study at +1.011R holdout.
It was a look-ahead bug — the entry is a bar's open and resolution started at
the next bar, skipping the bar in which the median weekend gap retraces 56% of
itself. Fixed, it is nothing.

## Verdict

**Live as section two**, weight 1.2, target 1:1 via
`target_r_multiple_by_family`, with a section breaker at 40 trades / 45%
losers — about four sigma against a 68% expectation.

**What is still not established:** this is HistData M15 bid bars. It contains
no Eightcap feed, no commission, no slippage, and no live spread. The spread
assumed is 0.04 ATR on FX; at 0.08 ATR the net halves to roughly +0.14R, and at
0.20 ATR it is gone. Cost is what has killed every other detector on this
account, and it is the one number this study had to assume rather than measure.

The breaker is set for exactly that failure.

## Related

- `analysis/level_retest.py` — the same idea without the impulse filter.
  Measured, gross positive, net negative everywhere; not live.
- `analysis/setup_lifecycle.py` — sends the other breakout families back to
  their level rather than to a pullback from the extreme.
- `scripts/measure_edges.py` — the harness, including the coin-flip control.

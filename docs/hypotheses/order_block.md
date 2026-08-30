# Hypothesis: order_block

## Claim

**The last candle going the other way before a violent move is where someone
was still being filled. When price comes back to it, they are still there.**

Not a level — an *area*. The body of the candle that got absorbed.

## Economic rationale

A bar whose body spans 1.5 ATR did not persuade anyone; it consumed them. Every
resting order between its open and its close was taken. The last candle
printing the *other* way immediately before it is, by construction, the last
price at which the losing side was still getting filled.

If that participant had size left — and a 1.5 ATR candle is what it looks like
when someone is trying to finish in a hurry — they did not get it done. The
claim is that they are still working the same order when price returns, and
that is what defends the zone.

**Why this is not section two.** Section two needs a twenty-bar channel to
break and stands in the queue that survives the break. This needs no level at
all: one violent candle and the candle before it. 91.4% of its entries fall on
bars where section two does not fire, and every number below is measured with
the other 8.6% removed.

**Why the tolerance goes *into* the body rather than around a price.** The
defended thing is an area with an edge. Reaching 0.25 ATR inside it is standing
at the near side of the queue; reaching 0.50 ATR in is reaching past the orders
and hoping.

## Predictions, stated in advance

1. The edge scales with the impulse — a small candle absorbed nobody.
2. It measures on the *body*, not the range: a wide bar with a small body is
   indecision, and should carry no edge.
3. The optimum target is near, for the same reason as section two: a local
   supply of orders is used up quickly.
4. It works where it can be paid for — FX — and not on the wide-spread
   instruments, and the difference will be cost rather than signal.

## Test design

HistData 2010–2022, sixteen instruments, six timeframes. The detector was one
of ten in a third search deliberately restricted to mechanisms that can fire
when a retest cannot: fractal swing sweeps, swing-break retests, Fibonacci
retracement at 38.2/50/61.8, prior-week breaks and fades, trend-day
continuation, session VWAP reversion, and cross-market peer divergence.

`scripts/lab/`. Rules as always: barrier-resolved only, same-bar ambiguity
counted as a loss, limit fills resolved from their own bar, the fill checked
before the failure, sigma clustered by day, Bonferroni over the whole grid, and
**every expectancy measured against a coin-flip control** rather than against
theory — random entries read +0.073R at a 3:1 target on this harness, and that
bias is subtracted.

Additionally, and specific to this section: **every bar on which section two
also fires, in the same direction and within three bars, is removed** before
anything is measured. A second section that borrows the first one's trades is
not a second section.

## Result

FX M30, disjoint from section two, target 1R, stop 1.0 ATR:

| | trades | hit | gross | control | spread | **net** |
|---|---|---|---|---|---|---|
| all | 31,376 | 62.1% | +0.242 | −0.002 | 0.080 | **+0.164** |
| train | | | | | | +0.168 |
| holdout | | | | | | +0.161 |

By year — eleven of eleven positive:

| | | | | | |
|---|---|---|---|---|---|
| 2012 +0.289 | 2013 +0.143 | 2014 +0.149 | 2015 +0.169 | 2016 +0.199 | 2017 +0.139 |
| 2018 +0.181 | 2019 +0.132 | 2020 +0.179 | 2021 +0.161 | 2022 +0.267 | |

114 of 114 months positive, worst month +0.9R. 72.1% of days positive, worst
day −14.2R. Worst drawdown −22.5R over 31,376 trades.

Prediction 1, the impulse threshold — most of the edge lives here:

| impulse | trades | net |
|---|---|---|
| ≥ 1.0 ATR | 122,927 | +0.083 |
| **≥ 1.5 ATR** | **37,049** | **+0.172** |
| ≥ 2.0 ATR | 12,441 | +0.185 |

Prediction 3, the target: 0.75R → +0.147, **1.0R → +0.172**, 1.5R → +0.099.

Prediction 4, by asset class at M30:

| asset | trades | hit | gross | spread | net |
|---|---|---|---|---|---|
| fx | 31,376 | 62.1% | +0.242 | 0.080 | **+0.164** |
| index | 9,357 | 59.3% | +0.187 | 0.160 | +0.029 |
| metal | 2,752 | 57.5% | +0.150 | 0.133 | +0.018 |

Gross positive on all three; only FX clears its own cost by a margin. As
predicted, the difference is cost.

Timeframes: H1 +0.193R over 18,527 trades, M30 +0.164R over 31,376, M15
+0.156R over 59,846. M30 ships — it is the best trade-off, and it is a
different clock from section two's M15, which keeps the two sections off each
other's bars.

### What was measured alongside it and failed

`liquidity_sweep` (wick through a confirmed swing, close back inside),
`swing_break_retest`, Fibonacci at three levels, `prior_week_break`,
`prior_week_fade`, `trend_day_continuation`, `session_vwap_reversion`. None
survived its holdout.

`retest_slow`, the runner-up from the previous search, was rejected on a
different ground: it measured +0.212R but shares 47.5% of its trades with
section two. Not a second strategy.

## Verdict

**Live as section three**, weight 1.0 (below section two's 1.2, because it is
real and smaller), target 1:1 via `target_r_multiple_by_family`, gold on a 1.5
ATR stop so it clears the spread gate, section breaker at 40 trades / 55%
losers against a 62% expectation.

**What is not established:** HistData bid bars — no Eightcap feed, no
commission, no slippage. The assumed FX spread is 0.04 ATR. At 0.12 ATR this
section is at zero. That is the number the breaker is watching for.

## Related

- `docs/hypotheses/impulse_retest.md` — section two, and the overlap analysis.
- `scripts/lab/zoo3.py` — the nine mechanisms tested beside this one.

# Hypothesis: level_reaction

> Written before any backtest. This one fails its own test at step one; the
> conclusion is recorded rather than the module quietly kept.

## Claim (as implemented)

When the current H1 candle's low comes within 0.35 ATR of the 5th percentile of
the previous 50 bars' lows, prints a lower wick more than 1.5× its upper wick,
and closes green, price continues up. Symmetrically for the short side.

## Economic rationale

**There isn't one, and that is the finding.**

The project's own rule is that a module needs a story for who is losing money
on the other side. I cannot write that story for this module, and I am not
going to invent one.

A rolling 5th/95th percentile of the last 50 H1 bars is a **statistical
artefact, not a level**. Nobody is watching it. No orders rest at it. It moves
every hour by construction. The mechanisms that make real support and
resistance work — a price where a large participant previously transacted and
will defend again, a round number where option barriers and human psychology
cluster, a prior day's high where stops sit — all depend on the level being
*fixed, visible, and remembered*. A rolling quantile is none of those.

What the module actually detects is "a wick, somewhere near the recent edge of
the range". That is a description of ordinary candle behaviour in a range. It
will fire constantly, and the wick-ratio filter selects for noise rather than
for a defended price.

Compare with what `docs/modules/market_structure.md` requires of a level:
touch count, age, and whether it ever flipped from resistance to support. None
of that is present here.

**Secondary problem: it double-counts.** In a range, the 5th percentile of the
lows and the 20-bar low that `liquidity_sweep` watches are frequently the same
price. Two modules voting on one event is exactly the failure the brief warned
about, and here it happens by construction rather than by coincidence.

## Predictions, stated in advance

If forced to predict: win rate near the base rate for a coin flip minus the
spread, so a negative expectancy of roughly −0.05R once costs are included.
Signals far more frequent than the other modules, which is itself a warning
sign — a genuine location-based edge is rare.

**What would prove me wrong:** a positive deflated Sharpe on the validation
split across at least two instruments it was not developed on. That is a real
possibility and the module stays in the tree so it can be tested. But it should
not be spending money while that test is pending.

## Test design

- Development instruments: EURUSD, GBPUSD, USDJPY
- Held back: AUDUSD, USDCAD
- Timeframe: H1
- Split: 60/20/20
- Parameters swept: quantile 0.02/0.05/0.10, proximity 0.2/0.35/0.5 ATR,
  wick ratio 1.2/1.5/2.0
- **Configurations tested: 27**
- **Required before any live weight:** replace the rolling quantile with an
  actual level definition — touch count, age, flip history, round numbers,
  prior day/week high and low — and re-register as a new hypothesis. The
  rewritten module is not this module.

## Result

Not run.

## Verdict

**Weight 0. Removed from `live_enabled_modules`.**

This is not a rejection of level-based trading — levels are one of the five
core modules in the plan and the concept is sound. It is a rejection of *this
implementation*, which uses a rolling statistic where it needs a remembered
price. Rewrite it against `analysis/levels.py` as originally specified (touch
count, age, S→R flips, daily/weekly opens, round numbers), register that as a
fresh hypothesis, and test it properly.

Kept in the tree at weight 0 so the record of having tried it survives.

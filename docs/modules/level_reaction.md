# Module: level_reaction

## Status: weight 0, not live-enabled

Rejected on economic grounds before testing. The full reasoning is in
`docs/hypotheses/level_reaction.md`; the summary is that the module treats a
rolling 5th/95th percentile of the last 50 H1 bars as a support/resistance
level, and a rolling quantile is a statistical artefact rather than a price
anyone defends. No counterparty can be named, which is the project's own
threshold for whether a module may carry weight.

It also overlaps with `liquidity_sweep` by construction: in a range, the 5th
percentile of the lows and the 20-bar low are often the same price.

## What it measures

Current H1 candle low within 0.35 ATR of the 5th percentile of the previous
50 bars' lows, with a lower wick more than 1.5× the upper wick and a green
close. Mirrored for the short side.

## Measured edge

Not measured, and not scheduled. Testing this implementation would spend
configurations from the deflated-Sharpe budget on a hypothesis with no
mechanism.

## Replacement

Rewrite against the `analysis/levels.py` specification in `PLAN.md`: touch
count, level age, whether it ever flipped support↔resistance, daily/weekly/
monthly opens, and round numbers. That is a different module and needs a fresh
pre-registered hypothesis. Level-based trading is sound; this implementation
of it is not.

## Weight

**0.** Kept in the tree so the record of having tried it survives.

# Section 17 — BTCUSD M15, trend pullback

> Promotion record, the shared caveat and the containment:
> [btcusd_discoveries.md](btcusd_discoveries.md). Read that first.

## Claim

On BTCUSD M15, price that dips into its own EMA20 while the trend is up, and
closes back above it as a green bar, continues.

## The rule, exactly

Per closed M15 bar, any hour, any day:

* long when `EMA20 > EMA50`, the bar's LOW touched or crossed the EMA20, the
  close is back above the EMA20, and the close is above the open
* short is the mirror: `EMA20 < EMA50`, the HIGH touched the EMA20, close back
  below it, close below open
* `polarity: +1` — this one trades what it reads, unlike 15 and 16

Stop 4.0 ATR, target 4.0 R, stop to entry at 1.0 R.

## Economic rationale

This is the only one of the three whose story does not depend on a fitted
direction. A trend that is being bought on dips has a counterparty you can
name: whoever is providing that liquidity at the mean and is wrong about the
trend ending. The bar closing back through the average is the evidence that
the dip was absorbed rather than that the trend broke.

It is also the oldest idea in this repository — `ema_pullback_resume` is the
same shape on other markets, and it carries weight 0.55 there for exactly the
reason this needs watching: the idea is sound and the edge is thin.

## The 4 R target is the whole risk profile

A 4 R target with break-even at 1 R means most trades scratch. The win rate
will look bad and that is by construction; the number to read is R per trade,
not the hit rate. A run that reports 25 % winners here is not a failing
section, and a run that reports 70 % winners means the break-even move is
firing on almost everything and the target is decorative.

The account's own `TARGET_RARELY_REACHED` gate asks precisely this question
against the historical reach rate, and it now runs on this section live. If it
refuses most setups, the 4 R target is the thing to revisit, not the gate.

## What would take it off

The section breaker. Beyond that: R per trade turning negative while the hit
rate stays flat means the losers got bigger, which on a 4.0 ATR stop means the
market's volatility moved out from under the parameter.

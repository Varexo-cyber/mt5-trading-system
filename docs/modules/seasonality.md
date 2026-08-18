# Module: seasonality

## What it measures

On the daily chart, over `lookback_days` (500):

1. Bar-to-bar returns in percent, so instruments at different price levels are
   on one scale and a standard error means the same thing on gold as on EURUSD.
2. The subset belonging to **the weekday of `ctx.now`** — the day the trade
   would be held through, not the weekday of the last closed bar, which would
   measure yesterday and act on it today.
3. A t-statistic: mean divided by standard error. Below `minimum_t` (2.0) it
   returns neutral and says what it measured, so the operator sees the module
   working rather than only seeing it silent.

It refuses outright below `minimum_samples` (60) observations of that weekday.

## Interface

- Score: ±25, the lowest in the table.
- Confidence: 0.30 + (|t| − 2.0) × 0.08, capped at **0.55**. That ceiling sits
  below the 0.65 `lone_module_minimum_confidence`, so the strongest weekday
  effect on record cannot open a position on its own. This is asserted as a
  test rather than left to arithmetic.
- invalidation_price: **none**. A weekday has no price at which it is wrong,
  and inventing one would put a stop on the chart that no observation supports.
  The setup's stop comes from whichever module actually found the level.
- details: timeframe, weekday, sample size, mean return, t-statistic.

## What it is for

Tipping a setup that two other readers already like. It is the only module here
whose input is not the shape of recent price, so it is the only one that cannot
correlate with the others — which is the whole argument for it, given that the
eight existing modules fire together and lose together.

## What it cannot escape

Five weekdays tested at t = 2 is roughly a one-in-four chance of a spurious
result per instrument, and the same test runs across the whole catalogue. Read
the WHEN PRESENT and ALONE tables together: this module can barely produce a
lone trade by construction, so its value is whether the setups it tips do
better than the same setups without it.

## Not live

Research weight only; absent from `live_enabled_modules`. Of the four families
added together, this is the one expected to be switched off first. See
`docs/hypotheses/seasonality.md`.

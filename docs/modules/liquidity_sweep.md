# Module: liquidity_sweep

## What it measures

A single M15 candle (H1 fallback) that trades through the 20-bar high or low
and closes back inside the range. Sweep depth is measured in ATR.

## Interface

- Score: ±75.
- Confidence: 0.55 + sweep depth in ATR × 0.25, capped at 0.9.
- key_levels: the swept 20-bar low and high.
- invalidation_price: the extreme of the sweep wick.

## Measured edge

Not measured. See `docs/hypotheses/liquidity_sweep.md` — 9 configurations,
plus a mandatory signal-correlation measurement against `market_structure`.

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---:|---:|---:|---:|---:|
| Train (60%) | not run | — | — | — | — |
| Validation (20%) | untouched | — | — | — | — |
| Holdout (20%) | untouched | — | — | — | — |

## Correlation with other modules

**Not yet measured, and this is the most important open item for this module.**
`market_structure` detects equal highs and lows, which are frequently the same
market feature. Two modules voting on one event inflates the confluence score
by double-counting. Measure before either carries live weight.

## Known limitations

- Accepts any sweep depth; a 0.1 ATR wick and a 1.5 ATR raid score the same
  apart from confidence.
- One-candle confirmation is weak. No requirement that the return inside the
  range holds.
- The mechanism justifies a reaction, not necessarily one large enough to pay
  for the spread and the stop the setup requires.

## Weight

**Research 0.8. Live-enabled without measured evidence,** as an accepted
deviation for the EUR 100 experiment.

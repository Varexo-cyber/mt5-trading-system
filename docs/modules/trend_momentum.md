# Module: trend_momentum

## What it measures

EMA(20) vs EMA(50) alignment plus the 5-bar slope of the EMA(20), required to
agree on **both** H4 and H1. Disagreement between the two timeframes returns a
neutral signal rather than a weak directional one.

## Interface

- Score: ±65 when both timeframes agree, 0 otherwise.
- Confidence: 0.5 + EMA separation in ATR × 0.2, capped at 0.9.
- invalidation_price: the extreme of the last 12 H1 bars against the direction.
- details: EMA values, slope, and ATR per timeframe.

## Measured edge

Not measured. See `docs/hypotheses/trend_momentum.md` for the pre-registered
test — 27 configurations, EURUSD/GBPUSD/USDJPY development, AUDUSD/USDCAD held
back.

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---:|---:|---:|---:|---:|
| Train (60%) | not run | — | — | — | — |
| Validation (20%) | untouched | — | — | — | — |
| Holdout (20%) | untouched | — | — | — | — |

## Known limitations

- The momentum literature is strong at monthly horizons on equity and thin at
  H1 on FX. A real edge here is plausible but not established.
- EMAs lag by construction; the signal arrives after the move has begun.
- No regime gate of its own — it relies on `volatility_regime` to be silenced
  in ranges, and that module currently carries weight 0.

## Weight

**Research 1.0. Live-enabled via `config/eightcap.yaml` without measured
evidence,** as an accepted deviation for the EUR 100 experiment. See the
verdict section of the hypothesis.

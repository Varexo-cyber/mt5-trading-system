# Module: market regime router

## What it measures

The classifier measures two context variables from closed H1 and H4 bars:
directional efficiency (net movement divided by total travelled movement) and
the percentile of H1 ATR. It labels the observation `trend_up`, `trend_down`,
`range`, `transition`, or `extreme`.

The classifier is non-directional. It cannot vote long or short. In research
and paper modes the confluence engine uses the label to discount incompatible
evidence to 35% of its base weight. Live routing is disabled pending the
pre-registered out-of-sample test.

## Interface

- Score: always `0`.
- Confidence: `1` when enough valid closed bars exist, otherwise a neutral
  signal with confidence `0`.
- Invalidation price: none; this module does not define an entry or stop.
- Details: regime, both efficiency ratios and directions, ATR and ATR
  percentile.

## Measured edge

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---:|---:|---:|---:|---:|
| Train (60%) | not tested | | | | |
| Validation (20%) | untouched | | | | |
| Holdout (20%) | untouched | | | | |

Configurations tested: 0 of 9.

Deflated Sharpe: not available.

## Parameter stability

Not tested. The registered grid is lookback 20/24/30 crossed with incompatible
weight multiplier 0.20/0.35/0.50.

## Correlation with other modules

The output is context rather than direction, so directional signal correlation
is not the primary test. The required test is incremental paired performance:
the same candidate stream with static weights versus routed weights.

## Known limitations

Efficiency is descriptive and lagging. A market can switch state immediately
after classification. ATR says nothing about direction. Broker CFD stocks have
scheduled gaps and crypto trades continuously; neither is covered by the
initial FX hypothesis and neither may be used to rescue a failed FX result.

## Weight

`0.0`. The module is non-directional. Research/paper routing is enabled;
real-money routing is disabled until validation passes.

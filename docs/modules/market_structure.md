# Module: market structure

## What it measures

The module finds internal and external fractal swing points using only closed
bars. A pivot is not usable until the configured number of bars to its right
has closed. It classifies higher-high/higher-low structure as bullish and
lower-high/lower-low structure as bearish.

A close through the latest external swing is labelled BOS when it continues
the existing direction, CHoCH when it breaks against that direction, and break
when the prior structure is neutral.

Only a BOS aligned with the H4 external direction emits a directional signal.
CHoCH, neutral breaks, internal swings, and equal highs/lows remain diagnostics.

## Interface

- Score: analysis.market_structure.bos_score, positive for bullish and negative
  for bearish. Default magnitude: 70.
- Confidence: starts at the configured minimum and rises with the close distance
  beyond the broken level, measured in ATR.
- key_levels: broken external swing, opposing structural invalidation, then
  detected equal highs/lows.
- invalidation_price: latest opposing external swing. The trade manager must
  place the actual stop beyond it with the configured ATR buffer.
- details: H1 internal/external direction, H4 bias, event type, ATR, and counts
  of confirmed swings and equal levels.

## Visual verification

Synthetic plumbing example:

    python scripts/plot_market_structure.py --output market_structure_demo.png

Broker data requires UTC time, open, high, low, and close columns:

    python scripts/plot_market_structure.py --csv eurusd_h1.csv --timeframe H1 --output eurusd_h1_structure.png

External swings show the pivot bar and the later bar on which it became
knowable. This distinction is the visual look-ahead check.

## Measured edge

Not measured. The module remains weight 0 until the pre-registered test in
docs/hypotheses/market_structure.md can run on bid/ask-aware historical data.

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---:|---:|---:|---:|---:|
| Train (60%) | not run | — | — | — | — |
| Validation (20%) | untouched | — | — | — | — |
| Holdout (20%) | untouched | — | — | — | — |

Configurations registered: 9  
Deflated Sharpe: not calculated

## Parameter stability

Not measured. Registered sweep: external fractal lookback 2/3/4 and BOS close
buffer 0.00/0.05/0.10 ATR. No additional variant may be inspected without
incrementing the configuration count before seeing its result.

## Correlation with other modules

Not available because no other analysis module carries weight. Equal highs/lows
are deliberately descriptive here so a future liquidity-sweep module cannot
turn the same market event into two independent votes.

## Known limitations

- Fractals lag by their right-hand confirmation window; removing that delay
  would introduce look-ahead.
- Structure is not uniquely defined. Results must be stable across the
  pre-registered lookback range.
- Wick-only breaks are ignored.
- CHoCH is not a reversal trigger.
- XAUUSD is outside the registered FX hypothesis.
- A structural invalidation level is not yet an executable stop; the ATR buffer,
  spread, and broker stop distance belong to trade management.

## Weight

**0 — awaiting the registered out-of-sample validation.** No performance claim
has been made.

# Walk-forward index reversal

## Mechanism

The model identifies an H1 SPX500 state where recent returns, EMA displacement,
bar geometry, volatility, volume and session time jointly predict continued
movement over the next twelve bars. Validation selected the opposite polarity:
fade that crowded forecast. The counterparty is the late directional trader
paying for immediacy after an extended index move; the trade supplies that
liquidity and targets partial reversion.

## Pre-registered live form

- Symbol: SPX500 only
- Clock: H1
- Frozen model; no live retraining
- Absolute model threshold: 0.075
- Stop: 1 ATR
- Target: 1.5R
- Account: EUR 203, normal 2% ceiling and broker minimum-lot rules

Validation chose this form before the newest quarter was opened. Holdout:
92 trades, 44.6% wins, +0.074R per trade and EUR +12.02 after historical
spread and configured costs. The exact Jarvis replay with break-even management
returned 83 trades, 80.7% winning exits, +4.64R and EUR +15.20 over 45 days.

## Falsification

The live breaker stops the section after at least twenty trades if more than
70% lose, or after eight consecutive losses. The current evidence is thin:
83 live-shaped trades, +0.74 sigma, with July carrying 90% of the result. This
is an experimental positive edge, not a profit guarantee.

# Hypothesis: m1_micro_breakout

## Claim

On a liquid market whose closed M5 bars already show directional structure, a
fresh closed M1 breakout from a compact consolidation, with a meaningful body
and above-normal tick activity, is followed often enough by a short continuation
to justify a cost-aware quick trade.

This is a discrete event. Price remaining beyond the range is not a new signal.
The forming M1 candle is never used.

## Economic rationale

A short consolidation after directional movement is a temporary balance. A
close outside that balance with participation shows that aggressive orders have
overwhelmed the resting liquidity at its edge. Stops belonging to traders on
the wrong side and continuation orders can fund the next short leg.

The counterparty is the trader fading the first break of a genuinely directional
micro-range. The thesis is wrong when the apparent range is merely random M1
noise or when the break runs directly against established M5 structure.

## Pre-registered rules

- Context: M5 EMA9/EMA20 separation and EMA20 slope must agree with the trade.
- Base: the M1 bars immediately before the breakout must form a compact range.
- Break: the latest closed M1 candle must close beyond that range.
- Quality: its body must be meaningful relative to M1 ATR, close near the
  directional end of its own range, and tick volume must exceed its recent
  median.
- Freshness: only the first closed bar beyond the range can signal.
- Invalidation: beyond the opposite edge of the pre-break range plus an ATR
  buffer.
- Existing spread, cost, session, news, sizing, correlation, risk and AI gates
  remain unchanged.

## Predicted failure

False breaks during thin liquidity and rapid mean reversion. Losses should
cluster when M5 EMA separation is small, the M1 base is too wide, or execution
cost consumes a large part of the planned stop. The response is to strengthen
those measured quality conditions, never to widen risk or average down.

## Validation plan

- Development: EURUSD, GBPUSD, USDJPY, XAUUSD.
- Held back: AUDUSD, USDCAD, US500, BTCUSD.
- Oldest 60% train, next 20% validation, final 20% untouched.
- Minimum conclusion size: 100 trades.
- Compare with direction-shuffled entries at the same symbols and times.
- Report all parameter variants and cost assumptions.

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---:|---:|---:|---:|---:|
| Train (60%) | not run | — | — | — | — |
| Validation (20%) | untouched | — | — | — | — |
| Holdout (20%) | untouched | — | — | — | — |

Experimental-live observations remain research evidence, not proof of an edge.

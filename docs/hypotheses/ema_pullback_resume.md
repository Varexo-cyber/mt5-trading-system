# Hypothesis: ema_pullback_resume

## Claim

On a liquid market with a clearly separated 9/20 EMA trend on M5, a shallow
pullback into the fast/slow EMA band followed by a closed candle reclaiming the
fast EMA in the trend direction is followed by enough continuation to support
a short, cost-aware trade.

This is an **entry event**, not permission to buy every bar above an average or
sell every bar below one. A new proposal exists only after price first enters
the EMA band and then leaves it again in the established direction.

## Economic rationale

The trend leg represents active directional flow. The pullback represents
temporary opposing flow or profit taking. If price cannot remain through the
slow EMA and is reclaimed in the original direction, the countertrend orders
have failed to change the short-term auction. Their exits, plus continuation
orders re-entering after the pullback, can fund the next leg.

The counterparty is the trader treating the first pullback as a reversal. The
claim is wrong when the market is actually rotating in a range: both averages
flatten, price crosses the band repeatedly, and every apparent reclaim merely
pays another spread.

## What it adds

`fast_ema_cross` only speaks near the original cross. `impulse_break` only
speaks after an unusually large bar. `drift_continuation` needs an extended
sequence of aligned M15 closes. None of them can re-enter an existing M5 trend
after an ordinary pullback. This module is intended to create that missing
repeatable quick-entry setup without weakening those detectors.

## Pre-registered rules

- Trend: EMA9 is separated from EMA20 by at least 0.12 M5 ATR.
- Trend slope: EMA20 must still slope in the proposed direction.
- Pullback: within the last four closed M5 bars, price must touch or enter the
  EMA9/EMA20 band without closing materially through EMA20.
- Resume: the latest closed M5 bar must close back beyond EMA9 in the trend
  direction and have a body in that direction.
- Timing confirmation: when M1 history exists, its latest three-bar drift may
  not materially oppose the proposal.
- Invalidation: beyond the pullback extreme and beyond EMA20 by a small ATR
  buffer, whichever is farther away.
- Freshness: one reclaim event may create one proposal. Remaining above/below
  the average is a state, not repeated permission to enter.

## Predicted failure

Whipsaw in low-efficiency ranges. Losses should cluster where the EMA band is
flat or price has crossed EMA20 repeatedly in the preceding hour. If that is
observed, the correct response is a stronger trend-efficiency or recross floor,
not a wider stop and not averaging down.

The second failure is cost. A technically valid M5 setup can still have a stop
and target too small to pay spread, commission and slippage. Existing cost,
spread, sizing, target-reach and AI gates remain binding.

## Validation plan

- Development instruments: EURUSD, GBPUSD, USDJPY, XAUUSD.
- Held back: AUDUSD, USDCAD, US500, BTCUSD.
- Split: oldest 60% train, next 20% validation, final 20% untouched.
- Sweep: EMA pairs, minimum separation, pullback window and invalidation
  buffer. Report the number of configurations.
- Minimum conclusion size: 100 trades.
- Compare against the same timestamps with direction shuffled and against the
  existing `fast_ema_cross` alone.

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---:|---:|---:|---:|---:|
| Train (60%) | not run | — | — | — | — |
| Validation (20%) | untouched | — | — | — | — |
| Holdout (20%) | untouched | — | — | — | — |

Live use before this protocol is experimental and must remain attributable as
`ema_pullback_resume` in the journal. It is not evidence of an edge merely
because it creates more proposals.

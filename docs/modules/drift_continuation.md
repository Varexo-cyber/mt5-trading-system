# Module: drift_continuation

## What it measures

Net close-to-close movement over the last 8 M15 bars, expressed in ATR(14), and
the share of those bars that closed in the direction of that movement. Both
must clear their floors: 1.0 ATR travelled and 65% of bars agreeing.

The second term is what separates this from "price is different than it was".
A market that ends the hour lower having gone up, down, up and down has the
same net drift as one that ground steadily lower, and only the second is going
anywhere. Distance alone cannot tell them apart.

## Interface

- Score: ±55, below `market_structure` and `trend_momentum` at ±65 and
  `liquidity_sweep` at ±75. A drift says the move happened, not that it
  continues.
- Confidence: 0.45 + (travelled/2.0, capped at 1) × consistency × 0.40, capped
  at 0.85. Both terms multiply because they fail in different ways — a big move
  nobody sustained is a spike, and a perfectly consistent move of no size is
  noise with a tidy shape.
- invalidation_price: the extreme of the window against the direction. Price
  back through where the drift started means the thing this is built on is gone.
- details: timeframe, lookback, drift in ATR, consistency, ATR.

## Why it exists

12 August 2026. GBPUSD declined for most of the session; the engine produced
344 refusals reading "price is moving against the long" and never once proposed
a short. No gate stopped one — no module was looking. `trend_momentum` runs
20/50 EMAs on H4 and H1 and so falls silent well before a market turns and
speaks again long after the new move began; `liquidity_sweep` needs a wick
through a 20-bar extreme on the last candle and is a reversal pattern anyway.
An hour of clean one-way drift fell between them.

## What stops it trading chop

This is the module's whole risk: a detector that fires on "price moved" fires
constantly in a range, and alternately buying and selling a sideways market
empties an account through the spread rather than through being wrong.

1. The consistency floor, above.
2. Registration in `confluence.trend_continuation_modules`, which makes the
   confluence engine refuse it outright when `market_regime` measures a range.
3. `entry_quality` downstream, which refuses an entry sitting at the extreme of
   its recent range — the gate that stops it selling the low of the move it has
   only just noticed.

## Measured edge

Not measured. See `docs/hypotheses/drift_continuation.md` for the
pre-registered test — EURUSD/GBPUSD/USDJPY/XAUUSD development,
AUDUSD/USDCAD/US500/BTCUSD held back, `lookback_bars`, `minimum_drift_atr` and
`minimum_consistency` swept.

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---:|---:|---:|---:|---:|
| Train (60%) | not run | — | — | — | — |
| Validation (20%) | untouched | — | — | — | — |
| Holdout (20%) | untouched | — | — | — | — |

Live from 12 August 2026 on the owner's explicit instruction, ahead of that
protocol, alongside the three other modules in the same position. What will
grade it first is the live record rather than the backtest:
`decisions.signals` now records which module found each trade, and
`Brain.module_records()` reports realised R per detector once it has twenty of
them.

## Known limitations

- **The predicted way it loses is entering late.** By the time eight bars have
  confirmed a move, much of it may be spent. If the losses cluster at entries
  near the extreme of the window, that is this failing exactly as expected and
  the answer is a shorter window or a stricter `entry_quality`, not a wider
  stop.
- Short horizon, so the round trip is a larger share of the move than it is for
  the swing modules. On instruments whose spread is a meaningful fraction of an
  M15 ATR the cost gates will refuse most of what it finds, correctly.
- Around scheduled news a drift is a repricing that has already finished rather
  than an order still being worked, and the mechanism in the hypothesis does
  not apply. The news blackout covers the window before a release; it does not
  cover unscheduled moves.
- The mechanism has a natural expiry — when the parent order is filled the
  drift stops — and nothing here estimates how much of it is left.

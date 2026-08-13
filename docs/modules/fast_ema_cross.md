# Module: fast_ema_cross

## What it measures

An EMA(9) / EMA(20) cross on M5, subject to three conditions that all have to
hold on the current bar:

1. **Freshness** — the cross happened at most 3 bars ago. Beyond that it is a
   state, and the state is what `trend_momentum` already reports; treating it
   as an entry means entering the same move repeatedly.
2. **Separation** — the averages are at least 0.15 ATR apart. Two EMAs sitting
   on top of each other crossing back and forth is one market making no
   decision, and reading each touch as a signal is reading noise at high
   frequency.
3. **Price side** — the last close is on the signal side of the slow average.
   A cross with price already back through it is a cross that failed in the
   time it took to print.

## Interface

- Score: ±50, the lowest in the engine. Fastest and least corroborated
  evidence.
- Confidence: 0.45 + separation in ATR × 0.50, capped at 0.80.
- invalidation_price: the further of (a) the extreme printed since the cross,
  over at least `minimum_invalidation_bars`, and (b) the slow EMA plus
  `invalidation_buffer_atr`. The thesis dies when price closes back through
  that average — the module's own third floor — so that is where the stop
  belongs; the buffer keeps a single wick through it from counting as a close,
  and taking the further of the two keeps the stop out of a wick that has
  already printed.
- details: timeframe, bars since the cross, separation in ATR, the invalidation
  window and its distance in ATR, both EMA values, ATR.

### A note on the weight, which was documented wrongly

The 0.5 weight was described here and in config as making the module "need help
to clear the threshold". It does not. A lone module's confluence score is
`|raw score| × confidence`: the weight appears in both the numerator and the
denominator of the weighted average and cancels. The weight only matters when
several modules fire together. What actually gates a lone cross is the
threshold against `50 × confidence`, which is why `score_threshold` had to come
down to 26 before a cross at this module's own 0.15 ATR floor could trade at
all.

## Why it exists

Every other directional module reads a slow chart — 20/50 EMAs on H4 and H1, a
break of structure, or at the fastest a specific wick on M15. Nothing looked at
the timeframe a day trade is actually entered on, so intraday opportunity was
invisible to the engine even while it was measuring the move.

## What stops it trading noise

The three floors above, and then two things outside the module:

- Registration in `confluence.trend_continuation_modules`, so the engine
  refuses it outright when `market_regime` measures a range — the condition
  under which a fast cross is at its worst.
- Registration in `confluence.intraday_modules`, so the plan it produces uses
  M15 planning authority and a target twelve bars out, rather than a swing
  target twenty-four hours away on a five-minute signal.

`entry_quality` also still refuses an entry at the extreme of its own range.

## Measured edge

Not measured. See `docs/hypotheses/fast_ema_cross.md` for the pre-registered
test — EURUSD/GBPUSD/USDJPY/XAUUSD development, AUDUSD/USDCAD/US500/BTCUSD held
back, EMA periods, freshness window and separation floor swept.

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---:|---:|---:|---:|---:|
| Train (60%) | not run | — | — | — | — |
| Validation (20%) | untouched | — | — | — | — |
| Holdout (20%) | untouched | — | — | — | — |

Live from 12 August 2026 on the owner's explicit instruction, ahead of that
protocol. `Brain.module_records()` grades it from real fills at twenty trades.

## Known limitations

- **The predicted way it loses is the whipsaw**: a cross, an entry, a reversal
  inside the hour. If losses cluster at trades held under thirty minutes with
  the exit on the opposite side of the entry, the separation floor is too low —
  the answer is a wider floor, not a wider stop.
- The most widely watched signal in retail trading, which cuts both ways.
  Whatever edge exists is in the filtering, not in the cross.
- Highest signal rate in the engine by a wide margin, so it will dominate the
  review queue unless the ranking terms hold it back. Worth watching in
  `why_no_trades` after the first full day.
- The price-side check is nearly unreachable behind the other two: any move
  violent enough to put price back through the slow average within three bars
  inflates ATR and trips the separation floor first. Kept as a guard on a real
  state that a synthetic series cannot easily produce, and its test relaxes the
  separation floor deliberately so the branch actually runs.

# Module: impulse_break

## What it measures

Over the last `max_bars_since + 1` closed M15 bars it finds the largest body and
asks four questions of it:

1. **Size** — is the body above `minimum_body_atr` (1.00)? The body, not the
   range: a wide bar with a small body is indecision wearing a big candle.
2. **Conviction** — did it close in the far `minimum_close_location` (66%) of its
   own range? A large body with a large opposing wick is a rejection, and
   joining a rejection means joining the reversal of the move you meant to join.
3. **Freshness** — was it within `max_bars_since` (2)? The mechanism concerns the
   minutes after the repricing; an hour later the liquidity is back.
4. **Still held** — has price given back no more than `maximum_retracement`
   (50%) of the move since? Half of it retraced means the market rejected the
   break and the spike was the whole story.

## Interface

- Score: ±60. Above `drift_continuation` at ±55 because one bar demonstrably
  travelled a long way — a fact rather than an indicator relationship — and
  below `market_structure` at ±65 because it is not a confirmed break.
  Chosen so the weakest signal it can emit still clears the live threshold
  alone: 60 × 0.45 = 27 against 26. That arithmetic is not optional;
  `fast_ema_cross` shipped at 50 against a 35 threshold and could never trade
  on its own, which took a live funnel to notice.
- Confidence: 0.45 + (body in ATR − 1.00) × 0.25, capped at 0.80.
- invalidation_price: the **open** of the impulse bar. Price back through where
  the repricing started means it did not hold, which is precisely the claim.
- details: timeframe, body in ATR, bars since the impulse, close location, how
  much has been given back, the impulse origin, ATR.

## Registration

- `intraday_modules` — a three-hour thesis must not be handed H1 planning
  authority and a target a day out.
- **NOT** `trend_continuation_modules`. It measures a move rather than inferring
  a trend, so a range on H1 does not contradict it. See the note on that tuple
  for the 6,726 refusals the wrong classification cost.

## Measured edge

Not measured. See `docs/hypotheses/impulse_break.md` for the pre-registered
test and the two predicted failure modes.

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---:|---:|---:|---:|---:|
| Train (60%) | not run | — | — | — | — |
| Validation (20%) | untouched | — | — | — | — |
| Holdout (20%) | untouched | — | — | — | — |

## Known limitations

- **The obvious way it loses is buying the top of the spike.** If losses cluster
  at entries near the impulse extreme with immediate reversal, the mechanism is
  wrong and the answer is a stricter `maximum_retracement`, not a wider stop.
- A news release produces this exact bar shape and the mechanism does not apply
  — the repricing has already finished. The blackout covers scheduled releases
  only.
- One bar is one observation. This is the least corroborated evidence in the
  engine after `fast_ema_cross`, and it is weighted accordingly at 0.6.
- It says nothing about how much of the move is left.

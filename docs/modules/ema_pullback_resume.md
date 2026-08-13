# Module: ema_pullback_resume

## Purpose

Find a repeatable quick entry inside an existing M5 trend. Unlike
`fast_ema_cross`, it does not need the trend to have started recently. It waits
for an ordinary pullback into the EMA9/EMA20 band and speaks only when a closed
M5 candle reclaims EMA9 in the trend direction.

## Inputs and decision

- Closed M5 bars determine trend, pullback, reclaim and invalidation.
- EMA9/EMA20 separation and EMA20 slope must confirm an active trend.
- A recent bar must touch the EMA band without closing materially through the
  slow EMA.
- The latest bar must form a discrete reclaim with a confirming candle body.
- Closed M1 bars, when available, may veto materially opposing immediate flow.
- Remaining above/below EMA9 after the reclaim does not emit another signal.

## Output

- Score: ±58.
- Confidence: base 0.45 plus bounded contributions from EMA separation and the
  reclaim body, capped at 0.80.
- Invalidation: beyond both the pullback extreme and EMA20 plus a 0.20 ATR
  buffer.
- Horizon: `quick`, planned on M5 over six bars (roughly thirty minutes).
- Setup family: `ema_pullback_resume_m5`.

## What remains downstream

This detector only creates a proposal. Target reach, spread and commission,
entry quality, session/news filters, sizing, margin, portfolio limits,
post-review price revalidation and the AI veto all remain binding. It does not
guarantee a trade every five minutes and it never changes risk after a loss.

## Known failure

EMA-band whipsaw in ranges. The detector requires slope and separation but
these are imperfect. Attribute every result to this module and disable it if
the registered validation fails or the live record is negative after an
adequate sample; do not repair it by widening stops.

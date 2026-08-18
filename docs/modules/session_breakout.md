# Module: session_breakout

## What it measures

On M15, in the broker's own server clock:

1. **The range** built between `range_start_hour` (00:00) and `range_end_hour`
   (07:00) of the most recently CLOSED such window. Around midnight that is
   yesterday's; using today's date unconditionally would compare a break
   against a window that has not happened yet.
2. **Freshness** — the current bar must be within `breakout_window_hours` (4h)
   of that window closing. A break at 15:00 of a range that closed at 07:00 is
   an afternoon move, not that session resolving.
3. **Width** — the range must be between `minimum_range_atr` (0.75) and
   `maximum_range_atr` (4.0). A five-pip overnight range is a dead market and
   breaking it says nothing; a range wider than the day's usual travel has
   already had its move.
4. **The break** — the close must finish `breakout_close_share` (10% of the
   range) beyond an edge, so a wick is not a break.

## Interface

- Score: ±50.
- Confidence: flat at `base_confidence` (0.45). No scaling, deliberately: there
  is no continuous quantity here that the hypothesis says should scale
  confidence. A break is a break.
- invalidation_price: the **far** side of the range. The near edge is inside
  the range's own noise.
- key_levels: the range top and bottom.
- details: timeframe, both edges, range in ATR, the session window used, ATR.

## The clock

Configured, never inferred. Bars are stamped in broker server time, and a server
three hours off UTC would silently shift every session by three hours with
nothing looking wrong — the ranges would still be ranges and the breaks would
still be breaks. Check `range_start_hour`/`range_end_hour` against the broker's
actual clock before reading anything into a backtest of this module.

## Registration

- `intraday_modules`.
- **NOT** `trend_continuation_modules`.

## Not live

Research weight only; absent from `live_enabled_modules`. See
`docs/hypotheses/session_breakout.md`.

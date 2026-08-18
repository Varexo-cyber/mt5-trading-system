# Module: volatility_squeeze

## What it measures

Over the M15 chart it asks three questions in order, and stops at the first no:

1. **Is it coiled?** The high-low range of the twelve bars BEFORE the last one,
   ranked against the same measurement rolled over the previous 200 bars. It
   must sit in the lowest `compression_percentile` (25%) of that history.
   Ranked, not measured in pips: a fifteen-pip range is tight on gold and wide
   on EURCHF, and a threshold in price would mean a different thing on every
   symbol in the catalogue.
2. **Has it released?** The last bar's range as a multiple of the compression.
   Below `expansion_multiple` (1.8x) it is not an expansion, it is the next
   quiet bar.
3. **Did it resolve?** The close must finish `breakout_close_share` (25% of the
   compressed range) beyond the coil's edge. A wick through the boundary that
   closes back inside is an expansion that resolved nothing.

The compression window deliberately EXCLUDES the breakout bar. Including it is
the classic way to make this signal look better than it is: the expansion
widens the range it is being compared against, and the test grades itself.

## Interface

- Score: ±55. Between `impulse_break` (60) and `drift_continuation` (55) — one
  bar demonstrably repriced, which is a fact, but the claim about what came
  before it is an inference. At the live threshold of 26 the weakest signal it
  can emit clears alone: 55 × 0.50 = 27.5.
- Confidence: 0.50 + (expansion − 1.80) × 0.10, capped at 0.85.
- invalidation_price: beyond the **far** side of the compression, plus 0.25 ATR.
  The near edge sits inside the noise the coil is made of and the breakout bar
  has already traded through it. This makes the stop wide and the position
  small, which is the honest price of the pattern.
- key_levels: the coil's top and bottom.
- details: timeframe, compression percentile, compressed range, expansion
  multiple, both edges, ATR.

## Registration

- `intraday_modules` — the claim is about the hour or two after the break, not
  about a trend, so it must not be handed H1 planning authority.
- **NOT** `trend_continuation_modules`. It measures a release from compression
  rather than inferring a trend, so a range on H1 does not contradict it — the
  range IS the setup.

## Not live

Carries a research weight so `scripts/backtest_modules.py` measures it; absent
from `live_enabled_modules`, so the confluence engine zeroes its weight in live
mode. See `docs/hypotheses/volatility_squeeze.md`.

# Module: mean_reversion

## What it measures

On M15, over a 48-bar window:

1. **The mean and standard deviation** of the closes.
2. **The extreme** of the last `max_bars_since_extreme + 1` (4) bars, in
   standard deviations from that mean. Both sides are measured; the larger wins
   and sets the direction, which is the OPPOSITE of the extreme.
3. **The stall** — how far the last close has come back off that extreme,
   measured in ATR, against `stall_retrace_atr` (0.35).

The extreme is taken over recent bars rather than requiring the LAST bar to be
extreme. Requiring that would be requiring the fade not to have started, which
is precisely the knife-catch this module exists not to be.

The stall is measured in ATR and not as a share of the extreme bar's own range.
It was written the second way first and a test caught it: a bar spanning a tenth
of an ATR whose close sits halfway down it reads as "50% retraced" while price
has come off the high by a twentieth of an ATR — nothing at all. The module
would have faded markets that were still running.

## Interface

- Score: ±50. Below every continuation reader, because the entire family is
  unproven on this account: `liquidity_sweep` is the only non-negative module
  measured so far and its t is 0.59, which is noise. At the live threshold of
  26 the weakest signal clears alone at 50 × 0.45 = 22.5 — it does NOT, and
  that is deliberate for now.
- Confidence: 0.45 + (stretch − 2.20) × 0.12, capped at 0.80.
- invalidation_price: beyond the extreme itself, plus 0.35 ATR. Price through
  the extreme means the stretch was not the end of the move, which is the claim
  failing in the most direct way available.
- key_levels: the extreme and the mean.
- details: timeframe, stretch in SD, retracement in ATR, mean, standard
  deviation, extreme, ATR.

## Registration

- `intraday_modules`.
- **NOT** `trend_continuation_modules`, and this matters more here than
  anywhere: it is a contradiction reader by construction, so the regime
  discount that suppresses continuation claims in a range must not touch it.

## Not live

Research weight only; absent from `live_enabled_modules`. See
`docs/hypotheses/mean_reversion.md`.

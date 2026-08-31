# Hypothesis: order_block_fast

## What this document is, and is not

`order_block` on M1 instead of M30. Same detector, same thresholds, same target
ratio, same stop. **Only the clock differs.**

The mechanism is unchanged and is not restated here — read
[`order_block.md`](order_block.md). What is new is the clock, and this
document exists to say honestly on what basis that clock was chosen, because
it is a weaker basis than anything else trading real money on this account.

**This is not pre-registration.** Every other hypothesis document here was
written before the measurement. This one is written after, about a cell that
was picked out of a sweep because it was the best-looking row. That is the
definition of the thing this project's own rules exist to prevent, and the
only honest thing to do is say so at the top rather than dress it up as a
prediction.

## What was actually measured

14 days to 31 August 2026, core universe, break-even exit (what the account
runs), five markets — the other eleven were refused on cost before a bar was
walked:

| section | clock | trades | win | LIVE R | LIVE EUR | sigma |
|---|---|---|---|---|---|---|
| impulse_retest | M1 | 78 | 73% | +3.60 | −6.46 | +0.62 |
| impulse_retest | M5 | 73 | 70% | −12.40 | −81.68 | −3.43 |
| **order_block** | **M1** | **105** | **88%** | **+34.00** | **+121.97** | **+4.28** |
| order_block | M5 | 308 | 75% | −15.20 | −54.74 | −1.11 |

## Why that number is weaker than it looks

Four reasons, all of which were on the screen when the decision was taken.

1. **It is thin.** 105 trades, against the report's own 200-trade bar. Thirteen
   daily observations is a poor basis for the standard error that +4.28 is
   computed from, and daily R totals at an 88% win rate are heavily skewed —
   the normal approximation flatters a distribution shaped like that.

2. **M5 says the opposite.** Same section, same fourteen days, same five
   markets, one clock away: −15.20 R over 308 trades, −1.11 sigma. Adjacent
   clocks with opposite signs, and the one that looks good has a third of the
   trades. If the M1 edge were real and mechanical, M5 should be a diluted
   version of it, not an inversion.

3. **The run did not charge the trades their spread.** `_resolve` walked raw
   highs and lows; the cost model refused setups and then paid the survivors as
   if trading were free. On M1 the round trip is up to 12% of the stop (the
   sizer's own `max_cost_share_of_risk`), so roughly 10 R of that +34.00 is
   measurement. Fixed on 31 August; **the +121.97 predates the fix and has not
   been re-measured.**

4. **Five markets.** XAUUSD, US30, NDX100, SPX500, GER40. The eleven FX pairs
   the research was done on were all skipped — a round trip is 78% to 194% of
   an M1 stop on those. So this is an extrapolation of an FX finding onto
   indices and gold, on the clock where the extrapolation is furthest.

Points 2 and 3 push the same way: a plausible explanation of the whole result
is that M1 gets a free entry and a free exit relative to a stop one M1 ATR
wide, and the smaller the stop the larger that gift.

## Why it is live anyway

The owner's decision, made twice in explicit terms after each of the four
points above was put to him: *"risico's moeten genomen worden om te testen"*.
The account exists to find out, the alternative to a live test is another
offline number of the kind that has now been wrong five times, and EUR 216 is
what he is prepared to learn on.

That is the actual reason. It is not that the evidence cleared a bar.

## What would settle it

`sweep.cmd 14 M1 M5` re-run with the spread charged. If M1 is still positive in
euros after the haircut, this stops being a bet. If it is not, this module comes
off.

## Falsification

The strictest breaker on the account, and deliberately stricter than the M30
copy it is cloned from:

| | order_block (M30) | order_block_fast (M1) |
|---|---|---|
| window | 40 | **30** |
| minimum trades | 40 | **30** |
| maximum loss share | 0.55 | **0.50** |
| losing streak | 9 | **7** |

At the measured 7.5 trades a day that is four days of evidence rather than six.
If M1 is not what those fourteen days suggested, this section switches itself
off before any other one would.

Manual falsification, in order of how quickly it should be acted on:

1. Win rate under 50% over 30 trades — the claim is 88%, and 50% is far enough
   below it that the fourteen days were noise.
2. Negative in euros over 30 trades while positive in R. That is the
   minimum-lot override putting more money on the losers than the winners, and
   it is what already happened to `impulse_retest` on M1 (+3.60 R, −6.46 EUR).
3. The re-run with the spread charged coming back under +0.02 R a trade.

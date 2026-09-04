# Section eleven — gold-cross regime continuation

## Status

Fresh hypothesis, 4 September 2026. The previous random-feature model is
rejected: its untouched holdout lost on XAUEUR, XAUGBP, XAUAUD and XAUJPY.
Those fitted coefficients and their forced-live exception are not evidence for
this hypothesis and must not be reused.

## Mechanism fixed before the search

A gold cross has two moving parts: the dollar gold leg and the quote-currency
leg. A continuation should be most reliable when a closed bar breaks or
rejects a recent level while its medium-term trend already points the same
way. The other side consists of traders fading a move before both legs have
finished repricing. Entries use only closed bars and occur at the next bar's
open.

The bounded candidate set contains trend pullback, channel breakout, aligned
cross momentum, trend rejection, volatility-contraction breakout, and two-leg
trend/impulse agreement. These are seven
different mechanisms, not hundreds of tiny threshold variations. They are
tested on M1, M5, M15, M30 and H1 with one stop/target definition shared by all
four markets. The two-leg candidates require the XAUUSD contribution and the
quote-currency contribution to point in the same gold-cross direction; this is
the mechanism specific to these instruments rather than another chart shape.
Each mechanism is tested in its natural direction and with its direction
reversed. The reversed cell pays as a separate trial; it represents liquidity
providers fading a completed move rather than pretending the sign was chosen
in advance.

A final distinct candidate is an M5 momentum fair-value-gap retest aligned
with an H1-equivalent EMA20/EMA50 trend. It is based on the mechanism described
in Teghjot Bindra, *Multi-Asset A/A+ Momentum Fair Value Gap Strategy* (SSRN
6725880, 2026). Its published result is not treated as evidence here: the
broker, crosses, costs and test window differ, so it must clear the exact same
local protocol.

## Data and synthetic limitation

The local broker archive contains XAUUSD and EURUSD, GBPUSD, AUDUSD and USDJPY,
but not the four quoted gold crosses. Research bars are therefore reconstructed
as XAUUSD divided by the USD-quoted FX leg, or multiplied by USDJPY. High and
low use conservative component extremes. A synthetic bar cannot reproduce the
broker's executable cross spread, so this stage may reject a mechanism but may
not promote one by itself.

## Locked evaluation protocol

- The oldest 50% selects one mechanism and timeframe for the whole basket.
- The next 25% is validation and the newest 25% is an untouched holdout.
- One position per market; no overlapping entries or averaging.
- Signal on a completed bar, fill at the next bar's open.
- Stop is 1 ATR. Targets of 0.5, 0.75, 1.0, 1.5 and 2.0 R are separate
  search cells and all count toward the multiplicity penalty. A 0.02 R round
  trip cost is charged to every result.
- Each rule is tested either all day or in one of four non-overlapping UTC
  regimes: Asia 00-07, London 07-13, New York 13-20 and late 20-24. Every
  regime is a separate search cell; hours are never removed after inspecting
  their loss curve.
- If stop and target occur in one bar, the stop wins.
- Unresolved positions close at the fixed horizon; no disappearing losers.
- Selection pays a Bonferroni penalty for every mechanism/timeframe cell.
- Promotion requires positive validation and holdout, positive holdout on every
  market, both holdout time-halves positive, at least 200 holdout trades and at
  least +2 day-clustered sigma on the holdout.

## Promotion rule

The pooled basket is tested first. If it fails, each cross may select its own
single mechanism/timeframe from the same bounded grid, paying its own full
multiplicity penalty. A passing market becomes a separate section so its
breaker, clock and evidence cannot be hidden by another market.

Even if a synthetic per-market test clears every line, that section remains
shadowed until the frozen rule clears a replay on the actual broker cross with
real spreads and execution gates. Failure leaves it off the real-money
allowlist. No force flag can change that conclusion.

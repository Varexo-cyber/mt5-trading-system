# Section 16 — BTCUSD M5, adaptive channel, inverted, weekdays

> Promotion record, the shared caveat and the containment:
> [btcusd_discoveries.md](btcusd_discoveries.md). Read that first.

## Claim

The same channel reading as section 15, with a far lower strength threshold,
on a five-minute clock, all day, on weekdays only.

## The rule, exactly

Per closed M5 bar, Monday to Friday, no session window:

* `fast` = EMA20, `slow` = EMA50
* `strong` = `|fast − slow| / ATR > 0.25` (the `_25` in the mechanism name)
* `ceiling` / `floor` = previous 20 bars' extreme
* strong + break up → natural long, strong + break down → natural short,
  swapped when not strong
* `polarity: -1` inverts it

Stop 6.0 ATR, target 2.0 R, **no break-even move** — the broker stop stands.

## What separates it from section 15

Only two numbers: the strength threshold (0.25 against 1.00) and the clock.
That is worth saying plainly, because two sections that differ in two
parameters are not two independent pieces of evidence. If both work, that is
one finding measured twice; if one fails, it says something about the other.
`_clock_overlap` in the dry run is what answers whether they take the same
move.

## Weekdays only, and why that is a real restriction

BTCUSD trades the weekend and this section refuses it. The search found the
weekend worse; the mechanism has no story for why. Treat it as a fitted
parameter, not a thesis.

## The unmanaged stop

`shadow_break_even_at_r: 0.0` means nothing moves the stop after entry. On a
6.0 ATR stop, a trade that goes 1.9 R in favour and turns around gives all of
it back and then some. That is a deliberate part of what was measured — adding
a break-even move now would make the live result a measurement of a different
strategy — but it is the largest single difference between this section and
everything else the account runs.

## What would take it off

The section breaker. Beyond that: if `_clock_overlap` shows section 15 and 16
taking the same move more than half the time, running both is doubling the
stake rather than doubling the chances, and one of them should go.

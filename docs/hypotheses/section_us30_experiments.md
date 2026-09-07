# US30 — four experimental sections

> Written **before** the replay runs, which is the only order in which this
> document is worth anything. Shared record for
> `section_us30_impulse_m1`, `section_us30_impulse_m5`,
> `section_us30_orderblock_m1` and `section_us30_orderblock_m5`.

## What is being asked

Two mechanisms this account has already measured, on a market and two clocks
they were **never** measured on.

| section | mechanism | clock | market |
|---|---|---|---|
| `section_us30_impulse_m1` | impulse retest | M1 | US30 |
| `section_us30_impulse_m5` | impulse retest | M5 | US30 |
| `section_us30_orderblock_m1` | order block | M1 | US30 |
| `section_us30_orderblock_m5` | order block | M5 | US30 |

All four are **shadow**: enabled so a replay can judge them, weight 0.0, and
absent from `live_enabled_modules`. No existing section was changed.

## The detectors are called, not copied

`ImpulseRetest` and `OrderBlock` are constructed with US30's own config and
asked the same question sections two and three ask them. Duplicating the
detection logic would produce a second implementation that drifts from the
measured one the first time either is touched — the single most repeated
defect in this repository, and the reason `analysis/mechanisms.py` and
`analysis/legs_gap.py` exist.

What `analysis/section_us30.py` adds is only what is new: the US30 gate, one
entry per setup, and a named refusal on every path.

## The parameters are the stock defaults, and that is the caveat

Every number in the brief is already the default:

| | impulse retest | order block |
|---|---|---|
| ATR period | 14 | 14 |
| lookback | 96 | 96 |
| minimum impulse | 1.0 ATR | 1.5 ATR |
| impulse span | 1.5 ATR | 1.5 ATR |
| tolerance | 0.15 ATR | 0.25 ATR |
| stop beyond | 0.85 ATR | 1.0 ATR |
| block search | — | 5 bars |

Deliberate: the question is what the **measured** mechanisms do on a new
market and two new clocks, and changing the numbers at the same time would
answer neither question.

It is also the warning. Those numbers were measured on **M15 and M30**, on FX
and index CFDs, over eleven years. Three things follow:

1. **A one-ATR impulse on M15 and on M1 are different events.** The parameter
   is scale-free; the market structure it is trying to catch is not.
2. **Cost grows as the clock shrinks.** The stop is roughly one ATR, so the
   same spread is a small share of an M15 stop and a large share of an M1 one.
   That is exactly what turned section eleven from +0.047 R a trade into
   −0.18 once the cost model charged M1 what M1 costs. Expect M1 to be the
   one that dies here.
3. **Four cells is four searches.** Two mechanisms × two clocks on one market.
   A single positive result among four is weaker evidence than a single
   positive result on its own, and the report's sigma does not know that four
   were tried. Read the four together.

## What has to be true before any of this is promoted

Not a positive euro figure. The account has been fooled by one of those
already — one month carried 74 % of a 180-day result.

- enough trades of its own to be judged (the report says "unjudged" below its
  own threshold, and it is right)
- a result not carried by a single month
- a cost share per trade the account can actually pay, visible in `THE COST
  WALL`
- and the gate counterfactuals: a section that is only positive because a gate
  refused thousands of setups is not a section, it is a gate. Section fifteen
  is the live example — +€15.62 while its spread gate held back −2,241.89 R.

## How to measure it

    us30.cmd 180

Same replay shape as `hoeveel.cmd`: real Eightcap bars, real spread per bar,
the real sizer, the four live gates, the shared position book. Anything else
would make the numbers incomparable with the rest of the book, which is the
only thing they can be judged against.

## Status

**Not measured.** No claim about profitability is made anywhere in this
change, and none should be made until the owner has run the replay on the VPS.

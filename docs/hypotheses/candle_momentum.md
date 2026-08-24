# Hypothesis: candle_momentum

## The claim

This is the shape of the fast bots — watch M1, buy when a candle jumps, sell
straight back out — and the shape is not the claim. A green minute happens about
half the time, and a strategy whose entire thesis is "the last candle was green"
is a coin flip paying a spread on every flip.

The claim is narrower: **when the slower charts have already decided a
direction, a decisive M1 candle in that direction is the moment the standing
agreement becomes actionable.**

```
M15   the direction the session is going          decides the side
M5    the direction the last half hour is going   must not contradict
M1    the candle that just closed                 the trigger, and only that
```

All three or nothing. Two out of three is not a majority, it is a
disagreement, and on a trade that lives for minutes a disagreement is a coin
flip with costs attached.

## The counterparty

Whoever is providing liquidity into a move that is not finished. A decisive
minute inside an established direction is usually a participant working an
order that is larger than one minute of book — the flow continues while they
are still working it, and the market maker filling them has not yet stepped
back. The trade is a few minutes long because that is how long the imbalance
lasts.

**This is the weakest counterparty story of any hypothesis in this directory,
and that is stated deliberately.** The mechanism is real but it is also the
mechanism most crowded with other fast systems, and it decays fastest. Nothing
here should be read as a claim of durable edge.

## When the claim is false

**The candle is an event, not momentum.** A minute carrying several times its
own normal activity is a release, a headline or a stop cascade. It produces the
strongest-looking candle such a bot will ever print, and what follows it is a
spread that takes the account apart. This is how these systems actually die —
not by being slightly wrong many times, but by reading the first candle of a
red-folder release as the best signal of its life.

`extreme_volume_multiple` refuses it. The owner's instruction was explicit:
when the volume is extreme, do not trade.

**The candle is the end of the move.** A close hard against its own extreme
after the run has already travelled is the last buyer, not the first.
`exhaustion_close_position` refuses it, measured against the instrument's own
range because "already run" means one thing on gold and another on EURUSD.

**The candle is mostly wick.** Nobody finished the minute in control of it, so
there is no imbalance to ride.

## Why the news blackout is not in this module

The module reads bars. It has no calendar, and it should not grow one.

A second copy of the news rules living beside the real one is a copy that
eventually disagrees with it, and the direction it disagrees in is "traded
through a release nobody meant to trade through". So the blackout is enforced
where it already lives: `news_filter`, `headline_filter`, and the runner's
observer, which refuses to record a scalp when this cycle was stopped by either
— or when the calendar could not be reached at all.

That last case is deliberate. An unreachable calendar is the state this whole
account already refuses to trade in; a paper section that quietly kept going
would be measuring a strategy nobody would ever run.

## Why nothing else covers it

`m1_micro_breakout` is the closest and asks a different question: it wants a
discrete range to break. This wants no range at all — only a decisive candle
inside an agreement that already exists.

| module | why it is a different population |
|---|---|
| `m1_micro_breakout` | needs a base and a break of it; this needs neither |
| `impulse_break` | one bar against ATR, on any timeframe, with no multi-frame requirement |
| `fast_ema_cross` | an indicator relationship, not a candle |

It shares the `momentum` evidence family with `impulse_break` on purpose.
Giving it a family of its own would let it "corroborate" the impulse reader on
one observation seen twice, which is exactly the failure families exist to
prevent.

## Cut the loser fast, keep the winner

The first version had the exits the other way round — stop 1.1 spans, target
0.8 — on the owner's description *"als het maar een haartje verkeerd gaat
eruit, gaat het maar ietsjes goed ook gelijk eruit"*. Both halves of that are
"get out fast", and running the arithmetic showed they point in opposite
directions.

On gold at a $0.25 spread:

| target | stop | net win | net loss | hit rate needed |
|---|---|---|---|---|
| 0.20 | 1.65 | −$0.05 | −$1.90 | impossible |
| 1.20 | 1.90 | +$0.95 | −$2.15 | 69% |
| 1.20 | 0.60 | +$0.95 | −$0.85 | 47% |
| 1.20 | 0.40 | +$0.95 | −$0.65 | **41%** |
| 0.50 | 0.40 | +$0.25 | −$0.65 | 72% |

Cutting the **loser** fast takes 69% down to 41%. Cutting the **winner** fast
takes it back up to 72%, because the spread does not shrink with the target.
The first row is the instinct taken to its conclusion: a $0.20 target is below
the spread, so the net win is minus five cents and no hit rate on earth pays
for it.

So the stop is a fraction of the trigger candle — if the minute was read wrong
that is known almost immediately — and the target is larger than the candle,
because the winners are the only thing paying for any of this. It is still
fast; a trade that does not work is cut inside 0.4 of a candle span.

## Where it may trade, and how much at once

**Only where commission is zero.** Derived from `risk.commission_by_asset_class`
rather than written out as a second list, because two lists of asset classes
fall out of step and the direction they fall out of step in is "scalping forex
at EUR 5.50 a lot". A scalp's whole margin is a few spreads wide; a fixed fee
per lot does not fit inside it.

**At most two positions at once**, inside the account's overall cap of four
rather than beside it — and honoured on paper too, or the record measures the
returns of a book the account has no room to hold.

## Only when it is convincing

Not a slogan. The entry thresholds were raised across the board once the
flipped exits removed the need for a 67% hit rate:

| | was | now |
|---|---|---|
| body as a share of the candle | 55% | **70%** |
| body against its recent average | 1.6x | **2.2x** |
| volume that counts as an event | 4.0x | **3.0x** |
| target clearance over the spread | 5 | **8 spreads** |

## What has to be true before this trades

It does not trade. It carries a weight so `backtest_modules.py` measures it, it
is absent from `live_enabled_modules`, and the observer records what it would
have done for the resolver to settle.

```
scripts/scorecard.py --days 30
```

→ `SECTION TWO, ON PAPER`, row `SECTION_6_OBSERVED`

The expectation on record: **a hit rate above 41%**, which is what the shipped
geometry needs to break even on gold at a typical spread. That is a number a
selective multi-timeframe filter can plausibly beat, where the original 67%
could not be reasoned about at all.

Note what changed about the failure mode. The first version's danger zone was
55-64% — winning most trades and losing money, which feels like success. The
flipped exits move the danger somewhere far more visible: below 41% the losses
show up immediately in the hit rate itself, because winners are now bigger than
losers and a losing record means a genuinely losing strategy rather than an
arithmetic trap.

## Related

- `docs/hypotheses/drift_burst.md` — the other paper section, fading rather than
  following
- `filters/news_filter.py` — where the blackout actually lives, and the only
  place it should

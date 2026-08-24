# Hypothesis: momentum_scalp

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

## In small, out small

The target is **smaller than the stop**, and that is a choice rather than an
oversight about reward-to-risk.

The owner's description: *"als het maar een haartje verkeerd gaat eruit, gaat
het maar ietsjes goed ook gelijk eruit"*. The stop is the far side of the
candle that triggered it — the level that says the minute was read wrong — and
the target is a fraction of that span. A scalp reaching for more is not a scalp;
it is a swing trade wearing a scalp's stop, and it will be stopped like one.

The arithmetic this forces is worth stating plainly. At a reward-to-risk below
1, the break-even hit rate `(1 + cost) / (1 + RR)` climbs fast: at 0.73 RR and
a tenth of an R in cost it needs about **64%**, and every basis point of spread
pushes it higher. This shape only works with a high hit rate, and the whole
point of the multi-timeframe agreement is to buy one.

## What has to be true before this trades

It does not trade. It carries a weight so `backtest_modules.py` measures it, it
is absent from `live_enabled_modules`, and the observer records what it would
have done for the resolver to settle.

```
scripts/scorecard.py --days 30
```

→ `SECTION TWO, ON PAPER`, row `SECTION_6_OBSERVED`

The expectation on record: **a hit rate above 64%, or it is not worth having.**
Below that the arithmetic above says it loses money regardless of how good the
setups look, and no amount of tuning the entry fixes a reward-to-risk that
small. A hit rate in the high seventies would make it the strongest thing on
the account; anything between 55% and 64% is the exact trap this shape is
famous for — a strategy that wins most of the time and loses money.

## Related

- `docs/hypotheses/drift_burst.md` — the other paper section, fading rather than
  following
- `filters/news_filter.py` — where the blackout actually lives, and the only
  place it should

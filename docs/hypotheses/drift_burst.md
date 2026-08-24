# Hypothesis: drift_burst

## The claim

A price moved. There are two entirely different reasons that can happen, and on
a chart they look identical:

1. **Drift** — an actual repricing. Information arrived and the level changed.
2. **Volatility plus a hurry** — nothing changed, but one participant had to be
   out of their position *now* and paid whatever the book asked.

Those two have opposite futures. The first stays; the second comes back.

Christensen, Oomen and Renò (*The Drift Burst Hypothesis*, Journal of
Econometrics, 2022) give a way to tell them apart: a non-parametric statistic
that is a t-test on the local mean return, scaled by the local volatility.

```
T = sqrt(n_eff) * mu_hat / sigma_hat
```

Above roughly 4, the move is larger than the instrument's own volatility can
explain. Over a multi-year tick sample spanning equities, fixed income,
currencies and commodities they identify **more than a thousand** such events —
about one a week per instrument, typically 25 to 200 basis points — and find
that **two thirds are followed by price reversion**.

So this module fades.

## The counterparty

Someone who needed immediacy and could not wait for a good price. A margin call
being met, a stop cascade, a fund that must be flat by a deadline, an algorithm
unwinding into a thin book.

They are not wrong about the market. They are simply out of time, and being out
of time has a price. The reversion is that price being paid back to whoever was
willing to be on the other side.

**This matters more than the statistic.** A pattern that repeats is a pattern
that stops repeating once enough people trade it. A fee for a service does not
go away while the service is still needed — and someone will always need to be
out of a position immediately. The short-term reversal literature reaches the
same conclusion from a completely different direction, calling the same return
"compensation for supplying immediacy". Two fields converging on one mechanism
is a stronger reason to believe it than either alone.

## When the claim is false

**The move was information.** A third of them are, and those do not come back —
they continue, against the position. This is not a defect to engineer away; it
is the shape of a two-thirds edge and the reason the stop sits beyond the
burst's own extreme rather than at some fixed multiple.

**The reversion is real but not collectable.** Studying intraday reversals, the
finding is that on the NYSE "the large widening of the bid-ask spread eliminates
most of the profits that can be achieved by a contrarian strategy", while on the
NASDAQ, where the spread stays put, the same strategy pays. A burst *is* a
liquidity shock, so the spread widening alongside it is the normal case. Section
four is exactly this condition, measured against the baseline this account
already learns per symbol per hour.

**The statistic sees a ratio, not a move.** A dead-quiet instrument that twitches
produces a large `t` on a two-basis-point move. `minimum_move_bp` refuses it: an
event smaller than the research describes is not the research's event.

## Why nothing else covers it

Every other reader in section one measures the price series and concludes
something about **direction**. This one measures the price series and concludes
something about **whether the measurement means anything** — and then trades
against the answer.

| module | why it is a different population |
|---|---|
| `impulse_break` | a big bar is a signal to FOLLOW; here it is the thing being faded |
| `drift_continuation` | consistency of a move as evidence it continues |
| `volatility_regime` | how rare the volatility is; carries no direction at all |
| `mean_reversion` | fades a distance from a mean; this fades a *statistical* excess |

The last row is the closest and still a different claim. `mean_reversion` asks
how far price is from where it usually sits. This asks whether the move that
just happened is bigger than the market's own noise can account for — a
question about the *generating process*, not about a level.

It sits in its own evidence family (`immediacy`) for that reason.

## The open question this exists to answer

**The research is measured on tick data. This account has M1 bars.**

The statistic survives coarsening in principle, with less power and with
microstructure noise the paper itself calls awkward. Whether it survives in
practice on minute bars is not answerable from any paper, and it is answerable
for free.

So it does not trade. It carries a weight so `backtest_modules.py` measures it,
it is absent from `live_enabled_modules` so `effective_weights` zeroes it before
the engine scores anything, and the runner records what it *would* have done as
a shadow trade the existing resolver settles against real later prices.

## What was already measured, on this estimator

Two things had to be established before the first observation was worth
recording, and neither came from the paper.

**One window did not work.** With drift and volatility over the same window the
test was blind to the shape it exists for: over 3,000 synthetic paths a 20-bar
burst fired 68% of the time while a **5-bar burst covering 127 basis points**
— harder and faster — fired **0.0%**. Never once. The burst was inflating its
own denominator. With volatility taken over 120 bars that are mostly the calm
before the move, the same 5-bar burst fires 96-100%.

**The threshold is calibrated, not borrowed.** Over 4,000 pure random walks at
these settings the statistic reaches a 99th percentile of 2.58 and a maximum of
3.55, so 4.0 is a false-positive rate under one in four thousand. The paper's
threshold is also near 4, which is reassuring rather than load-bearing.

| threshold | false positive on noise | 5-bar burst | 3-bar burst |
|---|---|---|---|
| 4.0 | 0 in 6,000 | 18% | 41% |
| 3.5 | 1 in 6,000 | 46% | 73% |
| 3.0 | 1 in 333 | 72% | 91% |

Section two takes 4.0; section four takes 3.5 and adds the spread condition.

## What the record has to show before this goes live

```
scripts/scorecard.py --days 30
```

→ `SECTION TWO, ON PAPER — WHAT IT WOULD HAVE MADE`

The expectation on record: **a hit rate near the paper's two thirds**. Near 67%
and the statistic survived the coarsening. Near 45% and it did not, the M1 bars
were too coarse, and the honest answer is that this belongs to a system with
tick data and not to this one.

Caveat that travels with it: one false positive in four thousand *per reading*
is not one in four thousand per day. Across 845 instruments read every minute
they accumulate, and `minimum_move_bp` rather than the threshold is what
actually removes them.

## Related

- `docs/hypotheses/basket_divergence.md` — the other non-chart reader, and the
  one that could go live immediately because it has no coarsening question
- `learning/counterfactual.py` — the resolver that settles these observations

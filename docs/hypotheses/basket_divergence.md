# Hypothesis: basket_divergence

## The claim

Equity indices are not independent instruments. SPX500, NDX100, US30, FRA40 and
UK100 are five different weightings of the same two things: the price of risk
and the price of money. Intraday they run 80-95% together, and that co-movement
is not a correlation that happens to have held — it is what being an equity
index means.

So when the basket moves forty basis points over fifteen minutes and one member
has moved ten, the claim is narrow and mechanical: **the thing that moved four
of them has not yet moved the fifth, and it will.** The gap closes.

This is not a forecast of the market. The basket and the laggard may both rise,
both fall, or neither. The trade pays when the *difference* closes, which means
it is possible to be wrong about direction and right about the trade — something
no other reader on this account can do.

## The counterparty

Whoever is slow. Concretely, three of them:

**Index arbitrage that has not run yet.** The professional flow that keeps
correlated indices in line is not instantaneous; it is a desk or an algorithm
acting on a threshold. Between the move and their response there is a window,
and that window is the entire trade. The counterparty here is not losing money
— they are being paid for closing the gap, and this rides alongside them.

**Order flow that arrived in one market first.** A macro headline hits the US
futures, and the European cash indices follow with a lag because their
participants are fewer and slower at that hour. The counterparty is the market
maker in the lagging index who has not yet repriced, and who repriced a moment
later at a worse level.

**Liquidity, briefly.** One index absorbs a large order and its price is pushed
away from its peers by the size rather than by information. The counterparty is
whoever needed to trade that size immediately, and they paid for immediacy. That
is the same fee `drift_burst` collects, arriving through a different door.

## When the claim is false

**A genuine decoupling.** One index moves alone for a real, index-specific
reason — a national holiday, a single dominant constituent, a country risk
event. The gap does not close; it widens. This is where the losses are and no
amount of measurement removes it, only the position size does.

**A stale price mistaken for a divergence.** FRA40 does not keep SPX500's hours.
A closed market's last print looks like a laggard that never catches up, because
it is not lagging — it is not trading. Left unguarded this would manufacture its
largest, most confident setups at exactly the times they are worthless.

The module refuses on a **last-bar timestamp**, not on a session table. A
calendar can be wrong about a holiday, a half-day, or a broker's own hours; a
timestamp cannot. Every peer must have printed within `peer_max_age_seconds` or
it does not count, and fewer than `minimum_peers` fresh peers means no reading.

**The basket did not actually move.** A gap is a difference, so it is equally
large when the basket ran and when *this* market ran while the basket sat still.
Those are opposite trades — the second is the decoupling above, arriving
disguised as the setup. `minimum_basket_bp` and a sign check refuse it.

## Why nothing else covers it

**No module on this account has ever looked at a second instrument.** All nine
live readers take one `MarketContext` and read one price series. That is not an
oversight in any of them; it is the shape of the interface, and it means the
corroboration rule — two independent families before a setup may be carried —
has never had a genuinely independent family to draw on. Four ways of reading
one chart are still one chart.

| module | why it is a different population |
|---|---|
| `drift_continuation` | measures this instrument's own consistency; blind to peers |
| `trend_momentum` | infers a trend from this instrument's own EMAs |
| `impulse_break` | one bar against this instrument's own ATR |
| `market_structure` | this instrument's own swings |
| `drift_burst` | a hypothesis test on this instrument's own volatility |

Every row says "its own". This module has no reading at all when it is alone,
which is the precise inverse, and it is why it sits in its own evidence family
(`relative`) rather than beside any of them.

## Why it may go live where sections two and four may not

`drift_burst` rests on a statistic Christensen, Oomen and Renò measured on
**tick** data. Whether it survives being computed on M1 bars is a real open
question, so both its bands run as shadow trades until the record answers it.

There is no equivalent question here. A move between two M1 closes is exactly as
well measured on M1 as on ticks — the coarsening costs nothing because nothing
about the claim depends on resolution below a minute. There is no experiment to
wait out, which is the whole reason this one carries weight from the start.

## What is not yet known

**There is no measured record of this on this account.** The weight of 0.75 is
reasoned from where it sits between the other readers, not from a number. It is
the first new live weight added in this cycle of work and it should be the first
thing re-read once the scorecard has a sample:

```
scripts/scorecard.py --days 30
```

→ `WHICH DETECTOR WAS BEHIND IT`, and `WHICH DETECTOR, IN WHICH REGIME`

The expectation on record, so it can be wrong in public: **a hit rate of 65-80%
and three to ten setups a day** across the seven indices in the catalogue,
concentrated in the hours where two or more of them are open together. A hit
rate near 90% would mean the target is too small to pay for its own spread
rather than that the module is unusually good.

## Related

- `docs/hypotheses/drift_burst.md` — the other non-chart reader, collecting the
  same immediacy premium through the price path rather than through peers
- `filters/currency_exposure.py` — the existing decomposition of a position into
  what it is really a bet on, and the nearest thing to this that already existed

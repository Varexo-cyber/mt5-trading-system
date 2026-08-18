# Hypothesis: volatility_squeeze

## The claim

When a market's range contracts to an extreme against **its own history** and
then a single bar spans a large multiple of that contraction, the contraction
was an inventory build rather than an absence of interest — and the bar that
breaks it is the start of the repricing rather than the whole of it.

## The counterparty

Whoever was selling the range. A contracting range is quoted by participants who
believe the current price is fair and are being paid to say so; the tighter it
gets, the larger the position they accumulate at one level and the less room
they have to be wrong. When the range breaks, they are not merely stepping back
— they are holding an inventory that is now offside and must be covered in the
same direction as the break.

That is a claim about **the hour or two after the break**, not about a trend.
The plan it produces is an M15 one and it expires with the move.

## Why nothing else covers it

`volatility_regime` already measures compression and is the one module in this
system carrying a weight of **zero**. It scores no direction by construction, so
the single reader that knows a market is coiled has never been able to say that
it has uncoiled. Everything else looks at direction first:

| module | why it is silent on a coil |
|---|---|
| `trend_momentum` | EMAs converge during compression; there is no alignment to read |
| `drift_continuation` | needs a majority of bars closing one way, which a coil by definition does not have |
| `market_structure` | wants a swing break; a coil has no swings worth the name |
| `impulse_break` | measures the bar against ATR, not against the compression, so it fires on the same bar in an already-volatile market |

The last row is the important one. `impulse_break` and this module will
sometimes agree, and when they do the agreement is real corroboration: one says
the bar was large in absolute terms, the other that it was large relative to
what this market had been doing.

## The predicted way it loses

**The scheduled release.** A calendar event produces exactly this shape — a
market quiet ahead of the number, one enormous bar on it — and the mechanism
does not apply: nobody was accumulating inventory, they were waiting. If losses
cluster within an hour of a release the answer is a tighter news blackout for
this module and not a wider stop. This is the failure the author expects first.

**The session boundary.** The first bar of London on an instrument that sleeps
overnight is a large bar following a compressed range on the clock rather than
in the book. The percentile test over 200 bars is what should prevent this, but
it will not prevent it on an instrument whose overnight session is most of its
history.

**The false break.** Price closes through the edge, the coil re-forms, and the
stop on the far side is a full range away. The invalidation deliberately sits
beyond the OPPOSITE edge for this reason, which makes the stop wide and the
position small — that is the honest price of the pattern, not a fault in it.

## Pre-registered test

Development: EURUSD, GBPUSD, USDJPY, XAUUSD.
Held back: AUDUSD, USDCAD, US500, BTCUSD.
Swept: `compression_percentile`, `expansion_multiple`, `breakout_close_share`,
`compression_bars`.

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---:|---:|---:|---:|---:|
| Train (60%) | not run | — | — | — | — |
| Validation (20%) | untouched | — | — | — | — |
| Holdout (20%) | untouched | — | — | — | — |

**Not live.** It carries a research weight so `scripts/backtest_modules.py`
measures it, and it is absent from `live_enabled_modules`, so the confluence
engine zeroes its weight in live mode. It reaches the allowlist when that
backtest says it earns a place and not before — the same discipline that
measured `trend_momentum` at -0.382R a trade over 62 trades, t = -3.26, which is
the only hard finding this account has produced.

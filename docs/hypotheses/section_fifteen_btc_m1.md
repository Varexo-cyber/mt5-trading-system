# Section 15 — BTCUSD M1, adaptive channel, inverted

> Promotion record, the shared caveat and the containment:
> [btcusd_discoveries.md](btcusd_discoveries.md). Read that first — it carries
> the part that matters more than anything below.

## Claim

On BTCUSD, during the Asian hours, a 20-bar channel break resolves AGAINST
itself when the trend is strong and WITH itself when it is not — and the
inversion of that reading is tradeable on a one-minute clock.

## The rule, exactly

Per closed M1 bar, inside 00:00–07:00 UTC:

* `fast` = EMA20 of close, `slow` = EMA50 of close
* `strong` = `|fast − slow| / ATR > 1.00` (the `_100` in the mechanism name)
* `ceiling` / `floor` = the highest high / lowest low of the previous 20 bars
* strong **and** close above the ceiling → natural long; strong **and** close
  below the floor → natural short; when NOT strong the two are swapped
* `polarity: -1` inverts that natural reading, so the section trades the
  opposite of it

Stop 3.0 ATR, target 2.0 R, stop moved to entry at 1.5 R.

## Economic rationale, and its weakness

A channel break in a quiet book is a liquidity sweep: there is not enough flow
behind it to continue, so it retraces, and the counterparty is whoever chased
it. That is the honest half. The Asian window is where BTCUSD is thinnest,
which is consistent with it.

The weakness is that the same sentence would also justify the un-inverted
version, and `polarity` was chosen by the search rather than by the argument.
A mechanism whose direction is a fitted parameter is a mechanism whose story
was written after the fact. That is not fatal — the fill has a counterparty
either way — but it is the reason this needs forward data and not another
backtest.

## Why M1 is the expensive choice here

A 3.0 ATR stop on an M1 clock is still a small distance in price, and the
spread is charged against it. The M1 lane is the one where the cost model
turned section eleven from +0.047 R a trade into −0.18. Whatever this shows
live, the cost share per trade is the first column to read.

## What would take it back off

The section breaker, on its own, without anyone present: 60-trade window,
more than 60 % losers or eight in a row. Short of that, a cost share per trade
materially above what the replay charged means the replay was measuring a
cheaper instrument than the one being traded.

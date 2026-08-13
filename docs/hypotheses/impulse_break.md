# Hypothesis: impulse_break

## The claim

A closed bar whose **body** exceeds one ATR and which closes in the far third of
its own range is a repricing that exhausted the resting liquidity at the old
level. For a short period afterwards the book is thin on that side, so the next
marginal order has to reach further and price continues.

## The counterparty

Whoever was quoting the old level. Market makers and resting limit orders sized
for normal flow are filled through in a single bar, and they do not immediately
requote at the same depth — they widen and step back while they work out whether
the move is informed. The edge, if it exists, is the interval between the
liquidity being consumed and it being replaced.

This is a claim about **minutes**, not days. It has a natural expiry, which is
why `max_bars_since` is two M15 bars and the plan produced is an intraday one.

## Why nothing else covers it

Found in live data on 13 August, not reasoned into being. A GBPCAD move the
reviewer described as "a large, fresh down-impulse — M15 last candle body -1.34
ATR, M5 3-bar drift -2.3 ATR" fired **no directional module at all**:

| module | why it was silent |
|---|---|
| `drift_continuation` | needs 65% of eight M15 bars to close with the move; an impulse inside otherwise flat bars gives about 25% |
| `fast_ema_cross` | needs a 9/20 cross inside six M5 bars, and a vertical move has usually already crossed |
| `market_structure` | wants a confirmed break of a swing level, which arrives later or not at all |
| `liquidity_sweep` | is a reversal pattern and points the other way |

Only `trend_momentum` spoke, off an H4/H1 EMA alignment — so the trade went out
as a 24-hour swing on a slow indicator instead of a three-hour intraday trade on
the move that was actually happening, and was refused for chasing.

## The predicted way it loses

**Buying the top of the spike.** If the losses cluster on entries taken at the
extreme of the impulse with immediate reversal, the mechanism is wrong: the bar
was not consuming liquidity, it was the last participant paying up. The answer
then is a stricter `maximum_retracement` or waiting for a pullback — not a wider
stop.

Second predicted failure: **scheduled news.** A release produces exactly this
bar shape and the mechanism does not apply, because the move is a repricing that
has already finished rather than an order still being worked. The news blackout
covers the window around a release; it does not cover unscheduled moves.

## Pre-registered test

Development: EURUSD, GBPUSD, USDJPY, XAUUSD.
Held back: AUDUSD, USDCAD, US500, BTCUSD.
Swept: `minimum_body_atr`, `minimum_close_location`, `max_bars_since`,
`maximum_retracement`.

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---:|---:|---:|---:|---:|
| Train (60%) | not run | — | — | — | — |
| Validation (20%) | untouched | — | — | — | — |
| Holdout (20%) | untouched | — | — | — | — |

Live from 13 August 2026 on the owner's explicit instruction, ahead of that
protocol, alongside the other modules in the same position. What grades it first
is the live record: `decisions.signals` records which module found each trade
and `Brain.module_records()` reports realised R per detector once it has twenty.

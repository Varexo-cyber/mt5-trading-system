# Hypothesis: session_breakout

## The claim

A market that trades thinly overnight builds a range that represents the
opinions of few participants. When the busy session opens, the orders that
arrive were formed while the market was closed to their owners and they are
larger than the range can absorb. A break of that range within a few hours of
the session opening is those orders being worked, and the flow continues while
they are.

## The counterparty

Whoever held the overnight range. Thin-session market makers quote a narrow band
because nothing is happening and they are paid for the spread, not for a view.
They are not positioned for size. When the London or New York book opens and a
real order arrives, they have neither the inventory nor the mandate to hold the
level, so they step aside — and the range they were defending stops existing.

The claim expires. This is about the hours immediately after the session change,
which is why `breakout_window_hours` exists and why a break at 15:00 of a range
that closed at 07:00 is refused. By then, whatever the range meant is gone.

## Why nothing else covers it

**Time is used only to refuse trades in this system.** A news blackout, a runway
wind-down, an evening flat, a `MARKET_TOO_QUIET`. The entire time dimension is
subtractive. The fact that a market builds a range overnight and resolves it
when a session opens has never been available as a reason to take a position.

The break readers already present break whatever range is in front of them,
whenever it happens:

| module | why it is a different population |
|---|---|
| `impulse_break` | measures one bar against ATR; fires all day, on any bar |
| `m1_micro_breakout` | breaks a range measured in minutes, not in sessions |
| `market_structure` | wants a swing pivot, which an overnight range rarely has |

Those fire continuously; this fires a handful of times a day. Two readers that
agree all day are one reader, and that is precisely what the eight existing
modules turned out to be.

## The predicted way it loses

**The false break at the open.** The first order through takes out the stops
resting just beyond the range, and price returns inside — which is the
`liquidity_sweep` pattern seen from the other side, and the two modules will
directly contradict each other on exactly these bars. That contradiction is
useful information rather than a bug: if `liquidity_sweep` is right and this is
wrong, the confluence engine will be handed the disagreement and the module
backtest will show which side of it pays.

**The wrong clock.** Bars are stamped in broker server time and the window is
configured, not inferred. A server three hours off UTC silently moves every
session by three hours, and nothing looks wrong — the ranges are still ranges
and the breaks are still breaks. The module would measure the wrong hours of the
day for the life of the account. This is the failure that would not show up as a
loss so much as a null.

**Instruments with no overnight session.** A single share CFD has no meaningful
00:00-07:00 bars at all, and the `minimum_range_atr` floor is what should turn
that into silence rather than into a break of a two-bar range.

## Pre-registered test

Development: EURUSD, GBPUSD, USDJPY, XAUUSD.
Held back: AUDUSD, USDCAD, US500, BTCUSD.
Swept: `range_start_hour`, `range_end_hour`, `breakout_window_hours`,
`minimum_range_atr`, `maximum_range_atr`.

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---:|---:|---:|---:|---:|
| Train (60%) | not run | — | — | — | — |
| Validation (20%) | untouched | — | — | — | — |
| Holdout (20%) | untouched | — | — | — | — |

Note on the sweep: session hours are the most over-fittable parameter in this
document. Five start hours by five end hours is twenty-five variants, and one of
them will look excellent on any history. The registered window is the one
argued for above — the thin session, not the best-performing one — and any other
choice must be justified by the mechanism rather than by the result.

**Not live.** Research weight only; absent from `live_enabled_modules`.

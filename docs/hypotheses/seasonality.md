# Hypothesis: seasonality

## The claim

Some instruments have a recurring weekday flow — a settlement, a fixing, a
rebalancing, a payroll cycle — that shows up as a mean daily return on that
weekday which is distinguishable from zero given the instrument's own noise.

The claim is deliberately narrow. It is **not** "Mondays reverse" or "sell in
May". It is: this symbol, over the daily history this broker carries, has or has
not a weekday mean that survives a t-test at the configured threshold. On most
instruments on most days it will not, and saying so is the module working.

## The counterparty

Nobody, in the usual sense — which is the honest weakness of this hypothesis and
the reason it carries the lowest weight in the table.

Where a real mechanism exists it is an institution trading on a calendar rather
than on a price: month-end index rebalancing, a weekly options expiry, a
scheduled fixing, a payroll flow. Those participants are price-insensitive on
their day, and price-insensitive flow is the only thing that produces a
repeatable calendar effect.

Where no such mechanism exists, a weekday effect is a coincidence that will not
repeat, and this module cannot tell the two apart. That is why it can never
carry a trade alone.

## Why nothing else covers it

No module in this system reads the calendar. The news filter reads the economic
calendar and uses it exclusively to **block**. Nothing has ever asked whether
today is a day this instrument tends to move.

It is also the only module here whose input is not the shape of recent price. It
cannot correlate with the other readers, because it is not looking at the same
thing — which is the entire argument for having it, given that the eight
existing modules fire together and lose together.

## The predicted way it loses

**Multiple comparisons, and this is not a risk but a certainty.** Five weekdays
tested at t = 2 gives roughly a one-in-four chance of a spurious "significant"
weekday per instrument, before considering that the same test runs across
hundreds of symbols. Across an 843-symbol catalogue, dozens of instruments will
show a two-sigma weekday that means nothing at all.

The mitigations are deliberate and all three are needed:

1. **Weight 0.25**, the lowest in the table.
2. **Confidence ceiling 0.55**, below the 0.65 `lone_module_minimum_confidence`,
   so the strongest weekday effect on record cannot open a position by itself.
   This is asserted as a test, not left to arithmetic.
3. **`minimum_t` is configurable** precisely so it can be raised when the
   backtest shows the family behaving like noise.

**Regime change.** A flow that existed for two years stops when the fund that
caused it closes, and a 500-day lookback keeps reporting it for a year and a
half afterwards. A weekday edge that is real is also perishable, and nothing
here detects the moment it stops.

## Pre-registered test

Development: EURUSD, GBPUSD, USDJPY, XAUUSD.
Held back: AUDUSD, USDCAD, US500, BTCUSD.
Swept: `minimum_t`, `minimum_samples`, `lookback_days`.

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---:|---:|---:|---:|---:|
| Train (60%) | not run | — | — | — | — |
| Validation (20%) | untouched | — | — | — | — |
| Holdout (20%) | untouched | — | — | — | — |

The result to look for is not "does it win". It is whether the modules it tips
over the line do better than the same modules without it — which the ALONE and
WHEN PRESENT tables in `scripts/backtest_modules.py` answer directly, since this
module can barely produce a lone trade by construction.

**Not live.** Research weight only; absent from `live_enabled_modules`. Of the
four families added together, this is the one the author expects to be switched
off first.

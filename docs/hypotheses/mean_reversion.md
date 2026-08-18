# Hypothesis: mean_reversion

## The claim

When price reaches an extreme distance from its own recent mean — measured in
that market's own standard deviations, not in pips — and then **stops
extending**, the move was liquidity-driven rather than information-driven, and
some of it is given back.

The stall is the whole claim. Distance alone is not evidence of anything: a
trending market spends its life far from a trailing mean, and fading it because
it has gone a long way is how accounts are destroyed.

## The counterparty

Whoever had to trade regardless of price — a stop cascade, a margin call, a
rebalancing flow, an option hedge. Those participants are not expressing a view;
they are meeting an obligation, and when the obligation is met the pressure
stops. The price they left behind is one nobody chose.

The stall is how that exhaustion becomes visible: the flow that was pushing has
finished and nothing has replaced it. Before the stall there is no way to
distinguish a forced seller from an informed one, which is why this module says
nothing until price has come off the extreme by a measurable fraction of an ATR.

## Why nothing else covers it

Every directional module here is a continuation reader in some form — a trend,
a cross, a break of structure, a drift, a resumed pullback, a micro-break. The
one exception is `liquidity_sweep`, and over 180 days on five symbols it is the
only module that is not negative:

| module | trades | per trade | t |
|---|---:|---:|---:|
| `trend_momentum` | 62 | -0.382R | -3.26 |
| `impulse_break` | 28 | -0.229R | -1.18 |
| `drift_continuation` | 42 | -0.106R | -0.69 |
| `liquidity_sweep` | 23 | **+0.119R** | +0.59 |

That is a hint and explicitly not a proof: liquidity_sweep's own t of 0.59 is
noise, and it needs roughly 268 trades to establish itself at that expectancy.
The way to find out whether reversion is the family that works on this account
is to add a second, differently-constructed reversion reader and measure them
together — not to add a ninth continuation reader to seven that lose.

Distinct from the `range_fade` playbook, which needs an identified range with
touched edges and was measured at -0.174R over 898 trades. This asks a purely
statistical question and does not require a range to exist.

## The predicted way it loses

**Catching a knife.** A genuine repricing — a central bank, a default, a
gap — goes far, pauses, and then goes further. The stall test will be satisfied
by the pause. If losses cluster on days with scheduled events or on the first
bar after a gap, the mechanism is wrong for those conditions and the answer is
to exclude them, not to widen `entry_z`.

**Being early in a trend.** A strong trend produces a sequence of extremes, each
one stalling briefly before the next leg. Each stall is a small loss and there
can be many in a row. This is the failure mode that will show up as a poor max
drawdown with a tolerable win rate, and it is the reason the module is not on
the live allowlist on the strength of an argument.

**Mean drift.** The 48-bar mean moves toward price as price moves, so on a slow
grind the z-score never reaches the threshold and the module simply never
speaks. That is a false-negative rather than a loss, but it means a low trade
count is expected and a low trade count is not evidence of anything.

## Pre-registered test

Development: EURUSD, GBPUSD, USDJPY, XAUUSD.
Held back: AUDUSD, USDCAD, US500, BTCUSD.
Swept: `entry_z`, `stall_retrace_atr`, `lookback`, `max_bars_since_extreme`.

| Split | Trades | Win rate | Avg R | Expectancy | Max DD |
|---|---:|---:|---:|---:|---:|
| Train (60%) | not run | — | — | — | — |
| Validation (20%) | untouched | — | — | — | — |
| Holdout (20%) | untouched | — | — | — | — |

**Not live.** Research weight only; absent from `live_enabled_modules`, so the
confluence engine zeroes its weight in live mode. The module backtest decides
whether it reaches the allowlist.

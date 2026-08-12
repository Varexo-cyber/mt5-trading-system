# Hypothesis: drift_continuation

> Written before any backtest. The account is live and the owner authorised
> switching this on ahead of the protocol below; that is recorded here rather
> than dressed up, and the Result section stays empty until it is filled by a
> test rather than by a good day.

## Claim

When a liquid FX pair or index has travelled at least 1 ATR in one direction
over the last eight M15 bars, with at least 65% of those bars closing that way,
price continues in that direction often enough to pay for a 1.5–2R plan — and
this remains true for the two hours or so a plan on that timeframe lives.

Restricted to the four asset classes already scanned (forex, metal, index,
crypto) and explicitly **not** claimed in a measured range, where the
confluence engine refuses it outright.

## Economic rationale

Who is on the other side. A sustained one-way grind on a fifteen-minute chart
is usually an order that is too large to fill at once being worked through the
book. The counterparty is whoever is taking the other side of each slice —
market makers absorbing flow and laying it off, and traders fading a move on
the assumption it has gone far enough. While the parent order is still working,
the fade is early and pays for it. That is a real mechanism with an identifiable
loser, and it has a natural expiry: when the order is done, the drift stops.

That expiry is also why this is deliberately short-horizon and why the
consistency term matters more than the distance term. A move that arrived in
one spike and then went sideways is not an order being worked, it is news
already priced.

**Where the mechanism does not exist:** in a range. There is no parent order,
only two-sided noise, and 1 ATR of drift over eight bars happens constantly.
This is the failure mode the module is most exposed to, which is why it is
registered in `trend_continuation_modules` and refused when the regime
classifier measures a range.

## What prompted it

12 August 2026. GBPUSD declined for most of the session. The engine produced
344 refusals reading "price is moving against the long" and never once proposed
a short, because no module was looking for one:

- `trend_momentum` runs 20/50 EMAs on H4 and H1, so it falls silent well before
  a market turns and only speaks again long after the new move has begun.
- `liquidity_sweep` needs a wick through a 20-bar extreme on the final candle.
  Rare by construction, and a reversal pattern rather than a continuation one.
- `market_structure` needs a break of structure.

Between the slow module and the rare one there is a hole, and an hour of clean
one-way drift sits in it. The two longs that did get through both lost.

## Predictions, stated in advance

- Expected win rate: **35–45%** (a 1.5–2R plan; below 33% it cannot pay)
- Expected average R: **+0.05 to +0.20**, and honestly it may be zero
- Expected signals per instrument per year: **150–400** — far more than the
  swing modules, which is the point and also the risk
- Conditions where this should **fail**:
  - Ranging and mean-reverting markets. Explicitly guarded, and if it still
    loses there the guard is not strong enough.
  - Around scheduled news, where drift is a repricing that has finished rather
    than an order being worked.
  - Instruments whose spread is a large share of an M15 ATR: the horizon is
    short, so the round trip is a bigger fraction of the move than it is for
    the swing modules.
  - **The specific way this is expected to lose: entering late.** By the time
    eight bars have confirmed a move, much of it may be spent. If the losses
    cluster at entries near the extreme of the window, that is this failing in
    its predicted way and not bad luck.

## Test design

- Instruments (development set): EURUSD, GBPUSD, USDJPY, XAUUSD
- Held back for the cross-instrument test: AUDUSD, USDCAD, US500, BTCUSD
- Timeframe: M15 signal, entries judged on the live tick as everything else is
- Data split: oldest 60% train / next 20% validate / final 20% untouched
- Parameters to sweep, and their ranges:
  - `lookback_bars` 4–16
  - `minimum_drift_atr` 0.5–2.0
  - `minimum_consistency` 0.50–0.85
- **Number of configurations tested: not yet run**

## Result

Not filled in. This module is live on the owner's explicit instruction before
the protocol was run, alongside the three others that are in the same position
— see `live_enabled_modules` in `config/eightcap.yaml` and
`scripts/audit_live_promotion.py`, which lists exactly what is still missing.

The live account is itself now generating the attribution needed:
`decisions.signals` records which module found each trade, and
`Brain.module_records()` grades each detector by realised R once it has twenty
trades. That is not a substitute for the split above — it is in-sample by
definition and cannot be un-looked-at — but it will show a clearly losing
detector long before the backtest is written.

- [ ] ≥100 trades
- [ ] Broad parameter plateau, not a spike
- [ ] Holds on instruments it was not developed on
- [ ] Deflated Sharpe still positive after correcting for the config count
- [ ] Final 20% looked at exactly once

## Verdict

`weight > 0` — 0.7, below the 1.0 of the swing modules and the 0.8 of
`liquidity_sweep`, because a drift says the move happened and not that it
continues. To be revisited against `module_records()` at twenty trades and
removed if it is losing.

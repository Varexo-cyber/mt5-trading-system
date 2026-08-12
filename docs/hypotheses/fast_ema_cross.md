# Hypothesis: fast_ema_cross

> Written before any backtest. The account is live and the owner authorised
> switching this on ahead of the protocol; that is recorded here rather than
> dressed up, and the Result section stays empty until a test fills it.

## Claim

A 9/20 EMA cross on M5 that is at most three bars old, with the averages at
least 0.15 ATR apart and price closing on the signal side of the slow average,
is followed by continuation often enough to pay a 1.5–2R intraday plan on
liquid FX, metals and indices.

Explicitly not claimed in a measured range, where the confluence engine refuses
it outright.

## Economic rationale

Who is on the other side. A fast EMA crossing a slower one on a five-minute
chart is a compressed statement that the last forty-five minutes have gone
somewhere the previous hundred did not. When that happens on a liquid
instrument it is usually flow — an order being worked, or a level giving way
and the stops behind it filling. The counterparty is whoever placed those stops
and whoever is fading the break early.

The honest weakness of the mechanism: it is the most widely watched signal in
retail trading, which cuts both ways. Widely watched levels attract the stops
that make a break move, and they also attract everyone trying to trade the same
break at the same price. If there is edge here it is small and it is in the
filtering, not in the cross.

**Where the mechanism does not exist:** in a range. There is no flow, only two
averages oscillating around each other, and each oscillation is a round trip
paid to the broker. This is the dominant failure mode and it is guarded three
ways — freshness, separation, and the range gate.

## What prompted it

The owner's observation that every directional module in the engine reads a
slow chart: 20/50 EMAs on H4 and H1, a break of structure, or at the fastest a
specific wick on M15. Nothing looked at the timeframe a day trade is actually
entered on, so intraday opportunity — particularly on the short side during
the 12 August GBPUSD decline — was invisible.

## Predictions, stated in advance

- Expected win rate: **35–45%** on a 1.5–2R intraday plan
- Expected average R: **-0.05 to +0.15**. It may well be negative; the spread
  is a larger share of an M5 move than of anything else here.
- Expected signals per instrument per year: **500–1500**. By far the most
  frequent module, which is the point and the risk in one number.
- Conditions where this should **fail**:
  - Ranges. Guarded; if it still loses there, the guard is too weak.
  - Instruments where the spread is a meaningful fraction of an M5 ATR. The
    cost gates should refuse most of those, correctly.
  - **The specific predicted failure: the whipsaw.** A cross, an entry, a
    reversal within the hour. If the losses cluster at trades held under thirty
    minutes with the exit on the opposite side of the entry, that is this
    failing exactly as expected — and the answer is a wider separation floor,
    not a wider stop.

## Test design

- Instruments (development set): EURUSD, GBPUSD, USDJPY, XAUUSD
- Held back for the cross-instrument test: AUDUSD, USDCAD, US500, BTCUSD
- Timeframe: M5 signal, M15 planning via the intraday horizon profile
- Data split: oldest 60% train / next 20% validate / final 20% untouched
- Parameters to sweep, and their ranges:
  - `fast_ema` 5–13, `slow_ema` 15–34
  - `max_bars_since_cross` 1–8
  - `minimum_separation_atr` 0.05–0.50
- **Number of configurations tested: not yet run**

## Result

Not filled in. Live on the owner's explicit instruction ahead of the protocol,
alongside the four other modules in that position — see `live_enabled_modules`
in `config/eightcap.yaml` and `scripts/audit_live_promotion.py`.

`decisions.signals` records which module found each trade and
`Brain.module_records()` grades each detector by realised R at twenty trades.
In-sample and no substitute for the split above, but it will show a losing
detector long before the backtest exists.

- [ ] ≥100 trades
- [ ] Broad parameter plateau, not a spike
- [ ] Holds on instruments it was not developed on
- [ ] Deflated Sharpe still positive after correcting for the config count
- [ ] Final 20% looked at exactly once

## Verdict

`weight > 0` — 0.5, the lowest in the table. It is the fastest and least
corroborated evidence in the engine and should need help to clear the
threshold rather than carrying a trade alone. To be removed if
`module_records()` shows it losing at twenty trades.
